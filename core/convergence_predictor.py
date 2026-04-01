"""
convergence_predictor.py — Signal Aggregation Layer (Section 5).

Collects all signals emitted by active concepts on a single bar, applies
per-category confidence multipliers, and produces a Convergence Score.

Formula (Section 5.1):
  convergence_score = Σ (signal.confidence × category_weight × nucleus_alignment_bonus)
  normalized to [0, 1]

Category weights (higher = more weight given to that signal type):
  STRUCTURE      : 1.5   (MSS, BOS, CHOCH are "anchor" signals)
  ORDER_BLOCK    : 1.4
  FVG            : 1.3
  LIQUIDITY      : 1.2
  TIME           : 1.1   (kill zone alignment)
  BIAS_FRAMEWORK : 1.0
  PREMIUM_DISCOUNT: 1.0
  CORRELATION    : 0.9   (SMT divergence — confirmation only)
  ANOMALY        : 0.8   (ATR regime — filters only)
  default        : 1.0

Nucleus alignment bonus: +20% if the signal's direction aligns with
the current dominant nucleus.

Threshold tiers:
  STRONG   : convergence_score ≥ 0.75
  MODERATE : convergence_score ≥ 0.55
  WEAK     : convergence_score ≥ 0.35
  NOISE    : below 0.35 → suppressed
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from technical.bar_snapshot import BarSnapshot, Signal, Direction, SignalStrength
from core.nucleus_engine import NucleusState, NucleusType

logger = logging.getLogger(__name__)

# ── Category weights ──────────────────────────────────────────────────────
CATEGORY_WEIGHTS: Dict[str, float] = {
    "STRUCTURE":       1.5,
    "ORDER_BLOCK":     1.4,
    "FVG":             1.3,
    "LIQUIDITY":       1.2,
    "TIME":            1.1,
    "BIAS_FRAMEWORK":  1.0,
    "PREMIUM_DISCOUNT":1.0,
    "CORRELATION":     0.9,
    "ANOMALY":         0.8,
}

NUCLEUS_ALIGNMENT_BONUS = 0.20  # +20% when signal aligns with nucleus

# ── Threshold tiers ───────────────────────────────────────────────────────
THRESHOLD_STRONG   = 0.75
THRESHOLD_MODERATE = 0.55
THRESHOLD_WEAK     = 0.35


class ConvergenceTier(str, Enum):
    STRONG   = "STRONG"
    MODERATE = "MODERATE"
    WEAK     = "WEAK"
    NOISE    = "NOISE"


@dataclass
class ConvergenceResult:
    score:        float
    tier:         ConvergenceTier
    direction:    Optional[Direction]
    signals:      List[Signal]            = field(default_factory=list)
    contributors: Dict[str, float]        = field(default_factory=dict)  # concept → weighted_score
    nucleus_state: Optional[NucleusState] = None


class ConvergencePredictor:
    """
    Aggregates per-bar signals from all concepts into a single convergence score.
    """

    def __init__(self, min_signals: int = 2):
        """
        Parameters
        ----------
        min_signals : int
            Minimum number of confirming signals required to produce a result
            stronger than NOISE. Default 2 (avoid single-concept "conviction").
        """
        self.min_signals = min_signals

    def predict(
        self,
        signals: List[Signal],
        nucleus_state: Optional[NucleusState] = None,
    ) -> ConvergenceResult:
        """
        Parameters
        ----------
        signals      : all signals from SignalRouter for this bar
        nucleus_state: output of NucleusEngine.identify() for this bar (optional)

        Returns
        -------
        ConvergenceResult
        """
        if not signals:
            return ConvergenceResult(0.0, ConvergenceTier.NOISE, None)

        # ── Vote on direction ─────────────────────────────────────────────
        long_votes  = sum(1 for s in signals if s.direction == Direction.LONG)
        short_votes = sum(1 for s in signals if s.direction == Direction.SHORT)
        dominant_dir = Direction.LONG if long_votes >= short_votes else Direction.SHORT
        aligned = [s for s in signals if s.direction == dominant_dir]

        if len(aligned) < self.min_signals:
            return ConvergenceResult(0.0, ConvergenceTier.NOISE, dominant_dir,
                                     signals=signals, nucleus_state=nucleus_state)

        # ── Weighted score accumulation ───────────────────────────────────
        total_weight  = 0.0
        weighted_sum  = 0.0
        contributors: Dict[str, float] = {}

        for sig in aligned:
            category = self._infer_category(sig)
            w        = CATEGORY_WEIGHTS.get(category, 1.0)

            # Nucleus alignment bonus
            if nucleus_state is not None:
                if self._nucleus_aligns(sig, nucleus_state):
                    w *= (1 + NUCLEUS_ALIGNMENT_BONUS)

            contrib        = sig.confidence * w
            weighted_sum  += contrib
            total_weight  += w
            contributors[sig.concept] = round(contrib, 4)

        raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        # Normalise: weighted average of confidences already in [0,1], so clip
        score = min(1.0, max(0.0, raw_score))

        tier = self._tier(score)

        logger.debug(
            "ConvergencePredictor: score=%.3f tier=%s dir=%s (%d/%d signals aligned)",
            score, tier.value, dominant_dir.value, len(aligned), len(signals),
        )

        return ConvergenceResult(
            score         = score,
            tier          = tier,
            direction     = dominant_dir,
            signals       = signals,
            contributors  = contributors,
            nucleus_state = nucleus_state,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _infer_category(sig: Signal) -> str:
        """Infer category from concept name for weight lookup."""
        name = sig.concept.upper()
        if any(x in name for x in ("MSS", "BOS", "CHOCH", "STRUCTURE", "SHIFT", "BREAK", "CHANGE")):
            return "STRUCTURE"
        if "ORDER" in name and "BLOCK" in name:
            return "ORDER_BLOCK"
        if "FVG" in name or "FAIR" in name:
            return "FVG"
        if "LIQUIDITY" in name:
            return "LIQUIDITY"
        if "KILL" in name or "ZONE" in name or "POWER" in name or "THREE" in name:
            return "TIME"
        if "BIAS" in name or "DAILY" in name:
            return "BIAS_FRAMEWORK"
        if "PREMIUM" in name or "DISCOUNT" in name:
            return "PREMIUM_DISCOUNT"
        if "SMT" in name or "DIVERGENCE" in name:
            return "CORRELATION"
        if "ATR" in name or "REGIME" in name or "QUANT" in name:
            return "ANOMALY"
        return "default"

    @staticmethod
    def _nucleus_aligns(sig: Signal, ns: NucleusState) -> bool:
        """
        Returns True if the signal direction is consistent with the dominant nucleus.
        Bullish nuclei: FVG demand, premium/discount at discount, order block support.
        Bearish nuclei: FVG supply, premium/discount at premium, order block resistance.
        """
        # Simple heuristic: if nucleus score is high and direction is LONG, assume bullish nucleus
        # The more nuanced version would check sig.metadata["zone"] for premium/discount
        nucleus = ns.dominant
        if nucleus in (NucleusType.EQUILIBRIUM, NucleusType.INSTITUTIONAL_RANGE):
            return True   # Neutral nucleus — no alignment bonus
        meta = sig.metadata or {}
        zone = str(meta.get("zone", "")).upper()
        if sig.direction == Direction.LONG and zone not in ("PREMIUM",):
            return True
        if sig.direction == Direction.SHORT and zone not in ("DISCOUNT",):
            return True
        return False

    @staticmethod
    def _tier(score: float) -> ConvergenceTier:
        if score >= THRESHOLD_STRONG:
            return ConvergenceTier.STRONG
        if score >= THRESHOLD_MODERATE:
            return ConvergenceTier.MODERATE
        if score >= THRESHOLD_WEAK:
            return ConvergenceTier.WEAK
        return ConvergenceTier.NOISE
