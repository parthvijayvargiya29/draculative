#!/usr/bin/env python3
"""
core/ict2_convergence_engine.py
ICT2 Convergence Engine
========================
Aggregates ALL ICT1 and ICT2 signals into a single directional convergence
score (-1.0 to +1.0) and a NucleusIdentification-adjusted final score.

Weights are loaded from configs/convergence_weights.yaml.

Signal groups:
  structure    : fvg_bias, bos_choch, premium_discount
  liquidity    : ssl_bsl_sweep, turtle_soup, nwog_nmog
  entry_model  : silver_bullet, po3, propulsion_block, bpr
  time_context : kill_zone          (applied as a multiplier on structure+entry)
  displacement : displacement
  fundamental  : value, quality, growth, sentiment
  news         : news_sentiment, news_event

Usage:
    engine = ICT2ConvergenceEngine()
    score  = engine.score(all_results_dict)

all_results_dict keys (all optional — missing keys score 0):
  fvg_result         : FVGAnalysisResult
  kill_zone          : KillZoneResult
  displacement       : DisplacementResult
  nwog               : NWOGResult
  propulsion_block   : PropulsionBlockResult
  bpr                : BPRResult
  turtle_soup        : TurtleSoupResult
  po3                : PO3Result
  silver_bullet      : SilverBulletResult
  fundamental_score  : float  (-1 to +1 from StockPredictor)
  news_score         : float  (-1 to +1 from StockPredictor)
  nucleus_score      : float  (0 to 1 from NucleusValidator, default 0.7)
  current_price      : float
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_CFG_PATH  = _REPO_ROOT / "configs" / "convergence_weights.yaml"


def _load_cfg() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    return {}


_CFG = _load_cfg()

_GROUP_W = _CFG.get("groups", {
    "structure":    0.25,
    "liquidity":    0.20,
    "entry_model":  0.20,
    "time_context": 0.10,
    "displacement": 0.10,
    "fundamental":  0.10,
    "news":         0.05,
})
_SIGNAL_LIMITS = _CFG.get("signal_limits", {})
_NUC_MIN = float(_CFG.get("nucleus_min_multiplier", 0.5))
_NUC_MAX = float(_CFG.get("nucleus_max_multiplier", 1.5))
_DIR_T = _CFG.get("direction_thresholds", {
    "strong_buy":  0.40,
    "buy":         0.15,
    "hold_lo":    -0.15,
    "sell":       -0.40,
})


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class ConvergenceScore:
    raw_score: float                # -1.0 to +1.0
    nucleus_multiplier: float       # 0.5–1.5
    final_score: float              # raw × nucleus, clamped ±1.0
    direction: str                  # STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL
    confidence: float               # 0–1
    top_signals: List[str]          # top 3 contributing signal names
    signal_breakdown: Dict[str, float]  # group → group_score
    conflicting_signals: List[str]  # signals pointing opposite to final direction


# ── Engine ────────────────────────────────────────────────────────────────────
class ICT2ConvergenceEngine:
    """
    Aggregates all ICT1 and ICT2 signals into a single convergence score.
    Weights are loaded from configs/convergence_weights.yaml.
    """

    SIGNAL_GROUPS = {
        "structure":    ["fvg_bias", "bos_choch", "premium_discount"],
        "liquidity":    ["ssl_bsl_sweep", "turtle_soup", "nwog_nmog"],
        "entry_model":  ["silver_bullet", "po3", "propulsion_block", "bpr"],
        "time_context": ["kill_zone"],
        "displacement": ["displacement"],
        "fundamental":  ["value", "quality", "growth", "sentiment"],
        "news":         ["news_sentiment", "news_event"],
    }

    def score(self, all_results: Dict[str, Any]) -> ConvergenceScore:
        """
        Compute convergence score from all signal results.

        Parameters
        ----------
        all_results : dict
            Keys described in module docstring. All optional.
        """
        raw_signals: Dict[str, float] = {}

        # ── Structure signals ──────────────────────────────────────────────────
        fvg = all_results.get("fvg_result")
        if fvg is not None:
            # fvg_bias
            if fvg.bias == "bullish":
                s = 0.3 if fvg.bias_strength == "strong" else 0.15
            elif fvg.bias == "bearish":
                s = -0.3 if fvg.bias_strength == "strong" else -0.15
            else:
                s = 0.0
            raw_signals["fvg_bias"] = s

            # bos_choch
            bos_s = 0.0
            st = fvg.structure
            if st.last_choch == "bullish":
                bos_s = 0.2
            elif st.last_choch == "bearish":
                bos_s = -0.2
            elif st.last_bos == "bullish":
                bos_s = 0.15
            elif st.last_bos == "bearish":
                bos_s = -0.15
            raw_signals["bos_choch"] = bos_s

            # premium_discount
            pd_z = fvg.premium_discount
            if pd_z.current_position == "discount":
                raw_signals["premium_discount"] = 0.10
            elif pd_z.current_position == "premium":
                raw_signals["premium_discount"] = -0.10
            else:
                raw_signals["premium_discount"] = 0.0

            # ssl_bsl_sweep (from fvg liquidity sweeps)
            sweep_s = 0.0
            for sw in fvg.liquidity_sweeps[-3:]:
                if sw.kind == "ssl_sweep" and sw.rejection:
                    sweep_s += 0.1
                elif sw.kind == "bsl_sweep" and sw.rejection:
                    sweep_s -= 0.1
            raw_signals["ssl_bsl_sweep"] = max(-0.3, min(0.3, sweep_s))

        # ── Turtle Soup ────────────────────────────────────────────────────────
        ts = all_results.get("turtle_soup")
        if ts is not None and ts.detected:
            limit = float(_SIGNAL_LIMITS.get("turtle_soup", 0.40))
            s = limit if ts.direction == "long" else -limit
            raw_signals["turtle_soup"] = s * ts.confidence

        # ── NWOG / NMOG ────────────────────────────────────────────────────────
        nwog = all_results.get("nwog")
        if nwog is not None:
            limit = float(_SIGNAL_LIMITS.get("nwog_nmog", 0.20))
            if nwog.bias_from_gaps == "bullish":
                raw_signals["nwog_nmog"] = limit
            elif nwog.bias_from_gaps == "bearish":
                raw_signals["nwog_nmog"] = -limit
            else:
                raw_signals["nwog_nmog"] = 0.0

            # Additional CE proximity weight
            price = float(all_results.get("current_price", 1.0))
            if price > 0:
                for attr, factor in [("nearest_ce_above", -0.05), ("nearest_ce_below", 0.05)]:
                    ce = getattr(nwog, attr, None)
                    if ce and abs(ce - price) / price < 0.02:
                        raw_signals["nwog_nmog"] = raw_signals.get("nwog_nmog", 0) + factor

        # ── Silver Bullet ──────────────────────────────────────────────────────
        sb = all_results.get("silver_bullet")
        if sb is not None and sb.setup_valid:
            limit = float(_SIGNAL_LIMITS.get("silver_bullet", 0.30))
            # Determine direction from entry vs target
            s = limit if sb.target_price > sb.entry_zone_midpoint else -limit
            raw_signals["silver_bullet"] = s * sb.confidence

        # ── Power of Three ─────────────────────────────────────────────────────
        po3 = all_results.get("po3")
        if po3 is not None and po3.phase == "distribution":
            limit = float(_SIGNAL_LIMITS.get("po3_distribution", 0.35))
            s = limit if po3.expected_direction == "bullish" else -limit
            raw_signals["po3"] = s * po3.confidence
        elif po3 is not None and po3.phase == "manipulation":
            # Manipulation phase hints at upcoming opposite move
            s = -0.15 if po3.expected_direction == "bullish" else 0.15
            raw_signals["po3"] = s * po3.confidence

        # ── Propulsion Block ───────────────────────────────────────────────────
        pb = all_results.get("propulsion_block")
        if pb is not None and pb.detected and not pb.mitigated:
            limit = float(_SIGNAL_LIMITS.get("propulsion_block", 0.25))
            s = limit if pb.direction == "bullish" else -limit
            raw_signals["propulsion_block"] = s * pb.confluence_score

        # ── Balanced Price Range ───────────────────────────────────────────────
        bpr_res = all_results.get("bpr")
        if bpr_res is not None:
            limit = float(_SIGNAL_LIMITS.get("bpr_ce", 0.15))
            price = float(all_results.get("current_price", 1.0))
            if price > 0:
                if bpr_res.nearest_bpr_above and \
                   abs(bpr_res.nearest_bpr_above.ce - price) / price < 0.02:
                    raw_signals["bpr"] = -limit   # CE above acts as magnet/resistance
                elif bpr_res.nearest_bpr_below and \
                     abs(bpr_res.nearest_bpr_below.ce - price) / price < 0.02:
                    raw_signals["bpr"] = limit    # CE below acts as support
                else:
                    raw_signals["bpr"] = 0.0

        # ── Displacement ───────────────────────────────────────────────────────
        disp = all_results.get("displacement")
        if disp is not None and disp.detected:
            limit = float(_SIGNAL_LIMITS.get("displacement", 0.30))
            s = limit if disp.direction == "bullish" else -limit
            # Decay by age
            age_decay = max(0.0, 1.0 - disp.candles_ago * 0.15)
            raw_signals["displacement"] = s * disp.strength * age_decay

        # ── Kill Zone ─────────────────────────────────────────────────────────
        kz = all_results.get("kill_zone")
        kz_multiplier = 1.0
        if kz is not None:
            kz_multiplier = 0.5 + kz.zone_strength * 0.5  # 0.5 to 1.0
            # Kill zone with bias alignment gives a slight directional nudge
            if kz.zone_strength >= 0.7:
                if kz.bias_direction == "bullish":
                    raw_signals["kill_zone"] = 0.05
                elif kz.bias_direction == "bearish":
                    raw_signals["kill_zone"] = -0.05
                else:
                    raw_signals["kill_zone"] = 0.0
            else:
                raw_signals["kill_zone"] = 0.0

        # ── Fundamental ────────────────────────────────────────────────────────
        fund_raw = all_results.get("fundamental_score", 0.0)
        if isinstance(fund_raw, (int, float)):
            raw_signals["value"]     = float(fund_raw) * 0.5
            raw_signals["quality"]   = float(fund_raw) * 0.3
            raw_signals["growth"]    = float(fund_raw) * 0.2
            raw_signals["sentiment"] = 0.0

        # ── News ───────────────────────────────────────────────────────────────
        news_raw = all_results.get("news_score", 0.0)
        if isinstance(news_raw, (int, float)):
            raw_signals["news_sentiment"] = float(news_raw) * 0.7
            raw_signals["news_event"]     = float(news_raw) * 0.3

        # ── Group aggregation ──────────────────────────────────────────────────
        group_scores: Dict[str, float] = {}
        for group, signal_names in self.SIGNAL_GROUPS.items():
            group_w = float(_GROUP_W.get(group, 0.1))
            sigs = [raw_signals.get(name, 0.0) for name in signal_names]
            # Normalise within group
            n_sig = len([s for s in sigs if s != 0.0])
            if n_sig > 0:
                group_raw = sum(sigs) / n_sig
            else:
                group_raw = 0.0
            group_scores[group] = group_raw * group_w

        # Apply kill zone multiplier to structure and entry_model groups
        group_scores["structure"]   *= kz_multiplier
        group_scores["entry_model"] *= kz_multiplier

        raw_score = max(-1.0, min(1.0, sum(group_scores.values())))

        # ── Nucleus multiplier ─────────────────────────────────────────────────
        # Map nucleus_score (0–1) linearly:  0.0 → 0.5,  0.5 → 1.0,  1.0 → 1.5
        # (Spec formula: multiplier = 0.5 + nucleus_score)
        nucleus_score = all_results.get("nucleus_score", None)
        if nucleus_score is not None:
            nuc_mult = 0.5 + float(nucleus_score)   # linear, spec-exact
        else:
            nuc_mult = 1.0   # degraded mode: no nucleus data available
        nuc_mult = max(_NUC_MIN, min(_NUC_MAX, nuc_mult))

        final_score = max(-1.0, min(1.0, raw_score * nuc_mult))

        # ── Direction ──────────────────────────────────────────────────────────
        dt = _DIR_T
        if final_score >= float(dt.get("strong_buy", 0.40)):
            direction = "STRONG_BUY"
        elif final_score >= float(dt.get("buy", 0.15)):
            direction = "BUY"
        elif final_score >= float(dt.get("hold_lo", -0.15)):
            direction = "HOLD"
        elif final_score >= float(dt.get("sell", -0.40)):
            direction = "SELL"
        else:
            direction = "STRONG_SELL"

        # ── Confidence ────────────────────────────────────────────────────────
        # Based on fraction of signals that agree with final direction
        final_sign = 1 if final_score >= 0 else -1
        agreeing = sum(1 for v in raw_signals.values() if v != 0 and
                       (1 if v > 0 else -1) == final_sign)
        total_active = sum(1 for v in raw_signals.values() if v != 0)
        confidence = (agreeing / total_active) if total_active > 0 else 0.5

        # ── Top 3 signals ──────────────────────────────────────────────────────
        sorted_sigs = sorted(
            [(k, abs(v)) for k, v in raw_signals.items() if v != 0],
            key=lambda x: x[1], reverse=True
        )
        top_signals = [name for name, _ in sorted_sigs[:3]]

        # ── Conflicting signals ────────────────────────────────────────────────
        conflicting = [
            name for name, val in raw_signals.items()
            if val != 0 and (1 if val > 0 else -1) != final_sign
        ]

        return ConvergenceScore(
            raw_score=round(raw_score, 4),
            nucleus_multiplier=round(nuc_mult, 4),
            final_score=round(final_score, 4),
            direction=direction,
            confidence=round(confidence, 4),
            top_signals=top_signals,
            signal_breakdown={k: round(v, 4) for k, v in group_scores.items()},
            conflicting_signals=conflicting,
        )


# ── Convenience function ──────────────────────────────────────────────────────
def compute_convergence(all_results: Dict[str, Any]) -> ConvergenceScore:
    """Module-level convenience wrapper."""
    return ICT2ConvergenceEngine().score(all_results)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from trading_system.ict_signals.killzone_filter import KillZoneDetector
    from trading_system.ict_signals.displacement_detector import DisplacementDetector
    from trading_system.ict_signals.nwog_detector import NWOGDetector
    from trading_system.ict_signals.turtle_soup_detector import TurtleSoupDetector
    from trading_system.ict_signals.power_of_three import PowerOfThreeDetector
    from trading_system.ict_signals.silver_bullet_setup import SilverBulletDetector
    from trading_system.ict_signals.balanced_price_range import BPRDetector
    from trading_system.ict_signals.propulsion_block_detector import PropulsionBlockDetector

    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="1y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    kz_det   = KillZoneDetector(htf_bias="bullish")
    disp_det = DisplacementDetector()
    nwog_det = NWOGDetector()
    ts_det   = TurtleSoupDetector()
    po3_det  = PowerOfThreeDetector(expected_direction="bullish")
    sb_det   = SilverBulletDetector()
    bpr_det  = BPRDetector()
    pb_det   = PropulsionBlockDetector()

    engine = ICT2ConvergenceEngine()

    i = len(df) - 1
    sl = df.iloc[:i+1]
    price = float(df["close"].iloc[-1])

    kz_r   = kz_det.process(None, price)   # daily bar
    disp_r = disp_det.update(sl)
    nwog_r = nwog_det.update(sl)
    ts_r   = ts_det.update(sl)
    po3_r  = po3_det.update(sl)
    sb_r   = sb_det.update(sl)
    bpr_r  = bpr_det.update(sl)
    pb_r   = pb_det.update(sl)

    result = engine.score({
        "kill_zone":       kz_r,
        "displacement":    disp_r,
        "nwog":            nwog_r,
        "turtle_soup":     ts_r,
        "po3":             po3_r,
        "silver_bullet":   sb_r,
        "bpr":             bpr_r,
        "propulsion_block": pb_r,
        "fundamental_score": 0.2,
        "news_score":      0.1,
        "nucleus_score":   0.75,
        "current_price":   price,
    })

    print(f"=== ICT2 Convergence Engine — {ticker} @ ${price:.2f} ===")
    print(f"  Raw score          : {result.raw_score:+.4f}")
    print(f"  Nucleus multiplier : {result.nucleus_multiplier:.4f}")
    print(f"  Final score        : {result.final_score:+.4f}")
    print(f"  Direction          : {result.direction}")
    print(f"  Confidence         : {result.confidence:.2%}")
    print(f"  Top signals        : {result.top_signals}")
    print(f"  Conflicting        : {result.conflicting_signals[:5]}")
    print(f"  Breakdown          :")
    for group, score in result.signal_breakdown.items():
        print(f"    {group:<15} {score:+.4f}")
    print("  PASS ✓")
