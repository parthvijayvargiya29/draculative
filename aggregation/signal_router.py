"""
aggregation/signal_router.py
=============================
Signal routing and weighting engine. Routes TC signals based on active regime.

PHILOSOPHY
----------
Not all signals are created equal. The macro regime (NUCLEUS) determines which
TCs are amplified and which are suppressed. This module:
  1. Takes raw TC signals
  2. Applies regime-based weights
  3. Filters out TCs not appropriate for current regime
  4. Outputs weighted, regime-adjusted signals

ROUTING RULES
-------------
TRENDING     → Weight TC-01/07/08/11 at 1.0x, suppress mean reversion TCs to 0.3x
CORRECTIVE   → Weight TC-02/06/10/13 at 1.0x, suppress trend-following TCs to 0.3x
HIGH_VOL     → Keep only TC-02/06 at 0.5x, suppress all others to 0.0x
RANGING      → Weight TC-02/06/13 at 1.0x, suppress breakout TCs to 0.0x

CONFLICT RESOLUTION
-------------------
If multiple TCs fire on the same bar:
  - Same direction → combine scores (weighted average)
  - Opposite directions → use highest-confidence signal if confidence > 0.70,
                          else output FLAT

Usage
-----
    from aggregation.signal_router import SignalRouter
    from aggregation.regime_classifier import RegimeState
    from technicals.base_signal import TechnicalSignal
    
    router = SignalRouter()
    tc_signals = [tc1.compute(df), tc2.compute(df), ...]
    regime = RegimeState(regime="TRENDING", ...)
    
    routed = router.route(tc_signals, regime)
    print(routed.direction, routed.score)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import from existing modules
try:
    from technical.bar_snapshot import Signal, Direction, SignalStrength
    from aggregation.regime_classifier import RegimeState
except ImportError:
    # Fallback for standalone testing
    from enum import Enum
    from dataclasses import dataclass as _dataclass
    
    class Direction(str, Enum):
        LONG = "LONG"
        SHORT = "SHORT"
        FLAT = "FLAT"
    
    class SignalStrength(str, Enum):
        STRONG_BUY = "STRONG_BUY"
        BUY = "BUY"
        HOLD = "HOLD"
        SELL = "SELL"
        STRONG_SELL = "STRONG_SELL"
    
    @_dataclass
    class Signal:
        direction: Direction
        strength: SignalStrength
        score: float
        confidence: float
        stop_loss: float
        take_profit: float
        metadata: Dict = field(default_factory=dict)
    
    @_dataclass
    class RegimeState:
        regime: str
        active_tc_ids: List[str]
        confidence: float = 1.0


# ── Regime → TC weight mappings ───────────────────────────────────────────────

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "TRENDING": {
        "TC-01": 1.0,   # Supertrend
        "TC-07": 1.0,   # ADX Gate
        "TC-08": 1.0,   # MA Cross
        "TC-11": 1.0,   # ChoCH
        "TC-02": 0.3,   # BB+RSI2 (mean reversion)
        "TC-06": 0.3,   # VWAP deviation
        "TC-10": 0.5,   # Liquidity sweep
        "TC-13": 0.3,   # Stoch RSI
    },
    "CORRECTIVE": {
        "TC-02": 1.0,   # BB+RSI2
        "TC-06": 1.0,   # VWAP deviation
        "TC-10": 1.0,   # Liquidity sweep
        "TC-13": 1.0,   # Stoch RSI
        "TC-01": 0.3,   # Supertrend
        "TC-07": 0.3,   # ADX Gate
        "TC-08": 0.3,   # MA Cross
        "TC-11": 0.5,   # ChoCH
    },
    "HIGH_VOL": {
        "TC-02": 0.5,   # BB+RSI2 only
        "TC-06": 0.5,   # VWAP deviation only
        # All others suppressed to 0.0
    },
    "RANGING": {
        "TC-02": 1.0,   # BB+RSI2
        "TC-06": 1.0,   # VWAP deviation
        "TC-13": 1.0,   # Stoch RSI
        "TC-10": 0.5,   # Liquidity sweep (cautious)
        # Breakout/trend-following suppressed to 0.0
    },
}


# ── Conflict resolution ───────────────────────────────────────────────────────

HIGH_CONFIDENCE_THRESHOLD = 0.70  # Override conflicts if one signal > this


# ── Routed Signal Output ──────────────────────────────────────────────────────

@dataclass
class RoutedSignal:
    """
    Output from signal router. Combines multiple TC signals into one weighted signal.
    """
    direction:       Direction
    strength:        SignalStrength
    score:           float              # -1.0 to +1.0
    confidence:      float              # 0.0 to 1.0
    stop_loss:       float
    take_profit:     float
    
    # Metadata
    regime:          str
    contributing_tcs: List[str]        # TC IDs that contributed to this signal
    tc_scores:       Dict[str, float]  # Individual TC scores (weighted)
    conflict_count:  int = 0           # Number of conflicting signals suppressed
    metadata:        Dict = field(default_factory=dict)
    
    def to_signal(self) -> Signal:
        """Convert to standard Signal object for execution."""
        return Signal(
            direction=self.direction,
            strength=self.strength,
            score=self.score,
            confidence=self.confidence,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            metadata={
                "regime": self.regime,
                "contributing_tcs": self.contributing_tcs,
                "tc_scores": self.tc_scores,
                "conflict_count": self.conflict_count,
                **self.metadata,
            },
        )


# ── Signal Router ─────────────────────────────────────────────────────────────

class SignalRouter:
    """
    Routes and weights TC signals based on active macro regime.
    This is the signal aggregation layer that sits between TCs and execution.
    """
    
    def __init__(self, regime_weights: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize router with regime weight mappings.
        
        Parameters
        ----------
        regime_weights : dict, optional
            Custom regime → TC weight mappings. If None, uses REGIME_WEIGHTS.
        """
        self.regime_weights = regime_weights or REGIME_WEIGHTS
    
    def route(
        self,
        tc_signals: List[Tuple[str, Signal]],  # List of (tc_id, signal)
        regime: RegimeState,
    ) -> RoutedSignal:
        """
        Route and weight TC signals based on regime.
        
        Parameters
        ----------
        tc_signals : List[Tuple[str, Signal]]
            List of (tc_id, signal) tuples from active TCs.
            Example: [("TC-01", signal1), ("TC-02", signal2), ...]
        
        regime : RegimeState
            Current macro regime from RegimeClassifier.
        
        Returns
        -------
        RoutedSignal
            Weighted, regime-adjusted signal ready for execution.
        """
        if not tc_signals:
            return self._flat_signal(regime)
        
        # ── Get regime weights ────────────────────────────────────────────────
        weights = self.regime_weights.get(regime.regime, {})
        
        # ── Apply regime weights and filter ───────────────────────────────────
        weighted_signals: List[Tuple[str, Signal, float]] = []  # (tc_id, signal, weight)
        
        for tc_id, signal in tc_signals:
            if signal.direction == Direction.FLAT:
                continue
            
            weight = weights.get(tc_id, 0.0)
            if weight == 0.0:
                logger.debug(f"Suppressed {tc_id} (weight=0.0 in {regime.regime} regime)")
                continue
            
            weighted_signals.append((tc_id, signal, weight))
        
        if not weighted_signals:
            return self._flat_signal(regime)
        
        # ── Aggregate signals ─────────────────────────────────────────────────
        return self._aggregate(weighted_signals, regime)
    
    def _aggregate(
        self,
        weighted_signals: List[Tuple[str, Signal, float]],
        regime: RegimeState,
    ) -> RoutedSignal:
        """
        Aggregate multiple weighted signals into one.
        
        Logic:
          - If all signals same direction → weighted average
          - If mixed directions:
              * If one signal confidence > 0.70 → use that signal
              * Else → FLAT (conflict)
        """
        if len(weighted_signals) == 1:
            tc_id, signal, weight = weighted_signals[0]
            return self._single_signal(tc_id, signal, weight, regime)
        
        # ── Check for directional conflicts ───────────────────────────────────
        directions = [sig.direction for _, sig, _ in weighted_signals]
        has_long = Direction.LONG in directions
        has_short = Direction.SHORT in directions
        
        if has_long and has_short:
            # Conflict: check for high-confidence override
            high_conf_signals = [
                (tc_id, sig, wt)
                for tc_id, sig, wt in weighted_signals
                if sig.confidence > HIGH_CONFIDENCE_THRESHOLD
            ]
            
            if len(high_conf_signals) == 1:
                tc_id, signal, weight = high_conf_signals[0]
                logger.info(f"Conflict resolved by high-confidence {tc_id} (conf={signal.confidence:.2%})")
                return self._single_signal(tc_id, signal, weight, regime, conflict_count=len(weighted_signals) - 1)
            
            # No clear winner → FLAT
            logger.warning(f"Directional conflict in {regime.regime}: {len(weighted_signals)} signals, no high-confidence override")
            return self._flat_signal(regime, conflict_count=len(weighted_signals))
        
        # ── All signals same direction → weighted average ─────────────────────
        total_weight = sum(wt for _, _, wt in weighted_signals)
        
        # Weighted scores
        weighted_score = sum(sig.score * wt for _, sig, wt in weighted_signals) / total_weight
        weighted_conf = sum(sig.confidence * wt for _, sig, wt in weighted_signals) / total_weight
        
        # Direction (consensus)
        direction = Direction.LONG if has_long else Direction.SHORT
        
        # Stop/target (use tightest stop, furthest target)
        if direction == Direction.LONG:
            stop = max(sig.stop_loss for _, sig, _ in weighted_signals)
            target = max(sig.take_profit for _, sig, _ in weighted_signals)
        else:
            stop = min(sig.stop_loss for _, sig, _ in weighted_signals)
            target = min(sig.take_profit for _, sig, _ in weighted_signals)
        
        # Strength
        strength = self._score_to_strength(weighted_score, direction)
        
        # Metadata
        contributing_tcs = [tc_id for tc_id, _, _ in weighted_signals]
        tc_scores = {tc_id: sig.score * wt for tc_id, sig, wt in weighted_signals}
        
        return RoutedSignal(
            direction=direction,
            strength=strength,
            score=weighted_score,
            confidence=weighted_conf,
            stop_loss=stop,
            take_profit=target,
            regime=regime.regime,
            contributing_tcs=contributing_tcs,
            tc_scores=tc_scores,
            conflict_count=0,
        )
    
    def _single_signal(
        self,
        tc_id: str,
        signal: Signal,
        weight: float,
        regime: RegimeState,
        conflict_count: int = 0,
    ) -> RoutedSignal:
        """Convert single TC signal to RoutedSignal."""
        return RoutedSignal(
            direction=signal.direction,
            strength=signal.strength,
            score=signal.score * weight,
            confidence=signal.confidence * weight,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            regime=regime.regime,
            contributing_tcs=[tc_id],
            tc_scores={tc_id: signal.score * weight},
            conflict_count=conflict_count,
            metadata=signal.metadata.copy() if hasattr(signal, 'metadata') else {},
        )
    
    def _flat_signal(self, regime: RegimeState, conflict_count: int = 0) -> RoutedSignal:
        """Return FLAT signal (no trade)."""
        return RoutedSignal(
            direction=Direction.FLAT,
            strength=SignalStrength.HOLD,
            score=0.0,
            confidence=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            regime=regime.regime,
            contributing_tcs=[],
            tc_scores={},
            conflict_count=conflict_count,
        )
    
    @staticmethod
    def _score_to_strength(score: float, direction: Direction) -> SignalStrength:
        """Convert numeric score to SignalStrength enum."""
        abs_score = abs(score)
        
        if direction == Direction.LONG:
            if abs_score >= 0.75:
                return SignalStrength.STRONG_BUY
            elif abs_score >= 0.40:
                return SignalStrength.BUY
            else:
                return SignalStrength.HOLD
        elif direction == Direction.SHORT:
            if abs_score >= 0.75:
                return SignalStrength.STRONG_SELL
            elif abs_score >= 0.40:
                return SignalStrength.SELL
            else:
                return SignalStrength.HOLD
        else:
            return SignalStrength.HOLD


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("=" * 80)
    print("  SIGNAL ROUTER — Example")
    print("=" * 80)
    
    # Mock signals
    signal1 = Signal(
        direction=Direction.LONG,
        strength=SignalStrength.BUY,
        score=0.65,
        confidence=0.80,
        stop_loss=495.0,
        take_profit=510.0,
        metadata={"tc_id": "TC-01"},
    )
    
    signal2 = Signal(
        direction=Direction.LONG,
        strength=SignalStrength.STRONG_BUY,
        score=0.85,
        confidence=0.90,
        stop_loss=496.0,
        take_profit=512.0,
        metadata={"tc_id": "TC-07"},
    )
    
    tc_signals = [("TC-01", signal1), ("TC-07", signal2)]
    
    # Mock regime
    class MockRegime:
        regime = "TRENDING"
        active_tc_ids = ["TC-01", "TC-07"]
        confidence = 0.85
    
    regime = MockRegime()
    
    # Route
    router = SignalRouter()
    routed = router.route(tc_signals, regime)
    
    print(f"\nInput Signals:")
    print(f"  TC-01: {signal1.direction.value} @ {signal1.score:.2f} (conf={signal1.confidence:.2%})")
    print(f"  TC-07: {signal2.direction.value} @ {signal2.score:.2f} (conf={signal2.confidence:.2%})")
    
    print(f"\nRegime: {regime.regime} (conf={regime.confidence:.2%})")
    
    print(f"\nRouted Signal:")
    print(f"  Direction         : {routed.direction.value}")
    print(f"  Strength          : {routed.strength.value}")
    print(f"  Score             : {routed.score:.3f}")
    print(f"  Confidence        : {routed.confidence:.2%}")
    print(f"  Stop / Target     : {routed.stop_loss:.2f} / {routed.take_profit:.2f}")
    print(f"  Contributing TCs  : {', '.join(routed.contributing_tcs)}")
    print(f"  TC Scores         : {routed.tc_scores}")
    print("=" * 80)
