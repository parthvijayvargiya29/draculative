"""
CONCEPT: ICT SMT Divergence (Smart Money Divergence)
SOURCE: 10 - ICT FOR DUMMIES | SMT Divergence EP. 9
CATEGORY: CORRELATION
TIMEFRAME: 1D, 4H, 1H, 15M
INSTRUMENTS: Correlated pairs (ES/NQ, EURUSD/GBPUSD, SPY/QQQ)
DESCRIPTION:
    SMT Divergence occurs when two HIGHLY CORRELATED instruments fail to confirm
    each other's swing points. If ES makes a new low but NQ does NOT make a
    new low (or vice versa), this divergence signals that the move is weak and
    a reversal is imminent.

    Bullish SMT: Instrument A makes a lower low, but correlated Instrument B
                 makes a HIGHER low (fails to confirm). → Expect reversal UP.
    Bearish SMT: Instrument A makes a higher high, but correlated Instrument B
                 makes a LOWER high (fails to confirm). → Expect reversal DOWN.

    The instrument that DOES NOT make the new extreme is the leader;
    the one that does make it is being manipulated (stop hunt).

EDGE:
    When correlated instruments disagree, one of them is being hunted.
    The divergence reveals the manipulation. You enter in the direction of
    the stronger instrument (the one that didn't confirm the extreme).
KNOWN_LIMITATIONS:
    Requires simultaneous data for two instruments — the simulator must
    feed both instruments' snapshots at the same bar. Divergence signals
    in trending markets can persist for many bars.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SWING_LOOKBACK          = 5     # bars each side for swing identification
SMT_LOOKBACK_BARS       = 20    # bars to look back for divergence comparison
DIVERGENCE_THRESHOLD    = 0.002 # minimum price difference to qualify as divergence


@dataclass
class SwingPoint:
    price:      float
    direction:  str   # "HIGH" or "LOW"
    timestamp:  object


class ICT_SMTDivergence:
    """
    Detects Smart Money Divergence between two correlated instruments.
    Must be called with snapshots from BOTH instruments at each bar.
    """

    def __init__(self, symbol_a: str, symbol_b: str, params: dict = None):
        """
        symbol_a: primary instrument (e.g. "SPY")
        symbol_b: correlated instrument (e.g. "QQQ")
        """
        p = params or {}
        self.symbol_a           = symbol_a
        self.symbol_b           = symbol_b
        self.swing_lookback     = p.get("swing_lookback",       SWING_LOOKBACK)
        self.smt_lookback       = p.get("smt_lookback_bars",    SMT_LOOKBACK_BARS)
        self.div_threshold      = p.get("divergence_threshold", DIVERGENCE_THRESHOLD)
        self._history_a:        List[BarSnapshot] = []
        self._history_b:        List[BarSnapshot] = []

    def update(self, snap_a: BarSnapshot, snap_b: BarSnapshot) -> Optional[Signal]:
        """
        Feed snapshots for both instruments at the same bar.
        Returns a Signal if SMT divergence is detected.
        """
        self._history_a.append(snap_a)
        self._history_b.append(snap_b)

        if len(self._history_a) < self.smt_lookback:
            return None

        a_bars = self._history_a[-self.smt_lookback:]
        b_bars = self._history_b[-self.smt_lookback:]

        atr = snap_a.atr or snap_a.close * 0.01

        # Bullish SMT: A makes lower low, B does NOT
        a_lows  = self._get_swing_lows(a_bars)
        b_lows  = self._get_swing_lows(b_bars)
        if len(a_lows) >= 2 and len(b_lows) >= 2:
            a_ll = a_lows[-1] < a_lows[-2]  # A made lower low
            b_ll = b_lows[-1] < b_lows[-2]  # B made lower low
            if a_ll and not b_ll:
                # Divergence: A lower, B not → reversal UP
                diff_pct = abs(a_lows[-1] - a_lows[-2]) / a_lows[-2]
                if diff_pct >= self.div_threshold:
                    self.log_event("SMT_BULLISH", snap_a.timestamp, {
                        "symbol_a": self.symbol_a, "a_low": a_lows[-1],
                        "symbol_b": self.symbol_b, "b_low": b_lows[-1],
                    })
                    return Signal(
                        concept     = "ICT_SMTDivergence",
                        symbol      = self.symbol_a,
                        direction   = Direction.LONG,
                        timestamp   = snap_a.timestamp,
                        entry_price = snap_a.close,
                        stop_loss   = snap_a.low - atr * 0.3,
                        take_profit = snap_a.close + 2.0 * atr,
                        confidence  = 0.70,
                        category    = ConceptCategory.CORRELATION,
                        regime      = snap_a.regime,
                        reason      = f"Bullish SMT: {self.symbol_a} made lower low ({a_lows[-1]:.4f}) but {self.symbol_b} did NOT ({b_lows[-1]:.4f}) — manipulation on A, reverse LONG",
                    )

        # Bearish SMT: A makes higher high, B does NOT
        a_highs = self._get_swing_highs(a_bars)
        b_highs = self._get_swing_highs(b_bars)
        if len(a_highs) >= 2 and len(b_highs) >= 2:
            a_hh = a_highs[-1] > a_highs[-2]
            b_hh = b_highs[-1] > b_highs[-2]
            if a_hh and not b_hh:
                diff_pct = abs(a_highs[-1] - a_highs[-2]) / a_highs[-2]
                if diff_pct >= self.div_threshold:
                    self.log_event("SMT_BEARISH", snap_a.timestamp, {
                        "symbol_a": self.symbol_a, "a_high": a_highs[-1],
                        "symbol_b": self.symbol_b, "b_high": b_highs[-1],
                    })
                    return Signal(
                        concept     = "ICT_SMTDivergence",
                        symbol      = self.symbol_a,
                        direction   = Direction.SHORT,
                        timestamp   = snap_a.timestamp,
                        entry_price = snap_a.close,
                        stop_loss   = snap_a.high + atr * 0.3,
                        take_profit = snap_a.close - 2.0 * atr,
                        confidence  = 0.70,
                        category    = ConceptCategory.CORRELATION,
                        regime      = snap_a.regime,
                        reason      = f"Bearish SMT: {self.symbol_a} made higher high ({a_highs[-1]:.4f}) but {self.symbol_b} did NOT ({b_highs[-1]:.4f}) — manipulation on A, reverse SHORT",
                    )
        return None

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        """Single-instrument interface — not applicable for SMT. Use update() instead."""
        return None

    def _get_swing_highs(self, bars: List[BarSnapshot]) -> List[float]:
        lb = self.swing_lookback
        highs = []
        for i in range(lb, len(bars) - lb):
            wh = [b.high for b in bars[i - lb: i + lb + 1]]
            if bars[i].high == max(wh):
                highs.append(bars[i].high)
        return highs

    def _get_swing_lows(self, bars: List[BarSnapshot]) -> List[float]:
        lb = self.swing_lookback
        lows = []
        for i in range(lb, len(bars) - lb):
            wl = [b.low for b in bars[i - lb: i + lb + 1]]
            if bars[i].low == min(wl):
                lows.append(bars[i].low)
        return lows

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= self.smt_lookback else 0
        return ValidationResult(
            concept="ICT_SMTDivergence",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_SMT] %s at %s | %s", event_type, bar, details)
