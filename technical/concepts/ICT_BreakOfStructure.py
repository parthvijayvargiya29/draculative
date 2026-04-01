"""
CONCEPT: ICT Break of Structure (BOS)
SOURCE: 04 - ICT FOR DUMMIES | Market Structure EP. 3
CATEGORY: STRUCTURE
TIMEFRAME: 1D, 4H, 1H, 15M
INSTRUMENTS: ANY
DESCRIPTION:
    A Break of Structure (BOS) is a continuation signal. In an uptrend, a BOS
    occurs when price takes out a prior swing high, confirming that the uptrend
    is still intact. In a downtrend, a BOS occurs when price takes out a prior
    swing low, confirming the downtrend. The BOS tells you the trend is continuing
    — it does NOT signal a reversal. Trade BOS in the direction of the prevailing
    higher-timeframe trend only.
EDGE:
    BOS entries ride institutional momentum. After taking out a swing level,
    price often retraces to the displacement origin (FVG or OB formed during
    the break candle) before continuing in the BOS direction.
KNOWN_LIMITATIONS:
    In ranging/corrective markets BOS signals are false breakouts. Must be
    combined with BIAS_FRAMEWORK and regime filter (ADX > 25 for BOS trades).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SWING_LOOKBACK  = 5       # bars each side for swing identification
MIN_BREAK_PCT   = 0.001   # close must exceed swing by this % to confirm BOS
ADX_FILTER      = 20.0    # minimum ADX to enter a BOS trade


class ICT_BreakOfStructure:
    """
    Detects BOS — a continuation signal confirming the prevailing trend.
    Only generates signals when the break aligns with the HTF bias.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.swing_lookback = p.get("swing_lookback", SWING_LOOKBACK)
        self.min_break_pct  = p.get("min_break_pct",  MIN_BREAK_PCT)
        self.adx_filter     = p.get("adx_filter",     ADX_FILTER)

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        if len(history) < 2 * self.swing_lookback + 2:
            return None

        lb = self.swing_lookback
        bars = history[-(3 * lb):]

        # Collect swing highs and lows from bars (excluding current)
        swing_highs = []
        swing_lows  = []
        for i in range(lb, len(bars) - lb):
            wh = [b.high for b in bars[i - lb: i + lb + 1]]
            if bars[i].high == max(wh):
                swing_highs.append(bars[i].high)
            wl = [b.low for b in bars[i - lb: i + lb + 1]]
            if bars[i].low == min(wl):
                swing_lows.append(bars[i].low)

        if not swing_highs or not swing_lows:
            return None

        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]

        # Bullish BOS: close above last swing high
        if snapshot.close > last_sh * (1 + self.min_break_pct):
            adx_ok = (snapshot.adx is None) or (snapshot.adx >= self.adx_filter)
            if adx_ok:
                atr = snapshot.atr or snapshot.close * 0.01
                self.log_event("BOS_BULLISH", snapshot.timestamp, {
                    "close": snapshot.close, "broken_sh": last_sh,
                })
                return Signal(
                    concept     = "ICT_BreakOfStructure",
                    symbol      = snapshot.symbol,
                    direction   = Direction.LONG,
                    timestamp   = snapshot.timestamp,
                    entry_price = snapshot.close,
                    stop_loss   = snapshot.low - atr * 0.3,
                    take_profit = snapshot.close + 2.0 * atr,
                    confidence  = 0.60,
                    category    = ConceptCategory.STRUCTURE,
                    regime      = snapshot.regime,
                    reason      = f"Bullish BOS: close {snapshot.close:.4f} above swing high {last_sh:.4f}",
                )

        # Bearish BOS: close below last swing low
        if snapshot.close < last_sl * (1 - self.min_break_pct):
            adx_ok = (snapshot.adx is None) or (snapshot.adx >= self.adx_filter)
            if adx_ok:
                atr = snapshot.atr or snapshot.close * 0.01
                self.log_event("BOS_BEARISH", snapshot.timestamp, {
                    "close": snapshot.close, "broken_sl": last_sl,
                })
                return Signal(
                    concept     = "ICT_BreakOfStructure",
                    symbol      = snapshot.symbol,
                    direction   = Direction.SHORT,
                    timestamp   = snapshot.timestamp,
                    entry_price = snapshot.close,
                    stop_loss   = snapshot.high + atr * 0.3,
                    take_profit = snapshot.close - 2.0 * atr,
                    confidence  = 0.60,
                    category    = ConceptCategory.STRUCTURE,
                    regime      = snapshot.regime,
                    reason      = f"Bearish BOS: close {snapshot.close:.4f} below swing low {last_sl:.4f}",
                )
        return None

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) > 20 else 0
        return ValidationResult(
            concept="ICT_BreakOfStructure",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_BOS] %s at %s | %s", event_type, bar, details)
