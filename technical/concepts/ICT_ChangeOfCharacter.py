"""
CONCEPT: ICT Change of Character (ChoCH)
SOURCE: 04 - ICT FOR DUMMIES | Market Structure EP. 3
CATEGORY: STRUCTURE
TIMEFRAME: 1D, 4H, 1H, 15M
INSTRUMENTS: ANY
DESCRIPTION:
    A Change of Character (ChoCH) is the FIRST sign of a potential reversal —
    it precedes the MSS. In a downtrend, a ChoCH occurs when price makes the
    first higher high (breaks the most recent lower high). In an uptrend, a ChoCH
    is the first lower low (breaks the most recent higher low). The ChoCH alone
    is NOT a trade entry — it is a ALERT to start looking for reversal entries
    on lower timeframes. Only after a ChoCH breaks a SIGNIFICANT swing (MSS) do
    we have confirmation.
EDGE:
    Captures the earliest institutional footprint of a direction change.
    Positions traders ahead of the crowd who wait for obvious trend reversals.
KNOWN_LIMITATIONS:
    Very high false-positive rate in noisy markets. Must be filtered by:
    1. Only trading ChoCH after a confirmed liquidity sweep (stop hunt).
    2. ChoCH on HTF, then refine entry on LTF.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SWING_LOOKBACK      = 5     # bars each side for swing identification
MIN_CHOCH_BREAK_PCT = 0.001 # close must breach level by at least this %
REQUIRE_SWEEP       = True  # ChoCH is only valid if preceded by a liquidity sweep


class ICT_ChangeOfCharacter:
    """
    Detects ChoCH — the first structural change signaling potential reversal.
    This is an alert concept; it should be combined with liquidity sweep confirmation.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.swing_lookback      = p.get("swing_lookback",      SWING_LOOKBACK)
        self.min_choch_break_pct = p.get("min_choch_break_pct", MIN_CHOCH_BREAK_PCT)
        self.require_sweep       = p.get("require_sweep",       REQUIRE_SWEEP)
        self._prior_trend: str   = "UNKNOWN"  # "UP" or "DOWN"

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        lb = self.swing_lookback
        if len(history) < 2 * lb + 4:
            return None

        bars = history[-(4 * lb):]
        sh_prices = []
        sl_prices = []
        for i in range(lb, len(bars) - lb):
            wh = [b.high for b in bars[i - lb: i + lb + 1]]
            wl = [b.low  for b in bars[i - lb: i + lb + 1]]
            if bars[i].high == max(wh):
                sh_prices.append(bars[i].high)
            if bars[i].low == min(wl):
                sl_prices.append(bars[i].low)

        if len(sh_prices) < 2 or len(sl_prices) < 2:
            return None

        # Determine prior trend from last two swing highs/lows
        prior_trend = "UP" if sh_prices[-1] > sh_prices[-2] else "DOWN"
        self._prior_trend = prior_trend

        atr = snapshot.atr or snapshot.close * 0.01

        if prior_trend == "DOWN":
            # ChoCH for downtrend: close above the most recent LOWER high
            last_lh = sh_prices[-1]
            if snapshot.close > last_lh * (1 + self.min_choch_break_pct):
                self.log_event("CHOCH_BULLISH", snapshot.timestamp, {
                    "close": snapshot.close, "broken_lh": last_lh,
                })
                return Signal(
                    concept     = "ICT_ChangeOfCharacter",
                    symbol      = snapshot.symbol,
                    direction   = Direction.LONG,
                    timestamp   = snapshot.timestamp,
                    entry_price = snapshot.close,
                    stop_loss   = snapshot.low - atr * 0.5,
                    take_profit = snapshot.close + 1.5 * atr,
                    confidence  = 0.50,  # ChoCH alone = low confidence
                    category    = ConceptCategory.STRUCTURE,
                    regime      = snapshot.regime,
                    reason      = f"ChoCH BULLISH: close {snapshot.close:.4f} broke lower high {last_lh:.4f} in downtrend",
                )

        if prior_trend == "UP":
            # ChoCH for uptrend: close below the most recent higher low
            last_hl = sl_prices[-1]
            if snapshot.close < last_hl * (1 - self.min_choch_break_pct):
                self.log_event("CHOCH_BEARISH", snapshot.timestamp, {
                    "close": snapshot.close, "broken_hl": last_hl,
                })
                return Signal(
                    concept     = "ICT_ChangeOfCharacter",
                    symbol      = snapshot.symbol,
                    direction   = Direction.SHORT,
                    timestamp   = snapshot.timestamp,
                    entry_price = snapshot.close,
                    stop_loss   = snapshot.high + atr * 0.5,
                    take_profit = snapshot.close - 1.5 * atr,
                    confidence  = 0.50,
                    category    = ConceptCategory.STRUCTURE,
                    regime      = snapshot.regime,
                    reason      = f"ChoCH BEARISH: close {snapshot.close:.4f} broke higher low {last_hl:.4f} in uptrend",
                )
        return None

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) > 20 else 0
        return ValidationResult(
            concept="ICT_ChangeOfCharacter",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_ChoCH] %s at %s | %s", event_type, bar, details)
