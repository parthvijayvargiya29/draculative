"""
CONCEPT: ICT Premium and Discount Zones
SOURCE: 09 - ICT FOR DUMMIES | Premium and Discount EP. 8
CATEGORY: PREMIUM_DISCOUNT
TIMEFRAME: 1D, 4H, 1H, 15M
INSTRUMENTS: ANY
DESCRIPTION:
    The market is always either in a Premium (above 50% of the prior swing range)
    or a Discount (below 50%). Institutions BUY in discount and SELL in premium.
    The equilibrium (50%) level is the dividing line.

    The Optimal Trade Entry (OTE) zone sits at the 61.8%–78.6% Fibonacci retracement.
    This is the deepest discount in a bullish leg / deepest premium in a bearish leg.
    OTE entries provide the best risk-reward because stop placement is close and
    the potential move to the prior extreme (or beyond) is large.

    Premium: price > 50% of (swing high - swing low) — look to SELL
    Discount: price < 50% of (swing high - swing low) — look to BUY
    OTE Bullish: 61.8%–78.6% retracement of the prior up-leg = LONG entry zone
    OTE Bearish: 61.8%–78.6% retracement of the prior down-leg = SHORT entry zone

EDGE:
    Institutions accumulate at discounts (filling buy orders without moving price
    adversely) and distribute at premiums (filling sell orders). Retail traders
    buy breakouts at premiums and sell breakdowns at discounts — the opposite.
KNOWN_LIMITATIONS:
    Requires a clear swing high and swing low to define the range. In a strong
    trending market, OTE may not be reached before price continues.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SWING_LOOKBACK    = 10     # bars each side for major swing identification
OTE_LOW_FIB       = 0.618  # OTE lower Fibonacci level (61.8%)
OTE_HIGH_FIB      = 0.786  # OTE upper Fibonacci level (78.6%)
EQUILIBRIUM_FIB   = 0.500  # 50% midpoint — premium vs discount divider
MIN_SWING_ATR_MULT = 3.0   # swing must be >= this * ATR to be significant


@dataclass
class SwingRange:
    high:       float
    low:        float
    direction:  str     # "BULLISH" (up-leg) or "BEARISH" (down-leg)
    equilibrium: float  = 0.0
    ote_high:   float   = 0.0
    ote_low:    float   = 0.0

    def __post_init__(self):
        rng = self.high - self.low
        self.equilibrium = self.low + rng * EQUILIBRIUM_FIB
        if self.direction == "BULLISH":
            # Retracement from high downward
            self.ote_high = self.high - rng * OTE_LOW_FIB
            self.ote_low  = self.high - rng * OTE_HIGH_FIB
        else:
            # Retracement from low upward
            self.ote_low  = self.low + rng * OTE_LOW_FIB
            self.ote_high = self.low + rng * OTE_HIGH_FIB

    def is_in_ote(self, price: float) -> bool:
        return self.ote_low <= price <= self.ote_high

    def is_premium(self, price: float) -> bool:
        return price > self.equilibrium

    def is_discount(self, price: float) -> bool:
        return price < self.equilibrium


class ICT_PremiumDiscount:
    """
    Identifies premium/discount zones and generates OTE entry signals.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.swing_lookback      = p.get("swing_lookback",       SWING_LOOKBACK)
        self.ote_low_fib         = p.get("ote_low_fib",          OTE_LOW_FIB)
        self.ote_high_fib        = p.get("ote_high_fib",         OTE_HIGH_FIB)
        self.min_swing_atr_mult  = p.get("min_swing_atr_mult",   MIN_SWING_ATR_MULT)
        self._current_range: Optional[SwingRange] = None

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        lb = self.swing_lookback
        if len(history) < 2 * lb + 2:
            return None

        atr = snapshot.atr or snapshot.close * 0.01
        self._current_range = self._identify_swing_range(history, atr)

        if self._current_range is None:
            return None

        sr = self._current_range

        # Bullish setup: price is in discount AND in the OTE zone → LONG
        if sr.direction == "BULLISH" and sr.is_discount(snapshot.close) and sr.is_in_ote(snapshot.close):
            self.log_event("OTE_BULLISH", snapshot.timestamp, {
                "price": snapshot.close,
                "ote_low": sr.ote_low,
                "ote_high": sr.ote_high,
                "equilibrium": sr.equilibrium,
            })
            return Signal(
                concept     = "ICT_PremiumDiscount",
                symbol      = snapshot.symbol,
                direction   = Direction.LONG,
                timestamp   = snapshot.timestamp,
                entry_price = snapshot.close,
                stop_loss   = sr.low - atr * 0.25,
                take_profit = sr.high,
                confidence  = 0.65,
                category    = ConceptCategory.PREMIUM_DISCOUNT,
                regime      = snapshot.regime,
                reason      = f"Price {snapshot.close:.4f} in OTE bullish zone ({sr.ote_low:.4f}–{sr.ote_high:.4f}), discount zone",
            )

        # Bearish setup: price is in premium AND in the OTE zone → SHORT
        if sr.direction == "BEARISH" and sr.is_premium(snapshot.close) and sr.is_in_ote(snapshot.close):
            self.log_event("OTE_BEARISH", snapshot.timestamp, {
                "price": snapshot.close,
                "ote_low": sr.ote_low,
                "ote_high": sr.ote_high,
                "equilibrium": sr.equilibrium,
            })
            return Signal(
                concept     = "ICT_PremiumDiscount",
                symbol      = snapshot.symbol,
                direction   = Direction.SHORT,
                timestamp   = snapshot.timestamp,
                entry_price = snapshot.close,
                stop_loss   = sr.high + atr * 0.25,
                take_profit = sr.low,
                confidence  = 0.65,
                category    = ConceptCategory.PREMIUM_DISCOUNT,
                regime      = snapshot.regime,
                reason      = f"Price {snapshot.close:.4f} in OTE bearish zone ({sr.ote_low:.4f}–{sr.ote_high:.4f}), premium zone",
            )
        return None

    def _identify_swing_range(self, history: List[BarSnapshot],
                               atr: float) -> Optional[SwingRange]:
        """Find the most recent significant swing range."""
        lb = self.swing_lookback
        bars = history[-(3 * lb):]
        if len(bars) < 2 * lb + 1:
            return None

        sh_idx, sh_price = None, 0.0
        sl_idx, sl_price = None, float("inf")

        for i in range(lb, len(bars) - lb):
            wh = [b.high for b in bars[i - lb: i + lb + 1]]
            wl = [b.low  for b in bars[i - lb: i + lb + 1]]
            if bars[i].high == max(wh) and bars[i].high > sh_price:
                sh_idx, sh_price = i, bars[i].high
            if bars[i].low == min(wl) and bars[i].low < sl_price:
                sl_idx, sl_price = i, bars[i].low

        if sh_idx is None or sl_idx is None:
            return None

        swing_range = sh_price - sl_price
        if swing_range < self.min_swing_atr_mult * atr:
            return None  # insignificant swing

        direction = "BULLISH" if sh_idx > sl_idx else "BEARISH"
        return SwingRange(high=sh_price, low=sl_price, direction=direction)

    def get_current_zone(self, price: float) -> str:
        """Returns 'PREMIUM', 'DISCOUNT', or 'EQUILIBRIUM' for a given price."""
        if self._current_range is None:
            return "UNKNOWN"
        if self._current_range.is_premium(price):
            return "PREMIUM"
        if self._current_range.is_discount(price):
            return "DISCOUNT"
        return "EQUILIBRIUM"

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= 2 * self.swing_lookback + 2 else 0
        return ValidationResult(
            concept="ICT_PremiumDiscount",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_PD] %s at %s | %s", event_type, bar, details)
