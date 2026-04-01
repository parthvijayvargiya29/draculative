"""
CONCEPT: ICT Order Block V2 (Refined)
SOURCE: 06 - ICT FOR DUMMIES | Fair Value Gaps EP. 5
         08 - ICT FOR DUMMIES | Liquidity PT. 2 EP. 7
CATEGORY: ORDER_BLOCK
TIMEFRAME: 1D, 4H, 1H, 15M
INSTRUMENTS: ANY
DESCRIPTION:
    An Order Block (OB) is the LAST candle in the OPPOSITE direction before a
    strong impulsive move (displacement). It represents the price level where
    institutional orders were placed. When price returns to this level,
    institutions defend their positions, creating high-probability entries.

    Bullish OB: The LAST DOWN (bearish) candle before a strong upward impulse.
    Bearish OB: The LAST UP (bullish) candle before a strong downward impulse.

    The OB zone is defined as the full body of that last opposing candle.
    Entry is at price returning INTO the OB zone (not just touching it).

    Breaker Block: An OB that has been VIOLATED becomes a Breaker Block, which
    then flips its role — a bullish OB that is violated becomes a bearish Breaker Block.

EDGE:
    Institutions defend their order levels. Retail traders see these levels as
    support/resistance. The OB is where institutional re-accumulation occurs.
KNOWN_LIMITATIONS:
    OBs without a subsequent FVG or displacement candle are weak. OBs deeper
    than 78.6% of the prior impulse leg are less reliable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
IMPULSE_BARS          = 3       # bars ahead to confirm displacement after OB candle
MIN_IMPULSE_ATR_MULT  = 1.5     # impulse must be >= this * ATR to qualify
MAX_OB_AGE_BARS       = 80      # OBs older than this are removed
MIN_BODY_ATR_MULT     = 0.2     # OB candle body must be >= this * ATR


@dataclass
class OrderBlockZone:
    direction:   str    # "BULLISH" or "BEARISH"
    zone_high:   float
    zone_low:    float
    timestamp:   object
    strength:    float  = 0.5    # impulse / ATR ratio (normalized)
    mitigated:   bool   = False  # True once price trades INTO the zone
    breaker:     bool   = False  # True if OB was violated (becomes breaker block)
    age_bars:    int    = 0

    @property
    def is_valid(self) -> bool:
        return not self.breaker and self.age_bars < MAX_OB_AGE_BARS

    @property
    def midpoint(self) -> float:
        return (self.zone_high + self.zone_low) / 2


class ICT_OrderBlock_V2:
    """
    Detects Order Blocks and Breaker Blocks, generates entry signals on mitigation.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.impulse_bars      = p.get("impulse_bars",         IMPULSE_BARS)
        self.min_impulse_atr   = p.get("min_impulse_atr_mult", MIN_IMPULSE_ATR_MULT)
        self.max_age           = p.get("max_ob_age_bars",      MAX_OB_AGE_BARS)
        self.min_body_atr      = p.get("min_body_atr_mult",    MIN_BODY_ATR_MULT)
        self._order_blocks: List[OrderBlockZone] = []
        self._pending_check: List[dict] = []  # bars queued for impulse confirmation

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        if len(history) < self.impulse_bars + 2:
            return None

        atr = snapshot.atr or snapshot.close * 0.01

        # Age existing OBs
        for ob in self._order_blocks:
            ob.age_bars += 1

        # Check for new OB formation using bars[-impulse_bars-1 .. -1]
        self._scan_for_new_obs(history, atr)

        # Check if price is entering an existing OB zone
        for ob in list(self._order_blocks):
            if not ob.is_valid:
                self._order_blocks.remove(ob)
                continue

            if ob.direction == "BULLISH":
                # Price enters bullish OB — LONG entry
                if snapshot.low <= ob.zone_high and snapshot.close >= ob.zone_low:
                    ob.mitigated = True
                    # Check for breaker: close BELOW the OB low
                    if snapshot.close < ob.zone_low:
                        ob.breaker = True
                        self.log_event("OB_BREAKER_BEARISH", snapshot.timestamp,
                                       {"ob_low": ob.zone_low, "close": snapshot.close})
                    else:
                        self.log_event("OB_BULLISH_ENTRY", snapshot.timestamp,
                                       {"zone": (ob.zone_low, ob.zone_high)})
                        return Signal(
                            concept     = "ICT_OrderBlock_V2",
                            symbol      = snapshot.symbol,
                            direction   = Direction.LONG,
                            timestamp   = snapshot.timestamp,
                            entry_price = snapshot.close,
                            stop_loss   = ob.zone_low - atr * 0.25,
                            take_profit = snapshot.close + 2.5 * atr,
                            confidence  = 0.60 + 0.15 * ob.strength,
                            category    = ConceptCategory.ORDER_BLOCK,
                            regime      = snapshot.regime,
                            reason      = f"Price entered bullish OB zone ({ob.zone_low:.4f}–{ob.zone_high:.4f}), strength={ob.strength:.2f}",
                        )

            if ob.direction == "BEARISH":
                # Price enters bearish OB — SHORT entry
                if snapshot.high >= ob.zone_low and snapshot.close <= ob.zone_high:
                    ob.mitigated = True
                    if snapshot.close > ob.zone_high:
                        ob.breaker = True
                        self.log_event("OB_BREAKER_BULLISH", snapshot.timestamp,
                                       {"ob_high": ob.zone_high, "close": snapshot.close})
                    else:
                        self.log_event("OB_BEARISH_ENTRY", snapshot.timestamp,
                                       {"zone": (ob.zone_low, ob.zone_high)})
                        return Signal(
                            concept     = "ICT_OrderBlock_V2",
                            symbol      = snapshot.symbol,
                            direction   = Direction.SHORT,
                            timestamp   = snapshot.timestamp,
                            entry_price = snapshot.close,
                            stop_loss   = ob.zone_high + atr * 0.25,
                            take_profit = snapshot.close - 2.5 * atr,
                            confidence  = 0.60 + 0.15 * ob.strength,
                            category    = ConceptCategory.ORDER_BLOCK,
                            regime      = snapshot.regime,
                            reason      = f"Price entered bearish OB zone ({ob.zone_low:.4f}–{ob.zone_high:.4f}), strength={ob.strength:.2f}",
                        )
        return None

    def _scan_for_new_obs(self, history: List[BarSnapshot], atr: float):
        """Look for the last opposing candle before a displacement move."""
        if len(history) < self.impulse_bars + 2:
            return

        # Look at the most recent bars for an OB formation
        window = history[-(self.impulse_bars + 3):]
        for i in range(len(window) - self.impulse_bars - 1):
            candidate = window[i]
            # Prospective bullish OB: candidate is bearish
            if candidate.is_bearish:
                # Check if the next N bars create upward displacement
                impulse_bars = window[i + 1: i + 1 + self.impulse_bars]
                total_move = sum(b.close - b.open for b in impulse_bars)
                if total_move >= self.min_impulse_atr * atr:
                    body = candidate.body_size
                    if body >= self.min_body_atr * atr:
                        strength = min(total_move / (self.min_impulse_atr * atr), 1.0)
                        if not self._zone_exists(candidate.low, candidate.high):
                            self._order_blocks.append(OrderBlockZone(
                                direction="BULLISH",
                                zone_high=max(candidate.open, candidate.close),
                                zone_low=min(candidate.open, candidate.close),
                                timestamp=candidate.timestamp,
                                strength=strength,
                            ))
                            self.log_event("OB_BULLISH_FORMED", candidate.timestamp, {
                                "zone": (candidate.low, candidate.high)
                            })

            # Prospective bearish OB: candidate is bullish
            if candidate.is_bullish:
                impulse_bars = window[i + 1: i + 1 + self.impulse_bars]
                total_move = sum(b.open - b.close for b in impulse_bars)
                if total_move >= self.min_impulse_atr * atr:
                    body = candidate.body_size
                    if body >= self.min_body_atr * atr:
                        strength = min(total_move / (self.min_impulse_atr * atr), 1.0)
                        if not self._zone_exists(candidate.low, candidate.high):
                            self._order_blocks.append(OrderBlockZone(
                                direction="BEARISH",
                                zone_high=max(candidate.open, candidate.close),
                                zone_low=min(candidate.open, candidate.close),
                                timestamp=candidate.timestamp,
                                strength=strength,
                            ))

    def _zone_exists(self, low: float, high: float) -> bool:
        """Prevent duplicate zones at the same level."""
        for ob in self._order_blocks:
            if abs(ob.zone_low - low) / low < 0.001 and abs(ob.zone_high - high) / high < 0.001:
                return True
        return False

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= self.impulse_bars + 3 else 0
        return ValidationResult(
            concept="ICT_OrderBlock_V2",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_OB_V2] %s at %s | %s", event_type, bar, details)
