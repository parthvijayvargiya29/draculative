"""
CONCEPT: ICT Liquidity Pool
SOURCE: 05 - ICT FOR DUMMIES | LIQUIDITY PT. 1 EP. 4
         08 - ICT FOR DUMMIES | Liquidity PT. 2 EP. 7
CATEGORY: LIQUIDITY
TIMEFRAME: 1D, 4H, 1H, 15M, 5M
INSTRUMENTS: ANY
DESCRIPTION:
    A Liquidity Pool is a cluster of resting orders (stop losses and pending orders)
    that accumulates above swing highs (Buy Side Liquidity / BSL) or below swing lows
    (Sell Side Liquidity / SSL). These pools represent the "draw on liquidity" —
    the target that institutional smart money is aiming for. Price is DRAWN to
    liquidity. Equal highs and equal lows are especially strong liquidity pools because
    retail traders place stop losses just beyond these obvious levels.

    The concept of trendline liquidity is also included: trendlines drawn by retail
    traders create stop loss clusters at the trendline extension level.

EDGE:
    Institutions need to fill large orders. They cannot fill at the current price
    without moving the market against themselves. Instead, they engineer price to
    sweep a liquidity pool (taking retail stops), fill their institutional order
    at a better price, then reverse. Knowing where the liquidity sits = knowing
    where price is going next.
KNOWN_LIMITATIONS:
    Liquidity pools that have already been swept are no longer valid. Always track
    whether a level has been taken. In strong trending markets, price may sweep
    multiple pools in succession without reversing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
LOOKBACK_BARS    = 50      # bars to scan for liquidity pool formation
EQUAL_THRESHOLD  = 0.001   # two highs/lows are "equal" if within this % of each other
SWING_LOOKBACK   = 5       # bars each side for swing identification
MIN_POOL_TOUCHES = 2       # minimum number of touches to qualify as a pool


@dataclass
class LiquidityLevel:
    price:    float
    side:     str      # "BSL" (above) or "SSL" (below)
    touches:  int
    swept:    bool = False


class ICT_LiquidityPool:
    """
    Identifies Buy Side Liquidity (BSL) and Sell Side Liquidity (SSL) pools.
    Signals a trade entry when price sweeps a pool and rejects.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.lookback_bars    = p.get("lookback_bars",    LOOKBACK_BARS)
        self.equal_threshold  = p.get("equal_threshold",  EQUAL_THRESHOLD)
        self.swing_lookback   = p.get("swing_lookback",   SWING_LOOKBACK)
        self.min_pool_touches = p.get("min_pool_touches", MIN_POOL_TOUCHES)
        self._pools: List[LiquidityLevel] = []

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        if len(history) < self.lookback_bars:
            return None

        # Rebuild pools from history
        self._build_pools(history[-self.lookback_bars:])

        atr = snapshot.atr or snapshot.close * 0.01

        for pool in self._pools:
            if pool.swept:
                continue

            # BSL sweep: price wicks above the pool then closes back below
            if pool.side == "BSL":
                swept = snapshot.high > pool.price and snapshot.close < pool.price
                if swept:
                    pool.swept = True
                    self.log_event("BSL_SWEPT", snapshot.timestamp, {
                        "pool_price": pool.price,
                        "high": snapshot.high,
                        "close": snapshot.close,
                    })
                    return Signal(
                        concept     = "ICT_LiquidityPool",
                        symbol      = snapshot.symbol,
                        direction   = Direction.SHORT,
                        timestamp   = snapshot.timestamp,
                        entry_price = snapshot.close,
                        stop_loss   = snapshot.high + atr * 0.25,
                        take_profit = snapshot.close - 2.0 * atr,
                        confidence  = 0.70,
                        category    = ConceptCategory.LIQUIDITY,
                        regime      = snapshot.regime,
                        reason      = f"BSL swept at {pool.price:.4f}: wick to {snapshot.high:.4f}, close back to {snapshot.close:.4f} — SHORT entry on rejection",
                    )

            # SSL sweep: price wicks below the pool then closes back above
            if pool.side == "SSL":
                swept = snapshot.low < pool.price and snapshot.close > pool.price
                if swept:
                    pool.swept = True
                    self.log_event("SSL_SWEPT", snapshot.timestamp, {
                        "pool_price": pool.price,
                        "low": snapshot.low,
                        "close": snapshot.close,
                    })
                    return Signal(
                        concept     = "ICT_LiquidityPool",
                        symbol      = snapshot.symbol,
                        direction   = Direction.LONG,
                        timestamp   = snapshot.timestamp,
                        entry_price = snapshot.close,
                        stop_loss   = snapshot.low - atr * 0.25,
                        take_profit = snapshot.close + 2.0 * atr,
                        confidence  = 0.70,
                        category    = ConceptCategory.LIQUIDITY,
                        regime      = snapshot.regime,
                        reason      = f"SSL swept at {pool.price:.4f}: wick to {snapshot.low:.4f}, close back to {snapshot.close:.4f} — LONG entry on rejection",
                    )
        return None

    def _build_pools(self, bars: List[BarSnapshot]):
        """Identify BSL and SSL pools from recent history."""
        self._pools = []
        lb = self.swing_lookback

        if len(bars) < 2 * lb + 1:
            return

        highs: List[float] = []
        lows:  List[float] = []

        for i in range(lb, len(bars) - lb):
            wh = [b.high for b in bars[i - lb: i + lb + 1]]
            wl = [b.low  for b in bars[i - lb: i + lb + 1]]
            if bars[i].high == max(wh):
                highs.append(bars[i].high)
            if bars[i].low == min(wl):
                lows.append(bars[i].low)

        # Group equal highs → BSL pools
        bsl_pools = self._group_equal_levels(highs)
        for price, touches in bsl_pools:
            if touches >= self.min_pool_touches:
                self._pools.append(LiquidityLevel(price, "BSL", touches))

        # Group equal lows → SSL pools
        ssl_pools = self._group_equal_levels(lows)
        for price, touches in ssl_pools:
            if touches >= self.min_pool_touches:
                self._pools.append(LiquidityLevel(price, "SSL", touches))

    def _group_equal_levels(self, prices: List[float]) -> List[Tuple[float, int]]:
        """Groups prices that are within equal_threshold of each other."""
        if not prices:
            return []
        groups: List[Tuple[float, int]] = []
        for p in sorted(set(prices)):
            merged = False
            for i, (gp, gc) in enumerate(groups):
                if abs(p - gp) / gp <= self.equal_threshold:
                    groups[i] = ((gp * gc + p) / (gc + 1), gc + 1)
                    merged = True
                    break
            if not merged:
                groups.append((p, 1))
        return groups

    def get_active_pools(self) -> List[LiquidityLevel]:
        return [p for p in self._pools if not p.swept]

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= self.lookback_bars else 0
        return ValidationResult(
            concept="ICT_LiquidityPool",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_LiquidityPool] %s at %s | %s", event_type, bar, details)
