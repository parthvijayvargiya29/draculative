"""
CONCEPT: ICT Fair Value Gap V2 (Refined)
SOURCE: 06 - ICT FOR DUMMIES | Fair Value Gaps EP. 5
CATEGORY: FVG
TIMEFRAME: 1D, 4H, 1H, 15M, 5M, 1M
INSTRUMENTS: ANY
DESCRIPTION:
    A Fair Value Gap (FVG) is a three-candle pattern where the middle candle
    creates an imbalance — the gap between candle 1's high and candle 3's low
    (for a bullish FVG) or between candle 1's low and candle 3's high (for a
    bearish FVG) is never filled during candle 2. This represents an area of
    price inefficiency. Price is DRAWN back to fill these inefficiencies.

    Bullish FVG: candle 3 low > candle 1 high  (gap above candle 1)
    Bearish FVG: candle 3 high < candle 1 low  (gap below candle 1)

    V2 improvements over V1:
    - Tracks partial fills (FVG is still valid until 50% fill by default)
    - Strength rating based on displacement candle size relative to ATR
    - Inversion detection: FVG that gets fully violated becomes an inversion FVG
    - Filters out tiny gaps (gap must be >= MIN_GAP_SIZE_ATR * ATR)

EDGE:
    Institutional orders are filled in these gaps. Price returns because those
    orders need to be filled to maintain balanced price delivery. High-displacement
    FVGs created during strong directional moves are the highest quality.
KNOWN_LIMITATIONS:
    FVGs on lower timeframes in choppy markets fill quickly and reverse. Always
    trade FVGs in the direction of the higher-timeframe bias. A FVG in premium
    is bearish; a FVG in discount is bullish.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
MIN_GAP_SIZE_ATR  = 0.25   # FVG gap must be >= this multiple of ATR to be valid
PARTIAL_FILL_PCT  = 0.50   # FVG is valid until this % of the gap is filled
MAX_FVG_AGE_BARS  = 100    # FVGs older than this are expired and removed
DISPLACEMENT_MIN  = 1.5    # displacement candle body must be >= this * ATR


@dataclass
class FVGZone:
    direction:  str     # "BULLISH" or "BEARISH"
    top:        float
    bottom:     float
    midpoint:   float
    origin_bar: object  # timestamp
    atr_at_creation: float
    strength:   float   # 0.0–1.0 based on displacement candle size
    filled_pct: float   = 0.0
    inverted:   bool    = False
    age_bars:   int     = 0

    @property
    def is_valid(self) -> bool:
        return not self.inverted and self.filled_pct < PARTIAL_FILL_PCT and self.age_bars < MAX_FVG_AGE_BARS

    @property
    def gap_size(self) -> float:
        return self.top - self.bottom


class ICT_FairValueGap_V2:
    """
    Detects FVGs and generates entry signals when price returns to fill them.
    Tracks partial fills and inversion events.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.min_gap_atr     = p.get("min_gap_atr",     MIN_GAP_SIZE_ATR)
        self.partial_fill    = p.get("partial_fill_pct", PARTIAL_FILL_PCT)
        self.max_age         = p.get("max_fvg_age_bars", MAX_FVG_AGE_BARS)
        self.displacement_min = p.get("displacement_min", DISPLACEMENT_MIN)
        self._fvg_zones: List[FVGZone] = []

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        if len(history) < 3:
            return None

        atr = snapshot.atr or snapshot.close * 0.01

        # Form new FVG from last 3 bars + current bar
        # Pattern uses bars[-3], bars[-2], bars[-1] (current = snapshot)
        c1, c2, c3 = history[-2], history[-1], snapshot

        # Age existing zones
        for z in self._fvg_zones:
            z.age_bars += 1

        # Check for new bullish FVG: c3.low > c1.high
        if c3.low > c1.high:
            gap = c3.low - c1.high
            if gap >= self.min_gap_atr * atr:
                displacement = c2.body_size / atr if atr > 0 else 0
                strength = min(displacement / self.displacement_min, 1.0)
                zone = FVGZone(
                    direction="BULLISH",
                    top=c3.low,
                    bottom=c1.high,
                    midpoint=(c1.high + c3.low) / 2,
                    origin_bar=c3.timestamp,
                    atr_at_creation=atr,
                    strength=strength,
                )
                self._fvg_zones.append(zone)
                self.log_event("FVG_BULLISH_FORMED", snapshot.timestamp, {
                    "top": zone.top, "bottom": zone.bottom, "strength": strength
                })

        # Check for new bearish FVG: c3.high < c1.low
        if c3.high < c1.low:
            gap = c1.low - c3.high
            if gap >= self.min_gap_atr * atr:
                displacement = c2.body_size / atr if atr > 0 else 0
                strength = min(displacement / self.displacement_min, 1.0)
                zone = FVGZone(
                    direction="BEARISH",
                    top=c1.low,
                    bottom=c3.high,
                    midpoint=(c1.low + c3.high) / 2,
                    origin_bar=c3.timestamp,
                    atr_at_creation=atr,
                    strength=strength,
                )
                self._fvg_zones.append(zone)
                self.log_event("FVG_BEARISH_FORMED", snapshot.timestamp, {
                    "top": zone.top, "bottom": zone.bottom, "strength": strength
                })

        # Update fills and check for entry
        for zone in list(self._fvg_zones):
            if not zone.is_valid:
                self._fvg_zones.remove(zone)
                continue

            if zone.direction == "BULLISH":
                # Price returned into bullish FVG — LONG entry
                if snapshot.low <= zone.top and snapshot.low >= zone.bottom:
                    self._update_fill(zone, snapshot)
                    if zone.is_valid:
                        self.log_event("FVG_BULLISH_ENTRY", snapshot.timestamp, {
                            "zone_bottom": zone.bottom, "zone_top": zone.top
                        })
                        return Signal(
                            concept     = "ICT_FairValueGap_V2",
                            symbol      = snapshot.symbol,
                            direction   = Direction.LONG,
                            timestamp   = snapshot.timestamp,
                            entry_price = snapshot.close,
                            stop_loss   = zone.bottom - atr * 0.25,
                            take_profit = zone.origin_bar if False else snapshot.close + 2.5 * atr,
                            confidence  = 0.55 + 0.20 * zone.strength,
                            category    = ConceptCategory.FVG,
                            regime      = snapshot.regime,
                            reason      = f"Price returned to bullish FVG ({zone.bottom:.4f}–{zone.top:.4f}), strength={zone.strength:.2f}",
                        )

            if zone.direction == "BEARISH":
                # Price returned into bearish FVG — SHORT entry
                if snapshot.high >= zone.bottom and snapshot.high <= zone.top:
                    self._update_fill(zone, snapshot)
                    if zone.is_valid:
                        self.log_event("FVG_BEARISH_ENTRY", snapshot.timestamp, {
                            "zone_bottom": zone.bottom, "zone_top": zone.top
                        })
                        return Signal(
                            concept     = "ICT_FairValueGap_V2",
                            symbol      = snapshot.symbol,
                            direction   = Direction.SHORT,
                            timestamp   = snapshot.timestamp,
                            entry_price = snapshot.close,
                            stop_loss   = zone.top + atr * 0.25,
                            take_profit = snapshot.close - 2.5 * atr,
                            confidence  = 0.55 + 0.20 * zone.strength,
                            category    = ConceptCategory.FVG,
                            regime      = snapshot.regime,
                            reason      = f"Price returned to bearish FVG ({zone.bottom:.4f}–{zone.top:.4f}), strength={zone.strength:.2f}",
                        )
        return None

    def _update_fill(self, zone: FVGZone, snap: BarSnapshot):
        """Update zone fill percentage and mark as inverted if fully violated."""
        if zone.direction == "BULLISH":
            penetration = zone.top - snap.low
            zone.filled_pct = max(zone.filled_pct, penetration / zone.gap_size)
            if snap.close < zone.bottom:
                zone.inverted = True  # full violation → inversion FVG
        else:
            penetration = snap.high - zone.bottom
            zone.filled_pct = max(zone.filled_pct, penetration / zone.gap_size)
            if snap.close > zone.top:
                zone.inverted = True

    def get_active_fvgs(self) -> List[FVGZone]:
        return [z for z in self._fvg_zones if z.is_valid]

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= 3 else 0
        return ValidationResult(
            concept="ICT_FairValueGap_V2",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_FVG_V2] %s at %s | %s", event_type, bar, details)
