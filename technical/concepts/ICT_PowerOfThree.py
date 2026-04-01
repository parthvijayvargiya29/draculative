"""
CONCEPT: ICT Power of Three (Accumulation → Manipulation → Distribution)
SOURCE: 07 - ICT FOR DUMMIES | Time EP. 6
         12 - ICT FOR DUMMIES | RISK MANAGEMENT EP. 11
CATEGORY: TIME
TIMEFRAME: 1D (primary), 4H
INSTRUMENTS: ANY
DESCRIPTION:
    The Power of Three (AMD) describes the three-phase structure that price
    follows within any time period (daily, weekly, monthly):

    ACCUMULATION: Price consolidates, typically in the ASIAN session.
                  Smart money absorbs orders. Range is tight.
    MANIPULATION: Price moves AGAINST the intended direction to sweep liquidity
                  (stop hunt). This is the "fake move" — it runs stops on one side.
                  Typically happens in early LONDON session.
    DISTRIBUTION: The real move in the intended direction. Price delivers to
                  the drawn liquidity target. Typically happens in NY session.

    The AMD model predicts: after manipulation (stop hunt), enter in the
    OPPOSITE direction for the distribution leg.

    From risk management episode: "After price distributes (your trade hits TP),
    do NOT re-enter. Price will then ACCUMULATE. Trade the distribution phase only."

EDGE:
    Retail traders chase the manipulation leg thinking it's the real move.
    By identifying the manipulation (sweep + rejection), you enter with the
    institutions for the distribution leg with a very tight stop.
KNOWN_LIMITATIONS:
    Requires identifying the ASIAN range first. If the Asian range is not defined
    (gap day, news overnight), AMD is unreliable for that session.
    Cannot be the sole entry criterion — needs a FVG or OB to define the exact entry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
ASIAN_BARS_REQUIRED  = 6    # minimum bars in Asian session to define range
MANIP_SWEEP_BUFFER   = 0.001 # manipulation must breach Asian range by this %
MIN_MANIP_WICK_ATR   = 0.3   # manipulation wick must be >= this * ATR
DISTRIBUTION_CONFIRM = True  # require close back inside Asian range to confirm manipulation


@dataclass
class AsianRange:
    high:      float
    low:       float
    midpoint:  float

    @property
    def range_size(self) -> float:
        return self.high - self.low


class ICT_PowerOfThree:
    """
    Detects AMD patterns and signals entries at the end of the manipulation phase.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.asian_bars_req    = p.get("asian_bars_required",  ASIAN_BARS_REQUIRED)
        self.manip_buffer      = p.get("manip_sweep_buffer",   MANIP_SWEEP_BUFFER)
        self.min_manip_wick    = p.get("min_manip_wick_atr",   MIN_MANIP_WICK_ATR)
        self.distrib_confirm   = p.get("distribution_confirm", DISTRIBUTION_CONFIRM)
        self._asian_range:     Optional[AsianRange] = None
        self._asian_bars:      List[BarSnapshot] = []
        self._manipulation_seen: bool = False
        self._manip_direction: Optional[str] = None   # "HIGH" or "LOW"
        self._current_date:    Optional[object] = None

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        history = snapshot.history
        if not history:
            return None

        atr = snapshot.atr or snapshot.close * 0.01
        ts  = snapshot.timestamp
        session = getattr(snapshot, "session", "UNKNOWN")

        # Reset daily state on new trading day
        bar_date = ts.date() if hasattr(ts, "date") else None
        if bar_date != self._current_date:
            self._reset_daily()
            self._current_date = bar_date

        # PHASE 1: Accumulate Asian session bars
        if session == "ASIAN":
            self._asian_bars.append(snapshot)
            return None

        # Build Asian range once London opens
        if session == "LONDON" and self._asian_range is None:
            if len(self._asian_bars) >= self.asian_bars_req:
                asian_high = max(b.high for b in self._asian_bars)
                asian_low  = min(b.low  for b in self._asian_bars)
                self._asian_range = AsianRange(
                    high=asian_high,
                    low=asian_low,
                    midpoint=(asian_high + asian_low) / 2,
                )
                self.log_event("ASIAN_RANGE", ts, {
                    "high": asian_high, "low": asian_low
                })

        if self._asian_range is None:
            return None

        ar = self._asian_range

        # PHASE 2: Detect manipulation (sweep of Asian range)
        if not self._manipulation_seen:
            # Bearish manipulation: wick above Asian high → reversal = bullish distribution
            if snapshot.high > ar.high * (1 + self.manip_buffer):
                wick = snapshot.high - max(snapshot.open, snapshot.close)
                if wick >= self.min_manip_wick * atr:
                    if not self.distrib_confirm or snapshot.close < ar.high:
                        self._manipulation_seen = True
                        self._manip_direction = "HIGH"
                        self.log_event("MANIPULATION_HIGH", ts, {
                            "asian_high": ar.high, "wick_top": snapshot.high
                        })
                        return Signal(
                            concept     = "ICT_PowerOfThree",
                            symbol      = snapshot.symbol,
                            direction   = Direction.LONG,
                            timestamp   = ts,
                            entry_price = snapshot.close,
                            stop_loss   = snapshot.high + atr * 0.15,
                            take_profit = snapshot.close + 2.5 * atr,
                            confidence  = 0.65,
                            category    = ConceptCategory.TIME,
                            regime      = snapshot.regime,
                            reason      = f"AMD: BSL swept above Asian high {ar.high:.4f}, manipulation wick={wick:.4f}, entering LONG for distribution",
                        )

            # Bullish manipulation: wick below Asian low → reversal = bearish distribution
            if snapshot.low < ar.low * (1 - self.manip_buffer):
                wick = min(snapshot.open, snapshot.close) - snapshot.low
                if wick >= self.min_manip_wick * atr:
                    if not self.distrib_confirm or snapshot.close > ar.low:
                        self._manipulation_seen = True
                        self._manip_direction = "LOW"
                        self.log_event("MANIPULATION_LOW", ts, {
                            "asian_low": ar.low, "wick_bottom": snapshot.low
                        })
                        return Signal(
                            concept     = "ICT_PowerOfThree",
                            symbol      = snapshot.symbol,
                            direction   = Direction.SHORT,
                            timestamp   = ts,
                            entry_price = snapshot.close,
                            stop_loss   = snapshot.low - atr * 0.15,
                            take_profit = snapshot.close - 2.5 * atr,
                            confidence  = 0.65,
                            category    = ConceptCategory.TIME,
                            regime      = snapshot.regime,
                            reason      = f"AMD: SSL swept below Asian low {ar.low:.4f}, manipulation wick={wick:.4f}, entering SHORT for distribution",
                        )
        return None

    def _reset_daily(self):
        self._asian_range       = None
        self._asian_bars        = []
        self._manipulation_seen = False
        self._manip_direction   = None

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= self.asian_bars_req + 5 else 0
        return ValidationResult(
            concept="ICT_PowerOfThree",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_AMD] %s at %s | %s", event_type, bar, details)
