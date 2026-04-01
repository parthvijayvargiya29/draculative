"""
nucleus_engine.py — Nucleus Identification Engine (Section 4).

The "Atom-Nucleus Hypothesis" states that price is always orbiting one
dominant structure. The engine scores 8 candidate nucleus types on every
bar and returns the one with the highest score as the current nucleus.

Nucleus candidates:
  1. PREMIUM_DISCOUNT_ZONE  — price inside OTE 61.8–78.6% fib range
  2. FAIR_VALUE_GAP         — open / unfilled FVG within 1 ATR
  3. ORDER_BLOCK            — validated order block confluence
  4. LIQUIDITY_POOL         — equal highs / lows cluster within reach
  5. KILL_ZONE              — inside Asian / London / NY kill zone
  6. SWING_LEVEL            — close proximity to key swing high/low
  7. INSTITUTIONAL_RANGE    — inside prior week's high-low range
  8. EQUILIBRIUM            — price near 50% of macro range (neutral)

Each scorer returns a float 0.0–1.0 confidence. The argmax nucleus
drives bias for the Convergence Predictor (Section 5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from technical.bar_snapshot import BarSnapshot

logger = logging.getLogger(__name__)


class NucleusType(str, Enum):
    PREMIUM_DISCOUNT_ZONE = "PREMIUM_DISCOUNT_ZONE"
    FAIR_VALUE_GAP        = "FAIR_VALUE_GAP"
    ORDER_BLOCK           = "ORDER_BLOCK"
    LIQUIDITY_POOL        = "LIQUIDITY_POOL"
    KILL_ZONE             = "KILL_ZONE"
    SWING_LEVEL           = "SWING_LEVEL"
    INSTITUTIONAL_RANGE   = "INSTITUTIONAL_RANGE"
    EQUILIBRIUM           = "EQUILIBRIUM"


@dataclass
class NucleusScore:
    nucleus_type: NucleusType
    score:        float           # 0.0 – 1.0
    level:        float           # price level of the nucleus
    reason:       str = ""


@dataclass
class NucleusState:
    dominant:     NucleusType
    dominant_score: float
    all_scores:   Dict[str, float] = field(default_factory=dict)
    dominant_level: float = 0.0


class NucleusEngine:
    """
    Stateless scorer — given a BarSnapshot (with df_ref), returns a NucleusState.
    """

    def identify(self, snap: BarSnapshot) -> NucleusState:
        scores: List[NucleusScore] = [
            self._score_premium_discount(snap),
            self._score_fvg(snap),
            self._score_order_block(snap),
            self._score_liquidity_pool(snap),
            self._score_kill_zone(snap),
            self._score_swing_level(snap),
            self._score_institutional_range(snap),
            self._score_equilibrium(snap),
        ]

        best = max(scores, key=lambda s: s.score)
        return NucleusState(
            dominant       = best.nucleus_type,
            dominant_score = best.score,
            dominant_level = best.level,
            all_scores     = {s.nucleus_type.value: s.score for s in scores},
        )

    # ── Individual scorers ────────────────────────────────────────────────

    def _score_premium_discount(self, snap: BarSnapshot) -> NucleusScore:
        """OTE zone 61.8–78.6% of recent swing range."""
        df  = snap.df_ref
        atr = snap.atr or 1.0
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 20:
            hi = df["high"].rolling(20).max().iloc[-1]
            lo = df["low"].rolling(20).min().iloc[-1]
            swing_range = hi - lo
            if swing_range > 0:
                fib_618 = hi - 0.618 * swing_range
                fib_786 = hi - 0.786 * swing_range
                lo_ote  = min(fib_618, fib_786)
                hi_ote  = max(fib_618, fib_786)
                if lo_ote <= snap.close <= hi_ote:
                    # Price inside OTE → strong nucleus
                    score = 0.85
                    level = (lo_ote + hi_ote) / 2
                elif abs(snap.close - lo_ote) <= atr or abs(snap.close - hi_ote) <= atr:
                    score = 0.55
                    level = lo_ote if snap.close < (lo_ote + hi_ote) / 2 else hi_ote

        return NucleusScore(NucleusType.PREMIUM_DISCOUNT_ZONE, score, level, "OTE fib zone")

    def _score_fvg(self, snap: BarSnapshot) -> NucleusScore:
        """Open FVG within 1 ATR of current price."""
        df  = snap.df_ref
        atr = snap.atr or 1.0
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 3:
            for i in range(len(df) - 3, max(0, len(df) - 50), -1):
                prev  = df.iloc[i]
                curr  = df.iloc[i + 1]
                nxt   = df.iloc[i + 2]
                # Bullish FVG: nxt.low > prev.high
                if nxt["low"] > prev["high"]:
                    fvg_lo = prev["high"]
                    fvg_hi = nxt["low"]
                    mid_fvg = (fvg_lo + fvg_hi) / 2
                    if abs(snap.close - mid_fvg) <= atr:
                        score = max(score, 0.80)
                        level = mid_fvg
                # Bearish FVG: nxt.high < prev.low
                elif nxt["high"] < prev["low"]:
                    fvg_lo = nxt["high"]
                    fvg_hi = prev["low"]
                    mid_fvg = (fvg_lo + fvg_hi) / 2
                    if abs(snap.close - mid_fvg) <= atr:
                        score = max(score, 0.80)
                        level = mid_fvg

        return NucleusScore(NucleusType.FAIR_VALUE_GAP, score, level, "FVG proximity")

    def _score_order_block(self, snap: BarSnapshot) -> NucleusScore:
        """Recent order block (last opposing candle before N-bar displacement)."""
        df  = snap.df_ref
        atr = snap.atr or 1.0
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 10:
            for i in range(len(df) - 2, max(0, len(df) - 30), -1):
                candle = df.iloc[i]
                future = df.iloc[i + 1:]
                if len(future) < 3:
                    continue
                displacement = abs(future["close"].iloc[2] - candle["close"])
                if displacement < 1.5 * atr:
                    continue
                # Check if candle is opposing direction
                is_bearish = candle["close"] < candle["open"]
                price_now_above = snap.close > candle["high"]
                price_now_below = snap.close < candle["low"]
                if is_bearish and price_now_above:
                    # Bullish order block (bearish candle, now price above it)
                    if abs(snap.close - candle["high"]) <= 1.5 * atr:
                        score = max(score, 0.75)
                        level = (candle["open"] + candle["close"]) / 2
                elif not is_bearish and price_now_below:
                    if abs(snap.close - candle["low"]) <= 1.5 * atr:
                        score = max(score, 0.75)
                        level = (candle["open"] + candle["close"]) / 2

        return NucleusScore(NucleusType.ORDER_BLOCK, score, level, "Order block proximity")

    def _score_liquidity_pool(self, snap: BarSnapshot) -> NucleusScore:
        """Equal highs/lows cluster within 1.5 ATR."""
        df  = snap.df_ref
        atr = snap.atr or 1.0
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 20:
            recent_highs = df["high"].tail(30).values
            recent_lows  = df["low"].tail(30).values
            tol = atr * 0.10
            # Count equal highs
            for h in recent_highs:
                cluster = np.sum(np.abs(recent_highs - h) <= tol)
                if cluster >= 3 and abs(snap.high - h) <= 1.5 * atr:
                    score = max(score, 0.70 + min(cluster * 0.05, 0.20))
                    level = h
            # Count equal lows
            for l in recent_lows:
                cluster = np.sum(np.abs(recent_lows - l) <= tol)
                if cluster >= 3 and abs(snap.low - l) <= 1.5 * atr:
                    score = max(score, 0.70 + min(cluster * 0.05, 0.20))
                    level = l

        return NucleusScore(NucleusType.LIQUIDITY_POOL, score, level, "Equal H/L cluster")

    def _score_kill_zone(self, snap: BarSnapshot) -> NucleusScore:
        """Inside Asian / London / NY kill zone windows."""
        from technical.concepts.ICT_KillZone import ICT_KillZone
        try:
            kz = ICT_KillZone()
            sig = kz.detect(snap)
            if sig is not None:
                return NucleusScore(NucleusType.KILL_ZONE, 0.65, snap.close, sig.reason)
        except Exception:
            pass
        return NucleusScore(NucleusType.KILL_ZONE, 0.0, snap.close, "Outside kill zone")

    def _score_swing_level(self, snap: BarSnapshot) -> NucleusScore:
        """Close proximity to a recent swing high or low."""
        df  = snap.df_ref
        atr = snap.atr or 1.0
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 5:
            swing_hi = df["high"].rolling(5, center=True).max().dropna()
            swing_lo = df["low"].rolling(5, center=True).min().dropna()
            for h in swing_hi.tail(10):
                if abs(snap.close - h) <= atr:
                    score = max(score, 0.60)
                    level = h
            for l in swing_lo.tail(10):
                if abs(snap.close - l) <= atr:
                    score = max(score, 0.60)
                    level = l

        return NucleusScore(NucleusType.SWING_LEVEL, score, level, "Swing H/L proximity")

    def _score_institutional_range(self, snap: BarSnapshot) -> NucleusScore:
        """Price inside prior-week high-low range."""
        df  = snap.df_ref
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 5:
            week_hi = df["high"].tail(5).max()
            week_lo = df["low"].tail(5).min()
            if week_lo <= snap.close <= week_hi:
                width = week_hi - week_lo
                center = (week_hi + week_lo) / 2
                proximity_to_edge = 1 - 2 * abs(snap.close - center) / width if width > 0 else 0
                score = max(0.30, 0.30 + proximity_to_edge * 0.30)
                level = center

        return NucleusScore(NucleusType.INSTITUTIONAL_RANGE, score, level, "Prior-week range")

    def _score_equilibrium(self, snap: BarSnapshot) -> NucleusScore:
        """Price near 50% equilibrium of macro range."""
        df  = snap.df_ref
        atr = snap.atr or 1.0
        score = 0.0
        level = snap.close

        if df is not None and len(df) >= 50:
            macro_hi = df["high"].tail(50).max()
            macro_lo = df["low"].tail(50).min()
            eq = (macro_hi + macro_lo) / 2
            if abs(snap.close - eq) <= atr:
                score = 0.50
                level = eq

        return NucleusScore(NucleusType.EQUILIBRIUM, score, level, "50% macro equilibrium")
