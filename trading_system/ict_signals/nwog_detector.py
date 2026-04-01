#!/usr/bin/env python3
"""
trading_system/ict_signals/nwog_detector.py
New Week Opening Gap (NWOG) & New Month Opening Gap (NMOG) Detector
====================================================================
ICT teaches that gaps between Friday close → Monday open (NWOG) and
last-day-of-month close → first-day-of-month open (NMOG) are institutional
reference points. Price is drawn to fill the Consequent Encroachment (CE = gap
midpoint) before resuming.

Works on daily bars only. Maintains rolling state: up to max_active_nwogs
unfilled NWOGs and max_active_nmogs unfilled NMOGs.

All parameters from configs/ict_signals.yaml under "nwog".

Usage:
    detector = NWOGDetector()
    for i in range(len(df)):
        result = detector.update(df.iloc[:i+1])
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
_CFG_PATH  = _REPO_ROOT / "configs" / "ict_signals.yaml"

def _load_cfg() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as fh:
            return yaml.safe_load(fh).get("nwog", {})
    return {}

_C = _load_cfg()
_MAX_NWOGS   = int(_C.get("max_active_nwogs", 4))
_MAX_NMOGS   = int(_C.get("max_active_nmogs", 2))
_FILL_TOL    = float(_C.get("fill_tolerance_pct", 0.10)) / 100.0
_BIAS_PROX   = float(_C.get("bias_weight_proximity_pct", 2.0)) / 100.0


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class GapLevel:
    gap_type: str       # "NWOG" | "NMOG"
    ce_price: float     # midpoint = consequent encroachment level
    top: float
    bottom: float
    direction: str      # "up" | "down"
    age_bars: int
    filled: bool


@dataclass
class NWOGResult:
    active_nwogs: List[GapLevel]
    active_nmogs: List[GapLevel]
    nearest_ce_above: Optional[float]
    nearest_ce_below: Optional[float]
    bias_from_gaps: str               # "bullish" | "bearish" | "neutral"


# ── Detector ──────────────────────────────────────────────────────────────────
class NWOGDetector:
    """
    Stateful daily-bar NWOG/NMOG tracker.
    Call update(df) where df = df.iloc[:i+1].
    """

    def __init__(self):
        self._nwogs: List[GapLevel] = []
        self._nmogs: List[GapLevel] = []

    def update(self, df: pd.DataFrame) -> NWOGResult:
        """
        Process bars up to df.iloc[-1]. No lookahead.
        """
        if len(df) < 2:
            return self._empty_result(0.0)

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        # Normalise date column
        date_col = next(
            (c for c in df.columns if "date" in c or "time" in c), None
        )
        if date_col and date_col != "date":
            df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])

        # Process every new bar pair for gap creation
        # We rebuild gaps from scratch on each call to keep state clean
        # (practical for small lookbacks; use incremental logic for live systems)
        self._nwogs = []
        self._nmogs = []

        for i in range(1, len(df)):
            prev_bar = df.iloc[i - 1]
            curr_bar = df.iloc[i]
            prev_close = float(prev_bar["close"])
            curr_open  = float(curr_bar["open"])
            curr_high  = float(curr_bar["high"])
            curr_low   = float(curr_bar["low"])
            curr_date  = curr_bar["date"]
            age = len(df) - 1 - i

            # Check if this is a NWOG (Monday bar)
            is_monday = (hasattr(curr_date, "weekday") and curr_date.weekday() == 0)
            if is_monday and curr_open != prev_close:
                gap = self._make_gap("NWOG", prev_close, curr_open, age)
                if gap:
                    self._nwogs.append(gap)

            # Check if this is a NMOG (first trading day of month)
            prev_date = prev_bar["date"]
            is_new_month = (
                hasattr(curr_date, "month")
                and hasattr(prev_date, "month")
                and curr_date.month != prev_date.month
            )
            if is_new_month and curr_open != prev_close:
                gap = self._make_gap("NMOG", prev_close, curr_open, age)
                if gap:
                    self._nmogs.append(gap)

        current_price = float(df.iloc[-1]["close"])

        # Mark filled gaps (CE tagged within tolerance)
        for gap_list in (self._nwogs, self._nmogs):
            for gap in gap_list:
                if not gap.filled:
                    gap.filled = abs(current_price - gap.ce_price) / gap.ce_price <= _FILL_TOL

        # Keep only unfilled, most recent
        unfilled_nwogs = [g for g in self._nwogs if not g.filled][-_MAX_NWOGS:]
        unfilled_nmogs = [g for g in self._nmogs if not g.filled][-_MAX_NMOGS:]

        # Nearest CEs above / below
        ces_above = sorted(
            [g.ce_price for g in unfilled_nwogs + unfilled_nmogs
             if g.ce_price > current_price]
        )
        ces_below = sorted(
            [g.ce_price for g in unfilled_nwogs + unfilled_nmogs
             if g.ce_price <= current_price],
            reverse=True,
        )

        nearest_above = ces_above[0] if ces_above else None
        nearest_below = ces_below[0] if ces_below else None

        bias = self._gap_bias(current_price, unfilled_nwogs + unfilled_nmogs)

        return NWOGResult(
            active_nwogs=unfilled_nwogs,
            active_nmogs=unfilled_nmogs,
            nearest_ce_above=nearest_above,
            nearest_ce_below=nearest_below,
            bias_from_gaps=bias,
        )

    @staticmethod
    def _make_gap(
        gap_type: str,
        prev_close: float,
        curr_open: float,
        age: int,
    ) -> Optional[GapLevel]:
        if curr_open > prev_close:
            top    = curr_open
            bottom = prev_close
            direction = "up"
        else:
            top    = prev_close
            bottom = curr_open
            direction = "down"

        gap_size = top - bottom
        if gap_size <= 0:
            return None

        ce = (top + bottom) / 2.0
        return GapLevel(
            gap_type=gap_type,
            ce_price=round(ce, 4),
            top=round(top, 4),
            bottom=round(bottom, 4),
            direction=direction,
            age_bars=age,
            filled=False,
        )

    @staticmethod
    def _gap_bias(price: float, gaps: List[GapLevel]) -> str:
        """
        Directional bias derived from unfilled gaps.
        If more unfilled CEs are below price than above → bias=bearish (drawn down).
        If more above → bias=bullish (drawn up).
        Proximity within _BIAS_PROX % of price adds extra weight.
        """
        score = 0.0
        for g in gaps:
            prox = abs(price - g.ce_price) / max(price, 1e-6)
            weight = 2.0 if prox <= _BIAS_PROX else 1.0
            if g.ce_price > price:
                score += weight   # magnet above → bullish pull
            else:
                score -= weight   # magnet below → bearish pull

        if score > 0.5:
            return "bullish"
        elif score < -0.5:
            return "bearish"
        return "neutral"

    @staticmethod
    def _empty_result(price: float) -> NWOGResult:
        return NWOGResult(
            active_nwogs=[], active_nmogs=[],
            nearest_ce_above=None, nearest_ce_below=None,
            bias_from_gaps="neutral",
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== NWOGDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="1y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = NWOGDetector()
    result = detector.update(df)

    price = float(df["close"].iloc[-1])
    print(f"  Symbol: {ticker}  Price: {price:.2f}")
    print(f"  Active NWOGs: {len(result.active_nwogs)}")
    for g in result.active_nwogs:
        print(f"    CE={g.ce_price:.2f}  top={g.top:.2f}  bot={g.bottom:.2f}  "
              f"dir={g.direction}  age={g.age_bars}d")
    print(f"  Active NMOGs: {len(result.active_nmogs)}")
    for g in result.active_nmogs:
        print(f"    CE={g.ce_price:.2f}  top={g.top:.2f}  bot={g.bottom:.2f}  "
              f"dir={g.direction}  age={g.age_bars}d")
    print(f"  Nearest CE above: {result.nearest_ce_above}")
    print(f"  Nearest CE below: {result.nearest_ce_below}")
    print(f"  Gap bias: {result.bias_from_gaps}")
    print("  PASS ✓")
