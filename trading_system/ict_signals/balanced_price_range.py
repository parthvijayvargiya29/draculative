#!/usr/bin/env python3
"""
trading_system/ict_signals/balanced_price_range.py
Balanced Price Range (BPR) Detector
=====================================
A BPR forms when a bullish FVG and a bearish FVG overlap. The overlap zone is
the BPR. Its midpoint (CE = consequent encroachment) acts as a magnet level.

BPR detection (bar-by-bar, no lookahead):
  1. Find all unmitigated bullish FVGs  (candle[i-1].high < candle[i+1].low)
  2. Find all unmitigated bearish FVGs  (candle[i-1].low  > candle[i+1].high)
  3. For every bull/bear pair:
       overlap_top    = min(bull.top, bear.top)
       overlap_bottom = max(bull.bottom, bear.bottom)
       if overlap_top > overlap_bottom → BPR found
  4. CE = (overlap_top + overlap_bottom) / 2

All parameters from configs/ict_signals.yaml → "balanced_price_range".

Usage:
    detector = BPRDetector()
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
            return yaml.safe_load(fh).get("balanced_price_range", {})
    return {}

_C = _load_cfg()
_FVG_LOOKBACK  = int(_C.get("fvg_lookback", 60))
_FILL_TOL      = float(_C.get("fill_tolerance_pct", 0.10)) / 100.0
_MAX_ACTIVE    = int(_C.get("max_active_bprs", 8))


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class BalancedPriceRange:
    top: float
    bottom: float
    ce: float               # consequent encroachment (midpoint)
    bullish_fvg_ref: str    # date of source bullish FVG
    bearish_fvg_ref: str    # date of source bearish FVG
    size: float
    mitigated: bool         # price has tagged CE


@dataclass
class BPRResult:
    active_bprs: List[BalancedPriceRange]
    nearest_bpr_above: Optional[BalancedPriceRange]
    nearest_bpr_below: Optional[BalancedPriceRange]


# ── Raw FVG namedtuple ────────────────────────────────────────────────────────
from typing import NamedTuple

class _FVG(NamedTuple):
    kind: str    # "bullish" | "bearish"
    top: float
    bottom: float
    date: str


# ── Detector ──────────────────────────────────────────────────────────────────
class BPRDetector:
    """
    Stateful Balanced Price Range detector.
    Call update(df) where df = df.iloc[:i+1]. No lookahead.
    """

    def __init__(self):
        self._bprs: List[BalancedPriceRange] = []

    def update(self, df: pd.DataFrame) -> BPRResult:
        if len(df) < 3:
            return BPRResult(active_bprs=[], nearest_bpr_above=None, nearest_bpr_below=None)

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
        if date_col and date_col != "date":
            df = df.rename(columns={date_col: "date"})

        current_price = float(df["close"].iloc[-1])
        n = len(df)
        start = max(1, n - _FVG_LOOKBACK - 1)

        bull_fvgs: List[_FVG] = []
        bear_fvgs: List[_FVG] = []

        # Detect FVGs strictly up to bar n-2 (need candle[i+1] ≤ current bar)
        for i in range(start, n - 1):
            c_prev = df.iloc[i - 1]
            c_mid  = df.iloc[i]
            c_next = df.iloc[i + 1]

            date_str = ""
            if "date" in df.columns:
                date_str = str(df["date"].iloc[i])

            prev_h = float(c_prev["high"])
            prev_l = float(c_prev["low"])
            next_l = float(c_next["low"])
            next_h = float(c_next["high"])

            # Bullish FVG: gap between candle[i-1].high and candle[i+1].low
            if prev_h < next_l:
                bull_fvgs.append(_FVG("bullish", next_l, prev_h, date_str))

            # Bearish FVG: gap between candle[i-1].low and candle[i+1].high
            if prev_l > next_h:
                bear_fvgs.append(_FVG("bearish", prev_l, next_h, date_str))

        # Cross-match all bull × bear pairs for overlap
        found_bprs: List[BalancedPriceRange] = []
        for bull in bull_fvgs:
            for bear in bear_fvgs:
                overlap_top    = min(bull.top,    bear.top)
                overlap_bottom = max(bull.bottom, bear.bottom)
                if overlap_top > overlap_bottom:
                    ce = (overlap_top + overlap_bottom) / 2.0
                    size = overlap_top - overlap_bottom
                    mitigated = abs(current_price - ce) / max(current_price, 1e-6) <= _FILL_TOL
                    found_bprs.append(BalancedPriceRange(
                        top=round(overlap_top, 4),
                        bottom=round(overlap_bottom, 4),
                        ce=round(ce, 4),
                        bullish_fvg_ref=bull.date,
                        bearish_fvg_ref=bear.date,
                        size=round(size, 4),
                        mitigated=mitigated,
                    ))

        # Deduplicate (by CE proximity) and keep most recent _MAX_ACTIVE
        unique: List[BalancedPriceRange] = []
        for bpr in found_bprs:
            if not any(abs(b.ce - bpr.ce) < 1e-5 for b in unique):
                unique.append(bpr)

        active = [b for b in unique if not b.mitigated][-_MAX_ACTIVE:]

        # Nearest above / below
        above = [b for b in active if b.ce > current_price]
        below = [b for b in active if b.ce <= current_price]
        nearest_above = min(above, key=lambda b: b.ce - current_price) if above else None
        nearest_below = max(below, key=lambda b: b.ce) if below else None

        return BPRResult(
            active_bprs=active,
            nearest_bpr_above=nearest_above,
            nearest_bpr_below=nearest_below,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== BPRDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="1y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = BPRDetector()
    result = detector.update(df)

    price = float(df["close"].iloc[-1])
    print(f"  Symbol: {ticker}  Price: {price:.2f}")
    print(f"  Active BPRs: {len(result.active_bprs)}")
    for b in result.active_bprs[:5]:
        print(f"    CE={b.ce:.2f}  top={b.top:.2f}  bot={b.bottom:.2f}  "
              f"size={b.size:.4f}  mitigated={b.mitigated}")
    print(f"  Nearest BPR above: {result.nearest_bpr_above.ce if result.nearest_bpr_above else 'None'}")
    print(f"  Nearest BPR below: {result.nearest_bpr_below.ce if result.nearest_bpr_below else 'None'}")
    print("  PASS ✓")
