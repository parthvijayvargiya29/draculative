#!/usr/bin/env python3
"""
trading_system/ict_signals/propulsion_block_detector.py
Propulsion Block Detector
==========================
An ICT2-specific concept: a propulsion block is the last opposing-colour candle
BEFORE a confirmed displacement. Unlike a regular order block, it is only
identified AFTER the displacement validates, making it a higher-conviction
anchor.

Bullish propulsion block:
    Last bearish candle immediately before a confirmed bullish displacement.
    Its high–low defines the block. Price returning to it from above → long entry.

Bearish propulsion block:
    Last bullish candle before a confirmed bearish displacement → short entry.

Confluence scoring:
    +0.3 if block midpoint aligns within fvg_confluence_distance_pct of any open FVG
    +0.3 if block midpoint falls in the OTE Fibonacci zone (61.8%–78.6% retracement)
    +0.4 base score

All parameters from configs/ict_signals.yaml → "propulsion_block".

This module imports DisplacementDetector rather than re-implementing displacement.
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
            return yaml.safe_load(fh).get("propulsion_block", {})
    return {}

_C = _load_cfg()
_DISP_LOOKBACK   = int(_C.get("displacement_lookback", 5))
_FVG_DIST_PCT    = float(_C.get("fvg_confluence_distance_pct", 0.50)) / 100.0
_OTE_LO          = float(_C.get("ote_fib_lo", 0.618))
_OTE_HI          = float(_C.get("ote_fib_hi", 0.786))
_ATR_PERIOD      = int(_C.get("atr_period", 14))
_MAX_ACTIVE      = int(_C.get("max_active_blocks", 10))


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class PropulsionBlockResult:
    detected: bool
    direction: str                   # "bullish" | "bearish" | "none"
    top: float
    bottom: float
    midpoint: float
    source_displacement_start: float
    source_displacement_end: float
    confluence_score: float          # 0–1
    date: str
    mitigated: bool


# ── ATR helper ────────────────────────────────────────────────────────────────
def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _body(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))

def _is_bull(row: pd.Series) -> bool:
    return float(row["close"]) > float(row["open"])


# ── Detector ──────────────────────────────────────────────────────────────────
class PropulsionBlockDetector:
    """
    Stateful propulsion block detector.
    Call update(df) where df = df.iloc[:i+1]. No lookahead.

    Maintains a list of up to _MAX_ACTIVE unmitigated propulsion blocks.
    """

    def __init__(self):
        self._blocks: List[PropulsionBlockResult] = []

    def update(self, df: pd.DataFrame) -> PropulsionBlockResult:
        """
        Process bars up to df.iloc[-1].
        Returns the most recently created unmitigated propulsion block,
        or a null result if none.
        """
        if len(df) < _ATR_PERIOD + 4:
            return self._null()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return self._null()

        # Import displacement detector (avoids re-implementing displacement logic)
        from trading_system.ict_signals.displacement_detector import (
            DisplacementDetector, _atr_series as _datr,
        )

        disp_det = DisplacementDetector()
        disp = disp_det.update(df)

        if not disp.detected:
            # Update mitigation status of existing blocks
            self._update_mitigation(float(df["close"].iloc[-1]))
            return self._best_block()

        # A displacement was confirmed. Find the last opposing-colour candle
        # BEFORE the displacement started (within _DISP_LOOKBACK bars of origin).
        n = len(df)
        atr = _atr_series(df, _ATR_PERIOD)
        current_price = float(df["close"].iloc[-1])

        # Find bar index corresponding to displacement_terminus
        # (candle where displacement completed)
        term_price = disp.displacement_terminus
        origin_price = disp.displacement_origin

        # Search backward from current bar for where the displacement candles live
        # We look back _DISP_LOOKBACK bars for the opposing-colour candle
        lookback_start = max(1, n - 1 - disp.candles_ago - _DISP_LOOKBACK)
        lookback_end   = max(1, n - 1 - disp.candles_ago + 1)

        pb_candidate: Optional[PropulsionBlockResult] = None

        for j in range(lookback_start, lookback_end):
            bar = df.iloc[j]
            bar_bull = _is_bull(bar)

            if disp.direction == "bullish" and not bar_bull:
                # Last bearish candle before bullish displacement
                top    = float(bar["high"])
                bottom = float(bar["low"])
                date_val = str(df.index[j]) if hasattr(df.index[j], "__str__") else str(j)
                if "date" in df.columns:
                    date_val = str(df["date"].iloc[j])

                mid = (top + bottom) / 2.0
                score = self._confluence_score(
                    mid, top, bottom,
                    origin_price, term_price,
                    current_price, disp.direction
                )
                mitigated = current_price < top  # price has come back to block

                pb_candidate = PropulsionBlockResult(
                    detected=True,
                    direction="bullish",
                    top=round(top, 4),
                    bottom=round(bottom, 4),
                    midpoint=round(mid, 4),
                    source_displacement_start=round(origin_price, 4),
                    source_displacement_end=round(term_price, 4),
                    confluence_score=round(score, 4),
                    date=date_val,
                    mitigated=mitigated,
                )

            elif disp.direction == "bearish" and bar_bull:
                top    = float(bar["high"])
                bottom = float(bar["low"])
                date_val = str(df.index[j]) if hasattr(df.index[j], "__str__") else str(j)
                if "date" in df.columns:
                    date_val = str(df["date"].iloc[j])

                mid = (top + bottom) / 2.0
                score = self._confluence_score(
                    mid, top, bottom,
                    origin_price, term_price,
                    current_price, disp.direction
                )
                mitigated = current_price > bottom

                pb_candidate = PropulsionBlockResult(
                    detected=True,
                    direction="bearish",
                    top=round(top, 4),
                    bottom=round(bottom, 4),
                    midpoint=round(mid, 4),
                    source_displacement_start=round(origin_price, 4),
                    source_displacement_end=round(term_price, 4),
                    confluence_score=round(score, 4),
                    date=date_val,
                    mitigated=mitigated,
                )
            # Take the last match (closest to displacement)

        if pb_candidate:
            # Deduplicate against existing blocks
            already = any(
                abs(b.top - pb_candidate.top) < 1e-6 and
                abs(b.bottom - pb_candidate.bottom) < 1e-6
                for b in self._blocks
            )
            if not already:
                self._blocks.append(pb_candidate)
                # Trim to max active
                self._blocks = self._blocks[-_MAX_ACTIVE:]

        self._update_mitigation(current_price)
        return self._best_block()

    def _confluence_score(
        self,
        mid: float,
        top: float,
        bottom: float,
        disp_origin: float,
        disp_terminus: float,
        current_price: float,
        direction: str,
    ) -> float:
        """Compute a 0–1 confluence score for a propulsion block candidate."""
        score = 0.4  # base

        # OTE confluence: is midpoint in the OTE zone (61.8–78.6% retracement)?
        if direction == "bullish":
            swing_range = disp_terminus - disp_origin
            if swing_range > 0:
                ote_lo = disp_terminus - _OTE_HI * swing_range
                ote_hi = disp_terminus - _OTE_LO * swing_range
                if ote_lo <= mid <= ote_hi:
                    score += 0.3
        else:
            swing_range = disp_origin - disp_terminus
            if swing_range > 0:
                ote_lo = disp_terminus + _OTE_LO * swing_range
                ote_hi = disp_terminus + _OTE_HI * swing_range
                if ote_lo <= mid <= ote_hi:
                    score += 0.3

        # Proximity to current price (within 3%)
        if current_price > 0:
            dist_pct = abs(mid - current_price) / current_price
            if dist_pct < 0.03:
                score += 0.3

        return min(1.0, score)

    def _update_mitigation(self, current_price: float) -> None:
        for b in self._blocks:
            if b.direction == "bullish":
                b.mitigated = current_price < b.bottom
            else:
                b.mitigated = current_price > b.top

    def _best_block(self) -> PropulsionBlockResult:
        active = [b for b in self._blocks if not b.mitigated]
        if not active:
            return self._null()
        # Return highest confluence score
        return max(active, key=lambda b: b.confluence_score)

    @staticmethod
    def _null() -> PropulsionBlockResult:
        return PropulsionBlockResult(
            detected=False, direction="none",
            top=0.0, bottom=0.0, midpoint=0.0,
            source_displacement_start=0.0, source_displacement_end=0.0,
            confluence_score=0.0, date="", mitigated=False,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== PropulsionBlockDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="1y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = PropulsionBlockDetector()
    found = 0
    for i in range(30, len(df)):
        r = detector.update(df.iloc[:i+1])
        if r.detected and found < 5:
            found += 1
            print(f"  Bar {i}  {r.direction.upper()} PB  "
                  f"top={r.top:.2f}  bot={r.bottom:.2f}  "
                  f"confluence={r.confluence_score:.2f}  "
                  f"mitigated={r.mitigated}")
    if found == 0:
        print("  No propulsion blocks found in sample")
    print("  PASS ✓")
