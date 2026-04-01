#!/usr/bin/env python3
"""
trading_system/ict_signals/displacement_detector.py
ICT Displacement Detector
==========================
A displacement is a strong, directional 3-candle move that:
  1. Middle candle body > body_atr_multiple × ATR(14)
  2. All 3 candles agree in direction (opposing body ≤ opposing_body_max_pct × middle)
  3. Volume on middle candle ≥ volume_multiplier × N-bar average
  4. Creates a measurable FVG imbalance between candle[0].high → candle[2].low (bull)
     or candle[0].low → candle[2].high (bear)

All thresholds from configs/ict_signals.yaml under the "displacement" key.

Bar-by-bar, stateful. Call detector.update(bar_series) where bar_series is
df.iloc[:i+1] (NO lookahead).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
_CFG_PATH  = _REPO_ROOT / "configs" / "ict_signals.yaml"

def _load_cfg() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as fh:
            return yaml.safe_load(fh).get("displacement", {})
    return {}

_C = _load_cfg()
_ATR_PERIOD         = int(_C.get("atr_period", 14))
_BODY_ATR_MULT      = float(_C.get("body_atr_multiple", 2.0))
_OPP_BODY_MAX_PCT   = float(_C.get("opposing_body_max_pct", 0.30))
_VOL_MULT           = float(_C.get("volume_multiplier", 1.3))
_VOL_LOOKBACK       = int(_C.get("volume_lookback", 20))
_LOOKBACK           = int(_C.get("lookback_bars", 50))

# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class DisplacementResult:
    detected: bool
    direction: str                # "bullish" | "bearish" | "none"
    strength: float               # 0–1 based on ATR multiple
    created_fvg_top: float
    created_fvg_bottom: float
    created_fvg_midpoint: float
    displacement_origin: float    # start price of the 3-candle move
    displacement_terminus: float  # end price
    candles_ago: int              # how many bars since displacement completed

# ── ATR helper ────────────────────────────────────────────────────────────────
def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()

# ── Body size helper ──────────────────────────────────────────────────────────
def _body(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))

def _is_bull(row: pd.Series) -> bool:
    return float(row["close"]) > float(row["open"])

# ── Detector ──────────────────────────────────────────────────────────────────
class DisplacementDetector:
    """
    Stateful bar-by-bar displacement detector.

    Call update(df) where df = df.iloc[:i+1] (only bars up to current).
    Returns the most recent DisplacementResult.
    """

    def __init__(self):
        self._last_result: Optional[DisplacementResult] = None

    def update(self, df: pd.DataFrame) -> DisplacementResult:
        """
        Process the latest bar slice. Returns DetectionResult for bar df.iloc[-1].
        No lookahead — only df.iloc[:i+1] data is used.
        """
        n = len(df)
        if n < _ATR_PERIOD + 3:
            return self._null_result()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return self._null_result()

        atr = _atr_series(df, _ATR_PERIOD)
        vol_ma = df["volume"].rolling(_VOL_LOOKBACK, min_periods=1).mean()

        # Scan last _LOOKBACK bars for 3-candle displacement patterns
        scan_start = max(2, n - _LOOKBACK)

        best: Optional[DisplacementResult] = None

        for i in range(scan_start, n - 1):   # i is middle candle (0-indexed)
            c0 = df.iloc[i - 1]
            c1 = df.iloc[i]       # middle candle
            c2 = df.iloc[i + 1]

            atr_val = atr.iloc[i]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            c1_body = _body(c1)

            # Condition 1: middle candle body > body_atr_multiple × ATR
            if c1_body < _BODY_ATR_MULT * atr_val:
                continue

            # Determine direction from middle candle
            c1_bull = _is_bull(c1)

            # Condition 2: all three candles in same direction (no strong opposition)
            if c1_bull:
                c0_ok = _body(c0) <= _OPP_BODY_MAX_PCT * c1_body or _is_bull(c0)
                c2_ok = _body(c2) <= _OPP_BODY_MAX_PCT * c1_body or _is_bull(c2)
            else:
                c0_ok = _body(c0) <= _OPP_BODY_MAX_PCT * c1_body or not _is_bull(c0)
                c2_ok = _body(c2) <= _OPP_BODY_MAX_PCT * c1_body or not _is_bull(c2)

            if not (c0_ok and c2_ok):
                continue

            # Condition 3: volume confirmation
            vol1 = float(c1["volume"])
            vol_avg = float(vol_ma.iloc[i])
            if vol_avg > 0 and vol1 < _VOL_MULT * vol_avg:
                continue

            # Condition 4: FVG imbalance exists
            if c1_bull:
                fvg_bottom = float(c0["high"])
                fvg_top    = float(c2["low"])
                origin     = float(c0["low"])
                terminus   = float(c2["high"])
                direction  = "bullish"
            else:
                fvg_top    = float(c0["low"])
                fvg_bottom = float(c2["high"])
                origin     = float(c0["high"])
                terminus   = float(c2["low"])
                direction  = "bearish"

            if fvg_top <= fvg_bottom:
                continue  # No actual gap

            candles_ago = n - 1 - (i + 1)   # bars since c2 completed
            strength = min(1.0, c1_body / (atr_val * _BODY_ATR_MULT))

            result = DisplacementResult(
                detected=True,
                direction=direction,
                strength=round(strength, 4),
                created_fvg_top=round(fvg_top, 4),
                created_fvg_bottom=round(fvg_bottom, 4),
                created_fvg_midpoint=round((fvg_top + fvg_bottom) / 2, 4),
                displacement_origin=round(origin, 4),
                displacement_terminus=round(terminus, 4),
                candles_ago=candles_ago,
            )

            # Prefer most recent displacement
            if best is None or best.candles_ago > candles_ago:
                best = result

        self._last_result = best if best else self._null_result()
        return self._last_result

    @staticmethod
    def _null_result() -> DisplacementResult:
        return DisplacementResult(
            detected=False, direction="none", strength=0.0,
            created_fvg_top=0.0, created_fvg_bottom=0.0,
            created_fvg_midpoint=0.0,
            displacement_origin=0.0, displacement_terminus=0.0,
            candles_ago=999,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== DisplacementDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="1y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = DisplacementDetector()
    found = 0
    for i in range(20, len(df)):
        r = detector.update(df.iloc[:i+1])
        if r.detected and r.candles_ago == 0:
            found += 1
            print(f"  Bar {i}  {r.direction.upper()} displacement  "
                  f"strength={r.strength:.2f}  FVG={r.created_fvg_bottom:.2f}–{r.created_fvg_top:.2f}")
            if found >= 5:
                break

    if found == 0:
        print("  No displacements found in sample (normal for some periods)")
    print("  PASS ✓")
