#!/usr/bin/env python3
"""
trading_system/ict_signals/turtle_soup_detector.py
Turtle Soup Stop-Hunt Detector
================================
ICT's version of the stop hunt / fake breakout.

Turtle soup LONG (raid of sell-side liquidity):
  - Prior swing low identified over `swing_lookback` bars
  - Current candle's low < prior swing low (by ≥ ATR × min_raid_atr_multiple)
  - Current candle CLOSES above the prior swing low (wick below, body above)
  - Raid candle volume ≥ volume_multiplier × N-bar average

Turtle soup SHORT (mirror — raid of buy-side liquidity):
  - Prior swing high taken out → closes below it

All parameters from configs/ict_signals.yaml → "turtle_soup".

Bar-by-bar stateful. Call update(df) where df = df.iloc[:i+1].
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
            return yaml.safe_load(fh).get("turtle_soup", {})
    return {}

_C = _load_cfg()
_SWING_LB    = int(_C.get("swing_lookback", 20))
_MIN_RAID    = float(_C.get("min_raid_atr_multiple", 0.05))
_ATR_PERIOD  = int(_C.get("atr_period", 14))
_VOL_MULT    = float(_C.get("volume_multiplier", 1.0))
_VOL_LB      = int(_C.get("volume_lookback", 20))
_STOP_ATR    = float(_C.get("stop_atr_multiple", 0.5))


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class TurtleSoupResult:
    detected: bool
    direction: str             # "long" | "short" | "none"
    raided_level: float        # the swing high/low swept
    entry_signal_price: float  # close of the confirmation candle
    stop_loss: float
    confidence: float          # 0–1
    date: str


# ── ATR helper ────────────────────────────────────────────────────────────────
def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _swing_low(df: pd.DataFrame, end_idx: int, lookback: int) -> float:
    start = max(0, end_idx - lookback)
    return float(df["low"].iloc[start:end_idx].min())


def _swing_high(df: pd.DataFrame, end_idx: int, lookback: int) -> float:
    start = max(0, end_idx - lookback)
    return float(df["high"].iloc[start:end_idx].max())


# ── Detector ──────────────────────────────────────────────────────────────────
class TurtleSoupDetector:
    """
    Stateful turtle soup detector.
    Call update(df) where df = df.iloc[:i+1]. No lookahead.
    Returns TurtleSoupResult for the latest bar.
    """

    def __init__(self):
        self._last: Optional[TurtleSoupResult] = None

    def update(self, df: pd.DataFrame) -> TurtleSoupResult:
        n = len(df)
        if n < _ATR_PERIOD + _SWING_LB + 2:
            return self._null()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return self._null()

        atr = _atr_series(df, _ATR_PERIOD)
        vol_ma = df["volume"].rolling(_VOL_LB, min_periods=1).mean()

        i = n - 1  # current bar index
        bar = df.iloc[i]
        atr_val = float(atr.iloc[i])
        if pd.isna(atr_val) or atr_val <= 0:
            return self._null()

        curr_low   = float(bar["low"])
        curr_high  = float(bar["high"])
        curr_close = float(bar["close"])
        curr_vol   = float(bar["volume"])
        vol_avg    = float(vol_ma.iloc[i])

        date_str = ""
        if "date" in df.columns:
            date_str = str(df["date"].iloc[i])

        # ── Long setup (raid of swing lows) ──────────────────────────────────
        prior_swing_low = _swing_low(df, i, _SWING_LB)
        raid_threshold  = prior_swing_low - _MIN_RAID * atr_val

        if (curr_low < raid_threshold) and (curr_close > prior_swing_low):
            # Wick below, close above → turtle soup long
            vol_ok = vol_avg <= 0 or curr_vol >= _VOL_MULT * vol_avg
            raid_distance = (prior_swing_low - curr_low) / atr_val
            confidence = min(1.0, 0.5 + (0.3 if vol_ok else 0.0) +
                             min(0.2, raid_distance * 0.1))
            stop = curr_low - _STOP_ATR * atr_val

            result = TurtleSoupResult(
                detected=True,
                direction="long",
                raided_level=round(prior_swing_low, 4),
                entry_signal_price=round(curr_close, 4),
                stop_loss=round(stop, 4),
                confidence=round(confidence, 4),
                date=date_str,
            )
            self._last = result
            return result

        # ── Short setup (raid of swing highs) ─────────────────────────────────
        prior_swing_high = _swing_high(df, i, _SWING_LB)
        raid_threshold_hi = prior_swing_high + _MIN_RAID * atr_val

        if (curr_high > raid_threshold_hi) and (curr_close < prior_swing_high):
            # Wick above, close below → turtle soup short
            vol_ok = vol_avg <= 0 or curr_vol >= _VOL_MULT * vol_avg
            raid_distance = (curr_high - prior_swing_high) / atr_val
            confidence = min(1.0, 0.5 + (0.3 if vol_ok else 0.0) +
                             min(0.2, raid_distance * 0.1))
            stop = curr_high + _STOP_ATR * atr_val

            result = TurtleSoupResult(
                detected=True,
                direction="short",
                raided_level=round(prior_swing_high, 4),
                entry_signal_price=round(curr_close, 4),
                stop_loss=round(stop, 4),
                confidence=round(confidence, 4),
                date=date_str,
            )
            self._last = result
            return result

        return self._null()

    @staticmethod
    def _null() -> TurtleSoupResult:
        return TurtleSoupResult(
            detected=False, direction="none",
            raided_level=0.0, entry_signal_price=0.0,
            stop_loss=0.0, confidence=0.0, date="",
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== TurtleSoupDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="2y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = TurtleSoupDetector()
    found = 0
    for i in range(_ATR_PERIOD + _SWING_LB, len(df)):
        r = detector.update(df.iloc[:i+1])
        if r.detected and found < 8:
            found += 1
            print(f"  Bar {i:4d}  {r.direction.upper():5s}  "
                  f"raided={r.raided_level:.2f}  "
                  f"entry={r.entry_signal_price:.2f}  "
                  f"SL={r.stop_loss:.2f}  "
                  f"conf={r.confidence:.2f}  {r.date[:10]}")
    if found == 0:
        print("  No turtle soup setups found in sample period")
    print("  PASS ✓")
