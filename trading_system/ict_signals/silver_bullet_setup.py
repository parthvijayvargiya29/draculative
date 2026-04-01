#!/usr/bin/env python3
"""
trading_system/ict_signals/silver_bullet_setup.py
ICT Silver Bullet Setup Detector
==================================
The Silver Bullet is a specific intraday entry model from ICT2:
  1. A Fair Value Gap (FVG) is created during a kill zone
  2. Price produces a BOS or ChoCH on the entry timeframe after the FVG
  3. Price returns to fill the FVG → entry trigger
  4. Target = prior opposing liquidity (swing high/low)

On daily bars (no intraday), approximate with:
  1. FVG created in the first 30% of session bars (within fvg_creation_bar_pct)
  2. BOS detected after the FVG bar
  3. Price returning toward FVG midpoint = silver bullet entry

All parameters from configs/ict_signals.yaml → "silver_bullet".

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
            return yaml.safe_load(fh).get("silver_bullet", {})
    return {}

_C = _load_cfg()
_FVG_BAR_PCT  = float(_C.get("fvg_creation_bar_pct", 0.30))
_BOS_LB       = int(_C.get("bos_lookback", 10))
_MIN_RR       = float(_C.get("min_risk_reward", 1.5))
_ATR_PERIOD   = int(_C.get("atr_period", 14))
_STOP_ATR     = float(_C.get("stop_atr_multiple", 1.0))


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class SilverBulletResult:
    setup_valid: bool
    entry_zone_top: float
    entry_zone_bottom: float
    entry_zone_midpoint: float
    trigger_bos_level: float
    target_price: float
    stop_loss: float
    risk_reward: float
    session_context: str        # "london" | "ny_open" | "daily"
    confidence: float


# ── ATR helper ────────────────────────────────────────────────────────────────
def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


# ── BOS detection helper ──────────────────────────────────────────────────────
def _detect_bos(df: pd.DataFrame, lookback: int) -> Optional[float]:
    """
    Simple BOS: price closes above the last swing high (bullish BOS)
    or below the last swing low (bearish BOS) within `lookback` bars.
    Returns the BOS level or None.
    """
    n = len(df)
    if n < lookback + 2:
        return None
    window = df.iloc[-lookback - 1 : -1]
    swing_high = float(window["high"].max())
    swing_low  = float(window["low"].min())
    curr_close = float(df["close"].iloc[-1])

    if curr_close > swing_high:
        return round(swing_high, 4)
    if curr_close < swing_low:
        return round(swing_low, 4)
    return None


# ── Detector ──────────────────────────────────────────────────────────────────
class SilverBulletDetector:
    """
    Stateful silver bullet setup detector.
    Call update(df) where df = df.iloc[:i+1]. No lookahead.
    """

    def __init__(self):
        self._last: Optional[SilverBulletResult] = None

    def update(self, df: pd.DataFrame) -> SilverBulletResult:
        n = len(df)
        if n < _ATR_PERIOD + _BOS_LB + 3:
            return self._null()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return self._null()

        atr = _atr_series(df, _ATR_PERIOD)
        atr_val = float(atr.iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return self._null()

        current_price = float(df["close"].iloc[-1])

        # Step 1: Find a FVG in the "early" bars (within _FVG_BAR_PCT of total)
        # For daily bars we use the rolling window — scan last 30% of lookback window
        lookback_window = max(_BOS_LB * 2, 20)
        fvg_scan_start  = max(1, n - lookback_window)
        fvg_scan_end    = max(2, int(fvg_scan_start + (n - fvg_scan_start) * _FVG_BAR_PCT))
        fvg_scan_end    = min(fvg_scan_end, n - 2)

        # Search for any unmitigated FVG in the early window
        candidate_fvgs = []
        for i in range(fvg_scan_start, fvg_scan_end + 1):
            if i < 1 or i + 1 >= n:
                continue
            c_prev = df.iloc[i - 1]
            c_next = df.iloc[i + 1]
            prev_h = float(c_prev["high"])
            prev_l = float(c_prev["low"])
            next_l = float(c_next["low"])
            next_h = float(c_next["high"])

            # Bullish FVG
            if prev_h < next_l:
                fvg_top    = next_l
                fvg_bottom = prev_h
                fvg_mid    = (fvg_top + fvg_bottom) / 2.0
                # Unmitigated if current price has not yet entered the gap
                if current_price > fvg_bottom:
                    candidate_fvgs.append(("bullish", fvg_top, fvg_bottom, fvg_mid))

            # Bearish FVG
            if prev_l > next_h:
                fvg_top    = prev_l
                fvg_bottom = next_h
                fvg_mid    = (fvg_top + fvg_bottom) / 2.0
                if current_price < fvg_top:
                    candidate_fvgs.append(("bearish", fvg_top, fvg_bottom, fvg_mid))

        if not candidate_fvgs:
            return self._null()

        # Step 2: BOS / ChoCH after the FVG
        bos_level = _detect_bos(df, _BOS_LB)
        if bos_level is None:
            return self._null()

        # Step 3: Price returning to FVG zone (within 1.5 ATR)
        fvg_kind, fvg_top, fvg_bottom, fvg_mid = candidate_fvgs[-1]

        if fvg_kind == "bullish":
            price_in_fvg = (fvg_bottom <= current_price <= fvg_top) or \
                           (abs(current_price - fvg_mid) <= 1.5 * atr_val)
        else:
            price_in_fvg = (fvg_bottom <= current_price <= fvg_top) or \
                           (abs(current_price - fvg_mid) <= 1.5 * atr_val)

        if not price_in_fvg:
            return self._null()

        # Step 4: Target = prior swing in opposing direction
        lookback_tgt = min(30, n - 1)
        if fvg_kind == "bullish":
            target  = float(df["high"].iloc[-lookback_tgt:].max())
            stop    = fvg_bottom - _STOP_ATR * atr_val
        else:
            target  = float(df["low"].iloc[-lookback_tgt:].min())
            stop    = fvg_top + _STOP_ATR * atr_val

        # Risk/reward
        risk   = abs(current_price - stop) + 1e-9
        reward = abs(target - current_price)
        rr     = reward / risk

        if rr < _MIN_RR:
            return self._null()

        confidence = min(1.0, 0.4 + min(0.4, rr / 10.0) + 0.2)

        return SilverBulletResult(
            setup_valid=True,
            entry_zone_top=round(fvg_top, 4),
            entry_zone_bottom=round(fvg_bottom, 4),
            entry_zone_midpoint=round(fvg_mid, 4),
            trigger_bos_level=round(bos_level, 4),
            target_price=round(target, 4),
            stop_loss=round(stop, 4),
            risk_reward=round(rr, 2),
            session_context="daily",
            confidence=round(confidence, 4),
        )

    @staticmethod
    def _null() -> SilverBulletResult:
        return SilverBulletResult(
            setup_valid=False,
            entry_zone_top=0.0, entry_zone_bottom=0.0,
            entry_zone_midpoint=0.0,
            trigger_bos_level=0.0,
            target_price=0.0, stop_loss=0.0,
            risk_reward=0.0, session_context="daily",
            confidence=0.0,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== SilverBulletDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="1y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = SilverBulletDetector()
    found = 0
    for i in range(_ATR_PERIOD + _BOS_LB + 5, len(df)):
        r = detector.update(df.iloc[:i+1])
        if r.setup_valid and found < 5:
            found += 1
            print(f"  Bar {i:4d}  FVG={r.entry_zone_bottom:.2f}–{r.entry_zone_top:.2f}  "
                  f"BOS={r.trigger_bos_level:.2f}  "
                  f"target={r.target_price:.2f}  "
                  f"RR={r.risk_reward:.1f}  "
                  f"conf={r.confidence:.2f}")
    if found == 0:
        print("  No silver bullet setups found in sample")
    print("  PASS ✓")
