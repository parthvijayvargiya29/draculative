#!/usr/bin/env python3
"""
trading_system/ict_signals/power_of_three.py
ICT Power of Three (PO3) Detector
=====================================
The three-phase institutional delivery model:
  Phase 1 — Accumulation : overnight range / Asia session buildup
  Phase 2 — Manipulation : Judas swing, false break against expected direction
  Phase 3 — Distribution : true directional move delivered (NY AM session)

On daily bars:
  Accumulation  = prior close → today's open range
  Manipulation  = first move away from open against expected direction (≥ 0.30×ATR)
  Distribution  = reversal past open in the expected direction

Detection requires an external HTF bias (expected_direction).
Set detector.expected_direction = "bullish" | "bearish" externally as weekly
bias is updated.

All parameters from configs/ict_signals.yaml → "power_of_three".

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
            return yaml.safe_load(fh).get("power_of_three", {})
    return {}

_C = _load_cfg()
_MANIP_MIN_ATR  = float(_C.get("manipulation_min_atr", 0.30))
_ATR_PERIOD     = int(_C.get("atr_period", 14))
_DIST_OTE_FIB   = float(_C.get("distribution_ote_fib", 0.50))
_LB_BARS        = int(_C.get("lookback_bars", 5))


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class PO3Result:
    phase: str                   # "accumulation"|"manipulation"|"distribution"|"unknown"
    expected_direction: str      # "bullish"|"bearish"|"neutral"
    manipulation_low: float
    manipulation_high: float
    open_price: float
    distribution_target: float   # estimated OTE of daily range
    confidence: float


# ── ATR helper ────────────────────────────────────────────────────────────────
def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


# ── Detector ──────────────────────────────────────────────────────────────────
class PowerOfThreeDetector:
    """
    Stateful PO3 phase detector. Works on daily bars.

    Set detector.expected_direction = "bullish" | "bearish" | "neutral"
    from an external weekly/monthly bias signal before calling update().
    """

    def __init__(self, expected_direction: str = "neutral"):
        self.expected_direction = expected_direction

    def update(self, df: pd.DataFrame) -> PO3Result:
        """
        Process bars up to df.iloc[-1]. No lookahead.
        Identifies which PO3 phase the current daily bar is in.
        """
        n = len(df)
        if n < _ATR_PERIOD + 2:
            return self._unknown()

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return self._unknown()

        atr = _atr_series(df, _ATR_PERIOD)
        atr_val = float(atr.iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return self._unknown()

        # Current bar
        bar = df.iloc[-1]
        curr_open  = float(bar["open"])
        curr_high  = float(bar["high"])
        curr_low   = float(bar["low"])
        curr_close = float(bar["close"])

        # Accumulation: opening range vs prior close
        prior_close = float(df["close"].iloc[-2])
        accum_range = abs(curr_open - prior_close)

        # Determine phase based on open + expected direction
        direction = self.expected_direction

        if direction == "bullish":
            # Manipulation = Judas swing DOWN first (below open)
            # Distribution = price closes above open (bullish delivery)

            manip_low  = curr_low
            manip_high = curr_open  # manipulation doesn't go up in bullish PO3

            # Was there a manipulation? Low below open by ≥ MANIP_MIN_ATR × ATR
            judas_swing = (curr_open - curr_low) >= _MANIP_MIN_ATR * atr_val

            if curr_close > curr_open:
                # Closed above open → distribution phase (true bullish move)
                phase = "distribution"
                confidence = min(1.0, 0.5 + (0.4 if judas_swing else 0.0) +
                                 min(0.1, (curr_close - curr_open) / (atr_val + 1e-9) * 0.1))
            elif judas_swing:
                # Below open but hasn't reversed yet → manipulation
                phase = "manipulation"
                confidence = min(1.0, 0.5 + min(0.3, (curr_open - curr_low) / atr_val * 0.3))
            else:
                # Small range, no move yet → accumulation
                phase = "accumulation"
                confidence = 0.4

            # Distribution target: OTE of daily range (open to projected high)
            daily_range = max(curr_high - curr_low, atr_val)
            dist_target = curr_open + (1 - _DIST_OTE_FIB) * daily_range

        elif direction == "bearish":
            # Manipulation = Judas swing UP (above open)
            # Distribution = price closes below open (bearish delivery)

            manip_high = curr_high
            manip_low  = curr_open

            judas_swing = (curr_high - curr_open) >= _MANIP_MIN_ATR * atr_val

            if curr_close < curr_open:
                phase = "distribution"
                confidence = min(1.0, 0.5 + (0.4 if judas_swing else 0.0) +
                                 min(0.1, (curr_open - curr_close) / (atr_val + 1e-9) * 0.1))
            elif judas_swing:
                phase = "manipulation"
                confidence = min(1.0, 0.5 + min(0.3, (curr_high - curr_open) / atr_val * 0.3))
            else:
                phase = "accumulation"
                confidence = 0.4

            daily_range = max(curr_high - curr_low, atr_val)
            dist_target = curr_open - (1 - _DIST_OTE_FIB) * daily_range

        else:
            # Neutral / unknown bias
            manip_low  = curr_low
            manip_high = curr_high
            phase = "unknown"
            confidence = 0.2
            dist_target = curr_open

        return PO3Result(
            phase=phase,
            expected_direction=direction,
            manipulation_low=round(manip_low, 4),
            manipulation_high=round(manip_high, 4),
            open_price=round(curr_open, 4),
            distribution_target=round(dist_target, 4),
            confidence=round(confidence, 4),
        )

    def _unknown(self) -> PO3Result:
        return PO3Result(
            phase="unknown",
            expected_direction=self.expected_direction,
            manipulation_low=0.0,
            manipulation_high=0.0,
            open_price=0.0,
            distribution_target=0.0,
            confidence=0.0,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== PowerOfThreeDetector smoke test ===")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="60d", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]

    detector = PowerOfThreeDetector(expected_direction="bullish")
    phase_counts = {"accumulation": 0, "manipulation": 0, "distribution": 0, "unknown": 0}

    for i in range(_ATR_PERIOD + 2, len(df)):
        r = detector.update(df.iloc[:i+1])
        phase_counts[r.phase] = phase_counts.get(r.phase, 0) + 1

    print(f"  Symbol: {ticker}  HTF bias: bullish  Bars: {len(df)}")
    for phase, cnt in phase_counts.items():
        print(f"  {phase:<15} {cnt:>4} bars")

    # Show last result
    r = detector.update(df)
    print(f"\n  Latest bar:")
    print(f"    Phase      : {r.phase}")
    print(f"    Open       : {r.open_price:.2f}")
    print(f"    Manip low  : {r.manipulation_low:.2f}")
    print(f"    Manip high : {r.manipulation_high:.2f}")
    print(f"    Dist target: {r.distribution_target:.2f}")
    print(f"    Confidence : {r.confidence:.2f}")
    print("  PASS ✓")
