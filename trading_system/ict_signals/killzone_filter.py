#!/usr/bin/env python3
"""
trading_system/ict_signals/killzone_filter.py
ICT Kill Zone Time Filter
=========================
The most foundational ICT concept: high-probability setups are only taken
inside named kill zones. All other signals should be gated through this filter.

Kill zones (US Eastern, configurable in configs/ict_signals.yaml):
  Asia        20:00 – 00:00  (SSL/BSL raids, accumulation phase)
  London      02:00 – 05:00  (highest institutional flow)
  NY Open     07:00 – 10:00  (power of three manipulation + distribution)
  NY Lunch    12:00 – 13:00  (low volume — avoid)
  NY PM       13:30 – 16:00  (continuation or reversal)

For daily bars (no intraday time), all bars are labelled "daily_bar" with
zone_strength=config.daily_bar_strength and in_high_prob_window=True.

Usage:
    detector = KillZoneDetector()
    for i, row in df.iterrows():
        result = detector.process(row['datetime'], row['close'])
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── Config loader ─────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
_CFG_PATH  = _REPO_ROOT / "configs" / "ict_signals.yaml"

def _load_cfg() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as fh:
            return yaml.safe_load(fh).get("kill_zone", {})
    return {}

_CFG = _load_cfg()

_ZONES_CFG = _CFG.get("zones", {
    "asia":     {"start": "20:00", "end": "00:00", "strength": 0.6},
    "london":   {"start": "02:00", "end": "05:00", "strength": 1.0},
    "ny_open":  {"start": "07:00", "end": "10:00", "strength": 0.9},
    "ny_lunch": {"start": "12:00", "end": "13:00", "strength": 0.0},
    "ny_pm":    {"start": "13:30", "end": "16:00", "strength": 0.7},
})
_DAILY_BAR_STRENGTH   = float(_CFG.get("daily_bar_strength", 0.5))
_HIGH_PROB_ZONES      = {"london", "ny_open"}

def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))

_ZONE_RANGES = {
    name: (
        _parse_time(cfg["start"]),
        _parse_time(cfg["end"]),
        float(cfg["strength"]),
    )
    for name, cfg in _ZONES_CFG.items()
}

# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class KillZoneResult:
    active_zone: str           # "asia"|"london"|"ny_open"|"ny_lunch"|"ny_pm"|"none"|"daily_bar"
    zone_strength: float       # 0–1
    minutes_into_zone: int
    bias_direction: str        # "bullish"|"bearish"|"neutral"
    in_high_prob_window: bool  # True if london or ny_open

# ── Detector ──────────────────────────────────────────────────────────────────
class KillZoneDetector:
    """
    Stateful ICT kill zone detector. Call process(dt, close) bar-by-bar.

    Parameters
    ----------
    htf_bias : str
        External higher-timeframe bias: "bullish" | "bearish" | "neutral".
        Can be updated at any time by setting detector.htf_bias.
    """

    def __init__(self, htf_bias: str = "neutral"):
        self.htf_bias = htf_bias

    def process(
        self,
        bar_datetime: Optional[datetime],
        close: float,               # not used internally; kept for interface parity
    ) -> KillZoneResult:
        """
        Determine which kill zone (if any) bar_datetime falls inside.

        For daily bars (bar_datetime is date-only), returns "daily_bar".
        """
        if bar_datetime is None:
            return KillZoneResult(
                active_zone="none", zone_strength=0.0,
                minutes_into_zone=0, bias_direction=self.htf_bias,
                in_high_prob_window=False,
            )

        # Coerce to datetime
        if isinstance(bar_datetime, (pd.Timestamp,)):
            bar_datetime = bar_datetime.to_pydatetime()

        # If no time component (daily bar), treat as daily_bar zone
        if (
            not hasattr(bar_datetime, "hour")
            or (bar_datetime.hour == 0 and bar_datetime.minute == 0
                and bar_datetime.second == 0)
        ):
            return KillZoneResult(
                active_zone="daily_bar",
                zone_strength=_DAILY_BAR_STRENGTH,
                minutes_into_zone=0,
                bias_direction=self.htf_bias,
                in_high_prob_window=True,
            )

        # Localise to Eastern — try pytz, fall back to naive offset
        try:
            import pytz
            tz = pytz.timezone(_CFG.get("timezone", "US/Eastern"))
            if bar_datetime.tzinfo is None:
                # Assume UTC, convert to Eastern
                bar_datetime = pytz.utc.localize(bar_datetime).astimezone(tz)
            else:
                bar_datetime = bar_datetime.astimezone(tz)
        except ImportError:
            # pytz missing — treat as-is (naive Eastern)
            pass

        bar_time = bar_datetime.time()

        for zone_name, (start_t, end_t, strength) in _ZONE_RANGES.items():
            minutes_in = _minutes_into_zone(bar_time, start_t, end_t)
            if minutes_in is not None:
                return KillZoneResult(
                    active_zone=zone_name,
                    zone_strength=strength,
                    minutes_into_zone=minutes_in,
                    bias_direction=self.htf_bias,
                    in_high_prob_window=(zone_name in _HIGH_PROB_ZONES),
                )

        return KillZoneResult(
            active_zone="none", zone_strength=0.0,
            minutes_into_zone=0, bias_direction=self.htf_bias,
            in_high_prob_window=False,
        )

    def process_series(self, df: pd.DataFrame) -> list[KillZoneResult]:
        """
        Batch process. df must have a 'datetime' column.
        Returns a list of KillZoneResult (same order as df rows).
        """
        dt_col = next(
            (c for c in df.columns if "date" in c.lower() or "time" in c.lower()),
            None
        )
        results = []
        for _, row in df.iterrows():
            dt = pd.to_datetime(row[dt_col]) if dt_col else None
            results.append(self.process(dt, float(row.get("close", 0))))
        return results


def _minutes_into_zone(t: time, start: time, end: time) -> Optional[int]:
    """
    Returns minutes since zone start if t is inside [start, end),
    handling midnight wraparound. Returns None if outside zone.
    """
    # Convert all times to minutes since midnight
    def _mins(x: time) -> int:
        return x.hour * 60 + x.minute

    t_m     = _mins(t)
    start_m = _mins(start)
    end_m   = _mins(end)

    if start_m < end_m:
        # Normal (no wraparound)
        if start_m <= t_m < end_m:
            return t_m - start_m
    else:
        # Wraparound (e.g. Asia: 20:00 – 00:00)
        if t_m >= start_m or t_m < end_m:
            if t_m >= start_m:
                return t_m - start_m
            else:
                return (1440 - start_m) + t_m
    return None


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    print("=== KillZoneDetector smoke test ===")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = yf.Ticker(ticker).history(period="5d", interval="1h").reset_index()
    df.columns = [c.lower() for c in df.columns]
    dt_col = next(c for c in df.columns if "date" in c or "time" in c)

    detector = KillZoneDetector(htf_bias="bullish")
    results = detector.process_series(df)

    from collections import Counter
    zone_counts = Counter(r.active_zone for r in results)
    print(f"  Symbol: {ticker}  Bars: {len(df)}")
    for zone, cnt in zone_counts.most_common():
        print(f"  {zone:<15} {cnt:>4} bars")
    print("  PASS ✓")
