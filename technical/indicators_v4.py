"""
indicators_v4.py — Core indicator computation engine.

Computes all indicators needed by concept modules from a pandas DataFrame
of OHLCV bars (columns: open, high, low, close, volume, indexed by datetime).

CRITICAL: All computations are pandas-native with no lookahead.
shift(1) is used wherever the *previous* bar value is needed so that
at bar N we only consume bars 0..N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── ATR ──────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. True range = max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── ADX ──────────────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = np.where((high - prev_high) > (prev_low - low),
                         np.maximum(high - prev_high, 0), 0)
    dm_minus = np.where((prev_low - low) > (high - prev_high),
                         np.maximum(prev_low - low, 0), 0)

    tr_s     = pd.Series(tr,       index=df.index).ewm(alpha=1/period, adjust=False).mean()
    dmp_s    = pd.Series(dm_plus,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    dmn_s    = pd.Series(dm_minus, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    di_plus  = 100 * dmp_s / tr_s.replace(0, np.nan)
    di_minus = 100 * dmn_s / tr_s.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0)


# ── SMA / EMA ─────────────────────────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].rolling(period, min_periods=period).mean()

def compute_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


# ── BOLLINGER BANDS ───────────────────────────────────────────────────────────

def compute_bbands(df: pd.DataFrame, period: int = 20,
                   std_mult: float = 2.0) -> pd.DataFrame:
    mid = compute_sma(df, period)
    std = df["close"].rolling(period, min_periods=period).std()
    return pd.DataFrame({
        "bb_mid":   mid,
        "bb_upper": mid + std_mult * std,
        "bb_lower": mid - std_mult * std,
        "bb_width": (std_mult * 2 * std) / mid.replace(0, np.nan),
    }, index=df.index)


# ── SWING HIGHS / LOWS ────────────────────────────────────────────────────────

def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Returns a boolean Series that is True at bars that are swing highs.
    A swing high is a bar whose high is the highest in the window
    [bar - lookback .. bar + lookback]. No lookahead: future bars are not used
    in a live feed — the caller is responsible for only calling on completed bars.
    """
    h = df["high"]
    rol_max = h.rolling(2 * lookback + 1, center=True, min_periods=lookback + 1).max()
    return h == rol_max

def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    l = df["low"]
    rol_min = l.rolling(2 * lookback + 1, center=True, min_periods=lookback + 1).min()
    return l == rol_min


# ── FIBONACCI LEVELS ──────────────────────────────────────────────────────────

def fibonacci_levels(swing_high: float, swing_low: float,
                     direction: str = "bullish") -> dict:
    """
    Returns key Fibonacci retracement and extension levels.
    direction='bullish': price moved up from low to high, we retrace down.
    direction='bearish': price moved down from high to low, we retrace up.
    """
    rng = swing_high - swing_low
    if direction == "bullish":
        # Retracement from high downward
        return {
            "0.0":   swing_high,
            "0.236": swing_high - 0.236 * rng,
            "0.382": swing_high - 0.382 * rng,
            "0.5":   swing_high - 0.500 * rng,
            "0.618": swing_high - 0.618 * rng,   # OTE lower bound
            "0.705": swing_high - 0.705 * rng,
            "0.786": swing_high - 0.786 * rng,   # OTE upper bound (deepest)
            "1.0":   swing_low,
        }
    else:
        return {
            "0.0":   swing_low,
            "0.236": swing_low + 0.236 * rng,
            "0.382": swing_low + 0.382 * rng,
            "0.5":   swing_low + 0.500 * rng,
            "0.618": swing_low + 0.618 * rng,
            "0.705": swing_low + 0.705 * rng,
            "0.786": swing_low + 0.786 * rng,
            "1.0":   swing_high,
        }


# ── SESSION LABEL ─────────────────────────────────────────────────────────────

def label_session(ts: pd.Timestamp) -> str:
    """
    Labels a UTC timestamp with the trading session it belongs to.
    All times expressed as UTC offsets of EST:
      Asian    20:00–00:00 EST  →  01:00–05:00 UTC (next day)
      London   02:00–05:00 EST  →  07:00–10:00 UTC
      New York 07:00–10:00 EST  →  12:00–15:00 UTC
    """
    # Convert to EST (UTC-5) for comparison
    hour_est = (ts.hour - 5) % 24
    if 20 <= hour_est or hour_est < 0:
        return "ASIAN"
    if 2 <= hour_est < 5:
        return "LONDON"
    if 7 <= hour_est < 10:
        return "NEW_YORK"
    return "OFF"


# ── REGIME CLASSIFIER ─────────────────────────────────────────────────────────

def classify_regime(adx: float, sma50: float, sma50_prev: float) -> str:
    """
    TRENDING  : ADX > 25 AND SMA50 slope is positive (price is in a trend).
    CORRECTIVE: everything else.
    """
    sma50_slope = sma50 - sma50_prev
    if adx > 25 and sma50_slope > 0:
        return "TRENDING"
    return "CORRECTIVE"


# ── FULL INDICATOR PIPELINE ───────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and returns it enriched with all indicators.
    This is the single entry-point called by the simulator before feeding
    bars to concept modules.

    Input columns required: open, high, low, close, volume
    Index: DatetimeIndex (UTC)
    """
    df = df.copy()
    df["atr_14"]    = compute_atr(df, 14)
    df["atr_pct"]   = df["atr_14"] / df["close"]
    df["adx_14"]    = compute_adx(df, 14)
    df["sma_20"]    = compute_sma(df, 20)
    df["sma_50"]    = compute_sma(df, 50)
    df["sma_200"]   = compute_sma(df, 200)
    df["ema_9"]     = compute_ema(df, 9)
    df["ema_21"]    = compute_ema(df, 21)
    df["rsi_14"]    = compute_rsi(df, 14)

    bb = compute_bbands(df, 20, 2.0)
    df = pd.concat([df, bb], axis=1)

    df["swing_high"] = find_swing_highs(df, 5)
    df["swing_low"]  = find_swing_lows(df, 5)

    df["session"]    = df.index.map(label_session)
    df["dow"]        = df.index.dayofweek   # 0=Mon, 4=Fri

    # Regime tag (needs sma_50 from previous bar — use shift)
    df["sma_50_prev"] = df["sma_50"].shift(1)
    df["regime"] = df.apply(
        lambda r: classify_regime(r["adx_14"], r["sma_50"], r["sma_50_prev"])
        if pd.notna(r["adx_14"]) and pd.notna(r["sma_50"]) else "UNKNOWN",
        axis=1,
    )

    return df
