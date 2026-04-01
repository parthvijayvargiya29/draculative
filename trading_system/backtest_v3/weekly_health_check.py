#!/usr/bin/env python3
"""
weekly_health_check.py  --  V4.0 Weekly Strategy Health Check
==============================================================
Run every Friday after market close.
Computes regime classification, rolling performance metrics, open position
summary, signal decay indicators, and drawdown state per Section 6.

USAGE
-----
  cd backtest_v3
  export ALPACA_API_KEY="..."
  export ALPACA_SECRET_KEY="..."
  export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
  python weekly_health_check.py --alpaca --daily
  python weekly_health_check.py --alpaca --daily --output weekly_2026_W12.txt

REGIME CLASSIFICATION (Section 6)
-----------------------------------
  Trending : SPY ADX(14) avg > 25 AND SPY SMA50 positive slope > 15 days/month
  Corrective: SPY net return negative AND avg ATR percentile above 60th
  Choppy   : Neither trending nor corrective

The regime table feeds the monthly summary and annual review in Section 6.

SIGNAL DECAY INDICATORS (Section 6, annual review)
----------------------------------------------------
  1. Rolling 30-trade WR declining for 3 consecutive quarters
  2. Rolling 30-trade PF < 1.0 for 2 consecutive quarters
  3. Avg holding period increasing > 30% vs first 30 trades
  4. TIME_STOP exit % increasing in recent 30 trades

If 2+ indicators present: PAUSE new entries and run 18-month fresh backtest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Dict

HERE = Path(__file__).parent.resolve()
TRADE_LOG    = HERE / "paper_trades.json"
MONITOR_LOG  = HERE / "paper_monitor.log"
WEEKLY_LOG   = HERE / "weekly_health.log"

# Validated benchmarks (Phase 1 locked, 19-Mar-2026)
VALIDATED = {
    "test_pf":          1.104,
    "test_wr":          52.94,
    "test_wl_ratio":    1.74,
    "test_signal_rate": 3.6,
    "test_n_trades":    34,
    "go_min_trades":    30,
    "go_pf_floor":      1.10,
    "go_wr_floor":      38.0,
    "go_wr_ceiling":    68.0,
    "go_max_drawdown":  25.0,
    "time_stop_pct_watch": 30.0,
    "instruments": ["GLD", "WMT", "USMV", "NVDA", "AMZN",
                    "GOOGL", "COST", "XOM", "HD", "MA"],
}

# Kelly ramp (mirrors paper_trading_monitor.py constants)
KELLY_PHASES = [
    (30,   0.005,  "Phase 1 (0–30 trades): 0.5%"),
    (60,   0.021,  "Phase 2 (31–60 trades): 2.1%"),
    (None, 0.0365, "Phase 3 (61+ trades): 3.65%"),
]
KELLY_PHASE1_PF_FLOOR   = 1.0
KELLY_PHASE1_MAX_DD_PCT = 15.0
KELLY_PHASE2_PF_FLOOR   = 1.0
KELLY_PHASE2_MAX_DD_PCT = 15.0

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_trades() -> List[dict]:
    if not TRADE_LOG.exists():
        return []
    with open(TRADE_LOG) as f:
        data = json.load(f)
    return data.get("trades", [])


def load_spy_data(use_alpaca: bool, n_bars: int = 200) -> Optional[object]:
    """
    Load SPY daily bars for regime classification.
    Returns a pandas DataFrame with columns: open, high, low, close, volume.
    Returns None if data unavailable (graceful degradation).
    """
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not available -- regime classification skipped.")
        return None

    if use_alpaca:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import Adjustment

            api_key    = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            if not api_key or not secret_key:
                print("  WARNING: ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
                return None

            client = StockHistoricalDataClient(api_key, secret_key)
            end   = datetime.now()
            start = end - timedelta(days=n_bars * 2)  # extra for weekends/holidays
            req   = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment=Adjustment.ALL,
            )
            bars = client.get_stock_bars(req)
            df   = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs("SPY", level=0)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().tail(n_bars)
            return df
        except Exception as exc:
            print(f"  WARNING: Alpaca data fetch failed ({exc}). Regime check skipped.")
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_adx(df, period: int = 14):
    """
    Compute ADX(period) on a OHLC DataFrame.
    Returns a Series of ADX values.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return None

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    dm_plus   = pd.Series(
        [u if u > d and u > 0 else 0.0 for u, d in zip(up_move, down_move)],
        index=df.index)
    dm_minus  = pd.Series(
        [d if d > u and d > 0 else 0.0 for u, d in zip(up_move, down_move)],
        index=df.index)

    # Wilder smoothing
    def wilder_smooth(s, p):
        result = [float("nan")] * len(s)
        s_vals = s.values
        # First value: sum of first p values
        first  = sum(v for v in s_vals[:p] if not math.isnan(v))
        result[p-1] = first
        for i in range(p, len(s_vals)):
            result[i] = result[i-1] - result[i-1]/p + s_vals[i]
        return pd.Series(result, index=s.index)

    atr_sm  = wilder_smooth(tr, period)
    dmp_sm  = wilder_smooth(dm_plus, period)
    dmm_sm  = wilder_smooth(dm_minus, period)

    di_plus  = 100 * dmp_sm / atr_sm.replace(0, float("nan"))
    di_minus = 100 * dmm_sm / atr_sm.replace(0, float("nan"))
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, float("nan"))
    adx      = wilder_smooth(dx.fillna(0), period)
    return adx


def compute_sma_slope(series, window: int = 50) -> object:
    """Return sign of SMA(window) slope over last 5 bars."""
    try:
        import pandas as pd
        sma = series.rolling(window).mean()
        if len(sma.dropna()) < 6:
            return "unknown"
        recent = sma.dropna().tail(6).values
        slope  = recent[-1] - recent[-6]
        if slope > 0.05 * recent[-1]:
            return "positive"
        elif slope < -0.05 * recent[-1]:
            return "negative"
        else:
            return "flat"
    except Exception:
        return "unknown"


def classify_regime(spy_df) -> dict:
    """
    Classify the current market regime using last 20 bars of SPY.
    Returns dict with classification and supporting metrics.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return {"regime": "UNKNOWN", "reason": "pandas not available"}

    if spy_df is None or len(spy_df) < 60:
        return {"regime": "UNKNOWN", "reason": "Insufficient SPY data"}

    last20 = spy_df.tail(20)
    last5  = spy_df.tail(5)

    # ADX
    adx_series = compute_adx(spy_df)
    adx_last20_avg = float(adx_series.dropna().tail(20).mean()) if adx_series is not None else None

    # SMA50 slope
    sma50_slope = compute_sma_slope(spy_df["close"], 50)

    # SPY net return last 20 bars
    spy_return_20 = (float(last20["close"].iloc[-1]) /
                     float(last20["close"].iloc[0]) - 1) * 100

    # ATR percentile (last 20 vs 200 bars)
    close  = spy_df["close"]
    prev_c = close.shift(1)
    high   = spy_df["high"]
    low    = spy_df["low"]
    tr     = (pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1)
               .max(axis=1))
    atr_pct = spy_df["close"].copy()
    # Compute ATR percentile rank of last bar vs last 200
    tr_vals = tr.dropna().tail(200).values
    last_tr = tr_vals[-1]
    pct_rank = (tr_vals < last_tr).mean() * 100

    # Classification logic (Section 6)
    if adx_last20_avg is not None and adx_last20_avg > 25 and sma50_slope == "positive":
        # Count positive slope days
        sma50 = spy_df["close"].rolling(50).mean()
        positive_days = sum(1 for i in range(1, min(21, len(sma50.dropna())))
                            if sma50.dropna().iloc[-i] > sma50.dropna().iloc[-i-1])
        if positive_days > 15:
            regime = "TRENDING"
        else:
            regime = "CHOPPY"
    elif spy_return_20 < -1.0 and pct_rank > 60:
        regime = "CORRECTIVE"
    else:
        regime = "CHOPPY"

    return {
        "regime":           regime,
        "adx_14_avg_20bar": round(adx_last20_avg, 1) if adx_last20_avg is not None else None,
        "sma50_slope":      sma50_slope,
        "spy_return_20d_%": round(spy_return_20, 2),
        "atr_percentile":   round(pct_rank, 1),
        "spy_close":        round(float(spy_df["close"].iloc[-1]), 2),
        "spy_close_date":   str(spy_df.index[-1])[:10],
        "note": {
            "TRENDING":   "SPY ADX > 25 + positive SMA50 slope > 15 days. Mean-reversion edge may be reduced.",
            "CORRECTIVE": "SPY declining + elevated ATR. Mean-reversion historically favorable.",
            "CHOPPY":     "Range-bound, moderate ATR. Mean-reversion edge typically strongest.",
            "UNKNOWN":    "Cannot classify -- check manually.",
        }.get(regime, ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics (mirror of paper_trading_monitor.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: List[dict], last_n: int = None) -> dict:
    if last_n:
        trades = trades[-last_n:]
    if not trades:
        return {"n": 0}

    n      = len(trades)
    pnls   = [float(t.get("net_pnl", 0)) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    wr       = len(wins) / n * 100
    avg_win  = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    pf       = sum(wins) / abs(sum(losses)) if sum(losses) < 0 else (
               float("inf") if wins else 0.0)

    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t.get("exit_reason", "UNKNOWN")] += 1

    equity = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        equity += p
        peak   = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0

    bars = [int(t.get("bar_count_at_exit", 0)) for t in trades if t.get("bar_count_at_exit")]

    return {
        "n":             n,
        "win_rate_%":    round(wr, 2),
        "profit_factor": round(pf, 3),
        "wl_ratio":      round(wl_ratio, 3),
        "total_pnl_$":   round(sum(pnls), 2),
        "max_dd_%":      round(max_dd_pct, 2),
        "exit_breakdown":dict(exit_counts),
        "avg_bars_held": round(sum(bars)/len(bars), 1) if bars else None,
    }


def check_signal_decay(trades: List[dict]) -> dict:
    if len(trades) < 60:
        return {"indicators_present": 0, "details": ["< 60 trades, decay check premature"],
                "recommendation": "HEALTHY"}
    indicators = []; details = []

    q_sz = max(len(trades) // 4, 15)
    qs   = [trades[i:i+q_sz] for i in range(0, len(trades), q_sz) if trades[i:i+q_sz]]

    if len(qs) >= 4:
        wrs = [sum(1 for t in q if float(t.get("net_pnl",0)) > 0) / len(q) * 100 for q in qs[-4:]]
        if all(wrs[i] > wrs[i+1] for i in range(len(wrs)-1)):
            indicators.append("WR_DECLINING_QUARTERLY")
            details.append(f"WR 3 quarters: {[f'{w:.0f}%' for w in wrs]}")

    if len(qs) >= 3:
        def pf_q(q):
            w = sum(float(t.get("net_pnl",0)) for t in q if float(t.get("net_pnl",0)) > 0)
            l = abs(sum(float(t.get("net_pnl",0)) for t in q if float(t.get("net_pnl",0)) <= 0))
            return w/l if l > 0 else float("inf")
        pfs = [pf_q(q) for q in qs[-3:]]
        if sum(1 for p in pfs[-2:] if p < 1.0) >= 2:
            indicators.append("PF_BELOW_1_TWO_QUARTERS")
            details.append(f"PF last 2 quarters: {[f'{p:.3f}' for p in pfs[-2:]]}")

    early = [int(t.get("bar_count_at_exit",0)) for t in trades[:30] if t.get("bar_count_at_exit")]
    late  = [int(t.get("bar_count_at_exit",0)) for t in trades[-30:] if t.get("bar_count_at_exit")]
    if early and late:
        ae, al = sum(early)/len(early), sum(late)/len(late)
        if al > ae * 1.30:
            indicators.append("HOLDING_PERIOD_INCREASING")
            details.append(f"Avg bars: early={ae:.1f}, recent={al:.1f} (+{(al/ae-1)*100:.0f}%)")

    all_exits    = [t.get("exit_reason","") for t in trades]
    recent30_ex  = [t.get("exit_reason","") for t in trades[-30:]]
    time_all     = sum(1 for e in all_exits   if e == "TIME") / len(all_exits)   * 100
    time_recent  = sum(1 for e in recent30_ex if e == "TIME") / len(recent30_ex) * 100
    if time_recent > VALIDATED["time_stop_pct_watch"] and time_recent > time_all * 1.5:
        indicators.append("TIME_STOP_INCREASING")
        details.append(f"TIME exits: overall={time_all:.0f}%, recent30={time_recent:.0f}%")

    n   = len(indicators)
    rec = "PAUSE_AND_RETEST" if n >= 2 else "HEALTHY"
    if n >= 2:
        details.append("2+ decay indicators. Pause and run 18-month fresh backtest.")

    return {"indicators_present": n, "indicators": indicators,
            "details": details, "recommendation": rec}


# ─────────────────────────────────────────────────────────────────────────────
# Open positions (from paper_open_positions.json)
# ─────────────────────────────────────────────────────────────────────────────

def load_open_positions() -> List[dict]:
    pos_path = HERE / "paper_open_positions.json"
    if not pos_path.exists():
        return []
    with open(pos_path) as f:
        return json.load(f).get("positions", [])


# ─────────────────────────────────────────────────────────────────────────────
# Weekly signals summary from trades log
# ─────────────────────────────────────────────────────────────────────────────

def trades_this_week(trades: List[dict]) -> List[dict]:
    """Trades entered or exited in the last 7 calendar days."""
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    return [t for t in trades
            if t.get("entry_date", "") >= cutoff or t.get("exit_date", "") >= cutoff]


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(use_alpaca: bool) -> str:
    now    = datetime.now()
    sep    = "=" * 80
    sep2   = "-" * 78
    trades = load_trades()
    open_p = load_open_positions()

    # Fetch SPY data for regime
    spy_df = load_spy_data(use_alpaca) if use_alpaca else None
    regime = classify_regime(spy_df)

    # Metrics
    all_m    = compute_metrics(trades)
    roll30_m = compute_metrics(trades, last_n=30)
    week_t   = trades_this_week(trades)
    decay    = check_signal_decay(trades)

    lines = []

    lines.append(sep)
    lines.append(f"  COMBO C V4.0 -- WEEKLY HEALTH CHECK")
    lines.append(f"  Week ending: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Total trades logged: {len(trades)}")
    lines.append(sep)

    # ── Regime ──────────────────────────────────────────────────────────────
    lines.append(f"\n  MARKET REGIME")
    lines.append(f"  {sep2}")
    lines.append(f"  Classification   : {regime['regime']}")
    if regime.get("adx_14_avg_20bar") is not None:
        lines.append(f"  SPY ADX(14) avg  : {regime['adx_14_avg_20bar']} (20-bar avg)")
    if regime.get("sma50_slope"):
        lines.append(f"  SPY SMA50 slope  : {regime['sma50_slope']}")
    if regime.get("spy_return_20d_%") is not None:
        lines.append(f"  SPY 20-bar return: {regime['spy_return_20d_%']:+.2f}%")
    if regime.get("atr_percentile") is not None:
        lines.append(f"  ATR percentile   : {regime['atr_percentile']:.0f}th (vs 200-bar history)")
    if regime.get("spy_close"):
        lines.append(f"  SPY close        : {regime['spy_close']} ({regime.get('spy_close_date','?')})")
    if regime.get("note"):
        lines.append(f"  Note: {regime['note']}")

    regime_note = {
        "TRENDING":   "Mean-reversion edge historically REDUCED in trending markets. "
                      "Monitor closely. Consider reducing to 10% size if DD>8%.",
        "CORRECTIVE": "Mean-reversion edge historically FAVORABLE in corrective regimes. "
                      "Standard operating conditions.",
        "CHOPPY":     "Mean-reversion edge typically STRONGEST in choppy regimes. "
                      "Standard operating conditions.",
        "UNKNOWN":    "Cannot classify. Proceed with caution.",
    }.get(regime["regime"], "")
    if regime_note:
        lines.append(f"  >>> {regime_note}")

    # ── Open positions ───────────────────────────────────────────────────────
    lines.append(f"\n  OPEN POSITIONS ({len(open_p)} open)")
    lines.append(f"  {sep2}")
    if open_p:
        lines.append(f"  {'Symbol':<8} {'Entry Date':<12} {'Entry Px':>9} "
                     f"{'Shares':>7} {'Bars':>5} {'ACCEL_SL':>10} {'BB_MID':>10}")
        lines.append(f"  {'-'*70}")
        for p in open_p:
            alert = " ** TIME STOP NEAR" if int(p.get("bar_count", 0)) >= 8 else ""
            lines.append(f"  {p.get('symbol','?'):<8} {p.get('entry_date','?'):<12} "
                         f"{p.get('entry_price', 0):>9.2f} {p.get('shares', 0):>7.0f} "
                         f"{p.get('bar_count', 0):>5} "
                         f"{str(p.get('accel_sl','---')):>10} "
                         f"{str(p.get('bb_mid','---')):>10}{alert}")
    else:
        lines.append("  No open positions this week.")

    # ── This week's activity ─────────────────────────────────────────────────
    lines.append(f"\n  THIS WEEK'S ACTIVITY")
    lines.append(f"  {sep2}")
    if week_t:
        lines.append(f"  {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Reason':<14} {'Net P&L':>9} {'WL'}")
        lines.append(f"  {'-'*70}")
        for t in week_t:
            sign = "+" if float(t.get("net_pnl", 0)) >= 0 else ""
            wl   = "WIN" if t.get("won") else "LOS"
            sym  = t.get("instrument", t.get("symbol", "?"))
            lines.append(f"  {sym:<8} {t.get('entry_date','?'):<12} "
                         f"{t.get('exit_date','?'):<12} "
                         f"{t.get('exit_reason','?'):<14} "
                         f"{sign}{float(t.get('net_pnl',0)):>9.2f} {wl}")
    else:
        lines.append("  No entries or exits this week.")

    # ── Running metrics ──────────────────────────────────────────────────────
    lines.append(f"\n  RUNNING METRICS")
    lines.append(f"  {sep2}")
    if trades:
        lines.append(f"  {'Metric':<28} {'All-time':<14} {'Rolling-30':<14} {'Validated'}")
        lines.append(f"  {'-'*70}")

        def mrow(lbl, ak, rk, bk, fmt=".3f"):
            av = all_m.get(ak)
            rv = roll30_m.get(rk) if roll30_m.get("n", 0) >= 10 else None
            bv = VALIDATED.get(bk)
            avs = f"{av:{fmt}}" if av is not None else "---"
            rvs = f"{rv:{fmt}}" if rv is not None else "---"
            bvs = f"{bv:{fmt}}" if bv is not None else "---"
            return f"  {lbl:<28} {avs:<14} {rvs:<14} {bvs}"

        lines.append(mrow("N trades", "n", "n", "test_n_trades", "d"))
        lines.append(mrow("Profit Factor", "profit_factor", "profit_factor", "test_pf"))
        lines.append(mrow("Win Rate (%)", "win_rate_%", "win_rate_%", "test_wr", ".1f"))
        lines.append(mrow("W/L Ratio", "wl_ratio", "wl_ratio", "test_wl_ratio"))
        lines.append(f"  {'Total P&L ($)':<28} {all_m.get('total_pnl_$', 0):<14.2f}")
        lines.append(f"  {'Max Drawdown (%)':<28} {all_m.get('max_dd_%', 0):<14.1f}")

        if all_m.get("exit_breakdown"):
            eb  = all_m["exit_breakdown"]
            tot = all_m["n"]
            parts = "  ".join(f"{k}:{v}({v/tot*100:.0f}%)" for k, v in sorted(eb.items()))
            lines.append(f"  Exit breakdown: {parts}")
    else:
        lines.append("  No trades logged yet.")

    # ── Go/no-go status ──────────────────────────────────────────────────────
    lines.append(f"\n  GO/NO-GO STATUS")
    lines.append(f"  {sep2}")
    n     = all_m.get("n", 0)
    pf    = all_m.get("profit_factor", 0)
    wr    = all_m.get("win_rate_%", 0)
    dd    = all_m.get("max_dd_%", 0)

    if n < VALIDATED["go_min_trades"]:
        months_left = max(0, (VALIDATED["go_min_trades"] - n) / VALIDATED["test_signal_rate"])
        lines.append(f"  ACCUMULATING: {n}/{VALIDATED['go_min_trades']} trades. "
                     f"~{months_left:.1f} months to go/no-go gate.")
    else:
        pass_count = sum([pf >= VALIDATED["go_pf_floor"],
                          wr >= VALIDATED["go_wr_floor"],
                          wr <= VALIDATED["go_wr_ceiling"],
                          dd < VALIDATED["go_max_drawdown"]])
        verdict = "FULL GO" if pass_count == 4 else f"PARTIAL ({pass_count}/4 required passing)"
        lines.append(f"  {verdict}")
        lines.append(f"  PF={pf:.3f} {'OK' if pf>=1.10 else 'FAIL'}  |  "
                     f"WR={wr:.1f}% {'OK' if 38<=wr<=68 else 'FAIL'}  |  "
                     f"DD={dd:.1f}% {'OK' if dd<25 else 'FAIL'}")

    # ── Signal decay ─────────────────────────────────────────────────────────
    if len(trades) >= 60:
        lines.append(f"\n  SIGNAL DECAY CHECK: {decay['recommendation']} "
                     f"({decay['indicators_present']}/4 indicators)")
        for d in decay["details"]:
            lines.append(f"    - {d}")

    # ── Drawdown state ───────────────────────────────────────────────────────
    lines.append(f"\n  DRAWDOWN STATE")
    lines.append(f"  {sep2}")
    dd_val = all_m.get("max_dd_%", 0)
    if dd_val < 8:
        dd_level = "NORMAL"
        dd_msg   = "No size adjustments needed."
    elif dd_val < 15:
        dd_level = "WATCH"
        dd_msg   = (f"DD {dd_val:.1f}%. If TRENDING regime, reduce to 50% size until DD < 8%.")
    elif dd_val < 25:
        dd_level = "HALT"
        dd_msg   = (f"DD {dd_val:.1f}%. HALT new entries. Run 18-month fresh backtest.")
    else:
        dd_level = "FULL_PAUSE"
        dd_msg   = (f"DD {dd_val:.1f}%. FULL STRATEGY PAUSE. Go/no-go failed.")
    lines.append(f"  [{dd_level}] {dd_msg}")

    # ── Kelly ramp status ────────────────────────────────────────────────────
    lines.append(f"\n  KELLY RAMP STATUS")
    lines.append(f"  {sep2}")
    try:
        from paper_trading_monitor import (
            count_completed_trades, get_kelly_phase,
            check_kelly_phase_gate, get_current_risk_fraction, _quick_metrics,
        )
        n_live     = count_completed_trades(trades)
        phase_info = get_kelly_phase(n_live)
        gate       = check_kelly_phase_gate(trades)
        eff_frac   = get_current_risk_fraction(trades)
        gate_held  = gate["hold_at_phase"] is not None
        lines.append(f"  Completed live trades : {n_live}")
        lines.append(f"  Current phase         : {phase_info['label']}")
        lines.append(f"  Nominal risk fraction : {phase_info['risk_fraction']*100:.2f}%")
        if gate_held:
            lines.append(f"  EFFECTIVE risk frac   : {eff_frac*100:.2f}%  "
                         f"[GATE HELD — Phase {gate['hold_at_phase']} gate blocked]")
        else:
            lines.append(f"  Effective risk frac   : {eff_frac*100:.2f}%")
        if phase_info["next_threshold"]:
            lines.append(f"  Next threshold        : {phase_info['next_threshold']} trades  "
                         f"({phase_info['trades_to_next']} to go)")
        else:
            lines.append(f"  Next threshold        : N/A (final phase — quarter-Kelly active)")
        if gate["gate_messages"]:
            for msg in gate["gate_messages"]:
                icon = "WW" if "BLOCKED" in msg else "OK"
                lines.append(f"    {icon} {msg}")
        else:
            lines.append(f"    -- Phase gate not yet evaluable (need 30 live trades)")
        # Phase gate check tables (only shown when phase boundary reached)
        live_trades = [t for t in trades if t.get("period") in ("LIVE_RAMP", "LIVE_FULL")]
        ph1 = live_trades[:30]
        if len(ph1) >= 30:
            m1   = _quick_metrics(ph1)
            p1ok = "OK" if m1["pf"] >= 1.0         else "XX"
            d1ok = "OK" if m1["max_dd_pct"] < 15.0 else "XX"
            lines.append(f"  Phase 1 gate (1→2 advance requires both):")
            lines.append(f"    {p1ok} PF >= 1.0    : {m1['pf']:.3f}")
            lines.append(f"    {d1ok} max DD < 15% : {m1['max_dd_pct']:.1f}%")
        ph2 = live_trades[30:60]
        if len(ph2) >= 30:
            m2   = _quick_metrics(ph2)
            p2ok = "OK" if m2["pf"] >= 1.0         else "XX"
            d2ok = "OK" if m2["max_dd_pct"] < 15.0 else "XX"
            lines.append(f"  Phase 2 gate (2→3 advance requires both):")
            lines.append(f"    {p2ok} PF >= 1.0    : {m2['pf']:.3f}")
            lines.append(f"    {d2ok} max DD < 15% : {m2['max_dd_pct']:.1f}%")
    except ImportError:
        lines.append(f"  [Kelly ramp import error — verify paper_trading_monitor.py is in same dir]")

    # ── Monthly regime table prompt ──────────────────────────────────────────
    lines.append(f"\n  MONTHLY REGIME TABLE ENTRY (add to running log)")
    lines.append(f"  {sep2}")
    lines.append(f"  Month     : {now.strftime('%Y-%m')}")
    lines.append(f"  Regime    : {regime['regime']}")
    lines.append(f"  N signals : [ fill in from weekly scan logs ]")
    lines.append(f"  N trades  : {n}")
    lines.append(f"  Monthly PF: [ fill in from: python paper_trading_monitor.py monthly "
                 f"--month {now.strftime('%Y-%m')} ]")
    lines.append(f"  Notes     :")

    # ── Portfolio overview (only shown when trend module is deployed) ─────────
    # Detects trend module deployment by checking for trend_trades.csv in cwd
    _trend_trades_file = HERE / "trend_trades.csv"
    if _trend_trades_file.exists():
        lines.append(f"\n  PORTFOLIO OVERVIEW")
        lines.append(f"  {sep2}")
        try:
            import numpy as _np
            import pandas as _pd
            _cc_trades   = trades  # already loaded above
            _trend_df    = _pd.read_csv(_trend_trades_file)
            _trend_live  = _trend_df.copy()

            n_trend = len(_trend_live)
            try:
                from paper_trading_monitor import count_completed_trades as _ctc
                n_cc_live = _ctc(_cc_trades)
            except Exception:
                n_cc_live = sum(1 for t in _cc_trades if t.get("period") in ("LIVE_RAMP", "LIVE_FULL"))

            lines.append(f"  Combo C live trades   : {n_cc_live}")
            lines.append(f"  Trend module trades   : {n_trend}")

            # Rolling 60-trade correlation
            if "net_pnl" in _trend_live.columns and len(_trend_live) >= 5:
                cc_pnl_series = _pd.Series([
                    float(t.get("net_pnl", t.get("pnl", 0)))
                    for t in _cc_trades
                    if t.get("period") in ("LIVE_RAMP", "LIVE_FULL")
                ])
                tr_pnl_series = _trend_live["net_pnl"].astype(float)
                min_n = min(len(cc_pnl_series), len(tr_pnl_series))
                if min_n >= 5:
                    full_corr = float(_np.corrcoef(
                        cc_pnl_series.iloc[:min_n].values,
                        tr_pnl_series.iloc[:min_n].values
                    )[0, 1])
                    corr_tag = ("IN RANGE ✓" if abs(full_corr) < 0.30
                                else ("WATCH ⚠" if abs(full_corr) < 0.50
                                      else "FLAG ✗ — reduce trend size 50%"))
                    lines.append(f"  Rolling correlation   : {full_corr:.3f}  [{corr_tag}]")

                    # Regime-conditional correlation (basic: split by SPY DD proxy)
                    if min_n >= 10:
                        roll60 = (cc_pnl_series.rolling(60, min_periods=5)
                                  .corr(tr_pnl_series)).iloc[:min_n]
                        valid  = roll60.dropna()
                        if len(valid) > 0:
                            lines.append(f"  60-trade rolling corr : "
                                         f"min={valid.min():.3f}  "
                                         f"mean={valid.mean():.3f}  "
                                         f"max={valid.max():.3f}")
                        hi_pct = (valid > 0.50).mean() * 100
                        if hi_pct > 20:
                            lines.append(f"  ⚠ {hi_pct:.0f}% of rolling windows above 0.50 — "
                                         f"consider reducing trend size")
            else:
                lines.append(f"  Correlation           : [need >= 5 trend trades]")

            # Combined equity at risk
            lines.append(f"  Capital allocation    : "
                         f"Combo C={50}%  Trend={50}%  (review with run_portfolio_v1.py)")
            lines.append(f"  Run full analysis: python run_portfolio_v1.py "
                         f"--combo-c-trades paper_trades_export.csv "
                         f"--trend-trades trend_trades.csv")
        except Exception as _e:
            lines.append(f"  [Portfolio overview error: {_e}]")
    else:
        lines.append(f"\n  PORTFOLIO OVERVIEW: not active (trend module not yet deployed)")
        lines.append(f"  (When trend_trades.csv is present, combined metrics will appear here)")

    lines.append(f"\n{sep}\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Weekly strategy health check -- Section 6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alpaca",  action="store_true",
                   help="Fetch live SPY data from Alpaca for regime classification")
    p.add_argument("--daily",   action="store_true",
                   help="Use daily bars (required with --alpaca)")
    p.add_argument("--output",  default=None,
                   help="Save report to file")
    return p.parse_args()


def main():
    args      = parse_args()
    use_alpaca = args.alpaca

    report = generate_report(use_alpaca=use_alpaca)
    print(report)

    # Auto-save timestamped copy
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    auto = HERE / f"weekly_health_{ts}.txt"
    with open(auto, "w") as f:
        f.write(report)
    print(f"[Saved to {auto}]")

    # Append to rolling weekly log
    with open(WEEKLY_LOG, "a") as f:
        f.write(report + "\n")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"[Also saved to {args.output}]")


if __name__ == "__main__":
    main()
