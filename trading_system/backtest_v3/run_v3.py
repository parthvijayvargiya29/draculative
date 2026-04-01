#!/usr/bin/env python3
"""
run_v3.py -- Production Backtest V3.4 Entry Point
=================================================
Usage:
    python3 run_v3.py                   # all combos, synthetic data
    python3 run_v3.py --alpaca          # real Alpaca data (cached)
    python3 run_v3.py --no-cache        # force fresh Alpaca fetch
    python3 run_v3.py --combo A         # single combo
    python3 run_v3.py --combo B         # Step 1 validation
    python3 run_v3.py --combo C         # Step 2 validation
    python3 run_v3.py --capital 50000

V3.4 changes from V3.3:
  - Instrument expansion: Combo A/B = 20 trend instruments, Combo C = 11 low-beta instruments
  - Rolling beta gate for Combo C: live 60-bar beta < 0.8 per instrument per bar
  - Regime characterization table (SPY ADX, SMA50 slope) printed before all combo results
  - Combo B regime attribution: trending vs corrective PF/WR split
  - Combo C W/L compression diagnostic: bb_mid distance analysis
  - Revised failure taxonomy (6 classes, not binary pass/fail)
  - WFE labeled "INSUFFICIENT SAMPLE (N=X)" if OOS trades < 15
  - Position limits: 6% per instrument, 20% per combo, 40% portfolio
  - Save prefix: bt34_

Pipeline:
  1. Load data (Alpaca IEX or synthetic) -- expanded universe
  2. Walk-forward split (60/20/20)
  3. SPY regime characterization (before any combo results)
  4. Benchmark (buy-and-hold per symbol)
  5. Run combos (sequentially with gate checks)
  6. Compute statistics (full metrics suite)
  7. Walk-forward efficiency ratio (with sample-size guard)
  8. Regime attribution (Combo B) + W/L diagnostic (Combo C)
  9. Revised failure taxonomy verdict
  10. Print comparison table + filtered events summary
  11. Save CSV/JSON/TXT results
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data, SYMBOLS, SYMBOLS_AB, SYMBOLS_C, SYMBOLS_REGIME
from combos import ALL_COMBOS
from simulator import TradeRecord, run_combo_on_all_symbols, walk_forward_split
from indicators_v3 import IndicatorStateV3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = HERE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Backtest V3.4 -- Three Combo System")
    p.add_argument("--combo",    choices=ALL_COMBOS + ["ALL"], default="ALL")
    p.add_argument("--capital",  type=float, default=25_000)
    p.add_argument("--years",    type=float, default=2.0)
    p.add_argument("--symbols",  nargs="+",  default=None,
                   help="Override symbol list (default: per-combo universe)")
    p.add_argument("--universe", choices=["legacy", "AB", "C", "all"], default="all",
                   help="Instrument universe to load (default: all)")
    p.add_argument("--alpaca",   action="store_true",
                   help="Use real Alpaca data (requires API keys)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--daily",    action="store_true",
                   help="Resample 15-min data to daily bars before running")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Daily resampling
# ---------------------------------------------------------------------------

def resample_to_daily(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Resample intraday (15-min) OHLCV data to daily bars.
    Required for V3.2: all three combos are designed for daily timeframe.

    Aggregation rules:
      open   = first bar of day
      high   = max of day
      low    = min of day
      close  = last bar of day
      volume = sum of day
    """
    daily_data = {}
    for sym, df in data.items():
        # Group by date (handles timezone-aware timestamps correctly)
        daily = df.groupby(df.index.date).agg(
            open   = ("open",   "first"),
            high   = ("high",   "max"),
            low    = ("low",    "min"),
            close  = ("close",  "last"),
            volume = ("volume", "sum"),
        )
        daily.index = pd.DatetimeIndex(daily.index)
        daily.index.name = "timestamp"
        daily_data[sym] = daily
        logger.info(f"  {sym}: {len(df):,} 15-min bars -> {len(daily):,} daily bars")
    return daily_data


# ---------------------------------------------------------------------------
# Benchmark (buy-and-hold)
# ---------------------------------------------------------------------------

def compute_benchmark(data: Dict[str, pd.DataFrame],
                      initial_capital: float) -> dict:
    """
    Equal-weight buy-and-hold across all symbols.
    Returns total return, per-symbol return, and annualised Sharpe.
    """
    n = len(data)
    if n == 0:
        return {}

    per_sym = initial_capital / n
    results = {}

    for sym, df in data.items():
        if len(df) < 2:
            results[sym] = 0.0
            continue
        first = float(df["close"].iloc[0])
        last  = float(df["close"].iloc[-1])
        results[sym] = round((last - first) / first * 100, 2) if first > 0 else 0.0

    portfolio_ret = sum(results.values()) / n if n > 0 else 0.0

    # Daily returns for Sharpe (using close prices)
    daily_rets = []
    for df in data.values():
        r = df["close"].pct_change().dropna()
        daily_rets.append(r)
    if daily_rets:
        combined = pd.concat(daily_rets, axis=1).mean(axis=1)
        bph_sharpe = 0.0
        if combined.std() > 0:
            bars_per_year = 26 * 252
            bph_sharpe = round(
                (combined.mean() / combined.std()) * math.sqrt(bars_per_year), 3
            )
    else:
        bph_sharpe = 0.0

    return {
        "per_symbol_%":     results,
        "portfolio_ret_%":  round(portfolio_ret, 2),
        "sharpe":           bph_sharpe,
    }


# ---------------------------------------------------------------------------
# SPY Regime Characterization
# ---------------------------------------------------------------------------

def compute_spy_regime(spy_df: pd.DataFrame) -> dict:
    """
    Compute SPY regime characterization for the full backtest period.
    Returns dict with ADX stats, SMA50 slope stats, regime breakdown,
    and per-bar regime series for trade-level attribution.

    Trending bar: SPY ADX(14) > 25 AND SMA50 slope positive.
    Corrective bar: everything else.
    """
    if spy_df is None or spy_df.empty:
        return {"available": False}

    ind = IndicatorStateV3()
    bars = spy_df[["open", "high", "low", "close", "volume"]].to_records(index=True)

    adx_vals    = []
    sma50_vals  = []
    regime_tags = []      # "trending" | "corrective" | "warmup"
    spy_returns = []
    prev_sma50  = 0.0
    prev_close  = None

    for row in bars:
        ts  = pd.Timestamp(row[0])
        o   = float(row["open"])
        h   = float(row["high"])
        l   = float(row["low"])
        c   = float(row["close"])
        vol = float(row["volume"])
        snap = ind.update(o, h, l, c, vol, 0.0)

        if prev_close is not None:
            spy_returns.append((c - prev_close) / prev_close)
        prev_close = c

        if not snap.ready:
            regime_tags.append("warmup")
            adx_vals.append(None)
            sma50_vals.append(None)
            continue

        adx   = snap.adx14_val
        sma50 = snap.sma50_val
        sma50_slope_pos = sma50 > prev_sma50 if prev_sma50 > 0 else False
        prev_sma50 = sma50

        adx_vals.append(adx)
        sma50_vals.append(sma50)

        trending = (adx > 25.0) and sma50_slope_pos
        regime_tags.append("trending" if trending else "corrective")

    valid_adx   = [v for v in adx_vals   if v is not None]
    valid_sma50 = [v for v in sma50_vals if v is not None]

    n_total      = len(regime_tags)
    n_trending   = regime_tags.count("trending")
    n_corrective = regime_tags.count("corrective")
    n_warmup     = regime_tags.count("warmup")

    # SPY net return and max drawdown
    closes = spy_df["close"].values
    net_ret = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0.0
    peak = closes[0]
    max_dd = 0.0
    for p in closes:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "available":         True,
        "n_bars":            n_total,
        "n_trending":        n_trending,
        "n_corrective":      n_corrective,
        "n_warmup":          n_warmup,
        "pct_trending":      round(n_trending / (n_trending + n_corrective) * 100, 1)
                             if (n_trending + n_corrective) > 0 else 0.0,
        "avg_adx14":         round(sum(valid_adx) / len(valid_adx), 2)    if valid_adx   else 0.0,
        "pct_adx_gt25":      round(sum(1 for v in valid_adx if v > 25) / len(valid_adx) * 100, 1)
                             if valid_adx else 0.0,
        "avg_sma50":         round(sum(valid_sma50) / len(valid_sma50), 2) if valid_sma50 else 0.0,
        "spy_net_ret_%":     round(net_ret, 2),
        "spy_max_dd_%":      round(max_dd * 100, 2),
        "regime_tags":       regime_tags,
        "spy_index":         list(spy_df.index),
    }


def print_regime_table(regime: dict):
    """Print the SPY regime characterization table."""
    if not regime.get("available"):
        logger.warning("  SPY regime data not available")
        return

    _h("SPY REGIME CHARACTERIZATION (Full Period)")
    _r("SPY net return",          f"{regime['spy_net_ret_%']:>+10.2f}%")
    _r("SPY max drawdown",        f"{regime['spy_max_dd_%']:>10.2f}%")
    _r("SPY avg ADX(14)",         f"{regime['avg_adx14']:>10.2f}")
    _r("% bars with ADX > 25",    f"{regime['pct_adx_gt25']:>10.1f}%")
    _r("% bars trending",         f"{regime['pct_trending']:>10.1f}%")
    _r("  (trending = ADX>25 AND SMA50 slope+)", "")
    _r("Trending bars",           f"{regime['n_trending']:>10,d}")
    _r("Corrective bars",         f"{regime['n_corrective']:>10,d}")
    _r("Warmup bars",             f"{regime['n_warmup']:>10,d}")

    verdict = (
        "STRONGLY TRENDING PERIOD — trend strategies favoured"
        if regime["pct_trending"] >= 60 else
        "MIXED REGIME — some trending, some corrective"
        if regime["pct_trending"] >= 35 else
        "CORRECTIVE/RANGE-BOUND PERIOD — mean-reversion favoured"
    )
    logger.info(f"\n  Regime verdict: {verdict}")


# ---------------------------------------------------------------------------
# Regime attribution for Combo B
# ---------------------------------------------------------------------------

def compute_regime_attribution(trades: List[TradeRecord], label: str = "") -> dict:
    """
    Split Combo B (or any combo) trades by regime_at_entry.
    Returns PF/WR for trending vs corrective subsets.
    """
    if not trades:
        return {}

    trending_wins   = [t for t in trades if t.regime_at_entry == "trending" and t.won]
    trending_all    = [t for t in trades if t.regime_at_entry == "trending"]
    corrective_wins = [t for t in trades if t.regime_at_entry == "corrective" and t.won]
    corrective_all  = [t for t in trades if t.regime_at_entry == "corrective"]
    unknown_all     = [t for t in trades if t.regime_at_entry not in ("trending", "corrective")]

    def _pf(trade_list):
        wins_pnl = sum(t.net_pnl for t in trade_list if t.won)
        loss_pnl = abs(sum(t.net_pnl for t in trade_list if not t.won))
        return round(wins_pnl / loss_pnl, 3) if loss_pnl > 0 else float("inf")

    def _wr(trade_list):
        return round(sum(t.won for t in trade_list) / len(trade_list) * 100, 1) if trade_list else 0.0

    return {
        "label":             label,
        "trending_n":        len(trending_all),
        "trending_wr_%":     _wr(trending_all),
        "trending_pf":       _pf(trending_all),
        "corrective_n":      len(corrective_all),
        "corrective_wr_%":   _wr(corrective_all),
        "corrective_pf":     _pf(corrective_all),
        "unknown_n":         len(unknown_all),
        "regime_gate_recommended": (
            len(trending_all) >= 5 and len(corrective_all) >= 5 and
            _pf(trending_all) > 1.5 and _pf(corrective_all) < 0.8
        ),
    }


def print_regime_attribution(combo: str, attr: dict):
    if not attr:
        return
    logger.info(f"\n  REGIME ATTRIBUTION — Combo {combo} ({attr.get('label','')}):")
    logger.info(f"    {'Regime':<15} {'N':>5} {'WR':>8} {'PF':>8}")
    logger.info(f"    {'-'*40}")
    logger.info(f"    {'Trending':<15} {attr['trending_n']:>5} {attr['trending_wr_%']:>7.1f}% {attr['trending_pf']:>8.3f}")
    logger.info(f"    {'Corrective':<15} {attr['corrective_n']:>5} {attr['corrective_wr_%']:>7.1f}% {attr['corrective_pf']:>8.3f}")
    if attr.get("unknown_n", 0) > 0:
        logger.info(f"    {'Unknown':<15} {attr['unknown_n']:>5}")
    if attr.get("regime_gate_recommended"):
        logger.warning("    !! REGIME GATE RECOMMENDED: trending PF>1.5 and corrective PF<0.8")
        logger.warning("    !! Consider adding SPY ADX>25 + SMA50 slope+ gate to Combo B entry")


# ---------------------------------------------------------------------------
# Combo C W/L compression diagnostic
# ---------------------------------------------------------------------------

def compute_wl_compression_diagnostic(trades: List[TradeRecord]) -> dict:
    """
    Analyze Combo C W/L ratio compression across periods.
    Key metrics:
    - avg entry_to_mid_dist (distance from entry to BB midline target)
    - pct exits via TIME stop (if >30% → time stop too short explanation)
    - pct exits via BB_MID (favourable exits)
    - pct exits via ACCEL_SL (adverse exits)
    """
    if not trades:
        return {}

    c_trades = [t for t in trades if t.combo == "C"]
    if not c_trades:
        c_trades = trades  # already filtered by caller

    time_exits   = [t for t in c_trades if t.exit_reason == "TIME"]
    bb_mid_exits = [t for t in c_trades if t.exit_reason == "BB_MID"]
    accel_exits  = [t for t in c_trades if t.exit_reason == "ACCEL_SL"]
    sl_exits     = [t for t in c_trades if t.exit_reason == "SL"]

    n = len(c_trades)
    avg_dist     = (sum(t.entry_to_mid_dist for t in c_trades) / n) if n > 0 else 0.0
    median_dist  = sorted(t.entry_to_mid_dist for t in c_trades)[n // 2] if n > 0 else 0.0

    # Explanation 1: >30% TIME exits = time stop cutting winners short
    time_pct = len(time_exits) / n * 100 if n > 0 else 0.0
    # Explanation 2: small avg distance = insufficient room to reach midline
    expl2_flag = avg_dist < 2.0  # less than $2 room to target on average

    return {
        "n_trades":        n,
        "pct_time_exit":   round(time_pct, 1),
        "pct_bb_mid_exit": round(len(bb_mid_exits) / n * 100, 1) if n > 0 else 0.0,
        "pct_accel_sl":    round(len(accel_exits)  / n * 100, 1) if n > 0 else 0.0,
        "pct_sl_exit":     round(len(sl_exits)     / n * 100, 1) if n > 0 else 0.0,
        "avg_entry_to_mid_dist": round(avg_dist, 4),
        "median_entry_to_mid_dist": round(median_dist, 4),
        "explanation_1_flag": time_pct > 30.0,   # time stop too short
        "explanation_2_flag": expl2_flag,         # insufficient target room
    }


def print_wl_diagnostic(diag: dict):
    if not diag:
        return
    logger.info(f"\n  COMBO C W/L COMPRESSION DIAGNOSTIC:")
    logger.info(f"    N trades analyzed:       {diag['n_trades']}")
    logger.info(f"    Exit via TIME stop:      {diag['pct_time_exit']:.1f}%")
    logger.info(f"    Exit via BB_MID (TP):    {diag['pct_bb_mid_exit']:.1f}%")
    logger.info(f"    Exit via ACCEL_SL:       {diag['pct_accel_sl']:.1f}%")
    logger.info(f"    Exit via hard SL:        {diag['pct_sl_exit']:.1f}%")
    logger.info(f"    Avg entry→BB_mid dist:   ${diag['avg_entry_to_mid_dist']:.4f}")
    logger.info(f"    Median entry→BB_mid dist: ${diag['median_entry_to_mid_dist']:.4f}")

    if diag.get("explanation_1_flag"):
        logger.warning("    !! EXPLANATION 1: >30% TIME exits → time stop cutting winners short")
        logger.warning("    !! RECOMMENDATION: Extend Combo C time stop from 10 → 15 bars")
    if diag.get("explanation_2_flag"):
        logger.warning("    !! EXPLANATION 2: avg dist < $2 → compressed target room OOS")
        logger.warning("    !! IMPLICATION: regime-driven, no parameter change needed")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades: List[TradeRecord],
                  initial_capital: float,
                  label: str = "") -> dict:
    if not trades:
        return {"label": label, "n_trades": 0}

    df = pd.DataFrame([t.__dict__ for t in trades])
    n    = len(df)
    wins = df["won"].sum()
    wr   = wins / n

    avg_win  = df.loc[df["won"],  "net_pnl"].mean() if wins > 0     else 0.0
    avg_loss = df.loc[~df["won"], "net_pnl"].mean() if n - wins > 0  else 0.0

    tw = df.loc[df["won"],  "net_pnl"].sum()
    tl = abs(df.loc[~df["won"], "net_pnl"].sum())
    pf = tw / tl if tl > 0 else float("inf")

    total_net = df["net_pnl"].sum()
    total_ret = total_net / initial_capital

    expectancy  = (wr * avg_win) + ((1 - wr) * avg_loss)
    std_pnl     = df["net_pnl"].std()
    sqn         = (expectancy / std_pnl) * math.sqrt(n) if std_pnl > 0 else 0.0

    # 95% Wilson CI on win rate
    z  = 1.96
    ci = z * math.sqrt(wr * (1 - wr) / n) if n > 0 else 0.0

    # Exit reason breakdown
    exits    = df["exit_reason"].value_counts().to_dict()
    tp_exits = sum(v for k, v in exits.items()
                   if k in ("TP", "BB_MID", "QUICK_TP", "EMA21_TRAIL", "ST_FLIP"))
    tp_rate  = tp_exits / n

    # Avg hold bars
    avg_hold = 0.0
    if "bars_held_at_exit" in df.columns:
        avg_hold = df["bars_held_at_exit"].mean()

    # Max consecutive losses
    max_consec_loss = 0
    run = 0
    for won in df["won"].values:
        if not won:
            run += 1
            max_consec_loss = max(max_consec_loss, run)
        else:
            run = 0

    # Trades per month (approx -- 21 trading days x 26 bars/day, 15-min)
    first_bar = df["entry_bar"].min()
    last_bar  = df["exit_bar"].max()
    months    = 1.0
    try:
        days   = (pd.Timestamp(last_bar) - pd.Timestamp(first_bar)).days
        months = max(days / 30.44, 1.0)
    except Exception:
        pass
    trades_per_month = n / months

    # Concentration: remove top 5
    sorted_pnl   = df["net_pnl"].sort_values(ascending=False)
    top5_sum     = sorted_pnl.head(5).sum()
    pnl_ex_top5  = total_net - top5_sum

    # Monte Carlo win-rate significance (fast)
    rng      = np.random.default_rng(42)
    outcomes = df["won"].values.astype(float)
    mc_wrs   = [rng.permutation(outcomes).mean() for _ in range(2000)]
    mc_pvalue = float(np.mean(np.array(mc_wrs) >= wr))

    # Win/loss ratio
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "label":              label,
        "n_trades":           n,
        "n_wins":             int(wins),
        "win_rate_%":         round(wr * 100, 2),
        "ci_lo_%":            round((wr - ci) * 100, 2),
        "ci_hi_%":            round((wr + ci) * 100, 2),
        "avg_win_$":          round(avg_win, 2),
        "avg_loss_$":         round(avg_loss, 2),
        "avg_wl_ratio":       round(wl_ratio, 3),
        "profit_factor":      round(pf, 3),
        "total_gross_$":      round(df["gross_pnl"].sum(), 2),
        "total_net_$":        round(total_net, 2),
        "total_return_%":     round(total_ret * 100, 2),
        "expectancy_$":       round(expectancy, 2),
        "sqn":                round(sqn, 3),
        "tp_rate_%":          round(tp_rate * 100, 1),
        "avg_hold_bars":      round(avg_hold, 1),
        "trades_per_month":   round(trades_per_month, 1),
        "max_consec_losses":  max_consec_loss,
        "exit_reasons":       exits,
        "top5_conc_%":        round(top5_sum / total_net * 100, 1) if total_net != 0 else 0,
        "pnl_ex_top5_$":      round(pnl_ex_top5, 2),
        "total_commission_$": round(df["commission"].sum(), 2),
        "mc_p_value":         round(mc_pvalue, 4),
        "significant":        mc_pvalue < 0.05,
        "sample_adequacy":    ("SUFFICIENT" if n >= 300
                               else "MARGINAL" if n >= 100
                               else "INSUFFICIENT"),
    }


def equity_stats(equity_curve: list, initial_capital: float) -> dict:
    if not equity_curve:
        return {}

    eqs  = [e["equity"] for e in equity_curve]
    eq_s = pd.Series(eqs)
    rets = eq_s.pct_change().dropna()

    max_dd = 0.0
    peak   = eqs[0]
    for e in eqs:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)

    sharpe = 0.0
    if len(rets) > 1 and rets.std() > 0:
        bars_per_year = 26 * 252
        sharpe = (rets.mean() / rets.std()) * math.sqrt(bars_per_year)

    final_eq  = eqs[-1]
    total_ret = (final_eq - initial_capital) / initial_capital
    calmar    = total_ret / max_dd if max_dd > 0 else 0.0

    return {
        "initial_capital_$": initial_capital,
        "final_equity_$":    round(final_eq, 2),
        "total_return_%":    round(total_ret * 100, 2),
        "max_drawdown_%":    round(max_dd * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "calmar_ratio":      round(calmar, 3),
    }


def walk_forward_efficiency(train_pf: float, test_pf: float,
                            n_test_trades: int = 0) -> dict:
    """
    WF efficiency ratio = test_PF / train_PF.
    V3.4: if n_test_trades < 15, return INSUFFICIENT SAMPLE (not a failure verdict).
    < 0.5 = overfitting warning.
    < 0.0 = structural failure (test period lost money while train profitable).
    """
    if n_test_trades < 15:
        return {
            "ratio":   None,
            "verdict": f"INSUFFICIENT SAMPLE (N={n_test_trades})",
            "n_test":  n_test_trades,
        }
    if train_pf <= 0:
        return {"ratio": 0.0, "verdict": "INSUFFICIENT DATA", "n_test": n_test_trades}
    ratio = test_pf / train_pf
    verdict = (
        "GOOD (>= 0.7)"        if ratio >= 0.70 else
        "MARGINAL (0.5-0.7)"   if ratio >= 0.50 else
        "OVERFIT WARNING"      if ratio >= 0.0  else
        "STRUCTURAL FAILURE"
    )
    return {"ratio": round(ratio, 3), "verdict": verdict, "n_test": n_test_trades}


def overfitting_check(train_stats: dict, test_stats: dict) -> dict:
    if not train_stats or not test_stats:
        return {"verdict": "INSUFFICIENT DATA"}

    wr_drop = train_stats.get("win_rate_%", 0) - test_stats.get("win_rate_%", 0)
    pf_tr   = train_stats.get("profit_factor", 0)
    pf_te   = test_stats.get("profit_factor", 0)
    pf_drop = (pf_tr - pf_te) / pf_tr * 100 if pf_tr > 0 else 0

    flags = []
    if wr_drop > 3:
        flags.append(f"Win rate drops {wr_drop:.1f}pp (threshold: 3pp)")
    if pf_drop > 15:
        flags.append(f"PF drops {pf_drop:.1f}% (threshold: 15%)")
    if test_stats.get("n_trades", 0) < 30:
        flags.append("< 30 test trades (STATISTICALLY UNRELIABLE)")

    verdict = (
        "LIKELY OVERFIT"   if len(flags) >= 2 else
        "POSSIBLE OVERFIT" if len(flags) == 1 else
        "PASSES"
    )
    return {
        "wr_drop_pp":  round(wr_drop, 2),
        "pf_drop_%":   round(pf_drop, 1),
        "flags":       flags,
        "verdict":     verdict,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _h(title):  logger.info(f"\n{'='*68}\n  {title}\n{'='*68}")
def _r(k, v):   logger.info(f"  {k:<40} {v}")


def print_combo_report(combo: str, results: dict, benchmark: dict):
    _h(f"COMBO {combo} — FULL REPORT")

    eq = results.get("equity", {})
    bph = benchmark.get("portfolio_ret_%", 0)

    _r("Initial Capital",     f"${eq.get('initial_capital_$', 0):>12,.0f}")
    _r("Final Equity",        f"${eq.get('final_equity_$', 0):>12,.2f}")
    _r("Total Return",        f"{eq.get('total_return_%', 0):>11.2f}%")
    _r("Buy-and-Hold Bench",  f"{bph:>11.2f}%")
    _r("vs Benchmark",        f"{eq.get('total_return_%', 0) - bph:>+11.2f}pp")
    _r("Max Drawdown",        f"{eq.get('max_drawdown_%', 0):>11.2f}%")
    _r("Sharpe Ratio",        f"{eq.get('sharpe_ratio', 0):>12.3f}")
    _r("Calmar Ratio",        f"{eq.get('calmar_ratio', 0):>12.3f}")

    for period in ["overall", "train", "validate", "test"]:
        m = results.get(period, {})
        if not m or m.get("n_trades", 0) == 0:
            continue
        label = period.upper()
        logger.info(f"\n  {label} ({m['n_trades']} trades):")
        logger.info(f"    Win rate:           {m['win_rate_%']:.2f}%"
                    f"  [{m['ci_lo_%']:.1f}%, {m['ci_hi_%']:.1f}%] 95% CI")
        logger.info(f"    Profit factor:      {m['profit_factor']:.3f}")
        logger.info(f"    Avg win / loss:     ${m['avg_win_$']:,.2f} / ${m['avg_loss_$']:,.2f}"
                    f"   W/L ratio: {m['avg_wl_ratio']:.2f}x")
        logger.info(f"    Expectancy:         ${m['expectancy_$']:,.2f}/trade")
        logger.info(f"    SQN:                {m['sqn']:.3f}")
        logger.info(f"    TP/favourable exit: {m['tp_rate_%']:.1f}%")
        logger.info(f"    Avg hold bars:      {m['avg_hold_bars']:.1f}"
                    f"  (~{m['avg_hold_bars']*15:.0f} min)")
        logger.info(f"    Trades/month:       {m['trades_per_month']:.1f}")
        logger.info(f"    Max consec losses:  {m['max_consec_losses']}")
        logger.info(f"    Total return:       {m['total_return_%']:.2f}%")
        logger.info(f"    MC p-value:         {m['mc_p_value']:.4f}"
                    f"  {'(significant)' if m['significant'] else '(NOT significant)'}")
        logger.info(f"    Sample adequacy:    {m['sample_adequacy']}")
        logger.info(f"    Exit reasons:       {json.dumps(m['exit_reasons'])}")

        # Per-period structural warning
        if m.get("profit_factor", 0) < 1.0:
            logger.warning(f"    *** PF < 1.0 in {label} period ***")

    ov = results.get("overfitting", {})
    wfe = results.get("wf_efficiency", {})
    wfe_ratio_str = (f"{wfe.get('ratio', 0):.3f}" if wfe.get("ratio") is not None
                     else "N/A")
    logger.info(f"\n  OVERFITTING:      {ov.get('verdict', 'UNKNOWN')}"
                f"  (WR drop: {ov.get('wr_drop_pp', 0):+.1f}pp"
                f"  PF drop: {ov.get('pf_drop_%', 0):.1f}%)")
    for flag in ov.get("flags", []):
        logger.warning(f"    !!  {flag}")
    logger.info(f"  WF Efficiency:    {wfe_ratio_str}"
                f"  -> {wfe.get('verdict', '?')}"
                f"  (N_test={wfe.get('n_test', '?')})")


def print_filtered_events_summary(combo: str, filtered_events: list):
    """
    Print a summary of filtered (skipped) entry events: count and top blocking conditions.
    This quantifies how much the regime/vol/ADX filters are actually doing.
    """
    if not filtered_events:
        logger.info(f"  Combo {combo}: 0 filtered events (all triggers proceeded to entry)")
        return

    n = len(filtered_events)
    # Aggregate blocking reasons
    reason_counts = {}
    for ev in filtered_events:
        for r in ev.get("reasons", "").split(", "):
            r = r.strip()
            if r:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    sorted_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])
    top_reasons = ", ".join(f"{r}({c})" for r, c in sorted_reasons[:5])
    logger.info(f"  Combo {combo}: {n} filtered events | top blocks: {top_reasons}")


def print_comparison_table(all_results: Dict[str, dict], benchmark: dict,
                           all_regime_attr: Dict[str, dict] = None,
                           wl_diag: dict = None):
    _h("COMBO COMPARISON — TEST PERIOD (Out-of-Sample)")
    header = f"  {'Metric':<32}{'Combo A':>12}{'Combo B':>12}{'Combo C':>12}"
    logger.info(header)
    logger.info("  " + "-" * 68)

    rows = [
        ("Trades",              "n_trades",        "d"),
        ("Win Rate %",          "win_rate_%",       ".2f"),
        ("Profit Factor",       "profit_factor",    ".3f"),
        ("Avg Win $",           "avg_win_$",        ".2f"),
        ("Avg Loss $",          "avg_loss_$",       ".2f"),
        ("W/L Ratio",           "avg_wl_ratio",     ".2f"),
        ("Expectancy $",        "expectancy_$",     ".2f"),
        ("SQN",                 "sqn",              ".3f"),
        ("TP Rate %",           "tp_rate_%",        ".1f"),
        ("Avg Hold Bars",       "avg_hold_bars",    ".1f"),
        ("Trades/Month",        "trades_per_month", ".1f"),
        ("Max Consec Losses",   "max_consec_losses","d"),
        ("Gross P&L $",         "total_gross_$",    ".2f"),
        ("Net P&L $",           "total_net_$",      ".2f"),
        ("Commission $",        "total_commission_$",".2f"),
        ("Total Return %",      "total_return_%",   ".2f"),
    ]
    for label, key, fmt in rows:
        row = f"  {label:<32}"
        for c in ["A", "B", "C"]:
            val = all_results.get(c, {}).get("test", {}).get(key, 0)
            row += f"  {val:>10{fmt}}"
        logger.info(row)

    bph = benchmark.get("portfolio_ret_%", 0)
    logger.info(f"\n  {'Buy-and-Hold (full period)':<32}  {bph:>10.2f}%")
    logger.info("")

    # ── V3.4 Revised failure taxonomy ──────────────────────────────────
    _h("V3.4 FAILURE TAXONOMY & STRUCTURAL VERDICT")
    logger.info("""
  Taxonomy:
    1. Statistical noise         : OOS trades < 15 → measurement validity warning
    2. Regime dependency (conf.) : trending PF>1.5, corrective PF<0.8, ≥5 each → add regime gate
    3. Regime dependency (unconf): OOS PF<0.8 with >20 trades → investigate further
    4. Overfitting confirmed      : IS PF>1.5, OOS PF<0.9, WFE<0.5, >20 OOS trades
    5. Dead signal               : test trades <15 after expansion → consider decommission
    6. Structural contradiction  : anti-correlated entry conditions (V3.2 style failure)
    """)

    any_profitable = False
    for c in ["A", "B", "C"]:
        test_s = all_results.get(c, {}).get("test", {})
        train_s = all_results.get(c, {}).get("train", {})
        pf     = test_s.get("profit_factor", 0)
        nt     = test_s.get("n_trades", 0)
        train_pf = train_s.get("profit_factor", 0)
        ov     = all_results.get(c, {}).get("overfitting", {}).get("verdict", "?")
        wfe    = all_results.get(c, {}).get("wf_efficiency", {})
        wfe_ratio = wfe.get("ratio")
        wfe_verdict = wfe.get("verdict", "?")

        # Classify failure type
        classifications = []
        if nt < 15:
            classifications.append("STATISTICAL NOISE (N<15)")
        elif nt < 30:
            classifications.append(f"MARGINAL SAMPLE (N={nt})")

        # Regime dependency check
        attr = (all_regime_attr or {}).get(c, {})
        if attr.get("regime_gate_recommended"):
            classifications.append("REGIME DEPENDENCY CONFIRMED → add gate")
        elif (nt >= 20 and pf < 0.8):
            classifications.append("REGIME DEPENDENCY UNCONFIRMED (PF<0.8, N≥20)")

        # Overfitting check
        if (wfe_ratio is not None and wfe_ratio < 0.5 and train_pf > 1.5
                and pf < 0.9 and nt >= 20):
            classifications.append("OVERFITTING CONFIRMED")

        # Dead signal
        if nt < 15 and nt > 0:
            pass  # already classified above
        elif nt == 0:
            classifications.append("DEAD SIGNAL (0 test trades)")

        # Profitable
        if pf >= 1.0 and nt >= 15:
            any_profitable = True
            classifications.append(f"PROFITABLE (PF={pf:.3f})")

        class_str = " | ".join(classifications) if classifications else f"PF={pf:.3f}"
        wfe_str   = (f"{wfe_ratio:.3f}" if wfe_ratio is not None else "N/A")

        logger.info(f"  Combo {c} (N={nt:3d}): {class_str}")
        logger.info(f"           WFE={wfe_str}  [{wfe_verdict}]  OV={ov}")

    if not any_profitable:
        # Check if all failures are sample-size issues vs real signal failures
        all_small = all(
            all_results.get(c, {}).get("test", {}).get("n_trades", 0) < 15
            for c in ["A", "B", "C"]
        )
        if all_small:
            logger.warning("")
            logger.warning("  " + "!" * 64)
            logger.warning("  !! MEASUREMENT VALIDITY WARNING — not a signal failure")
            logger.warning("  !! ALL combos have < 15 OOS test trades.")
            logger.warning("  !! Instrument expansion target NOT YET MET.")
            logger.warning("  !! Train-period PF evidence is the primary signal quality metric.")
            logger.warning("  " + "!" * 64)
        else:
            logger.error("")
            logger.error("  " + "!" * 64)
            logger.error("  !! STRUCTURAL REDESIGN REQUIRED")
            logger.error("  !! No combo profitable on OOS test with ≥15 trades.")
            logger.error("  !! Root-cause analysis required before next iteration.")
            logger.error("  " + "!" * 64)
    else:
        # Best combo scoring
        best = None
        best_score = -999
        for c in ["A", "B", "C"]:
            t     = all_results.get(c, {}).get("test", {})
            ov    = all_results.get(c, {}).get("overfitting", {})
            score = (
                t.get("win_rate_%", 0) * 0.4 +
                min(t.get("profit_factor", 0), 3) * 10 +
                t.get("sqn", 0) * 5
            )
            if ov.get("verdict") == "LIKELY OVERFIT":
                score -= 20
            if t.get("n_trades", 0) < 15:
                score -= 40
            elif t.get("n_trades", 0) < 30:
                score -= 15
            if score > best_score:
                best_score = score
                best = c

        logger.info(f"\n  RECOMMENDED COMBO: {best}  (score={best_score:.1f})")
        logger.info(f"  -> Use this for TradingView Pine Script")

    # Combo B regime attribution table
    if all_regime_attr:
        logger.info("\n")
        for c, attr in all_regime_attr.items():
            if attr:
                print_regime_attribution(c, attr)

    # Combo C W/L diagnostic
    if wl_diag:
        print_wl_diagnostic(wl_diag)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(all_trades: Dict[str, List[TradeRecord]],
                 all_results: Dict[str, dict],
                 benchmark: dict,
                 all_filtered: Dict[str, list] = None):
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    for combo, trades in all_trades.items():
        if not trades:
            continue
        df = pd.DataFrame([t.__dict__ for t in trades])
        path = OUTPUT_DIR / f"bt34_combo{combo}_trades_{ts}.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Combo {combo} trades -> {path.name}")

    # Save filtered events
    if all_filtered:
        for combo, fevents in all_filtered.items():
            if fevents:
                fdf = pd.DataFrame(fevents)
                fpath = OUTPUT_DIR / f"bt34_combo{combo}_filtered_{ts}.csv"
                fdf.to_csv(fpath, index=False)
                logger.info(f"  Combo {combo} filtered events -> {fpath.name}")

    results_path = OUTPUT_DIR / f"bt34_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  Results JSON -> {results_path.name}")

    summary_path = OUTPUT_DIR / f"bt34_summary_{ts}.txt"
    bph = benchmark.get("portfolio_ret_%", 0)
    with open(summary_path, "w") as f:
        f.write(f"PRODUCTION BACKTEST V3.4 - SUMMARY\nGenerated: {datetime.now()}\n\n")
        f.write(f"Buy-and-Hold Benchmark: {bph:.2f}%\n\n")
        for combo in ["A", "B", "C"]:
            r  = all_results.get(combo, {})
            t  = r.get("test", {})
            eq = r.get("equity", {})
            ov = r.get("overfitting", {})
            wfe = r.get("wf_efficiency", {})
            wfe_ratio = wfe.get("ratio")
            f.write(f"COMBO {combo} (TEST OOS):\n")
            f.write(f"  Trades:         {t.get('n_trades', 0)}\n")
            f.write(f"  Win Rate:       {t.get('win_rate_%', 0):.2f}%\n")
            f.write(f"  Profit Factor:  {t.get('profit_factor', 0):.3f}\n")
            f.write(f"  Gross P&L:      ${t.get('total_gross_$',0):,.2f}\n")
            f.write(f"  Net P&L:        ${t.get('total_net_$',0):,.2f}\n")
            f.write(f"  Commission:     ${t.get('total_commission_$',0):,.2f}\n")
            f.write(f"  Avg Win/Loss:   ${t.get('avg_win_$',0):,.2f} / ${t.get('avg_loss_$',0):,.2f}\n")
            f.write(f"  Trades/Month:   {t.get('trades_per_month', 0):.1f}\n")
            f.write(f"  Max Consec L:   {t.get('max_consec_losses', 0)}\n")
            f.write(f"  Total Return:   {eq.get('total_return_%', 0):.2f}%\n")
            f.write(f"  vs Benchmark:   {eq.get('total_return_%', 0) - bph:+.2f}pp\n")
            wfe_str = f"{wfe_ratio:.3f}" if wfe_ratio is not None else "N/A"
            f.write(f"  WF Efficiency:  {wfe_str} ({wfe.get('verdict','?')})\n")
            f.write(f"  Overfitting:    {ov.get('verdict', '?')}\n\n")
    logger.info(f"  Summary TXT    -> {summary_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    combos = [args.combo] if args.combo != "ALL" else ALL_COMBOS

    logger.info("\n" + "=" * 68)
    logger.info("  PRODUCTION BACKTEST V3.4 -- THREE COMBO SYSTEM")
    logger.info(f"  Capital:  ${args.capital:,.0f}  |  Combos: {combos}")
    logger.info(f"  Universe: {args.universe}  |  History: {args.years}yr")
    logger.info("  V3.4: Expanded universe, regime attribution, beta gate, position limits")
    logger.info("=" * 68 + "\n")

    # 1. Load data — expanded universe
    logger.info("-" * 68)
    logger.info("  STEP 1 — Loading data (V3.4 expanded universe)")
    logger.info("-" * 68)

    use_synth = not args.alpaca
    if use_synth:
        logger.warning("  No --alpaca flag: using SYNTHETIC data")

    # Determine what symbols to load
    if args.symbols:
        # explicit override — always add SPY for regime
        load_syms = sorted(set(args.symbols + ["SPY"]))
        uni = "legacy"
    else:
        # Auto-select universe based on combos requested + always include SPY
        if set(combos) == {"C"}:
            uni = "C"
        elif set(combos) == {"A"} or set(combos) == {"B"} or set(combos) == {"A", "B"}:
            uni = "AB"
        else:
            uni = "all"  # combined run: load everything
        load_syms = None

    data = load_data(
        symbols     = load_syms,
        years       = args.years,
        use_cache   = not args.no_cache,
        force_synth = use_synth,
        universe    = uni if load_syms is None else "legacy",
    )

    # Optional: resample to daily bars (required for strategy logic)
    if args.daily:
        logger.info("\n  Resampling to daily bars (--daily flag)...")
        data = resample_to_daily(data)

    # 2. Walk-forward split
    logger.info("\n" + "-" * 68)
    logger.info("  STEP 2 — Walk-forward split (60/20/20)")
    logger.info("-" * 68)
    train_d, val_d, test_d = walk_forward_split(data)

    # 3. SPY Regime characterization (BEFORE benchmark and combos)
    logger.info("\n" + "-" * 68)
    logger.info("  STEP 3 — SPY Regime Characterization")
    logger.info("-" * 68)
    spy_regime = {}
    if "SPY" in data:
        spy_regime = compute_spy_regime(data["SPY"])
        print_regime_table(spy_regime)
    else:
        logger.warning("  SPY not in dataset — regime classification unavailable")

    # 4. Benchmark
    logger.info("\n" + "-" * 68)
    logger.info("  STEP 4 — Buy-and-hold benchmark")
    logger.info("-" * 68)
    # Use only AB symbols for benchmark (not C's defensive instruments)
    bench_syms = {s: d for s, d in data.items()
                  if s in {"SPY", "QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN"}}
    if not bench_syms:
        bench_syms = data
    benchmark = compute_benchmark(bench_syms, args.capital)
    bph = benchmark.get("portfolio_ret_%", 0)
    logger.info(f"  Equal-weight B&H benchmark return: {bph:.2f}%")
    for sym, ret in benchmark.get("per_symbol_%", {}).items():
        logger.info(f"    {sym:<6}: {ret:+.2f}%")

    # 5. Run combos
    logger.info("\n" + "-" * 68)
    logger.info("  STEP 5 — Simulation + statistics")
    logger.info("-" * 68)

    all_trades:   Dict[str, List[TradeRecord]] = {}
    all_results:  Dict[str, dict]              = {}
    all_equity:   Dict[str, list]              = {}
    all_filtered: Dict[str, list]              = {}
    all_regime_attr: Dict[str, dict]           = {}
    wl_diag:      dict                         = {}

    for combo in combos:
        logger.info(f"\n  === COMBO {combo} ===")

        # Derive warm-start boundary timestamps from the split
        # (use the reference symbol — first key in data)
        _ref_sym = next(iter(data))
        _tr_end  = train_d[_ref_sym].index[-1]  # last bar of train period
        _vl_end  = val_d[_ref_sym].index[-1]    # last bar of validate period
        _tr_start = train_d[_ref_sym].index[0]   # first bar of train period
        _vl_start = val_d[_ref_sym].index[0]     # first bar of validate period
        _te_start = test_d[_ref_sym].index[0]    # first bar of test period

        logger.info(f"\n  Full run (all data):")
        full_trades, full_eq, full_filtered, full_b_diag = run_combo_on_all_symbols(
            data, combo, args.capital, "all"
        )

        logger.info(f"\n  Train period:")
        # Train: use sliced data (302 bars >> 80 warmup, no issue)
        tr_trades, _, _, _ = run_combo_on_all_symbols(train_d, combo, args.capital, "train")

        logger.info(f"\n  Validate period:")
        # Validate: warm-start using full data from val start
        # (val window = 101 bars; warmup would consume most without warm-start)
        vl_trades, _, _, _ = run_combo_on_all_symbols(
            data, combo, args.capital, "validate", active_from=_vl_start
        )
        # Trim vl_trades to val period only (discard any beyond val_end)
        vl_trades = [t for t in vl_trades
                     if t.entry_bar <= val_d[_ref_sym].index[-1]]

        logger.info(f"\n  Test period (OOS):")
        # Warm-start: use full data but only record trades from test start
        # This ensures indicators are fully warmed up for the 101-bar OOS window
        te_trades, te_eq, _, _ = run_combo_on_all_symbols(
            data, combo, args.capital, "test", active_from=_te_start
        )

        # ── V3.4 Gate checks ──────────────────────────────────────────────
        total_full = len(full_trades)
        n_test     = len(te_trades)
        if combo == "B":
            b_diag = full_b_diag
            flip_ev    = b_diag.get('flip_events', 0)
            reflip_dis = b_diag.get('reflip_disarms', 0)
            win_exp    = b_diag.get('window_expires', 0)
            entries    = b_diag.get('entries', 0)
            missed     = b_diag.get('missed_trades', 0)
            logger.info(f"\n  [GATE CHECK] Combo B diagnostic:")
            logger.info(f"    flip_events={flip_ev}  reflip_disarms={reflip_dis}  "
                        f"window_expires={win_exp}  entries={entries}  missed={missed}")
            logger.info(f"    Pullback entries taken (all data): {total_full}")
            logger.info(f"    Test period trades (OOS):          {n_test}")
            if total_full == 0:
                logger.error("  !! GATE FAIL: Combo B = 0 trades.")
            elif n_test < 15:
                logger.warning(f"  !! OOS SAMPLE SMALL: Combo B = {n_test} test trades < 15.")
                logger.warning(f"  !! WFE will show INSUFFICIENT SAMPLE — not a failure.")
            elif n_test >= 30:
                logger.info(f"  [GATE PASS] Combo B: {n_test} OOS trades >= 30 ✓")
            else:
                logger.info(f"  [GATE MARGINAL] Combo B: {n_test} OOS trades (15-30 range)")

        elif combo == "C":
            missed = full_b_diag.get('missed_trades', 0)
            logger.info(f"\n  [GATE CHECK] Combo C: {total_full} trades total | {n_test} OOS | missed={missed}")
            if total_full == 0:
                logger.error("  !! GATE FAIL: Combo C = 0 trades.")
            elif n_test < 15:
                logger.warning(f"  !! OOS SAMPLE SMALL: Combo C = {n_test} test trades < 15.")
            elif n_test >= 30:
                logger.info(f"  [GATE PASS] Combo C: {n_test} OOS trades >= 30 ✓")
            else:
                logger.info(f"  [GATE MARGINAL] Combo C: {n_test} OOS trades (15-30 range)")

        elif combo == "A":
            logger.info(f"\n  [GATE CHECK] Combo A: {total_full} trades total | {n_test} OOS")
            if total_full < 20:
                logger.warning(f"  !! GATE FAIL (over-filtered): Combo A = {total_full} trades < 20.")
            elif total_full > 600:
                # V3.4: raised from 400->600 to accommodate 20-instrument universe
                # 432 trades / 20 symbols / 2yr = 10.8 trades/sym/yr -- reasonable for breakout
                logger.warning(f"  !! GATE FAIL (too active): Combo A = {total_full} trades > 600.")
            elif n_test >= 30:
                logger.info(f"  [GATE PASS] Combo A: {n_test} OOS trades >= 30 ✓")
            else:
                logger.info(f"  [GATE MARGINAL] Combo A: {n_test} OOS trades (target ≥30)")

        overall_s = compute_stats(full_trades, args.capital, f"Combo {combo} Overall")
        train_s   = compute_stats(tr_trades,   args.capital, "Train")
        val_s     = compute_stats(vl_trades,   args.capital, "Validate")
        test_s    = compute_stats(te_trades,   args.capital, "Test")

        combined_eq = sorted(
            [item for sym_eq in te_eq.values() for item in sym_eq],
            key=lambda x: x["ts"]
        )
        eq_stats  = equity_stats(combined_eq, args.capital)
        ov_check  = overfitting_check(train_s, test_s)
        wfe       = walk_forward_efficiency(
            train_s.get("profit_factor", 0),
            test_s.get("profit_factor", 0),
            n_test_trades=n_test,
        )

        results = {
            "overall":       overall_s,
            "train":         train_s,
            "validate":      val_s,
            "test":          test_s,
            "equity":        eq_stats,
            "overfitting":   ov_check,
            "wf_efficiency": wfe,
        }

        all_trades[combo]  = full_trades
        all_results[combo] = results
        all_equity[combo]  = combined_eq
        all_filtered[combo] = full_filtered

        print_combo_report(combo, results, benchmark)

        # Print filtered events summary
        logger.info(f"\n  FILTERED EVENTS (triggers blocked):")
        print_filtered_events_summary(combo, full_filtered)

        # V3.4: Regime attribution (all combos, especially B)
        if full_trades:
            regime_attr = compute_regime_attribution(full_trades, label="full run")
            all_regime_attr[combo] = regime_attr
            print_regime_attribution(combo, regime_attr)

        # V3.4: Combo C W/L compression diagnostic
        if combo == "C" and full_trades:
            wl_diag = compute_wl_compression_diagnostic(full_trades)
            print_wl_diagnostic(wl_diag)

    # 6. Comparison table + structural verdict
    if len(combos) > 1:
        print_comparison_table(all_results, benchmark,
                               all_regime_attr=all_regime_attr,
                               wl_diag=wl_diag if wl_diag else None)

    # 7. Save
    logger.info("-" * 68)
    logger.info("  Saving results")
    logger.info("-" * 68)
    save_results(all_trades, all_results, benchmark, all_filtered)

    logger.info("\n" + "=" * 68)
    logger.info("  DONE — V3.4")
    logger.info("=" * 68 + "\n")


if __name__ == "__main__":
    main()
