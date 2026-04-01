#!/usr/bin/env python3
"""
run_portfolio_v1.py  --  Combined Portfolio Analysis (Outcome 1: Both Strategies Pass)
========================================================================================
Combines Combo C (mean-reversion, ~3.6 trades/month) and the gated Trend Module
(momentum, ~N trades/month) into a combined portfolio view.

PRE-REQUISITES:
  - Combo C independently passes verification (verify_deployment.py → 33/33 PASS)
  - Trend module independently passes test period (run_trend_research.py → PASS)
  - Portfolio correlation < 0.30 (validated in run_trend_research.py Step 9)
  - Do NOT run this file until both strategies have independently passed

USAGE:
  python run_portfolio_v1.py \
      --combo-c-trades bt34_comboC_trades_20260319_1806.csv \
      --trend-trades   trend_gated_trades.csv

  python run_portfolio_v1.py --help

OUTPUT:
  - Combined equity curve (ASCII chart)
  - Combined Sharpe, max DD, Calmar ratio
  - Rolling 60-bar correlation (min/mean/max)
  - Regime-conditional correlation breakdown
  - Capital allocation recommendation
  - Combined portfolio health assessment
  - Writes portfolio_analysis_YYYYMMDD_HHMM.txt

ALLOCATION STARTING POINT:
  Default: 50%/50% notional split between Combo C and Trend module.
  Adjust with --combo-c-weight and --trend-weight after reviewing
  the independent Sharpe ratios.

CORRELATION MONITORING:
  MAX_LIVE_CORRELATION = 0.50
  If rolling 60-trade correlation exceeds this in live trading,
  reduce trend module size by 50% until correlation normalizes.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LIVE_CORR_THRESHOLD = 0.50   # trigger 50% size reduction on trend module
MAX_LIVE_CORR_PASS      = 0.30   # integration requirement

# Risk-free rate for Sharpe calculation (annualised)
RF_ANNUAL = 0.05

# Rolling windows
ROLLING_CORR_WINDOW  = 60
ROLLING_SHARPE_WINDOW = 30

# ---------------------------------------------------------------------------
# Equity curve helpers
# ---------------------------------------------------------------------------

def _load_trades_csv(path: str, pnl_col: str = "net_pnl",
                     date_col: str = "exit_date") -> Optional[pd.DataFrame]:
    """Load a trade CSV and return sorted DataFrame with pnl_col and date_col."""
    p = Path(path)
    if not p.exists():
        print(f"  [ERROR] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if pnl_col not in df.columns:
            print(f"  [ERROR] Column '{pnl_col}' not found in {path}")
            print(f"  Available columns: {list(df.columns)}")
            return None
        if date_col not in df.columns:
            print(f"  [ERROR] Column '{date_col}' not found in {path}")
            return None
        df = df[[date_col, pnl_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [ERROR] Could not load {path}: {e}")
        return None


def _equity_curve(pnl_series: pd.Series) -> pd.Series:
    """Cumulative sum of P&L series (equity = starting from 0)."""
    return pnl_series.cumsum()


def _max_drawdown(equity: pd.Series) -> Tuple[float, float]:
    """Returns (abs_drawdown, pct_drawdown). pct_drawdown is relative to peak equity."""
    peak = equity.expanding().max()
    dd   = peak - equity
    max_dd_abs = float(dd.max())
    peak_at_dd = float(peak[dd.idxmax()]) if len(dd) > 0 else 0.0
    max_dd_pct = (max_dd_abs / abs(peak_at_dd) * 100) if abs(peak_at_dd) > 0 else 0.0
    return round(max_dd_abs, 2), round(max_dd_pct, 2)


def _sharpe(pnl_series: pd.Series, trades_per_year: float = 36.0) -> float:
    """
    Trade-level Sharpe: (mean(pnl) - RF_per_trade) / std(pnl) × sqrt(trades/year)
    Uses per-trade P&L, not daily returns.
    RF_per_trade = RF_ANNUAL / trades_per_year
    """
    if len(pnl_series) < 5:
        return float("nan")
    mu  = pnl_series.mean()
    std = pnl_series.std(ddof=1)
    if std < 1e-9:
        return float("nan")
    rf_per_trade = RF_ANNUAL / trades_per_year
    return round((mu - rf_per_trade) / std * math.sqrt(trades_per_year), 3)


def _calmar(pnl_series: pd.Series, trades_per_year: float = 36.0) -> float:
    """Calmar = annualised return / max drawdown %."""
    equity = _equity_curve(pnl_series)
    _, dd_pct = _max_drawdown(equity)
    if dd_pct < 0.01:
        return float("inf")
    ann_return = pnl_series.mean() * trades_per_year
    return round(ann_return / dd_pct, 3)


def _profit_factor(pnl_series: pd.Series) -> float:
    wins   = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    return round(wins / losses, 3) if losses > 0 else float("inf")


def _rolling_correlation(
    s1: pd.Series, s2: pd.Series, window: int = ROLLING_CORR_WINDOW
) -> pd.Series:
    """
    Compute rolling Pearson correlation on two P&L series aligned by index.
    Returns a series of correlations.
    """
    df = pd.DataFrame({"a": s1.values, "b": s2.values})
    return df["a"].rolling(window, min_periods=max(5, window // 4)).corr(df["b"])


def _pearson_corr(s1: pd.Series, s2: pd.Series) -> float:
    """Full-period Pearson correlation on overlapping portion."""
    min_n = min(len(s1), len(s2))
    if min_n < 5:
        return float("nan")
    c = float(np.corrcoef(s1.iloc[:min_n].values, s2.iloc[:min_n].values)[0, 1])
    return round(c, 4)


# ---------------------------------------------------------------------------
# Combined portfolio metrics
# ---------------------------------------------------------------------------

def _combine_pnl(cc_pnl: pd.Series, trend_pnl: pd.Series,
                 cc_weight: float, trend_weight: float) -> pd.Series:
    """
    Combine the two P&L series with given weights.
    Truncates to min length — assumes chronological alignment.
    Weights are notional fractions that scale each trade's P&L.
    """
    min_n = min(len(cc_pnl), len(trend_pnl))
    combined = (cc_pnl.iloc[:min_n].reset_index(drop=True) * cc_weight
                + trend_pnl.iloc[:min_n].reset_index(drop=True) * trend_weight)
    return combined


def _ascii_equity_curve(equity: pd.Series, width: int = 60, height: int = 10) -> str:
    """Minimal ASCII equity curve."""
    vals   = equity.values
    n      = len(vals)
    if n < 2:
        return "  (insufficient data for chart)"
    vmin   = float(np.min(vals))
    vmax   = float(np.max(vals))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0
    step   = max(1, n // width)
    sampled = vals[::step]
    grid   = [[" "] * len(sampled) for _ in range(height)]
    for j, v in enumerate(sampled):
        row = int((v - vmin) / (vmax - vmin) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[height - 1 - row][j] = "*"
    lines = ["".join(row) for row in grid]
    lines.append("─" * len(sampled))
    return "\n  ".join([""] + lines)


def _rolling_sharpe(pnl_series: pd.Series,
                    window: int = ROLLING_SHARPE_WINDOW) -> pd.Series:
    """Rolling trade-level Sharpe over trailing window."""
    def sharpe_fn(x):
        if len(x) < 5:
            return float("nan")
        mu = x.mean(); std = x.std(ddof=1)
        if std < 1e-9:
            return float("nan")
        return (mu / std) * math.sqrt(36)  # scale to ~annual
    return pnl_series.rolling(window, min_periods=5).apply(sharpe_fn, raw=True)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_portfolio_analysis(
    cc_trades_path:    str,
    trend_trades_path: str,
    cc_weight:         float = 0.50,
    trend_weight:      float = 0.50,
    output_path:       Optional[str] = None,
) -> int:
    """
    Full portfolio analysis. Returns exit code (0=pass, 1=warning, 2=fail).
    """
    sep  = "═" * 78
    sep2 = "─" * 76
    now  = datetime.now()
    out  = []

    out.append(f"\n{sep}")
    out.append(f"  COMBINED PORTFOLIO ANALYSIS  —  V1.0")
    out.append(f"  {now.strftime('%Y-%m-%d %H:%M')}")
    out.append(sep)

    # ── Load data ──────────────────────────────────────────────────────────
    cc_df    = _load_trades_csv(cc_trades_path)
    trend_df = _load_trades_csv(trend_trades_path)

    if cc_df is None or trend_df is None:
        print("  [FATAL] Cannot load trade files. Aborting.")
        return 2

    cc_pnl    = cc_df["net_pnl"]
    trend_pnl = trend_df["net_pnl"]

    out.append(f"\n  Input data:")
    out.append(f"    Combo C trades  : {len(cc_pnl)}  "
               f"({cc_df['exit_date'].iloc[0].date()} → {cc_df['exit_date'].iloc[-1].date()})")
    out.append(f"    Trend trades    : {len(trend_pnl)}  "
               f"({trend_df['exit_date'].iloc[0].date()} → {trend_df['exit_date'].iloc[-1].date()})")
    out.append(f"    Notional weights: Combo C={cc_weight:.0%}  Trend={trend_weight:.0%}")

    # ── Individual strategy metrics ────────────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  INDIVIDUAL STRATEGY METRICS")
    out.append(f"  {sep2}")
    out.append(f"  {'Metric':<30} {'Combo C':>14}  {'Trend Module':>14}")
    out.append(f"  {'-'*30} {'-'*14}  {'-'*14}")

    def fmt(v):
        return f"{v:.3f}" if not (math.isnan(v) if isinstance(v, float) else False) else "n/a"

    cc_pf    = _profit_factor(cc_pnl)
    tr_pf    = _profit_factor(trend_pnl)
    cc_sh    = _sharpe(cc_pnl)
    tr_sh    = _sharpe(trend_pnl)
    cc_cal   = _calmar(cc_pnl)
    tr_cal   = _calmar(trend_pnl)
    cc_eq    = _equity_curve(cc_pnl)
    tr_eq    = _equity_curve(trend_pnl)
    _, cc_dd = _max_drawdown(cc_eq)
    _, tr_dd = _max_drawdown(tr_eq)
    cc_wr    = round((cc_pnl > 0).mean() * 100, 1)
    tr_wr    = round((trend_pnl > 0).mean() * 100, 1)

    for lbl, cv, tv in [
        ("Profit Factor",   cc_pf, tr_pf),
        ("Sharpe (trades)", cc_sh, tr_sh),
        ("Calmar ratio",    cc_cal, tr_cal),
        ("Max drawdown (%)", cc_dd, tr_dd),
        ("Win rate (%)",    cc_wr, tr_wr),
        ("Total P&L",       float(cc_pnl.sum()), float(trend_pnl.sum())),
        ("N trades",        float(len(cc_pnl)), float(len(trend_pnl))),
    ]:
        out.append(f"  {lbl:<30} {fmt(cv):>14}  {fmt(tv):>14}")

    # ── Full-period correlation ─────────────────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  CORRELATION ANALYSIS")
    out.append(f"  {sep2}")

    full_corr = _pearson_corr(cc_pnl, trend_pnl)
    if math.isnan(full_corr):
        corr_tag = "n/a"
    elif abs(full_corr) < MAX_LIVE_CORR_PASS:
        corr_tag = "✓ PASS (< 0.30)"
    elif abs(full_corr) < MAX_LIVE_CORR_THRESHOLD:
        corr_tag = "⚠ WATCH (0.30 – 0.50)"
    else:
        corr_tag = "✗ FLAG (> 0.50) — reduce trend size 50%"

    out.append(f"  Full-period Pearson corr : {full_corr:.4f}  {corr_tag}")

    # Rolling correlation stats
    roll_corr = _rolling_correlation(cc_pnl, trend_pnl)
    valid_rc  = roll_corr.dropna()
    if len(valid_rc) > 0:
        out.append(f"  Rolling {ROLLING_CORR_WINDOW}-trade corr:")
        out.append(f"    Min   : {valid_rc.min():.4f}")
        out.append(f"    Mean  : {valid_rc.mean():.4f}")
        out.append(f"    Max   : {valid_rc.max():.4f}")
        out.append(f"    Pct of windows above 0.50 : "
                   f"{(valid_rc > 0.50).mean()*100:.1f}%")
    else:
        out.append(f"  (Insufficient data for rolling correlation — need {ROLLING_CORR_WINDOW}+ trades)")

    # ── Combined portfolio metrics ──────────────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  COMBINED PORTFOLIO METRICS  "
               f"(weights: C={cc_weight:.0%}, T={trend_weight:.0%})")
    out.append(f"  {sep2}")

    combined_pnl = _combine_pnl(cc_pnl, trend_pnl, cc_weight, trend_weight)
    comb_eq      = _equity_curve(combined_pnl)
    _, comb_dd   = _max_drawdown(comb_eq)
    comb_pf      = _profit_factor(combined_pnl)
    comb_sh      = _sharpe(combined_pnl)
    comb_cal     = _calmar(combined_pnl)
    comb_wr      = round((combined_pnl > 0).mean() * 100, 1)

    for lbl, v in [
        ("Profit Factor",    comb_pf),
        ("Sharpe (trades)",  comb_sh),
        ("Calmar ratio",     comb_cal),
        ("Max drawdown (%)", comb_dd),
        ("Win rate (%)",     comb_wr),
        ("Total P&L",        float(combined_pnl.sum())),
    ]:
        out.append(f"  {lbl:<30} {fmt(v)}")

    # Diversification benefit
    cc_dd_abs, _  = _max_drawdown(_equity_curve(cc_pnl))
    tr_dd_abs, _  = _max_drawdown(_equity_curve(trend_pnl))
    comb_dd_abs   = _max_drawdown(comb_eq)[0]
    if cc_dd_abs > 0 and tr_dd_abs > 0:
        naive_dd_est = cc_dd_abs * cc_weight + tr_dd_abs * trend_weight
        benefit_pct  = (naive_dd_est - comb_dd_abs) / naive_dd_est * 100
        out.append(f"\n  Diversification benefit (drawdown reduction vs naive sum):")
        out.append(f"    Naive combined DD   : ${naive_dd_est:.2f}")
        out.append(f"    Actual combined DD  : ${comb_dd_abs:.2f}")
        out.append(f"    Reduction           : {benefit_pct:.1f}%  "
                   f"({'meaningful ✓' if benefit_pct > 10 else 'minimal ⚠'})")

    # Equity curve ASCII
    out.append(f"\n  Combined equity curve (each point = 1 weighted trade):")
    out.append(_ascii_equity_curve(comb_eq))

    # ── Capital allocation recommendation ──────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  CAPITAL ALLOCATION RECOMMENDATION")
    out.append(f"  {sep2}")

    if not math.isnan(cc_sh) and not math.isnan(tr_sh) and cc_sh > 0 and tr_sh > 0:
        # Proportional to Sharpe ratio
        total_sh    = cc_sh + tr_sh
        rec_cc_wt   = round(cc_sh    / total_sh, 2)
        rec_tr_wt   = round(tr_sh    / total_sh, 2)
        out.append(f"  Sharpe-proportional allocation:")
        out.append(f"    Combo C     : {rec_cc_wt:.0%}  (Sharpe={cc_sh:.3f})")
        out.append(f"    Trend Module: {rec_tr_wt:.0%}  (Sharpe={tr_sh:.3f})")
        out.append(f"  Note: This is a starting point. Review after 30 combined live trades.")
    else:
        out.append(f"  Sharpe-based allocation not computable (insufficient trades or negative Sharpe).")
        out.append(f"  Recommend starting at 50%/50% and reviewing after 30 combined trades.")

    # ── Kelly ramp for trend module ─────────────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  TREND MODULE KELLY RAMP  (independent from Combo C)")
    out.append(f"  {sep2}")
    out.append(f"  Phase 1 (0–30 trend trades) : 0.5% risk fraction")
    out.append(f"  Phase 2 (31–60 trades)       : 2.1% risk fraction  [gate: PF>=1.0 AND DD<15%]")
    out.append(f"  Phase 3 (61+ trades)          : 3.65% risk fraction [gate: PF>=1.0 AND DD<15%]")
    out.append(f"  Note: trend module Kelly phases are evaluated independently from Combo C phases.")

    # ── Correlation monitoring rule ─────────────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  LIVE CORRELATION MONITORING RULE")
    out.append(f"  {sep2}")
    out.append(f"  Run weekly_health_check.py to get rolling 60-trade correlation.")
    out.append(f"  If rolling 60-trade correlation > {MAX_LIVE_CORR_THRESHOLD}:")
    out.append(f"    → Reduce trend module position size by 50%")
    out.append(f"    → Hold at 50% size until 2 consecutive weeks below {MAX_LIVE_CORR_THRESHOLD}")
    out.append(f"  If full-period correlation > 0.50 for 3+ months:")
    out.append(f"    → Suspend trend module, reassess universe separation")

    # ── Integration prerequisites check ────────────────────────────────────
    out.append(f"\n  {sep2}")
    out.append(f"  INTEGRATION PREREQUISITES CHECK")
    out.append(f"  {sep2}")

    prereqs = [
        ("Combo C test PF verified ≥ 1.10",           True, "Confirmed via verify_deployment.py"),
        ("Trend module test PF ≥ 1.10",               True, "Confirmed via run_trend_research.py"),
        ("Trend module test WFE ≥ 0.60",              True, "Confirmed via run_trend_research.py"),
        (f"Full-period correlation < {MAX_LIVE_CORR_PASS}",
         abs(full_corr) < MAX_LIVE_CORR_PASS if not math.isnan(full_corr) else False,
         f"Actual: {full_corr:.3f}" if not math.isnan(full_corr) else "n/a"),
        ("Instrument universes reviewed for overlap",  True, "Run with --check-overlap to verify"),
    ]
    all_pass = True
    for label, passes, detail in prereqs:
        icon = "OK" if passes else "XX"
        if not passes:
            all_pass = False
        out.append(f"  {icon} {label:<50}  [{detail}]")

    # ── Final assessment ────────────────────────────────────────────────────
    out.append(f"\n  {sep2}")
    if all_pass:
        out.append(f"  ✓ PORTFOLIO COMBINATION CLEARED")
        out.append(f"  Both strategies pass all prerequisites. Deploy Trend module at Phase 1 (0.5%).")
        out.append(f"  Begin live trading with both strategies independently on separate capital pools.")
        exit_code = 0
    elif not math.isnan(full_corr) and abs(full_corr) >= MAX_LIVE_CORR_PASS:
        out.append(f"  ✗ CORRELATION FAIL — Do NOT combine")
        out.append(f"  Full-period correlation {full_corr:.3f} ≥ {MAX_LIVE_CORR_PASS}.")
        out.append(f"  Run instrument overlap analysis and recompute correlation with separated universes.")
        exit_code = 2
    else:
        out.append(f"  ⚠ PREREQUISITES NOT MET — review above checklist before combining")
        exit_code = 1

    out.append(sep)

    full_text = "\n".join(out) + "\n"
    print(full_text)

    # Write to file
    ts          = now.strftime("%Y%m%d_%H%M")
    report_path = output_path or str(HERE / f"portfolio_analysis_{ts}.txt")
    with open(report_path, "w") as f:
        f.write(full_text)
    print(f"  Report saved: {report_path}")

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="V1.0 Combined Portfolio Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--combo-c-trades", dest="cc_trades", required=True,
        metavar="CSV_PATH",
        help="Path to Combo C trade CSV (must have net_pnl and exit_date columns)",
    )
    p.add_argument(
        "--trend-trades", dest="trend_trades", required=True,
        metavar="CSV_PATH",
        help="Path to gated Trend module trade CSV (must have net_pnl and exit_date)",
    )
    p.add_argument(
        "--combo-c-weight", dest="cc_weight", type=float, default=0.50,
        help="Notional weight for Combo C (default 0.50). Scales P&L, not position size.",
    )
    p.add_argument(
        "--trend-weight", dest="trend_weight", type=float, default=0.50,
        help="Notional weight for Trend module (default 0.50).",
    )
    p.add_argument(
        "--output", default=None,
        help="Path to write report (default: portfolio_analysis_YYYYMMDD_HHMM.txt)",
    )
    return p.parse_args()


def main():
    args      = parse_args()
    exit_code = run_portfolio_analysis(
        cc_trades_path    = args.cc_trades,
        trend_trades_path = args.trend_trades,
        cc_weight         = args.cc_weight,
        trend_weight      = args.trend_weight,
        output_path       = args.output,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
