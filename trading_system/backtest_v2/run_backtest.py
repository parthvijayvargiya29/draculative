#!/usr/bin/env python3
"""
run_backtest.py — Production Backtest CLI Entry Point
═══════════════════════════════════════════════════════
Usage:
    python3 run_backtest.py                      # full 2-year run
    python3 run_backtest.py --capital 50000      # custom starting capital
    python3 run_backtest.py --symbols NVDA AAPL  # subset of symbols

What this script does:
  1. Downloads and validates 2 years of 15-min OHLCV data
  2. Runs bar-by-bar simulation (NO lookahead by design)
  3. Computes full statistics (Sharpe, drawdown, win rate CI, etc.)
  4. Runs walk-forward validation (first 60% in-sample, last 40% out-of-sample)
  5. Saves CSV + JSON results to this directory

All output files are prefixed bt2_ so they never conflict with v1.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from data_loader import DataLoader, SYMBOLS
from engine import BacktestEngine, Statistics, monthly_breakdown

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = HERE   # save files next to the scripts


# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Production-grade backtester v2")
    p.add_argument('--capital',  type=float, default=25_000,
                   help='Starting capital in USD (default: 25000)')
    p.add_argument('--symbols',  nargs='+',  default=SYMBOLS,
                   help=f'Symbols to trade (default: {SYMBOLS})')
    p.add_argument('--years',    type=float, default=2.0,
                   help='Lookback years (default: 2.0)')
    p.add_argument('--no-cache', action='store_true',
                   help='Force re-download even if cached data exists')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(symbols, years, force_download=False):
    logger.info("━"*64)
    logger.info("  STEP 1 / 5 — Loading & validating market data")
    logger.info("━"*64)

    loader = DataLoader(symbols=symbols, lookback_years=years)
    if force_download:
        import shutil
        cache_dir = Path.home() / '.backtest_cache'
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("  Cleared cache — forcing fresh download")

    clean_data = loader.load_all()

    if not clean_data:
        logger.error("  No data loaded. Check internet connection or symbol names.")
        sys.exit(1)

    logger.info(f"\n  ✓ Loaded {len(clean_data)} symbols:")
    for sym, df in clean_data.items():
        start = df.index.min().date()
        end   = df.index.max().date()
        bars  = len(df)
        logger.info(f"    {sym:6s}  {start} → {end}  ({bars:,} bars)")

    return clean_data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Full simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(clean_data, capital, symbols):
    logger.info("\n" + "━"*64)
    logger.info("  STEP 2 / 5 — Running bar-by-bar simulation")
    logger.info("━"*64)
    logger.info("  Every bar = a live trading moment. No lookahead.\n")

    engine = BacktestEngine(initial_capital=capital, symbols=symbols)
    trades = engine.run(clean_data)

    logger.info(f"\n  ✓ Simulation complete")
    logger.info(f"    Total trades:   {len(trades)}")
    logger.info(f"    Final equity:   ${engine.portfolio.equity:,.2f}")

    return engine, trades


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(engine, trades):
    logger.info("\n" + "━"*64)
    logger.info("  STEP 3 / 5 — Computing statistics")
    logger.info("━"*64)

    stats = Statistics(
        trades          = trades,
        initial_capital = engine.initial_capital,
        equity_curve    = engine.portfolio.equity_curve,
    )
    results = stats.compute()
    stats.print_report(results)
    return stats, results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Monthly breakdown
# ─────────────────────────────────────────────────────────────────────────────

def print_monthly(trades):
    logger.info("\n" + "━"*64)
    logger.info("  STEP 4 / 5 — Monthly breakdown")
    logger.info("━"*64)

    monthly = monthly_breakdown(trades)
    if monthly.empty:
        logger.warning("  No monthly data available.")
        return monthly

    # Show test period only (unbiased view)
    test_m = monthly[monthly['period'] == 'test']
    if not test_m.empty:
        logger.info("\n  TEST PERIOD — Monthly results:")
        logger.info(test_m.to_string(index=False))
    else:
        # Fall back to all periods
        logger.info("\n  ALL PERIODS — Monthly results:")
        logger.info(monthly.to_string(index=False))

    # Consistency metrics
    df = pd.DataFrame([t.__dict__ for t in trades])
    df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')
    monthly_pnl = df.groupby('month')['net_pnl'].sum()
    positive_months = (monthly_pnl > 0).sum()
    total_months    = len(monthly_pnl)

    logger.info(f"\n  Profitable months: {positive_months}/{total_months} "
                f"({positive_months/total_months*100:.0f}%)")

    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(engine, trades, results, monthly):
    logger.info("\n" + "━"*64)
    logger.info("  STEP 5 / 5 — Saving results")
    logger.info("━"*64)

    ts = datetime.now().strftime('%Y%m%d_%H%M')

    # ── Trades CSV ──────────────────────────────────────────────────────────
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    # Convert Direction objects to strings for serialisation
    for col in trades_df.columns:
        trades_df[col] = trades_df[col].apply(
            lambda x: str(x) if hasattr(x, '__class__') and
                      x.__class__.__name__ == 'Direction' else x
        )
    trades_path = OUTPUT_DIR / f'bt2_trades_{ts}.csv'
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"  ✓ Trades      → {trades_path}")

    # ── Equity curve CSV ────────────────────────────────────────────────────
    equity_df   = pd.DataFrame(engine.portfolio.equity_curve)
    equity_path = OUTPUT_DIR / f'bt2_equity_{ts}.csv'
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"  ✓ Equity curve → {equity_path}")

    # ── Monthly CSV ─────────────────────────────────────────────────────────
    if not monthly.empty:
        monthly_path = OUTPUT_DIR / f'bt2_monthly_{ts}.csv'
        monthly.to_csv(monthly_path, index=False)
        logger.info(f"  ✓ Monthly      → {monthly_path}")

    # ── Full results JSON ───────────────────────────────────────────────────
    results_path = OUTPUT_DIR / f'bt2_results_{ts}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  ✓ Results JSON → {results_path}")

    # ── Summary TXT (human-readable quick ref) ──────────────────────────────
    summary_path = OUTPUT_DIR / f'bt2_summary_{ts}.txt'
    with open(summary_path, 'w') as f:
        ov   = results.get('overall', {})
        eq   = results.get('equity', {})
        sig  = results.get('significance', {})
        ofs  = results.get('overfitting', {})

        f.write("PRODUCTION BACKTEST V2 — QUICK SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("EQUITY\n")
        f.write(f"  Initial:       ${eq.get('initial_capital',0):>12,.2f}\n")
        f.write(f"  Final:         ${eq.get('final_equity',0):>12,.2f}\n")
        f.write(f"  Return:        {eq.get('total_return_%',0):>10.2f}%\n")
        f.write(f"  Max Drawdown:  {eq.get('max_drawdown_%',0):>10.2f}%\n")
        f.write(f"  Sharpe Ratio:  {eq.get('sharpe_ratio',0):>10.3f}\n")
        f.write(f"  Calmar Ratio:  {eq.get('calmar_ratio',0):>10.3f}\n\n")

        f.write("OVERALL\n")
        f.write(f"  Trades:        {ov.get('n_trades',0)}\n")
        f.write(f"  Win Rate:      {ov.get('win_rate',0):.2f}%"
                f"  [{ov.get('win_rate_ci_lo',0):.1f}%, {ov.get('win_rate_ci_hi',0):.1f}%]\n")
        f.write(f"  Profit Factor: {ov.get('profit_factor',0):.3f}\n")
        f.write(f"  Expectancy:    ${ov.get('expectancy_$',0):,.2f}/trade\n")
        f.write(f"  SQN:           {ov.get('sqn',0):.3f}\n\n")

        f.write("VALIDITY\n")
        f.write(f"  Sample size:   {sig.get('verdict','?')}\n")
        f.write(f"  Monte Carlo p: {sig.get('monte_carlo_p',1.0):.4f}\n")
        f.write(f"  Significant:   {'YES' if sig.get('significant_5pct') else 'NO'}\n")
        f.write(f"  Overfitting:   {ofs.get('verdict','?')}\n")

    logger.info(f"  ✓ Summary      → {summary_path}")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward split helper (diagnostic only — simulation already splits)
# ─────────────────────────────────────────────────────────────────────────────

def print_walk_forward_summary(results):
    """
    Print a side-by-side train / validate / test comparison.
    Flagging when test degrades severely from train (classic overfitting sign).
    """
    logger.info("\n" + "═"*64)
    logger.info("  WALK-FORWARD VALIDATION SUMMARY")
    logger.info("═"*64)

    by_period = results.get('by_period', {})
    cols = ['n_trades', 'win_rate', 'profit_factor', 'expectancy_$', 'sqn',
            'total_return_%']

    header = f"  {'Metric':<25}"
    for period in ['train', 'validate', 'test']:
        m = by_period.get(period, {})
        if m:
            header += f"  {period.upper():>12}"
    logger.info(header)
    logger.info("  " + "─"*62)

    for col in cols:
        row = f"  {col:<25}"
        for period in ['train', 'validate', 'test']:
            m = by_period.get(period, {})
            if not m:
                continue
            val = m.get(col, 0)
            if isinstance(val, float):
                row += f"  {val:>12.2f}"
            else:
                row += f"  {val:>12}"
        logger.info(row)

    logger.info("")
    ov = results.get('overfitting', {})
    verdict = ov.get('verdict', 'UNKNOWN')
    flags   = ov.get('overfitting_flags', [])

    if verdict == 'LIKELY OVERFIT':
        logger.warning(f"  ⚠  VERDICT: {verdict}")
    elif verdict == 'POSSIBLE OVERFIT':
        logger.warning(f"  ⚠  VERDICT: {verdict}")
    else:
        logger.info(f"  ✓  VERDICT: {verdict}")

    for flag in flags:
        logger.warning(f"     • {flag}")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("\n" + "═"*64)
    logger.info("  PRODUCTION BACKTEST V2 — ZERO LOOKAHEAD")
    logger.info(f"  Capital: ${args.capital:,.0f}")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  History: {args.years} years at 15-min granularity")
    logger.info("═"*64 + "\n")

    # 1. Data
    clean_data = load_data(args.symbols, args.years, args.no_cache)

    # 2. Simulation
    engine, trades = run_simulation(clean_data, args.capital, args.symbols)

    if not trades:
        logger.error("\n  ✗ Zero trades executed. Check signal generator thresholds.")
        sys.exit(1)

    # 3. Stats
    _, results = compute_stats(engine, trades)

    # 4. Monthly
    monthly = print_monthly(trades)

    # 5. Walk-forward comparison
    print_walk_forward_summary(results)

    # 6. Save
    save_results(engine, trades, results, monthly)

    # Final verdict
    eq  = results.get('equity', {})
    sig = results.get('significance', {})
    ov  = results.get('overfitting', {})
    n   = len(trades)

    logger.info("═"*64)
    logger.info("  FINAL VERDICT")
    logger.info("═"*64)
    logger.info(f"  Total return:    {eq.get('total_return_%', 0):.2f}%")
    logger.info(f"  Sharpe ratio:    {eq.get('sharpe_ratio', 0):.3f}")
    logger.info(f"  Trade count:     {n}  (need ≥300 for statistical validity)")
    logger.info(f"  Significant:     {'YES ✓' if sig.get('significant_5pct') else 'NO ✗'}")
    logger.info(f"  Overfitting:     {ov.get('verdict', 'UNKNOWN')}")

    # Interpret
    if sig.get('significant_5pct') and ov.get('verdict') == 'PASSES BASIC CHECK':
        logger.info("\n  ✓ Strategy passes initial validation — suitable for paper trading.")
    else:
        logger.warning("\n  ⚠  Strategy requires further investigation before live deployment.")

    logger.info("")


if __name__ == '__main__':
    main()
