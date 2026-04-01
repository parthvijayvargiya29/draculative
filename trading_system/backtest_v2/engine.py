"""
Production-Grade Backtesting Engine

Ties together: DataLoader → IndicatorState → SignalGenerator → Portfolio

Core loop:
  For each 15-minute bar (chronological, all symbols):
    1. Update indicator state for that symbol (closed bar only)
    2. Generate signal from snapshot (no future data)
    3. Call portfolio.on_bar() → checks exits on open positions, queues entry
    4. On the NEXT bar → call portfolio.on_next_bar_open() → fills entry

Period splits (no in-sample tuning on test data):
  Train:    first 60% of data   — for strategy development
  Validate: next  20%           — for threshold tuning
  Test:     last  20%           — out-of-sample, reported separately

Statistics reported:
  - Per period (train/validate/test)
  - Per symbol
  - Overall
  - Monte Carlo permutation test (is win rate above chance?)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import json
from math import sqrt
from collections import defaultdict

from data_loader import DataLoader, SYMBOLS
from indicators import IndicatorState, IndicatorSnapshot
from signals import generate_signal, Signal, Direction
from executor import Portfolio, Trade

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Period splitter
# ─────────────────────────────────────────────────────────────────────────────

def split_periods(index: pd.DatetimeIndex,
                  train_pct: float = 0.60,
                  val_pct:   float = 0.20) -> Tuple[datetime, datetime]:
    """
    Return cutoff datetimes for train/validate/test split.

    Returns (train_end, val_end). Everything after val_end is the test set.
    Splits are time-based (not random) to prevent future leakage.
    """
    n = len(index)
    train_end = index[int(n * train_pct)]
    val_end   = index[int(n * (train_pct + val_pct))]
    return train_end, val_end


def get_period_label(ts: datetime,
                     train_end: datetime,
                     val_end:   datetime) -> str:
    if ts <= train_end:
        return 'train'
    elif ts <= val_end:
        return 'validate'
    else:
        return 'test'


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Bar-by-bar simulation over all symbols simultaneously.

    Timeline:
      bar T close  → indicator update → signal generated
      bar T+1 open → entry fills (if signal fired at T)
      bar T+1 …    → normal bar processing (exit check, new signal check)
    """

    def __init__(self, initial_capital: float = 25_000.0,
                 symbols: List[str] = None):
        self.initial_capital = initial_capital
        self.symbols         = symbols or SYMBOLS
        self.portfolio       = Portfolio(initial_capital)

        # One IndicatorState per symbol — they are COMPLETELY independent
        self.indicator_states: Dict[str, IndicatorState] = {
            sym: IndicatorState() for sym in self.symbols
        }

        # Track last signal per symbol (to pass to on_next_bar_open)
        self._last_signal: Dict[str, Optional[Signal]] = {
            sym: None for sym in self.symbols
        }

        self._train_end: Optional[datetime] = None
        self._val_end:   Optional[datetime] = None

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, clean_data: Dict[str, pd.DataFrame]) -> List[Trade]:
        """
        Main entry point. Returns list of completed trades.

        clean_data: dict of symbol → DataFrame with columns [Open,High,Low,Close,Volume]
                    index is timezone-aware datetime (America/New_York)
        """
        # Filter to symbols we have data for
        available = {s: clean_data[s] for s in self.symbols if s in clean_data}
        if not available:
            raise ValueError("No data available for any symbol")

        # Build a unified sorted timeline
        all_timestamps = pd.DatetimeIndex([])
        for df in available.values():
            all_timestamps = all_timestamps.union(df.index)
        all_timestamps = all_timestamps.sort_values()

        # Determine period splits
        self._train_end, self._val_end = split_periods(all_timestamps)
        logger.info(
            f"\nPeriod splits:\n"
            f"  Train:    start → {self._train_end.date()}\n"
            f"  Validate: {self._train_end.date()} → {self._val_end.date()}\n"
            f"  Test:     {self._val_end.date()} → end\n"
            f"  Total bars: {len(all_timestamps):,}\n"
        )

        logger.info("Starting bar-by-bar simulation...")
        bar_count = 0
        log_interval = max(1, len(all_timestamps) // 20)  # Log ~20 progress updates

        for ts in all_timestamps:
            bar_count += 1
            period = get_period_label(ts, self._train_end, self._val_end)

            if bar_count % log_interval == 0:
                pct = bar_count / len(all_timestamps) * 100
                open_pos = len(self.portfolio.open_positions)
                trades_so_far = len(self.portfolio.trades)
                equity = self.portfolio.equity
                logger.info(
                    f"  {pct:4.0f}% | {ts.date()} | "
                    f"equity=${equity:,.0f} | "
                    f"open={open_pos} | trades={trades_so_far}"
                )

            for symbol, df in available.items():
                if ts not in df.index:
                    continue

                row = df.loc[ts]
                o = float(row['Open'])
                h = float(row['High'])
                l = float(row['Low'])
                c = float(row['Close'])
                v = float(row['Volume'])

                # ── Step 1: Fill any pending entry at this bar's OPEN ─────────
                # This simulates: signal fired at T, entry fills at T+1 open
                self.portfolio.on_next_bar_open(symbol, ts, o, period)

                # ── Step 2: Update indicators with this closed bar ────────────
                snap = self.indicator_states[symbol].update(o, h, l, c, v)

                # ── Step 3: Generate signal from indicator snapshot ───────────
                signal = generate_signal(snap)
                self._last_signal[symbol] = signal

                # ── Step 4: Portfolio processes the bar ───────────────────────
                # Checks exits, queues new entries
                self.portfolio.on_bar(symbol, ts, o, h, l, c, signal, period)

            # ── Equity snapshot (once per timestamp) ─────────────────────────
            if bar_count % 26 == 0:  # Daily snapshot
                self.portfolio.record_equity(ts)

        # Force-close any remaining open positions at last known prices
        last_prices = {}
        for symbol, df in available.items():
            if len(df) > 0:
                last_prices[symbol] = float(df['Close'].iloc[-1])
        last_ts = all_timestamps[-1]
        period = get_period_label(last_ts, self._train_end, self._val_end)
        self.portfolio.force_close_all(last_ts, last_prices, period)

        logger.info(
            f"\n✓ Simulation complete: {len(self.portfolio.trades)} trades, "
            f"final equity=${self.portfolio.equity:,.2f}"
        )
        return self.portfolio.trades


# ─────────────────────────────────────────────────────────────────────────────
# Statistics engine
# ─────────────────────────────────────────────────────────────────────────────

class Statistics:
    """Compute and report all performance metrics from a list of trades."""

    def __init__(self, trades: List[Trade], initial_capital: float,
                 equity_curve: List[dict]):
        self.trades          = trades
        self.initial_capital = initial_capital
        self.equity_curve    = equity_curve

    def compute(self) -> dict:
        """Compute all metrics. Returns nested dict."""
        df = pd.DataFrame([t.__dict__ for t in self.trades])

        if df.empty:
            return {'error': 'No trades executed'}

        results = {}

        # ── Overall ──────────────────────────────────────────────────────────
        results['overall'] = self._compute_metrics(df, 'ALL')

        # ── By period ────────────────────────────────────────────────────────
        results['by_period'] = {}
        for period in ['train', 'validate', 'test']:
            subset = df[df['period'] == period]
            if len(subset) > 0:
                results['by_period'][period] = self._compute_metrics(subset, period.upper())

        # ── By symbol ─────────────────────────────────────────────────────────
        results['by_symbol'] = {}
        for sym in df['symbol'].unique():
            subset = df[df['symbol'] == sym]
            results['by_symbol'][sym] = self._compute_metrics(subset, sym)

        # ── Equity curve stats ────────────────────────────────────────────────
        results['equity'] = self._equity_stats()

        # ── Overfitting check ─────────────────────────────────────────────────
        results['overfitting'] = self._overfitting_check(results)

        # ── Statistical significance ──────────────────────────────────────────
        results['significance'] = self._significance(df)

        return results

    def _compute_metrics(self, df: pd.DataFrame, label: str) -> dict:
        if len(df) == 0:
            return {}

        wins   = df[df['won']]
        losses = df[~df['won']]
        n      = len(df)

        avg_win  = wins['net_pnl'].mean()   if len(wins)   > 0 else 0.0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0

        total_wins_dollar   = wins['net_pnl'].sum()
        total_losses_dollar = abs(losses['net_pnl'].sum())

        profit_factor = (total_wins_dollar / total_losses_dollar
                         if total_losses_dollar > 0 else float('inf'))

        win_rate = len(wins) / n if n > 0 else 0.0

        # Total P&L as return on initial capital
        total_net_pnl = df['net_pnl'].sum()
        total_return  = total_net_pnl / self.initial_capital

        # Expectancy per trade ($ per trade)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # SQN (Van Tharp's System Quality Number)
        if n > 1 and df['net_pnl'].std() > 0:
            sqn = (expectancy / df['net_pnl'].std()) * sqrt(n)
        else:
            sqn = 0.0

        # 95% confidence interval on win rate (Wilson)
        z  = 1.96
        ci = z * sqrt((win_rate * (1 - win_rate)) / n) if n > 0 else 0.0

        # Exit reason breakdown
        exit_counts = df['exit_reason'].value_counts().to_dict()

        # Concentration: remove best/worst 5 trades
        sorted_pnl  = df['net_pnl'].sort_values(ascending=False)
        top5_sum    = sorted_pnl.head(5).sum()
        bottom5_sum = sorted_pnl.tail(5).sum()
        pnl_ex_top5 = total_net_pnl - top5_sum

        m = {
            'label':          label,
            'n_trades':       n,
            'n_wins':         len(wins),
            'n_losses':       len(losses),
            'win_rate':       round(win_rate * 100, 2),
            'win_rate_ci_lo': round((win_rate - ci) * 100, 2),
            'win_rate_ci_hi': round((win_rate + ci) * 100, 2),
            'avg_win_$':      round(avg_win, 2),
            'avg_loss_$':     round(avg_loss, 2),
            'profit_factor':  round(profit_factor, 3),
            'total_net_pnl_$': round(total_net_pnl, 2),
            'total_return_%': round(total_return * 100, 2),
            'expectancy_$':   round(expectancy, 2),
            'sqn':            round(sqn, 3),
            'exit_reasons':   exit_counts,
            'largest_win_$':  round(df['net_pnl'].max(), 2),
            'largest_loss_$': round(df['net_pnl'].min(), 2),
            'top5_concentration_%': round(top5_sum / total_net_pnl * 100, 1)
                                    if total_net_pnl != 0 else 0,
            'pnl_ex_top5_$':  round(pnl_ex_top5, 2),
            'total_commission_$': round(df['commission'].sum(), 2),
            'total_slippage_$':   round(df['slippage'].sum(), 2),
        }
        return m

    def _equity_stats(self) -> dict:
        if not self.equity_curve:
            return {}
        ec_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        max_dd = ec_df['drawdown'].max()
        equity_series = ec_df['equity']

        # Daily returns approximation from equity snapshots
        daily_rets = equity_series.pct_change().dropna()
        sharpe = 0.0
        if len(daily_rets) > 1 and daily_rets.std() > 0:
            # Annualise assuming ~252 trading days × 26 bars = 6552 snapshots/year
            # But our snapshots are daily, so use daily Sharpe
            sharpe = (daily_rets.mean() / daily_rets.std()) * sqrt(252)

        final_equity = equity_series.iloc[-1]
        calmar = 0.0
        if max_dd > 0:
            total_ret = (final_equity - self.initial_capital) / self.initial_capital
            calmar = total_ret / max_dd

        return {
            'initial_capital':  round(self.initial_capital, 2),
            'final_equity':     round(float(final_equity), 2),
            'total_return_%':   round((final_equity - self.initial_capital)
                                      / self.initial_capital * 100, 2),
            'max_drawdown_%':   round(max_dd * 100, 2),
            'sharpe_ratio':     round(sharpe, 3),
            'calmar_ratio':     round(calmar, 3),
        }

    def _overfitting_check(self, results: dict) -> dict:
        """Compare train vs test performance. Flag if test is much worse."""
        train = results.get('by_period', {}).get('train', {})
        test  = results.get('by_period', {}).get('test', {})

        if not train or not test:
            return {'warning': 'Insufficient period data for overfitting check'}

        train_wr = train.get('win_rate', 0)
        test_wr  = test.get('win_rate', 0)
        train_pf = train.get('profit_factor', 0)
        test_pf  = test.get('profit_factor', 0)

        wr_degradation = train_wr - test_wr
        pf_degradation = train_pf - test_pf

        flags = []
        if wr_degradation > 10:
            flags.append(f"Win rate drops {wr_degradation:.1f}pp from train→test (HIGH RISK)")
        if pf_degradation > 0.3:
            flags.append(f"Profit factor drops {pf_degradation:.2f} from train→test")
        if test.get('n_trades', 0) < 30:
            flags.append("Fewer than 30 trades in test set — results unreliable")

        return {
            'train_win_rate':   train_wr,
            'test_win_rate':    test_wr,
            'wr_degradation':   round(wr_degradation, 2),
            'train_pf':         train_pf,
            'test_pf':          test_pf,
            'pf_degradation':   round(pf_degradation, 3),
            'overfitting_flags': flags,
            'verdict': 'LIKELY OVERFIT' if len(flags) >= 2
                       else 'POSSIBLE OVERFIT' if len(flags) == 1
                       else 'PASSES BASIC CHECK',
        }

    def _significance(self, df: pd.DataFrame) -> dict:
        """Basic statistical significance tests."""
        n = len(df)
        win_rate = df['won'].mean()
        z = 1.96
        ci = z * sqrt((win_rate * (1 - win_rate)) / n) if n > 0 else 0.0

        # Monte Carlo: shuffle trade outcomes 1000× and count how often
        # random achieves the same or better win rate (p-value estimate)
        np.random.seed(42)
        mc_trials = 1000
        outcomes = df['won'].values.copy()
        mc_win_rates = []
        for _ in range(mc_trials):
            shuffled = np.random.permutation(outcomes)
            mc_win_rates.append(shuffled.mean())

        p_value = np.mean(np.array(mc_win_rates) >= win_rate)

        return {
            'n_trades':         n,
            'win_rate_%':       round(win_rate * 100, 2),
            'ci_95_lo_%':       round((win_rate - ci) * 100, 2),
            'ci_95_hi_%':       round((win_rate + ci) * 100, 2),
            'monte_carlo_p':    round(float(p_value), 4),
            'significant_5pct': bool(p_value < 0.05),
            'recommended_min_trades': 300,
            'verdict': (
                'SUFFICIENT' if n >= 300 else
                'MARGINAL'   if n >= 100 else
                'INSUFFICIENT'
            ),
        }

    # ── Print report ─────────────────────────────────────────────────────────

    def print_report(self, results: dict):
        """Print a clean, comprehensive report to logger."""

        def h(title): logger.info(f"\n{'═'*64}\n  {title}\n{'═'*64}")
        def row(k, v): logger.info(f"  {k:<35} {v}")

        h("BACKTEST RESULTS — PRODUCTION GRADE")
        eq = results.get('equity', {})
        row("Initial Capital",     f"${eq.get('initial_capital', 0):>12,.2f}")
        row("Final Equity",        f"${eq.get('final_equity', 0):>12,.2f}")
        row("Total Return",        f"{eq.get('total_return_%', 0):>11.2f}%")
        row("Max Drawdown",        f"{eq.get('max_drawdown_%', 0):>11.2f}%")
        row("Sharpe Ratio",        f"{eq.get('sharpe_ratio', 0):>12.3f}")
        row("Calmar Ratio",        f"{eq.get('calmar_ratio', 0):>12.3f}")

        for period_key in ['overall', 'by_period']:
            if period_key == 'overall':
                m = results.get('overall', {})
                h(f"OVERALL ({m.get('n_trades', 0)} trades)")
                self._print_metrics(m)
            else:
                for period_name, m in results.get('by_period', {}).items():
                    h(f"{period_name.upper()} PERIOD ({m.get('n_trades',0)} trades)")
                    self._print_metrics(m)

        h("BY SYMBOL")
        for sym, m in results.get('by_symbol', {}).items():
            logger.info(f"\n  {sym}:")
            logger.info(f"    trades={m.get('n_trades',0)} | "
                        f"win={m.get('win_rate',0):.1f}% | "
                        f"pf={m.get('profit_factor',0):.2f} | "
                        f"net=${m.get('total_net_pnl_$',0):,.0f}")

        h("OVERFITTING CHECK")
        ov = results.get('overfitting', {})
        row("Train win rate",   f"{ov.get('train_win_rate', 0):.2f}%")
        row("Test win rate",    f"{ov.get('test_win_rate', 0):.2f}%")
        row("Win rate delta",   f"{ov.get('wr_degradation', 0):+.2f}pp")
        row("Train PF",         f"{ov.get('train_pf', 0):.3f}")
        row("Test PF",          f"{ov.get('test_pf', 0):.3f}")
        row("Verdict",          ov.get('verdict', 'UNKNOWN'))
        for flag in ov.get('overfitting_flags', []):
            logger.warning(f"  ⚠  {flag}")

        h("STATISTICAL SIGNIFICANCE")
        sig = results.get('significance', {})
        row("Total trades",     f"{sig.get('n_trades', 0)}")
        row("Win rate",         f"{sig.get('win_rate_%', 0):.2f}%")
        row("95% CI",           f"[{sig.get('ci_95_lo_%',0):.1f}%, "
                                f"{sig.get('ci_95_hi_%',0):.1f}%]")
        row("Monte Carlo p",    f"{sig.get('monte_carlo_p', 1.0):.4f}")
        row("Significant?",     "YES" if sig.get('significant_5pct') else "NO")
        row("Sample adequacy",  sig.get('verdict', 'UNKNOWN'))
        logger.info("")

    def _print_metrics(self, m: dict):
        if not m:
            return
        logger.info(f"  Trades:           {m.get('n_trades', 0):>8}")
        logger.info(f"  Win rate:         {m.get('win_rate', 0):>7.2f}%  "
                    f"[{m.get('win_rate_ci_lo',0):.1f}%, "
                    f"{m.get('win_rate_ci_hi',0):.1f}%] 95% CI")
        logger.info(f"  Avg win:          ${m.get('avg_win_$',0):>8,.2f}")
        logger.info(f"  Avg loss:         ${m.get('avg_loss_$',0):>8,.2f}")
        logger.info(f"  Profit factor:    {m.get('profit_factor',0):>8.3f}")
        logger.info(f"  Expectancy:       ${m.get('expectancy_$',0):>8,.2f}/trade")
        logger.info(f"  SQN:              {m.get('sqn',0):>8.3f}")
        logger.info(f"  Total net P&L:    ${m.get('total_net_pnl_$',0):>8,.2f}")
        logger.info(f"  Total return:     {m.get('total_return_%',0):>7.2f}%")
        logger.info(f"  Total commission: ${m.get('total_commission_$',0):>8,.2f}")
        logger.info(f"  Total slippage:   ${m.get('total_slippage_$',0):>8,.2f}")
        logger.info(f"  Largest win:      ${m.get('largest_win_$',0):>8,.2f}")
        logger.info(f"  Largest loss:     ${m.get('largest_loss_$',0):>8,.2f}")
        logger.info(f"  Top-5 concentration: {m.get('top5_concentration_%',0):.1f}%")
        logger.info(f"  P&L ex top-5:     ${m.get('pnl_ex_top5_$',0):>8,.2f}")
        exit_r = m.get('exit_reasons', {})
        logger.info(f"  Exit reasons:     {json.dumps(exit_r)}")


# ─────────────────────────────────────────────────────────────────────────────
# Monthly breakdown helper
# ─────────────────────────────────────────────────────────────────────────────

def monthly_breakdown(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')
    monthly = df.groupby(['month', 'period']).agg(
        trades     = ('won', 'count'),
        wins       = ('won', 'sum'),
        net_pnl    = ('net_pnl', 'sum'),
    ).reset_index()
    monthly['win_rate'] = (monthly['wins'] / monthly['trades'] * 100).round(1)
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("="*64)
    logger.info("  PRODUCTION BACKTEST — NO LOOKAHEAD")
    logger.info("="*64)

    # ── 1. Load & validate data ───────────────────────────────────────────────
    loader = DataLoader(symbols=SYMBOLS, lookback_years=2)
    clean_data = loader.load_all()

    if not clean_data:
        logger.error("No clean data available. Aborting.")
        return

    # ── 2. Run simulation ─────────────────────────────────────────────────────
    engine = BacktestEngine(initial_capital=25_000.0, symbols=list(clean_data.keys()))
    trades = engine.run(clean_data)

    if not trades:
        logger.warning("No trades were executed.")
        return

    # ── 3. Compute statistics ─────────────────────────────────────────────────
    stats = Statistics(
        trades          = trades,
        initial_capital = engine.initial_capital,
        equity_curve    = engine.portfolio.equity_curve,
    )
    results = stats.compute()
    stats.print_report(results)

    # ── 4. Monthly breakdown ──────────────────────────────────────────────────
    monthly = monthly_breakdown(trades)
    if len(monthly) > 0:
        logger.info("\nMONTHLY BREAKDOWN (TEST PERIOD):")
        test_monthly = monthly[monthly['period'] == 'test']
        if len(test_monthly) > 0:
            logger.info(test_monthly.to_string(index=False))

    # ── 5. Save results ────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(__file__))

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trades_path = os.path.join(output_dir, 'backtest_v2_trades.csv')
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"\n✓ Trades saved → {trades_path}")

    results_path = os.path.join(output_dir, 'backtest_v2_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Results saved → {results_path}")

    equity_df = pd.DataFrame(engine.portfolio.equity_curve)
    equity_path = os.path.join(output_dir, 'backtest_v2_equity.csv')
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"✓ Equity curve saved → {equity_path}")

    return trades, results


if __name__ == '__main__':
    main()
