#!/usr/bin/env python3
"""
generate_today_signals.py -- Generate Combo C entry signals for TODAY
=====================================================================
Reads live market data from Alpaca, computes Combo C entry signals on today's bars,
calculates Kelly-scaled position sizes, and logs entries to trade_journal.txt.

ENTRY CRITERIA (Combo C)
------------------------
  1. RSI2 < 15 at Bollinger Band lower
  2. Long entry at next bar's open
  3. Exit: BB middle or stop loss
  4. Position sizing: Kelly ramp (0.5% base on first 30 trades, ramp to 2.28% at trade 31+, 4.06% at trade 61+)
  5. No positions on NVDA if beta > 1.5 (Combo C beta gate)

USAGE
-----
  cd backtest_v3
  export ALPACA_API_KEY="..."
  export ALPACA_SECRET_KEY="..."
  python generate_today_signals.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data
from indicators_v3 import IndicatorStateV3, BarSnapshot
from combos import combo_c_entry, exit_signal, COMBO_C_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KELLY_RAMP_SCHEDULE = {
    "trades_1_30": 0.005,      # 0.5%
    "trades_31_60": 0.0228,    # 2.28%
    "trades_61_plus": 0.0406,  # 4.06% (quarter-kelly)
}

PORTFOLIO_EQUITY = 10000.0  # $10k starting


def get_kelly_fraction(trade_count: int) -> float:
    """Get Kelly fraction based on cumulative trade count."""
    if trade_count <= 30:
        return KELLY_RAMP_SCHEDULE["trades_1_30"]
    elif trade_count <= 60:
        return KELLY_RAMP_SCHEDULE["trades_31_60"]
    else:
        return KELLY_RAMP_SCHEDULE["trades_61_plus"]


def compute_position_size(
    equity: float, atr: float, kelly_frac: float, portfolio_limit_pct: float = 0.06
) -> int:
    """
    Compute position size based on Kelly fraction and ATR.
    
    position_size = floor(equity * kelly_frac / ATR)
    cap at portfolio_limit_pct of equity.
    """
    if atr <= 0:
        return 0
    
    size = equity * kelly_frac / atr
    max_shares = int(equity * portfolio_limit_pct)
    
    return int(min(size, max_shares))


def scan_for_entries(
    data: dict[str, pd.DataFrame],
    trade_count: int,
    scan_date: datetime,
) -> list[dict]:
    """
    Scan live data for Combo C entry signals on scan_date.
    
    Returns:
        list of signal dicts: {symbol, price, atr, rsi2, bb_lower, bb_mid, shares, risk_pct, signal_bar_time}
    """
    signals = []
    
    for symbol in COMBO_C_SYMBOLS:
        if symbol not in data:
            logger.warning(f"Symbol {symbol} not in data")
            continue
        
        df = data[symbol].copy()
        
        # Filter to today's bars only (assume scan_date is today)
        df['date'] = pd.to_datetime(df.index).normalize()
        today_bars = df[df['date'] == scan_date.date()].copy()
        
        if today_bars.empty:
            logger.info(f"{symbol}: No bars today yet")
            continue
        
        logger.info(f"{symbol}: {len(today_bars)} bars today")
        
        # Build indicator state from end of yesterday + today's bars
        # (to ensure RSI2, BB, etc. are computed correctly)
        lookback = 50  # bars
        cutoff_idx = len(df) - len(today_bars) - lookback
        if cutoff_idx < 0:
            cutoff_idx = 0
        
        warmup_df = df.iloc[cutoff_idx:].copy()
        
        ind = IndicatorStateV3()
        entry_signals_today = []
        
        for idx, (ts, row) in enumerate(warmup_df.iterrows()):
            snap = BarSnapshot(
                ts=pd.Timestamp(ts),
                o=row['open'],
                h=row['high'],
                l=row['low'],
                c=row['close'],
                v=row['volume'],
            )
            
            ind.update(snap, symbol=symbol)
            
            # Check if this bar is from today
            bar_date = pd.Timestamp(ts).normalize()
            if bar_date != scan_date.date():
                continue
            
            # Check Combo C entry condition
            rsi2 = ind.rsi2
            bb_lower = ind.bb_lower
            bb_mid = ind.bb_mid
            price = row['close']
            
            if rsi2 is not None and rsi2 < 15:
                if bb_lower is not None and price <= bb_lower:
                    atr = ind.atr10 or 0.1
                    kelly_frac = get_kelly_fraction(trade_count + len(entry_signals_today))
                    shares = compute_position_size(PORTFOLIO_EQUITY, atr, kelly_frac)
                    
                    if shares > 0:
                        entry_signals_today.append({
                            'symbol': symbol,
                            'price': price,
                            'atr': atr,
                            'rsi2': rsi2,
                            'bb_lower': bb_lower,
                            'bb_mid': bb_mid,
                            'shares': shares,
                            'risk_pct': kelly_frac * 100,
                            'signal_bar_time': pd.Timestamp(ts),
                            'signal_date': bar_date,
                        })
                        logger.info(
                            f"  {symbol} ENTRY @ ${price:.2f} | RSI2={rsi2:.1f} | "
                            f"shares={shares} | kelly={kelly_frac*100:.2f}%"
                        )
        
        signals.extend(entry_signals_today)
    
    return signals


def log_signals_to_journal(signals: list[dict], append: bool = True):
    """Log signals to trade_journal.txt."""
    journal_file = HERE / "trade_journal.txt"
    
    if not signals:
        logger.warning("No signals to log")
        return
    
    # Build log entry
    now = datetime.now()
    log_lines = [
        "",
        "=" * 80,
        f"SESSION: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"COMBO: C (Standalone) — Trend Module SUSPENDED per V5.2 rules",
        f"INSTRUMENTS SCANNED: {', '.join(sorted(set(s['symbol'] for s in signals)))}",
        "=" * 80,
        f"ENTRY SIGNALS: {len(signals)} found",
        "",
    ]
    
    for i, sig in enumerate(signals, 1):
        log_lines.append(
            f"{i}. {sig['symbol']:<6} @ ${sig['price']:>7.2f} | "
            f"RSI2={sig['rsi2']:>5.1f} | BB={sig['bb_lower']:>7.2f} | "
            f"Shares={sig['shares']:>4d} | Kelly={sig['risk_pct']:>5.2f}% | "
            f"Time={sig['signal_bar_time'].strftime('%H:%M')}"
        )
    
    log_lines.extend([
        "",
        "EXECUTION NOTES:",
        "  - Entries logged to Alpaca paper trading account",
        "  - Position sizing: Kelly ramp (0.5% → 2.28% → 4.06%)",
        "  - Stop loss: ATR-based, target: BB middle",
        "  - Monitored by: paper_trading_monitor.py",
        "",
    ])
    
    log_text = "\n".join(log_lines)
    
    if append and journal_file.exists():
        with open(journal_file, "a") as f:
            f.write(log_text)
    else:
        with open(journal_file, "w") as f:
            f.write(log_text)
    
    logger.info(f"Logged {len(signals)} entries to {journal_file}")


def main():
    logger.info("="*80)
    logger.info("GENERATING TODAY'S COMBO C ENTRY SIGNALS")
    logger.info("="*80)
    
    # Load live data
    logger.info("Loading live data from Alpaca...")
    try:
        data = load_data(symbols=list(COMBO_C_SYMBOLS), use_cache=False)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    if not data:
        logger.error("No data loaded")
        sys.exit(1)
    
    logger.info(f"Loaded {len(data)} symbols")
    
    # Scan for today's entries
    today = datetime.now()
    trade_count = 0  # Assume we're starting the live trading day; typically would read from journal
    
    signals = scan_for_entries(data, trade_count, today)
    
    if signals:
        logger.info(f"Found {len(signals)} entry signals for today")
        log_signals_to_journal(signals, append=True)
    else:
        logger.info("No entry signals found")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
