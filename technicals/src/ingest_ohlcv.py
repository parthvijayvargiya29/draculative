"""Ingest OHLCV market data from Yahoo Finance.

Fetches daily/intraday price data for technical analysis.

Usage:
    python ingest_ohlcv.py --ticker NVDA --period 2y --interval 1d
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd


def fetch_ohlcv(ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    # Normalize columns
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Reset index to have date as column
    df = df.reset_index()
    
    # Find the date/datetime column and rename it
    date_cols = [c for c in df.columns if c.lower() in ('date', 'datetime', 'index')]
    if date_cols:
        df = df.rename(columns={date_cols[0]: 'datetime'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'datetime'})
    
    # Add ticker column
    df['ticker'] = ticker
    
    return df


def save_ohlcv(ticker: str, df: pd.DataFrame, interval: str = '1d',
               base_path: str = 'technicals/data'):
    """Save OHLCV data to parquet and JSON."""
    os.makedirs(base_path, exist_ok=True)
    
    suffix = f"_{interval}" if interval != '1d' else ""
    
    # Save parquet
    parquet_path = os.path.join(base_path, f'{ticker}_ohlcv{suffix}.parquet')
    df.to_parquet(parquet_path, index=False)
    
    # Save JSON metadata
    meta = {
        'ticker': ticker,
        'interval': interval,
        'start_date': str(df['datetime'].min()),
        'end_date': str(df['datetime'].max()),
        'records': len(df),
        'columns': list(df.columns),
        'fetched_at': datetime.now().isoformat()
    }
    
    json_path = os.path.join(base_path, f'{ticker}_ohlcv{suffix}_meta.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved: {parquet_path}")
    print(f"Saved: {json_path}")
    
    return parquet_path


def main():
    parser = argparse.ArgumentParser(description='Fetch OHLCV market data')
    parser.add_argument('--ticker', '-t', required=True, help='Stock ticker symbol')
    parser.add_argument('--period', '-p', default='2y', 
                        help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)')
    parser.add_argument('--interval', '-i', default='1d',
                        help='Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)')
    args = parser.parse_args()
    
    print(f"Fetching {args.ticker} OHLCV data ({args.period}, {args.interval})...")
    
    df = fetch_ohlcv(args.ticker, args.period, args.interval)
    
    print(f"\nData range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Records: {len(df)}")
    print(f"\nLatest data:")
    print(df.tail())
    
    save_ohlcv(args.ticker, df, args.interval)


if __name__ == '__main__':
    main()
