"""Merge TradingView and yfinance data into canonical JSON and Parquet.

Produces two files for a ticker:
- `companies/data/{TICKER}_merged.json` (canonical long-form JSON)
- `companies/data/{TICKER}_merged.parquet` (long-form parquet table)

Long-form table columns: `ticker, period, metric, value, source`

Usage:
  python companies/src/merge_tv_yfinance.py --ticker IONQ
"""

import argparse
import json
import os
from typing import Dict, List

import pandas as pd


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def tradingview_to_long(ticker: str, tv_json: Dict, statement: str) -> pd.DataFrame:
    # tv_json expected shape: { 'periods': [...], 'rows': {label: [vals...] } }
    rows = []
    if not tv_json:
        return pd.DataFrame(columns=['ticker','period','metric','value','source'])
    periods: List[str] = tv_json.get('periods', [])
    data_rows = tv_json.get('rows', {})
    for metric, vals in data_rows.items():
        for idx, v in enumerate(vals):
            period = periods[idx] if idx < len(periods) else f'col{idx}'
            rows.append({'ticker': ticker, 'period': period, 'metric': metric, 'value': v, 'source': f'tradingview_{statement}'})
    return pd.DataFrame(rows)


def yfinance_to_long(ticker: str, yf_json: Dict) -> pd.DataFrame:
    rows = []
    if not yf_json:
        return pd.DataFrame(columns=['ticker','period','metric','value','source'])
    # history
    history = yf_json.get('history', [])
    for rec in history:
        # try to find a date key
        date = rec.get('Date') or rec.get('date') or rec.get('Datetime')
        # create price row
        price = rec.get('Close') or rec.get('close') or rec.get('Adj Close')
        if date is not None and price is not None:
            rows.append({'ticker': ticker, 'period': str(date), 'metric': 'price', 'value': price, 'source': 'yfinance_history'})
    # financials/bs/cf are structured as metric -> {period -> value}
    for section in ['financials', 'balance_sheet', 'cashflow']:
        sec = yf_json.get(section) or {}
        if isinstance(sec, dict):
            for metric, period_vals in sec.items():
                if not isinstance(period_vals, dict):
                    continue
                for period, value in period_vals.items():
                    rows.append({'ticker': ticker, 'period': str(period), 'metric': str(metric), 'value': value, 'source': f'yfinance_{section}'})
    # top-level info: eps, last price
    info = yf_json.get('info') or {}
    if info:
        if 'regularMarketPrice' in info:
            rows.append({'ticker': ticker, 'period': 'latest', 'metric': 'price', 'value': info.get('regularMarketPrice'), 'source': 'yfinance_info'})
        if 'trailingEps' in info:
            rows.append({'ticker': ticker, 'period': 'latest', 'metric': 'eps', 'value': info.get('trailingEps'), 'source': 'yfinance_info'})
    return pd.DataFrame(rows)


def merge_and_write(ticker: str, out_dir: str = 'companies/data') -> None:
    os.makedirs(out_dir, exist_ok=True)
    yf_path = os.path.join('companies', 'data', f'{ticker}.json')
    tv_base = os.path.join('companies', 'data', 'tradingview')
    yf = load_json(yf_path)
    tv_income = load_json(os.path.join(tv_base, f'{ticker}_income.json'))
    tv_balance = load_json(os.path.join(tv_base, f'{ticker}_balance.json'))
    tv_cash = load_json(os.path.join(tv_base, f'{ticker}_cash.json'))

    df_tv_income = tradingview_to_long(ticker, tv_income, 'income')
    df_tv_balance = tradingview_to_long(ticker, tv_balance, 'balance')
    df_tv_cash = tradingview_to_long(ticker, tv_cash, 'cash')
    df_yf = yfinance_to_long(ticker, yf)

    df = pd.concat([df_tv_income, df_tv_balance, df_tv_cash, df_yf], ignore_index=True, sort=False)

    # normalize value types where possible (strings with commas etc.) — keep as-is for now

    out_json_path = os.path.join(out_dir, f'{ticker}_merged.json')
    out_parquet_path = os.path.join(out_dir, f'{ticker}_merged.parquet')

    # write JSON long-form
    df_records = df.where(pd.notnull(df), None).to_dict(orient='records')
    with open(out_json_path, 'w') as f:
        json.dump({'ticker': ticker, 'rows': df_records}, f, indent=2)

    # normalize 'value' to string (JSON-safe) for Parquet compatibility
    def _safe_val(v):
        if v is None:
            return None
        # keep simple scalars as-is stringified
        try:
            return str(v)
        except Exception:
            try:
                return json.dumps(v)
            except Exception:
                return None

    df['value'] = df['value'].apply(_safe_val)

    # write parquet
    df.to_parquet(out_parquet_path, index=False)
    print('Wrote:', out_json_path, out_parquet_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', required=True)
    parser.add_argument('--out-dir', default='companies/data')
    args = parser.parse_args()
    merge_and_write(args.ticker, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
