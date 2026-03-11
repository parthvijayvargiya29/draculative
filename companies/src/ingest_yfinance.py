"""Ingestion adapter using Yahoo Finance (via yfinance).

This is a reliable, low-friction adapter to fetch company info, recent financials,
and historical prices. It writes a JSON file to `companies/data/{TICKER}.json`.

Usage:
  python companies/src/ingest_yfinance.py --ticker IONQ
"""

import argparse
import json
import os
from typing import Dict

import pandas as pd


def fetch_yahoo(ticker: str) -> Dict:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    out: Dict = {}
    try:
        out["info"] = tk.info
    except Exception:
        out["info"] = {}
    def _serialize_df(df):
        if df is None or df.empty:
            return {}
        d = df.fillna("").to_dict(orient="index")
        # ensure keys are strings (timestamps etc.)
        safe = {}
        for k, v in d.items():
            ks = str(k)
            inner = {}
            for ik, iv in v.items():
                inner[str(ik)] = iv
            safe[ks] = inner
        return safe

    try:
        # prefer quarterly if available, fallback to annual financials
        if hasattr(tk, "quarterly_financials") and not tk.quarterly_financials.empty:
            out["financials"] = _serialize_df(tk.quarterly_financials)
        else:
            out["financials"] = _serialize_df(tk.financials)
    except Exception:
        out["financials"] = {}
    try:
        hist = tk.history(period="5y", auto_adjust=False)
        out["history"] = [ {k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k,v in row.items()} for row in hist.reset_index().to_dict(orient="records") ]
    except Exception:
        out["history"] = []
    # balance sheet / cashflow
    try:
        out["balance_sheet"] = _serialize_df(tk.balance_sheet)
    except Exception:
        out["balance_sheet"] = {}
    try:
        out["cashflow"] = _serialize_df(tk.cashflow)
    except Exception:
        out["cashflow"] = {}
    return out


def save_data(ticker: str, data: Dict, data_dir: str = "companies/data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{ticker}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", "-t", required=True, help="Ticker to fetch (e.g., IONQ)")
    parser.add_argument("--data-dir", "-d", default="companies/data", help="Directory to write data")
    args = parser.parse_args()
    print(f"Fetching {args.ticker} from Yahoo Finance...")
    data = fetch_yahoo(args.ticker)
    outpath = save_data(args.ticker, data, data_dir=args.data_dir)
    print(f"Saved data to {outpath}")


if __name__ == "__main__":
    main()
