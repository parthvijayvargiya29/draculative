"""Dask-based batch runner for backtests and batch predictions.

This script is a template that demonstrates how to parallelize work across
tickers using `dask.distributed`. It is deliberately defensive: if Dask is
not installed, it prints actionable install instructions.

Usage (local):
  python predictor/scripts/dask_backtest.py --tickers AAPL MSFT TSLA

To run on a cluster, start a scheduler and workers and pass the scheduler
address with `--scheduler-address`.
"""
import argparse
import sys
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', nargs='+', required=True, help='Tickers to run backtest on')
    p.add_argument('--scheduler-address', default=None, help='Dask scheduler address (optional)')
    return p.parse_args()


def run_local(tickers: List[str], scheduler_address: str = None):
    try:
        from dask.distributed import Client, as_completed
    except Exception:
        print("Dask is not installed. Install with: python -m pip install 'dask[distributed]'", file=sys.stderr)
        return

    if scheduler_address:
        client = Client(address=scheduler_address)
        print(f"Connected to scheduler: {scheduler_address}")
    else:
        client = Client()
        print(f"Started local Dask cluster: {client}")

    def _backtest(ticker: str) -> dict:
        # Placeholder backtest function. Replace with your real runner.
        # Keep this pure-functional if possible so it serializes well.
        import time
        time.sleep(0.5)
        return {"ticker": ticker, "result": "ok"}

    futures = [client.submit(_backtest, t) for t in tickers]
    for fut in as_completed(futures):
        res = fut.result()
        print(res)


def main():
    args = parse_args()
    run_local(args.tickers, scheduler_address=args.scheduler_address)


if __name__ == '__main__':
    main()
