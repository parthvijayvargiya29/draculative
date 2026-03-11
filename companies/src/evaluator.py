#!/usr/bin/env python3
"""Starter evaluator for company fundamentals.

This script is a small, opinionated starting point. It expects per-company CSV/JSON
files with common columns and computes a few example metrics and a normalized score.
"""

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml

DEFAULT_WEIGHTS = {
    "revenue_growth": 0.4,
    "net_margin": 0.3,
    "pe_ratio": 0.2,
    "debt_equity": 0.1,
}


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def load_company_financials(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    csv_path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    json_path = os.path.join(data_dir, f"{ticker}.json")
    if os.path.exists(json_path):
        return pd.read_json(json_path)
    raise FileNotFoundError(f"No data found for {ticker} in {data_dir}")


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "revenue" in df.columns:
        rev = df["revenue"].dropna()
        if len(rev) >= 2:
            metrics["revenue_growth"] = (rev.iloc[-1] / rev.iloc[-2]) - 1.0
        else:
            metrics["revenue_growth"] = float("nan")
    if "net_income" in df.columns and "revenue" in df.columns:
        metrics["net_margin"] = df["net_income"].iloc[-1] / df["revenue"].iloc[-1]
    if "eps" in df.columns and "price" in df.columns:
        eps = df["eps"].iloc[-1]
        metrics["pe_ratio"] = df["price"].iloc[-1] / eps if eps != 0 else float("inf")
    if "total_debt" in df.columns and "shareholders_equity" in df.columns:
        se = df["shareholders_equity"].iloc[-1]
        metrics["debt_equity"] = df["total_debt"].iloc[-1] / se if se != 0 else float("inf")
    return metrics


def score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    total_w = 0.0
    for k, w in weights.items():
        val = metrics.get(k)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            continue
        if k == "pe_ratio" or k == "debt_equity":
            norm = 1.0 / (1.0 + val)
        else:
            norm = 1.0 / (1.0 + np.exp(-val))
        s += norm * w
        total_w += w
    return s / total_w if total_w > 0 else 0.0


def evaluate(ticker: str, data_dir: str = "data", weights: Dict[str, float] = None) -> Dict:
    df = load_company_financials(ticker, data_dir)
    metrics = compute_metrics(df)
    w = weights if weights else DEFAULT_WEIGHTS
    return {"ticker": ticker, "metrics": metrics, "score": score(metrics, w)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate company fundamentals")
    parser.add_argument("--company", "-c", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--data-dir", "-d", default="companies/data", help="Data directory")
    parser.add_argument("--weights", "-w", help="YAML file with weights (optional)")
    args = parser.parse_args()
    weights = None
    if args.weights:
        cfg = load_config(args.weights)
        # expect YAML with top-level mapping of metric->weight
        weights = cfg
    result = evaluate(args.company, data_dir=args.data_dir, weights=weights)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
