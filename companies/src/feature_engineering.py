"""Feature engineering for company fundamentals.

Computes derived features from the wide table:
- Growth rates (QoQ, YoY)
- Financial ratios (margins, leverage, efficiency, valuation)
- Rolling statistics (moving averages, volatility)

Produces `{TICKER}_features.parquet` and `{TICKER}_features.json`.

Usage:
  python companies/src/feature_engineering.py --ticker IONQ
"""

import argparse
import json
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_fiscal_period(fp: str):
    """Parse fiscal_period string to (year, quarter) tuple for sorting."""
    if fp == 'latest':
        return (9999, 5)
    if fp == 'TTM':
        return (9999, 4)
    m = re.match(r"(\d{4})-Q(\d)", str(fp))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.match(r"^(\d{4})$", str(fp))
    if m2:
        return (int(m2.group(1)), 0)
    return (0, 0)


def sort_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by fiscal_period chronologically."""
    df = df.copy()
    df['_sort'] = df['fiscal_period'].apply(parse_fiscal_period)
    df = df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    return df


def compute_growth(series: pd.Series, periods: int = 1) -> pd.Series:
    """Compute period-over-period growth rate."""
    shifted = series.shift(periods)
    # Avoid division by zero/negative
    growth = (series - shifted) / shifted.abs().replace(0, np.nan)
    return growth


def compute_yoy_growth(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute year-over-year growth (4 quarters back)."""
    return compute_growth(df[col], periods=4)


def compute_qoq_growth(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute quarter-over-quarter growth."""
    return compute_growth(df[col], periods=1)


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Safe division handling zeros and NaNs."""
    return num / denom.replace(0, np.nan)


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add growth rate features."""
    growth_cols = ['revenue', 'gross_profit', 'net_income', 'ebitda', 'total_assets', 'free_cash_flow']
    
    for col in growth_cols:
        if col in df.columns:
            df[f'{col}_qoq_growth'] = compute_qoq_growth(df, col)
            df[f'{col}_yoy_growth'] = compute_yoy_growth(df, col)
    
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add financial ratio features."""
    
    # Profitability ratios
    if 'gross_profit' in df.columns and 'revenue' in df.columns:
        df['gross_margin'] = safe_divide(df['gross_profit'], df['revenue'])
    
    if 'net_income' in df.columns and 'revenue' in df.columns:
        df['net_margin'] = safe_divide(df['net_income'], df['revenue'])
    
    if 'ebitda' in df.columns and 'revenue' in df.columns:
        df['ebitda_margin'] = safe_divide(df['ebitda'], df['revenue'])
    
    if 'operating_income' in df.columns and 'revenue' in df.columns:
        df['operating_margin'] = safe_divide(df['operating_income'], df['revenue'])
    
    # Leverage ratios
    if 'total_debt' in df.columns and 'total_equity' in df.columns:
        df['debt_to_equity'] = safe_divide(df['total_debt'], df['total_equity'])
    
    if 'total_debt' in df.columns and 'total_assets' in df.columns:
        df['debt_to_assets'] = safe_divide(df['total_debt'], df['total_assets'])
    
    if 'total_liabilities' in df.columns and 'total_assets' in df.columns:
        df['liabilities_to_assets'] = safe_divide(df['total_liabilities'], df['total_assets'])
    
    # Liquidity ratios
    if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
        df['current_ratio'] = safe_divide(df['current_assets'], df['current_liabilities'])
    
    if 'cash' in df.columns and 'current_liabilities' in df.columns:
        df['cash_ratio'] = safe_divide(df['cash'], df['current_liabilities'])
    
    # Efficiency ratios
    if 'revenue' in df.columns and 'total_assets' in df.columns:
        df['asset_turnover'] = safe_divide(df['revenue'], df['total_assets'])
    
    # Return ratios
    if 'net_income' in df.columns and 'total_assets' in df.columns:
        df['roa'] = safe_divide(df['net_income'], df['total_assets'])
    
    if 'net_income' in df.columns and 'total_equity' in df.columns:
        df['roe'] = safe_divide(df['net_income'], df['total_equity'])
    
    if 'net_income' in df.columns and 'shares_outstanding' in df.columns:
        df['earnings_per_share'] = safe_divide(df['net_income'], df['shares_outstanding'])
    
    # Valuation ratios (if price available)
    if 'price' in df.columns and 'eps' in df.columns:
        df['pe_ratio'] = safe_divide(df['price'], df['eps'].abs())
    
    if 'price' in df.columns and 'revenue' in df.columns and 'shares_outstanding' in df.columns:
        revenue_per_share = safe_divide(df['revenue'], df['shares_outstanding'])
        df['ps_ratio'] = safe_divide(df['price'], revenue_per_share)
    
    # Cash flow ratios
    if 'free_cash_flow' in df.columns and 'revenue' in df.columns:
        df['fcf_margin'] = safe_divide(df['free_cash_flow'], df['revenue'])
    
    if 'cash_from_operating' in df.columns and 'revenue' in df.columns:
        df['operating_cash_margin'] = safe_divide(df['cash_from_operating'], df['revenue'])
    
    if 'capex' in df.columns and 'revenue' in df.columns:
        df['capex_to_revenue'] = safe_divide(df['capex'].abs(), df['revenue'])
    
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int] = [2, 4]) -> pd.DataFrame:
    """Add rolling statistics (moving averages, std)."""
    
    rolling_cols = ['revenue', 'net_income', 'gross_profit', 'ebitda', 'free_cash_flow', 'price']
    
    for col in rolling_cols:
        if col not in df.columns:
            continue
        for w in windows:
            # Rolling mean
            df[f'{col}_ma{w}'] = df[col].rolling(window=w, min_periods=1).mean()
            # Rolling std (volatility)
            if w >= 2:
                df[f'{col}_std{w}'] = df[col].rolling(window=w, min_periods=2).std()
    
    # Revenue momentum (current vs moving average)
    if 'revenue' in df.columns and 'revenue_ma4' in df.columns:
        df['revenue_momentum'] = safe_divide(df['revenue'], df['revenue_ma4']) - 1
    
    return df


def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 4]) -> pd.DataFrame:
    """Add lagged features for key metrics."""
    
    lag_cols = ['revenue', 'net_income', 'eps', 'price']
    
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators."""
    
    # Revenue trend: count of consecutive growth quarters
    if 'revenue_qoq_growth' in df.columns:
        growth_positive = (df['revenue_qoq_growth'] > 0).astype(int)
        # Simple streak counter
        streaks = []
        streak = 0
        for g in growth_positive:
            if g == 1:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        df['revenue_growth_streak'] = streaks
    
    # Profitability trend
    if 'net_income' in df.columns:
        df['is_profitable'] = (df['net_income'] > 0).astype(int)
    
    return df


def engineer_features(ticker: str) -> pd.DataFrame:
    """Load wide table and compute all features."""
    
    wide_path = f'companies/data/{ticker}_wide.parquet'
    if not os.path.exists(wide_path):
        raise FileNotFoundError(f"Wide table not found: {wide_path}")
    
    df = pd.read_parquet(wide_path)
    
    # Filter to only fiscal periods (exclude 'latest', 'TTM', col* artifacts)
    df = df[df['fiscal_period'].str.match(r'^\d{4}(-Q\d)?$', na=False)].copy()
    
    # Sort chronologically
    df = sort_by_period(df)
    
    # Apply feature engineering
    df = add_growth_features(df)
    df = add_ratio_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_trend_features(df)
    
    return df


def save_features(ticker: str, df: pd.DataFrame, out_dir: str = 'companies/data') -> None:
    """Save feature table to Parquet and JSON."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    parquet_path = os.path.join(out_dir, f'{ticker}_features.parquet')
    json_path = os.path.join(out_dir, f'{ticker}_features.json')
    
    # Save parquet
    df.to_parquet(parquet_path, index=False)
    
    # Save JSON
    records = df.where(pd.notnull(df), None).to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump({'ticker': ticker, 'rows': records}, f, indent=2, default=str)
    
    print(f'Wrote: {parquet_path} {json_path}')


def summarize_features(df: pd.DataFrame) -> None:
    """Print summary of computed features."""
    
    # Categorize columns
    base_cols = ['fiscal_period', 'revenue', 'net_income', 'gross_profit', 'ebitda', 
                 'total_assets', 'total_liabilities', 'total_debt', 'total_equity',
                 'cash', 'free_cash_flow', 'eps', 'price', 'shares_outstanding']
    
    growth_cols = [c for c in df.columns if '_growth' in c]
    ratio_cols = [c for c in df.columns if c.endswith('_margin') or c.endswith('_ratio') 
                  or c in ['roa', 'roe', 'asset_turnover', 'earnings_per_share']]
    rolling_cols = [c for c in df.columns if '_ma' in c or '_std' in c]
    lag_cols = [c for c in df.columns if '_lag' in c]
    trend_cols = [c for c in df.columns if 'streak' in c or 'is_' in c or 'momentum' in c]
    
    print(f"\nFeature summary ({len(df)} periods, {len(df.columns)} columns):")
    print(f"  Growth features ({len(growth_cols)}): {growth_cols[:5]}...")
    print(f"  Ratio features ({len(ratio_cols)}): {ratio_cols[:5]}...")
    print(f"  Rolling features ({len(rolling_cols)}): {rolling_cols[:5]}...")
    print(f"  Lag features ({len(lag_cols)}): {lag_cols[:5]}...")
    print(f"  Trend features ({len(trend_cols)}): {trend_cols}")
    
    # Show sample of latest period
    if len(df) > 0:
        latest = df.iloc[-1]
        print(f"\nLatest period ({latest['fiscal_period']}) sample metrics:")
        sample_cols = ['revenue', 'revenue_qoq_growth', 'revenue_yoy_growth', 
                       'gross_margin', 'net_margin', 'debt_to_assets', 'roa', 'revenue_momentum']
        for c in sample_cols:
            if c in df.columns:
                val = latest[c]
                if pd.notna(val):
                    if 'growth' in c or 'margin' in c or 'ratio' in c or c in ['roa', 'roe', 'momentum']:
                        print(f"    {c}: {val:.2%}")
                    else:
                        print(f"    {c}: {val:,.0f}" if abs(val) > 100 else f"    {c}: {val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', required=True, help='Ticker symbol')
    parser.add_argument('--out-dir', default='companies/data', help='Output directory')
    args = parser.parse_args()
    
    print(f"Engineering features for {args.ticker}...")
    df = engineer_features(args.ticker)
    save_features(args.ticker, df, out_dir=args.out_dir)
    summarize_features(df)


if __name__ == '__main__':
    main()
