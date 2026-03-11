"""Normalize merged data and produce a wide Parquet per ticker ready for modeling.

Steps performed:
- Load merged long-form data (`{TICKER}_merged.parquet`) and TradingView stats JSON.
- Apply metric name mapping to canonical names.
- Parse and coerce numeric values (handles commas, parentheses, K/M/B/T suffixes, percentages).
- Normalize period strings into fiscal keys (`YYYY-Qn` or `YYYY`, leave 'TTM' and 'latest').
- Pivot to wide form: one row per `fiscal_period`, columns per canonical metric.
- Write `companies/data/{TICKER}_wide.parquet` and `{TICKER}_wide.json` (metadata + rows).

Usage:
  python companies/src/normalize_and_widen.py --ticker IONQ
"""

import argparse
import json
import os
import re
from typing import Optional

import pandas as pd

# mapping of common TradingView and yfinance metric names to canonical metric keys
METRIC_MAP = {
    # Revenue
    'Total revenue': 'revenue',
    'Total Revenue': 'revenue',
    'Revenue': 'revenue',
    'Operating Revenue': 'revenue',
    # Net income
    'Net income': 'net_income',
    'Net Income': 'net_income',
    'Net Income Common Stockholders': 'net_income',
    'Net Income From Continuing Operations': 'net_income',
    'Net Income Including Noncontrolling Interests': 'net_income_incl_minority',
    # Gross profit
    'Gross profit': 'gross_profit',
    'Gross Profit': 'gross_profit',
    # Operating income
    'Operating income': 'operating_income',
    'Operating Income': 'operating_income',
    'Total Operating Income As Reported': 'operating_income',
    # EBITDA
    'EBITDA': 'ebitda',
    'Normalized EBITDA': 'normalized_ebitda',
    'EBIT': 'ebit',
    # Pretax income
    'Pretax income': 'pretax_income',
    'Pretax Income': 'pretax_income',
    # EPS
    'EPS (Basic)': 'eps',
    'Earnings per share (EPS)': 'eps',
    'Basic EPS': 'eps',
    'Diluted EPS': 'eps_diluted',
    # Balance sheet - Assets
    'Total assets': 'total_assets',
    'Total Assets': 'total_assets',
    'Current Assets': 'current_assets',
    'Total Non Current Assets': 'non_current_assets',
    'Cash And Cash Equivalents': 'cash',
    'Cash Cash Equivalents And Short Term Investments': 'cash_and_investments',
    'Accounts Receivable': 'accounts_receivable',
    'Inventory': 'inventory',
    'Goodwill And Other Intangible Assets': 'goodwill_intangibles',
    'Net PPE': 'net_ppe',
    'Net Tangible Assets': 'net_tangible_assets',
    # Balance sheet - Liabilities
    'Total liabilities': 'total_liabilities',
    'Total Liabilities Net Minority Interest': 'total_liabilities',
    'Current Liabilities': 'current_liabilities',
    'Total Non Current Liabilities Net Minority Interest': 'non_current_liabilities',
    'Accounts Payable': 'accounts_payable',
    'Total Debt': 'total_debt',
    'Long Term Debt And Capital Lease Obligation': 'long_term_debt',
    'Current Debt And Capital Lease Obligation': 'current_debt',
    # Balance sheet - Equity
    'Total Equity Gross Minority Interest': 'total_equity',
    'Common Stock Equity': 'common_equity',
    'Stockholders Equity': 'shareholders_equity',
    'Additional Paid In Capital': 'additional_paid_in_capital',
    'Retained Earnings': 'retained_earnings',
    # Cash flow
    'Cash from operating activities': 'cash_from_operating',
    'Cash Flow From Continuing Operating Activities': 'cash_from_operating',
    'Operating Cash Flow': 'cash_from_operating',
    'Cash from investing activities': 'cash_from_investing',
    'Cash Flow From Continuing Investing Activities': 'cash_from_investing',
    'Investing Cash Flow': 'cash_from_investing',
    'Cash from financing activities': 'cash_from_financing',
    'Cash Flow From Continuing Financing Activities': 'cash_from_financing',
    'Financing Cash Flow': 'cash_from_financing',
    'Free cash flow': 'free_cash_flow',
    'Free Cash Flow': 'free_cash_flow',
    'Capital Expenditure': 'capex',
    # Shares
    'Market Cap': 'market_cap',
    'Shares outstanding': 'shares_outstanding',
    'Ordinary Shares Number': 'shares_outstanding',
    'Share Issued': 'shares_issued',
    'Basic Average Shares': 'avg_shares_basic',
    'Diluted Average Shares': 'avg_shares_diluted',
    # Valuation ratios
    'PE ratio': 'pe_ratio',
    'Price/Earnings (P/E)': 'pe_ratio',
    # Expenses
    'Cost Of Revenue': 'cost_of_revenue',
    'Total Expenses': 'total_expenses',
    'Research And Development': 'r_and_d',
    'General And Administrative Expense': 'g_and_a',
    # Price
    'price': 'price',
}


def canonical_metric(name: str) -> str:
    name = name.strip()
    if name in METRIC_MAP:
        return METRIC_MAP[name]
    # fallback: lowercase, remove non-word and replace spaces with underscore
    s = re.sub(r"[^0-9a-zA-Z]+", ' ', name).strip().lower().replace(' ', '_')
    return s


def parse_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s == '' or s in ['-', '—', 'NaN', 'None']:
        return None
    # percentages
    if s.endswith('%'):
        try:
            return float(s.rstrip('%').replace(',', '')) / 100.0
        except Exception:
            return None
    # handle parentheses negative
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1]
    s = s.replace(',', '').replace('\u202f', '')
    # handle abbreviations: K, M, B, T
    m = re.match(r'^([\d\.\-]+)\s*([KMBTkmBt]?)$', s)
    if m:
        base = float(m.group(1))
        suf = m.group(2).upper()
        mult = 1.0
        if suf == 'K':
            mult = 1e3
        elif suf == 'M':
            mult = 1e6
        elif suf == 'B':
            mult = 1e9
        elif suf == 'T':
            mult = 1e12
        val = base * mult
        return -val if neg else val
    # try to parse plain float
    try:
        return float(s)
    except Exception:
        return None


def parse_period_to_fiscal(period: str) -> str:
    if period is None:
        return 'unknown'
    p = str(period).strip()
    if p.lower() in ('latest', ''):
        return 'latest'
    if p.upper() == 'TTM':
        return 'TTM'
    # Quarter format like Q3 '20 or Q3 2020
    m = re.search(r"Q(\d)\s*'?(\d{2,4})", p)
    if m:
        q = int(m.group(1))
        y = int(m.group(2))
        if y < 100:
            y += 2000
        return f"{y}-Q{q}"
    # ISO-like date
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", p)
    if m2:
        y = int(m2.group(1))
        month = int(m2.group(2))
        q = (month - 1) // 3 + 1
        return f"{y}-Q{q}"
    # year-only
    m3 = re.search(r"^(20\d{2}|19\d{2})$", p)
    if m3:
        return m3.group(1)
    # fallback to raw cleaned string
    return re.sub(r"\s+", "_", p)


def normalize_and_widen(ticker: str) -> None:
    base = os.path.join('companies', 'data')
    merged_parquet = os.path.join(base, f'{ticker}_merged.parquet')
    tv_stats = os.path.join(base, 'tradingview', f'{ticker}_stats.json')
    if not os.path.exists(merged_parquet):
        raise FileNotFoundError(merged_parquet)

    df = pd.read_parquet(merged_parquet)

    # canonicalize metric names
    df['metric_canon'] = df['metric'].fillna('').apply(canonical_metric)

    # coerce values
    df['value_num'] = df['value'].apply(parse_number)

    # normalize periods
    df['fiscal_period'] = df['period'].apply(parse_period_to_fiscal)

    # prefer numeric value where available, else keep None
    df['value_final'] = df['value_num']

    # pivot to wide form
    pivot = df.pivot_table(index='fiscal_period', columns='metric_canon', values='value_final', aggfunc='first')
    pivot = pivot.reset_index()

    # attach stats (snapshot metrics) into 'latest' row or separate columns
    if os.path.exists(tv_stats):
        with open(tv_stats, 'r') as f:
            stats = json.load(f)
        # stats JSON comes from the scraper; stored periods and rows - try to extract rows
        stats_rows = stats.get('rows', {}) if isinstance(stats, dict) else {}
        # parse each stat value and put into pivot under fiscal_period='latest'
        latest_row = pivot[pivot['fiscal_period'] == 'latest'] if 'latest' in pivot['fiscal_period'].values else None
        for raw_name, vals in stats_rows.items():
            key = canonical_metric(raw_name)
            # stats page likely has single-cell values; find first numeric
            v = None
            if isinstance(vals, list) and len(vals) > 0:
                v = parse_number(vals[0])
            elif isinstance(vals, (str, int, float)):
                v = parse_number(vals)
            # assign into pivot
            if 'latest' in pivot['fiscal_period'].values:
                pivot.loc[pivot['fiscal_period'] == 'latest', key] = v
            else:
                # append a latest row
                new = {c: None for c in pivot.columns}
                new['fiscal_period'] = 'latest'
                new[key] = v
                pivot = pd.concat([pivot, pd.DataFrame([new])], ignore_index=True, sort=False)

    # sort fiscal_periods (put 'latest' last)
    def sort_key(fp):
        if fp == 'latest':
            return (9999, 4)
        if fp == 'TTM':
            return (9998, 4)
        m = re.match(r"(\d{4})-Q(\d)", str(fp))
        if m:
            return (int(m.group(1)), int(m.group(2)))
        if re.match(r"^\d{4}$", str(fp)):
            return (int(fp), 4)
        return (0, 0)

    pivot['__sort'] = pivot['fiscal_period'].apply(sort_key)
    pivot = pivot.sort_values('__sort').drop(columns='__sort')

    # ===== Clean up noisy columns =====
    # Drop columns whose name is > 50 chars (garbage from DOM scraping)
    good_cols = ['fiscal_period'] + [c for c in pivot.columns if c != 'fiscal_period' and len(c) <= 50]
    pivot = pivot[good_cols]

    # Drop columns that contain digits concatenated with underscores (e.g., prices embedded)
    noise_pattern = re.compile(r'^\d+_|_\d{2}_\d{2}_|_\d+_\d+_\d+')
    good_cols = ['fiscal_period'] + [c for c in pivot.columns if c != 'fiscal_period' and not noise_pattern.search(c)]
    pivot = pivot[good_cols]

    # Known DOM artifact column names to drop
    DOM_NOISE = {
        'skip_to_main_content', 'ionq', 'search_products_community_markets_brokers_more_en_get_started',
    }
    good_cols = ['fiscal_period'] + [c for c in pivot.columns if c != 'fiscal_period' and c.lower() not in DOM_NOISE and c.lower() != ticker.lower()]
    pivot = pivot[good_cols]

    # Drop empty columns (all NaN)
    pivot = pivot.dropna(axis=1, how='all')

    out_dir = base
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{ticker}_wide.parquet')
    out_json = os.path.join(out_dir, f'{ticker}_wide.json')
    pivot.to_parquet(out_path, index=False)
    pivot_records = pivot.where(pd.notnull(pivot), None).to_dict(orient='records')
    with open(out_json, 'w') as f:
        json.dump({'ticker': ticker, 'rows': pivot_records}, f, indent=2)
    print('Wrote wide:', out_path, out_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', required=True)
    args = parser.parse_args()
    normalize_and_widen(args.ticker)


if __name__ == '__main__':
    main()
