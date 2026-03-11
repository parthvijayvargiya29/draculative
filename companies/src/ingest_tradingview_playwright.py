"""Prototype TradingView financials scraper using Playwright.

This script renders the TradingView financials pages and attempts to extract the
financial tables (income statement, balance sheet, cash flow) into a canonical
JSON structure saved under `companies/data/tradingview/{TICKER}_{statement}.json`.

Notes
- TradingView is JS-heavy and the page structure may change. This is a heuristic
  prototyping extractor and will likely require adjustments for robustness.
- Installing Playwright is required (`pip install playwright` and `playwright install`).

Usage:
  python companies/src/ingest_tradingview_playwright.py --ticker IONQ --statement income
  python companies/src/ingest_tradingview_playwright.py --ticker IONQ --statement balance
  python companies/src/ingest_tradingview_playwright.py --ticker IONQ --statement cash

Output schema (example):
{
  "ticker": "IONQ",
  "statement": "income",
  "periods": ["Q3 '20", "Q4 '20", ...],
  "rows": {
    "Total revenue": [61390000, ...],
    "Net income": [753670000, ...]
  }
}
"""

import argparse
import json
import os
import re
from typing import Dict, List


def _canonical_statement_url(ticker: str, statement: str) -> str:
    base = f"https://www.tradingview.com/symbols/NYSE-{ticker}/"
    if statement == "income":
        return base + "financials-income-statement/"
    if statement == "balance":
        return base + "financials-balance-sheet/"
    if statement == "cash":
        return base + "financials-cash-flow/"
    if statement == "stats":
        return base + "financials-statistics-and-ratios/"
    raise ValueError("unknown statement type")


def _clean_text(s: str) -> str:
    # remove excess whitespace, unicode nbsp, and currency symbols
    return re.sub(r"\s+", " ", s).replace('\xa0', ' ').strip()


def _to_number(s: str):
    if s is None:
        return None
    s = s.replace(',', '').replace('$', '').replace('—', '').strip()
    if s == '' or s in ['-', '—']:
        return None
    try:
        # remove parentheses for negatives
        if s.startswith('(') and s.endswith(')'):
            return -float(s[1:-1])
        return float(s)
    except Exception:
        return s


def extract_table_from_page(page) -> Dict:
    # Heuristic: find the header row (periods) by searching for a node containing Q\d or 'TTM'
    content = page.content()
    # If we can find a row of period labels in the DOM via regex, we will then
    # search neighbors for label rows. However, Playwright gives us nice locators,
    # so use JS evaluation: find all elements containing numeric-looking cells.
    header_js = """
    (() => {
      const textNodes = Array.from(document.querySelectorAll('div,span,td'))
        .filter(el => /Q\d|TTM|\d{4}/.test(el.textContent));
      if (textNodes.length === 0) return null;
      // collect unique candidate header texts from the first few matches
      const hdrs = textNodes.slice(0, 60).map(e => e.textContent.trim()).filter(Boolean);
      return hdrs.slice(0, 30);
    })()
    """
    hdrs = page.evaluate(header_js)
    # fallback: empty periods list
    periods: List[str] = []
    if hdrs:
        # filter strings that look like Qx 'yy or year
        periods = [ _clean_text(h) for h in hdrs if re.search(r"Q\d|'\d{2}|TTM|\d{4}", h) ]
        # dedupe while preserving order
        seen = set()
        periods = [p for p in periods if not (p in seen or seen.add(p))]

    # Now find rows by looking for elements that contain known metric words, then collect siblings
    # We'll attempt to find elements containing text with at least 1 alphabetical char and neighboring numeric cells
    rows = {}

    # JS evaluator to find candidate metric elements and extract their adjacent cell texts
    js_extract = """
    (function() {
      const out = [];
      const candidates = Array.from(document.querySelectorAll('div,span,td'));
      for (const el of candidates) {
        const txt = (el.textContent || '').trim();
        if (!txt) continue;
        // skip purely numeric
        if (/^[\d\-\$\,\.\(\)\s]+$/.test(txt)) continue;
        // try to find sibling numeric cells: look at parent and parent's next siblings
        const row = el.closest('tr') || el.parentElement;
        if (!row) continue;
        const cells = Array.from(row.querySelectorAll('div,span,td'))
          .map(c => c.textContent.trim())
          .filter(Boolean);
        if (cells.length <= 1) continue;
        out.push({label: cells[0], cells: cells.slice(1, Math.min(cells.length, 32))});
      }
      return out.slice(0, 300);
    })();
    """

    try:
        extracted = page.evaluate(js_extract)
    except Exception:
        extracted = []

    for item in extracted:
        label = _clean_text(item.get('label') or '')
        if not label:
            continue
        values = [ _to_number(_clean_text(str(v))) for v in item.get('cells', []) ]
        # only keep rows that have at least one numeric value
        if any(isinstance(v, (int, float)) or (isinstance(v, str) and re.search(r'\d', v)) for v in values):
            rows[label] = values

    return {
        'periods': periods,
        'rows': rows
    }


def save_output(ticker: str, statement: str, data: Dict, out_dir: str = 'companies/data/tradingview') -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}_{statement}.json")
    with open(path, 'w') as f:
        json.dump({'ticker': ticker, 'statement': statement, 'periods': data.get('periods', []), 'rows': data.get('rows', {})}, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', required=True)
    parser.add_argument('--statement', '-s', required=True, choices=['income', 'balance', 'cash', 'stats'])
    args = parser.parse_args()
    url = _canonical_statement_url(args.ticker, args.statement)
    print(f'Opening {url} ...')

    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print('Playwright not installed. Run: pip install playwright  && playwright install')
        raise

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')
        # give some extra time for dynamic widgets
        page.wait_for_timeout(1200)
        data = extract_table_from_page(page)
        outpath = save_output(args.ticker, args.statement, data)
        print('Saved:', outpath)
        browser.close()


if __name__ == '__main__':
    main()
