"""Capture TradingView XHR/Fetch network requests for financial pages.

Saves JSON responses (or raw text) for XHR/fetch requests while loading the
financials pages. This helps discover the underlying API endpoints used by
TradingView to populate the tables.

Usage:
  python companies/src/capture_tv_xhrs.py --ticker IONQ --statement income
  python companies/src/capture_tv_xhrs.py --ticker IONQ --all

Output:
  companies/data/tradingview/xhr/{ticker}_{statement}_{n}.json
  companies/data/tradingview/xhr/{ticker}_{statement}_requests.json  (mapping of saved files to URLs)
"""

import argparse
import json
import os
import re
from urllib.parse import urlparse


def _safe_filename(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]', '_', s)[:200]


def _statement_url(ticker: str, statement: str) -> str:
    base = f"https://www.tradingview.com/symbols/NYSE-{ticker}/"
    if statement == 'income':
        return base + 'financials-income-statement/'
    if statement == 'balance':
        return base + 'financials-balance-sheet/'
    if statement == 'cash':
        return base + 'financials-cash-flow/'
    if statement == 'stats':
        return base + 'financials-statistics-and-ratios/'
    raise ValueError('unknown statement')


def capture_for_statement(playwright, ticker: str, statement: str, out_dir: str):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    def handle_response(response):
        try:
            req = response.request
            if req.resource_type not in ('xhr', 'fetch'):
                return
            url = response.url
            # Heuristic: only capture requests that likely contain financials data
            if not any(x in url.lower() for x in ('financial', 'fundament', 'fundamental', 'financials', 'ratios', 'statistics', 'symbol')) and 'json' not in (response.headers.get('content-type') or '').lower():
                # still allow some JSON fetches
                return
            try:
                body = response.body()
            except Exception:
                return
            if not body:
                return
            try:
                text = body.decode('utf-8')
            except Exception:
                text = body.decode('latin1', errors='ignore')
            # try parse as JSON
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = text
            # save
            idx = len(saved) + 1
            parsed_name = _safe_filename(urlparse(url).path + ('_' + (urlparse(url).query or '')))
            file_name = f"{ticker}_{statement}_{idx}_{parsed_name}.json"
            file_path = os.path.join(out_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                if isinstance(parsed, str):
                    json.dump({'url': url, 'text': parsed}, f, ensure_ascii=False, indent=2)
                else:
                    json.dump({'url': url, 'json': parsed}, f, ensure_ascii=False, indent=2)
            saved.append({'url': url, 'file': file_name})
        except Exception:
            pass

    page.on('response', handle_response)

    url = _statement_url(ticker, statement)
    print('Navigating to', url)
    page.goto(url, wait_until='networkidle')
    page.wait_for_timeout(2000)
    # scroll to ensure lazy-loaded requests
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(1200)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(800)

    # write mapping
    map_path = os.path.join(out_dir, f"{ticker}_{statement}_requests.json")
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(saved, f, indent=2)
    print('Saved', len(saved), 'responses to', out_dir)
    browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', required=True)
    parser.add_argument('--statement', '-s', choices=['income', 'balance', 'cash', 'stats'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--out-dir', default='companies/data/tradingview/xhr')
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print('Playwright not installed. Run: pip install playwright && playwright install')
        raise

    with sync_playwright() as p:
        if args.all:
            for s in ('income', 'balance', 'cash', 'stats'):
                capture_for_statement(p, args.ticker, s, args.out_dir)
        else:
            if not args.statement:
                parser.error('Specify --statement or --all')
            capture_for_statement(p, args.ticker, args.statement, args.out_dir)


if __name__ == '__main__':
    main()
