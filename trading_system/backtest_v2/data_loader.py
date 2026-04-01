"""
Production-Grade Data Loader & Validator

Fetches 2 years of 15-minute OHLCV data for multiple symbols.
Performs strict validation: OHLC integrity, gap detection, volume checks, split adjustment.

CRITICAL: Data is loaded ONCE and stored. The simulation engine then replays
bars strictly forward — no future data ever leaks into decisions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'QQQ']

# Market hours: 9:30–16:00 ET → 26 bars of 15 min each
BARS_PER_DAY = 26
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN = 0

MAX_GAP_MINUTES = 20       # Alert if gap > 1 bar (15 min + 5 min tolerance)
MAX_GAP_FRACTION = 0.01    # Alert if >1% of bars are gaps


# ─────────────────────────────────────────────────────────────────────────────
# Data Loader
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Fetches and validates historical 15-minute OHLCV data.

    yfinance limitation: 15-min data is only available for the past ~60 days
    via a single API call. To get 2 years, we fetch in 60-day chunks.
    """

    def __init__(self, symbols: List[str] = None, lookback_years: int = 2):
        self.symbols = symbols or SYMBOLS
        self.lookback_years = lookback_years
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.clean_data: Dict[str, pd.DataFrame] = {}
        self.validation_report: Dict[str, dict] = {}

    # ── Public ──────────────────────────────────────────────────────────────

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch + validate all symbols. Returns clean data keyed by symbol."""
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=365 * self.lookback_years)

        logger.info(f"Loading {len(self.symbols)} symbols | "
                    f"{start_date.date()} → {end_date.date()} | 15-min bars")

        for symbol in self.symbols:
            logger.info(f"  Fetching {symbol}...")
            df = self._fetch_chunked(symbol, start_date, end_date)
            if df is None or len(df) == 0:
                logger.error(f"  ✗ {symbol}: No data retrieved")
                continue

            self.raw_data[symbol] = df
            clean, report = self._validate_and_clean(symbol, df)
            self.clean_data[symbol] = clean
            self.validation_report[symbol] = report

            status = "✓" if report['passed'] else "⚠"
            logger.info(
                f"  {status} {symbol}: {len(clean):,} bars | "
                f"gaps={report['gap_count']} | "
                f"bad_ohlc={report['bad_ohlc_count']} | "
                f"zero_vol={report['zero_vol_count']} | "
                f"trading_days={report['trading_days']}"
            )

        self._print_validation_summary()
        return self.clean_data

    def get_aligned_index(self) -> pd.DatetimeIndex:
        """Return the union of all timestamps across symbols (market hours only)."""
        if not self.clean_data:
            raise RuntimeError("Call load_all() first")
        all_idx = pd.DatetimeIndex([])
        for df in self.clean_data.values():
            all_idx = all_idx.union(df.index)
        return all_idx.sort_values()

    # ── Fetch ────────────────────────────────────────────────────────────────

    def _fetch_chunked(self, symbol: str, start: datetime,
                       end: datetime) -> Optional[pd.DataFrame]:
        """
        yfinance only returns ~60 days of 15-min data per call.
        Split the full range into 50-day chunks and concatenate.
        """
        chunks = []
        chunk_start = start
        chunk_days = 50  # Stay safely under the 60-day limit

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            for attempt in range(3):
                try:
                    df = yf.download(
                        symbol,
                        start=chunk_start.strftime('%Y-%m-%d'),
                        end=chunk_end.strftime('%Y-%m-%d'),
                        interval='15m',
                        progress=False,
                        auto_adjust=True,   # Adjusts for splits & dividends
                        prepost=False,       # Regular hours only
                    )
                    if df is not None and len(df) > 0:
                        chunks.append(df)
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"    Failed chunk {chunk_start.date()} "
                                       f"→ {chunk_end.date()}: {e}")
                    time.sleep(1)

            chunk_start = chunk_end
            time.sleep(0.3)  # Respect rate limits

        if not chunks:
            return None

        combined = pd.concat(chunks)
        # Remove duplicates from overlapping chunk boundaries
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)

        # Flatten MultiIndex columns if present (yfinance v0.2+)
        if isinstance(combined.columns, pd.MultiIndex):
            combined.columns = combined.columns.get_level_values(0)

        return combined

    # ── Validate & Clean ────────────────────────────────────────────────────

    def _validate_and_clean(self, symbol: str,
                            df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Run all validation checks and return clean DataFrame + report.

        Checks:
        1. Required columns present
        2. OHLC integrity (high >= max(open,close), low <= min(open,close))
        3. Volume > 0
        4. No NaN values
        5. Market hours only (9:30–16:00 ET)
        6. Gap detection
        7. Monotonic timestamps (no backward jumps)
        """
        report = {
            'symbol': symbol,
            'raw_bars': len(df),
            'bad_ohlc_count': 0,
            'zero_vol_count': 0,
            'nan_count': 0,
            'gap_count': 0,
            'large_gaps': [],
            'trading_days': 0,
            'passed': True,
            'warnings': [],
        }

        # ── 1. Column check ──────────────────────────────────────────────────
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            report['passed'] = False
            report['warnings'].append(f"Missing columns: {missing}")
            return df, report

        df = df[required].copy()

        # ── 2. Ensure numeric ────────────────────────────────────────────────
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ── 3. Drop NaN rows ─────────────────────────────────────────────────
        nan_mask = df.isnull().any(axis=1)
        report['nan_count'] = int(nan_mask.sum())
        df = df[~nan_mask]

        # ── 4. Market hours filter (9:30–15:45 ET) ───────────────────────────
        # yfinance with auto_adjust=True returns timezone-aware timestamps
        if df.index.tz is not None:
            df.index = df.index.tz_convert('America/New_York')
        else:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

        market_mask = (
            (df.index.time >= pd.Timestamp('09:30').time()) &
            (df.index.time <= pd.Timestamp('15:45').time()) &
            (df.index.dayofweek < 5)  # Mon–Fri only
        )
        df = df[market_mask]

        if len(df) == 0:
            report['passed'] = False
            report['warnings'].append("No bars remain after market hours filter")
            return df, report

        # ── 5. OHLC integrity ────────────────────────────────────────────────
        bad_high = df['High'] < df[['Open', 'Close']].max(axis=1)
        bad_low = df['Low'] > df[['Open', 'Close']].min(axis=1)
        bad_ohlc = bad_high | bad_low
        report['bad_ohlc_count'] = int(bad_ohlc.sum())
        if report['bad_ohlc_count'] > 0:
            report['warnings'].append(
                f"Fixing {report['bad_ohlc_count']} bars with bad OHLC"
            )
            # Fix by clamping
            df.loc[bad_high, 'High'] = df.loc[bad_high, ['Open', 'Close']].max(axis=1)
            df.loc[bad_low, 'Low'] = df.loc[bad_low, ['Open', 'Close']].min(axis=1)

        # ── 6. Volume > 0 ─────────────────────────────────────────────────────
        zero_vol = df['Volume'] <= 0
        report['zero_vol_count'] = int(zero_vol.sum())
        if report['zero_vol_count'] > 0:
            report['warnings'].append(
                f"Dropping {report['zero_vol_count']} zero-volume bars"
            )
            df = df[~zero_vol]

        # ── 7. Monotonic index ────────────────────────────────────────────────
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
            report['warnings'].append("Timestamps were not monotonic — sorted")

        # ── 8. Gap detection ──────────────────────────────────────────────────
        time_diffs = df.index.to_series().diff().dropna()
        # Only look at consecutive bars within the same day
        # (gaps between 15:45 and next day 09:30 are expected)
        same_day = df.index.to_series().dt.date == df.index.to_series().shift(1).dt.date
        intraday_diffs = time_diffs[same_day]

        large_gaps = intraday_diffs[intraday_diffs > pd.Timedelta(minutes=MAX_GAP_MINUTES)]
        report['gap_count'] = len(large_gaps)
        if len(large_gaps) > 0:
            report['large_gaps'] = [
                f"{ts}: {gap.total_seconds()/60:.0f} min"
                for ts, gap in large_gaps.head(5).items()
            ]
            gap_fraction = len(large_gaps) / len(df)
            if gap_fraction > MAX_GAP_FRACTION:
                report['warnings'].append(
                    f"High gap rate: {gap_fraction:.2%} of bars have gaps"
                )

        # ── 9. Trading day count ──────────────────────────────────────────────
        report['trading_days'] = df.index.normalize().nunique()
        report['clean_bars'] = len(df)

        return df, report

    # ── Summary ──────────────────────────────────────────────────────────────

    def _print_validation_summary(self):
        logger.info("\n" + "═" * 60)
        logger.info("DATA VALIDATION SUMMARY")
        logger.info("═" * 60)
        for sym, r in self.validation_report.items():
            logger.info(f"\n{sym}:")
            logger.info(f"  Raw bars:       {r.get('raw_bars', 'N/A'):>8,}")
            logger.info(f"  Clean bars:     {r.get('clean_bars', 0):>8,}")
            logger.info(f"  Trading days:   {r.get('trading_days', 0):>8,}")
            logger.info(f"  Bad OHLC fixed: {r.get('bad_ohlc_count', 0):>8,}")
            logger.info(f"  Zero-vol drops: {r.get('zero_vol_count', 0):>8,}")
            logger.info(f"  NaN drops:      {r.get('nan_count', 0):>8,}")
            logger.info(f"  Intraday gaps:  {r.get('gap_count', 0):>8,}")
            for w in r.get('warnings', []):
                logger.warning(f"  ⚠  {w}")
        logger.info("\n" + "═" * 60)
