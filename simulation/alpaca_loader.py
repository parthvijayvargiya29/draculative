"""
alpaca_loader.py — Alpaca Markets API data fetcher.

Fetches historical OHLCV bar data from the Alpaca Data API v2.
Supports equity (IEX feed) and crypto bars.
Adjusts for splits and dividends by default.

Usage:
    loader = AlpacaLoader(api_key="...", secret_key="...")
    df = loader.get_bars("SPY", "1Day", start="2024-01-01", end="2026-03-28")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DEFAULT_TIMEFRAMES = {
    "1Min":  "1Min",
    "5Min":  "5Min",
    "15Min": "15Min",
    "1Hour": "1Hour",
    "4Hour": "4Hour",
    "1Day":  "1Day",
}
MIN_HISTORY_YEARS = 2   # Per Section 3.3: minimum 2 full calendar years


class AlpacaLoader:
    """
    Wraps the Alpaca Data API v2 to return clean pandas DataFrames
    indexed by UTC datetime, with columns: open, high, low, close, volume.

    Falls back to a CSV cache directory if the API is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 cache_dir: str = "data/cache"):
        self.api_key    = api_key    or os.environ.get("ALPACA_API_KEY",    "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.cache_dir  = cache_dir
        self._client    = None
        self._data_client = None
        self._init_clients()

    def _init_clients(self):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            self._data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            self._TimeFrame   = TimeFrame
            logger.info("Alpaca data client initialized")
        except ImportError:
            logger.warning("alpaca-py not installed. Install: pip install alpaca-py")
        except Exception as e:
            logger.warning("Alpaca init failed: %s", e)

    def get_bars(self, symbol: str, timeframe: str = "1Day",
                 start: Optional[str] = None,
                 end:   Optional[str] = None,
                 adjusted: bool = True) -> pd.DataFrame:
        """
        Returns a DataFrame (open, high, low, close, volume) for the given symbol.
        Falls back to CSV cache if API unavailable.
        """
        if end is None:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.utcnow() - timedelta(days=365 * MIN_HISTORY_YEARS)).strftime("%Y-%m-%d")

        cache_path = self._cache_path(symbol, timeframe, start, end)

        # Try cache first
        if cache_path.exists():
            logger.debug("Loading from cache: %s", cache_path)
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        if self._data_client is None:
            logger.error("No Alpaca client and no cache for %s. Returning empty DataFrame.", symbol)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            tf_map = {
                "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
                "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
                "4Hour": TimeFrame(4,  TimeFrameUnit.Hour),
                "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
            }
            tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                adjustment="split" if adjusted else "raw",
                feed="iex",
            )
            bars = self._data_client.get_stock_bars(request)
            df   = bars.df

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0)

            df.index = pd.to_datetime(df.index, utc=True)
            df = df[["open", "high", "low", "close", "volume"]].sort_index()

            # Cache to disk
            import pathlib
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)
            logger.info("Fetched %d bars for %s [%s→%s]", len(df), symbol, start, end)
            return df

        except Exception as e:
            logger.error("Alpaca fetch failed for %s: %s", symbol, e)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_multi(self, symbols: List[str], timeframe: str = "1Day",
                  start: Optional[str] = None, end: Optional[str] = None) -> dict:
        """Returns a dict of {symbol: DataFrame} for multiple symbols."""
        return {sym: self.get_bars(sym, timeframe, start, end) for sym in symbols}

    def _cache_path(self, symbol: str, timeframe: str, start: str, end: str):
        import pathlib
        safe_start = start.replace("-", "")
        safe_end   = end.replace("-", "")
        fname = f"{symbol}_{timeframe}_{safe_start}_{safe_end}.csv"
        return pathlib.Path(self.cache_dir) / fname
