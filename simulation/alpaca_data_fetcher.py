"""
alpaca_data_fetcher.py — Universe data loader for Nucleus Validation.

Fetches 2+ years of daily OHLCV bars for the full nucleus-scoring universe:

  Core indices / volatility:
    SPY, QQQ, VXX, TLT, GLD, UUP

  Sector ETFs (10 SPDR sectors):
    XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLC

  Mega-cap single stocks:
    NVDA, AAPL, MSFT, GOOGL, META, AMZN, TSLA

Data is cached to data/cache/universe/ as CSV files.
Cache is valid for 24 hours; force-refresh with refresh=True.

Requires:
  ALPACA_API_KEY and ALPACA_SECRET_KEY in environment OR in .env file
  at the repo root.

  pip install alpaca-trade-api python-dotenv
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE: Dict[str, str] = {
    # Core
    "SPY":   "SPDR S&P 500 ETF",
    "QQQ":   "Invesco QQQ ETF",
    "VXX":   "iPath Series B S&P 500 VIX Short-Term Futures ETN",
    "TLT":   "iShares 20+ Year Treasury Bond ETF",
    "GLD":   "SPDR Gold Shares",
    "UUP":   "Invesco DB US Dollar Index Bullish Fund",
    # Sectors
    "XLK":   "Technology",
    "XLF":   "Financials",
    "XLE":   "Energy",
    "XLV":   "Health Care",
    "XLI":   "Industrials",
    "XLY":   "Consumer Discretionary",
    "XLP":   "Consumer Staples",
    "XLU":   "Utilities",
    "XLRE":  "Real Estate",
    "XLC":   "Communication Services",
    # Mega-caps
    "NVDA":  "NVIDIA",
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet",
    "META":  "Meta Platforms",
    "AMZN":  "Amazon",
    "TSLA":  "Tesla",
}

LOOKBACK_YEARS  = 2
CACHE_DIR       = Path("data/cache/universe")
CACHE_MAX_AGE   = 24 * 3600     # 1 day in seconds
REQUIRED_SYMBOL = "SPY"         # Validation fails without this one


class AlpacaDataFetcher:
    """
    Loads 2 years of daily bars for the entire nucleus universe.

    Priority order:
      1. Fresh Alpaca API call (if keys present and cache stale)
      2. Disk cache CSV (if fresh enough)
      3. yfinance fallback (if alpaca-trade-api not installed or keys missing)
    """

    def __init__(self, api_key: str = "", secret_key: str = "",
                 base_url: str = "https://paper-api.alpaca.markets"):
        # Load .env if keys not passed directly
        self._load_dotenv()
        self.api_key    = api_key    or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url   = base_url

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_universe(self, refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all symbols in UNIVERSE. Returns {symbol: DataFrame}.
        DataFrame columns: open, high, low, close, volume
        Index: pd.DatetimeIndex (UTC, daily)
        """
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS + 30)

        result: Dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for sym in UNIVERSE:
            cached = self._load_cache(sym)
            if cached is not None and not refresh:
                result[sym] = cached
                logger.info("  [cache] %s  (%d bars)", sym, len(cached))
            else:
                missing.append(sym)

        if missing:
            logger.info("Fetching %d symbols from API/fallback: %s", len(missing), missing)
            fetched = self._fetch_batch(missing, start_date, end_date)
            for sym, df in fetched.items():
                if not df.empty:
                    self._save_cache(sym, df)
                    result[sym] = df
                    logger.info("  [fetch] %s  (%d bars)", sym, len(df))
                else:
                    logger.warning("  [empty] %s — no data returned", sym)

        if REQUIRED_SYMBOL not in result or result[REQUIRED_SYMBOL].empty:
            raise RuntimeError(
                f"Could not load {REQUIRED_SYMBOL} data. "
                "Check ALPACA_API_KEY / ALPACA_SECRET_KEY in .env, "
                "or install yfinance as fallback: pip install yfinance"
            )

        logger.info(
            "Universe loaded: %d/%d symbols, SPY period %s → %s",
            len(result), len(UNIVERSE),
            result[REQUIRED_SYMBOL].index[0].date(),
            result[REQUIRED_SYMBOL].index[-1].date(),
        )
        return result

    # ── Fetch strategies ──────────────────────────────────────────────────────

    def _fetch_batch(self, symbols: list[str],
                     start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Try Alpaca first; fall back to yfinance."""
        if self.api_key and self.secret_key:
            try:
                return self._fetch_alpaca(symbols, start, end)
            except Exception as exc:
                logger.warning("Alpaca fetch failed (%s) — falling back to yfinance", exc)

        return self._fetch_yfinance(symbols, start, end)

    def _fetch_alpaca(self, symbols: list[str],
                      start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch from Alpaca Data API v2 using alpaca-trade-api."""
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version="v2",
        )

        start_str = start.strftime("%Y-%m-%d")
        end_str   = end.strftime("%Y-%m-%d")
        result: Dict[str, pd.DataFrame] = {}

        # Fetch in batches of 10 to avoid rate limits
        for i in range(0, len(symbols), 10):
            batch = symbols[i : i + 10]
            try:
                bars = api.get_bars(
                    batch,
                    "1Day",
                    start=start_str,
                    end=end_str,
                    adjustment="split",
                    feed="iex",
                ).df

                if bars.empty:
                    continue

                # Multi-symbol response has symbol as index level 0
                if isinstance(bars.index, pd.MultiIndex):
                    for sym in batch:
                        try:
                            df = bars.xs(sym, level=0).copy()
                            df.index = pd.to_datetime(df.index, utc=True)
                            df.columns = [c.lower() for c in df.columns]
                            result[sym] = df[["open", "high", "low", "close", "volume"]]
                        except KeyError:
                            logger.debug("No data for %s in batch response", sym)
                else:
                    # Single symbol — assign directly
                    sym = batch[0]
                    bars.index = pd.to_datetime(bars.index, utc=True)
                    bars.columns = [c.lower() for c in bars.columns]
                    result[sym] = bars[["open", "high", "low", "close", "volume"]]

                if i + 10 < len(symbols):
                    time.sleep(0.4)   # Respect rate limit

            except Exception as exc:
                logger.warning("Batch fetch error for %s: %s", batch, exc)
                # Try each symbol individually as fallback
                for sym in batch:
                    try:
                        df = api.get_bars(
                            sym, "1Day",
                            start=start_str, end=end_str,
                            adjustment="split", feed="iex",
                        ).df
                        if not df.empty:
                            df.index = pd.to_datetime(df.index, utc=True)
                            df.columns = [c.lower() for c in df.columns]
                            result[sym] = df[["open", "high", "low", "close", "volume"]]
                    except Exception as e2:
                        logger.debug("Individual fetch failed for %s: %s", sym, e2)

        return result

    def _fetch_yfinance(self, symbols: list[str],
                        start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Fallback: yfinance download."""
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError(
                "Neither alpaca-trade-api nor yfinance is available. "
                "Install one: pip install yfinance"
            )

        result: Dict[str, pd.DataFrame] = {}
        start_str = start.strftime("%Y-%m-%d")
        end_str   = end.strftime("%Y-%m-%d")

        for sym in symbols:
            try:
                raw = yf.download(
                    sym, start=start_str, end=end_str,
                    auto_adjust=True, progress=False,
                )
                if raw.empty:
                    continue

                # yfinance returns MultiIndex columns when multi=False sometimes
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                raw.columns = [c.lower() for c in raw.columns]
                raw.index   = pd.to_datetime(raw.index, utc=True)
                df = raw[["open", "high", "low", "close", "volume"]].copy()
                df = df.dropna(subset=["close"])
                result[sym] = df
                time.sleep(0.1)
            except Exception as exc:
                logger.warning("yfinance failed for %s: %s", sym, exc)

        return result

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str) -> Path:
        return CACHE_DIR / f"{symbol}_1Day.csv"

    def _load_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > CACHE_MAX_AGE:
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df.columns = [c.lower() for c in df.columns]
            if "close" not in df.columns:
                return None
            return df
        except Exception as exc:
            logger.debug("Cache load failed for %s: %s", symbol, exc)
            return None

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._cache_path(symbol))
        except Exception as exc:
            logger.debug("Cache save failed for %s: %s", symbol, exc)

    @staticmethod
    def _load_dotenv() -> None:
        """Load .env file from repo root if present."""
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            # dotenv not installed — keys must come from os.environ directly
            pass
