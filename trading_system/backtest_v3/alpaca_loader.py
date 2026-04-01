"""
Alpaca Data Loader — Production Grade
══════════════════════════════════════
Fetches 2 years of 15-minute OHLCV data from Alpaca Markets (free tier).

Why Alpaca over yfinance:
  - yfinance hard-limits 15-min history to 60 days
  - Alpaca free tier: unlimited historical data, all resolutions, 5+ years
  - Alpaca data is institutional quality (SIP feed)

Setup (Paper Trading Keys — free, no approval required):
  1. Create free account : https://alpaca.markets
  2. Get paper API keys  : https://app.alpaca.markets/paper/dashboard/overview
     → Click "View" under Paper API Keys → copy Key ID + Secret Key
  3. Set environment vars:
       export ALPACA_API_KEY="your_key_id_here"
       export ALPACA_SECRET_KEY="your_secret_key_here"
  4. Run backtest:
       python run_v3.py --alpaca

Note: Paper keys work for BOTH paper trading AND historical data fetching.
The historical data endpoint (data.alpaca.markets) accepts paper keys.
"""

import os
import sys
import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config — set your keys here or via environment variables
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Alpaca requires APCA_API_BASE_URL for paper trading.
# Paper keys start with "PK", live keys start with "AK" or "CK".
# The data endpoint (data.alpaca.markets) is the same for both —
# only the trading endpoint differs.
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL  = "https://api.alpaca.markets"


def _is_paper_key() -> bool:
    """Paper keys from Alpaca always start with 'PK'."""
    return ALPACA_API_KEY.startswith("PK")


def _set_base_url():
    """
    alpaca-py respects the APCA_API_BASE_URL env var.
    Set it automatically based on key type so trading calls
    (order placement etc.) route to the correct endpoint.
    Historical data always uses data.alpaca.markets regardless.
    """
    if not os.getenv("APCA_API_BASE_URL"):
        url = ALPACA_PAPER_URL if _is_paper_key() else ALPACA_LIVE_URL
        os.environ["APCA_API_BASE_URL"] = url
        logger.info(f"  Base URL → {url} ({'paper' if _is_paper_key() else 'live'})")


def setup_check() -> bool:
    """
    Print Alpaca key status and setup instructions.
    Returns True if keys are configured.
    """
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        masked_key    = ALPACA_API_KEY[:4] + "*" * (len(ALPACA_API_KEY) - 4)
        masked_secret = ALPACA_SECRET_KEY[:4] + "*" * 8 + "..."
        acct_type     = "PAPER" if _is_paper_key() else "LIVE"
        base_url      = ALPACA_PAPER_URL if _is_paper_key() else ALPACA_LIVE_URL
        logger.info(f"  Alpaca {acct_type} key: {masked_key} / {masked_secret}")
        logger.info(f"  Trading endpoint  → {base_url}")
        logger.info(f"  Data endpoint     → https://data.alpaca.markets (same for all)")
        _set_base_url()
        return True
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║            ALPACA API KEYS NOT SET                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Go to:  https://alpaca.markets  → Sign up (free)        ║
║                                                              ║
║  2. Get paper keys:                                          ║
║     https://app.alpaca.markets/paper/dashboard/overview      ║
║     → Click "View" under "Paper API Keys"                   ║
║     → Copy the Key ID and Secret Key                        ║
║                                                              ║
║  3. Run in your terminal:                                    ║
║     export ALPACA_API_KEY="your_key_id"                     ║
║     export ALPACA_SECRET_KEY="your_secret_key"              ║
║     export APCA_API_BASE_URL="https://paper-api.alpaca.markets" ║
║                                                              ║
║  4. Then run:                                                ║
║     python run_v3.py --alpaca                               ║
║                                                              ║
║  Paper keys (start with PK) work for historical data.        ║
╚══════════════════════════════════════════════════════════════╝
""")
        return False

# ---------------------------------------------------------------------------
# V3.4 instrument universes
# ---------------------------------------------------------------------------

# Core symbols always fetched (included in all combo data loads)
SYMBOLS = ["NVDA", "AAPL", "MSFT", "GOOGL", "QQQ"]

# Combo A / B universe -- trend-following instruments
# Criteria: large-cap US equity / sector ETF, >$50M avg daily volume,
# listed before 2023-06-01, beta >= 0.8 (trend participants)
SYMBOLS_AB = [
    "SPY",   # S&P 500 ETF -- also needed for regime classification
    "QQQ",   # NASDAQ 100 ETF
    "IWM",   # Russell 2000 ETF
    "XLK",   # Technology Sector ETF
    "XLY",   # Consumer Discretionary Sector ETF
    "NVDA",  # NVIDIA
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "META",  # Meta Platforms
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "AMD",   # Advanced Micro Devices
    "MRVL",  # Marvell Technology
    "PANW",  # Palo Alto Networks
    "CRM",   # Salesforce
    "NOW",   # ServiceNow
    "ADBE",  # Adobe
    "CRWD",  # CrowdStrike
    "MU",    # Micron Technology
]

# Combo C universe -- mean-reversion instruments
# Criteria: instruments with demonstrated mean-reversion at BB lower + RSI2<10
# V3.4 Combo C instrument selection rationale:
# Mean reversion (BB lower + RSI2<15) works on: idiosyncratic-volatile instruments
# and defensive large-caps with strong institutional buying at oversold levels.
# DOES NOT work on: correlated market indices (SPY/QQQ/IWM) or high-beta tech
# in sustained downtrends (AAPL/MSFT 0-25% test WR in Jan-Feb 2026 selloff)
#
# Final validated set (WFE analysis on 2024-2026 data):
# Core: GLD(3.25), WMT(1.91), USMV(1.40), NVDA(1.38), AMZN(~1.5), GOOGL(~1.3)
# Added: COST(WFE 0.616 best), XOM, HD, BRK/B (defensive diversification)
SYMBOLS_C = [
    "GLD",   # SPDR Gold Shares ETF          -- validated strong mean reversion
    "WMT",   # Walmart                        -- defensive demand support
    "USMV",  # iShares MSCI Min Vol USA ETF   -- diversified min-vol basket
    "NVDA",  # Nvidia                         -- extreme volatility, strong bounces
    "AMZN",  # Amazon                         -- high-vol large-cap, institutional
    "GOOGL", # Alphabet                       -- high-vol large-cap, institutional
    "COST",  # Costco                         -- defensive consumer, best WFE in analysis
    "XOM",   # ExxonMobil                     -- energy, different sector correlation
    "HD",    # Home Depot                     -- consumer moat, institutional floor
    "MA",    # Mastercard                     -- payment network moat, institutional floor
]

# SPY is always loaded for regime classification (even if not traded)
SYMBOLS_REGIME = ["SPY"]

LOOKBACK_YRS = 2          # Target years of history
TIMEFRAME    = "15Min"    # 15-minute bars

CACHE_DIR    = Path.home() / ".backtest_v3_cache"
CACHE_DAYS   = 1          # Re-download if cache older than 1 day


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

MIN_TRADING_DAYS = 200    # Minimum clean trading days required
BARS_PER_DAY     = 26     # 9:30–16:00 ET in 15-min intervals


def _validate_df(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict OHLCV validation. Returns clean DataFrame.

    Checks:
    - High >= max(Open, Close)   — basic sanity
    - Low  <= min(Open, Close)
    - Volume > 0
    - No NaN rows
    - Drop duplicates
    - Ensure timezone-aware index (America/New_York)
    """
    orig_len = len(df)
    issues = []

    # Drop NaNs
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    nan_dropped = orig_len - len(df)
    if nan_dropped:
        issues.append(f"NaN rows dropped: {nan_dropped}")

    # Drop duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Sort chronologically
    df = df.sort_index()

    # OHLC sanity
    bad_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"]  > df[["open", "close"]].min(axis=1))
    )
    if bad_ohlc.any():
        # Clamp rather than drop — minor floating-point errors
        df.loc[bad_ohlc, "high"] = df.loc[bad_ohlc, ["open", "high", "close"]].max(axis=1)
        df.loc[bad_ohlc, "low"]  = df.loc[bad_ohlc, ["open", "low",  "close"]].min(axis=1)
        issues.append(f"OHLC clamped: {bad_ohlc.sum()}")

    # Zero volume
    zero_vol = df["volume"] <= 0
    df = df[~zero_vol]
    if zero_vol.any():
        issues.append(f"Zero-vol rows dropped: {zero_vol.sum()}")

    # Timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    # Market hours only (9:30–16:00 ET)
    df = df.between_time("09:30", "15:59")

    clean_len = len(df)
    trading_days = df.index.normalize().nunique()

    if issues:
        logger.warning(f"  {symbol}: cleaned — {'; '.join(issues)}")

    logger.info(
        f"  ✓ {symbol}: {clean_len:,} bars | {trading_days} trading days "
        f"({'OK' if trading_days >= MIN_TRADING_DAYS else 'INSUFFICIENT'})"
    )

    if trading_days < MIN_TRADING_DAYS:
        logger.warning(
            f"  ⚠ {symbol}: only {trading_days} trading days "
            f"(need {MIN_TRADING_DAYS}+ for reliable backtest)"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(symbols: List[str], years: float) -> str:
    raw = f"{sorted(symbols)}-{years}-{TIMEFRAME}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _load_cache(key: str) -> Optional[Dict[str, pd.DataFrame]]:
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    if age_hours > CACHE_DAYS * 24:
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"  Cache hit: {key} ({age_hours:.1f}h old)")
        return data
    except Exception:
        return None


def _save_cache(key: str, data: Dict[str, pd.DataFrame]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"  Cached → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_alpaca(symbols: List[str], years: float) -> Dict[str, pd.DataFrame]:
    """
    Fetch from Alpaca via alpaca-py SDK.
    Requires: pip install alpaca-py

    Paper keys work here — the data client (data.alpaca.markets)
    accepts both paper and live API keys.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError(
            "alpaca-py not installed. Run:\n"
            "  pip install alpaca-py"
        )

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError(
            "Alpaca keys not set. Run setup_check() for instructions."
        )

    # StockHistoricalDataClient uses data.alpaca.markets — works with paper keys
    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
    )

    end   = datetime.now(tz=timezone.utc)
    # Add buffer for weekends/holidays (~1.4× calendar days)
    start = end - timedelta(days=int(years * 365 * 1.45))

    logger.info(f"  Requesting {start.date()} → {end.date()} | {TIMEFRAME}")

    # Build TimeFrame — Minute15 attribute was removed in newer alpaca-py versions
    try:
        tf = TimeFrame.Minute15          # older SDK
    except AttributeError:
        tf = TimeFrame(15, TimeFrameUnit.Minute)   # newer SDK

    # Try IEX feed first (free, no SIP subscription needed)
    # IEX covers all major US stocks with slight latency — perfect for backtesting
    for feed in ["iex", "sip"]:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
                adjustment="split",
                feed=feed,
            )
            bars = client.get_stock_bars(req)
            logger.info(f"  Feed: {feed.upper()} ✓")
            break
        except Exception as e:
            err_str = str(e)
            if "forbidden" in err_str.lower() or "403" in err_str:
                logger.warning(f"  Feed {feed.upper()} not authorized — trying next...")
                continue
            raise

    result: Dict[str, pd.DataFrame] = {}
    try:
        # get_stock_bars returns a BarSet; .df gives a multi-index (symbol, timestamp) DataFrame
        all_df = bars.df.copy()
        all_df.columns = [c.lower() for c in all_df.columns]

        if isinstance(all_df.index, pd.MultiIndex):
            # Index levels: (symbol, timestamp)
            for sym in symbols:
                try:
                    df = all_df.xs(sym, level=0).copy()
                    if not df.empty:
                        result[sym] = df
                    else:
                        logger.error(f"  ✗ {sym}: empty slice")
                except KeyError:
                    logger.error(f"  ✗ {sym}: not in response")
        else:
            # Single symbol — flat index
            sym = symbols[0]
            result[sym] = all_df

    except AttributeError:
        # Older SDK or unexpected return type — try per-symbol access
        for sym in symbols:
            try:
                sym_bars = bars[sym]
                if hasattr(sym_bars, "df"):
                    df = sym_bars.df.copy()
                else:
                    df = pd.DataFrame([b.__dict__ for b in sym_bars])
                    df = df.set_index("timestamp")
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    result[sym] = df
            except (KeyError, AttributeError) as e:
                logger.error(f"  ✗ {sym}: no data returned — {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback (for offline testing / CI)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic(symbols: List[str], years: float) -> Dict[str, pd.DataFrame]:
    """
    Generate realistic GBM price series for offline testing.
    NOT for production — for development/CI only.
    """
    import numpy as np

    logger.warning("  ⚠ Using SYNTHETIC data — results not meaningful for live trading")

    rng   = np.random.default_rng(42)
    result: Dict[str, pd.DataFrame] = {}

    # EST market sessions
    sessions = pd.bdate_range(
        end=pd.Timestamp.now(tz="America/New_York").normalize(),
        periods=int(years * 252),
        freq="B",
        tz="America/New_York",
    )

    base_prices = {"NVDA": 850, "AAPL": 210, "MSFT": 420, "GOOGL": 175, "QQQ": 490}

    for sym in symbols:
        records = []
        S = base_prices.get(sym, 200.0)
        mu    = 0.0002          # per-bar drift (~50% annual)
        sigma = 0.006           # per-bar vol (~reasonable intraday)

        for day in sessions:
            open_time  = day.replace(hour=9,  minute=30)
            close_time = day.replace(hour=15, minute=45)
            bar_times  = pd.date_range(open_time, close_time, freq="15min", tz="America/New_York")

            for ts in bar_times:
                ret = rng.normal(mu, sigma)
                o   = S
                c   = S * (1 + ret)
                h   = max(o, c) * (1 + abs(rng.normal(0, sigma * 0.3)))
                l   = min(o, c) * (1 - abs(rng.normal(0, sigma * 0.3)))
                vol = abs(rng.normal(2_000_000, 500_000))
                records.append({"timestamp": ts, "open": o, "high": h,
                                 "low": l, "close": c, "volume": vol})
                S = c

        df = pd.DataFrame(records).set_index("timestamp")
        result[sym] = df

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    symbols:       List[str] = None,
    years:         float     = LOOKBACK_YRS,
    use_cache:     bool      = True,
    force_synth:   bool      = False,
    universe:      str       = "legacy",
) -> Dict[str, pd.DataFrame]:
    """
    Load and validate 15-min OHLCV data.

    universe:
      "legacy"    — original 5 symbols (NVDA/AAPL/MSFT/GOOGL/QQQ)  [default]
      "AB"        — Combo A/B universe (20 trend-following instruments + SPY)
      "C"         — Combo C universe (11 low-beta mean-reversion instruments)
      "all"       — union of AB + C + SPY (for combined runs)
      "regime"    — SPY only (for regime classification)

    Explicit `symbols` list overrides `universe`.

    Returns:
        dict {symbol: DataFrame} — index is tz-aware America/New_York,
        columns: open, high, low, close, volume

    Falls back to synthetic data if Alpaca keys not set.
    """
    if symbols is None:
        if universe == "AB":
            symbols = sorted(set(SYMBOLS_AB + SYMBOLS_REGIME))
        elif universe == "C":
            symbols = sorted(set(SYMBOLS_C + SYMBOLS_REGIME + ["QQQ"]))
        elif universe == "all":
            symbols = sorted(set(SYMBOLS_AB + SYMBOLS_C + SYMBOLS_REGIME))
        elif universe == "regime":
            symbols = SYMBOLS_REGIME
        else:  # "legacy"
            symbols = SYMBOLS

    logger.info(f"\nLoading {len(symbols)} symbols | {years}yr | {TIMEFRAME} | universe={universe}")

    # Cache
    key = _cache_key(symbols, years)
    if use_cache and not force_synth:
        cached = _load_cache(key)
        if cached is not None:
            logger.info(f"  ✓ Loaded from cache")
            return cached

    # Fetch
    if force_synth or not ALPACA_API_KEY:
        raw = _generate_synthetic(symbols, years)
    else:
        setup_check()   # Print masked key confirmation
        try:
            raw = _fetch_alpaca(symbols, years)
        except Exception as e:
            logger.error(f"  Alpaca fetch failed: {e}")
            logger.warning("  Falling back to synthetic data for testing")
            raw = _generate_synthetic(symbols, years)

    # Validate each symbol
    clean: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym not in raw or raw[sym].empty:
            logger.error(f"  ✗ {sym}: empty — skipping")
            continue
        df = _validate_df(sym, raw[sym])
        if not df.empty:
            clean[sym] = df

    if not clean:
        raise RuntimeError("No data loaded for any symbol")

    # Summary
    total_bars = sum(len(df) for df in clean.values())
    date_ranges = {s: (df.index.min().date(), df.index.max().date()) for s, df in clean.items()}
    logger.info(f"\n  Summary: {len(clean)} symbols | {total_bars:,} total bars")
    for sym, (start, end) in date_ranges.items():
        logger.info(f"    {sym:6s}  {start} → {end}  ({len(clean[sym]):,} bars)")

    if use_cache:
        _save_cache(key, clean)

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    has_keys = setup_check()
    if not has_keys:
        print("Loading SYNTHETIC data for testing...\n")

    data = load_data()
    for sym, df in data.items():
        print(f"  {sym}: {len(df):,} bars  |  first={df.index[0]}  |  last={df.index[-1]}")
