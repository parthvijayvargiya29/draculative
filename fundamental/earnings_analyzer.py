"""
earnings_analyzer.py — Upcoming earnings event tracker.

Scans the watchlist for earnings dates within the next N days.
Flags symbols with earnings as HIGH_RISK for position sizing purposes.
Data source: yfinance calendar (free, no key required).
Cached in data/cache/earnings_cache.json (refreshed daily).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_FILE    = Path("data/cache/earnings_cache.json")
CACHE_MAX_AGE = 24 * 3600


@dataclass
class EarningsEvent:
    symbol:           str
    earnings_date:    str       # "YYYY-MM-DD"
    eps_estimate:     Optional[float] = None
    revenue_estimate: Optional[float] = None
    surprise_pct:     Optional[float] = None  # populated after event
    is_upcoming:      bool = True


class EarningsAnalyzer:
    """
    Fetch upcoming earnings for a list of symbols.
    """

    def __init__(self, watchlist: List[str] = None):
        self.watchlist = watchlist or []
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._cache:    Dict[str, EarningsEvent] = {}
        self._cache_ts: float = 0.0

    def refresh(self, symbols: List[str] = None) -> None:
        syms = symbols or self.watchlist
        if not syms:
            return
        cached = self._load_cache()
        if cached:
            self._cache = cached
            return
        for sym in syms:
            evt = self._fetch_symbol(sym)
            if evt:
                self._cache[sym] = evt
        self._cache_ts = time.time()
        self._save_cache()

    def is_near_earnings(self, symbol: str, days: int = 5) -> bool:
        """Returns True if symbol has earnings within `days` trading days."""
        if symbol not in self._cache:
            self.refresh([symbol])
        evt = self._cache.get(symbol)
        if evt is None:
            return False
        try:
            ed = datetime.strptime(evt.earnings_date, "%Y-%m-%d")
            return 0 <= (ed - datetime.utcnow()).days <= days
        except ValueError:
            return False

    def get_event(self, symbol: str) -> Optional[EarningsEvent]:
        return self._cache.get(symbol)

    def upcoming_events(self, days: int = 7) -> List[EarningsEvent]:
        cutoff = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
        today  = datetime.utcnow().strftime("%Y-%m-%d")
        return [e for e in self._cache.values()
                if today <= e.earnings_date <= cutoff]

    def _fetch_symbol(self, symbol: str) -> Optional[EarningsEvent]:
        try:
            import yfinance as yf
            tk  = yf.Ticker(symbol)
            cal = tk.calendar
            if cal is None or cal.empty if hasattr(cal, "empty") else not cal:
                return None
            # yfinance calendar columns vary — try both dict and DataFrame
            if hasattr(cal, "columns"):
                date_col = cal.get("Earnings Date", cal.get("earnings_date", None))
                if date_col is not None and len(date_col):
                    ed = str(date_col.iloc[0])[:10]
                    return EarningsEvent(symbol=symbol, earnings_date=ed)
            elif isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date", cal.get("earningsDate", [None]))[0]
                if ed_raw:
                    ed = str(ed_raw)[:10]
                    return EarningsEvent(symbol=symbol, earnings_date=ed)
        except Exception as exc:
            logger.debug("Earnings fetch for %s failed: %s", symbol, exc)
        return None

    def _load_cache(self) -> Optional[Dict[str, EarningsEvent]]:
        if not CACHE_FILE.exists():
            return None
        try:
            if time.time() - CACHE_FILE.stat().st_mtime > CACHE_MAX_AGE:
                return None
            with open(CACHE_FILE) as f:
                raw = json.load(f)
            return {k: EarningsEvent(**v) for k, v in raw.items()}
        except Exception:
            return None

    def _save_cache(self) -> None:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({k: asdict(v) for k, v in self._cache.items()}, f, indent=2)
        except Exception as exc:
            logger.debug("Earnings cache save failed: %s", exc)
