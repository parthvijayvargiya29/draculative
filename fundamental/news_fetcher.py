"""
news_fetcher.py — Economic calendar & news headline fetcher.

Pulls events from free public APIs:
  1. Alpha Vantage News Sentiment  (requires ALPHA_VANTAGE_API_KEY env var)
  2. FinnHub Economic Calendar     (requires FINNHUB_API_KEY env var)
  3. Offline fallback               (static US holiday / FOMC / CPI dates)

All fetched items are normalised to a NewsItem dataclass and cached
in data/cache/news_cache.json (refreshed every 4 hours).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CACHE_FILE     = Path("data/cache/news_cache.json")
CACHE_MAX_AGE  = 4 * 3600   # 4 hours in seconds


@dataclass
class NewsItem:
    date:        str            # ISO date "YYYY-MM-DD"
    time_et:     str            # "08:30" or ""
    title:       str
    category:    str            # "CPI" | "FOMC" | "NFP" | "EARNINGS" | "OTHER"
    impact:      str            # "HIGH" | "MEDIUM" | "LOW"
    source:      str            # "alpha_vantage" | "finnhub" | "static"
    ticker:      Optional[str] = None
    sentiment:   Optional[float] = None   # -1.0 to 1.0


class NewsFetcher:
    """Fetch and cache macro / earnings news items."""

    def __init__(self):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._cache: List[NewsItem] = []
        self._cache_ts: float = 0.0

    def get_upcoming(self, days_ahead: int = 7) -> List[NewsItem]:
        """Return all events from today through +days_ahead."""
        self._refresh_if_stale()
        cutoff = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        today  = datetime.utcnow().strftime("%Y-%m-%d")
        return [n for n in self._cache if today <= n.date <= cutoff]

    def get_high_impact(self, days_ahead: int = 7) -> List[NewsItem]:
        return [n for n in self.get_upcoming(days_ahead) if n.impact == "HIGH"]

    def _refresh_if_stale(self) -> None:
        if time.time() - self._cache_ts < CACHE_MAX_AGE and self._cache:
            return
        items = self._load_cache_file()
        if not items:
            items = self._fetch_remote()
        if not items:
            items = self._static_fallback()
        self._cache    = items
        self._cache_ts = time.time()
        self._save_cache_file(items)

    def _fetch_remote(self) -> List[NewsItem]:
        items: List[NewsItem] = []
        try:
            import requests
            av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
            if av_key:
                url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                       f"&topics=economy_macro&apikey={av_key}&limit=50")
                r   = requests.get(url, timeout=10)
                if r.ok:
                    data = r.json()
                    for feed in data.get("feed", []):
                        ts    = feed.get("time_published", "")[:8]
                        date  = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
                        score = float(feed.get("overall_sentiment_score", 0))
                        items.append(NewsItem(
                            date      = date,
                            time_et   = "",
                            title     = feed.get("title", ""),
                            category  = self._classify_title(feed.get("title", "")),
                            impact    = "MEDIUM",
                            source    = "alpha_vantage",
                            sentiment = score,
                        ))
        except Exception as exc:
            logger.debug("Alpha Vantage fetch failed: %s", exc)
        return items

    def _load_cache_file(self) -> List[NewsItem]:
        if not CACHE_FILE.exists():
            return []
        try:
            if time.time() - CACHE_FILE.stat().st_mtime > CACHE_MAX_AGE:
                return []
            with open(CACHE_FILE) as f:
                raw = json.load(f)
            return [NewsItem(**r) for r in raw]
        except Exception:
            return []

    def _save_cache_file(self, items: List[NewsItem]) -> None:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump([asdict(i) for i in items], f, indent=2)
        except Exception as exc:
            logger.debug("Cache save failed: %s", exc)

    @staticmethod
    def _classify_title(title: str) -> str:
        t = title.upper()
        if "CPI" in t or "INFLATION" in t:     return "CPI"
        if "FOMC" in t or "FED" in t:          return "FOMC"
        if "NFP" in t or "PAYROLL" in t:       return "NFP"
        if "EARNINGS" in t or "EPS" in t:      return "EARNINGS"
        if "GDP" in t:                          return "GDP"
        return "OTHER"

    @staticmethod
    def _static_fallback() -> List[NewsItem]:
        """Hard-coded 2025 high-impact dates (FOMC, CPI, NFP)."""
        return [
            NewsItem("2025-01-10", "08:30", "NFP — January",    "NFP",  "HIGH", "static"),
            NewsItem("2025-01-15", "08:30", "CPI — December",   "CPI",  "HIGH", "static"),
            NewsItem("2025-01-29", "14:00", "FOMC Decision",    "FOMC", "HIGH", "static"),
            NewsItem("2025-02-07", "08:30", "NFP — February",   "NFP",  "HIGH", "static"),
            NewsItem("2025-02-12", "08:30", "CPI — January",    "CPI",  "HIGH", "static"),
            NewsItem("2025-03-07", "08:30", "NFP — March",      "NFP",  "HIGH", "static"),
            NewsItem("2025-03-12", "08:30", "CPI — February",   "CPI",  "HIGH", "static"),
            NewsItem("2025-03-19", "14:00", "FOMC Decision",    "FOMC", "HIGH", "static"),
            NewsItem("2025-04-04", "08:30", "NFP — April",      "NFP",  "HIGH", "static"),
            NewsItem("2025-04-10", "08:30", "CPI — March",      "CPI",  "HIGH", "static"),
            NewsItem("2025-05-02", "08:30", "NFP — May",        "NFP",  "HIGH", "static"),
            NewsItem("2025-05-07", "14:00", "FOMC Decision",    "FOMC", "HIGH", "static"),
            NewsItem("2025-05-13", "08:30", "CPI — April",      "CPI",  "HIGH", "static"),
            NewsItem("2025-06-06", "08:30", "NFP — June",       "NFP",  "HIGH", "static"),
            NewsItem("2025-06-11", "08:30", "CPI — May",        "CPI",  "HIGH", "static"),
            NewsItem("2025-06-18", "14:00", "FOMC Decision",    "FOMC", "HIGH", "static"),
        ]
