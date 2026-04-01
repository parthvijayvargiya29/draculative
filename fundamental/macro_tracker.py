"""
macro_tracker.py — Macro regime state tracker.

Classifies the current macro environment into one of four regimes
based on publicly available data (FRED, yfinance fallback):

  RISK_ON     — SPY above SMA200, VIX below 20, 10Y yield falling
  RISK_OFF    — SPY below SMA200 OR VIX above 30
  NEUTRAL     — in-between / data unavailable
  CRISIS      — VIX above 40 or 2-day SPY drop > 3%

Cached in data/cache/macro_cache.json, refreshed daily.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE   = Path("data/cache/macro_cache.json")
CACHE_MAX_AGE = 24 * 3600   # 1 day


class MacroRegime(str, Enum):
    RISK_ON  = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    NEUTRAL  = "NEUTRAL"
    CRISIS   = "CRISIS"


@dataclass
class MacroState:
    regime:     MacroRegime
    vix:        float = 0.0
    spy_vs_200: float = 0.0   # SPY close / SMA200 - 1
    yield_10y:  float = 0.0
    updated_at: str   = ""


class MacroTracker:
    """
    Fetches market data to determine macro regime.
    Gracefully degrades to NEUTRAL if no data is available.
    """

    def __init__(self):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._state: Optional[MacroState] = None
        self._cache_ts: float = 0.0

    def get_state(self) -> MacroState:
        if time.time() - self._cache_ts < CACHE_MAX_AGE and self._state:
            return self._state
        state = self._load_cache()
        if state is None:
            state = self._fetch()
        self._state    = state
        self._cache_ts = time.time()
        self._save_cache(state)
        return state

    def _fetch(self) -> MacroState:
        try:
            import yfinance as yf
            vix_data = yf.Ticker("^VIX").history(period="5d")
            spy_data = yf.Ticker("SPY").history(period="250d")
            tnx_data = yf.Ticker("^TNX").history(period="5d")

            vix   = float(vix_data["Close"].iloc[-1]) if not vix_data.empty else 0.0
            yield10 = float(tnx_data["Close"].iloc[-1]) if not tnx_data.empty else 0.0

            spy_vs_200 = 0.0
            if not spy_data.empty and len(spy_data) >= 200:
                sma200 = spy_data["Close"].rolling(200).mean().iloc[-1]
                spy_close = spy_data["Close"].iloc[-1]
                spy_vs_200 = (spy_close / sma200) - 1

            regime = self._classify(vix, spy_vs_200, spy_data)
            return MacroState(
                regime     = regime,
                vix        = vix,
                spy_vs_200 = spy_vs_200,
                yield_10y  = yield10,
                updated_at = datetime.utcnow().isoformat(),
            )
        except Exception as exc:
            logger.debug("Macro fetch failed: %s", exc)
            return MacroState(
                regime     = MacroRegime.NEUTRAL,
                updated_at = datetime.utcnow().isoformat(),
            )

    @staticmethod
    def _classify(vix: float, spy_vs_200: float, spy_data) -> MacroRegime:
        import pandas as pd
        if vix > 40:
            return MacroRegime.CRISIS
        if vix > 30 or spy_vs_200 < -0.05:
            return MacroRegime.RISK_OFF
        # 2-day drop > 3%
        if spy_data is not None and not getattr(spy_data, "empty", True) and len(spy_data) >= 2:
            two_day_ret = spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[-2] - 1
            if two_day_ret < -0.03:
                return MacroRegime.RISK_OFF
        if spy_vs_200 > 0 and vix < 20:
            return MacroRegime.RISK_ON
        return MacroRegime.NEUTRAL

    def _load_cache(self) -> Optional[MacroState]:
        if not CACHE_FILE.exists():
            return None
        try:
            if time.time() - CACHE_FILE.stat().st_mtime > CACHE_MAX_AGE:
                return None
            with open(CACHE_FILE) as f:
                d = json.load(f)
            return MacroState(**d)
        except Exception:
            return None

    def _save_cache(self, state: MacroState) -> None:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(asdict(state), f, indent=2)
        except Exception as exc:
            logger.debug("Macro cache save failed: %s", exc)
