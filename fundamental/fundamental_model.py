"""
fundamental_model.py — Fundamental Alpha Score Aggregator.

Combines:
  1. Macro regime (MacroTracker)
  2. Upcoming high-impact news (NewsFetcher)
  3. Earnings proximity (EarningsAnalyzer)

…into a single FundamentalBias per symbol.

FundamentalBias:
  score      : float in [-1, +1]  (−1 full bearish, +1 full bullish, 0 neutral)
  trade_ok   : bool (False = do NOT trade this symbol right now)
  reason     : str

Rules:
  - CRISIS macro → trade_ok = False for everything
  - RISK_OFF      → score penalty −0.3, reduce size flag
  - Earnings within 3 days → trade_ok = False
  - High-impact news today  → reduce to 50% size
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from fundamental.macro_tracker import MacroTracker, MacroRegime
from fundamental.news_fetcher import NewsFetcher
from fundamental.earnings_analyzer import EarningsAnalyzer

logger = logging.getLogger(__name__)

EARNINGS_BLACKOUT_DAYS = 3
SIZE_REDUCTION_IMPACT  = 0.50


@dataclass
class FundamentalBias:
    symbol:         str
    score:          float       # −1.0 to +1.0
    trade_ok:       bool
    size_scale:     float = 1.0  # multiply intended shares by this
    macro_regime:   str   = "NEUTRAL"
    reasons:        List[str] = field(default_factory=list)


class FundamentalModel:
    """
    Instantiate once per session; call .assess(symbol) per bar.
    Refreshes underlying data at configured intervals.
    """

    def __init__(self, watchlist: List[str] = None):
        self.macro     = MacroTracker()
        self.news      = NewsFetcher()
        self.earnings  = EarningsAnalyzer(watchlist=watchlist or [])

    def assess(self, symbol: str) -> FundamentalBias:
        macro_state = self.macro.get_state()
        upcoming    = self.news.get_high_impact(days_ahead=1)
        near_earn   = self.earnings.is_near_earnings(symbol, days=EARNINGS_BLACKOUT_DAYS)

        reasons: List[str] = []
        score       = 0.0
        trade_ok    = True
        size_scale  = 1.0

        # ── Macro regime ──────────────────────────────────────────────────
        if macro_state.regime == MacroRegime.CRISIS:
            trade_ok = False
            reasons.append("MACRO CRISIS — all trading halted")
        elif macro_state.regime == MacroRegime.RISK_OFF:
            score -= 0.3
            size_scale *= 0.75
            reasons.append("RISK_OFF macro — size reduced 25%")
        elif macro_state.regime == MacroRegime.RISK_ON:
            score += 0.2
            reasons.append("RISK_ON macro — slight bullish tilt")

        # ── High-impact news today ────────────────────────────────────────
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_events = [n for n in upcoming if n.date == today and n.impact == "HIGH"]
        if today_events:
            event_names = ", ".join(e.title for e in today_events)
            size_scale *= SIZE_REDUCTION_IMPACT
            reasons.append(f"High-impact event today: {event_names} — size 50%")

        # ── Earnings proximity ────────────────────────────────────────────
        if near_earn:
            trade_ok = False
            reasons.append(f"{symbol} earnings within {EARNINGS_BLACKOUT_DAYS} days — no trade")

        # Clip score to [-1, +1]
        score = max(-1.0, min(1.0, score))

        return FundamentalBias(
            symbol       = symbol,
            score        = score,
            trade_ok     = trade_ok,
            size_scale   = round(size_scale, 2),
            macro_regime = macro_state.regime.value,
            reasons      = reasons,
        )

    def assess_watchlist(self, symbols: List[str]) -> Dict[str, FundamentalBias]:
        return {s: self.assess(s) for s in symbols}
