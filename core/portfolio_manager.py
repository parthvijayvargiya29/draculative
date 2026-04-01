"""
portfolio_manager.py — Position Sizing & Allocation Guardrails.

Enforces three-tiered limits from Section 3.3:
  - Per instrument : ≤ 6% of total equity
  - Per strategy   : ≤ 20% of total equity
  - Total portfolio: ≤ 40% of total equity (max concurrent exposure)

Correlation-aware reduction:
  If two instruments have rolling 20-day correlation ≥ 0.80, the second
  position is sized at 50% of what it would otherwise be.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Allocation limits ──────────────────────────────────────────────────────
MAX_INSTRUMENT_PCT = 0.06   # 6% per instrument
MAX_STRATEGY_PCT   = 0.20   # 20% per strategy (concept group)
MAX_TOTAL_PCT      = 0.40   # 40% total portfolio
CORR_THRESHOLD     = 0.80   # above this → cut size 50%
CORR_LOOKBACK      = 20     # trading days


@dataclass
class AllocationRequest:
    symbol:        str
    strategy_name: str
    entry_price:   float
    stop_loss:     float
    equity:        float   # current total equity
    direction:     str     # "long" | "short"


@dataclass
class AllocationResult:
    approved:      bool
    shares:        int
    dollar_risk:   float
    reason:        str = ""


class PortfolioManager:
    """
    Stateful manager — tracks open positions and capital deployed.
    """

    def __init__(self, initial_equity: float = 100_000):
        self._equity       = initial_equity
        # {symbol: dollar_value_open}
        self._instrument_exposure: Dict[str, float] = {}
        # {strategy_name: dollar_value_open}
        self._strategy_exposure:   Dict[str, float] = {}
        # {symbol: list of recent daily returns} for correlation
        self._return_history: Dict[str, List[float]] = {}

    @property
    def total_exposure(self) -> float:
        return sum(self._instrument_exposure.values())

    def request_allocation(self, req: AllocationRequest) -> AllocationResult:
        """
        Returns an AllocationResult indicating whether the trade is approved
        and how many shares to buy/sell.
        """
        equity = req.equity
        instr_used  = self._instrument_exposure.get(req.symbol, 0.0)
        strat_used  = self._strategy_exposure.get(req.strategy_name, 0.0)
        total_used  = self.total_exposure

        # ── Hard limits ──────────────────────────────────────────────────
        if instr_used / equity >= MAX_INSTRUMENT_PCT:
            return AllocationResult(False, 0, 0.0, f"{req.symbol} instrument limit reached")
        if strat_used / equity >= MAX_STRATEGY_PCT:
            return AllocationResult(False, 0, 0.0, f"{req.strategy_name} strategy limit reached")
        if total_used / equity >= MAX_TOTAL_PCT:
            return AllocationResult(False, 0, 0.0, "Total portfolio limit reached")

        # ── Base position size (1% risk) ─────────────────────────────────
        risk_dollars     = equity * 0.01
        risk_per_share   = abs(req.entry_price - req.stop_loss)
        if risk_per_share < 0.0001:
            return AllocationResult(False, 0, 0.0, "Stop too close to entry")

        shares = int(risk_dollars / risk_per_share)

        # ── Instrument exposure cap ───────────────────────────────────────
        max_instrument_dollars = equity * MAX_INSTRUMENT_PCT - instr_used
        max_by_instrument = int(max_instrument_dollars / req.entry_price)
        shares = min(shares, max(0, max_by_instrument))

        # ── Correlation haircut ──────────────────────────────────────────
        if self._is_correlated(req.symbol):
            shares = int(shares * 0.50)
            logger.debug("Correlation haircut applied to %s — 50%% size", req.symbol)

        if shares == 0:
            return AllocationResult(False, 0, 0.0, "Position size computed as 0 after limits")

        dollar_risk = risk_per_share * shares
        return AllocationResult(True, shares, dollar_risk)

    def open_position(self, symbol: str, strategy_name: str,
                      shares: int, price: float) -> None:
        value = abs(shares) * price
        self._instrument_exposure[symbol]     = self._instrument_exposure.get(symbol, 0.0) + value
        self._strategy_exposure[strategy_name] = self._strategy_exposure.get(strategy_name, 0.0) + value

    def close_position(self, symbol: str, strategy_name: str,
                       shares: int, price: float) -> None:
        value = abs(shares) * price
        self._instrument_exposure[symbol]      = max(0.0, self._instrument_exposure.get(symbol, 0.0) - value)
        self._strategy_exposure[strategy_name] = max(0.0, self._strategy_exposure.get(strategy_name, 0.0) - value)

    def update_equity(self, new_equity: float) -> None:
        self._equity = new_equity

    def record_return(self, symbol: str, daily_return: float) -> None:
        """Feed daily returns for correlation tracking."""
        hist = self._return_history.setdefault(symbol, [])
        hist.append(daily_return)
        if len(hist) > CORR_LOOKBACK * 2:
            self._return_history[symbol] = hist[-CORR_LOOKBACK * 2:]

    def _is_correlated(self, symbol: str) -> bool:
        """
        Returns True if this symbol has rolling correlation ≥ CORR_THRESHOLD
        with ANY other currently-held instrument.
        """
        held = [s for s, v in self._instrument_exposure.items()
                if v > 0 and s != symbol]
        if not held:
            return False

        hist_a = self._return_history.get(symbol, [])
        if len(hist_a) < CORR_LOOKBACK:
            return False

        for other in held:
            hist_b = self._return_history.get(other, [])
            if len(hist_b) < CORR_LOOKBACK:
                continue
            n   = min(len(hist_a), len(hist_b), CORR_LOOKBACK)
            arr = np.corrcoef(hist_a[-n:], hist_b[-n:])
            if abs(arr[0, 1]) >= CORR_THRESHOLD:
                return True
        return False

    def summary(self) -> dict:
        return {
            "equity":              self._equity,
            "total_exposure":      self.total_exposure,
            "total_exposure_pct":  self.total_exposure / self._equity if self._equity else 0,
            "instrument_exposure": dict(self._instrument_exposure),
            "strategy_exposure":   dict(self._strategy_exposure),
        }
