"""
CONCEPT: Risk Management Framework
SOURCE: 12 - ICT FOR DUMMIES | RISK MANAGEMENT EP. 11
CATEGORY: RISK_MANAGEMENT
TIMEFRAME: ANY
INSTRUMENTS: ANY (includes prop firm specific rules)
DESCRIPTION:
    The risk management framework from the ICT for Dummies series covers three
    phases for prop firm traders:

    PHASE 1 — EVAL:
    - Risk per trade: 1% of account
    - Max daily loss: 1 trade (one and done)
    - Logic: High win rate models with 1:1 to 1:3 RR. At 1%, 4 consecutive
      losses = max drawdown on a 4% max loss firm.
    - "Anybody can make money in the markets. The reason people are profitable
      is because they can KEEP that money."

    PHASE 2 — BUILDING BUFFER:
    - Risk per trade: 0.5% of account
    - Max daily loss: 1–2 trades
    - Buffer = cushion of capital = original max drawdown amount (e.g., $4k on 100k)
    - Goal: build account to original_balance + max_drawdown before requesting payout

    PHASE 3 — PAYOUT PHASE:
    - Buffer established → can increase risk back to 1%
    - Always maintain buffer to preserve drawdown wiggle room

    CONTRACT SIZING RULE:
    - Micros give the most accurate risk (1 mini = 10 micros)
    - Calculate contracts from target dollar risk / (stop_ticks * tick_value)
    - Never use minis on accounts < $150k; use micros for accuracy

    PSYCHOLOGICAL RULE:
    - "When entering a trade, always think about the POTENTIAL LOSS, not the gain"
    - "The moment you start feeling emotions, GET OFF the charts"
    - After a trade distributes (TP hit): do NOT re-enter — accumulation follows distribution

EDGE:
    Consistent percentage risk means position size adapts to volatility. A trader
    with a 50% win rate and 1:2 RR is profitable. Risk management turns an edge
    into a compounding machine.
KNOWN_LIMITATIONS:
    Prop firm rules vary. Always verify max drawdown type (trailing vs static).
    This framework is for funded/prop accounts — live accounts may use different sizing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from technical.bar_snapshot import Signal

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
EVAL_RISK_PCT       = 0.01   # 1.0% per trade in eval phase
BUFFER_RISK_PCT     = 0.005  # 0.5% per trade while building buffer
PAYOUT_RISK_PCT     = 0.01   # 1.0% per trade once buffer is built
MAX_DAILY_LOSS_EVAL = 1      # maximum trades per day in eval (stop after 1 loss)
MAX_DAILY_LOSS_BUF  = 2      # maximum trades per day building buffer
MIN_BUFFER_RATIO    = 1.0    # buffer = 1x the max drawdown amount
MICRO_CONTRACT_VALUE = 5.0   # $5 per point per micro (ES MNQ)
MINI_CONTRACT_VALUE  = 50.0  # $50 per point per mini (ES)
MIN_ACCOUNT_FOR_MINI = 150_000  # only use minis on accounts >= this


@dataclass
class RiskParameters:
    account_size:     float
    phase:            str     # "EVAL" | "BUFFER" | "PAYOUT"
    max_drawdown_pct: float   # e.g. 0.04 for 4%
    profit_target_pct: float  # e.g. 0.06 for 6%

    @property
    def risk_pct(self) -> float:
        if self.phase == "EVAL":
            return EVAL_RISK_PCT
        if self.phase == "BUFFER":
            return BUFFER_RISK_PCT
        return PAYOUT_RISK_PCT

    @property
    def dollar_risk(self) -> float:
        return self.account_size * self.risk_pct

    @property
    def max_drawdown_dollars(self) -> float:
        return self.account_size * self.max_drawdown_pct

    @property
    def profit_target_dollars(self) -> float:
        return self.account_size * self.profit_target_pct

    @property
    def buffer_target(self) -> float:
        """Account balance target before requesting payout."""
        return self.account_size + self.max_drawdown_dollars

    @property
    def max_daily_trades(self) -> int:
        if self.phase == "EVAL":
            return MAX_DAILY_LOSS_EVAL
        return MAX_DAILY_LOSS_BUF


class ICT_RiskManagement:
    """
    Runtime risk manager. Called by portfolio_manager.py before any trade entry.
    Enforces daily loss limits, position sizing, and phase-appropriate risk.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.account_size     = p.get("account_size",      100_000)
        self.phase            = p.get("phase",             "EVAL")
        self.max_drawdown_pct = p.get("max_drawdown_pct",  0.04)
        self.profit_target_pct = p.get("profit_target_pct", 0.06)
        self._params = RiskParameters(
            account_size=self.account_size,
            phase=self.phase,
            max_drawdown_pct=self.max_drawdown_pct,
            profit_target_pct=self.profit_target_pct,
        )
        self._daily_trades: int   = 0
        self._daily_pnl:    float = 0.0
        self._current_date: Optional[object] = None

    def compute_position_size(self, entry_price: float, stop_loss: float,
                               use_micros: bool = True) -> dict:
        """
        Calculates the correct number of contracts/shares for a given
        entry/stop combination and the current risk parameters.

        Returns a dict with: contracts, dollar_risk, stop_ticks, using_micros
        """
        dollar_risk = self._params.dollar_risk
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            return {"contracts": 0, "dollar_risk": 0, "stop_ticks": 0, "using_micros": use_micros}

        if use_micros:
            contracts = int(dollar_risk / (stop_distance * MICRO_CONTRACT_VALUE))
        else:
            contracts = int(dollar_risk / (stop_distance * MINI_CONTRACT_VALUE))

        actual_risk = contracts * stop_distance * (MICRO_CONTRACT_VALUE if use_micros else MINI_CONTRACT_VALUE)

        return {
            "contracts":    max(contracts, 1),
            "dollar_risk":  actual_risk,
            "stop_ticks":   stop_distance,
            "using_micros": use_micros,
        }

    def can_trade(self, timestamp, signal: Signal) -> tuple:
        """
        Returns (allowed: bool, reason: str).
        Checks daily loss limits and max trade count before entry.
        """
        today = timestamp.date() if hasattr(timestamp, "date") else None
        if today != self._current_date:
            self._daily_trades = 0
            self._daily_pnl    = 0.0
            self._current_date = today

        max_trades = self._params.max_daily_trades
        if self._daily_trades >= max_trades:
            return False, f"Daily trade limit reached ({max_trades} trades/day in {self.phase} phase)"

        max_daily_loss = self._params.max_drawdown_dollars * 0.5
        if self._daily_pnl <= -max_daily_loss:
            return False, f"Daily loss limit hit: P&L={self._daily_pnl:.2f}, limit={-max_daily_loss:.2f}"

        return True, "OK"

    def record_trade(self, pnl: float):
        """Call after each trade completes to update daily counters."""
        self._daily_trades += 1
        self._daily_pnl    += pnl
        logger.info("[RiskMgmt] Trade recorded: PnL=%.2f | Daily PnL=%.2f | Trades=%d",
                    pnl, self._daily_pnl, self._daily_trades)

    def update_account(self, new_balance: float):
        """Update account balance for accurate dollar risk calculation."""
        self.account_size = new_balance
        self._params = RiskParameters(
            account_size=new_balance,
            phase=self.phase,
            max_drawdown_pct=self.max_drawdown_pct,
            profit_target_pct=self.profit_target_pct,
        )

    def set_phase(self, phase: str):
        """Transition between EVAL → BUFFER → PAYOUT phases."""
        assert phase in ("EVAL", "BUFFER", "PAYOUT"), f"Invalid phase: {phase}"
        old = self.phase
        self.phase = phase
        self._params = RiskParameters(
            account_size=self.account_size,
            phase=phase,
            max_drawdown_pct=self.max_drawdown_pct,
            profit_target_pct=self.profit_target_pct,
        )
        logger.info("[RiskMgmt] Phase transition: %s → %s | New risk pct: %.1f%%",
                    old, phase, self._params.risk_pct * 100)

    def get_summary(self) -> dict:
        return {
            "phase":              self.phase,
            "account_size":       self.account_size,
            "risk_pct":           self._params.risk_pct,
            "dollar_risk":        self._params.dollar_risk,
            "max_drawdown":       self._params.max_drawdown_dollars,
            "profit_target":      self._params.profit_target_dollars,
            "buffer_target":      self._params.buffer_target,
            "daily_trades_used":  self._daily_trades,
            "daily_pnl":          self._daily_pnl,
        }
