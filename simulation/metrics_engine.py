"""
metrics_engine.py — All performance metric computations.

Takes a list of completed trades and returns the full SimulationResult metrics.
All metrics match the required format from Section 6.2.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from technical.bar_snapshot import SimulationResult, WalkForwardResult, RegimeMetrics, ApprovalStatus

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05  # 5% annual risk-free rate for Sharpe/Sortino
TRADING_DAYS   = 252


@dataclass
class TradeRecord:
    concept:      str
    symbol:       str
    direction:    str
    entry_time:   datetime
    exit_time:    datetime
    entry_price:  float
    exit_price:   float
    stop_loss:    float
    take_profit:  float
    pnl:          float       # dollar P&L after commission
    pnl_pct:      float       # return % on risk capital
    regime:       str = "UNKNOWN"
    hold_bars:    int = 0
    commission:   float = 0.0


class MetricsEngine:
    """
    Computes all required metrics from a list of TradeRecords.
    """

    @staticmethod
    def compute(trades: List[TradeRecord], concept_name: str,
                data_period: dict, universe: List[str],
                initial_equity: float = 100_000) -> SimulationResult:
        """
        Main entry point. Returns a fully populated SimulationResult.
        """
        result = SimulationResult(
            concept     = concept_name,
            run_date    = datetime.utcnow().strftime("%Y-%m-%d"),
            data_period = data_period,
            universe    = universe,
        )

        if not trades:
            result.approval_status  = ApprovalStatus.REJECTED
            result.rejection_reason = "No trades generated"
            return result

        # ── Basic trade stats ─────────────────────────────────────────────
        result.total_trades = len(trades)
        wins  = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.win_rate = len(wins) / len(trades)

        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        result.avg_hold_bars = np.mean([t.hold_bars for t in trades])

        # ── Equity curve ──────────────────────────────────────────────────
        equity = MetricsEngine._build_equity_curve(trades, initial_equity)
        daily_returns = equity["equity"].pct_change().dropna()

        result.sharpe_annualized = MetricsEngine._sharpe(daily_returns)
        result.sortino_ratio     = MetricsEngine._sortino(daily_returns)
        result.max_drawdown_pct  = MetricsEngine._max_drawdown(equity["equity"])
        result.calmar_ratio      = (
            MetricsEngine._annual_return(equity["equity"]) / abs(result.max_drawdown_pct)
            if result.max_drawdown_pct != 0 else 0.0
        )

        # ── Trades per month ──────────────────────────────────────────────
        if trades:
            first_date = trades[0].entry_time
            last_date  = trades[-1].exit_time
            months = max((last_date - first_date).days / 30.44, 1)
            result.trades_per_month = len(trades) / months

        # ── Regime attribution ────────────────────────────────────────────
        trending  = [t for t in trades if t.regime == "TRENDING"]
        corrective = [t for t in trades if t.regime == "CORRECTIVE"]
        result.regime_trending  = MetricsEngine._regime_metrics(trending)
        result.regime_corrective = MetricsEngine._regime_metrics(corrective)

        # ── Monthly breakdown ─────────────────────────────────────────────
        result.monthly_breakdown = MetricsEngine._monthly_breakdown(trades)

        return result

    @staticmethod
    def _build_equity_curve(trades: List[TradeRecord],
                             initial: float) -> pd.DataFrame:
        rows = [{"time": trades[0].entry_time, "equity": initial}]
        equity = initial
        for t in sorted(trades, key=lambda x: x.exit_time):
            equity += t.pnl
            rows.append({"time": t.exit_time, "equity": equity})
        df = pd.DataFrame(rows).set_index("time")
        df = df[~df.index.duplicated(keep="last")]
        return df

    @staticmethod
    def _sharpe(daily_returns: pd.Series) -> float:
        if daily_returns.std() == 0:
            return 0.0
        excess = daily_returns - RISK_FREE_RATE / TRADING_DAYS
        return float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS))

    @staticmethod
    def _sortino(daily_returns: pd.Series) -> float:
        downside = daily_returns[daily_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        excess = daily_returns.mean() - RISK_FREE_RATE / TRADING_DAYS
        return float(excess / downside.std() * math.sqrt(TRADING_DAYS))

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        return float(drawdown.min())

    @staticmethod
    def _annual_return(equity: pd.Series) -> float:
        if len(equity) < 2 or equity.iloc[0] == 0:
            return 0.0
        total = equity.iloc[-1] / equity.iloc[0] - 1
        first = equity.index[0]
        last  = equity.index[-1]
        years = max((last - first).days / 365.25, 0.1)
        return float((1 + total) ** (1 / years) - 1)

    @staticmethod
    def _regime_metrics(trades: List[TradeRecord]) -> RegimeMetrics:
        if not trades:
            return RegimeMetrics()
        wins = [t for t in trades if t.pnl > 0]
        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        return RegimeMetrics(
            trades=len(trades),
            win_rate=len(wins) / len(trades),
            pf=gp / gl if gl > 0 else float("inf"),
        )

    @staticmethod
    def _monthly_breakdown(trades: List[TradeRecord]) -> list:
        if not trades:
            return []
        rows = {}
        for t in trades:
            key = t.exit_time.strftime("%Y-%m")
            if key not in rows:
                rows[key] = {"month": key, "trades": 0, "wins": 0, "pnl": 0.0}
            rows[key]["trades"] += 1
            if t.pnl > 0:
                rows[key]["wins"] += 1
            rows[key]["pnl"] += t.pnl
        return sorted(rows.values(), key=lambda x: x["month"])
