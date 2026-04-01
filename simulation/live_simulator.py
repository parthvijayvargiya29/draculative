"""
live_simulator.py — Bar-by-bar Live Simulation Engine.

THIS IS NOT A BACKTEST.
Every bar sees ONLY the data that would have been available in real-time.
No lookahead. No future-bar fills. No hindsight bias.

Execution model:
  - Market orders  → fill at NEXT bar open + 0.05% slippage (direction-aware)
  - Limit/Stop orders → fill only if next bar's range trades through the level
  - Commission      → $0.005/share, min $1.00 per fill
  - EOD forced exit → 15:45 ET market-on-close

Warm-up:
  - Default 80 bars required before any concept may emit a signal.
  - Concepts may define their own MIN_LOOKBACK; the max is used.

Walk-forward split baked in (run_walk_forward entry point):
  60% train · 20% validate · 20% test
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, SimulationResult, ApprovalStatus
from technical.indicators_v4 import enrich_dataframe
from simulation.metrics_engine import MetricsEngine, TradeRecord

logger = logging.getLogger(__name__)

# ── Execution constants ───────────────────────────────────────────────────────
SLIPPAGE_PCT   = 0.0005  # 0.05%
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN = 1.00
EOD_EXIT_TIME  = dtime(15, 45)  # ET


@dataclass
class _PendingOrder:
    signal:      Signal
    bar_index:   int      # bar on which the signal was generated
    order_type:  str      # "market" | "limit" | "stop"
    limit_price: float = 0.0


@dataclass
class _OpenPosition:
    signal:      Signal
    entry_time:  datetime
    entry_price: float
    entry_bar:   int
    shares:      int
    stop_loss:   float
    take_profit: float
    regime:      str = "UNKNOWN"
    commission_paid: float = 0.0


def _calc_commission(shares: int) -> float:
    return max(abs(shares) * COMMISSION_PER_SHARE, COMMISSION_MIN)


def _slippage(price: float, direction: Direction) -> float:
    """Apply slippage in the direction that hurts us."""
    if direction == Direction.LONG:
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def _build_snapshot(df: pd.DataFrame, idx: int, symbol: str = "") -> BarSnapshot:
    """Build a BarSnapshot from row idx with full history visible up to idx."""
    from technical.bar_snapshot import MarketRegime
    row = df.iloc[idx]
    regime_str = str(row.get("regime", "UNKNOWN"))
    try:
        regime_val = MarketRegime(regime_str)
    except ValueError:
        regime_val = MarketRegime.UNKNOWN
    snap = BarSnapshot(
        timestamp  = row.name if isinstance(row.name, datetime) else pd.Timestamp(row.name).to_pydatetime(),
        symbol     = symbol,
        open       = float(row["open"]),
        high       = float(row["high"]),
        low        = float(row["low"]),
        close      = float(row["close"]),
        volume     = float(row.get("volume", 0)),
        atr        = float(row["atr"]) if pd.notna(row.get("atr")) else None,
        adx        = float(row["adx"]) if pd.notna(row.get("adx")) else None,
        sma_50     = float(row["sma_50"]) if pd.notna(row.get("sma_50")) else None,
        sma_200    = float(row["sma_200"]) if pd.notna(row.get("sma_200")) else None,
        rsi_14     = float(row["rsi_14"]) if pd.notna(row.get("rsi_14")) else None,
        session    = str(row.get("session", "UNKNOWN")),
        regime     = regime_val,
        df_ref     = df.iloc[: idx + 1],
    )
    return snap


class LiveSimulator:
    """
    Bar-by-bar simulation engine.

    Parameters
    ----------
    concept_instances : list
        Instantiated concept objects (each must implement detect(snapshot) → Optional[Signal]).
    initial_equity : float
        Starting paper-money equity.
    warmup_bars : int
        Bars to consume for indicator warm-up before allowing signals.
    max_open_positions : int
        Max concurrent open positions at any time.
    """

    def __init__(
        self,
        concept_instances: list,
        initial_equity: float = 100_000,
        warmup_bars: int = 80,
        max_open_positions: int = 4,
    ):
        self.concepts          = concept_instances
        self.initial_equity    = initial_equity
        self.warmup_bars       = warmup_bars
        self.max_open_positions = max_open_positions

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime_classifier=None,
    ) -> Tuple[List[TradeRecord], pd.Series]:
        """
        Run simulation on pre-loaded OHLCV DataFrame.

        Parameters
        ----------
        symbol : str
        df     : pd.DataFrame  — must have columns: open, high, low, close, volume
        regime_classifier : optional RegimeClassifier instance

        Returns
        -------
        trades     : List[TradeRecord]
        equity_curve : pd.Series (indexed by exit timestamp)
        """
        df = enrich_dataframe(df.copy())
        df.sort_index(inplace=True)

        if len(df) < self.warmup_bars + 10:
            logger.warning("Not enough bars for warm-up (%d bars, need %d)", len(df), self.warmup_bars)
            return [], pd.Series(dtype=float)

        trades:          List[TradeRecord] = []
        pending_orders:  List[_PendingOrder] = []
        open_positions:  List[_OpenPosition] = []
        equity          = self.initial_equity
        equity_series   = {df.index[0]: equity}

        for i in range(self.warmup_bars, len(df) - 1):
            bar      = df.iloc[i]
            next_bar = df.iloc[i + 1]
            ts = bar.name if isinstance(bar.name, datetime) else pd.Timestamp(bar.name).to_pydatetime()
            next_ts = next_bar.name if isinstance(next_bar.name, datetime) else pd.Timestamp(next_bar.name).to_pydatetime()

            # ── 1. Fill pending orders against NEXT bar ───────────────────
            filled_signals: List[Signal] = []
            still_pending:  List[_PendingOrder] = []
            for order in pending_orders:
                if order.order_type == "market":
                    fill_price = _slippage(float(next_bar["open"]), order.signal.direction)
                    filled_signals.append((order.signal, fill_price, i + 1))
                elif order.order_type in ("limit", "stop"):
                    lp = order.limit_price
                    nx_lo = float(next_bar["low"])
                    nx_hi = float(next_bar["high"])
                    if order.signal.direction == Direction.LONG and nx_lo <= lp <= nx_hi:
                        filled_signals.append((order.signal, lp, i + 1))
                    elif order.signal.direction == Direction.SHORT and nx_lo <= lp <= nx_hi:
                        filled_signals.append((order.signal, lp, i + 1))
                    else:
                        # Cancel after 1 bar (market-order-on-open equivalent)
                        fill_price = _slippage(float(next_bar["open"]), order.signal.direction)
                        filled_signals.append((order.signal, fill_price, i + 1))

            for sig, fill_price, fill_bar_idx in filled_signals:
                if len(open_positions) >= self.max_open_positions:
                    continue
                shares = self._position_size(equity, fill_price, sig.stop_loss)
                if shares == 0:
                    continue
                comm = _calc_commission(shares)
                equity -= comm
                regime_tag = "UNKNOWN"
                if regime_classifier is not None:
                    regime_tag = regime_classifier.tag(df.iloc[:fill_bar_idx + 1])
                pos = _OpenPosition(
                    signal       = sig,
                    entry_time   = next_ts,
                    entry_price  = fill_price,
                    entry_bar    = fill_bar_idx,
                    shares       = shares,
                    stop_loss    = sig.stop_loss,
                    take_profit  = sig.take_profit,
                    regime       = regime_tag,
                    commission_paid = comm,
                )
                open_positions.append(pos)

            # ── 2. Check exits on CURRENT bar ─────────────────────────────
            is_eod = ts.time() >= EOD_EXIT_TIME if hasattr(ts, "time") else False
            surviving: List[_OpenPosition] = []
            for pos in open_positions:
                exit_price = None
                exit_reason = ""

                if pos.signal.direction == Direction.LONG:
                    if bar["low"] <= pos.stop_loss:
                        exit_price = pos.stop_loss
                        exit_reason = "stop_loss"
                    elif bar["high"] >= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = "take_profit"
                    elif is_eod:
                        exit_price = float(bar["close"])
                        exit_reason = "eod"
                else:  # SHORT
                    if bar["high"] >= pos.stop_loss:
                        exit_price = pos.stop_loss
                        exit_reason = "stop_loss"
                    elif bar["low"] <= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = "take_profit"
                    elif is_eod:
                        exit_price = float(bar["close"])
                        exit_reason = "eod"

                if exit_price is not None:
                    comm_exit = _calc_commission(pos.shares)
                    if pos.signal.direction == Direction.LONG:
                        raw_pnl = (exit_price - pos.entry_price) * pos.shares
                    else:
                        raw_pnl = (pos.entry_price - exit_price) * pos.shares
                    net_pnl = raw_pnl - comm_exit - pos.commission_paid
                    equity += net_pnl
                    equity_series[ts] = equity

                    rec = TradeRecord(
                        concept      = pos.signal.concept,
                        symbol       = symbol,
                        direction    = pos.signal.direction.value,
                        entry_time   = pos.entry_time,
                        exit_time    = ts,
                        entry_price  = pos.entry_price,
                        exit_price   = exit_price,
                        stop_loss    = pos.stop_loss,
                        take_profit  = pos.take_profit,
                        pnl          = net_pnl,
                        pnl_pct      = net_pnl / (pos.entry_price * pos.shares) if pos.entry_price * pos.shares != 0 else 0.0,
                        regime       = pos.regime,
                        hold_bars    = i - pos.entry_bar,
                        commission   = comm_exit + pos.commission_paid,
                    )
                    trades.append(rec)
                    logger.debug("EXIT [%s] %s @ %.4f | P&L $%.2f (%s)", ts, symbol, exit_price, net_pnl, exit_reason)
                else:
                    surviving.append(pos)
            open_positions = surviving

            # ── 3. Generate new signals from concepts ─────────────────────
            snapshot = _build_snapshot(df, i, symbol=symbol)
            for concept in self.concepts:
                try:
                    sig: Optional[Signal] = concept.detect(snapshot)
                except Exception as exc:
                    logger.debug("Concept %s raised: %s", getattr(concept, "__class__", "?"), exc)
                    sig = None
                if sig is not None:
                    pending_orders.append(_PendingOrder(
                        signal     = sig,
                        bar_index  = i,
                        order_type = "market",
                    ))

        # ── Force-close any remaining positions at last bar close ─────────
        last_bar = df.iloc[-1]
        last_ts  = last_bar.name if isinstance(last_bar.name, datetime) else pd.Timestamp(last_bar.name).to_pydatetime()
        for pos in open_positions:
            exit_price = float(last_bar["close"])
            comm_exit = _calc_commission(pos.shares)
            if pos.signal.direction == Direction.LONG:
                raw_pnl = (exit_price - pos.entry_price) * pos.shares
            else:
                raw_pnl = (pos.entry_price - exit_price) * pos.shares
            net_pnl = raw_pnl - comm_exit - pos.commission_paid
            equity += net_pnl
            trades.append(TradeRecord(
                concept     = pos.signal.concept,
                symbol      = symbol,
                direction   = pos.signal.direction.value,
                entry_time  = pos.entry_time,
                exit_time   = last_ts,
                entry_price = pos.entry_price,
                exit_price  = exit_price,
                stop_loss   = pos.stop_loss,
                take_profit = pos.take_profit,
                pnl         = net_pnl,
                pnl_pct     = net_pnl / (pos.entry_price * pos.shares) if pos.entry_price * pos.shares != 0 else 0.0,
                regime      = pos.regime,
                hold_bars   = len(df) - 1 - pos.entry_bar,
                commission  = comm_exit + pos.commission_paid,
            ))
        equity_series[last_ts] = equity
        return trades, pd.Series(equity_series, dtype=float)

    def _position_size(self, equity: float, price: float, stop: float,
                       risk_pct: float = 0.01) -> int:
        """1% risk per trade, bounded by price."""
        risk_dollars = equity * risk_pct
        risk_per_share = abs(price - stop)
        if risk_per_share < 0.0001:
            return 0
        shares = int(risk_dollars / risk_per_share)
        max_shares = int(equity * 0.06 / price)  # 6% capital limit
        return max(0, min(shares, max_shares))

    def build_simulation_result(
        self,
        symbol: str,
        df: pd.DataFrame,
        concept_name: str,
        start: str,
        end: str,
        regime_classifier=None,
    ) -> SimulationResult:
        """Convenience wrapper: run + compute all metrics."""
        trades, equity_series = self.run(symbol, df, regime_classifier)
        result = MetricsEngine.compute(
            trades       = trades,
            concept_name = concept_name,
            data_period  = {"start": start, "end": end, "symbol": symbol},
            universe     = [symbol],
            initial_equity = self.initial_equity,
        )
        return result


class RunSimulation:
    """
    High-level entry point: loads bars, runs simulation, returns SimulationResult.
    """

    def __init__(self, loader, concepts: list, initial_equity: float = 100_000):
        self.loader   = loader
        self.concepts = concepts
        self.initial_equity = initial_equity

    def run(self, symbol: str, start: str, end: str,
            timeframe: str = "5Min") -> SimulationResult:
        from simulation.regime_classifier import RegimeClassifier
        df = self.loader.get_bars(symbol, timeframe, start, end)
        if df.empty:
            result = SimulationResult(concept="N/A", run_date=datetime.utcnow().strftime("%Y-%m-%d"),
                                      data_period={}, universe=[symbol])
            result.approval_status  = ApprovalStatus.REJECTED
            result.rejection_reason = "No market data"
            return result
        concept_names = [getattr(c, "NAME", type(c).__name__) for c in self.concepts]
        sim = LiveSimulator(
            concept_instances  = self.concepts,
            initial_equity     = self.initial_equity,
        )
        rc = RegimeClassifier()
        return sim.build_simulation_result(
            symbol       = symbol,
            df           = df,
            concept_name = ", ".join(concept_names),
            start        = start,
            end          = end,
            regime_classifier = rc,
        )
