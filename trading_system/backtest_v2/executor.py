"""
Trade Executor & Portfolio

Handles all order lifecycle for the simulation:
  - Entry at NEXT bar's OPEN price (not the signal bar's close)
  - Slippage model applied at entry and exit
  - Commission deducted per trade
  - ATR-based stop loss
  - Optional ATR-based take profit
  - End-of-day forced flat (no overnight holds)
  - Per-symbol position tracking (max 1 open position per symbol)

Capital management:
  - Risk a fixed % of CURRENT equity per trade
  - Position size = risk_amount / atr_stop_distance
  - Compound returns: each trade's risk is based on current equity
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, time as dtime
import pandas as pd

from signals import Signal, Direction


# ─────────────────────────────────────────────────────────────────────────────
# Execution parameters
# ─────────────────────────────────────────────────────────────────────────────

RISK_PER_TRADE       = 0.01        # 1% of current equity at risk per trade
ATR_STOP_MULT        = 1.5         # Stop = entry ± ATR_STOP_MULT × ATR
ATR_TP_MULT          = 3.0         # Take profit = entry ± ATR_TP_MULT × ATR
SLIPPAGE_PCT         = 0.0005      # 0.05% slippage per side (5 bps)
COMMISSION_PER_SHARE = 0.005       # $0.005 per share (Interactive Brokers rate)
MIN_COMMISSION       = 1.00        # Minimum $1.00 per order
EOD_EXIT_TIME        = dtime(15, 45)  # Force close at 15:45 (last full bar)
MAX_POSITION_PCT     = 0.20        # Max 20% of equity in any single position


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Complete record of one round-trip trade."""
    trade_id:    int
    symbol:      str
    direction:   str           # BUY / SELL

    # Signal context
    signal_time:     datetime = None
    signal_close:    float    = 0.0
    signal_reason:   str      = ""
    signal_conf:     float    = 0.0
    atr_at_signal:   float    = 0.0

    # Entry (NEXT bar open + slippage)
    entry_time:  datetime = None
    entry_price: float    = 0.0
    shares:      float    = 0.0
    stop_loss:   float    = 0.0
    take_profit: float    = 0.0
    risk_amount: float    = 0.0       # $ risked
    equity_at_entry: float = 0.0

    # Exit
    exit_time:   datetime = None
    exit_price:  float    = 0.0
    exit_reason: str      = ""        # SL / TP / EOD / NEW_SIGNAL

    # P&L
    gross_pnl:   float = 0.0          # $ before costs
    commission:  float = 0.0          # $ commission
    slippage:    float = 0.0          # $ slippage cost
    net_pnl:     float = 0.0          # $ after all costs
    net_pnl_pct: float = 0.0          # net_pnl / equity_at_entry

    # Classification
    won:         bool  = False
    period:      str   = ""           # 'train' / 'validate' / 'test'


# ─────────────────────────────────────────────────────────────────────────────
# Open position (live during simulation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    trade_id:    int
    symbol:      str
    direction:   str
    entry_time:  datetime
    entry_price: float
    shares:      float
    stop_loss:   float
    take_profit: float
    risk_amount: float
    equity_at_entry: float
    signal_reason: str  = ""
    signal_conf:   float = 0.0
    atr_at_signal: float = 0.0
    signal_close:  float = 0.0
    signal_time:   datetime = None


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio (tracks equity curve & all positions)
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Portfolio state machine.

    Responsibilities:
    - Track current equity (starting capital + cumulative net P&L)
    - Open / close positions per symbol
    - Apply slippage, commission, ATR-based sizing
    - Force EOD exits
    - Emit equity snapshots at every bar for drawdown analysis
    """

    def __init__(self, initial_capital: float = 25_000.0):
        self.initial_capital = initial_capital
        self.equity          = initial_capital
        self.peak_equity     = initial_capital

        self.open_positions: Dict[str, OpenPosition] = {}  # symbol → position
        self.trades:         List[Trade]              = []
        self.equity_curve:   List[dict]               = []  # timestamp → equity

        self._trade_counter    = 0
        self._pending_entries: Dict[str, dict] = {}   # queued entries for next bar

    # ── Bar-level update ─────────────────────────────────────────────────────

    def on_bar(self, symbol: str, ts: datetime,
               bar_open: float, bar_high: float,
               bar_low: float, bar_close: float,
               signal: Signal, period: str):
        """
        Called at every bar close for one symbol.

        Order of operations (mirrors live trading):
        1. Check if existing open position was stopped out or hit TP THIS bar
        2. Check for EOD forced exit
        3. If signal fires AND no open position → queue entry for next bar open
           (Entry is recorded NOW but executed at NEXT bar open via on_next_bar_open)
        """

        # ── 1. Manage existing open position ────────────────────────────────
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            closed = self._check_exit(pos, ts, bar_high, bar_low, bar_close, period)
            if closed:
                return   # Position closed this bar, don't open a new one

        # ── 2. EOD: close any remaining open position ─────────────────────────
        bar_time = ts.time() if isinstance(ts, datetime) else ts
        if bar_time >= EOD_EXIT_TIME and symbol in self.open_positions:
            self._close_position(
                symbol, ts, bar_close,
                exit_reason='EOD', period=period
            )
            return

        # ── 3. No open position + valid signal → record pending entry ─────────
        # The actual entry price is set in on_next_bar_open()
        if symbol not in self.open_positions and signal.is_valid:
            if signal.direction in (Direction.BUY, Direction.SELL):
                self._queue_entry(symbol, ts, signal, period)

    def on_next_bar_open(self, symbol: str, ts: datetime,
                         next_open: float, period: str):
        """
        Called at the OPEN of the bar AFTER a signal bar.
        This is where entry actually fills — simulating a market order at open.
        """
        if symbol not in self._pending_entries:
            return

        pending = self._pending_entries.pop(symbol, None)
        if pending is None:
            return

        direction = pending['direction']
        signal    = pending['signal']

        # Apply slippage (adverse: increases cost)
        if direction == Direction.BUY:
            entry_price = next_open * (1 + SLIPPAGE_PCT)
        else:
            entry_price = next_open * (1 - SLIPPAGE_PCT)

        # Position sizing: risk RISK_PER_TRADE × equity
        risk_amount  = self.equity * RISK_PER_TRADE
        atr          = signal.atr if signal.atr > 0 else entry_price * 0.01
        stop_dist    = ATR_STOP_MULT * atr

        if stop_dist <= 0:
            return

        shares = risk_amount / stop_dist
        # Cap position to MAX_POSITION_PCT of equity
        max_shares = (self.equity * MAX_POSITION_PCT) / entry_price
        shares = min(shares, max_shares)
        shares = max(shares, 0.01)   # Fractional shares allowed

        if direction == Direction.BUY:
            stop_loss   = entry_price - stop_dist
            take_profit = entry_price + ATR_TP_MULT * atr
        else:
            stop_loss   = entry_price + stop_dist
            take_profit = entry_price - ATR_TP_MULT * atr

        # Commission on entry
        commission = max(shares * COMMISSION_PER_SHARE, MIN_COMMISSION)
        self.equity -= commission

        self._trade_counter += 1
        pos = OpenPosition(
            trade_id       = self._trade_counter,
            symbol         = symbol,
            direction      = direction,
            entry_time     = ts,
            entry_price    = entry_price,
            shares         = shares,
            stop_loss      = stop_loss,
            take_profit    = take_profit,
            risk_amount    = risk_amount,
            equity_at_entry = self.equity,
            signal_reason  = signal.reason,
            signal_conf    = signal.confidence,
            atr_at_signal  = signal.atr,
            signal_close   = signal.close,
            signal_time    = pending['signal_time'],
        )
        self.open_positions[symbol] = pos

    def record_equity(self, ts: datetime):
        """Snapshot equity at this timestamp for the equity curve."""
        # Include mark-to-market of open positions would require current price.
        # For simplicity, equity curve is updated only on trade closes.
        self.equity_curve.append({
            'timestamp': ts,
            'equity':    self.equity,
            'drawdown':  (self.peak_equity - self.equity) / self.peak_equity
                          if self.peak_equity > 0 else 0.0,
        })
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    # ── Internal ─────────────────────────────────────────────────────────────

    def _queue_entry(self, symbol: str, ts: datetime,
                     signal: Signal, period: str):
        """Record that we want to enter at the NEXT bar's open."""
        self._pending_entries[symbol] = {
            'direction':   signal.direction,
            'signal':      signal,
            'signal_time': ts,
            'period':      period,
        }

    def _check_exit(self, pos: OpenPosition, ts: datetime,
                    bar_high: float, bar_low: float,
                    bar_close: float, period: str) -> bool:
        """
        Check if the current bar's high/low touched stop or take-profit.

        Important: We don't know which hit first (SL or TP) within the bar.
        Conservative approach: if both SL and TP are touched in the same bar,
        assume SL hit first (worst case — more realistic).
        """
        if pos.direction == Direction.BUY:
            hit_sl = bar_low  <= pos.stop_loss
            hit_tp = bar_high >= pos.take_profit
        else:
            hit_sl = bar_high >= pos.stop_loss
            hit_tp = bar_low  <= pos.take_profit

        if hit_sl:
            self._close_position(pos.symbol, ts, pos.stop_loss,
                                  exit_reason='SL', period=period)
            return True

        if hit_tp:
            self._close_position(pos.symbol, ts, pos.take_profit,
                                  exit_reason='TP', period=period)
            return True

        return False

    def _close_position(self, symbol: str, ts: datetime,
                        raw_exit_price: float,
                        exit_reason: str, period: str):
        """Close an open position, compute P&L, record trade."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions.pop(symbol)

        # Slippage on exit (adverse)
        if pos.direction == Direction.BUY:
            exit_price = raw_exit_price * (1 - SLIPPAGE_PCT)
        else:
            exit_price = raw_exit_price * (1 + SLIPPAGE_PCT)

        # P&L
        if pos.direction == Direction.BUY:
            gross_pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.shares

        commission = max(pos.shares * COMMISSION_PER_SHARE, MIN_COMMISSION)
        slippage_cost = (
            pos.entry_price * pos.shares * SLIPPAGE_PCT +
            raw_exit_price  * pos.shares * SLIPPAGE_PCT
        )
        net_pnl = gross_pnl - commission

        self.equity += gross_pnl - commission
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        net_pnl_pct = net_pnl / pos.equity_at_entry if pos.equity_at_entry > 0 else 0.0

        trade = Trade(
            trade_id        = pos.trade_id,
            symbol          = symbol,
            direction       = pos.direction,
            signal_time     = pos.signal_time,
            signal_close    = pos.signal_close,
            signal_reason   = pos.signal_reason,
            signal_conf     = pos.signal_conf,
            atr_at_signal   = pos.atr_at_signal,
            entry_time      = pos.entry_time,
            entry_price     = pos.entry_price,
            shares          = pos.shares,
            stop_loss       = pos.stop_loss,
            take_profit     = pos.take_profit,
            risk_amount     = pos.risk_amount,
            equity_at_entry = pos.equity_at_entry,
            exit_time       = ts,
            exit_price      = exit_price,
            exit_reason     = exit_reason,
            gross_pnl       = round(gross_pnl, 4),
            commission      = round(commission, 4),
            slippage        = round(slippage_cost, 4),
            net_pnl         = round(net_pnl, 4),
            net_pnl_pct     = round(net_pnl_pct, 6),
            won             = net_pnl > 0,
            period          = period,
        )
        self.trades.append(trade)

        # Also cancel any pending entry for this symbol if it somehow queued
        self._pending_entries.pop(symbol, None)

    def force_close_all(self, ts: datetime, prices: Dict[str, float], period: str):
        """Close all open positions at given prices (end of backtest)."""
        for symbol in list(self.open_positions.keys()):
            price = prices.get(symbol, self.open_positions[symbol].entry_price)
            self._close_position(symbol, ts, price, exit_reason='EOB', period=period)
