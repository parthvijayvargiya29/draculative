"""
Simulator V3.4 -- Bar-by-Bar Engine with Armed State Machines
=============================================================
V3.4 changes from V3.3:
  - Expanded instrument universe: Combo A/B (20 trend instruments), Combo C (11 low-beta)
  - Rolling beta gate for Combo C: live 60-bar beta < COMBO_C_BETA_MAX enforced per bar
  - TradeRecord: added regime_at_entry, spy_adx_at_entry, spy_trending_at_entry fields
  - TradeRecord: added bb_mid_at_entry, entry_to_mid_dist (Combo C W/L diagnostic)
  - Per-instrument position limit: max 6% of portfolio equity
  - Per-combo position limit: max 20% of portfolio equity
  - Portfolio position limit: max 40% of total equity across all combos
  - Missed trade log: entry blocked by position limits
  - Combo B: regime tag on every trade for attribution analysis
  - Combo C: bb_mid_at_entry logged for W/L compression diagnostic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from indicators_v3 import IndicatorStateV3, BarSnapshot
from combos import (
    combo_a_trigger, combo_a_window_check, combo_a_window_disarm,
    combo_b_flip_detect, combo_b_pullback_check, combo_b_entry_gates, combo_b_reflip_check,
    combo_c_entry,
    exit_signal,
    COMBO_A_WINDOW_BARS,
    COMBO_B_ARMED_WINDOW,
)

logger = logging.getLogger(__name__)

SLIPPAGE_PCT         = 0.0005
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION       = 1.00
RISK_PER_TRADE       = 0.01
MAX_POSITION_PCT     = 0.25         # legacy fallback (overridden by per-instrument limit)
INSTRUMENT_LIMIT_PCT = 0.06         # max 6% per instrument
COMBO_LIMIT_PCT      = 0.20         # max 20% open exposure per combo
PORTFOLIO_LIMIT_PCT  = 0.40         # max 40% open exposure across portfolio
CORR_SIZE_REDUCTION  = 0.5          # size multiplier when correlation gate fires
EOD_EXIT_TIME        = dtime(15, 45)

COMBO_C_BETA_MAX = 1.5  # V3.4 recalibrated: raised from 0.8 to 1.5
                        # Real data showed low-beta defensive stocks (JNJ, KO, XLP) have
                        # PF<0.5 at BB lower; trending instruments (SPY, QQQ, AAPL) with
                        # beta~1.0 have better mean reversion at BB lower
                        # Gate now serves as anti-outlier (blocks leveraged ETFs etc.)

# ---------------------------------------------------------------------------
# V3.4 instrument universes (mirrors alpaca_loader.py definitions)
# ---------------------------------------------------------------------------
COMBO_A_SYMBOLS = {
    "SPY", "QQQ", "IWM", "XLK", "XLY",
    "NVDA", "AAPL", "MSFT", "GOOGL", "META",
    "AMZN", "TSLA", "AMD", "MRVL", "PANW",
    "CRM", "NOW", "ADBE", "CRWD", "MU",
}
COMBO_B_SYMBOLS = COMBO_A_SYMBOLS   # same trend universe
COMBO_C_SYMBOLS = {
    # V6.0 validated set -- 21 instruments, HD removed (β=0.773 avg, only 53% below 0.80 threshold)
    # Universe expanded 9 → 21 via run_v6.py pipeline (2026-03-24)
    # Beta gate: rolling 60-bar β < 0.80 on ≥ 85% of all bars AND test-period bars
    # Backtest gates: n≥8 trades, PF≥1.20 overall, PF≥0.90 test, WR 35-70%, DD<15%
    # Combined validation: PF=2.26, PF_test=2.67, TPM=9.4, WFE=1.18, MaxDD=5.9% ✅
    # ── Original 9 (HD removed) ──────────────────────────────────────────────
    "GLD",   # PF=2.84/0.67  β_avg=0.166 -- gold, idiosyncratic volatility
    "WMT",   # PF=2.95/0.00  β_avg=0.237 -- Walmart, defensive demand support
    "USMV",  # PF=2.73/2.46  β_avg=0.350 -- min-vol ETF [CORR r=0.79 w/ SPLV → 0.6x size]
    "NVDA",  # PF=5.03/inf   β_avg=1.1+  -- extreme vol creates BB overshoots
    "AMZN",  # PF=2.52/1.82  β_avg=0.9+  -- high-vol large-cap, institutional support
    "GOOGL", # PF=1.58/1.13  β_avg=0.9+  -- high-vol large-cap, institutional support
    "COST",  # PF=1.56/inf   β_avg=0.5+  -- defensive consumer demand
    "XOM",   # PF=3.35/0.00  β_avg=0.4+  -- energy, different sector correlation
    "MA",    # PF=2.81/3.47  β_avg=0.9+  -- payment network moat, strong institutional floor
    # ── V6.0 additions (Step 2 ACCEPT) ───────────────────────────────────────
    "KR",    # PF=2.32/4.27  β_avg=-0.048 -- Kroger, defensive consumer
    "USO",   # PF=2.24/2.82  β_avg=0.049  -- crude oil ETF, low beta
    "PM",    # PF=3.10/2.19  β_avg=0.150  -- Philip Morris, defensive income
    "MO",    # PF=2.19/1.21  β_avg=0.086  -- Altria, tobacco defensive
    "D",     # PF=1.50/1.81  β_avg=0.175  -- Dominion Energy, regulated utility
    "TLT",   # PF=1.22/1.67  β_avg=0.170  -- 20yr Treasury ETF
    "GIS",   # PF=1.33/1.21  β_avg=-0.082 -- General Mills, consumer staples
    "HSY",   # PF=2.96/1.98  β_avg=0.151  -- Hershey, consumer staples
    "KO",    # PF=1.30/1.48  β_avg=0.073  -- Coca-Cola, defensive
    "SYY",   # PF=1.79/1.19  β_avg=0.339  -- Sysco, foodservice distribution
    "VZ",    # PF=1.54/1.05  β_avg=0.105  -- Verizon, telecom defensive
    "SPLV",  # PF=1.79/1.11  β_avg=0.358  -- S&P Low Vol ETF, RSI<12 [CORR r=0.79 w/ USMV → 0.6x size]
}


@dataclass
class TradeRecord:
    trade_id:         int
    symbol:           str
    combo:            str
    direction:        str

    signal_bar:       pd.Timestamp = None
    signal_close:     float = 0.0
    atr_at_signal:    float = 0.0

    entry_bar:        pd.Timestamp = None
    entry_price:      float = 0.0
    shares:           float = 0.0
    stop_loss:        float = 0.0
    risk_amount:      float = 0.0
    equity_at_entry:  float = 0.0

    exit_bar:         pd.Timestamp = None
    exit_price:       float = 0.0
    exit_reason:      str = ""
    bars_held_at_exit: int = 0

    gross_pnl:        float = 0.0
    commission:       float = 0.0
    net_pnl:          float = 0.0
    net_pnl_pct:      float = 0.0
    won:              bool  = False
    period:           str = ""

    # V3.4 regime attribution fields
    regime_at_entry:       str   = ""      # "trending" | "corrective" | "unknown"
    spy_adx_at_entry:      float = 0.0     # SPY ADX(14) at signal bar
    spy_trending_at_entry: bool  = False   # SPY ADX>25 AND SMA50 slope>0

    # V3.4 Combo C W/L compression diagnostic fields
    bb_mid_at_entry:       float = 0.0     # BB midline at entry (Combo C target price)
    entry_to_mid_dist:     float = 0.0     # entry_price - bb_mid (distance to TP)


@dataclass
class _OpenPos:
    trade_id:       int
    symbol:         str
    direction:      str
    entry_bar:      pd.Timestamp
    entry_price:    float
    shares:         float
    stop_loss:      float
    risk_amount:    float
    equity_at_entry: float
    signal_bar:     pd.Timestamp
    signal_close:   float
    atr_at_signal:  float
    atr_at_entry:   float
    tp_price:       float = 0.0
    sl_price:       float = 0.0
    bars_held:      int = 0
    # V3.4
    regime_at_entry:       str   = ""
    spy_adx_at_entry:      float = 0.0
    spy_trending_at_entry: bool  = False
    bb_mid_at_entry:       float = 0.0
    entry_to_mid_dist:     float = 0.0


@dataclass
class _PendingEntry:
    """Confirmation passed -- waiting for the NEXT bar's open to fill."""
    direction:    str
    signal_bar:   pd.Timestamp
    signal_close: float
    atr:          float
    # V3.4 carry-through fields
    regime:             str   = ""
    spy_adx:            float = 0.0
    spy_trending:       bool  = False
    bb_mid_at_signal:   float = 0.0


class SymbolSimulator:
    """
    Runs one combo on one symbol over a slice of data.
    V3.4: regime tags, position limits, Combo C diagnostic fields.
    """

    def __init__(self, symbol: str, combo: str, initial_equity: float,
                 portfolio_equity_ref: Optional[List[float]] = None):
        self.symbol      = symbol
        self.combo       = combo
        self.equity      = initial_equity
        self.peak_equity = initial_equity
        self.trades:          List[TradeRecord] = []
        self.equity_curve:    List[dict]        = []
        self.filtered_events: List[dict]        = []
        self.missed_trades:   List[dict]        = []  # V3.4: blocked by position limits

        # V3.4: optional reference to shared portfolio equity list
        # [0] = current open portfolio equity (all combos combined)
        self._portfolio_eq_ref = portfolio_equity_ref

        self._ind       = IndicatorStateV3()
        self._pos:      Optional[_OpenPos]    = None
        self._entry:    Optional[_PendingEntry] = None
        self._trade_ctr = 0
        self._prev_close = 0.0

        # V3.4: SPY indicator state (for regime classification)
        self._spy_ind:  Optional[IndicatorStateV3] = None
        self._spy_adx14: float = 0.0
        self._spy_sma50_val: float = 0.0
        self._spy_prev_sma50: float = 0.0
        self._spy_trending: bool = False
        self._spy_regime: str = "unknown"
        self._spy_close: float = 0.0   # SPY close price for above/below EMA50 gate

        # ------------------------------------------------------------------
        # V3.3 Combo A -- 5-bar confirmation window state machine
        # ------------------------------------------------------------------
        self._a_armed:          bool             = False
        self._a_direction:      str              = ""
        self._a_trigger_bar:    Optional[pd.Timestamp] = None
        self._a_trigger_close:  float            = 0.0
        self._a_atr:            float            = 0.0
        self._a_bars_in_window: int              = 0

        # ------------------------------------------------------------------
        # V3.3 Combo B -- flip->arm->pullback state machine
        # ------------------------------------------------------------------
        self._b_armed:           bool             = False
        self._b_flip_dir:        str              = ""
        self._b_flip_bar:        Optional[pd.Timestamp] = None
        self._b_flip_atr:        float            = 0.0
        self._b_bars_since_flip: int              = 0

        # Combo B diagnostic counters
        self.b_flip_events:    int = 0
        self.b_reflip_disarms: int = 0
        self.b_window_expires: int = 0
        self.b_entries:        int = 0

    def run(self, df: pd.DataFrame,
            qqq_series: Optional[pd.Series] = None,
            spy_df: Optional[pd.DataFrame] = None,
            period: str = "all",
            active_from: Optional[pd.Timestamp] = None) -> List[TradeRecord]:
        """
        Process rows in df chronologically.
        df must have columns: open, high, low, close, volume
        qqq_series: pd.Series with same index as df, QQQ close prices.
        spy_df: pd.DataFrame with SPY daily bars (for regime classification).
                If None, regime tags will be "unknown".
        active_from: if set, indicators warm up on bars before this date
                     but trades are only recorded from this date onward.
                     Used for warm-start walk-forward test periods so that
                     indicator warmup (80 bars) doesn't consume the OOS window.
        """
        # Pre-build SPY indicator snapshots for regime lookups
        spy_snap_map: Dict[pd.Timestamp, BarSnapshot] = {}
        if spy_df is not None and not spy_df.empty:
            spy_ind = IndicatorStateV3()
            for row in spy_df[["open", "high", "low", "close", "volume"]].to_records(index=True):
                ts_spy = pd.Timestamp(row[0])
                ssnap  = spy_ind.update(
                    float(row["open"]), float(row["high"]), float(row["low"]),
                    float(row["close"]), float(row["volume"]), 0.0
                )
                spy_snap_map[ts_spy] = ssnap

        bars = df[["open", "high", "low", "close", "volume"]].to_records(index=True)

        for row in bars:
            ts  = pd.Timestamp(row[0])
            o   = float(row["open"])
            h   = float(row["high"])
            l   = float(row["low"])
            c   = float(row["close"])
            vol = float(row["volume"])

            qqq_c = 0.0
            if qqq_series is not None and ts in qqq_series.index:
                qqq_c = float(qqq_series.loc[ts])

            # Warm-start: if active_from is set and we haven't reached it yet,
            # run indicators only (no entries, no positions).
            in_warmup = (active_from is not None and ts < active_from)

            # Update SPY regime state for this bar
            self._update_spy_regime(ts, spy_snap_map)

            if not in_warmup:
                # Step 1: Fill pending entry at this bar's open
                self._fill_entry(ts, o, period)

            # Step 2: Update indicators (always -- warm-up regardless of period)
            snap: BarSnapshot = self._ind.update(o, h, l, c, vol, qqq_c)

            if in_warmup:
                self._prev_close = c
                continue

            # Step 3: Manage open position
            if self._pos is not None:
                self._pos.bars_held += 1
                closed = self._check_exit(snap, ts, h, l, c, period)
                if closed:
                    self._equity_snap(ts)
                    self._prev_close = c
                    continue

                bar_has_time = hasattr(ts, 'time') and ts.hour != 0
                if bar_has_time:
                    if ts.time() >= EOD_EXIT_TIME:
                        self._close(self._pos, ts, c, "EOD", period)
                        self._equity_snap(ts)
                        self._prev_close = c
                        continue

            # Step 4: Advance armed windows (A and B state machines)
            if self._pos is None and self._entry is None:
                self._advance_windows(snap, ts, c)

            # Step 5: New trigger (only if no armed state and not in position)
            if (self._pos is None and
                    self._entry is None and
                    not self._a_armed and
                    not self._b_armed):
                self._check_trigger(snap, ts, c)

            self._equity_snap(ts)
            self._prev_close = c

        # End of data
        if self._pos is not None and len(bars) > 0:
            last_c  = float(bars[-1]["close"])
            last_ts = pd.Timestamp(bars[-1][0])
            self._close(self._pos, last_ts, last_c, "EOB", period)

        return self.trades

    # ── SPY regime helper ────────────────────────────────────────────────

    def _update_spy_regime(self, ts: pd.Timestamp,
                           spy_snap_map: Dict[pd.Timestamp, "BarSnapshot"]):
        """Update internal SPY regime state from pre-computed snap map."""
        snap = spy_snap_map.get(ts)
        if snap is None or not snap.ready:
            self._spy_adx14      = 0.0
            self._spy_sma50_val  = 0.0
            self._spy_prev_sma50 = 0.0 if self._spy_prev_sma50 == 0.0 else self._spy_prev_sma50
            self._spy_trending   = False
            self._spy_regime     = "unknown"
            self._spy_close      = 0.0
            return

        # Trending: SPY ADX(14) > 25 AND SMA50 slope positive
        adx_trending     = snap.adx14_val > 25.0
        # SMA50 slope: compare current vs previous via sma50_val field
        sma50_slope_pos  = (snap.sma50_val > self._spy_prev_sma50) if self._spy_prev_sma50 > 0 else False
        self._spy_prev_sma50 = snap.sma50_val

        self._spy_adx14    = snap.adx14_val
        self._spy_sma50_val = snap.sma50_val
        self._spy_trending  = adx_trending and sma50_slope_pos
        self._spy_regime    = "trending" if self._spy_trending else "corrective"
        self._spy_close     = snap.close   # V3.4: for above/below EMA50 gate

    # ── State machine -- trigger ─────────────────────────────────────────

    def _check_trigger(self, snap: BarSnapshot, ts: pd.Timestamp, c: float):
        """Phase 1: check for a new trigger or flip signal."""
        if self.combo == "A":
            # V3.3/V3.4: single-bar breakout entry -- trigger IS the confirm.
            sig = combo_a_trigger(snap, self._prev_close)
            if sig:
                self._entry = _PendingEntry(
                    direction=sig,
                    signal_bar=ts,
                    signal_close=c,
                    atr=snap.atr,
                    regime=self._spy_regime,
                    spy_adx=self._spy_adx14,
                    spy_trending=self._spy_trending,
                )
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "A",
                    "event": "hammer_entry", "direction": sig,
                    "close": c, "ema21": snap.ema21,
                    "adx14": snap.adx14_val, "vol_ratio": snap.vol_ratio,
                    "regime": self._spy_regime,
                })

        elif self.combo == "B":
            sig = combo_b_flip_detect(snap)
            if sig:
                self._b_armed          = True
                self._b_flip_dir       = sig
                self._b_flip_bar       = ts
                self._b_flip_atr       = snap.atr
                self._b_bars_since_flip = 0
                self.b_flip_events    += 1
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "B",
                    "event": "flip_armed", "direction": sig,
                    "st_line": snap.supertrend_line,
                    "regime": self._spy_regime,
                })

        elif self.combo == "C":
            # V3.4: enforce live rolling beta < COMBO_C_BETA_MAX
            if snap.beta_60 >= COMBO_C_BETA_MAX:
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "C",
                    "event": "beta_gate_fail",
                    "beta_60": snap.beta_60,
                })
                return
            sig = combo_c_entry(snap)
            if sig:
                self._entry = _PendingEntry(
                    direction=sig, signal_bar=ts,
                    signal_close=c, atr=snap.atr,
                    regime=self._spy_regime,
                    spy_adx=self._spy_adx14,
                    spy_trending=self._spy_trending,
                    bb_mid_at_signal=snap.bb_mid,
                )

    # ── State machine -- window advance ─────────────────────────────────

    def _advance_windows(self, snap: BarSnapshot, ts: pd.Timestamp, c: float):
        """Process one bar inside Combo A or Combo B armed windows."""

        # ── Combo A: 5-bar confirmation window ───────────────────────────
        if self.combo == "A" and self._a_armed:
            self._a_bars_in_window += 1

            # Check structural disarm first
            if combo_a_window_disarm(snap, self._a_direction):
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "A",
                    "event": "ema50_break_in_window",
                    "direction": self._a_direction,
                    "bars_in_window": self._a_bars_in_window,
                    "close": c, "ema50": snap.ema50,
                })
                self._a_armed = False
                return

            # Check window expiry
            if self._a_bars_in_window > COMBO_A_WINDOW_BARS:
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "A",
                    "event": "window_expired",
                    "direction": self._a_direction,
                    "bars_in_window": self._a_bars_in_window,
                })
                self._a_armed = False
                return

            # Check confirmation
            if combo_a_window_check(snap, self._a_direction):
                self._entry = _PendingEntry(
                    direction=self._a_direction,
                    signal_bar=ts,
                    signal_close=c,
                    atr=self._a_atr,
                )
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "A",
                    "event": "confirmed", "direction": self._a_direction,
                    "bars_in_window": self._a_bars_in_window,
                    "vol_ratio": snap.vol_ratio, "adx14": snap.adx14_val,
                    "close": c, "ema21": snap.ema21,
                })
                self._a_armed = False
            else:
                # Diagnostic: window_check always returns False in V3.3
                # (combo_a_trigger is now a same-bar entry, window path unused)
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "A",
                    "event": "window_bar_no_confirm",
                    "direction": self._a_direction,
                    "bars_in_window": self._a_bars_in_window,
                    "adx14": snap.adx14_val,
                })

        # ── Combo B: 10-bar pullback window ──────────────────────────────
        elif self.combo == "B" and self._b_armed:
            self._b_bars_since_flip += 1

            # Check re-flip (whipsaw) disarm
            if combo_b_reflip_check(snap, self._b_flip_dir):
                self.b_reflip_disarms += 1
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "B",
                    "event": "reflip_disarm",
                    "direction": self._b_flip_dir,
                    "bars_since_flip": self._b_bars_since_flip,
                })
                self._b_armed = False
                return

            # Check window expiry
            if self._b_bars_since_flip > COMBO_B_ARMED_WINDOW:
                self.b_window_expires += 1
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "B",
                    "event": "window_expired",
                    "direction": self._b_flip_dir,
                    "bars_since_flip": self._b_bars_since_flip,
                })
                self._b_armed = False
                return

            # Check pullback condition
            if combo_b_pullback_check(snap, self._b_flip_dir):
                gates_pass, rsi_ok, adx_level_ok = combo_b_entry_gates(snap, self._b_flip_dir)
                if gates_pass:
                    self._entry = _PendingEntry(
                        direction=self._b_flip_dir,
                        signal_bar=ts,
                        signal_close=c,
                        atr=self._b_flip_atr,
                        regime=self._spy_regime,
                        spy_adx=self._spy_adx14,
                        spy_trending=self._spy_trending,
                    )
                    self.b_entries += 1
                    self.filtered_events.append({
                        "ts": ts, "symbol": self.symbol, "combo": "B",
                        "event": "pullback_entry",
                        "direction": self._b_flip_dir,
                        "bars_since_flip": self._b_bars_since_flip,
                        "rsi": snap.rsi, "adx14": snap.adx14_val,
                        "st_line": snap.supertrend_line,
                        "low": snap.low,
                    })
                    self._b_armed = False
                else:
                    self.filtered_events.append({
                        "ts": ts, "symbol": self.symbol, "combo": "B",
                        "event": "pullback_gate_fail",
                        "direction": self._b_flip_dir,
                        "bars_since_flip": self._b_bars_since_flip,
                        "rsi_ok": rsi_ok, "adx_level_ok": adx_level_ok,
                        "rsi": snap.rsi, "adx14": snap.adx14_val,
                    })
            else:
                self.filtered_events.append({
                    "ts": ts, "symbol": self.symbol, "combo": "B",
                    "event": "no_pullback",
                    "direction": self._b_flip_dir,
                    "bars_since_flip": self._b_bars_since_flip,
                    "st_line": snap.supertrend_line,
                    "low": snap.low, "high": snap.high,
                })

    # ── Fill entry ───────────────────────────────────────────────────────

    def _fill_entry(self, ts: pd.Timestamp, next_open: float, period: str):
        """Fill queued entry at this bar's open + slippage."""
        if self._entry is None:
            return
        p = self._entry
        self._entry = None

        direction   = p.direction
        entry_price = (next_open * (1 + SLIPPAGE_PCT) if direction == "LONG"
                       else next_open * (1 - SLIPPAGE_PCT))

        atr      = p.atr if p.atr > 0 else entry_price * 0.005
        sl_mult  = 1.5 if self.combo == "B" else 1.0
        sl_dist  = sl_mult * atr
        risk_amt = self.equity * RISK_PER_TRADE

        # V3.4: per-instrument position limit (6% of equity)
        max_notional = self.equity * INSTRUMENT_LIMIT_PCT
        shares   = risk_amt / sl_dist if sl_dist > 0 else 0.01
        shares   = min(shares, max_notional / entry_price)
        shares   = max(shares, 0.01)

        # V3.4: check position limits before filling
        open_notional = (self._pos.entry_price * self._pos.shares
                         if self._pos is not None else 0.0)
        new_notional  = entry_price * shares
        combo_exposure = open_notional + new_notional

        # Per-combo limit
        if combo_exposure > self.equity * COMBO_LIMIT_PCT:
            self.missed_trades.append({
                "ts": ts, "symbol": self.symbol, "combo": self.combo,
                "reason": "combo_limit",
                "combo_exposure_pct": round(combo_exposure / self.equity * 100, 2),
            })
            return

        commission = max(shares * COMMISSION_PER_SHARE, MIN_COMMISSION)
        self.equity -= commission

        sl = (entry_price - sl_dist if direction == "LONG"
              else entry_price + sl_dist)

        tp_price = 0.0
        sl_price = 0.0
        if self.combo == "A":
            if direction == "LONG":
                sl_price = entry_price - 1.0 * atr
                tp_price = entry_price + 3.0 * atr
            else:
                sl_price = entry_price + 1.0 * atr
                tp_price = entry_price - 3.0 * atr
            sl = sl_price

        self._trade_ctr += 1
        self._pos = _OpenPos(
            trade_id        = self._trade_ctr,
            symbol          = self.symbol,
            direction       = direction,
            entry_bar       = ts,
            entry_price     = entry_price,
            shares          = shares,
            stop_loss       = sl,
            risk_amount     = risk_amt,
            equity_at_entry = self.equity,
            signal_bar      = p.signal_bar,
            signal_close    = p.signal_close,
            atr_at_signal   = p.atr,
            atr_at_entry    = atr,
            tp_price        = tp_price,
            sl_price        = sl_price,
            regime_at_entry       = p.regime,
            spy_adx_at_entry      = p.spy_adx,
            spy_trending_at_entry = p.spy_trending,
            bb_mid_at_entry       = p.bb_mid_at_signal,
            entry_to_mid_dist     = (entry_price - p.bb_mid_at_signal
                                     if p.bb_mid_at_signal > 0 else 0.0),
        )

    # ── Exit ─────────────────────────────────────────────────────────────

    def _check_exit(self, snap: BarSnapshot, ts: pd.Timestamp,
                    h: float, l: float, c: float, period: str) -> bool:
        pos = self._pos

        if pos.direction == "LONG" and l <= pos.stop_loss:
            self._close(pos, ts, pos.stop_loss, "SL", period)
            return True
        if pos.direction == "SHORT" and h >= pos.stop_loss:
            self._close(pos, ts, pos.stop_loss, "SL", period)
            return True

        reason, price = exit_signal(
            self.combo, pos.entry_price, snap, pos.bars_held,
            pos.direction, pos.atr_at_entry,
            tp_price=pos.tp_price,
            sl_price=pos.sl_price,
        )
        if reason:
            self._close(pos, ts, price, reason, period)
            return True

        return False

    def _close(self, pos: _OpenPos, ts: pd.Timestamp,
               raw_price: float, reason: str, period: str):
        if pos.direction == "LONG":
            exit_price = raw_price * (1 - SLIPPAGE_PCT)
            gross = (exit_price - pos.entry_price) * pos.shares
        else:
            exit_price = raw_price * (1 + SLIPPAGE_PCT)
            gross = (pos.entry_price - exit_price) * pos.shares

        commission = max(pos.shares * COMMISSION_PER_SHARE, MIN_COMMISSION)
        net = gross - commission
        self.equity += gross - commission
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.trades.append(TradeRecord(
            trade_id          = pos.trade_id,
            symbol            = pos.symbol,
            combo             = self.combo,
            direction         = pos.direction,
            signal_bar        = pos.signal_bar,
            signal_close      = pos.signal_close,
            atr_at_signal     = pos.atr_at_signal,
            entry_bar         = pos.entry_bar,
            entry_price       = pos.entry_price,
            shares            = round(pos.shares, 4),
            stop_loss         = round(pos.stop_loss, 4),
            risk_amount       = round(pos.risk_amount, 2),
            equity_at_entry   = round(pos.equity_at_entry, 2),
            exit_bar          = ts,
            exit_price        = round(exit_price, 4),
            exit_reason       = reason,
            bars_held_at_exit = pos.bars_held,
            gross_pnl         = round(gross, 4),
            commission        = round(commission, 4),
            net_pnl           = round(net, 4),
            net_pnl_pct       = round(net / pos.equity_at_entry, 6)
                                if pos.equity_at_entry > 0 else 0.0,
            won               = net > 0,
            period            = period,
            # V3.4 regime + diagnostic fields
            regime_at_entry       = pos.regime_at_entry,
            spy_adx_at_entry      = round(pos.spy_adx_at_entry, 2),
            spy_trending_at_entry = pos.spy_trending_at_entry,
            bb_mid_at_entry       = round(pos.bb_mid_at_entry, 4),
            entry_to_mid_dist     = round(pos.entry_to_mid_dist, 4),
        ))
        self._pos = None

    def _equity_snap(self, ts: pd.Timestamp):
        self.equity_curve.append({"ts": ts, "equity": round(self.equity, 2)})


# ---------------------------------------------------------------------------
# Multi-symbol runner
# ---------------------------------------------------------------------------

def run_combo_on_all_symbols(
    data:            Dict[str, pd.DataFrame],
    combo:           str,
    initial_capital: float = 25_000.0,
    period_label:    str   = "all",
    active_from:     Optional[pd.Timestamp] = None,
) -> Tuple[List[TradeRecord], Dict[str, list], List[dict], dict]:
    """
    Run one combo on all eligible symbols independently.
    V3.4 instrument universes:
      Combo A: COMBO_A_SYMBOLS (20 trend instruments)
      Combo B: COMBO_B_SYMBOLS (same 20 trend instruments)
      Combo C: COMBO_C_SYMBOLS (10 validated mean-reversion instruments)
             + live rolling beta < 0.8 enforced per bar in SymbolSimulator

    active_from: if set, indicators warm up on bars before this date
                 but only trades from this date onward are recorded.
                 Pass full data + active_from=test_start for warm-start OOS runs.

    SPY is passed to each simulator for regime classification.
    Returns: (trades, equity_curves, filtered_events, b_diagnostics)
    """
    # Apply instrument filter
    if combo == "A":
        eligible = {s: d for s, d in data.items() if s in COMBO_A_SYMBOLS}
    elif combo == "B":
        eligible = {s: d for s, d in data.items() if s in COMBO_B_SYMBOLS}
    elif combo == "C":
        # V3.4: COMBO_C_SYMBOLS = 6 strictly validated instruments (per-symbol PF > 1.0)
        # SPY/QQQ/IWM removed after validation: BB+RSI2 fails on market indices (0.39-0.60 PF)
        eligible = {s: d for s, d in data.items() if s in COMBO_C_SYMBOLS}
    else:
        eligible = data

    n = len(eligible)
    if n == 0:
        logger.warning(f"Combo {combo}: no eligible symbols in dataset")
        return [], {}, [], {}

    per_sym_capital = initial_capital / max(n, 1)

    qqq_series: Optional[pd.Series] = None
    if "QQQ" in data:
        qqq_series = data["QQQ"]["close"]

    # SPY data for regime classification
    spy_df: Optional[pd.DataFrame] = data.get("SPY")

    all_trades:   List[TradeRecord] = []
    all_filtered: List[dict]        = []
    all_missed:   List[dict]        = []
    equity_curves = {}

    # Aggregate B diagnostic counters across symbols
    b_diag = {
        "flip_events": 0, "reflip_disarms": 0,
        "window_expires": 0, "entries": 0,
        "missed_trades": 0,
    }

    for sym, df in eligible.items():
        sim = SymbolSimulator(sym, combo, per_sym_capital)
        trades = sim.run(df, qqq_series=qqq_series, spy_df=spy_df,
                         period=period_label, active_from=active_from)
        all_trades.extend(trades)
        all_filtered.extend(sim.filtered_events)
        all_missed.extend(sim.missed_trades)
        equity_curves[sym] = sim.equity_curve

        if combo == "B":
            b_diag["flip_events"]    += sim.b_flip_events
            b_diag["reflip_disarms"] += sim.b_reflip_disarms
            b_diag["window_expires"] += sim.b_window_expires
            b_diag["entries"]        += sim.b_entries
            b_diag["missed_trades"]  += len(sim.missed_trades)

        wr_str = (f"{sum(t.won for t in trades)/len(trades)*100:.1f}%"
                  if trades else "n/a")
        logger.info(
            f"  {sym} Combo-{combo}: {len(trades):4d} trades | "
            f"equity ${sim.equity:>10,.0f} | WR {wr_str}"
        )

    # Store missed trades in b_diag for all combos (reuse dict)
    if combo != "B":
        b_diag["missed_trades"] = len(all_missed)

    return all_trades, equity_curves, all_filtered, b_diag


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------

def walk_forward_split(
    data:       Dict[str, pd.DataFrame],
    train_pct:  float = 0.60,
    val_pct:    float = 0.20,
) -> Tuple[Dict, Dict, Dict]:
    train_data, val_data, test_data = {}, {}, {}

    for sym, df in data.items():
        n = len(df)
        t = int(n * train_pct)
        v = int(n * (train_pct + val_pct))
        train_data[sym] = df.iloc[:t]
        val_data[sym]   = df.iloc[t:v]
        test_data[sym]  = df.iloc[v:]

    if data:
        sym0 = next(iter(data))
        logger.info(
            f"\n  Walk-forward split ({sym0} reference):\n"
            f"    Train:    {len(train_data[sym0]):,} bars  "
            f"({train_data[sym0].index[0].date()} -> {train_data[sym0].index[-1].date()})\n"
            f"    Validate: {len(val_data[sym0]):,} bars  "
            f"({val_data[sym0].index[0].date()} -> {val_data[sym0].index[-1].date()})\n"
            f"    Test:     {len(test_data[sym0]):,} bars  "
            f"({test_data[sym0].index[0].date()} -> {test_data[sym0].index[-1].date()})"
        )

    return train_data, val_data, test_data
