#!/usr/bin/env python3
"""
sim_5000.py -- Combo C Simulation: $5,000 Starting Capital
"""
from __future__ import annotations
import sys, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from collections import Counter

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data
from simulator import walk_forward_split, COMBO_C_SYMBOLS
from combos import combo_c_entry
from indicators_v3 import IndicatorStateV3

logging.basicConfig(level=logging.WARNING)

STARTING_EQUITY = 5_000.0
RISK_FRAC       = 0.01     # 1% equity risked per trade
SLIPPAGE_PCT    = 0.0005   # 0.05% per leg
COMMISSION      = 2.00     # $2 round-trip
SL_ATR_MULT     = 1.0
TIME_BARS       = 10

print("Loading data from cache...")
raw = load_data(symbols=list(COMBO_C_SYMBOLS), use_cache=True)
print(f"Loaded {len(raw)} symbols")

daily = {}
for sym, df in raw.items():
    daily[sym] = df.resample("1D").agg({
        "open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}
    ).dropna()

train_d, val_d, test_d = walk_forward_split(daily)
ref = "AMZN"
train_end  = train_d[ref].index[-1]
val_end    = val_d[ref].index[-1]
test_start = test_d[ref].index[0]
test_end   = test_d[ref].index[-1]
full_start = train_d[ref].index[0]

print(f"\nPeriods:")
print(f"  Train : {train_d[ref].index[0].date()} -> {train_end.date()}")
print(f"  Val   : {val_d[ref].index[0].date()} -> {val_end.date()}")
print(f"  Test  : {test_start.date()} -> {test_end.date()}")
print(f"\nStarting equity : ${STARTING_EQUITY:,.2f}  | Risk/trade: {RISK_FRAC*100:.1f}%  | Commission: ${COMMISSION:.2f} RT\n")

@dataclass
class Trade:
    sym: str; entry_date: str; entry_price: float; stop_loss: float
    take_profit: float; exit_date: str; exit_price: float; exit_reason: str
    shares: int; sl_dist: float; tp_dist: float; rr: float
    gross_pnl: float; commission: float; net_pnl: float; won: bool; period: str

def simulate_symbol(sym, df, equity_ref):
    ind = IndicatorStateV3()
    trades = []
    in_pos = False; pending = False; pending_data = {}
    entry_price = stop_loss = take_profit = 0.0
    shares = bars_held = 0; entry_date = ""
    sl_dist = tp_dist = rr_val = atr_entry = 0.0

    for i, (ts, row) in enumerate(df.iterrows()):
        o,h,l,c,vol = float(row.open),float(row.high),float(row.low),float(row.close),float(row.volume)
        ts = pd.Timestamp(ts)

        # Fill pending at this bar's open
        if pending and not in_pos:
            eq  = equity_ref[0]
            atr = pending_data["atr"]
            tp  = pending_data["bb_mid"]
            shares = max(1, int(eq * RISK_FRAC / atr)) if atr > 0 else 1
            shares = min(shares, max(1, int(eq * 0.20 / max(o, 0.01))))
            entry_price = o * (1 + SLIPPAGE_PCT)
            stop_loss   = entry_price - SL_ATR_MULT * atr
            take_profit = tp if tp > entry_price else entry_price * 1.015
            sl_dist = entry_price - stop_loss
            tp_dist = take_profit - entry_price
            rr_val  = tp_dist / sl_dist if sl_dist > 0 else 0.0
            atr_entry = atr
            in_pos = True; bars_held = 0
            entry_date = ts.strftime("%Y-%m-%d"); pending = False

        snap = ind.update(o, h, l, c, vol, 0.0)

        if in_pos:
            bars_held += 1
            exit_reason = ""; exit_price = 0.0
            atr_now = (snap.atr_10 or atr_entry or 1.0)
            accel_sl = (snap.bb_lower or 0) - SL_ATR_MULT * atr_now
            if c <= stop_loss:
                exit_reason = "STOP_LOSS"; exit_price = max(stop_loss, l)
            elif accel_sl > 0 and c <= accel_sl:
                exit_reason = "ACCEL_SL";  exit_price = max(accel_sl, l)
            elif snap.bb_mid and c >= snap.bb_mid:
                exit_reason = "TAKE_PROFIT"; exit_price = snap.bb_mid
            elif bars_held >= TIME_BARS:
                exit_reason = "TIME"; exit_price = c

            if exit_reason:
                slip = 1 - SLIPPAGE_PCT
                exit_fill = exit_price * slip
                gross = (exit_fill - entry_price) * shares
                net   = gross - COMMISSION
                period = "train" if ts <= train_end else ("val" if ts <= val_end else "test")
                equity_ref[0] = max(100.0, equity_ref[0] + net)
                trades.append(Trade(sym,entry_date,round(entry_price,2),round(stop_loss,2),
                    round(take_profit,2),ts.strftime("%Y-%m-%d"),round(exit_fill,2),
                    exit_reason,shares,round(sl_dist,2),round(tp_dist,2),round(rr_val,2),
                    round(gross,2),COMMISSION,round(net,2),net>0,period))
                in_pos = False; pending = False

        if not in_pos and not pending and snap.ready:
            if combo_c_entry(snap) == "LONG":
                pending = True
                pending_data = {
                    "atr"    : snap.atr_10 if snap.atr_10 and snap.atr_10>0 else (snap.atr or 1.0),
                    "bb_mid" : snap.bb_mid or (c * 1.02),
                }
    return trades

equity_ref = [STARTING_EQUITY]
all_trades = []
print("Running simulation...")
for sym in sorted(COMBO_C_SYMBOLS):
    if sym not in daily: continue
    t = simulate_symbol(sym, daily[sym], equity_ref)
    all_trades.extend(t)
    n=len(t); wr=sum(x.won for x in t)/n*100 if n else 0
    net=sum(x.net_pnl for x in t)
    print(f"  {sym:<6}: {n:3d} trades | Net P&L: ${net:>+8.2f} | WR: {wr:.0f}%")

all_trades.sort(key=lambda x: x.entry_date)

# rebuild equity + max drawdown
eq=STARTING_EQUITY; peak=eq; max_dd=0.0
for t in all_trades:
    eq += t.net_pnl
    peak = max(peak, eq)
    max_dd = max(max_dd, (peak-eq)/peak*100)

def print_summary(trades, label, start_eq):
    if not trades:
        print(f"\nNo trades in {label}"); return
    n=len(trades); wins=sum(t.won for t in trades); losses=n-wins
    wr=wins/n*100; net=sum(t.net_pnl for t in trades)
    gross=sum(t.gross_pnl for t in trades); comm=sum(t.commission for t in trades)
    ending=start_eq+net; ret=net/start_eq*100
    wp=[t.net_pnl for t in trades if t.won]; lp=[t.net_pnl for t in trades if not t.won]
    avg_w=np.mean(wp) if wp else 0; avg_l=np.mean(lp) if lp else 0
    best=max(t.net_pnl for t in trades); worst=min(t.net_pnl for t in trades)
    gw=sum(t.gross_pnl for t in trades if t.gross_pnl>0)
    gl=abs(sum(t.gross_pnl for t in trades if t.gross_pnl<=0))
    pf=gw/gl if gl>0 else float("inf")
    avg_sl=np.mean([t.sl_dist for t in trades]); avg_tp=np.mean([t.tp_dist for t in trades])
    avg_rr=np.mean([t.rr for t in trades]); avg_sh=np.mean([t.shares for t in trades])
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"  Starting capital: ${start_eq:>10,.2f}")
    print(f"{'═'*70}")
    print(f"  Trades:        {n:>5}    |  Wins: {wins:>3}   |  Losses: {losses:>3}")
    print(f"  Win Rate:      {wr:>6.1f}%  |  Avg shares/trade: {avg_sh:.1f}")
    print(f"  Profit Factor: {pf:>6.3f}")
    print(f"{'─'*70}")
    print(f"  Avg Stop Loss Dist:    ${avg_sl:>8.2f}  per share")
    print(f"  Avg Take Profit Dist:  ${avg_tp:>8.2f}  per share")
    print(f"  Avg R:R Ratio:              {avg_rr:.2f}x")
    print(f"{'─'*70}")
    print(f"  Gross P&L:      ${gross:>+11.2f}")
    print(f"  Commissions:    ${-comm:>+11.2f}")
    print(f"  Net P&L:        ${net:>+11.2f}")
    print(f"{'─'*70}")
    print(f"  Avg Win:        ${avg_w:>+11.2f}  |  Best Trade:  ${best:>+9.2f}")
    print(f"  Avg Loss:       ${avg_l:>+11.2f}  |  Worst Trade: ${worst:>+9.2f}")
    print(f"{'─'*70}")
    print(f"  ➤  Starting :  ${start_eq:>10,.2f}")
    print(f"  ➤  Ending   :  ${ending:>10,.2f}")
    print(f"  ➤  Net Return:     {ret:>+7.2f}%")
    print(f"{'═'*70}")
    exits=Counter(t.exit_reason for t in trades)
    print(f"\n  Exit Breakdown:")
    for r,cnt in sorted(exits.items(),key=lambda x:-x[1]):
        print(f"    {r:<16} {cnt:>4d}  ({cnt/n*100:>5.1f}%)")

test_trades=[t for t in all_trades if t.period=="test"]

print(f"\n{'═'*70}")
print(f"  PER-INSTRUMENT — Test Period ({test_start.date()} → {test_end.date()})")
print(f"{'═'*70}")
print(f"  {'Sym':<7} {'N':>4} {'WR%':>6} {'Avg SL/sh':>10} {'Avg TP/sh':>10} {'R:R':>5} {'Net P&L':>10}")
print(f"  {'─'*68}")
for sym in sorted(COMBO_C_SYMBOLS):
    st=[t for t in test_trades if t.sym==sym]
    if not st: print(f"  {sym:<7}  — (no signals)"); continue
    ns=len(st); wrs=sum(t.won for t in st)/ns*100; nets=sum(t.net_pnl for t in st)
    asl=np.mean([t.sl_dist for t in st]); atp=np.mean([t.tp_dist for t in st])
    arr=np.mean([t.rr for t in st])
    print(f"  {sym:<7} {ns:>4}  {wrs:>5.0f}%   ${asl:>7.2f}    ${atp:>7.2f}  {arr:>4.2f}x  ${nets:>+9.2f}")

print_summary(all_trades,  f"FULL BACKTEST  ({full_start.date()} → {test_end.date()})", STARTING_EQUITY)
print_summary(test_trades, f"TEST PERIOD ONLY  ({test_start.date()} → {test_end.date()})", STARTING_EQUITY)

if test_trades:
    print(f"\n  LAST 25 OOS TRADES")
    print(f"  {'#':>3} {'Entry Date':<12} {'Sym':<6} {'Entry $':>8} {'Stop $':>8} "
          f"{'Target $':>9} {'Exit $':>8} {'Shs':>4} {'Net P&L':>9} {'W/L':<3} Reason")
    print(f"  {'─'*92}")
    for i,t in enumerate(test_trades[-25:],1):
        wl="WIN" if t.won else "los"
        print(f"  {i:>3} {t.entry_date:<12} {t.sym:<6} "
              f"${t.entry_price:>7.2f} ${t.stop_loss:>7.2f} ${t.take_profit:>8.2f} "
              f"${t.exit_price:>7.2f} {t.shares:>4d} {t.net_pnl:>+9.2f}  {wl:<3} {t.exit_reason}")

print(f"\n  Max Drawdown (full run): {max_dd:.2f}%")
print(f"  Final Equity:           ${STARTING_EQUITY + sum(t.net_pnl for t in all_trades):,.2f}\n")
