#!/usr/bin/env python3
"""
place_combo_c_orders.py -- Place live Combo C signals on Alpaca paper account
=============================================================================
Reads today's Combo C signals, computes ATR-based position sizes (1% risk),
attaches stop-loss + take-profit bracket orders, and submits to Alpaca paper.
"""

from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data
from simulator import COMBO_C_SYMBOLS
from combos import combo_c_entry
from indicators_v3 import IndicatorStateV3

# ── Alpaca credentials ───────────────────────────────────────────────────────
API_KEY    = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_SECRET_KEY"]
BASE_URL   = "https://paper-api.alpaca.markets"
HEADERS    = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET,
              "Content-Type": "application/json"}

# ── Sizing constants (V6.0) ─────────────────────────────────────────────────
PORTFOLIO_VALUE = 100_000.0   # paper account portfolio value
RISK_FRAC       = 0.01        # 1% risk per trade
SL_ATR_MULT     = 1.0         # stop loss = entry - 1× ATR10
TP_ATR_MULT     = None        # TP = BB midline (not ATR-based)
MAX_POS_PCT     = 0.06        # cap at 6% of portfolio per instrument
MAX_CONCURRENT  = 6           # V6.0: 30% of 21 instruments (min 4, max 8)
# RSI threshold overrides per instrument (V6.0 Step 2)
RSI_THRESHOLD_OVERRIDES = {
    "SPLV": 12.0,   # SPLV only accepted at RSI<12
}
# High-correlation pairs (r > 0.70): reduce both positions to 60% when both signal
HIGH_CORR_PAIRS = [{"USMV", "SPLV"}]  # r=0.79

def get_account():
    r = requests.get(f"{BASE_URL}/v2/account", headers=HEADERS)
    return r.json()

def get_latest_quote(symbol: str) -> float:
    """Get latest ask price for market entry estimate."""
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
        headers=HEADERS,
        params={"feed": "iex"},
    )
    d = r.json()
    try:
        return float(d["quote"]["ap"])  # ask price
    except Exception:
        return 0.0

def place_bracket_order(symbol: str, qty: int, stop_price: float, tp_price: float) -> dict:
    """
    Place a market buy with bracket (stop-loss + take-profit).
    Uses Alpaca's bracket order type.
    """
    payload = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          "buy",
        "type":          "market",
        "time_in_force": "day",
        "order_class":   "bracket",
        "stop_loss": {
            "stop_price":  str(round(stop_price, 2)),
            "limit_price": str(round(stop_price * 0.995, 2)),  # 0.5% limit buffer
        },
        "take_profit": {
            "limit_price": str(round(tp_price, 2)),
        },
    }
    r = requests.post(f"{BASE_URL}/v2/orders", headers=HEADERS, data=json.dumps(payload))
    return r.json()

def scan_signals() -> list[dict]:
    """Scan all Combo C symbols for today's entry signals."""
    import combos as _combos
    raw = load_data(symbols=list(COMBO_C_SYMBOLS), use_cache=False)
    signals = []
    for sym in sorted(COMBO_C_SYMBOLS):
        if sym not in raw:
            continue
        df = raw[sym]
        daily = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        # Apply per-instrument RSI override
        orig_rsi = _combos.COMBO_C_RSI2_LEVEL
        _combos.COMBO_C_RSI2_LEVEL = RSI_THRESHOLD_OVERRIDES.get(sym, orig_rsi)
        ind = IndicatorStateV3()
        snap = None
        for _, row in daily.iterrows():
            snap = ind.update(float(row.open), float(row.high), float(row.low),
                              float(row.close), float(row.volume), 0.0)
        _combos.COMBO_C_RSI2_LEVEL = orig_rsi
        if snap and snap.ready and combo_c_entry(snap) == "LONG":
            signals.append({
                "sym":       sym,
                "close":     round(float(daily.iloc[-1].close), 2),
                "bb_mid":    round(snap.bb_mid, 2),
                "atr10":     round(snap.atr_10 or snap.atr or 1.0, 2),
                "rsi2":      round(snap.rsi2, 2),
                "bb_lower":  round(snap.bb_lower, 2),
            })
    return signals

# ── Main ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"  COMBO C ORDER PLACEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

acc = get_account()
portfolio_val = float(acc.get("portfolio_value", PORTFOLIO_VALUE))
buying_power  = float(acc.get("buying_power", 0))
print(f"  Account     : {acc.get('account_number')}")
print(f"  Portfolio   : ${portfolio_val:>10,.2f}")
print(f"  Buying Power: ${buying_power:>10,.2f}")
print(f"  Status      : {acc.get('status')}")
print()

print("  Scanning for Combo C signals...")
signals = scan_signals()

if not signals:
    print("  → No entry signals today. No orders placed.")
    sys.exit(0)

print(f"  → {len(signals)} signal(s) found:\n")

# V6.0: cap to MAX_CONCURRENT slots (sort by RSI2 ascending = strongest oversold first)
signals_sorted = sorted(signals, key=lambda x: x["rsi2"])
if len(signals_sorted) > MAX_CONCURRENT:
    print(f"  ⚠ {len(signals_sorted)} signals exceed MAX_CONCURRENT={MAX_CONCURRENT}.")
    print(f"    Taking top {MAX_CONCURRENT} by RSI2 (lowest = most oversold).")
    signals_sorted = signals_sorted[:MAX_CONCURRENT]

# Detect high-correlation pair conflicts
signal_syms = {s["sym"] for s in signals_sorted}
corr_reduced = set()
for pair in HIGH_CORR_PAIRS:
    if pair.issubset(signal_syms):
        print(f"  ⚠ High-corr pair {pair} both signaled → applying 0.6x size multiplier")
        corr_reduced |= pair

orders_placed = []
for sig in signals_sorted:
    sym      = sig["sym"]
    atr      = sig["atr10"]
    bb_mid   = sig["bb_mid"]
    corr_mult = 0.60 if sym in corr_reduced else 1.0

    # Get live ask for entry estimate
    ask = get_latest_quote(sym)
    if ask <= 0:
        ask = sig["close"]   # fallback to last close

    # Position sizing: shares = floor(portfolio * risk_frac / ATR) × corr_mult
    qty = max(1, int(portfolio_val * RISK_FRAC * corr_mult / atr))
    # Cap at 6% of portfolio
    max_qty = max(1, int(portfolio_val * MAX_POS_PCT / ask))
    qty = min(qty, max_qty)

    stop_price = round(ask - SL_ATR_MULT * atr, 2)
    tp_price   = round(bb_mid, 2)

    # Ensure tp > ask and stop < ask
    if tp_price <= ask:
        tp_price = round(ask * 1.015, 2)
    if stop_price >= ask:
        stop_price = round(ask * 0.985, 2)

    risk_dollar  = qty * (ask - stop_price)
    tp_dollar    = qty * (tp_price - ask)
    rr           = tp_dollar / risk_dollar if risk_dollar > 0 else 0

    print(f"  ┌─ {sym}")
    print(f"  │  RSI2={sig['rsi2']:.1f}  |  BB_lower=${sig['bb_lower']:.2f}  |  Close=${sig['close']:.2f}")
    print(f"  │  Ask     : ${ask:.2f}")
    print(f"  │  Stop    : ${stop_price:.2f}  (−${ask-stop_price:.2f}/sh, ATR={atr:.2f})")
    print(f"  │  Target  : ${tp_price:.2f}  (+${tp_price-ask:.2f}/sh, BB mid)")
    print(f"  │  Qty     : {qty} shares")
    print(f"  │  Risk $  : ${risk_dollar:.2f}  ({risk_dollar/portfolio_val*100:.2f}% of portfolio)")
    print(f"  │  R:R     : {rr:.2f}x")
    print(f"  └─ Placing bracket order...")

    result = place_bracket_order(sym, qty, stop_price, tp_price)

    if result.get("id"):
        print(f"     ✅ ORDER PLACED | ID: {result['id'][:8]}... | Status: {result.get('status')}")
        orders_placed.append({
            "symbol": sym, "qty": qty, "entry_est": ask,
            "stop": stop_price, "target": tp_price,
            "order_id": result["id"], "status": result.get("status"),
        })
    else:
        print(f"     ❌ ORDER FAILED: {result.get('message', result)}")
    print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"  SUMMARY: {len(orders_placed)}/{len(signals)} orders placed successfully")
print("=" * 65)
for o in orders_placed:
    print(f"  {o['symbol']:<6}  {o['qty']:>3} shares | "
          f"Stop: ${o['stop']:.2f} | Target: ${o['target']:.2f} | "
          f"Status: {o['status']}")
print()
