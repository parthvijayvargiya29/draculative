#!/usr/bin/env python3
"""
run_variants.py -- Phase 1 Pre-Live Variant Comparison for Combo C
===================================================================
Runs 5 variants of Combo C against the validated walk-forward split:

  Baseline : RSI<15, no volume gate, static ACCEL_SL (bb_lower - 1×ATR)   [validated]
  V1       : RSI<10, no volume gate, static ACCEL_SL                       [tighter RSI]
  V2       : RSI<5,  no volume gate, static ACCEL_SL                       [tightest RSI]
  V3       : RSI<15, volume<0.8×avg gate, static ACCEL_SL                  [Pine Script vol gate]
  V4       : RSI<15, no volume gate, ratcheting ACCEL_SL                   [Pine Script ratchet SL]

Exit rules shared across all variants (matches actual backtest):
  1. ACCEL_SL: static floor = live bb_lower - 1×ATR(10)  [or ratcheting in V4]
  2. BB_MID: close >= bb_mid
  3. TIME: bars_held >= 10
  Simulator layer 0: hard stop_loss from entry (entry_price - 1×ATR) → "SL"

CRITICAL BASELINE CLARIFICATION:
  - COMBO_C_RSI2_LEVEL = 15.0  (docstring in combos.py says "10" -- that is STALE)
  - No volume gate in combo_c_entry() (removed in V3.4 -- anti-correlated with bb_lower breaks)
  - "ACCEL_SL" in backtest trade records = dynamic floor: live bb_lower - 1×ATR(10) per bar
    This is NOT ratcheting. V4 tests true ratcheting for Pine Script alignment.

Usage:
    cd trading_system/backtest_v3
    export ALPACA_API_KEY="..."
    export ALPACA_SECRET_KEY="..."
    export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
    ../../.venv/bin/python run_variants.py --alpaca --daily [--no-cache]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data
from simulator import TradeRecord, run_combo_on_all_symbols, walk_forward_split
from indicators_v3 import BarSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = [
    {
        "name":        "Baseline",
        "rsi_level":   15.0,
        "vol_gate":    False,
        "ratchet_sl":  False,
        "description": "RSI<15, no vol gate, static floor (VALIDATED)",
    },
    {
        "name":        "V1_RSI10",
        "rsi_level":   10.0,
        "vol_gate":    False,
        "ratchet_sl":  False,
        "description": "RSI<10, no vol gate, static floor",
    },
    {
        "name":        "V2_RSI5",
        "rsi_level":   5.0,
        "vol_gate":    False,
        "ratchet_sl":  False,
        "description": "RSI<5, no vol gate, static floor",
    },
    {
        "name":        "V3_VolGate",
        "rsi_level":   15.0,
        "vol_gate":    True,
        "vol_mult":    0.8,
        "vol_len":     20,
        "ratchet_sl":  False,
        "description": "RSI<15, vol<0.8x20avg gate, static floor",
    },
    {
        "name":        "V4_RatchetSL",
        "rsi_level":   15.0,
        "vol_gate":    False,
        "ratchet_sl":  True,
        "description": "RSI<15, no vol gate, ratcheting ACCEL_SL",
    },
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 Variant Comparison -- Combo C")
    p.add_argument("--capital",  type=float, default=25_000)
    p.add_argument("--alpaca",   action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--daily",    action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Daily resampling (mirrors run_v3.py)
# ---------------------------------------------------------------------------

def resample_to_daily(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym, df in data.items():
        if df.empty:
            out[sym] = df
            continue
        daily = df.resample("D").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["open", "close"])
        daily = daily[daily["volume"] > 0]
        out[sym] = daily
    return out


# ---------------------------------------------------------------------------
# Patched combo_c_entry / combo_c_exit factories
# ---------------------------------------------------------------------------

def make_combo_c_entry(rsi_level: float, vol_gate: bool,
                       vol_mult: float = 0.8, vol_len: int = 20):
    """Return a combo_c_entry function with patched parameters."""
    def combo_c_entry_patched(snap: BarSnapshot):
        if not snap.ready:
            return None

        below_lower = snap.close < snap.bb_lower
        rsi2_sold   = snap.rsi2 < rsi_level

        if vol_gate:
            # Volume gate: require volume below vol_mult × vol_len-bar average.
            # snap.vol_ratio is computed as volume / vol_20_avg in indicators_v3.
            vol_ok = snap.vol_ratio < vol_mult
            if not (below_lower and rsi2_sold and vol_ok):
                return None
        else:
            if not (below_lower and rsi2_sold):
                return None

        return "LONG"
    return combo_c_entry_patched


def make_combo_c_exit(ratchet_sl: bool, sl_atr_mult: float = 1.0, time_bars: int = 10):
    """
    Return a combo_c_exit function.
    ratchet_sl=False: static floor = live bb_lower - sl_atr_mult×ATR(10)  [current backtest]
    ratchet_sl=True:  ratcheting stop -- arms on first profitable close,
                      then floor ratchets up to max(accel_sl, prev_accel_sl)
                      (never moves down). Matches Pine Script ACCEL_SL intent.
    """
    # Ratchet state per position -- stored in closure as mutable dict
    # Reset each time a new position is detected via entry_price change.
    _state = {"entry_price": None, "ratchet_level": None}

    if not ratchet_sl:
        def combo_c_exit_static(entry_price: float, snap: BarSnapshot,
                                bars_held: int, direction: str):
            _HOLD = (None, 0.0)
            c   = snap.close
            atr = snap.atr_10 if snap.atr_10 > 0 else (snap.atr if snap.atr > 0 else entry_price * 0.005)
            if direction == "LONG":
                accel_sl = snap.bb_lower - sl_atr_mult * atr
                if c <= accel_sl:    return ("ACCEL_SL", accel_sl)
                if c >= snap.bb_mid: return ("BB_MID", c)
            if bars_held >= time_bars:
                return ("TIME", c)
            return _HOLD
        return combo_c_exit_static
    else:
        def combo_c_exit_ratchet(entry_price: float, snap: BarSnapshot,
                                 bars_held: int, direction: str):
            _HOLD = (None, 0.0)
            c   = snap.close
            atr = snap.atr_10 if snap.atr_10 > 0 else (snap.atr if snap.atr > 0 else entry_price * 0.005)

            if direction == "LONG":
                # Reset ratchet when a new trade starts
                if _state["entry_price"] != entry_price:
                    _state["entry_price"]    = entry_price
                    _state["ratchet_level"]  = None   # unarmed

                # Arm: first bar where close > entry_price
                if _state["ratchet_level"] is None:
                    if c > entry_price:
                        # Arm at static floor level on first profitable bar
                        _state["ratchet_level"] = snap.bb_lower - sl_atr_mult * atr
                    else:
                        # Not yet profitable -- use static floor as hard stop
                        hard_floor = snap.bb_lower - sl_atr_mult * atr
                        if c <= hard_floor:
                            return ("ACCEL_SL", hard_floor)
                else:
                    # Ratchet: advance floor upward only
                    new_level = snap.bb_lower - sl_atr_mult * atr
                    _state["ratchet_level"] = max(_state["ratchet_level"], new_level)
                    if c <= _state["ratchet_level"]:
                        return ("ACCEL_SL", _state["ratchet_level"])

                if c >= snap.bb_mid:
                    return ("BB_MID", c)

            if bars_held >= time_bars:
                return ("TIME", c)
            return _HOLD
        return combo_c_exit_ratchet


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_pf(trades: List[TradeRecord]) -> float:
    if not trades:
        return 0.0
    wins  = sum(t.net_pnl for t in trades if t.won)
    loss  = abs(sum(t.net_pnl for t in trades if not t.won))
    return round(wins / loss, 3) if loss > 0 else float("inf")


def compute_wr(trades: List[TradeRecord]) -> float:
    if not trades:
        return 0.0
    return round(sum(t.won for t in trades) / len(trades) * 100, 1)


def compute_wfe(test_pf: float, train_pf: float) -> float:
    """Walk-forward efficiency: test_pf / train_pf."""
    if train_pf <= 0:
        return 0.0
    return round(test_pf / train_pf, 3)


def period_trades(trades: List[TradeRecord], period: str) -> List[TradeRecord]:
    return [t for t in trades if t.period == period]


def exit_breakdown(trades: List[TradeRecord]) -> str:
    """Compact exit reason summary: BB_MID/SL/ACCEL/TIME counts."""
    if not trades:
        return "—"
    from collections import Counter
    c = Counter(t.exit_reason for t in trades)
    parts = []
    for key in ("BB_MID", "SL", "ACCEL_SL", "TIME"):
        if key in c:
            parts.append(f"{key}:{c[key]}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

def run_variant(variant: dict, data: Dict[str, pd.DataFrame],
                initial_capital: float) -> dict:
    """
    Monkey-patch combos module and run walk-forward for one variant.

    simulator.py does:
        from combos import combo_c_entry, exit_signal
    so patching combos.combo_c_entry is NOT enough — we must also patch
    simulator.combo_c_entry and simulator.exit_signal (the names bound in
    simulator's module namespace at import time).

    Returns a result dict with per-period stats.
    """
    import combos
    import simulator

    # Build patched functions for this variant
    patched_entry = make_combo_c_entry(
        rsi_level=variant["rsi_level"],
        vol_gate=variant["vol_gate"],
        vol_mult=variant.get("vol_mult", 0.8),
        vol_len=variant.get("vol_len", 20),
    )
    patched_exit_fn = make_combo_c_exit(ratchet_sl=variant["ratchet_sl"])

    # Save originals for restore
    orig_combos_entry  = combos.combo_c_entry
    orig_combos_exit   = combos.combo_c_exit
    orig_combos_signal = combos.exit_signal
    orig_sim_entry     = simulator.combo_c_entry
    orig_sim_signal    = simulator.exit_signal

    # Build patched exit_signal dispatcher (routes Combo C to patched exit)
    def patched_exit_signal(combo, entry_price, snap, bars_held, direction,
                            atr_at_entry=0.0, tp_price=0.0, sl_price=0.0):
        if combo == "C":
            return patched_exit_fn(entry_price, snap, bars_held, direction)
        return orig_combos_signal(combo, entry_price, snap, bars_held, direction,
                                  atr_at_entry=atr_at_entry,
                                  tp_price=tp_price, sl_price=sl_price)

    # Patch both combos module AND simulator module namespace
    combos.combo_c_entry  = patched_entry
    combos.combo_c_exit   = patched_exit_fn
    combos.exit_signal    = patched_exit_signal
    simulator.combo_c_entry = patched_entry    # bound name in simulator's namespace
    simulator.exit_signal   = patched_exit_signal

    try:
        # Walk-forward split (for period boundaries + sliced train data)
        train_data, val_data, test_data = walk_forward_split(data)

        # Reference symbol for period start timestamps
        _ref_sym  = next(iter(train_data))
        _vl_start = val_data[_ref_sym].index[0]
        _te_start = test_data[_ref_sym].index[0]
        _vl_end   = val_data[_ref_sym].index[-1]

        # Train: sliced data (435 daily bars >> 80 warmup needed)
        all_train_trades, _, _, _ = run_combo_on_all_symbols(
            train_data, "C", initial_capital, period_label="train")

        # Validate: warm-start — pass full dataset, record from val start
        # (145-bar val window; indicators not warm if split independently)
        all_val_raw, _, _, _ = run_combo_on_all_symbols(
            data, "C", initial_capital, period_label="val", active_from=_vl_start)
        all_val_trades = [t for t in all_val_raw if t.entry_bar <= _vl_end]

        # Test (OOS): warm-start — pass full dataset, record from test start
        # This exactly mirrors run_v3.py's authoritative OOS evaluation
        all_test_trades, _, _, _ = run_combo_on_all_symbols(
            data, "C", initial_capital, period_label="test", active_from=_te_start)

    finally:
        # Always restore originals even if run fails
        combos.combo_c_entry  = orig_combos_entry
        combos.combo_c_exit   = orig_combos_exit
        combos.exit_signal    = orig_combos_signal
        simulator.combo_c_entry = orig_sim_entry
        simulator.exit_signal   = orig_sim_signal

    all_trades = all_train_trades + all_val_trades + all_test_trades

    train_pf   = compute_pf(all_train_trades)
    val_pf     = compute_pf(all_val_trades)
    test_pf    = compute_pf(all_test_trades)
    overall_pf = compute_pf(all_trades)
    wfe        = compute_wfe(test_pf, train_pf)

    return {
        "name":        variant["name"],
        "rsi":         variant["rsi_level"],
        "vol_gate":    "YES" if variant["vol_gate"] else "NO",
        "sl_type":     "Ratchet" if variant["ratchet_sl"] else "Static",
        "description": variant["description"],
        "n_total":     len(all_trades),
        "n_train":     len(all_train_trades),
        "n_val":       len(all_val_trades),
        "n_test":      len(all_test_trades),
        "train_pf":    train_pf,
        "val_pf":      val_pf,
        "test_pf":     test_pf,
        "overall_pf":  overall_pf,
        "wfe":         wfe,
        "train_wr":    compute_wr(all_train_trades),
        "val_wr":      compute_wr(all_val_trades),
        "test_wr":     compute_wr(all_test_trades),
        "test_exits":  exit_breakdown(all_test_trades),
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[dict]):
    sep  = "=" * 120
    sep2 = "-" * 120

    print(f"\n{sep}")
    print("  PHASE 1 VARIANT COMPARISON — COMBO C  (Baseline = validated 131-trade run)")
    print(sep)
    print(f"\n  {'Variant':<14} {'RSI':<6} {'VolGate':<9} {'SL_Type':<10} "
          f"{'N_Total':<9} {'N_Test':<8} "
          f"{'Train_PF':<10} {'Val_PF':<9} {'Test_PF':<9} {'Overall_PF':<12} "
          f"{'WFE':<8} {'Test_WR':<9}")
    print(f"  {sep2}")

    for r in results:
        marker = " ← VALIDATED" if r["name"] == "Baseline" else ""
        print(f"  {r['name']:<14} {r['rsi']:<6} {r['vol_gate']:<9} {r['sl_type']:<10} "
              f"{r['n_total']:<9} {r['n_test']:<8} "
              f"{r['train_pf']:<10.3f} {r['val_pf']:<9.3f} {r['test_pf']:<9.3f} "
              f"{r['overall_pf']:<12.3f} {r['wfe']:<8.3f} {r['test_wr']:<9.1f}%"
              f"{marker}")

    print(f"\n  {'Variant':<14} {'Description':<45} {'Test exits (N)'}")
    print(f"  {sep2}")
    for r in results:
        print(f"  {r['name']:<14} {r['description']:<45} {r['test_exits']}")

    print(f"\n{sep}")
    print("  PHASE 1 DECISION CRITERIA")
    print(sep)
    print("""
  Lock parameters if:
    (a) Baseline holds: Test_PF ≥ 1.10, WFE ≥ 0.65, N_Test ≥ 20
    (b) Tighter RSI (V1/V2): Test_PF improvement > +0.10 AND N_Test ≥ 15
        → Prefer RSI<10 or RSI<5 only if signal quality clearly improves with adequate sample
    (c) Volume gate (V3): Test_PF improvement > +0.10 over Baseline
        → Else: remove volume gate from Pine Script to match validated backtest
    (d) Ratchet SL (V4): Test_PF improvement > +0.10 over Baseline
        → Else: replace Pine Script ratcheting ACCEL_SL with static floor version
  """)

    # Determine winner
    valid = [r for r in results if r["n_test"] >= 15]
    if valid:
        best = max(valid, key=lambda r: r["test_pf"])
        print(f"\n  Best by Test_PF (N_Test≥15):  {best['name']}")
        print(f"  → Test_PF={best['test_pf']:.3f}, WFE={best['wfe']:.3f}, "
              f"N_Test={best['n_test']}, N_Total={best['n_total']}")

    baseline = next((r for r in results if r["name"] == "Baseline"), None)
    if baseline:
        print(f"\n  Baseline reference:  Test_PF={baseline['test_pf']:.3f}, "
              f"WFE={baseline['wfe']:.3f}, N_Test={baseline['n_test']}")

    print()


# ---------------------------------------------------------------------------
# Phase 1 decision statement generator
# ---------------------------------------------------------------------------

def print_phase1_decision(results: List[dict]):
    baseline = next((r for r in results if r["name"] == "Baseline"), None)
    if not baseline:
        return

    best_test_pf_result = max(results, key=lambda r: r["test_pf"] if r["n_test"] >= 15 else 0)
    winner = best_test_pf_result

    print("=" * 120)
    print("  PHASE 1 DECISION STATEMENT (auto-generated)")
    print("=" * 120)

    locked_rsi    = winner["rsi"]
    locked_vol    = winner["vol_gate"]
    locked_sl     = winner["sl_type"]

    improvement_vs_baseline = winner["test_pf"] - baseline["test_pf"]
    changed = winner["name"] != "Baseline"

    if changed and improvement_vs_baseline > 0.10:
        print(f"""
  PARAMETERS UPDATED from Baseline:
    RSI threshold : {locked_rsi}  (was 15.0)
    Volume gate   : {locked_vol}  (was NO)
    ACCEL_SL type : {locked_sl}  (was Static)

  Rationale: {winner['name']} improved Test_PF by +{improvement_vs_baseline:.3f} over Baseline
  ({baseline['test_pf']:.3f} → {winner['test_pf']:.3f}) with N_Test={winner['n_test']} trades.
  WFE={winner['wfe']:.3f} indicates {'acceptable' if winner['wfe'] >= 0.65 else 'marginal'} OOS persistence.

  ACTION: Update COMBO_C_RSI2_LEVEL={locked_rsi} in combos.py.
          {'Add volume gate to combo_c_entry().' if locked_vol == 'YES' else 'Volume gate ABSENT — remove from Pine Script.'}
          {'Add ratcheting ACCEL_SL to combo_c_exit().' if locked_sl == 'Ratchet' else 'Use static floor ACCEL_SL — fix Pine Script to match.'}
""")
    else:
        print(f"""
  PARAMETERS UNCHANGED — Baseline confirmed as optimal:
    RSI threshold : 15.0 (unchanged)
    Volume gate   : NO   (absent from backtest; REMOVE from Pine Script)
    ACCEL_SL type : Static floor = live bb_lower - 1×ATR(10)
                    (fix Pine Script ratcheting ACCEL_SL to use static version)

  Rationale: No variant improved Test_PF by the required +0.10 threshold over Baseline
  (Baseline Test_PF={baseline['test_pf']:.3f}, best variant={best_test_pf_result['name']} Test_PF={best_test_pf_result['test_pf']:.3f}).
  The validated 131-trade run (Aug 2025–Mar 2026 test window) stands as the locked configuration.

  Pine Script corrections required (2 discrepancies):
    1. Remove volume gate from entry condition (not in backtest)
    2. Replace ratcheting ACCEL_SL with: accel_sl = bb_lower - 1.0 × atr(10)  [static per-bar]
""")
    print("=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Determine if Alpaca keys are present (same logic as run_v3.py)
    import os
    has_alpaca_key = bool(os.environ.get("ALPACA_API_KEY", ""))
    use_synth = not args.alpaca or not has_alpaca_key

    logger.info("Loading data...")
    data = load_data(
        use_cache   = not args.no_cache,
        force_synth = use_synth,
        universe    = "all",
    )

    if args.daily:
        logger.info("Resampling to daily bars...")
        data = resample_to_daily(data)

    results = []
    for i, variant in enumerate(VARIANTS, 1):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Running variant {i}/{len(VARIANTS)}: {variant['name']}")
        logger.info(f"  {variant['description']}")
        logger.info(f"{'─'*60}")
        r = run_variant(variant, data, args.capital)
        results.append(r)
        logger.info(f"  DONE: Test_PF={r['test_pf']:.3f}, WFE={r['wfe']:.3f}, "
                    f"N_Test={r['n_test']}, N_Total={r['n_total']}")

    print_comparison_table(results)
    print_phase1_decision(results)


if __name__ == "__main__":
    main()
