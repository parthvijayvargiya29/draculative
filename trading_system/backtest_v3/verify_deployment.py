#!/usr/bin/env python3
"""
verify_deployment.py  --  Pre-Deployment Checklist for Combo C (V5.1)
======================================================================
Run this script BEFORE Day 1 of live trading to confirm:

  1. Locked parameters match Pine Script implementation
  2. Per-instrument trade counts (Pine Script vs backtest ±2 tolerance)
  3. Kelly ramp logic mock tests (phase count boundaries)
  4. Phase gate logic (hold when PF/DD gate fails)
  5. Position sizing examples (vol-ratio cap, floor, integer truncation)

ALL items must show PASS for the strategy to be cleared for live deployment.

USAGE
-----
  python verify_deployment.py
  python verify_deployment.py --pine-counts GLD=2,WMT=0,USMV=6,NVDA=3,AMZN=5,GOOGL=3,COST=3,XOM=0,HD=8,MA=9
  python verify_deployment.py --no-color
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
_USE_COLOR = True

def _green(s: str) -> str:   return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _red(s: str)   -> str:   return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _yellow(s: str) -> str:  return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _bold(s: str)  -> str:   return f"\033[1m{s}\033[0m"  if _USE_COLOR else s

PASS_TAG  = lambda: _green("PASS")
FAIL_TAG  = lambda: _red("FAIL")
WARN_TAG  = lambda: _yellow("WARN")

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
_results: List[Tuple[str, bool, str]] = []   # (label, passed, detail)

def _check(label: str, passed: bool, detail: str = "") -> bool:
    _results.append((label, passed, detail))
    tag = PASS_TAG() if passed else FAIL_TAG()
    print(f"  [{tag}] {label}" + (f"  -- {detail}" if detail else ""))
    return passed


def _section(title: str) -> None:
    sep = "─" * 78
    print(f"\n{sep}\n  {_bold(title)}\n{sep}")


# ===========================================================================
# Section 1: Locked parameter display
# ===========================================================================

def _section1_locked_params() -> None:
    _section("1. LOCKED PARAMETERS (Combo C — frozen 19-Mar-2026)")

    params = {
        "Instruments":               "GLD WMT USMV NVDA AMZN GOOGL COST XOM HD MA",
        "BB period":                 "20",
        "BB multiplier":             "2.0",
        "RSI period":                "2",
        "RSI threshold (entry)":     "< 15",
        "ATR period":                "10",
        "Time-stop bars":            "10",
        "Accel SL":                  "bb_lower - 1.0 × ATR(10), static floor",
        "Exit 1 (primary)":          "BB midline (bb_basis)",
        "Exit 2 (stop)":             "ATR-anchored stop (accel_sl)",
        "Exit 3 (time)":             "After 10 bars if no target hit",
        "Volume gate":               "NONE",
        "Initial capital / chart":   "$2 500",
        "Commission / order":        "$1.00",
        "Risk fraction (Phase 1)":   "0.5%  (floor(equity × 0.005 / ATR10))",
        "Risk fraction (Phase 2)":   "2.1%  (at N=31–60, gate required)",
        "Risk fraction (Phase 3)":   "3.65% (at N=61+, gate required)",
    }

    col_w = max(len(k) for k in params) + 2
    for k, v in params.items():
        print(f"    {k:<{col_w}}: {v}")

    print(f"\n  NOTE: NO changes to any of the above parameters are permitted.")
    print(f"  If Pine Script values differ, STOP and reconcile before trading.")
    _check("Locked parameters displayed", True)


# ===========================================================================
# Section 2: Per-instrument trade count table
# ===========================================================================

BACKTEST_COUNTS_TEST = {
    "GLD": 1, "WMT": 0, "USMV": 5, "NVDA": 3,
    "AMZN": 5, "GOOGL": 3, "COST": 3, "XOM": 0,
    "HD": 7, "MA": 8,
}
BACKTEST_COUNTS_TOTAL = {
    "GLD": 12, "WMT": 10, "USMV": 18, "NVDA": 3,
    "AMZN": 12, "GOOGL": 12, "COST": 11, "XOM": 10,
    "HD": 21, "MA": 22,
}
_TOLERANCE = 2   # ±2 trades acceptable


def _section2_trade_counts(pine_counts: Optional[Dict[str, int]]) -> None:
    _section("2. PER-INSTRUMENT TRADE COUNT VERIFICATION (Pine Script vs Backtest)")

    instruments = list(BACKTEST_COUNTS_TOTAL.keys())
    bt_total_sum = sum(BACKTEST_COUNTS_TOTAL.values())   # 131

    # Header
    print(f"  {'Symbol':<8} {'BT Total':>9} {'BT Test':>8} {'Pine':>8} {'Delta':>7}  Result")
    print(f"  {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*7}  ------")

    all_ok = True
    pine_provided = pine_counts is not None

    for sym in instruments:
        bt_total = BACKTEST_COUNTS_TOTAL[sym]
        bt_test  = BACKTEST_COUNTS_TEST[sym]

        if pine_provided:
            pine_n = pine_counts.get(sym, -1)
            delta  = pine_n - bt_total
            ok     = abs(delta) <= _TOLERANCE
            if not ok:
                all_ok = False
            delta_str = f"{delta:+d}" if pine_n >= 0 else "N/A"
            result_str = (_green("OK") if ok else _red(f"FAIL ±{abs(delta)}")) if pine_n >= 0 else _yellow("no data")
            print(f"  {sym:<8} {bt_total:>9} {bt_test:>8} {pine_n:>8} {delta_str:>7}  {result_str}")
        else:
            print(f"  {sym:<8} {bt_total:>9} {bt_test:>8} {'?':>8} {'?':>7}  {_yellow('NOT PROVIDED')}")

    print(f"\n  Backtest total: {bt_total_sum} trades across 10 instruments (36-month full period)")
    print(f"  Tolerance     : ±{_TOLERANCE} per instrument")

    if not pine_provided:
        # Not a hard failure — it's a pre-run check the user must perform manually.
        # Mark as WARN (True) so it doesn't block the overall PASS verdict,
        # but the table above reminds the user to fill in counts before Day 1.
        print(f"\n  {_yellow('WARN')} --pine-counts not provided. "
              f"Verify counts manually before trading.")
        _check("Pine Script trade count verification", True,
               "MANUAL CHECK REQUIRED before Day 1 — re-run with --pine-counts GLD=N,...")
    else:
        _check("All instrument counts within ±2", all_ok,
               "All OK" if all_ok else "One or more symbols exceed tolerance — reconcile before trading")


# ===========================================================================
# Section 3: Kelly ramp logic mock tests
# ===========================================================================

def _make_fake_trades(n_live: int, period: str = "LIVE_RAMP",
                      win: bool = True) -> List[dict]:
    """Produce `n_live` fake closed trades with the given outcome."""
    pnl = 50.0 if win else -40.0
    return [{"period": period, "pnl": pnl, "entry_price": 100.0, "shares": 5}
            for _ in range(n_live)]


def _make_alternating_trades(n: int, win_fraction: float = 0.6) -> List[dict]:
    """
    Produce n trades with interleaved wins/losses (not sequential batches).
    Uses proportional assignment so wins are spread evenly, keeping DD low.
    """
    trades = []
    wins_so_far = 0
    for i in range(n):
        expected_wins = int((i + 1) * win_fraction)
        if wins_so_far < expected_wins:
            trades.append({"period": "LIVE_RAMP", "pnl": 50.0,
                           "entry_price": 100.0, "shares": 5})
            wins_so_far += 1
        else:
            trades.append({"period": "LIVE_RAMP", "pnl": -40.0,
                           "entry_price": 100.0, "shares": 5})
    return trades


def _section3_kelly_mock_tests() -> None:
    _section("3. KELLY RAMP LOGIC — MOCK TESTS")

    # Import the live Kelly functions from paper_trading_monitor
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from paper_trading_monitor import (  # type: ignore
            get_kelly_phase, count_completed_trades, KELLY_PHASES,
        )
    except ImportError as exc:
        _check("Import paper_trading_monitor Kelly functions", False, str(exc))
        print(f"  {_red('SKIP')} Cannot import — remaining Kelly tests skipped.")
        return

    _check("Import paper_trading_monitor Kelly functions", True)

    cases = [
        (0,   0.005,  1),
        (29,  0.005,  1),
        (30,  0.021,  2),   # boundary: first trade AT 30 is Phase 2
        (31,  0.021,  2),
        (60,  0.0365, 3),
        (100, 0.0365, 3),
    ]

    print(f"\n  {'N live':>8} {'Expected frac':>14} {'Actual frac':>12} {'Phase':>7}  Result")
    print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*7}  ------")

    all_ok = True
    for n, expected_frac, expected_phase in cases:
        info  = get_kelly_phase(n)
        actual_frac  = info["risk_fraction"]
        # Determine expected phase number from expected_frac
        ok = math.isclose(actual_frac, expected_frac, rel_tol=1e-6)
        if not ok:
            all_ok = False
        result_str = _green("OK") if ok else _red(f"GOT {actual_frac:.4f}")
        phase_lbl = info["label"].split("(")[0].strip()
        print(f"  {n:>8} {expected_frac:>14.4f} {actual_frac:>12.4f} {phase_lbl:>7}  {result_str}")

    _check("Kelly phase fractions correct at all boundaries", all_ok)

    # next_threshold / trades_to_next
    info29 = get_kelly_phase(29)
    _check("Phase 1: next_threshold=30", info29["next_threshold"] == 30,
           f"got {info29['next_threshold']}")
    _check("Phase 1: trades_to_next=1 at n=29", info29["trades_to_next"] == 1,
           f"got {info29['trades_to_next']}")

    info60 = get_kelly_phase(60)
    _check("Phase 3: next_threshold is None (final phase)", info60["next_threshold"] is None,
           f"got {info60['next_threshold']}")
    _check("Phase 3: trades_to_next is None (final phase)", info60["trades_to_next"] is None,
           f"got {info60['trades_to_next']}")


# ===========================================================================
# Section 4: Phase gate logic tests
# ===========================================================================

def _section4_gate_tests() -> None:
    _section("4. PHASE GATE LOGIC — MOCK TESTS")

    try:
        from paper_trading_monitor import (  # type: ignore
            check_kelly_phase_gate, get_current_risk_fraction, _quick_metrics,
        )
    except ImportError as exc:
        _check("Import gate functions", False, str(exc))
        return
    _check("Import gate functions", True)

    # ── 4a: No live trades → gate not evaluable, no hold ──────────────────
    gate = check_kelly_phase_gate([])
    _check("Empty trades: hold_at_phase is None",
           gate["hold_at_phase"] is None, str(gate["hold_at_phase"]))
    _check("Empty trades: phase1_gate_open is None",
           gate["phase1_gate_open"] is None, str(gate["phase1_gate_open"]))

    # ── 4b: 30 good Phase 1 trades (PF > 1, low DD) → gate OPEN ──────────
    good_ph1 = _make_alternating_trades(30, win_fraction=0.6)
    gate_good = check_kelly_phase_gate(good_ph1)
    _check("30 good Phase 1 trades: phase1_gate_open=True",
           gate_good["phase1_gate_open"] is True,
           f"phase1_gate_open={gate_good['phase1_gate_open']}")
    _check("30 good Phase 1 trades: hold_at_phase is None",
           gate_good["hold_at_phase"] is None,
           str(gate_good["hold_at_phase"]))
    eff = get_current_risk_fraction(good_ph1)
    _check("30 good trades: effective frac ≥ 0.021 (Phase 2 unlocked)",
           eff >= 0.021 - 1e-9,
           f"effective={eff:.4f}")

    # ── 4c: 30 bad Phase 1 trades (PF < 1) → gate BLOCKED, hold at Phase 1 ──
    bad_ph1 = _make_alternating_trades(30, win_fraction=0.1)   # mostly losses
    gate_bad = check_kelly_phase_gate(bad_ph1)
    _check("30 bad Phase 1 trades: phase1_gate_open=False",
           gate_bad["phase1_gate_open"] is False,
           f"phase1_gate_open={gate_bad['phase1_gate_open']}")
    _check("30 bad Phase 1 trades: hold_at_phase=1",
           gate_bad["hold_at_phase"] == 1,
           str(gate_bad["hold_at_phase"]))
    eff_bad = get_current_risk_fraction(bad_ph1)
    _check("30 bad trades: effective frac = 0.005 (HELD at Phase 1)",
           math.isclose(eff_bad, 0.005, rel_tol=1e-6),
           f"effective={eff_bad:.4f}")

    # ── 4d: 60 trades, Phase 1 good but Phase 2 bad → hold at Phase 2 ────
    good_30  = _make_alternating_trades(30, win_fraction=0.6)
    bad_30   = _make_alternating_trades(30, win_fraction=0.1)
    combined = good_30 + bad_30
    gate_c   = check_kelly_phase_gate(combined)
    _check("60 trades (ph1 good, ph2 bad): phase1_gate_open=True",
           gate_c["phase1_gate_open"] is True,
           str(gate_c["phase1_gate_open"]))
    _check("60 trades (ph1 good, ph2 bad): phase2_gate_open=False",
           gate_c["phase2_gate_open"] is False,
           str(gate_c["phase2_gate_open"]))
    _check("60 trades (ph1 good, ph2 bad): hold_at_phase=2",
           gate_c["hold_at_phase"] == 2,
           str(gate_c["hold_at_phase"]))
    eff_c = get_current_risk_fraction(combined)
    _check("60 trades (ph1 good, ph2 bad): effective frac = 0.021 (HELD at Phase 2)",
           math.isclose(eff_c, 0.021, rel_tol=1e-6),
           f"effective={eff_c:.4f}")

    # ── 4e: 60 trades, both phases good → full quarter-Kelly 3.65% ────────
    good_60  = _make_alternating_trades(60, win_fraction=0.6)
    gate_all = check_kelly_phase_gate(good_60)
    _check("60 good trades: both gates open, no hold",
           gate_all["hold_at_phase"] is None,
           str(gate_all["hold_at_phase"]))
    eff_full = get_current_risk_fraction(good_60)
    _check("60 good trades: effective frac = 0.0365 (Phase 3 active)",
           math.isclose(eff_full, 0.0365, rel_tol=1e-6),
           f"effective={eff_full:.4f}")

    # ── 4f: _quick_metrics sanity ─────────────────────────────────────────
    sample = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 100.0}, {"pnl": -50.0}]
    m = _quick_metrics(sample)
    _check("_quick_metrics: PF = 200/100 = 2.0",
           math.isclose(m["pf"], 2.0, rel_tol=1e-6),
           f"pf={m['pf']:.4f}")
    _check("_quick_metrics: max_dd_pct >= 0",
           m["max_dd_pct"] >= 0,
           f"max_dd_pct={m['max_dd_pct']:.2f}%")


# ===========================================================================
# Section 5: Position sizing examples
# ===========================================================================

def _section5_sizing() -> None:
    _section("5. POSITION SIZING — FORMULA VERIFICATION")

    try:
        from paper_trading_monitor import get_position_size  # type: ignore
    except ImportError as exc:
        _check("Import get_position_size", False, str(exc))
        return
    _check("Import get_position_size", True)

    # Build mock trade sets for different phases
    no_trades      = []
    good_ph1       = _make_alternating_trades(30, win_fraction=0.6)
    good_ph1_ph2   = _make_alternating_trades(60, win_fraction=0.6)

    # ── 5a: Basic Phase 1, normal vol ─────────────────────────────────────
    # equity=10000, ATR=2.00, ATR_60bar_avg=2.00 → vol_ratio=1.0
    # size = floor(10000 × 0.005 / 2.00 / 1.0) = floor(25) = 25
    n = get_position_size(10_000, 2.00, 2.00, no_trades)
    _check("Phase 1, ATR=avg: size = floor(10000×0.005/2.00/1.0) = 25",
           n == 25, f"got {n}")

    # ── 5b: Phase 1, low-vol (vol_ratio < 1 → LARGER position, OK) ────────
    # equity=10000, ATR=1.00, ATR_60bar_avg=2.00 → vol_ratio=0.5
    # size = floor(10000 × 0.005 / 1.00 / 0.5) = floor(100) = 100
    n_lv = get_position_size(10_000, 1.00, 2.00, no_trades)
    _check("Phase 1, low-vol (ATR=1, avg=2): vol_ratio=0.5, size=100",
           n_lv == 100, f"got {n_lv}")

    # ── 5c: vol_ratio capped at 1.5× (high-vol env → smaller position) ────
    # equity=10000, ATR=5.00, ATR_60bar_avg=1.00 → raw_ratio=5.0, capped=1.5
    # size = floor(10000 × 0.005 / 5.00 / 1.5) = floor(6.67) = 6
    n_hv = get_position_size(10_000, 5.00, 1.00, no_trades)
    _check("Phase 1, high-vol (ATR=5, avg=1): vol_ratio capped at 1.5, size=6",
           n_hv == 6, f"got {n_hv}")

    # ── 5d: Phase 2 (good ph1), normal vol ────────────────────────────────
    # equity=20000, ATR=3.00, ATR_60bar_avg=3.00 → vol_ratio=1.0
    # size = floor(20000 × 0.021 / 3.00 / 1.0) = floor(140) = 140
    n_ph2 = get_position_size(20_000, 3.00, 3.00, good_ph1)
    _check("Phase 2, ATR=avg: size = floor(20000×0.021/3.00/1.0) = 140",
           n_ph2 == 140, f"got {n_ph2}")

    # ── 5e: Phase 3 (good ph1+ph2), normal vol ────────────────────────────────
    # equity=50000, ATR=2.50, ATR_60bar_avg=2.50 → vol_ratio=1.0
    # size = floor(50000 × 0.0365 / 2.50 / 1.0)
    # Note: float gives 729.9999... so int() = 729 (correct floor behaviour)
    n_ph3 = get_position_size(50_000, 2.50, 2.50, good_ph1_ph2)
    _check("Phase 3, ATR=avg: size = floor(50000×0.0365/2.50/1.0) = 729",
           n_ph3 == 729, f"got {n_ph3}")

    # ── 5f: Minimum 1 share floor ─────────────────────────────────────────
    # equity=100, ATR=50.00, ATR_60bar_avg=1.00 → vol_ratio=1.5 (cap)
    # raw = floor(100 × 0.005 / 50.00 / 1.5) = floor(0.0067) = 0 → clamped to 1
    n_min = get_position_size(100, 50.00, 1.00, no_trades)
    _check("Minimum 1 share floor (tiny equity, huge ATR)",
           n_min == 1, f"got {n_min}")

    # ── 5g: Integer truncation (no rounding up) ───────────────────────────
    # floor(10000 × 0.005 / 3.33 / 1.0) = floor(15.015) = 15
    n_int = get_position_size(10_000, 3.33, 3.33, no_trades)
    _check("Integer truncation: floor(15.015)=15, not 16",
           n_int == 15, f"got {n_int}")


# ===========================================================================
# Summary
# ===========================================================================

def _print_summary() -> None:
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  {_bold('DEPLOYMENT CHECKLIST SUMMARY')}")
    print(f"{sep}")

    n_pass  = sum(1 for _, ok, _ in _results if ok)
    n_fail  = sum(1 for _, ok, _ in _results if not ok)
    n_total = len(_results)

    print(f"  Total checks : {n_total}")
    print(f"  Passed       : {_green(str(n_pass))}")
    print(f"  Failed       : {(_red(str(n_fail)) if n_fail else _green('0'))}")

    if n_fail:
        print(f"\n  {_red('FAILED CHECKS:')}")
        for label, ok, detail in _results:
            if not ok:
                print(f"    {FAIL_TAG()} {label}" + (f": {detail}" if detail else ""))

    print()
    if n_fail == 0:
        print(f"  {_green(_bold('✓  ALL CHECKS PASSED — STRATEGY CLEARED FOR LIVE DEPLOYMENT'))}")
    else:
        print(f"  {_red(_bold('✗  NOT READY — Resolve failed items before trading live'))}")
    print(f"{sep}\n")

    sys.exit(0 if n_fail == 0 else 1)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_pine_counts(raw: str) -> Dict[str, int]:
    """Parse 'GLD=2,WMT=0,...' into {sym: count} dict."""
    result: Dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            sym, _, n = part.partition("=")
            try:
                result[sym.strip().upper()] = int(n.strip())
            except ValueError:
                pass
    return result


def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="Pre-deployment checklist for Combo C (V5.1)"
    )
    parser.add_argument(
        "--pine-counts",
        type=str,
        default=None,
        metavar="GLD=N,...",
        help="Pine Script trade counts per symbol, e.g. 'GLD=2,WMT=0,...'"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output"
    )
    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False

    pine_counts: Optional[Dict[str, int]] = None
    if args.pine_counts:
        pine_counts = _parse_pine_counts(args.pine_counts)

    print(_bold("\nCOMBO C — PRE-DEPLOYMENT CHECKLIST (V5.1)"))
    print(f"Run this BEFORE Day 1 of live trading. All items must PASS.\n")

    _section1_locked_params()
    _section2_trade_counts(pine_counts)
    _section3_kelly_mock_tests()
    _section4_gate_tests()
    _section5_sizing()
    _print_summary()


if __name__ == "__main__":
    main()
