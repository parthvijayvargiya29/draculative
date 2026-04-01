#!/usr/bin/env python3
"""
paper_trading_monitor.py  --  V4.0 Live Deployment Trade Log & Monitor
=======================================================================
Governs the live deployment of Combo C (beta-filtered mean reversion).
Supports all three deployment periods: PAPER, LIVE_RAMP, LIVE_FULL.

DEPLOYMENT PATH DECISION: PATH C -- Direct live at 20% position size.
----------------------------------------------------------------------
The 131-trade backtest on 36 months of clean data (WFE 0.724, five versions,
confirmed parameter stability) constitutes sufficient prior evidence. Paper
trading for 60 days yields ~7 trades (3.6/month x 2 months) -- far below the
N=30 statistical minimum and unable to validate go/no-go criteria. Path C:
enter live at 20% of target position size immediately. First 30 live trades ARE
the formal validation period. At N=30: pass all criteria -> ramp per Section 5;
fail -> halt and diagnose.
PREREQUISITE: Pine Script per-instrument trade count within +-2 of backtest for
all 10 instruments verified before Day 1.

COMMANDS
--------
  log     -- Log a completed trade (paper or live)
  check   -- Daily go/no-go assessment + drift analysis
  show    -- Full trade table
  monthly -- Monthly summary report
  export  -- CSV export
  open    -- Manage open positions (add / list / close)
  decide  -- Log a decision (skip / override / size adjustment)

USAGE EXAMPLES
--------------
  # Log a live trade (full schema)
  python paper_trading_monitor.py log \\
      --symbol GLD --period LIVE_RAMP \\
      --signal-date 2026-03-20 --signal-close 215.10 \\
      --bb-lower-signal 214.20 --bb-mid-signal 219.50 \\
      --rsi2-signal 12.3 --atr10-signal 2.85 --accel-sl-signal 211.35 \\
      --entry-date 2026-03-21 --entry-price 215.40 --shares 12 \\
      --equity-at-entry 10000 \\
      --exit-date 2026-03-25 --exit-price 218.90 \\
      --exit-reason BB_MID --bars-held 4 \\
      --bb-mid-exit 219.20 \\
      --entry-slippage-pct 0.05 --exit-slippage-pct 0.03 \\
      --earnings-checked --dividend-checked --news-checked \\
      --concurrent 1

  # Daily check
  python paper_trading_monitor.py check

  # Monthly summary
  python paper_trading_monitor.py monthly --month 2026-03

  # Open position
  python paper_trading_monitor.py open add --symbol HD \\
      --entry-date 2026-03-21 --entry-price 387.50 --shares 7 \\
      --accel-sl 383.10 --bb-mid 391.20 --bar-count 1
  python paper_trading_monitor.py open list
  python paper_trading_monitor.py open close --symbol HD

FILES
-----
  paper_trades.json         -- Master trade log (append-only)
  paper_open_positions.json -- Open positions (live state)
  paper_monitor.log         -- Daily check archive
  paper_decisions.log       -- Decision log (skips, overrides, adjustments)
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any

HERE = Path(__file__).parent.resolve()
TRADE_LOG      = HERE / "paper_trades.json"
OPEN_POSITIONS = HERE / "paper_open_positions.json"
MONITOR_LOG    = HERE / "paper_monitor.log"
DECISIONS_LOG  = HERE / "paper_decisions.log"

# ---------------------------------------------------------------------------
# Validated benchmarks & locked parameters (Phase 1 decision, 19-Mar-2026)
# ---------------------------------------------------------------------------
VALIDATED = {
    # Backtest test-period stats (Aug 2025 - Mar 2026, N=34)
    "test_pf":              1.104,
    "test_wr":              52.94,
    "test_avg_win":         4.56,
    "test_avg_loss":        -2.63,
    "test_wl_ratio":        1.74,
    "test_n_trades":        34,
    "test_n_days":          146,
    "test_signal_rate":     3.6,   # trades/month across 10 instruments

    # Per-instrument full-period trade counts (total, train+val+test)
    # Source: bt34_comboC_trades_20260319_1806.csv (131 total)
    "backtest_counts_total": {
        "GLD": 12, "WMT": 10, "USMV": 18, "NVDA": 3,
        "AMZN": 12, "GOOGL": 12, "COST": 11, "XOM": 10,
        "HD": 21, "MA": 22,
    },
    # Per-instrument test-period counts (inferred from signal_bar dates >= 2025-08)
    "backtest_counts_test": {
        "GLD": 1, "WMT": 0, "USMV": 5, "NVDA": 3,
        "AMZN": 5, "GOOGL": 3, "COST": 3, "XOM": 0,
        "HD": 7, "MA": 8,
    },

    # Go/no-go thresholds (Section 5)
    "go_min_trades":        30,
    "go_pf_floor":          1.10,
    "go_wr_floor":          38.0,   # test_WR - 15pp
    "go_wr_ceiling":        68.0,   # test_WR + 15pp
    "go_max_drawdown_pct":  25.0,

    # Hard risk limits (Section 7)
    "risk_per_trade_pct":      0.5,   # target: 0.5% equity at risk per trade
    "risk_per_trade_max_pct":  0.8,   # flag if > 0.8% (sizing formula error)
    "max_exposure_pct":        10.0,  # max 10% equity per instrument
    "max_total_at_risk_pct":   5.0,   # sum(stop_dist x shares) / equity
    "max_concurrent":          4,     # max simultaneous open positions
    "monthly_loss_limit_pct":  8.0,   # halt new entries for rest of month
    "annual_loss_limit_pct":   15.0,  # full strategy pause

    # Signal decay watch thresholds
    "time_stop_pct_watch":     30.0,  # flag if TIME exits > 30%

    # Locked parameters
    "instruments":    ["GLD", "WMT", "USMV", "NVDA", "AMZN",
                       "GOOGL", "COST", "XOM", "HD", "MA"],
    "bb_period":      20,
    "bb_mult":        2.0,
    "rsi_period":     2,
    "rsi_threshold":  15.0,
    "atr_period":     10,
    "time_stop_bars": 10,
    "accel_sl":       "bb_lower - 1.0 * ATR(10) [static floor, non-ratcheting]",
    "volume_gate":    "NONE",
    "pos_size":       "floor(equity * 0.005 / ATR(10))",
    "pos_cap":        "10% of equity per instrument",
    "initial_capital_per_chart": 2500,
    "commission_per_order":      1.00,
}

VALID_PERIODS      = {"PAPER", "LIVE_RAMP", "LIVE_FULL"}
VALID_EXIT_REASONS = {"BB_MID", "SL", "ACCEL_SL", "TIME", "MANUAL_OVERRIDE", "EOB"}
VALID_SYMBOLS      = set(VALIDATED["instruments"])

# Slippage thresholds (Section 8)
SLIPPAGE_OK     = 0.15   # % per leg -- acceptable
SLIPPAGE_REVIEW = 0.30   # % per leg -- switch to limit orders

# ---------------------------------------------------------------------------
# Kelly ramp schedule (V5.1 — locked 19-Mar-2026)
# ---------------------------------------------------------------------------
# Based on 131 backtest trades: WR=51.1%, W/L=1.34, full-Kelly=14.6%
# Quarter-Kelly = 3.65%.  Ramp in 3 phases to manage model uncertainty.
KELLY_PHASES = [
    # (max_trade_count_exclusive, risk_fraction, label)
    (30,  0.005,  "Phase 1 (0-30 trades) — Validated baseline 0.5%"),
    (60,  0.021,  "Phase 2 (31-60 trades) — Midpoint ramp 2.1%"),
    (None, 0.0365, "Phase 3 (61+ trades) — Full quarter-Kelly 3.65%"),
]

# Phase transition gates (BOTH must pass to advance; check is per-phase boundary)
KELLY_PHASE1_PF_FLOOR       = 1.0    # Phase 1 PF must be >= 1.0 before advancing to Phase 2
KELLY_PHASE1_MAX_DD_PCT     = 15.0   # Phase 1 max drawdown must be < 15% before advancing
KELLY_PHASE2_PF_FLOOR       = 1.0    # Phase 2 PF must be >= 1.0 before advancing to Phase 3
KELLY_PHASE2_MAX_DD_PCT     = 15.0   # Phase 2 max drawdown must be < 15% before advancing

# Minimum trade counts before phase gate is evaluable
# Fewer trades → PF estimate too noisy to be reliable
KELLY_PHASE1_MIN_EVALUABLE  = 15     # need >= 15 Phase 1 trades before evaluating Phase 1 gate
KELLY_PHASE2_MIN_EVALUABLE  = 30     # need >= 30 Phase 2 trades before evaluating Phase 2 gate


def count_completed_trades(trades: Optional[List[dict]] = None) -> int:
    """Count completed (closed) live trades from the trade log."""
    if trades is None:
        trades = load_trades()
    return sum(1 for t in trades if t.get("period") in ("LIVE_RAMP", "LIVE_FULL"))


def get_kelly_phase(n_live: int) -> dict:
    """
    Return the current Kelly phase info dict given completed live trade count.
    Does NOT apply phase transition gates — call check_kelly_phase_gate()
    to determine whether gate allows advancement.
    """
    for max_n, frac, label in KELLY_PHASES:
        if max_n is None or n_live < max_n:
            next_threshold = max_n
            return {
                "n_live":          n_live,
                "risk_fraction":   frac,
                "label":           label,
                "next_threshold":  next_threshold,
                "trades_to_next":  (next_threshold - n_live) if next_threshold else None,
            }
    # Should never reach here — last phase has max_n=None
    _, frac, label = KELLY_PHASES[-1]
    return {"n_live": n_live, "risk_fraction": frac, "label": label,
            "next_threshold": None, "trades_to_next": None}


def check_kelly_phase_gate(trades: List[dict]) -> dict:
    """
    Verify phase transition gate conditions.
    Returns a dict describing whether advancement is blocked and why.

    Gate logic:
      Phase 1 → Phase 2 (at N=30):
        PF of Phase 1 trades >= 1.0  AND  max_DD of Phase 1 trades < 15%
      Phase 2 → Phase 3 (at N=60):
        PF of Phase 2 trades >= 1.0  AND  max_DD of Phase 2 trades < 15%

    If gate fails: stay at current phase risk fraction regardless of trade count.
    """
    live_trades  = [t for t in trades if t.get("period") in ("LIVE_RAMP", "LIVE_FULL")]
    n_live = len(live_trades)

    result = {
        "n_live":            n_live,
        "phase1_gate_open":  None,   # None = not yet evaluable (< 30 phase trades)
        "phase2_gate_open":  None,   # None = not yet evaluable (< 60 phase trades)
        "gate_messages":     [],
        "hold_at_phase":     None,   # None = no hold, else 1 or 2
    }

    # Phase 1 trades = first 30 live trades
    ph1_trades = live_trades[:30]
    if len(ph1_trades) >= KELLY_PHASE1_MIN_EVALUABLE:
        if len(ph1_trades) < 30:
            # Enough trades to evaluate but Phase 1 boundary not yet reached
            result["gate_messages"].append(
                f"Phase 1 gate: {len(ph1_trades)}/{30} trades "
                f"(evaluable at {KELLY_PHASE1_MIN_EVALUABLE}, boundary at 30)")
            # Evaluate early health signal but do NOT block advancement yet
            m1_early = _quick_metrics(ph1_trades)
            if m1_early["pf"] < KELLY_PHASE1_PF_FLOOR or m1_early["max_dd_pct"] >= KELLY_PHASE1_MAX_DD_PCT:
                result["gate_messages"].append(
                    f"  Early warning: PF={m1_early['pf']:.3f}, DD={m1_early['max_dd_pct']:.1f}% "
                    f"(monitor — gate evaluates at 30 trades)")
        else:
            # Full Phase 1 boundary reached — gate is binding
            m1 = _quick_metrics(ph1_trades)
            pf1 = m1["pf"]; dd1 = m1["max_dd_pct"]
            ph1_blocked = False
            if pf1 < KELLY_PHASE1_PF_FLOOR:
                ph1_blocked = True
                result["gate_messages"].append(
                    f"Phase 1 gate BLOCKED: PF={pf1:.3f} < {KELLY_PHASE1_PF_FLOOR} "
                    f"(Phase 1 losing — investigate before sizing up)")
            if dd1 >= KELLY_PHASE1_MAX_DD_PCT:
                ph1_blocked = True
                result["gate_messages"].append(
                    f"Phase 1 gate BLOCKED: max_DD={dd1:.1f}% >= {KELLY_PHASE1_MAX_DD_PCT}% "
                    f"(drawdown too high — investigate before sizing up)")
            if ph1_blocked:
                result["phase1_gate_open"] = False
                result["hold_at_phase"] = 1
            else:
                result["phase1_gate_open"] = True
                result["gate_messages"].append(
                    f"Phase 1 gate OPEN: PF={pf1:.3f} >= {KELLY_PHASE1_PF_FLOOR}, "
                    f"DD={dd1:.1f}% < {KELLY_PHASE1_MAX_DD_PCT}%")

    # Phase 2 trades = live trades 31–60
    ph2_trades = live_trades[30:60]
    if len(ph2_trades) >= KELLY_PHASE2_MIN_EVALUABLE and result["phase1_gate_open"] is True:
        if len(ph2_trades) < 30:
            # Early health signal for Phase 2 — not yet binding
            result["gate_messages"].append(
                f"Phase 2 gate: {len(ph2_trades)}/{30} trades "
                f"(evaluable at {KELLY_PHASE2_MIN_EVALUABLE}, boundary at 60 total trades)")
            m2_early = _quick_metrics(ph2_trades)
            if m2_early["pf"] < KELLY_PHASE2_PF_FLOOR or m2_early["max_dd_pct"] >= KELLY_PHASE2_MAX_DD_PCT:
                result["gate_messages"].append(
                    f"  Early warning: PF={m2_early['pf']:.3f}, DD={m2_early['max_dd_pct']:.1f}% "
                    f"(monitor — gate evaluates at 60 total trades)")
        else:
            m2 = _quick_metrics(ph2_trades)
            pf2 = m2["pf"]; dd2 = m2["max_dd_pct"]
            ph2_blocked = False
            if pf2 < KELLY_PHASE2_PF_FLOOR:
                ph2_blocked = True
                result["gate_messages"].append(
                    f"Phase 2 gate BLOCKED: PF={pf2:.3f} < {KELLY_PHASE2_PF_FLOOR} "
                    f"(Phase 2 losing — hold at 2.1% until PF recovers)")
            if dd2 >= KELLY_PHASE2_MAX_DD_PCT:
                ph2_blocked = True
                result["gate_messages"].append(
                    f"Phase 2 gate BLOCKED: max_DD={dd2:.1f}% >= {KELLY_PHASE2_MAX_DD_PCT}% "
                    f"(drawdown too high — hold at 2.1% until DD recovers)")
            if ph2_blocked:
                result["phase2_gate_open"] = False
                result["hold_at_phase"] = 2
            else:
                result["phase2_gate_open"] = True
                result["gate_messages"].append(
                    f"Phase 2 gate OPEN: PF={pf2:.3f} >= {KELLY_PHASE2_PF_FLOOR}, "
                    f"DD={dd2:.1f}% < {KELLY_PHASE2_MAX_DD_PCT}%")

    return result


def _quick_metrics(trades: List[dict]) -> dict:
    """Minimal PF + max-drawdown computation for gate checks."""
    # Support both 'net_pnl' (real trade log schema) and 'pnl' (mock / legacy)
    pnls = [float(t.get("net_pnl", t.get("pnl", 0))) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if sum(losses) < 0 else (
         float("inf") if wins else 0.0)
    equity = 0.0; peak = 0.0; max_dd_abs = 0.0
    for p in pnls:
        equity += p
        if equity > peak: peak = equity
        dd = peak - equity
        if dd > max_dd_abs: max_dd_abs = dd
    max_dd_pct = (max_dd_abs / peak * 100) if peak > 0 else 0.0
    return {"pf": round(pf, 3), "max_dd_pct": round(max_dd_pct, 1)}


def get_current_risk_fraction(trades: Optional[List[dict]] = None) -> float:
    """
    Returns the current risk fraction based on live trade count AND phase gate.

    Ramp schedule (requires gate to pass before advancing):
      Trades  0–30:  0.005  (0.5% — validated baseline)
      Trades 31–60:  0.021  (2.1% — midpoint to quarter-Kelly)
      Trades 61+:    0.0365 (3.65% — full quarter-Kelly)

    Phase transition gate: advancing to next phase requires:
      PF of completed phase >= 1.0  AND  max drawdown of completed phase < 15%
      If gate fails: hold at current phase fraction regardless of trade count.
    """
    if trades is None:
        trades = load_trades()
    n_live = count_completed_trades(trades)
    gate   = check_kelly_phase_gate(trades)

    # Nominal phase by trade count
    phase_info = get_kelly_phase(n_live)
    nominal_frac = phase_info["risk_fraction"]

    # Apply gate: if hold_at_phase is set, cap the fraction at that phase
    # hold_at_phase is only set when the gate is evaluable (>= 30 phase trades)
    # and has explicitly FAILED — None means either no hold or not yet evaluable.
    if gate["hold_at_phase"] is not None:
        held_phase_n = gate["hold_at_phase"]
        # Risk fraction for the held phase
        held_frac = KELLY_PHASES[held_phase_n - 1][1]
        return held_frac

    return nominal_frac


def get_position_size(equity: float, atr: float,
                      atr_60bar_avg: float,
                      trades: Optional[List[dict]] = None) -> int:
    """
    Volatility-scaled position size with Kelly ramp.

    Formula: floor(equity × risk_fraction / ATR / vol_ratio)
    vol_ratio = ATR(10)_current / ATR(10)_60bar_avg, capped at 1.5×
    (cap prevents outsized positions when vol is temporarily suppressed)

    Args:
        equity:       current account equity in $
        atr:          ATR(10) at signal bar (same bar as entry trigger)
        atr_60bar_avg: 60-bar average of ATR(10) (use snap.atr_pct_rank_60 proxy
                       or compute from recent bars)
        trades:       trade log list; if None, loads from disk

    Returns:
        Integer share count (floor division, minimum 1).
        Returns 0 if inputs are invalid (atr <= 0 or equity <= 0).
    """
    if equity <= 0 or atr <= 0 or atr_60bar_avg <= 0:
        return 0
    risk_fraction = get_current_risk_fraction(trades)
    vol_ratio     = min(atr / atr_60bar_avg, 1.5)   # cap at 1.5×
    raw_size      = (equity * risk_fraction) / (atr * vol_ratio)
    return max(int(raw_size), 1)   # floor via int(), minimum 1 share


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_trades() -> List[dict]:
    if not TRADE_LOG.exists():
        return []
    with open(TRADE_LOG, "r") as f:
        data = json.load(f)
    return data.get("trades", [])


def save_trades(trades: List[dict]):
    data = {
        "schema_version":     "2.0",
        "deployment_path":    "C",
        "validated_baseline": VALIDATED,
        "trades":             trades,
        "last_updated":       datetime.now().isoformat(),
    }
    with open(TRADE_LOG, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_open_positions() -> List[dict]:
    if not OPEN_POSITIONS.exists():
        return []
    with open(OPEN_POSITIONS, "r") as f:
        return json.load(f).get("positions", [])


def save_open_positions(positions: List[dict]):
    data = {"positions": positions, "updated": datetime.now().isoformat()}
    with open(OPEN_POSITIONS, "w") as f:
        json.dump(data, f, indent=2, default=str)


def log_decision(decision_type: str, detail: str):
    """Append to decisions log (Section 9 audit trail)."""
    entry = {
        "timestamp":     datetime.now().isoformat(),
        "decision_type": decision_type,
        "detail":        detail,
    }
    with open(DECISIONS_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: List[dict],
                    period: Optional[str] = None,
                    last_n: Optional[int] = None) -> dict:
    if period:
        trades = [t for t in trades if t.get("period") == period]
    if last_n is not None:
        trades = trades[-last_n:]
    if not trades:
        return {"n": 0}

    n      = len(trades)
    pnls   = [float(t.get("net_pnl", 0)) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    wr       = len(wins) / n * 100
    avg_win  = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    pf       = sum(wins) / abs(sum(losses)) if sum(losses) < 0 else (
               float("inf") if wins else 0.0)

    # Drawdown on cumulative P&L
    equity = 0.0; peak = 0.0; max_dd_abs = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd_abs:
            max_dd_abs = dd
    max_dd_pct = (max_dd_abs / peak * 100) if peak > 0 else 0.0

    # Exit type breakdown
    exit_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        exit_counts[t.get("exit_reason", "UNKNOWN")] += 1

    # Slippage
    es_vals = [float(t["entry_slippage_pct"]) for t in trades
               if t.get("entry_slippage_pct") is not None]
    xs_vals = [float(t["exit_slippage_pct"]) for t in trades
               if t.get("exit_slippage_pct") is not None]
    avg_es  = sum(es_vals) / len(es_vals) if es_vals else None
    avg_xs  = sum(xs_vals) / len(xs_vals) if xs_vals else None

    # Avg bars held
    bars    = [int(t.get("bar_count_at_exit", 0)) for t in trades
               if t.get("bar_count_at_exit")]
    avg_bars = sum(bars) / len(bars) if bars else None

    # Distance to target at TIME exits
    t_trades = [t for t in trades if t.get("exit_reason") == "TIME"]
    d_vals   = [float(t["distance_to_target_at_exit"]) for t in t_trades
                if t.get("distance_to_target_at_exit") is not None]
    avg_dtm  = sum(d_vals) / len(d_vals) if d_vals else None

    dates = sorted(t.get("entry_date", "") for t in trades)
    return {
        "n":                     n,
        "n_wins":                len(wins),
        "n_losses":              len(losses),
        "win_rate_%":            round(wr, 2),
        "avg_win_$":             round(avg_win, 2),
        "avg_loss_$":            round(avg_loss, 2),
        "wl_ratio":              round(wl_ratio, 3),
        "profit_factor":         round(pf, 3),
        "total_pnl_$":           round(sum(pnls), 2),
        "max_drawdown_%":        round(max_dd_pct, 2),
        "exit_breakdown":        dict(exit_counts),
        "avg_entry_slippage_%":  round(avg_es, 4) if avg_es is not None else None,
        "avg_exit_slippage_%":   round(avg_xs, 4) if avg_xs is not None else None,
        "avg_holding_bars":      round(avg_bars, 1) if avg_bars is not None else None,
        "avg_dist_to_mid_time":  round(avg_dtm, 4) if avg_dtm is not None else None,
        "first_trade":           dates[0] if dates else None,
        "last_trade":            dates[-1] if dates else None,
    }


# ---------------------------------------------------------------------------
# Go/no-go decision matrix (Section 5)
# ---------------------------------------------------------------------------

def assess_go_nogo(metrics: dict) -> dict:
    v  = VALIDATED
    n  = metrics.get("n", 0)
    pf = metrics.get("profit_factor", 0.0)
    wr = metrics.get("win_rate_%", 0.0)
    dd = metrics.get("max_drawdown_%", 0.0)

    checks = {
        "sample_size": {
            "label":     f"N trades >= {v['go_min_trades']} (statistical minimum)",
            "value":     n,
            "threshold": v["go_min_trades"],
            "pass":      n >= v["go_min_trades"],
            "severity":  "REQUIRED",
        },
        "profit_factor": {
            "label":     f"Live PF >= {v['go_pf_floor']} (test-period floor)",
            "value":     round(pf, 3),
            "threshold": v["go_pf_floor"],
            "pass":      pf >= v["go_pf_floor"],
            "severity":  "REQUIRED",
        },
        "wr_floor": {
            "label":     f"Live WR >= {v['go_wr_floor']}% (backtest - 15pp)",
            "value":     round(wr, 1),
            "threshold": v["go_wr_floor"],
            "pass":      wr >= v["go_wr_floor"],
            "severity":  "REQUIRED",
        },
        "wr_ceiling": {
            "label":     f"Live WR <= {v['go_wr_ceiling']}% (lookahead guard)",
            "value":     round(wr, 1),
            "threshold": v["go_wr_ceiling"],
            "pass":      wr <= v["go_wr_ceiling"],
            "severity":  "REQUIRED",
        },
        "drawdown": {
            "label":     f"Max drawdown < {v['go_max_drawdown_pct']}% (~2x backtest)",
            "value":     round(dd, 1),
            "threshold": v["go_max_drawdown_pct"],
            "pass":      dd < v["go_max_drawdown_pct"],
            "severity":  "REQUIRED",
        },
        "pf_comfortable": {
            "label":     "Live PF >= 1.20 (comfortable margin, preferred)",
            "value":     round(pf, 3),
            "threshold": 1.20,
            "pass":      pf >= 1.20,
            "severity":  "RECOMMENDED",
        },
        "sample_preferred": {
            "label":     "N trades >= 50 (preferred robustness)",
            "value":     n,
            "threshold": 50,
            "pass":      n >= 50,
            "severity":  "RECOMMENDED",
        },
    }

    req = {k: c for k, c in checks.items() if c["severity"] == "REQUIRED"}

    # Section 5 decision matrix
    if n < v["go_min_trades"]:
        months_left = max(0, (v["go_min_trades"] - n) / v["test_signal_rate"])
        verdict = "ACCUMULATING"
        reason  = (f"N={n}/{v['go_min_trades']} trades. "
                   f"~{months_left:.1f} months to go at {v['test_signal_rate']:.1f} trades/month.")
        action  = "Trade at 20% position size. Log every trade before next trading day."
    elif all(c["pass"] for c in req.values()):
        verdict = "FULL GO"
        reason  = "All 5 required criteria pass."
        action  = ("Ramp to 50% position size (Month 1 post-go). "
                   "If M1 PF>=1.0 and DD<15%: 75%. "
                   "If M2 PF>=1.0 and DD<15%: 100%.")
    elif not checks["drawdown"]["pass"]:
        verdict = "PAUSE"
        reason  = f"Drawdown {dd:.1f}% exceeds 25% hard limit."
        action  = "Halt new entries. Let open positions close. Run 18-month fresh backtest."
    elif (not checks["profit_factor"]["pass"] and
          checks["wr_floor"]["pass"] and checks["wr_ceiling"]["pass"]):
        verdict = "CONDITIONAL"
        reason  = f"PF={pf:.3f} below 1.10 but WR in-bounds."
        action  = "Continue at current size for 15 more trades, then reassess."
    elif not checks["wr_floor"]["pass"]:
        verdict = "INVESTIGATE_WR_LOW"
        reason  = f"Live WR {wr:.1f}% below 38% floor."
        action  = ("STOP new entries. Check Pine Script: RSI condition (< 15 not <= 15), "
                   "BB period (20), adjusted data enabled.")
    elif not checks["wr_ceiling"]["pass"]:
        verdict = "INVESTIGATE_WR_HIGH"
        reason  = f"Live WR {wr:.1f}% above 68% ceiling -- likely lookahead or execution mismatch."
        action  = "STOP new entries. Audit Pine Script for forward references."
    else:
        failed  = [k for k, c in req.items() if not c["pass"]]
        verdict = "NO-GO"
        reason  = f"Failed: {', '.join(failed)}."
        action  = "Do NOT deploy additional capital."

    return {"verdict": verdict, "reason": reason, "action": action,
            "checks": checks, "n_trades": n}


# ---------------------------------------------------------------------------
# Drawdown state (Section 7)
# ---------------------------------------------------------------------------

def assess_drawdown_state(dd_pct: float) -> dict:
    if dd_pct < 8:
        return {"level": "NORMAL", "msg": "Normal trading. No size adjustments."}
    elif dd_pct < 15:
        return {"level": "WATCH",
                "msg": (f"DD {dd_pct:.1f}%: Check regime. If trending market, "
                        f"reduce to 50% size until DD < 8%.")}
    elif dd_pct < 25:
        return {"level": "HALT",
                "msg": (f"DD {dd_pct:.1f}%: HALT new entries. "
                        f"Let open positions close. Run 18-month fresh backtest.")}
    else:
        return {"level": "FULL_PAUSE",
                "msg": (f"DD {dd_pct:.1f}%: FULL STRATEGY PAUSE. "
                        f"Triggers go/no-go failure regardless of prior validation.")}


# ---------------------------------------------------------------------------
# Signal decay indicators (Section 6)
# ---------------------------------------------------------------------------

def check_signal_decay(trades: List[dict]) -> dict:
    if len(trades) < 60:
        return {"indicators_present": 0,
                "details": ["Fewer than 60 trades -- decay analysis premature."],
                "recommendation": "HEALTHY"}

    indicators = []
    details    = []

    # 1. Rolling 30-trade WR declining for 3 consecutive quarters
    q_sz = max(len(trades) // 4, 15)
    qs   = [trades[i:i+q_sz] for i in range(0, len(trades), q_sz) if trades[i:i+q_sz]]
    if len(qs) >= 4:
        q_wrs = [sum(1 for t in q if float(t.get("net_pnl", 0)) > 0) / len(q) * 100
                 for q in qs[-4:]]
        if all(q_wrs[i] > q_wrs[i+1] for i in range(len(q_wrs)-1)):
            indicators.append("WR_DECLINING_QUARTERLY")
            details.append(f"WR declining 3 quarters: {[f'{w:.0f}%' for w in q_wrs]}")

    # 2. PF below 1.0 for 2 consecutive recent quarters
    if len(qs) >= 3:
        def pf_q(q_trades):
            w = sum(float(t.get("net_pnl", 0)) for t in q_trades
                    if float(t.get("net_pnl", 0)) > 0)
            l = abs(sum(float(t.get("net_pnl", 0)) for t in q_trades
                        if float(t.get("net_pnl", 0)) <= 0))
            return w / l if l > 0 else float("inf")
        pfs = [pf_q(q) for q in qs[-3:]]
        if sum(1 for p in pfs[-2:] if p < 1.0) >= 2:
            indicators.append("PF_BELOW_1_TWO_QUARTERS")
            details.append(f"PF < 1.0 last 2 quarters: {[f'{p:.3f}' for p in pfs[-2:]]}")

    # 3. Avg holding period increasing >30%
    if len(trades) >= 60:
        early = [int(t.get("bar_count_at_exit", 0)) for t in trades[:30]
                 if t.get("bar_count_at_exit")]
        late  = [int(t.get("bar_count_at_exit", 0)) for t in trades[-30:]
                 if t.get("bar_count_at_exit")]
        if early and late:
            avg_e = sum(early) / len(early)
            avg_l = sum(late) / len(late)
            if avg_l > avg_e * 1.30:
                indicators.append("HOLDING_PERIOD_INCREASING")
                details.append(f"Avg bars held: early={avg_e:.1f}, recent={avg_l:.1f} "
                                f"(+{(avg_l/avg_e-1)*100:.0f}%)")

    # 4. TIME exit % increasing in recent trades
    all_exits    = [t.get("exit_reason", "") for t in trades]
    time_overall = sum(1 for e in all_exits if e == "TIME") / len(all_exits) * 100
    recent30_exits = [t.get("exit_reason", "") for t in trades[-30:]]
    time_recent  = sum(1 for e in recent30_exits if e == "TIME") / len(recent30_exits) * 100
    if time_recent > VALIDATED["time_stop_pct_watch"] and time_recent > time_overall * 1.5:
        indicators.append("TIME_STOP_INCREASING")
        details.append(f"TIME exits: overall={time_overall:.0f}%, recent30={time_recent:.0f}%")

    n = len(indicators)
    rec = "PAUSE_AND_RETEST" if n >= 2 else "HEALTHY"
    if n >= 2:
        details.append("2+ decay indicators. Pause new entries and run 18-month fresh backtest.")

    return {"indicators_present": n, "indicators": indicators,
            "details": details, "recommendation": rec}


# ---------------------------------------------------------------------------
# Drift analysis
# ---------------------------------------------------------------------------

def drift_analysis(metrics: dict) -> List[dict]:
    v     = VALIDATED
    flags = []

    if metrics.get("n", 0) < 10:
        return [{"level": "INFO", "msg": "Fewer than 10 trades -- deviation analysis premature."}]

    pf = metrics.get("profit_factor", 0.0)
    if pf < v["test_pf"] * 0.70:
        flags.append({"level": "ALERT",
                      "msg": f"Live PF ({pf:.3f}) >30% below validated ({v['test_pf']:.3f}). Regime shift."})
    elif pf < v["test_pf"]:
        flags.append({"level": "WARN",
                      "msg": f"Live PF ({pf:.3f}) below validated ({v['test_pf']:.3f}). Monitor."})
    else:
        flags.append({"level": "OK",
                      "msg": f"Live PF ({pf:.3f}) >= validated ({v['test_pf']:.3f})."})

    wr      = metrics.get("win_rate_%", 0.0)
    wr_diff = wr - v["test_wr"]
    if abs(wr_diff) > 15:
        flags.append({"level": "ALERT",
                      "msg": f"Live WR ({wr:.1f}%) deviates {wr_diff:+.1f}pp from backtest. Investigate."})
    else:
        flags.append({"level": "OK",
                      "msg": f"Live WR ({wr:.1f}%) within +-15pp of backtest ({v['test_wr']:.1f}%)."})

    wl = metrics.get("wl_ratio", 0.0)
    if wl < v["test_wl_ratio"] * 0.60:
        flags.append({"level": "ALERT",
                      "msg": f"W/L ratio ({wl:.2f}x) >40% below validated. Wins cut short?"})
    elif wl < v["test_wl_ratio"]:
        flags.append({"level": "WARN",
                      "msg": f"W/L ratio ({wl:.2f}x) below validated ({v['test_wl_ratio']:.2f}x)."})
    else:
        flags.append({"level": "OK",
                      "msg": f"W/L ratio ({wl:.2f}x) >= validated ({v['test_wl_ratio']:.2f}x)."})

    # Slippage check
    es = metrics.get("avg_entry_slippage_%")
    xs = metrics.get("avg_exit_slippage_%")
    if es is not None or xs is not None:
        avg_slip = ((es or 0) + (xs or 0)) / max(1, (1 if es else 0) + (1 if xs else 0))
        if avg_slip > SLIPPAGE_REVIEW:
            flags.append({"level": "ALERT",
                          "msg": f"Avg slippage {avg_slip:.3f}% > {SLIPPAGE_REVIEW}%. Switch to limit orders."})
        elif avg_slip > SLIPPAGE_OK:
            flags.append({"level": "WARN",
                          "msg": f"Avg slippage {avg_slip:.3f}% elevated (>{SLIPPAGE_OK}%). Review fills."})

    return flags


# ---------------------------------------------------------------------------
# Monthly summary
# ---------------------------------------------------------------------------

def compute_monthly_summary(trades: List[dict], month_str: str) -> dict:
    m_trades  = [t for t in trades if t.get("entry_date", "").startswith(month_str)]
    cum_trades = [t for t in trades if t.get("entry_date", "") <= month_str + "-99"]
    m  = compute_metrics(m_trades)
    c  = compute_metrics(cum_trades)
    eb = m.get("exit_breakdown", {})
    tot = m.get("n", 1) or 1
    eb_pct = {k: round(v / tot * 100, 1) for k, v in eb.items()}

    return {
        "month":            month_str,
        "n_this_month":     m.get("n", 0),
        "n_cumulative":     c.get("n", 0),
        "wr_month_%":       m.get("win_rate_%"),
        "wr_cumul_%":       c.get("win_rate_%"),
        "pf_month":         m.get("profit_factor"),
        "pf_cumul":         c.get("profit_factor"),
        "avg_net_pnl_$":    round(m.get("total_pnl_$", 0) / m.get("n", 1), 2)
                             if m.get("n") else None,
        "max_dd_month_%":   m.get("max_drawdown_%"),
        "exit_type_pct":    eb_pct,
        "avg_holding_bars": m.get("avg_holding_bars"),
        "avg_dtm_time":     m.get("avg_dist_to_mid_time"),
        "avg_entry_slip_%": m.get("avg_entry_slippage_%"),
        "avg_exit_slip_%":  m.get("avg_exit_slippage_%"),
        "vs_backtest_pf":   round(m.get("profit_factor", 0) - VALIDATED["test_pf"], 3)
                             if m.get("n") else None,
        "vs_backtest_wr":   round(m.get("win_rate_%", 0) - VALIDATED["test_wr"], 1)
                             if m.get("n") else None,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_log(args):
    trades = load_trades()

    symbol = args.symbol.upper()
    if symbol not in VALID_SYMBOLS:
        print(f"  WARNING: {symbol} not in validated instruments.")
        if not getattr(args, "force", False):
            print("  Use --force to log non-standard instruments.")
            return

    period = getattr(args, "period", "PAPER").upper()
    if period not in VALID_PERIODS:
        period = "PAPER"

    exit_reason = getattr(args, "exit_reason", "").upper()
    if exit_reason not in VALID_EXIT_REASONS:
        print(f"  WARNING: exit_reason '{exit_reason}' not recognized.")

    entry_price  = float(args.entry_price)
    exit_price   = float(args.exit_price)
    shares       = float(args.shares)
    commission   = 2.00   # $1/order x 2 orders round trip
    gross_pnl    = (exit_price - entry_price) * shares
    net_pnl      = gross_pnl - commission
    won          = net_pnl > 0

    eq           = float(getattr(args, "equity_at_entry", 0) or 0)
    accel_sig    = float(getattr(args, "accel_sl_signal", 0) or 0)
    pct_at_risk  = None
    if eq > 0 and accel_sig > 0:
        pct_at_risk = round((entry_price - accel_sig) * shares / eq * 100, 4)

    es_pct = getattr(args, "entry_slippage_pct", None)
    xs_pct = getattr(args, "exit_slippage_pct", None)

    bm_exit = float(getattr(args, "bb_mid_exit", 0) or 0) or None
    dtm     = round(bm_exit - exit_price, 4) if bm_exit else None

    trade: Dict[str, Any] = {
        "trade_id":                     len(trades) + 1,
        "period":                       period,
        "logged_at":                    datetime.now().isoformat(),
        # Signal bar
        "instrument":                   symbol,
        "signal_date":                  getattr(args, "signal_date", None),
        "signal_close":                 float(getattr(args, "signal_close", 0) or 0) or None,
        "bb_lower_at_signal":           float(getattr(args, "bb_lower_signal", 0) or 0) or None,
        "bb_midline_at_signal":         float(getattr(args, "bb_mid_signal", 0) or 0) or None,
        "rsi2_at_signal":               float(getattr(args, "rsi2_signal", 0) or 0) or None,
        "atr10_at_signal":              float(getattr(args, "atr10_signal", 0) or 0) or None,
        "accel_sl_at_signal":           accel_sig or None,
        # Entry
        "entry_date":                   args.entry_date,
        "entry_price":                  entry_price,
        "entry_slippage_pct":           float(es_pct) if es_pct is not None else None,
        "shares":                       shares,
        "equity_at_entry":              eq or None,
        "pct_equity_at_risk":           pct_at_risk,
        # Exit
        "bar_count_at_exit":            int(args.bars_held),
        "exit_date":                    args.exit_date,
        "exit_price":                   exit_price,
        "exit_reason":                  exit_reason,
        "exit_slippage_pct":            float(xs_pct) if xs_pct is not None else None,
        # P&L
        "gross_pnl":                    round(gross_pnl, 4),
        "commission":                   commission,
        "net_pnl":                      round(net_pnl, 4),
        "won":                          won,
        # Diagnostics
        "bb_midline_at_exit":           bm_exit,
        "distance_to_target_at_exit":   dtm,
        # Checklist
        "earnings_checked":             bool(getattr(args, "earnings_checked", False)),
        "dividend_checked":             bool(getattr(args, "dividend_checked", False)),
        "news_checked":                 bool(getattr(args, "news_checked", False)),
        # Portfolio
        "concurrent_positions_at_entry": int(getattr(args, "concurrent", 0) or 0),
        "notes":                        getattr(args, "notes", "") or "",
    }

    # Checklist warning
    missing_checks = [k for k in ("earnings_checked", "dividend_checked", "news_checked")
                      if not trade[k]]
    if missing_checks:
        print(f"  WARNING: Pre-entry checklist incomplete: {missing_checks}")
        log_decision("CHECKLIST_INCOMPLETE",
                     f"Trade #{trade['trade_id']} {symbol} {args.entry_date}: {missing_checks}")

    # Risk size warning
    if pct_at_risk is not None and pct_at_risk > VALIDATED["risk_per_trade_max_pct"]:
        print(f"  WARNING: Risk {pct_at_risk:.3f}% > {VALIDATED['risk_per_trade_max_pct']}% -- check position sizing.")

    trades.append(trade)
    save_trades(trades)

    sign = "+" if net_pnl >= 0 else ""
    print(f"\n  Trade #{trade['trade_id']} [{period}] {symbol} "
          f"{args.entry_date}->{args.exit_date} | {exit_reason} | "
          f"net={sign}{net_pnl:.2f} | {'WIN' if won else 'LOSS'}")
    print(f"  Total trades: {len(trades)}")

    # Remove from open positions
    open_pos = load_open_positions()
    open_pos = [p for p in open_pos if p.get("symbol") != symbol]
    save_open_positions(open_pos)


def cmd_check(args):
    trades     = load_trades()
    all_m      = compute_metrics(trades)
    roll30_m   = compute_metrics(trades, last_n=30)
    assessment = assess_go_nogo(all_m)
    dd_state   = assess_drawdown_state(all_m.get("max_drawdown_%", 0))
    flags      = drift_analysis(all_m) if trades else []
    open_pos   = load_open_positions()

    now  = datetime.now()
    sep  = "=" * 80
    sep2 = "-" * 78
    out  = []

    # Period counts
    n_paper     = sum(1 for t in trades if t.get("period") == "PAPER")
    n_live_ramp = sum(1 for t in trades if t.get("period") == "LIVE_RAMP")
    n_live_full = sum(1 for t in trades if t.get("period") == "LIVE_FULL")

    out.append(f"\n{sep}")
    out.append(f"  COMBO C v4.0 -- LIVE DEPLOYMENT MONITOR  [PATH C]")
    out.append(f"  {now.strftime('%Y-%m-%d %H:%M')}  |  "
               f"Total: {len(trades)}  (Paper:{n_paper} Ramp:{n_live_ramp} Full:{n_live_full})")
    out.append(sep)

    if trades:
        out.append(f"\n  ALL-PERIOD METRICS  (N={all_m['n']})")
        out.append(f"  {sep2}")
        out.append(f"  {'Metric':<28} {'Live':<14} {'Validated':<14} {'Delta'}")
        out.append(f"  {sep2}")

        def row(lbl, val, bk, fmt=".3f"):
            base = VALIDATED.get(bk)
            if val is not None and base is not None:
                d = val - base; s = "+" if d >= 0 else ""
                return f"  {lbl:<28} {val:<14{fmt}} {base:<14{fmt}} {s}{d:{fmt}}"
            vs = f"{val:{fmt}}" if val is not None else "---"
            return f"  {lbl:<28} {vs:<14} ---"

        out.append(row("Profit Factor",  all_m.get("profit_factor"), "test_pf"))
        out.append(row("Win Rate (%)",   all_m.get("win_rate_%"),    "test_wr",  ".1f"))
        out.append(row("Avg Win ($)",    all_m.get("avg_win_$"),     "test_avg_win",  ".2f"))
        out.append(row("Avg Loss ($)",   all_m.get("avg_loss_$"),    "test_avg_loss", ".2f"))
        out.append(row("W/L Ratio",      all_m.get("wl_ratio"),      "test_wl_ratio"))
        out.append(f"  {'Total P&L ($)':<28} {all_m.get('total_pnl_$', 0):<14.2f}")
        out.append(f"  {'Max Drawdown (%)':<28} {all_m.get('max_drawdown_%', 0):<14.1f}")
        if all_m.get("avg_holding_bars"):
            out.append(f"  {'Avg Bars Held':<28} {all_m['avg_holding_bars']:<14.1f}")
        if all_m.get("exit_breakdown"):
            eb = all_m["exit_breakdown"]
            n  = all_m["n"]
            parts = "  ".join(f"{k}:{v}({v/n*100:.0f}%)" for k, v in sorted(eb.items()))
            out.append(f"  Exit breakdown: {parts}")

        if roll30_m.get("n", 0) >= 15:
            out.append(f"\n  ROLLING LAST-30: PF={roll30_m.get('profit_factor','---')} | "
                       f"WR={roll30_m.get('win_rate_%','---')}%")
    else:
        out.append(f"\n  No trades logged. Expected first signal soon at ~3.6 trades/month.")
        out.append(f"  N=30 expected in ~8.3 months from first trade.")

    # Open positions
    if open_pos:
        out.append(f"\n  OPEN POSITIONS ({len(open_pos)} / {VALIDATED['max_concurrent']} max)")
        out.append(f"  {sep2}")
        for p in open_pos:
            time_warn = " <-- TIME STOP NEAR" if int(p.get("bar_count", 0)) >= 8 else ""
            out.append(f"  {p.get('symbol','?'):<6} entry={p.get('entry_date','?')} @ "
                       f"{p.get('entry_price', 0):.2f}  bars={p.get('bar_count','?')}  "
                       f"accel_sl={p.get('accel_sl','---')}  bb_mid={p.get('bb_mid','---')}"
                       f"{time_warn}")
    else:
        out.append(f"\n  Open positions: NONE")

    # Drawdown state
    out.append(f"\n  DRAWDOWN STATE [{dd_state['level']}]: {dd_state['msg']}")

    # Verdict
    out.append(f"\n  {sep2}")
    out.append(f"  VERDICT : {assessment['verdict']}")
    out.append(f"  REASON  : {assessment['reason']}")
    out.append(f"  ACTION  : {assessment['action']}")
    out.append(f"  {sep2}")
    for k, c in assessment["checks"].items():
        icon = "OK" if c["pass"] else "XX"
        sev  = f"[{c['severity']}]"
        out.append(f"  {icon} {sev:<14} {c['label']:<45} = {c['value']}")

    # Drift flags
    if flags:
        out.append(f"\n  DRIFT ANALYSIS:")
        lvl_icons = {"OK": "OK", "WARN": "WW", "ALERT": "!!", "INFO": "--"}
        for fl in flags:
            ico = lvl_icons.get(fl["level"], "??")
            out.append(f"  {ico} [{fl['level']:<5}] {fl['msg']}")

    # Decay check (>= 60 trades)
    if len(trades) >= 60:
        decay = check_signal_decay(trades)
        out.append(f"\n  SIGNAL DECAY: {decay['recommendation']} "
                   f"({decay['indicators_present']}/4 indicators present)")
        for d in decay["details"]:
            out.append(f"    - {d}")

    # Kelly ramp status
    out.append(f"\n  {sep2}")
    out.append(f"  KELLY RAMP STATUS")
    out.append(f"  {sep2}")
    n_live      = count_completed_trades(trades)
    phase_info  = get_kelly_phase(n_live)
    gate        = check_kelly_phase_gate(trades)
    eff_frac    = get_current_risk_fraction(trades)
    gate_held   = gate["hold_at_phase"] is not None
    out.append(f"  Completed live trades : {n_live}")
    out.append(f"  Current phase         : {phase_info['label']}")
    out.append(f"  Nominal risk fraction : {phase_info['risk_fraction']*100:.2f}%")
    if gate_held:
        out.append(f"  EFFECTIVE risk frac   : {eff_frac*100:.2f}%  [GATE HELD at Phase {gate['hold_at_phase']}]")
    else:
        out.append(f"  Effective risk frac   : {eff_frac*100:.2f}%")
    if phase_info["next_threshold"]:
        out.append(f"  Next threshold        : {phase_info['next_threshold']} trades  "
                   f"({phase_info['trades_to_next']} to go)")
    else:
        out.append(f"  Next threshold        : N/A (final phase)")
    out.append(f"  Gate messages:")
    if gate["gate_messages"]:
        for msg in gate["gate_messages"]:
            icon = "WW" if "BLOCKED" in msg else ("EW" if "warning" in msg.lower() else "OK")
            out.append(f"    {icon} {msg}")
    else:
        ph1_evaluable_at = KELLY_PHASE1_MIN_EVALUABLE
        ph1_boundary     = 30
        out.append(f"    -- Phase gate not yet evaluable (need {ph1_evaluable_at} live trades; "
                   f"binding at {ph1_boundary})")
        out.append(f"    -- Currently in Phase 1 baseline period. Risk fraction fixed at 0.5%.")

    # Size example at current equity (use most recent trade's equity if available)
    recent_equity = None
    recent_atr    = None
    if trades:
        for t in reversed(trades):
            if t.get("equity_at_entry") and t.get("atr10_signal"):
                recent_equity = float(t["equity_at_entry"])
                recent_atr    = float(t["atr10_signal"])
                break
    if recent_equity and recent_atr:
        ex_size = get_position_size(recent_equity, recent_atr, recent_atr, trades)
        out.append(f"  Example size (last equity=${recent_equity:.0f}, "
                   f"ATR={recent_atr:.2f}, vol_ratio=1.0): {ex_size} shares")

    # Daily checklist
    out.append(f"\n  {sep2}")
    out.append(f"  DAILY CHECKLIST")
    out.append(f"  {sep2}")
    items = [
        "Open position audit: bar_count, current ACCEL_SL, distance to BB_MID",
        "Signal scan: close < BB_lower(20,2) AND RSI(2) < 15 on any flat instrument",
        "Pre-entry: earnings within 5 days? ex-div within 3 days? company news?",
        "Concurrent cap: max 4 open; if >4 signals, pick lowest RSI(2) first",
        "At-risk check: sum(stop_dist x shares) / equity < 5%",
        "Log exits:  python paper_trading_monitor.py log ...",
        "Update open: python paper_trading_monitor.py open list",
    ]
    for i, item in enumerate(items, 1):
        out.append(f"  {i}. {item}")

    out.append(f"\n{sep}\n")
    full = "\n".join(out)
    print(full)
    with open(MONITOR_LOG, "a") as f:
        f.write(full + "\n")


def cmd_show(args):
    trades = load_trades()
    period_filter = getattr(args, "period", None)
    if period_filter:
        trades = [t for t in trades if t.get("period") == period_filter.upper()]
    if not trades:
        print("No trades logged yet.")
        return

    print(f"\n  COMBO C TRADE LOG  ({len(trades)} trades)")
    print(f"  {'--' * 60}")
    hdr = (f"  {'#':>4} {'Per':>8} {'Sym':<6} {'Entry':>12} {'Px':>8} "
           f"{'Exit':>12} {'ExPx':>8} {'Reason':<14} {'Bars':>5} "
           f"{'NetPnL':>9} {'WL':<4} {'Chk'}")
    print(hdr)
    print(f"  {'--' * 60}")

    for t in trades:
        sign = "+" if float(t.get("net_pnl", 0)) >= 0 else ""
        wl   = "WIN" if t.get("won") else "LOS"
        chk  = "Y" if (t.get("earnings_checked") and t.get("dividend_checked")
                       and t.get("news_checked")) else "N"
        sym  = t.get("instrument", t.get("symbol", "?"))
        print(f"  {t['trade_id']:>4} {t.get('period','?'):>8} {sym:<6} "
              f"{t.get('entry_date','?'):>12} {t.get('entry_price',0):>8.2f} "
              f"{t.get('exit_date','?'):>12} {t.get('exit_price',0):>8.2f} "
              f"{t.get('exit_reason','?'):<14} {t.get('bar_count_at_exit',0):>5} "
              f"{sign}{float(t.get('net_pnl',0)):>9.2f} {wl:<4} {chk}")

    print(f"  {'--' * 60}")
    m = compute_metrics(trades)
    print(f"\n  N={m['n']} | PF={m.get('profit_factor','---')} | "
          f"WR={m.get('win_rate_%','---')}% | Total=${m.get('total_pnl_$','---')} | "
          f"MaxDD={m.get('max_drawdown_%','---')}%\n")


def cmd_monthly(args):
    trades    = load_trades()
    month_str = getattr(args, "month", None)
    months    = ([month_str] if month_str else
                 sorted(set(t.get("entry_date", "")[:7]
                            for t in trades if t.get("entry_date"))))
    if not months:
        print("No trades logged yet.")
        return

    sep = "-" * 70
    for mth in months:
        ms = compute_monthly_summary(trades, mth)
        print(f"\n  MONTHLY SUMMARY: {mth}")
        print(f"  {sep}")
        print(f"  N this month     : {ms['n_this_month']}   (cumulative: {ms['n_cumulative']})")
        if ms.get("wr_month_%") is not None:
            print(f"  Win rate         : {ms['wr_month_%']:.1f}%  (cumul: {ms['wr_cumul_%']:.1f}%)")
            print(f"  Profit factor    : {ms['pf_month']:.3f}   (cumul: {ms['pf_cumul']:.3f})")
            print(f"  Avg net P&L/trade: ${ms['avg_net_pnl_$']:.2f}")
            print(f"  Max DD (month)   : {ms['max_dd_month_%']:.1f}%")
            print(f"  Exit type %%      : {ms['exit_type_pct']}")
            if ms.get("avg_holding_bars"):
                print(f"  Avg holding bars : {ms['avg_holding_bars']:.1f}")
            if ms.get("avg_dtm_time") is not None:
                print(f"  Avg dist-to-mid at TIME exit: ${ms['avg_dtm_time']:.2f}")
            if ms.get("avg_entry_slip_%") is not None:
                print(f"  Avg entry slip   : {ms['avg_entry_slip_%']:.3f}%")
            if ms.get("avg_exit_slip_%") is not None:
                print(f"  Avg exit slip    : {ms['avg_exit_slip_%']:.3f}%")
            print(f"  vs Backtest PF   : {ms['vs_backtest_pf']:+.3f}   WR: {ms['vs_backtest_wr']:+.1f}pp")
        else:
            print("  No trades this month.")
        print(f"  {sep}")


def cmd_open(args):
    sub       = getattr(args, "open_cmd", "list")
    positions = load_open_positions()

    if sub == "add":
        sym = args.symbol.upper()
        pos = {
            "symbol":      sym,
            "signal_date": getattr(args, "signal_date", None),
            "entry_date":  args.entry_date,
            "entry_price": float(args.entry_price),
            "shares":      float(args.shares),
            "accel_sl":    float(getattr(args, "accel_sl", 0) or 0) or None,
            "bb_mid":      float(getattr(args, "bb_mid", 0) or 0) or None,
            "bar_count":   int(getattr(args, "bar_count", 1)),
            "added_at":    datetime.now().isoformat(),
        }
        positions = [p for p in positions if p.get("symbol") != sym]
        positions.append(pos)
        save_open_positions(positions)
        print(f"  Added {sym} @ {pos['entry_price']} | accel_sl={pos['accel_sl']} | bb_mid={pos['bb_mid']}")

    elif sub == "close":
        sym = args.symbol.upper()
        positions = [p for p in positions if p.get("symbol") != sym]
        save_open_positions(positions)
        print(f"  Removed {sym} from open positions. Use 'log' to record the completed trade.")

    elif sub == "list":
        if not positions:
            print("  No open positions.")
            return
        print(f"\n  OPEN POSITIONS ({len(positions)} / {VALIDATED['max_concurrent']} max)")
        print(f"  {'-'*70}")
        print(f"  {'Sym':<6} {'Entry Date':<12} {'Entry Px':>9} {'Shares':>7} {'Bar':>4} "
              f"{'ACCEL_SL':>10} {'BB_MID':>10}")
        print(f"  {'-'*70}")
        for p in positions:
            warn = " *** TIME STOP NEAR" if int(p.get("bar_count", 0)) >= 8 else ""
            print(f"  {p.get('symbol','?'):<6} {p.get('entry_date','?'):<12} "
                  f"{p.get('entry_price', 0):>9.2f} {p.get('shares', 0):>7.0f} "
                  f"{p.get('bar_count', 0):>4} "
                  f"{str(p.get('accel_sl', '---')):>10} "
                  f"{str(p.get('bb_mid', '---')):>10}{warn}")
        print(f"  {'-'*70}")
        if len(positions) >= VALIDATED["max_concurrent"]:
            print(f"  *** AT MAX CONCURRENT POSITIONS -- do not add more. ***")
    else:
        print(f"Unknown sub-command '{sub}'. Use: add / list / close")


def cmd_journal(args):
    """
    Append or display the operational trade journal (trade_journal.txt).

    Format per entry:
        DATE: YYYY-MM-DD
        REGIME: [TRENDING/CORRECTIVE/CHOPPY]
        OPEN POSITIONS: [list or NONE]
        ACTIVITY:
          [SIGNAL/ENTRY/EXIT/SKIP/NEAR_MISS] lines
        NOTES: free text

    Usage:
        python paper_trading_monitor.py journal show
        python paper_trading_monitor.py journal add \\
          --date 2026-03-21 --regime TRENDING \\
          --activity "[ENTRY] GLD — fill=175.20, size=28, stop=172.10, phase=1, risk_frac=0.5%" \\
          --notes "First live trade. No slippage issues."
    """
    sub     = getattr(args, "journal_sub", "show")
    journal = HERE / "trade_journal.txt"

    if sub == "show":
        if not journal.exists():
            print("  No journal yet. Use: python paper_trading_monitor.py journal add --date ... --activity ...")
            return
        print(journal.read_text())
        return

    # sub == "add"
    date_str  = getattr(args, "date", None) or datetime.now().strftime("%Y-%m-%d")
    regime    = (getattr(args, "regime", None) or "---").upper()
    activity  = getattr(args, "activity", None) or ""
    notes     = getattr(args, "notes", None)    or ""

    # Open positions for the header line
    positions = load_open_positions()
    if positions:
        pos_str = ", ".join(
            f"{p.get('symbol','?')} bar={p.get('bar_count','?')}"
            for p in positions
        )
    else:
        pos_str = "NONE"

    sep = "-" * 60
    entry_lines = [
        f"\n{sep}",
        f"DATE: {date_str}",
        f"REGIME: {regime}",
        f"OPEN POSITIONS: {pos_str}",
        f"ACTIVITY:",
    ]
    for line in activity.split(";"):
        line = line.strip()
        if line:
            entry_lines.append(f"  {line}")
    if notes:
        entry_lines.append(f"NOTES: {notes}")
    entry_lines.append(sep)

    entry_text = "\n".join(entry_lines) + "\n"
    with open(journal, "a") as f:
        f.write(entry_text)
    print(f"  Journal entry added for {date_str}")
    print(f"  File: {journal}")


def cmd_export(args):
    trades = load_trades()
    if not trades:
        print("No trades to export.")
        return

    out_path   = HERE / "paper_trades_export.csv"
    fieldnames = [
        "trade_id", "period", "instrument",
        "signal_date", "signal_close", "bb_lower_at_signal", "bb_midline_at_signal",
        "rsi2_at_signal", "atr10_at_signal", "accel_sl_at_signal",
        "entry_date", "entry_price", "entry_slippage_pct", "shares",
        "equity_at_entry", "pct_equity_at_risk",
        "bar_count_at_exit", "exit_date", "exit_price", "exit_reason",
        "exit_slippage_pct", "gross_pnl", "commission", "net_pnl", "won",
        "bb_midline_at_exit", "distance_to_target_at_exit",
        "earnings_checked", "dividend_checked", "news_checked",
        "concurrent_positions_at_entry", "notes",
    ]
    with open(out_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            row = dict(t)
            row.setdefault("instrument", row.get("symbol", ""))
            writer.writerow(row)
    print(f"  Exported {len(trades)} trades -> {out_path}  ({len(fieldnames)} columns)")


def cmd_decide(args):
    """Log a decision to the audit trail (Section 9)."""
    decision_type = getattr(args, "dtype", "OTHER").upper()
    detail        = getattr(args, "detail", "")
    log_decision(decision_type, detail)
    print(f"  Decision logged: [{decision_type}] {detail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p   = argparse.ArgumentParser(
        description="Combo C V4.0 Deployment Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # log
    lp = sub.add_parser("log")
    lp.add_argument("--symbol",            required=True)
    lp.add_argument("--period",            default="PAPER",
                    choices=["PAPER", "LIVE_RAMP", "LIVE_FULL"])
    lp.add_argument("--signal-date",       dest="signal_date")
    lp.add_argument("--signal-close",      dest="signal_close",    type=float)
    lp.add_argument("--bb-lower-signal",   dest="bb_lower_signal", type=float)
    lp.add_argument("--bb-mid-signal",     dest="bb_mid_signal",   type=float)
    lp.add_argument("--rsi2-signal",       dest="rsi2_signal",     type=float)
    lp.add_argument("--atr10-signal",      dest="atr10_signal",    type=float)
    lp.add_argument("--accel-sl-signal",   dest="accel_sl_signal", type=float)
    lp.add_argument("--entry-date",        required=True, dest="entry_date")
    lp.add_argument("--entry-price",       required=True, dest="entry_price",  type=float)
    lp.add_argument("--shares",            required=True, type=float)
    lp.add_argument("--equity-at-entry",   dest="equity_at_entry", type=float)
    lp.add_argument("--entry-slippage-pct",dest="entry_slippage_pct", type=float)
    lp.add_argument("--exit-date",         required=True, dest="exit_date")
    lp.add_argument("--exit-price",        required=True, dest="exit_price",   type=float)
    lp.add_argument("--exit-reason",       required=True, dest="exit_reason",
                    choices=["BB_MID", "SL", "ACCEL_SL", "TIME", "MANUAL_OVERRIDE", "EOB"])
    lp.add_argument("--bars-held",         required=True, dest="bars_held", type=int)
    lp.add_argument("--bb-mid-exit",       dest="bb_mid_exit",     type=float)
    lp.add_argument("--exit-slippage-pct", dest="exit_slippage_pct", type=float)
    lp.add_argument("--earnings-checked",  dest="earnings_checked",  action="store_true")
    lp.add_argument("--dividend-checked",  dest="dividend_checked",  action="store_true")
    lp.add_argument("--news-checked",      dest="news_checked",      action="store_true")
    lp.add_argument("--concurrent",        type=int, default=0)
    lp.add_argument("--notes",             default="")
    lp.add_argument("--force",             action="store_true")

    # check
    sub.add_parser("check")

    # show
    sp = sub.add_parser("show")
    sp.add_argument("--period", choices=["PAPER", "LIVE_RAMP", "LIVE_FULL"], default=None)

    # monthly
    mp = sub.add_parser("monthly")
    mp.add_argument("--month", help="YYYY-MM")

    # export
    sub.add_parser("export")

    # open
    op     = sub.add_parser("open")
    op_sub = op.add_subparsers(dest="open_cmd")
    oa = op_sub.add_parser("add")
    oa.add_argument("--symbol",      required=True)
    oa.add_argument("--signal-date", dest="signal_date")
    oa.add_argument("--entry-date",  required=True, dest="entry_date")
    oa.add_argument("--entry-price", required=True, dest="entry_price", type=float)
    oa.add_argument("--shares",      required=True, type=float)
    oa.add_argument("--accel-sl",    dest="accel_sl",  type=float)
    oa.add_argument("--bb-mid",      dest="bb_mid",    type=float)
    oa.add_argument("--bar-count",   dest="bar_count", type=int, default=1)
    oc = op_sub.add_parser("close")
    oc.add_argument("--symbol",      required=True)
    op_sub.add_parser("list")

    # decide
    dp = sub.add_parser("decide")
    dp.add_argument("--type",   dest="dtype", required=True,
                    choices=["SKIP", "MANUAL_OVERRIDE", "SIZE_ADJUST", "RISK_LIMIT", "OTHER"])
    dp.add_argument("--detail", dest="detail", required=True)

    # journal
    jp      = sub.add_parser("journal", help="Operational trade journal (daily activity log)")
    jp_sub  = jp.add_subparsers(dest="journal_sub")
    jp_sub.add_parser("show", help="Print the full journal")
    ja = jp_sub.add_parser("add", help="Append a new journal entry")
    ja.add_argument("--date",     default=None, help="YYYY-MM-DD (default: today)")
    ja.add_argument("--regime",   default="---",
                    choices=["TRENDING", "CORRECTIVE", "CHOPPY", "---"],
                    help="Market regime from weekly_health_check.py")
    ja.add_argument("--activity", default="",
                    help="Activity lines separated by semicolons. "
                         "Prefix with [SIGNAL], [ENTRY], [EXIT], [SKIP], or [NEAR_MISS].")
    ja.add_argument("--notes",    default="", help="Free-form notes")

    return p.parse_args()


def main():
    args = parse_args()
    dispatch = {
        "log":     cmd_log,
        "check":   cmd_check,
        "show":    cmd_show,
        "monthly": cmd_monthly,
        "export":  cmd_export,
        "open":    cmd_open,
        "decide":  cmd_decide,
        "journal": cmd_journal,
    }
    cmd = args.command or "check"
    if cmd in dispatch:
        dispatch[cmd](args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()
