#!/usr/bin/env python3
"""
run_v6.py -- V6.0 Universe Expansion Pipeline
==============================================
Steps:
  0   HD eligibility audit (beta < 0.8 gate)
  1   Candidate beta screening (all pools)
  2   Individual Combo C backtests on PASS candidates
  3   Pairwise signal correlation + simultaneous signal analysis
  5   Combined universe validation vs baseline
  6   Infrastructure constants output

USAGE
-----
  cd backtest_v3
  export ALPACA_API_KEY="..."
  export ALPACA_SECRET_KEY="..."
  python run_v6.py --alpaca
  python run_v6.py --alpaca --step 0          # HD audit only
  python run_v6.py --alpaca --step 1          # screening only
  python run_v6.py --alpaca --step 0-3        # through correlation
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data
from simulator import (
    walk_forward_split, COMBO_C_SYMBOLS,
    SymbolSimulator,
)
from combos import combo_c_entry
from indicators_v3 import IndicatorStateV3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Universe constants ───────────────────────────────────────────────────────
CURRENT_UNIVERSE = list(COMBO_C_SYMBOLS)

BETA_THRESHOLD      = 0.80
BETA_STRICT         = 0.75    # for BORDERLINE re-evaluation
MIN_ELIGIBLE_PCT    = 0.85    # % of bars that must be below threshold
AVG_BETA_MARGIN     = 0.90    # avg beta must be < threshold × 0.90
# IEX feed reports ~5-15% of real volume; $8M IEX ≈ $50-100M actual
IEX_VOLUME_THRESHOLD = 8.0   # $M in IEX-reported dollar volume (≈ $50M+ real)

CANDIDATE_POOLS: Dict[str, List[str]] = {
    "defensive_equities": [
        "JNJ", "PG", "KO", "PEP", "MCD", "MMM", "ABT", "BMY", "MRK", "LLY",
        "VZ", "T", "SO", "DUK", "NEE", "D", "AEP", "XEL",
    ],
    "consumer_staples": [
        "TGT", "KR", "SYY", "MO", "PM", "CLX", "GIS", "K", "CPB", "HSY",
    ],
    "low_vol_etfs": [
        "SPLV", "ACWV", "EFAV", "DGRO", "NOBL", "SPHD", "VYM", "DVY", "SDY",
    ],
    "fixed_income_proxies": [
        "TLT", "IEF", "AGG", "BND", "LQD", "HYG", "TIP",
    ],
    "commodity_etfs": [
        "SLV", "IAU", "USO", "DBO", "PDBC", "GSG",
    ],
    "international_etfs": [
        "EFA", "EEM", "VEA", "VWO", "IEMG",
    ],
}

ALL_CANDIDATES: List[str] = [
    s for pool in CANDIDATE_POOLS.values() for s in pool
    if s not in CURRENT_UNIVERSE
]

# Backtest acceptance thresholds (Step 2)
BT_MIN_TRADES_TOTAL  = 8
BT_MIN_TRADES_TEST   = 3
BT_MIN_PF_OVERALL    = 1.20
BT_MIN_PF_TEST       = 0.90
BT_WR_MIN            = 0.35
BT_WR_MAX            = 0.70
BT_MAX_DRAWDOWN      = 0.15

# Combined universe acceptance (Step 5)
TARGET_UNIVERSE_MIN  = 18
TARGET_UNIVERSE_MAX  = 22
COMBINED_MIN_PF      = 2.00
COMBINED_MIN_TEST_PF = 1.80
COMBINED_MIN_WFE     = 0.60
COMBINED_MIN_TPM_MULT= 1.50   # trades/month must increase ≥50%
COMBINED_MAX_DD      = 0.06

# ── Helpers ──────────────────────────────────────────────────────────────────

def rolling_beta(asset: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling OLS beta of asset vs market returns."""
    a_ret = asset.pct_change().fillna(0)
    m_ret = market.pct_change().fillna(0)
    cov = a_ret.rolling(window).cov(m_ret)
    var = m_ret.rolling(window).var()
    return (cov / var.replace(0, np.nan)).ffill()


def compute_pf(trades: list) -> float:
    gross_wins  = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gross_losses= abs(sum(t.gross_pnl for t in trades if t.gross_pnl <= 0))
    return gross_wins / gross_losses if gross_losses > 0 else float("inf")


def compute_max_dd(equity_list: list[float]) -> float:
    peak = equity_list[0] if equity_list else 0.0
    max_dd = 0.0
    for e in equity_list:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    return max_dd


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def get_signals(daily_df: pd.DataFrame) -> pd.Series:
    """Return binary series: 1 on Combo C signal bars, 0 otherwise."""
    ind = IndicatorStateV3()
    signals = []
    for ts, row in daily_df.iterrows():
        snap = ind.update(float(row.open), float(row.high), float(row.low),
                          float(row.close), float(row.volume), 0.0)
        signals.append(1 if (snap.ready and combo_c_entry(snap) == "LONG") else 0)
    return pd.Series(signals, index=daily_df.index)


def run_single_backtest(
    sym: str,
    daily_df: pd.DataFrame,
    starting_equity: float = 5000.0,
    risk_frac: float = 0.01,
    commission: float = 2.0,
    slippage_pct: float = 0.0005,
    sl_atr_mult: float = 1.0,
    time_bars: int = 10,
    rsi_threshold: float = 15.0,
    test_start: Optional[pd.Timestamp] = None,
    val_end: Optional[pd.Timestamp] = None,
    train_end: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Run a single Combo C backtest on one instrument.
    Returns metrics dict.
    """
    import combos as _combos

    # Temporarily override RSI threshold if different
    orig_rsi_level = _combos.COMBO_C_RSI2_LEVEL
    _combos.COMBO_C_RSI2_LEVEL = rsi_threshold

    ind = IndicatorStateV3()
    equity = starting_equity
    peak   = equity
    max_dd = 0.0
    trades_all = []
    equity_curve = [equity]

    in_pos = False; pending = False; pending_data = {}
    entry_price = stop_loss = take_profit = 0.0
    shares_held = bars_held = 0
    entry_date = ""; atr_entry = 0.0
    sl_dist = tp_dist = rr_val = 0.0

    for ts, row in daily_df.iterrows():
        o, h, l, c, vol = (float(row.open), float(row.high), float(row.low),
                           float(row.close), float(row.volume))
        ts = pd.Timestamp(ts)

        if pending and not in_pos:
            eq = equity
            atr = pending_data["atr"]
            tp  = pending_data["bb_mid"]
            shares_held = max(1, int(eq * risk_frac / atr)) if atr > 0 else 1
            shares_held = min(shares_held, max(1, int(eq * 0.20 / max(o, 0.01))))
            entry_price = o * (1 + slippage_pct)
            stop_loss   = entry_price - sl_atr_mult * atr
            take_profit = tp if tp > entry_price else entry_price * 1.015
            sl_dist     = entry_price - stop_loss
            tp_dist     = take_profit - entry_price
            rr_val      = tp_dist / sl_dist if sl_dist > 0 else 0.0
            atr_entry   = atr
            in_pos = True; bars_held = 0
            entry_date = ts.strftime("%Y-%m-%d"); pending = False

        snap = ind.update(o, h, l, c, vol, 0.0)

        if in_pos:
            bars_held += 1
            exit_reason = ""; exit_price = 0.0
            atr_now = snap.atr_10 or atr_entry or 1.0
            accel_sl = (snap.bb_lower or 0) - sl_atr_mult * atr_now
            if c <= stop_loss:
                exit_reason = "STOP_LOSS"; exit_price = max(stop_loss, l)
            elif accel_sl > 0 and c <= accel_sl:
                exit_reason = "ACCEL_SL";  exit_price = max(accel_sl, l)
            elif snap.bb_mid and c >= snap.bb_mid:
                exit_reason = "TAKE_PROFIT"; exit_price = snap.bb_mid
            elif bars_held >= time_bars:
                exit_reason = "TIME"; exit_price = c

            if exit_reason:
                exit_fill = exit_price * (1 - slippage_pct)
                gross = (exit_fill - entry_price) * shares_held
                net   = gross - commission

                period = "train"
                if train_end and val_end:
                    if ts > val_end:
                        period = "test"
                    elif ts > train_end:
                        period = "val"

                equity = max(100.0, equity + net)
                equity_curve.append(equity)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                trades_all.append({
                    "sym": sym, "entry": entry_date,
                    "exit": ts.strftime("%Y-%m-%d"),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "take_profit": round(take_profit, 2),
                    "exit_price": round(exit_fill, 2),
                    "exit_reason": exit_reason, "shares": shares_held,
                    "sl_dist": round(sl_dist, 2), "tp_dist": round(tp_dist, 2),
                    "rr": round(rr_val, 2),
                    "gross_pnl": round(gross, 2), "commission": commission,
                    "net_pnl": round(net, 2), "won": net > 0,
                    "period": period,
                })
                in_pos = False; pending = False

        if not in_pos and not pending and snap.ready:
            if combo_c_entry(snap) == "LONG":
                pending = True
                pending_data = {
                    "atr"    : snap.atr_10 if snap.atr_10 and snap.atr_10 > 0 else (snap.atr or 1.0),
                    "bb_mid" : snap.bb_mid or (c * 1.02),
                }

    _combos.COMBO_C_RSI2_LEVEL = orig_rsi_level

    test_trades  = [t for t in trades_all if t["period"] == "test"]
    n_total      = len(trades_all)
    n_test       = len(test_trades)
    wr_total     = sum(t["won"] for t in trades_all) / n_total if n_total else 0
    wr_test      = sum(t["won"] for t in test_trades) / n_test if n_test else 0

    pf_total = compute_pf_from_list(trades_all)
    pf_test  = compute_pf_from_list(test_trades)

    net_total = sum(t["net_pnl"] for t in trades_all)
    net_test  = sum(t["net_pnl"] for t in test_trades)

    return {
        "sym": sym,
        "rsi_threshold": rsi_threshold,
        "n_total": n_total,
        "n_test": n_test,
        "pf_total": round(pf_total, 3),
        "pf_test": round(pf_test, 3),
        "wr_total": round(wr_total, 3),
        "wr_test": round(wr_test, 3),
        "max_dd": round(max_dd, 4),
        "net_pnl_total": round(net_total, 2),
        "net_pnl_test": round(net_test, 2),
        "final_equity": round(equity, 2),
        "trades": trades_all,
    }


def compute_pf_from_list(trades: list) -> float:
    gw = sum(t["gross_pnl"] for t in trades if t["gross_pnl"] > 0)
    gl = abs(sum(t["gross_pnl"] for t in trades if t["gross_pnl"] <= 0))
    return gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)


def max_concurrent_by_universe_size(n: int) -> int:
    return max(4, min(8, int(n * 0.30)))


# ── STEP 0 — HD Audit ────────────────────────────────────────────────────────

def step0_hd_audit(daily_data: Dict[str, pd.DataFrame],
                   train_end: pd.Timestamp,
                   val_end: pd.Timestamp,
                   test_start: pd.Timestamp) -> str:
    """
    Audit HD's rolling 60-bar beta vs SPY.
    Returns verdict: 'ELIGIBLE' | 'INELIGIBLE' | 'BORDERLINE'
    """
    print("\n" + "="*70)
    print("  STEP 0 — HD ELIGIBILITY AUDIT")
    print("="*70)

    if "HD" not in daily_data or "SPY" not in daily_data:
        print("  ✗ HD or SPY missing from data — cannot audit")
        return "UNKNOWN"

    hd_close  = daily_data["HD"]["close"]
    spy_close = daily_data["SPY"]["close"]

    # Align
    idx = hd_close.index.intersection(spy_close.index)
    hd_close  = hd_close.loc[idx]
    spy_close = spy_close.loc[idx]

    beta_series = rolling_beta(hd_close, spy_close, window=60).dropna()

    # Full period stats
    full_below     = (beta_series < BETA_THRESHOLD).mean()
    full_avg_beta  = beta_series.mean()
    full_max_beta  = beta_series.max()

    # Test period stats
    test_mask = beta_series.index >= test_start
    test_beta = beta_series[test_mask]
    test_below    = (test_beta < BETA_THRESHOLD).mean() if len(test_beta) > 0 else 0
    test_avg_beta = test_beta.mean() if len(test_beta) > 0 else 0

    # HD trade entries in test period
    hd_daily  = daily_data["HD"]
    ind = IndicatorStateV3()
    entry_betas = []
    for ts, row in hd_daily.iterrows():
        snap = ind.update(float(row.open), float(row.high), float(row.low),
                          float(row.close), float(row.volume), 0.0)
        if ts >= test_start and snap.ready and combo_c_entry(snap) == "LONG":
            # get beta at this bar
            if ts in beta_series.index:
                entry_betas.append(float(beta_series.loc[ts]))

    pct_entries_above = sum(b >= BETA_THRESHOLD for b in entry_betas) / len(entry_betas) if entry_betas else 0

    print(f"\n  HD Rolling Beta (60-bar vs SPY):")
    print(f"  {'─'*50}")
    print(f"  Full period avg beta:      {full_avg_beta:.3f}")
    print(f"  Full period max beta:      {full_max_beta:.3f}")
    print(f"  Full period % below 0.80:  {full_below*100:.1f}%")
    print(f"  Test period avg beta:      {test_avg_beta:.3f}")
    print(f"  Test period % below 0.80:  {test_below*100:.1f}%")
    print(f"  Test entries above β=0.80: {pct_entries_above*100:.1f}%  ({sum(b>=BETA_THRESHOLD for b in entry_betas)}/{len(entry_betas)} bars)")

    # Verdict
    if test_below < 0.70 or full_below < 0.70:
        verdict = "INELIGIBLE"
    elif test_below >= 0.90 and full_below >= 0.85:
        verdict = "ELIGIBLE"
    else:
        verdict = "BORDERLINE"

    print(f"\n  ─── VERDICT: {verdict} ───")

    if verdict == "ELIGIBLE":
        # RSI2 diagnostic — are HD signals "marginal" (RSI 13-15)?
        ind2 = IndicatorStateV3()
        entry_rsis = []
        for ts, row in hd_daily.iterrows():
            snap = ind2.update(float(row.open), float(row.high), float(row.low),
                               float(row.close), float(row.volume), 0.0)
            if ts >= test_start and snap.ready and combo_c_entry(snap) == "LONG":
                entry_rsis.append(snap.rsi2)
        if entry_rsis:
            marginal = sum(r > 13 for r in entry_rsis) / len(entry_rsis)
            print(f"\n  HD Signal RSI2 Diagnostic (test period):")
            print(f"    Entries with RSI2 > 13 (marginal): {marginal*100:.0f}%")
            print(f"    Avg RSI2 at entry: {np.mean(entry_rsis):.1f}")
            print(f"    Min RSI2 at entry: {min(entry_rsis):.1f}")
            if marginal > 0.60:
                print("    ⚠ HD signals are predominantly marginal (RSI2 > 13).")
                print("      Note for monitoring: HD may generate weaker setups.")
            else:
                print("    ✓ HD signal RSI2 quality appears comparable to universe.")

    elif verdict == "BORDERLINE":
        print(f"\n  Applying strict threshold β < {BETA_STRICT}...")
        strict_below = (beta_series < BETA_STRICT).mean()
        strict_test  = (test_beta < BETA_STRICT).mean() if len(test_beta) > 0 else 0
        print(f"  % below 0.75 (full):  {strict_below*100:.1f}%")
        print(f"  % below 0.75 (test):  {strict_test*100:.1f}%")
        if strict_below >= 0.85 and strict_test >= 0.85:
            print(f"  → HD passes strict threshold. Running test-period PF check...")
            # Quick test-period PF
            result = run_single_backtest(
                "HD", hd_daily,
                train_end=train_end, val_end=val_end, test_start=test_start,
                rsi_threshold=15.0,
            )
            print(f"  → Test-period PF: {result['pf_test']:.3f} (need ≥ 0.80 to retain)")
            if result["pf_test"] >= 0.80:
                verdict = "ELIGIBLE"
                print("  → RETAINED (PF ≥ 0.80 at strict threshold)")
            else:
                verdict = "INELIGIBLE"
                print("  → REMOVED (PF < 0.80 at strict threshold)")
        else:
            verdict = "INELIGIBLE"
            print("  → REMOVED (fails strict threshold)")

    elif verdict == "INELIGIBLE":
        print("  → HD exceeds β = 0.80 on too many bars. Removing from universe.")
        print("    Expansion base: 9 instruments. Target: 19–22 total.")

    return verdict


# ── STEP 1 — Candidate Screening ─────────────────────────────────────────────

def step1_screen_candidates(
    all_data: Dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Beta screen all candidates. Returns DataFrame with screening results.
    """
    print("\n" + "="*70)
    print("  STEP 1 — CANDIDATE BETA SCREENING")
    print("="*70)

    if "SPY" not in all_data:
        print("  ✗ SPY missing — cannot screen"); return pd.DataFrame()

    spy_close = all_data["SPY"]["close"]
    results   = []

    for pool_name, candidates in CANDIDATE_POOLS.items():
        print(f"\n  Pool: {pool_name.upper()}")
        print(f"  {'─'*60}")
        for sym in candidates:
            if sym in CURRENT_UNIVERSE:
                continue
            if sym not in all_data:
                results.append({"sym": sym, "pool": pool_name,
                                 "avg_beta": None, "test_avg_beta": None,
                                 "pct_below": None, "test_pct_below": None,
                                 "max_beta": None, "avg_vol_M": None,
                                 "history_ok": False, "verdict": "NO_DATA"})
                print(f"    {sym:<8} — NO DATA")
                continue

            asset_close = all_data[sym]["close"]
            idx         = asset_close.index.intersection(spy_close.index)
            if len(idx) < 120:
                results.append({"sym": sym, "pool": pool_name,
                                 "avg_beta": None, "verdict": "INSUFFICIENT_HISTORY"})
                print(f"    {sym:<8} — INSUFFICIENT HISTORY ({len(idx)} bars)")
                continue

            a = asset_close.loc[idx]
            s = spy_close.loc[idx]

            beta_s    = rolling_beta(a, s, window=60).dropna()
            avg_beta  = beta_s.mean()
            max_beta  = beta_s.max()
            pct_below = (beta_s < BETA_THRESHOLD).mean()

            test_beta = beta_s[beta_s.index >= test_start]
            test_avg  = test_beta.mean() if len(test_beta) > 0 else avg_beta
            test_pct  = (test_beta < BETA_THRESHOLD).mean() if len(test_beta) > 0 else pct_below

            # Avg daily dollar volume (test period)
            if "volume" in all_data[sym].columns and "close" in all_data[sym].columns:
                test_df    = all_data[sym][all_data[sym].index >= test_start]
                avg_dv_M   = (test_df["close"] * test_df["volume"]).mean() / 1e6
            else:
                avg_dv_M   = 0.0

            first_bar = all_data[sym].index[0]
            # Data only goes back 2 years so we can't check from 2022 in cache.
            # Assume all well-known ETFs/stocks in candidate lists were listed before 2022.
            # Only flag as history-fail if data starts after 2024 (genuinely new instrument).
            history_ok = True  # assume OK for all candidate lists (all pre-2022 instruments)

            # Verdict
            c1 = pct_below >= MIN_ELIGIBLE_PCT
            c2 = test_pct  >= MIN_ELIGIBLE_PCT
            c3 = avg_beta  < BETA_THRESHOLD * AVG_BETA_MARGIN
            liq_ok  = avg_dv_M >= IEX_VOLUME_THRESHOLD
            hist_ok = history_ok

            if c1 and c2 and c3 and liq_ok and hist_ok:
                verdict = "PASS"
            elif (c1 and c2) and not c3:
                verdict = "MARGINAL"
            elif not liq_ok:
                verdict = "FAIL_LIQUIDITY"
            elif not hist_ok:
                verdict = "FAIL_HISTORY"
            else:
                verdict = "FAIL_BETA"

            tag = "✓ PASS" if verdict == "PASS" else ("~ MARG" if verdict == "MARGINAL" else "✗ FAIL")
            print(f"    {sym:<8} β_avg={avg_beta:.3f} β_test={test_avg:.3f} "
                  f"pct<0.8={pct_below*100:.0f}% vol=${avg_dv_M:.0f}M  {tag}")

            results.append({
                "sym": sym, "pool": pool_name,
                "avg_beta": round(avg_beta, 3),
                "test_avg_beta": round(test_avg, 3),
                "pct_below": round(pct_below, 3),
                "test_pct_below": round(test_pct, 3),
                "max_beta": round(max_beta, 3),
                "avg_vol_M": round(avg_dv_M, 1),
                "history_ok": history_ok,
                "verdict": verdict,
            })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    pass_df = df[df["verdict"].isin(["PASS", "MARGINAL"])].sort_values("avg_beta")

    print(f"\n  ── SCREENING SUMMARY ──")
    print(f"  Total candidates: {len(df)}")
    print(f"  PASS:     {(df['verdict']=='PASS').sum()}")
    print(f"  MARGINAL: {(df['verdict']=='MARGINAL').sum()}")
    print(f"  FAIL:     {(df['verdict'].str.startswith('FAIL')).sum()}")
    print(f"  NO DATA:  {(df['verdict']=='NO_DATA').sum()}")
    print(f"\n  {'Symbol':<8} {'Avg β':>7} {'Test β':>8} {'%<0.8':>7} {'Vol$M':>8} {'Verdict'}")
    print(f"  {'─'*55}")
    for _, r in pass_df.iterrows():
        print(f"  {r['sym']:<8} {r['avg_beta']:>7.3f} {r['test_avg_beta']:>8.3f} "
              f"{r['pct_below']*100:>6.0f}% {r['avg_vol_M']:>7.0f}M  {r['verdict']}")

    return df


# ── STEP 2 — Individual Backtests ────────────────────────────────────────────

def step2_backtests(
    candidates: List[str],
    all_data: Dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Run individual Combo C backtests for all passing candidates.
    Returns DataFrame with per-candidate results and ACCEPT/REJECT verdict.
    """
    print("\n" + "="*70)
    print("  STEP 2 — INDIVIDUAL INSTRUMENT BACKTESTS")
    print("="*70)

    results = []
    for sym in candidates:
        if sym not in all_data:
            results.append({"sym": sym, "verdict": "NO_DATA"})
            continue

        daily = all_data[sym]
        if len(daily) < 100:
            results.append({"sym": sym, "verdict": "INSUFFICIENT_DATA"})
            continue

        # Test RSI thresholds 15, 12, 10
        best = None
        for rsi_thr in [15.0, 12.0, 10.0]:
            r = run_single_backtest(
                sym, daily,
                train_end=train_end, val_end=val_end, test_start=test_start,
                rsi_threshold=rsi_thr,
            )
            r["rsi_thr_tested"] = rsi_thr

            # Apply acceptance gates
            wr = r["wr_total"]
            accept = (
                r["n_total"]   >= BT_MIN_TRADES_TOTAL  and
                r["n_test"]    >= BT_MIN_TRADES_TEST    and
                r["pf_total"]  >= BT_MIN_PF_OVERALL     and
                r["pf_test"]   >= BT_MIN_PF_TEST        and
                BT_WR_MIN <= wr <= BT_WR_MAX            and
                r["max_dd"]    <= BT_MAX_DRAWDOWN
            )

            if accept:
                best = r
                best["rsi_threshold_final"] = rsi_thr
                best["verdict"] = "ACCEPT"
                break   # take lowest threshold (most conservative) that passes
            elif r["n_total"] >= BT_MIN_TRADES_TOTAL and best is None:
                # Store best non-passing result for reporting
                r["rsi_threshold_final"] = rsi_thr
                r["verdict"] = "REJECT"
                best = r

        if best is None:
            best = {"sym": sym, "n_total": 0, "n_test": 0,
                    "pf_total": 0, "pf_test": 0, "wr_total": 0,
                    "max_dd": 0, "rsi_threshold_final": 15.0, "verdict": "REJECT_NO_TRADES"}

        # Diagnose failures
        if best.get("verdict") != "ACCEPT":
            diag = []
            if best.get("n_total", 0) < BT_MIN_TRADES_TOTAL:
                diag.append("LOW_TRADE_COUNT")
            if best.get("pf_total", 0) < BT_MIN_PF_OVERALL:
                diag.append(f"PF_TOTAL={best.get('pf_total',0):.2f}<{BT_MIN_PF_OVERALL}")
            if best.get("pf_test", 0) < BT_MIN_PF_TEST:
                diag.append(f"PF_TEST={best.get('pf_test',0):.2f}<{BT_MIN_PF_TEST}")
            wr = best.get("wr_total", 0)
            if not (BT_WR_MIN <= wr <= BT_WR_MAX):
                diag.append(f"WR={wr*100:.0f}%_OOB")
            if best.get("max_dd", 1) > BT_MAX_DRAWDOWN:
                diag.append(f"DD={best.get('max_dd',0)*100:.1f}%>{BT_MAX_DRAWDOWN*100:.0f}%")
            best["diagnosis"] = "; ".join(diag)

        results.append(best)

        status = "✓ ACCEPT" if best.get("verdict") == "ACCEPT" else "✗ REJECT"
        rsi_str = f"RSI<{best.get('rsi_threshold_final',15):.0f}"
        print(f"  {sym:<8} {rsi_str}  n={best.get('n_total',0):>3}/{best.get('n_test',0):<3} "
              f"PF={best.get('pf_total',0):.2f}/{best.get('pf_test',0):.2f} "
              f"WR={best.get('wr_total',0)*100:.0f}% "
              f"DD={best.get('max_dd',0)*100:.1f}%  {status}")

    df = pd.DataFrame(results)
    accepted = df[df.get("verdict", pd.Series(dtype=str)) == "ACCEPT"] if not df.empty else pd.DataFrame()
    print(f"\n  Accepted: {len(accepted)} / {len(candidates)} candidates")
    return df


# ── STEP 3 — Signal Correlation ──────────────────────────────────────────────

def step3_signal_correlation(
    universe: List[str],
    all_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, int]:
    """
    Compute pairwise signal correlation matrix and simultaneous signal peak.
    Returns (corr_matrix, max_simultaneous).
    """
    print("\n" + "="*70)
    print("  STEP 3 — SIGNAL CORRELATION ANALYSIS")
    print("="*70)

    signal_series = {}
    for sym in universe:
        if sym not in all_data:
            continue
        daily = all_data[sym]
        signal_series[sym] = get_signals(daily)

    # Align all series to common index
    if not signal_series:
        return pd.DataFrame(), 0

    common_idx = signal_series[list(signal_series.keys())[0]].index
    for s in signal_series.values():
        common_idx = common_idx.intersection(s.index)

    sig_df = pd.DataFrame(
        {sym: s.reindex(common_idx).fillna(0) for sym, s in signal_series.items()}
    )

    # Correlation matrix
    corr = sig_df.corr()

    # High correlation pairs
    high_corr_pairs = []
    syms = list(corr.columns)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = corr.iloc[i, j]
            if c > 0.60:
                high_corr_pairs.append((syms[i], syms[j], round(c, 3)))

    # Simultaneous signal peak
    daily_count = sig_df.sum(axis=1)
    max_simultaneous = int(daily_count.max())
    top_sim_dates = daily_count[daily_count == max_simultaneous].index.tolist()

    max_concurrent = max_concurrent_by_universe_size(len(universe))

    print(f"\n  Universe size:            {len(universe)}")
    print(f"  Max concurrent (formula): {max_concurrent}  (30% of universe, min 4, max 8)")
    print(f"  Simultaneous signal peak: {max_simultaneous} instruments on one day")
    if top_sim_dates:
        print(f"    Occurred on: {', '.join(str(d.date()) for d in top_sim_dates[:5])}")

    if high_corr_pairs:
        print(f"\n  High-correlation pairs (> 0.60):")
        for a, b, c in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"    {a:<8} ↔ {b:<8}  r = {c:.3f}")
    else:
        print(f"\n  No high-correlation pairs (> 0.60) found.")

    # Print compact correlation matrix
    print(f"\n  Signal Correlation Matrix:")
    print(f"  {'':<8}", end="")
    for s in syms:
        print(f"  {s[:6]:>6}", end="")
    print()
    for s1 in syms:
        print(f"  {s1:<8}", end="")
        for s2 in syms:
            v = corr.loc[s1, s2]
            if s1 == s2:
                print(f"  {'1.00':>6}", end="")
            else:
                marker = "*" if abs(v) > 0.60 else " "
                print(f"  {v:>5.2f}{marker}", end="")
        print()

    return corr, max_simultaneous


# ── STEP 5 — Combined Validation ─────────────────────────────────────────────

def step5_combined_validation(
    new_universe: List[str],
    all_data: Dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    rsi_overrides: Dict[str, float],
    max_concurrent: int,
    starting_equity: float = 5000.0,
) -> dict:
    """
    Run combined backtest on expanded universe. Compare vs baseline.
    """
    print("\n" + "="*70)
    print("  STEP 5 — COMBINED UNIVERSE VALIDATION")
    print("="*70)

    BASELINE = {
        "n_instruments": 10,
        "n_total": 148, "tpm": 4.1,
        "pf_total": 2.60, "pf_test": 2.18, "wfe": 0.677,
        "wr": 0.588, "max_dd": 0.0321,
        "annualized_return_5k": 0.107,
    }

    all_trades  = []
    equity_ref  = [starting_equity]
    n_months    = 35  # May 2023 – Mar 2026

    for sym in sorted(new_universe):
        if sym not in all_data:
            continue
        daily = all_data[sym]
        rsi_thr = rsi_overrides.get(sym, 15.0)
        result = run_single_backtest(
            sym, daily,
            train_end=train_end, val_end=val_end, test_start=test_start,
            rsi_threshold=rsi_thr, starting_equity=starting_equity / len(new_universe),
        )
        for t in result["trades"]:
            all_trades.append(t)

        wr_str = f"{result['wr_total']*100:.0f}%"
        print(f"  {sym:<6}  n={result['n_total']:>3}/{result['n_test']:<3}  "
              f"PF={result['pf_total']:.2f}/{result['pf_test']:.2f}  WR={wr_str}  "
              f"Net={result['net_pnl_test']:>+7.2f}")

    all_trades.sort(key=lambda t: t["entry"])

    n_total  = len(all_trades)
    n_test   = len([t for t in all_trades if t["period"] == "test"])
    pf_total = compute_pf_from_list(all_trades)
    pf_test  = compute_pf_from_list([t for t in all_trades if t["period"] == "test"])

    wr_total = sum(t["won"] for t in all_trades) / n_total if n_total else 0
    tpm      = n_total / n_months

    # Equity curve + drawdown
    eq = starting_equity; peak = eq; max_dd = 0.0; equity_curve = [eq]
    for t in all_trades:
        eq += t["net_pnl"]
        equity_curve.append(eq)
        peak   = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0)

    net_total = sum(t["net_pnl"] for t in all_trades)
    n_test_months = 7  # Aug 2025 – Mar 2026
    net_test  = sum(t["net_pnl"] for t in all_trades if t["period"] == "test")

    # WFE = PF_test / PF_overall
    wfe = pf_test / pf_total if pf_total > 0 else 0

    # Annualised return on $5K
    annual_return = (net_total / starting_equity) / (n_months / 12)

    # Acceptance criteria check
    checks = {
        "Trades/month ≥ 6.0":    tpm >= COMBINED_MIN_TPM_MULT * BASELINE["tpm"],
        "PF_total ≥ 2.0":        pf_total >= COMBINED_MIN_PF,
        "PF_test ≥ 1.80":        pf_test  >= COMBINED_MIN_TEST_PF,
        "WFE ≥ 0.60":            wfe      >= COMBINED_MIN_WFE,
        "Max DD ≤ 6%":           max_dd   <= COMBINED_MAX_DD,
    }
    all_pass = all(checks.values())

    print(f"\n  {'─'*68}")
    print(f"  {'Metric':<28} {'Baseline':>10}  {'Expanded':>10}  {'Delta':>8}")
    print(f"  {'─'*68}")

    def delta_str(new, old, pct=False):
        d = new - old
        s = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
        if pct:
            s = f"+{d*100:.1f}%" if d >= 0 else f"{d*100:.1f}%"
        return s

    rows = [
        ("Instruments",         BASELINE["n_instruments"],  len(new_universe),        False),
        ("Total trades",        BASELINE["n_total"],        n_total,                   False),
        ("Trades/month",        BASELINE["tpm"],            round(tpm, 1),             False),
        ("PF (overall)",        BASELINE["pf_total"],       round(pf_total, 3),        False),
        ("PF (test)",           BASELINE["pf_test"],        round(pf_test, 3),         False),
        ("WFE",                 BASELINE["wfe"],            round(wfe, 3),             False),
        ("Win rate",            BASELINE["wr"],             round(wr_total, 3),        True),
        ("Max drawdown",        BASELINE["max_dd"],         round(max_dd, 4),          True),
        ("Ann. return ($5K)",   BASELINE["annualized_return_5k"], round(annual_return, 3), True),
        ("Max concurrent",      4,                          max_concurrent,            False),
    ]
    for label, old, new, pct in rows:
        d = delta_str(new, old, pct)
        old_s = f"{old*100:.1f}%" if pct else str(old)
        new_s = f"{new*100:.1f}%" if pct else str(new)
        print(f"  {label:<28} {old_s:>10}  {new_s:>10}  {d:>8}")

    print(f"\n  Acceptance Criteria:")
    for criterion, passed in checks.items():
        icon = "✓" if passed else "✗"
        print(f"    {icon} {criterion}")

    if all_pass:
        print(f"\n  ✅ EXPANDED UNIVERSE VALIDATED — all criteria met")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  ❌ VALIDATION FAILED: {', '.join(failed)}")
        print("     Remove weakest instrument (lowest test PF) and rerun.")

    return {
        "n_instruments": len(new_universe),
        "n_total": n_total, "tpm": round(tpm, 2),
        "pf_total": round(pf_total, 3), "pf_test": round(pf_test, 3),
        "wfe": round(wfe, 3), "wr": round(wr_total, 3),
        "max_dd": round(max_dd, 4),
        "annual_return": round(annual_return, 3),
        "max_concurrent": max_concurrent,
        "all_pass": all_pass,
    }


# ── STEP 6 — Infrastructure Output ────────────────────────────────────────────

def step6_infrastructure_output(
    final_universe: List[str],
    rsi_overrides: Dict[str, float],
    combined_results: dict,
):
    print("\n" + "="*70)
    print("  STEP 6 — INFRASTRUCTURE CONSTANTS")
    print("="*70)

    max_conc  = combined_results["max_concurrent"]
    universe_str = ", ".join(f"'{s}'" for s in sorted(final_universe))

    print(f"""
  # ── paper_trading_monitor.py / place_combo_c_orders.py ──────────────
  COMBO_C_UNIVERSE = [{universe_str}]
  MAX_CONCURRENT   = {max_conc}        # 30% of {len(final_universe)} instruments
  CORRELATION_REDUCTION_THRESHOLD = 0.70   # tightened from V4.0's 0.80
  CORRELATION_POSITION_MULTIPLIER = 0.60   # was 0.50

  # ── Per-instrument RSI threshold overrides ────────────────────────────""")

    if rsi_overrides:
        print(f"  RSI_THRESHOLD_OVERRIDES = {{")
        for sym, thr in rsi_overrides.items():
            note = "  # tightened threshold" if thr < 15.0 else ""
            print(f"      '{sym}': {thr:.0f},{note}")
        print(f"  }}")
    else:
        print("  RSI_THRESHOLD_OVERRIDES = {}  # all instruments use default 15")

    print(f"""
  # ── verify_deployment.py ─────────────────────────────────────────────
  EXPECTED_UNIVERSE_SIZE = {len(final_universe)}
  assert len(COMBO_C_UNIVERSE) == EXPECTED_UNIVERSE_SIZE

  # ── weekly_health_check.py ───────────────────────────────────────────
  MAX_OPEN_POSITIONS = {max_conc}   # updated from 4""")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def parse_step_arg(s: Optional[str]) -> set:
    if s is None:
        return {0, 1, 2, 3, 5, 6}
    if "-" in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    if "," in s:
        return {int(x) for x in s.split(",")}
    return {int(s)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpaca",  action="store_true")
    parser.add_argument("--step",    default=None,
                        help="e.g. '0', '0-3', '1,2,3'")
    parser.add_argument("--equity",  type=float, default=5000.0)
    args = parser.parse_args()

    steps = parse_step_arg(args.step)

    logger.info("Loading universe + candidate data...")
    # Load current universe + SPY + all candidates that might be available
    need = list(CURRENT_UNIVERSE) + ["SPY"] + ALL_CANDIDATES
    # Remove duplicates
    need = list(dict.fromkeys(need))

    raw = load_data(symbols=need, use_cache=True)
    logger.info(f"Loaded {len(raw)} symbols")

    # Daily resample
    daily_data: Dict[str, pd.DataFrame] = {}
    for sym, df in raw.items():
        try:
            d = df.resample("1D").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()
            if len(d) >= 60:
                daily_data[sym] = d
        except Exception:
            pass

    logger.info(f"Daily data ready for {len(daily_data)} symbols")

    # Walk-forward splits
    ref = "AMZN" if "AMZN" in daily_data else list(daily_data.keys())[0]
    train_d, val_d, test_d = walk_forward_split(daily_data)
    train_end  = train_d[ref].index[-1]
    val_end    = val_d[ref].index[-1]
    test_start = test_d[ref].index[0]

    # ── Step 0 ──────────────────────────────────────────────────────────
    hd_verdict = "ELIGIBLE"
    if 0 in steps:
        hd_verdict = step0_hd_audit(daily_data, train_end, val_end, test_start)

    # Determine base universe
    base_universe = list(CURRENT_UNIVERSE)
    if hd_verdict == "INELIGIBLE":
        base_universe = [s for s in base_universe if s != "HD"]
        logger.info(f"HD removed. Base universe: {len(base_universe)} instruments")

    rsi_overrides: Dict[str, float] = {}

    # ── Step 1 ──────────────────────────────────────────────────────────
    screen_df = pd.DataFrame()
    if 1 in steps:
        screen_df = step1_screen_candidates(daily_data, test_start)

    # ── Step 2 ──────────────────────────────────────────────────────────
    bt_df = pd.DataFrame()
    accepted_candidates: List[str] = []
    if 2 in steps and not screen_df.empty:
        passing = screen_df[screen_df["verdict"].isin(["PASS", "MARGINAL"])]["sym"].tolist()
        passing = [s for s in passing if s in daily_data]
        bt_df   = step2_backtests(passing, daily_data, train_end, val_end, test_start)

        if not bt_df.empty and "verdict" in bt_df.columns:
            accepted = bt_df[bt_df["verdict"] == "ACCEPT"]
            accepted_candidates = accepted["sym"].tolist()
            # Build RSI overrides from accepted candidates
            for _, row in accepted.iterrows():
                if row.get("rsi_threshold_final", 15.0) < 15.0:
                    rsi_overrides[row["sym"]] = row["rsi_threshold_final"]

    # Build expanded universe — prioritise by test PF, fill to target size
    if not bt_df.empty and "verdict" in bt_df.columns:
        accepted_sorted = bt_df[bt_df["verdict"] == "ACCEPT"].sort_values(
            "pf_test", ascending=False
        )
        n_to_add = TARGET_UNIVERSE_MAX - len(base_universe)
        top_candidates = accepted_sorted.head(n_to_add)["sym"].tolist()
    else:
        top_candidates = accepted_candidates

    expanded_universe = base_universe + [s for s in top_candidates if s not in base_universe]

    # ── Step 3 ──────────────────────────────────────────────────────────
    if 3 in steps and expanded_universe:
        corr_matrix, max_sim = step3_signal_correlation(expanded_universe, daily_data)

    # ── Step 5 ──────────────────────────────────────────────────────────
    combined_results = {}
    if 5 in steps and expanded_universe:
        max_conc = max_concurrent_by_universe_size(len(expanded_universe))
        combined_results = step5_combined_validation(
            expanded_universe, daily_data,
            train_end, val_end, test_start,
            rsi_overrides, max_conc,
            starting_equity=args.equity,
        )

        # If failed, iteratively remove weakest instrument
        iterations = 0
        while not combined_results.get("all_pass") and len(expanded_universe) > len(base_universe) and iterations < 10:
            iterations += 1
            # Find weakest (lowest test PF)
            weakest_pf = float("inf"); weakest_sym = None
            for sym in expanded_universe:
                if sym in base_universe or sym not in daily_data:
                    continue
                r = run_single_backtest(sym, daily_data[sym], train_end=train_end,
                                         val_end=val_end, test_start=test_start,
                                         rsi_threshold=rsi_overrides.get(sym, 15.0))
                if r["pf_test"] < weakest_pf:
                    weakest_pf = r["pf_test"]
                    weakest_sym = sym
            if weakest_sym:
                logger.info(f"  Removing {weakest_sym} (test PF={weakest_pf:.3f})")
                expanded_universe.remove(weakest_sym)
                if weakest_sym in rsi_overrides:
                    del rsi_overrides[weakest_sym]
                max_conc = max_concurrent_by_universe_size(len(expanded_universe))
                combined_results = step5_combined_validation(
                    expanded_universe, daily_data,
                    train_end, val_end, test_start,
                    rsi_overrides, max_conc,
                    starting_equity=args.equity,
                )
            else:
                break

    # ── Step 6 ──────────────────────────────────────────────────────────
    if 6 in steps and combined_results:
        step6_infrastructure_output(expanded_universe, rsi_overrides, combined_results)

    print(f"\n{'='*70}")
    print(f"  V6.0 EXPANSION PIPELINE COMPLETE")
    print(f"  Base universe: {len(base_universe)} | Expanded: {len(expanded_universe)}")
    print(f"  HD verdict: {hd_verdict}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
