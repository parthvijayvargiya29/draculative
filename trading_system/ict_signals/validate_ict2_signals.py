#!/usr/bin/env python3
"""
trading_system/ict_signals/validate_ict2_signals.py
====================================================
Walk-forward bar-simulation validator for all 8 ICT2 signal modules.

Architecture
------------
For each (signal, symbol) pair:
  1. Load 2-year daily bars (yfinance or AlpacaDataFetcher)
  2. Split:  60% train / 20% validation / 20% test  (time-ordered)
  3. Bar-by-bar simulation:
       • Detect signal on bar i
       • Entry next bar open + 0.05% slippage
       • Stop: 1.5× ATR (structure signals) or 1.0× ATR (liquidity signals)
       • Max hold: 10 bars
  4. Compute PF (Profit Factor), Win Rate, expectancy, WFE
  5. Gate:   val PF ≥ 1.10  AND  test PF ≥ 0.90  AND  WFE ≥ 0.60
  6. Assign: DEPLOY | RETUNE | REJECT

Output: data/ict2_validation_report.yaml + pretty console summary

Usage
-----
    # From repo root
    python -m trading_system.ict_signals.validate_ict2_signals [--symbols SPY NVDA ...] [--period 2y]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime as dt

import numpy as np
import pandas as pd
import yaml

# ── Repo root on path ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

# ── Signal module imports ─────────────────────────────────────────────────────
from trading_system.ict_signals.killzone_filter import KillZoneDetector
from trading_system.ict_signals.displacement_detector import DisplacementDetector
from trading_system.ict_signals.nwog_detector import NWOGDetector
from trading_system.ict_signals.propulsion_block_detector import PropulsionBlockDetector
from trading_system.ict_signals.balanced_price_range import BPRDetector
from trading_system.ict_signals.turtle_soup_detector import TurtleSoupDetector
from trading_system.ict_signals.power_of_three import PowerOfThreeDetector
from trading_system.ict_signals.silver_bullet_setup import SilverBulletDetector

# ── Constants ─────────────────────────────────────────────────────────────────
SLIPPAGE_PCT     = 0.0005   # 0.05%
MAX_HOLD_BARS    = 10
ATR_PERIOD       = 14
STRUCTURE_ATR    = 1.5      # stop multiple for structure signals
LIQUIDITY_ATR    = 1.0      # stop multiple for liquidity signals

GATE_VAL_PF      = 1.10
GATE_TEST_PF     = 0.90
GATE_WFE         = 0.60

STRUCTURE_SIGS   = {"displacement", "propulsion_block", "silver_bullet", "bpr", "po3"}
LIQUIDITY_SIGS   = {"turtle_soup", "nwog", "kill_zone"}

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "VXX", "TLT", "GLD", "UUP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC",
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
]

# ── Data helpers ──────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def load_bars(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Load daily OHLCV bars. Falls back to yfinance if Alpaca not configured."""
    try:
        from simulation.alpaca_data_fetcher import AlpacaDataFetcher
        fetcher = AlpacaDataFetcher()
        df = fetcher.load_symbol(symbol, timeframe="1Day")
        if df is not None and len(df) >= 100:
            df.columns = [c.lower() for c in df.columns]
            df.reset_index(inplace=True, drop=False)
            return df
    except Exception:
        pass

    # yfinance fallback
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def split_bars(
    df: pd.DataFrame,
    train_end_date: Optional[pd.Timestamp] = None,
    val_end_date:   Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """FIX 3: Date-based 60/20/20 walk-forward split.
    If train_end_date / val_end_date are supplied (computed from first symbol),
    ALL subsequent symbols use the same calendar dates for true OOS isolation.
    Otherwise compute dates from this df's index.
    """
    n   = len(df)
    t1  = int(n * 0.60)
    t2  = int(n * 0.80)

    # Try to use the date column for slicing
    date_col = "date" if "date" in df.columns else None
    if date_col:
        dates = pd.to_datetime(df[date_col], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        # Fallback to positional split if no date info
        return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]

    if train_end_date is None:
        train_end_date = dates.iloc[t1]
    if val_end_date is None:
        val_end_date   = dates.iloc[t2]

    train = df[dates <= train_end_date]
    val   = df[(dates > train_end_date) & (dates <= val_end_date)]
    test  = df[dates > val_end_date]
    return train, val, test


# ── Simulation engine ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar  : int
    entry_price: float
    direction  : str   # "long" | "short"
    stop_loss  : float
    exit_bar   : Optional[int]   = None
    exit_price : Optional[float] = None
    pnl_pct    : float = 0.0


def _profit_factor(trades: List[Trade]) -> float:
    gains  = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    losses = sum(-t.pnl_pct for t in trades if t.pnl_pct < 0)
    return gains / losses if losses > 0 else (999.0 if gains > 0 else 1.0)


def _win_rate(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl_pct > 0) / len(trades)


def _expectancy(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(t.pnl_pct for t in trades) / len(trades)


def simulate_signal(
    signal_name: str,
    df: pd.DataFrame,
    atr_multiple: float,
) -> List[Trade]:
    """
    Bar-by-bar simulation for a single signal on a given DataFrame slice.
    Returns list of closed Trade objects.
    """
    trades: List[Trade] = []
    atr_series = _atr(df)
    n = len(df)

    # Detector factory (re-instantiated per slice to reset state)
    def make_detector():
        if signal_name == "displacement":
            return DisplacementDetector()
        elif signal_name == "nwog":
            return NWOGDetector()
        elif signal_name == "propulsion_block":
            return PropulsionBlockDetector()
        elif signal_name == "bpr":
            return BPRDetector()
        elif signal_name == "turtle_soup":
            return TurtleSoupDetector()
        elif signal_name == "po3":
            return PowerOfThreeDetector(expected_direction="bullish")
        elif signal_name == "silver_bullet":
            return SilverBulletDetector()
        elif signal_name == "kill_zone":
            return KillZoneDetector(htf_bias="bullish")
        else:
            raise ValueError(f"Unknown signal: {signal_name}")

    detector = make_detector()
    # min bars needed before detecting
    min_bars = max(20, ATR_PERIOD + 5)

    open_trade: Optional[Trade] = None

    for i in range(min_bars, n - 1):
        slice_df = df.iloc[: i + 1]

        # ── Manage open trade ──────────────────────────────────────────────────
        if open_trade is not None:
            cur_high  = float(df["high"].iloc[i])
            cur_low   = float(df["low"].iloc[i])
            cur_close = float(df["close"].iloc[i])

            # ─── EXIT HIERARCHY (FIX 2: Tiered exit) ──────────────────────────
            # Priority 1: Stop loss hit
            if open_trade.direction == "long" and cur_low <= open_trade.stop_loss:
                exit_p = open_trade.stop_loss * (1 - SLIPPAGE_PCT)
                open_trade.exit_bar   = i
                open_trade.exit_price = exit_p
                open_trade.pnl_pct    = (exit_p - open_trade.entry_price) / open_trade.entry_price
                trades.append(open_trade)
                open_trade = None
                continue
            elif open_trade.direction == "short" and cur_high >= open_trade.stop_loss:
                exit_p = open_trade.stop_loss * (1 + SLIPPAGE_PCT)
                open_trade.exit_bar   = i
                open_trade.exit_price = exit_p
                open_trade.pnl_pct    = (open_trade.entry_price - exit_p) / open_trade.entry_price
                trades.append(open_trade)
                open_trade = None
                continue

            # Priority 2: Take profit — price reaches nearest opposing ICT level
            # Use 2× stop distance as TP proxy (nearest FVG/OB approximation)
            cur_atr_i = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else \
                        float(df["close"].iloc[i]) * 0.01
            tp_dist   = atr_multiple * 2.0 * cur_atr_i
            if open_trade.direction == "long":
                tp_level = open_trade.entry_price + tp_dist
                if cur_high >= tp_level:
                    exit_p = tp_level * (1 - SLIPPAGE_PCT)
                    open_trade.exit_bar   = i
                    open_trade.exit_price = exit_p
                    open_trade.pnl_pct    = (exit_p - open_trade.entry_price) / open_trade.entry_price
                    trades.append(open_trade)
                    open_trade = None
                    continue
            else:
                tp_level = open_trade.entry_price - tp_dist
                if cur_low <= tp_level:
                    exit_p = tp_level * (1 + SLIPPAGE_PCT)
                    open_trade.exit_bar   = i
                    open_trade.exit_price = exit_p
                    open_trade.pnl_pct    = (open_trade.entry_price - exit_p) / open_trade.entry_price
                    trades.append(open_trade)
                    open_trade = None
                    continue

            # Priority 3: Signal reversal — same detector fires opposite direction
            try:
                rev_result = detector.update(slice_df)
                rev_det, rev_dir = _extract_signal(signal_name, rev_result)
                if rev_det and rev_dir != open_trade.direction:
                    exit_p = cur_close * (
                        (1 - SLIPPAGE_PCT) if open_trade.direction == "long" else (1 + SLIPPAGE_PCT)
                    )
                    open_trade.exit_bar   = i
                    open_trade.exit_price = exit_p
                    open_trade.pnl_pct    = (
                        (exit_p - open_trade.entry_price) / open_trade.entry_price
                        if open_trade.direction == "long"
                        else (open_trade.entry_price - exit_p) / open_trade.entry_price
                    )
                    trades.append(open_trade)
                    open_trade = None
                    continue
            except Exception:
                pass

            # Priority 4: Time stop (max hold bars)
            if i - open_trade.entry_bar >= MAX_HOLD_BARS:
                exit_p = float(df["open"].iloc[i]) * (
                    (1 - SLIPPAGE_PCT) if open_trade.direction == "long" else (1 + SLIPPAGE_PCT)
                )
                open_trade.exit_bar   = i
                open_trade.exit_price = exit_p
                open_trade.pnl_pct    = (
                    (exit_p - open_trade.entry_price) / open_trade.entry_price
                    if open_trade.direction == "long"
                    else (open_trade.entry_price - exit_p) / open_trade.entry_price
                )
                trades.append(open_trade)
                open_trade = None
            # else stay in trade
            continue

        # ── Detect signal ──────────────────────────────────────────────────────
        try:
            if signal_name == "kill_zone":
                current_price = float(df["close"].iloc[i])
                result = detector.process(None, current_price)
                if not result.in_high_prob_window:
                    continue
                direction = "long" if result.bias_direction == "bullish" else "short"
                signal_detected = True
            else:
                result = detector.update(slice_df)
                signal_detected, direction = _extract_signal(signal_name, result)
        except Exception:
            continue

        if not signal_detected:
            continue

        # Entry next bar open + slippage
        next_open = float(df["open"].iloc[i + 1])
        cur_atr   = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else \
                    float(df["close"].iloc[i]) * 0.01

        entry_price = next_open * (1 + SLIPPAGE_PCT if direction == "long" else (1 - SLIPPAGE_PCT))
        stop_dist   = atr_multiple * cur_atr

        if direction == "long":
            stop_loss = entry_price - stop_dist
        else:
            stop_loss = entry_price + stop_dist

        open_trade = Trade(
            entry_bar=i + 1,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
        )

    # Close any remaining open trade at last close
    if open_trade is not None:
        last_close = float(df["close"].iloc[-1])
        exit_p = last_close * (1 - SLIPPAGE_PCT if open_trade.direction == "long" else (1 + SLIPPAGE_PCT))
        open_trade.exit_bar   = n - 1
        open_trade.exit_price = exit_p
        open_trade.pnl_pct    = (
            (exit_p - open_trade.entry_price) / open_trade.entry_price
            if open_trade.direction == "long"
            else (open_trade.entry_price - exit_p) / open_trade.entry_price
        )
        trades.append(open_trade)

    return trades


def _extract_signal(signal_name: str, result) -> Tuple[bool, str]:
    """
    Map a signal result object to (signal_detected: bool, direction: str).
    Returns ("long" | "short") as direction.
    """
    if signal_name == "displacement":
        if result.detected:
            return True, ("long" if result.direction == "bullish" else "short")
    elif signal_name == "nwog":
        if result.bias_from_gaps in ("bullish", "bearish"):
            return True, ("long" if result.bias_from_gaps == "bullish" else "short")
    elif signal_name == "propulsion_block":
        if result.detected and not result.mitigated:
            return True, ("long" if result.direction == "bullish" else "short")
    elif signal_name == "bpr":
        if result.nearest_bpr_below or result.nearest_bpr_above:
            # Use CE proximity: if nearest is below → bullish pull, above → bearish
            if result.nearest_bpr_below:
                return True, "long"
            else:
                return True, "short"
    elif signal_name == "turtle_soup":
        if result.detected:
            return True, ("long" if result.direction == "long" else "short")
    elif signal_name == "po3":
        if result.phase == "distribution":
            return True, ("long" if result.expected_direction == "bullish" else "short")
    elif signal_name == "silver_bullet":
        if result.setup_valid:
            direction = "long" if result.target_price > result.entry_zone_midpoint else "short"
            return True, direction
    return False, "long"


# ── Per-signal, per-symbol result ─────────────────────────────────────────────

@dataclass
class SignalSymbolResult:
    signal_name    : str
    symbol         : str
    n_train_trades : int
    n_val_trades   : int
    n_test_trades  : int
    train_pf       : float
    val_pf         : float
    test_pf        : float
    val_win_rate   : float
    test_win_rate  : float
    wfe            : float       # Walk-Forward Efficiency
    expectancy     : float
    status         : str         # DEPLOY | RETUNE | REJECT


@dataclass
class AggregateResult:
    signal_name       : str
    symbols_tested    : int
    deploy_count      : int
    retune_count      : int
    reject_count      : int
    mean_val_pf       : float
    mean_test_pf      : float
    mean_wfe          : float
    recommended_status: str
    per_symbol        : List[SignalSymbolResult] = field(default_factory=list)


# ── Main validation runner ────────────────────────────────────────────────────

SIGNAL_NAMES = [
    "displacement",
    "nwog",
    "propulsion_block",
    "bpr",
    "turtle_soup",
    "po3",
    "silver_bullet",
    "kill_zone",
]


def validate_all(symbols: List[str], period: str = "2y") -> Dict[str, AggregateResult]:
    results: Dict[str, AggregateResult] = {}

    # FIX 3: Compute shared date boundaries from first successful symbol load
    _shared_train_end: Optional[pd.Timestamp] = None
    _shared_val_end:   Optional[pd.Timestamp] = None

    # Pre-load bars and establish the canonical date split
    print("  Pre-loading bars to establish canonical date split …")
    bars_cache: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_bars(sym, period=period)
            if len(df) >= 100:
                bars_cache[sym] = df
                if _shared_train_end is None:
                    n = len(df)
                    t1, t2 = int(n * 0.60), int(n * 0.80)
                    date_col = "date" if "date" in df.columns else None
                    if date_col:
                        dates = pd.to_datetime(df[date_col], utc=True)
                        _shared_train_end = dates.iloc[t1]
                        _shared_val_end   = dates.iloc[t2]
                        print(f"    Date split: train ≤ {_shared_train_end.date()}  "
                              f"val ≤ {_shared_val_end.date()}  test > {_shared_val_end.date()}")
        except Exception as ex:
            print(f"    [{sym}] load error: {ex}")

    for sig in SIGNAL_NAMES:
        atr_mult = LIQUIDITY_ATR if sig in LIQUIDITY_SIGS else STRUCTURE_ATR
        agg = AggregateResult(
            signal_name=sig,
            symbols_tested=0,
            deploy_count=0,
            retune_count=0,
            reject_count=0,
            mean_val_pf=0.0,
            mean_test_pf=0.0,
            mean_wfe=0.0,
            recommended_status="REJECT",
        )

        for sym in symbols:
            print(f"  [{sig}] {sym} ... ", end="", flush=True)
            try:
                df = bars_cache.get(sym)
                if df is None or len(df) < 100:
                    print("SKIP (not cached or too few bars)")
                    continue

                train_df, val_df, test_df = split_bars(df, _shared_train_end, _shared_val_end)

                train_trades = simulate_signal(sig, train_df, atr_mult)
                val_trades   = simulate_signal(sig, val_df,   atr_mult)
                test_trades  = simulate_signal(sig, test_df,  atr_mult)

                val_pf   = _profit_factor(val_trades)
                test_pf  = _profit_factor(test_trades)
                train_pf = _profit_factor(train_trades)

                # Walk-Forward Efficiency = out-of-sample PF / in-sample PF
                wfe = (test_pf / train_pf) if train_pf > 0 else 0.0

                if val_pf >= GATE_VAL_PF and test_pf >= GATE_TEST_PF and wfe >= GATE_WFE:
                    status = "DEPLOY"
                elif val_pf >= GATE_VAL_PF * 0.85 or test_pf >= GATE_TEST_PF * 0.85:
                    status = "RETUNE"
                else:
                    status = "REJECT"

                sr = SignalSymbolResult(
                    signal_name=sig,
                    symbol=sym,
                    n_train_trades=len(train_trades),
                    n_val_trades=len(val_trades),
                    n_test_trades=len(test_trades),
                    train_pf=round(train_pf, 3),
                    val_pf=round(val_pf, 3),
                    test_pf=round(test_pf, 3),
                    val_win_rate=round(_win_rate(val_trades), 3),
                    test_win_rate=round(_win_rate(test_trades), 3),
                    wfe=round(wfe, 3),
                    expectancy=round(_expectancy(test_trades) * 100, 4),  # as %
                    status=status,
                )
                agg.per_symbol.append(sr)
                agg.symbols_tested += 1
                if status == "DEPLOY":
                    agg.deploy_count += 1
                elif status == "RETUNE":
                    agg.retune_count += 1
                else:
                    agg.reject_count += 1

                print(f"val_pf={val_pf:.3f} test_pf={test_pf:.3f} wfe={wfe:.3f} → {status}")

            except Exception as ex:
                print(f"ERROR: {ex}")
                continue

        # Aggregate stats
        if agg.per_symbol:
            agg.mean_val_pf  = round(float(np.mean([r.val_pf  for r in agg.per_symbol])), 3)
            agg.mean_test_pf = round(float(np.mean([r.test_pf for r in agg.per_symbol])), 3)
            agg.mean_wfe     = round(float(np.mean([r.wfe     for r in agg.per_symbol])), 3)

            deploy_frac = agg.deploy_count / agg.symbols_tested if agg.symbols_tested else 0
            if deploy_frac >= 0.50:
                agg.recommended_status = "DEPLOY"
            elif deploy_frac + agg.retune_count / max(agg.symbols_tested, 1) >= 0.50:
                agg.recommended_status = "RETUNE"
            else:
                agg.recommended_status = "REJECT"

        results[sig] = agg

    return results


# ── Failure recovery protocol ─────────────────────────────────────────────────────

FAILURE_RECOVERY_MAP = {
    "LOW_TRADE_COUNT": {
        "diagnosis": "Signal fires too infrequently to be statistically reliable.",
        "fix_options": [
            "Loosen the primary condition threshold (check ict_signals.yaml)",
            "Reduce lookback_bars for swing high/low detection",
            "Add this signal as a confluence filter only, not standalone entry",
        ],
    },
    "PF_FAILURE": {
        "diagnosis": "Signal has positive trades but losers outweigh winners.",
        "fix_options": [
            "Tighten the ATR stop multiplier from 1.5 to 1.0",
            "Add kill zone filter (only take signals in london/ny_open zones)",
            "Require displacement confirmation before entry",
            "Add BPR proximity filter (only enter when near a BPR level)",
        ],
    },
    "WFE_FAILURE": {
        "diagnosis": "Signal overfits to training data. Test PF << Train PF.",
        "fix_options": [
            "Remove the most recent parameter change",
            "Simplify the entry condition (fewer conjunctive gates = more robust)",
            "Switch from fixed threshold to ATR-relative threshold",
        ],
    },
}


def print_failure_recovery(results: Dict[str, AggregateResult]) -> None:
    """Print failure diagnosis and fix options for each failed signal."""
    failed = {sig: agg for sig, agg in results.items() if agg.recommended_status != "DEPLOY"}
    if not failed:
        print("\n  ✅ All signals passed — no failure recovery needed.")
        return

    print("\n" + "=" * 72)
    print("ICT2 SIGNAL FAILURE RECOVERY PROTOCOL")
    print("=" * 72)

    for sig, agg in failed.items():
        n = agg.symbols_tested or 1
        # Determine failure type
        if agg.symbols_tested > 0 and (agg.deploy_count + agg.retune_count) == 0:
            total_test_trades = sum(
                r.n_test_trades for r in agg.per_symbol
            )
            if total_test_trades < 5:
                fail_key = "LOW_TRADE_COUNT"
            elif agg.mean_wfe < GATE_WFE:
                fail_key = "WFE_FAILURE"
            else:
                fail_key = "PF_FAILURE"
        elif agg.mean_wfe < GATE_WFE:
            fail_key = "WFE_FAILURE"
        else:
            fail_key = "PF_FAILURE"

        rec = FAILURE_RECOVERY_MAP.get(fail_key, FAILURE_RECOVERY_MAP["PF_FAILURE"])
        gate_label = (
            f"val_pf={agg.mean_val_pf:.3f} test_pf={agg.mean_test_pf:.3f} "
            f"wfe={agg.mean_wfe:.3f}"
        )
        print(f"\n  ✗ {sig}: {agg.recommended_status}  ({gate_label})")
        print(f"    Diagnosis: {rec['diagnosis']}")
        print(f"    Fix options:")
        for opt_i, opt in enumerate(rec["fix_options"], 1):
            print(f"      {opt_i}. {opt}")


# ── Reporting ──────────────────────────────────────────────────────────────────

def save_report(results: Dict[str, AggregateResult], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": dt.datetime.now().isoformat(),
        "gates": {
            "val_pf_min": GATE_VAL_PF,
            "test_pf_min": GATE_TEST_PF,
            "wfe_min": GATE_WFE,
        },
        "signals": {},
    }
    for sig, agg in results.items():
        agg_dict = asdict(agg)
        report["signals"][sig] = agg_dict

    with open(out_path, "w") as fh:
        yaml.safe_dump(report, fh, default_flow_style=False, sort_keys=False)
    print(f"\n✅ YAML report saved → {out_path}")


def save_markdown_report(results: Dict[str, AggregateResult], out_path: Path) -> None:
    """Generate human-readable markdown report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ICT2 Signal Validation Report",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Universe: {next(iter(results.values())).symbols_tested} symbols, 2-year Alpaca daily bars",
        f"Gates: val_pf ≥ {GATE_VAL_PF} | test_pf ≥ {GATE_TEST_PF} | wfe ≥ {GATE_WFE}",
        "",
        "## Summary",
        "| Signal | Trades (train/test) | PF (train/val/test) | WFE | Gate |",
        "|--------|---------------------|---------------------|-----|------|`",
    ]

    # Fix the table header line (remove trailing backtick)
    lines[-1] = lines[-1].replace("`", "")

    status_emoji = {"DEPLOY": "✅ DEPLOY", "RETUNE": "⚠ RETUNE", "REJECT": "❌ REJECT"}

    for sig, agg in results.items():
        total_train = sum(r.n_train_trades for r in agg.per_symbol)
        total_test  = sum(r.n_test_trades  for r in agg.per_symbol)
        emoji       = status_emoji.get(agg.recommended_status, agg.recommended_status)
        lines.append(
            f"| {sig:<20} | {total_train:>6} / {total_test:<5} "
            f"| {agg.mean_val_pf:.2f} / {agg.mean_val_pf:.2f} / {agg.mean_test_pf:.2f} "
            f"| {agg.mean_wfe:.2f} | {emoji} |"
        )

    lines += [
        "",
        "## Per-Signal Detail",
    ]
    for sig, agg in results.items():
        lines.append(f"\n### `{sig}` — {agg.recommended_status}")
        lines.append(f"- Mean val PF: {agg.mean_val_pf:.3f} | Mean test PF: {agg.mean_test_pf:.3f} | Mean WFE: {agg.mean_wfe:.3f}")
        if agg.per_symbol:
            best  = max(agg.per_symbol, key=lambda r: r.test_pf)
            worst = min(agg.per_symbol, key=lambda r: r.test_pf)
            lines.append(f"- Best symbol:  {best.symbol} (test_pf={best.test_pf:.3f})")
            lines.append(f"- Worst symbol: {worst.symbol} (test_pf={worst.test_pf:.3f})")

    lines += [
        "",
        "## Recommended Weight Adjustments",
        ""
        "Signals that **DEPLOY** retain their `convergence_weights.yaml` weight.  ",
        "Signals that **RETUNE** get weight × 0.6.  ",
        "Signals that **REJECT** get weight → 0.0 (use as confluence filter only).",
        "",
    ]

    # Load current convergence weights
    cw_path = _ROOT / "configs" / "convergence_weights.yaml"
    current_weights: Dict[str, float] = {}
    if cw_path.exists():
        with open(cw_path) as fh:
            import yaml as _yaml
            cw = _yaml.safe_load(fh) or {}
            current_weights = {k: float(v) for k, v in cw.get("signal_limits", {}).items()}

    lines.append("| Signal | Current Weight | Recommended Weight | Reason |")
    lines.append("|--------|---------------|-------------------|--------|")
    for sig, agg in results.items():
        cur_w = current_weights.get(sig, 0.20)
        if agg.recommended_status == "DEPLOY":
            rec_w, reason = cur_w, "Passes all gates"
        elif agg.recommended_status == "RETUNE":
            rec_w, reason = round(cur_w * 0.6, 3), f"Partial pass (wfe={agg.mean_wfe:.2f})"
        else:
            rec_w, reason = 0.0, "Fails gates — confluence only"
        lines.append(f"| {sig:<20} | {cur_w:.3f} | {rec_w:.3f} | {reason} |")

    lines += ["", f"---", f"*Auto-generated by Draculative Alpha Engine*"]
    out_path.write_text("\n".join(lines))
    print(f"✅ Markdown report saved → {out_path}")


def print_summary(results: Dict[str, AggregateResult]):
    print("\n" + "=" * 72)
    print("ICT2 SIGNAL WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 72)
    fmt = "{:<20} {:>6} {:>8} {:>9} {:>6}  {:>7}  {}"
    print(fmt.format("Signal", "Syms", "ValPF", "TestPF", "WFE", "Deploy%", "Status"))
    print("-" * 72)
    for sig, agg in results.items():
        n = agg.symbols_tested or 1
        dp = f"{agg.deploy_count}/{n}"
        print(fmt.format(
            sig,
            agg.symbols_tested,
            f"{agg.mean_val_pf:.3f}",
            f"{agg.mean_test_pf:.3f}",
            f"{agg.mean_wfe:.3f}",
            dp,
            agg.recommended_status,
        ))
    print("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICT2 signal walk-forward validator"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Symbols to validate (default: full 23-symbol universe)"
    )
    parser.add_argument(
        "--period", default="2y",
        help="Data period string (default: 2y)"
    )
    parser.add_argument(
        "--out", default="data/ict2_validation_report.yaml",
        help="Output YAML report path (relative to repo root)"
    )
    args = parser.parse_args()

    print(f"ICT2 Validation — {len(args.symbols)} symbols, period={args.period}")
    print(f"Signals: {SIGNAL_NAMES}\n")

    results = validate_all(args.symbols, period=args.period)
    print_summary(results)
    print_failure_recovery(results)

    yaml_path = _ROOT / args.out
    save_report(results, yaml_path)
    md_path   = yaml_path.with_suffix(".md")
    save_markdown_report(results, md_path)


if __name__ == "__main__":
    main()
