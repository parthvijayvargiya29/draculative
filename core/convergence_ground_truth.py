#!/usr/bin/env python3
"""
core/convergence_ground_truth.py
=================================
Answers: "When the ICT2ConvergenceEngine says STRONG_BUY (final_score ≥ 0.65),
how often does the asset close higher the next day?"

METHODOLOGY
-----------
  1. Load 2-year daily bars for symbol (default: SPY) from Alpaca cache.
  2. For each bar i from bar 80 onward:
       a. Run all 8 ICT2 detectors on bars [0..i]
       b. Run NucleusIdentificationEngine + ICT2 adjustments
       c. Run ICT2ConvergenceEngine.score()
       d. Record: date, final_score, direction, nucleus_multiplier, etc.
       e. Record: next_day_return = (close[i+1] - close[i]) / close[i]
  3. Directional accuracy on high-confidence bars (|final_score| ≥ 0.65)
  4. Score calibration (Spearman rank correlation across 0.1-width buckets)
  5. Per-signal-group accuracy (which groups add real predictive value)
  6. Auto-generate weight adjustment suggestions

GATES
-----
  directional_accuracy ≥ 0.52  AND  calibration_spearman ≥ 0.30

OUTPUT
------
  data/convergence_ground_truth_report.yaml
  Console formatted table

CACHE
-----
  data/convergence_cache/<symbol>_<date_range_hash>_<cfg_hash>.pkl

Usage:
  python -m core.convergence_ground_truth [--symbol SPY] [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ── Repo root ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
HIGH_CONF_THRESHOLD = 0.65
GATE_DIR_ACCURACY   = 0.52
GATE_SPEARMAN       = 0.30
CACHE_DIR           = _ROOT / "data" / "convergence_cache"
REPORT_PATH         = _ROOT / "data" / "convergence_ground_truth_report.yaml"
WARMUP_BARS         = 80
BUCKET_WIDTH        = 0.1

# ── ICT2 imports ──────────────────────────────────────────────────────────────
try:
    from trading_system.ict_signals.killzone_filter import KillZoneDetector
    from trading_system.ict_signals.displacement_detector import DisplacementDetector
    from trading_system.ict_signals.nwog_detector import NWOGDetector
    from trading_system.ict_signals.propulsion_block_detector import PropulsionBlockDetector
    from trading_system.ict_signals.balanced_price_range import BPRDetector
    from trading_system.ict_signals.turtle_soup_detector import TurtleSoupDetector
    from trading_system.ict_signals.power_of_three import PowerOfThreeDetector
    from trading_system.ict_signals.silver_bullet_setup import SilverBulletDetector
    from core.ict2_convergence_engine import ICT2ConvergenceEngine, ConvergenceScore
    _ICT2_OK = True
except Exception as _e:
    print(f"[convergence_ground_truth] ICT2 modules unavailable: {_e}")
    _ICT2_OK = False

try:
    from core.nucleus_validator import StandaloneNucleusScorer, NUCLEUS_TYPES
    from core.nucleus_validator import NucleusValidator as _NV
    _NUCLEUS_OK = True
except Exception as _e:
    print(f"[convergence_ground_truth] Nucleus modules unavailable: {_e}")
    _NUCLEUS_OK = False

# ── Data loader ───────────────────────────────────────────────────────────────

def _load_bars(symbol: str) -> pd.DataFrame:
    """Load via Alpaca cache, fall back to yfinance."""
    try:
        from simulation.alpaca_data_fetcher import AlpacaDataFetcher
        fetcher = AlpacaDataFetcher()
        df = fetcher.load_symbol(symbol, timeframe="1Day")
        if df is not None and len(df) >= 100:
            df.columns = [c.lower() for c in df.columns]
            df.reset_index(drop=False, inplace=True)
            return df
    except Exception:
        pass
    import yfinance as yf
    df = yf.Ticker(symbol).history(period="2y", interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = None
        for c in ("date", "timestamp", "datetime", "index"):
            if c in df.columns:
                date_col = c
                break
        if date_col:
            df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index, utc=True)
    return df


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cfg_hash() -> str:
    cfg_files = [
        _ROOT / "configs" / "convergence_weights.yaml",
        _ROOT / "configs" / "nucleus_ict2_adjustments.yaml",
    ]
    h = hashlib.md5()
    for p in cfg_files:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _cache_key(symbol: str, df: pd.DataFrame) -> str:
    date_range = f"{df.index[0].date()}_{df.index[-1].date()}" if isinstance(df.index, pd.DatetimeIndex) else str(len(df))
    return f"{symbol}_{date_range}_{_cfg_hash()}"


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def _load_cache(key: str) -> Optional[List[dict]]:
    p = _cache_path(key)
    if p.exists():
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
    return None


def _save_cache(key: str, data: List[dict]) -> None:
    p = _cache_path(key)
    with open(p, "wb") as fh:
        pickle.dump(data, fh)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BarRecord:
    bar_date:           str
    final_score:        float
    raw_score:          float
    nucleus_multiplier: float
    direction:          str
    confidence:         float
    top_signals:        List[str]
    signal_breakdown:   Dict[str, float]   # group → score
    next_day_return:    float              # NaN on last bar


@dataclass
class GroundTruthReport:
    symbol:                         str
    n_bars_evaluated:               int
    n_high_confidence_predictions:  int
    directional_accuracy:           float        # gate: ≥ 0.52
    directional_accuracy_pass:      bool
    score_calibration_spearman:     float        # gate: ≥ 0.30
    calibration_pass:               bool
    per_bucket_returns:             Dict[str, float]
    per_signal_accuracy:            Dict[str, float]
    best_performing_signal:         str
    worst_performing_signal:        str
    false_positive_rate:            float
    false_negative_rate:            float
    overall_gate_pass:              bool
    recommendations:                List[str]
    bar_records:                    List[BarRecord] = field(default_factory=list)


# ── Evaluation engine ─────────────────────────────────────────────────────────

def _run_bar_by_bar(symbol: str, df: pd.DataFrame, refresh: bool = False) -> List[BarRecord]:
    """
    Main loop: bar-by-bar convergence scoring.
    Cached to disk unless refresh=True.
    """
    key = _cache_key(symbol, df)
    if not refresh:
        cached = _load_cache(key)
        if cached:
            print(f"  [cache HIT] {key}")
            return [BarRecord(**r) for r in cached]

    print(f"  [cache MISS] Computing {len(df)} bars for {symbol} …")
    records: List[BarRecord] = []

    # Instantiate detectors (stateful — reset each bar via slice)
    if not _ICT2_OK:
        print("  WARNING: ICT2 modules unavailable — scores will be 0")

    # Nucleus scorer needs universe data dict; we only have the single symbol here
    nucleus_scorer = None
    if _NUCLEUS_OK:
        try:
            nucleus_scorer = StandaloneNucleusScorer({symbol: df, "SPY": df})
        except Exception:
            pass

    engine = ICT2ConvergenceEngine() if _ICT2_OK else None

    n = len(df)
    for i in range(WARMUP_BARS, n - 1):
        slice_df = df.iloc[: i + 1]
        bar_date = str(df.index[i].date()) if isinstance(df.index, pd.DatetimeIndex) else str(i)
        current_price = float(df["close"].iloc[i])
        next_day_return = float(
            (df["close"].iloc[i + 1] - df["close"].iloc[i]) / df["close"].iloc[i]
        )

        # ── ICT2 signals ───────────────────────────────────────────────────────
        ict2_results: Dict[str, Any] = {"current_price": current_price}

        if _ICT2_OK:
            try:
                ict2_results["displacement"]     = DisplacementDetector().update(slice_df)
                ict2_results["nwog"]             = NWOGDetector().update(slice_df)
                ict2_results["turtle_soup"]      = TurtleSoupDetector().update(slice_df)
                ict2_results["propulsion_block"] = PropulsionBlockDetector().update(slice_df)
                ict2_results["bpr"]              = BPRDetector().update(slice_df)
                ict2_results["po3"]              = PowerOfThreeDetector("bullish").update(slice_df)
                ict2_results["silver_bullet"]    = SilverBulletDetector().update(slice_df)
                ict2_results["kill_zone"]        = KillZoneDetector("bullish").process(None, current_price)
            except Exception:
                pass

        # ── Nucleus multiplier ─────────────────────────────────────────────────
        nucleus_score_val = 0.70   # default
        if nucleus_scorer is not None:
            try:
                date_ts = df.index[i]
                scores = nucleus_scorer.score(date_ts, lookback=20)
                if scores:
                    nucleus_score_val = float(max(scores.values()))
            except Exception:
                pass
        ict2_results["nucleus_score"] = nucleus_score_val

        # ── Convergence ────────────────────────────────────────────────────────
        if engine is not None:
            try:
                conv: ConvergenceScore = engine.score(ict2_results)
            except Exception:
                continue
        else:
            continue

        records.append(BarRecord(
            bar_date           = bar_date,
            final_score        = conv.final_score,
            raw_score          = conv.raw_score,
            nucleus_multiplier = conv.nucleus_multiplier,
            direction          = conv.direction,
            confidence         = conv.confidence,
            top_signals        = conv.top_signals,
            signal_breakdown   = dict(conv.signal_breakdown),
            next_day_return    = next_day_return,
        ))

        if i % 50 == 0:
            print(f"    Bar {i}/{n}  {bar_date}  score={conv.final_score:+.3f}  dir={conv.direction}")

    # Cache results
    _save_cache(key, [asdict(r) for r in records])
    print(f"  Saved {len(records)} records → cache")
    return records


# ── Statistics ────────────────────────────────────────────────────────────────

def _spearman(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation."""
    if len(x) < 3:
        return 0.0
    from scipy.stats import spearmanr
    try:
        rho, _ = spearmanr(x, y)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        # Manual fallback
        n = len(x)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        d2 = np.sum((rx - ry) ** 2)
        return float(1 - 6 * d2 / (n * (n ** 2 - 1)))


def _bucket_label(score: float) -> str:
    lo = round(np.floor(score / BUCKET_WIDTH) * BUCKET_WIDTH, 1)
    hi = round(lo + BUCKET_WIDTH, 1)
    return f"[{lo:+.1f},{hi:+.1f})"


def build_report(symbol: str, records: List[BarRecord]) -> GroundTruthReport:
    """Compute all metrics and return GroundTruthReport."""
    # Filter valid records (non-NaN next_day_return)
    valid = [r for r in records if not np.isnan(r.next_day_return)]
    n = len(valid)

    # ── Directional accuracy on high-confidence bars ──────────────────────────
    high_conf = [r for r in valid if abs(r.final_score) >= HIGH_CONF_THRESHOLD]
    n_hc = len(high_conf)
    if n_hc > 0:
        correct_hc = sum(
            1 for r in high_conf
            if (r.final_score > 0 and r.next_day_return > 0) or
               (r.final_score < 0 and r.next_day_return < 0)
        )
        dir_acc = correct_hc / n_hc
    else:
        dir_acc = 0.0

    # ── Score calibration (bucket returns) ───────────────────────────────────
    bucket_returns: Dict[str, List[float]] = defaultdict(list)
    for r in valid:
        bucket_returns[_bucket_label(r.final_score)].append(r.next_day_return)

    per_bucket = {
        lbl: round(float(np.mean(rets)), 6)
        for lbl, rets in sorted(bucket_returns.items())
        if len(rets) >= 3
    }

    # Spearman: sorted bucket labels → mean returns
    bucket_scores = []
    bucket_means  = []
    for lbl, mean_ret in sorted(per_bucket.items()):
        lo = float(lbl[1:lbl.index(",")])
        bucket_scores.append(lo)
        bucket_means.append(mean_ret)

    spearman = _spearman(bucket_scores, bucket_means) if len(bucket_scores) >= 3 else 0.0

    # ── Per-signal group accuracy ─────────────────────────────────────────────
    group_correct: Dict[str, int] = defaultdict(int)
    group_total:   Dict[str, int] = defaultdict(int)

    for r in valid:
        if not r.signal_breakdown:
            continue
        # Top group = largest absolute contribution
        top_group = max(r.signal_breakdown, key=lambda k: abs(r.signal_breakdown[k]))
        correct = (r.final_score > 0 and r.next_day_return > 0) or \
                  (r.final_score < 0 and r.next_day_return < 0)
        group_correct[top_group] += int(correct)
        group_total[top_group]   += 1

    per_signal_acc: Dict[str, float] = {
        g: round(group_correct[g] / group_total[g], 4)
        for g in group_total if group_total[g] >= 5
    }

    best_sig  = max(per_signal_acc, key=per_signal_acc.get) if per_signal_acc else "N/A"
    worst_sig = min(per_signal_acc, key=per_signal_acc.get) if per_signal_acc else "N/A"

    # ── False positive / negative rates ──────────────────────────────────────
    strong_buy_bars  = [r for r in valid if r.direction == "STRONG_BUY"]
    strong_sell_bars = [r for r in valid if r.direction == "STRONG_SELL"]

    fp_rate = (
        sum(1 for r in strong_buy_bars if r.next_day_return < -0.01) / len(strong_buy_bars)
        if strong_buy_bars else 0.0
    )
    fn_rate = (
        sum(1 for r in strong_sell_bars if r.next_day_return > 0.01) / len(strong_sell_bars)
        if strong_sell_bars else 0.0
    )

    # ── Weight recommendations ────────────────────────────────────────────────
    cfg_path = _ROOT / "configs" / "convergence_weights.yaml"
    current_weights: Dict[str, float] = {}
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        current_weights = {k: float(v) for k, v in cfg.get("groups", {}).items()}

    recommendations: List[str] = []
    for group, acc in per_signal_acc.items():
        cur_w = current_weights.get(group, 0.10)
        if acc < 0.48:
            recommendations.append(
                f"Reduce weight of '{group}' in convergence_weights.yaml "
                f"(current accuracy: {acc:.1%}, below 0.48 threshold). "
                f"Suggested: {cur_w * 0.7:.3f} (30% reduction)."
            )
        elif acc > 0.58:
            recommendations.append(
                f"Increase weight of '{group}' in convergence_weights.yaml "
                f"(current accuracy: {acc:.1%}). "
                f"Suggested: {cur_w * 1.25:.3f} (25% increase)."
            )

    if not recommendations and dir_acc < GATE_DIR_ACCURACY:
        recommendations.append(
            f"Overall directional accuracy {dir_acc:.1%} is below the {GATE_DIR_ACCURACY:.0%} gate. "
            "Consider raising HIGH_CONF_THRESHOLD to 0.75 to restrict predictions to "
            "only the strongest bars, or add a kill-zone filter to the convergence engine."
        )

    overall_pass = (dir_acc >= GATE_DIR_ACCURACY) and (spearman >= GATE_SPEARMAN)

    return GroundTruthReport(
        symbol                        = symbol,
        n_bars_evaluated              = n,
        n_high_confidence_predictions = n_hc,
        directional_accuracy          = round(dir_acc, 4),
        directional_accuracy_pass     = dir_acc >= GATE_DIR_ACCURACY,
        score_calibration_spearman    = round(spearman, 4),
        calibration_pass              = spearman >= GATE_SPEARMAN,
        per_bucket_returns            = per_bucket,
        per_signal_accuracy           = per_signal_acc,
        best_performing_signal        = best_sig,
        worst_performing_signal       = worst_sig,
        false_positive_rate           = round(fp_rate, 4),
        false_negative_rate           = round(fn_rate, 4),
        overall_gate_pass             = overall_pass,
        recommendations               = recommendations,
        bar_records                   = valid,
    )


# ── Console output ────────────────────────────────────────────────────────────

def print_report(report: GroundTruthReport) -> None:
    w = 68
    print(f"\n{'='*w}")
    print(f"  CONVERGENCE GROUND TRUTH REPORT — {report.symbol}")
    print(f"  Bars evaluated: {report.n_bars_evaluated}  |  High-confidence: {report.n_high_confidence_predictions}")
    print(f"{'='*w}")

    da_icon = "✓" if report.directional_accuracy_pass else "✗"
    sp_icon = "✓" if report.calibration_pass else "✗"
    print(f"\n  GATE CHECKS")
    print(f"  {'─'*50}")
    print(f"  {da_icon}  Directional accuracy : {report.directional_accuracy:.1%}  (gate ≥ {GATE_DIR_ACCURACY:.0%})")
    print(f"  {sp_icon}  Spearman calibration : {report.score_calibration_spearman:+.3f}  (gate ≥ {GATE_SPEARMAN:.2f})")

    print(f"\n  SCORE BUCKET → MEAN NEXT-DAY RETURN")
    print(f"  {'─'*50}")
    for lbl, mean_ret in sorted(report.per_bucket_returns.items()):
        bar_char = "▲" if mean_ret > 0 else "▼"
        bar_len  = min(20, int(abs(mean_ret) * 1000))
        print(f"  {lbl:<15} {mean_ret:+.4f}  {bar_char * bar_len}")

    print(f"\n  PER-SIGNAL-GROUP DIRECTIONAL ACCURACY")
    print(f"  {'─'*50}")
    for g, acc in sorted(report.per_signal_accuracy.items(), key=lambda x: -x[1]):
        icon = "✓" if acc >= 0.52 else "✗"
        print(f"  {icon}  {g:<15} {acc:.1%}")

    print(f"\n  ERROR RATES")
    print(f"  {'─'*50}")
    print(f"  False positive rate (STRONG_BUY → next day < -1%) : {report.false_positive_rate:.1%}")
    print(f"  False negative rate (STRONG_SELL → next day > +1%) : {report.false_negative_rate:.1%}")

    overall_icon = "✅" if report.overall_gate_pass else "❌"
    print(f"\n  {overall_icon}  OVERALL GATE: {'PASS' if report.overall_gate_pass else 'FAIL'}")

    if report.recommendations:
        print(f"\n  WEIGHT ADJUSTMENT RECOMMENDATIONS")
        print(f"  {'─'*50}")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  [{i}] {rec}")
    print(f"{'='*w}\n")


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(report: GroundTruthReport) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(report)
    # Don't save full bar_records to YAML (too large); save separately
    bar_records = d.pop("bar_records", [])
    with open(REPORT_PATH, "w") as fh:
        yaml.safe_dump(d, fh, default_flow_style=False, sort_keys=False)

    # Save bar records as CSV
    csv_path = REPORT_PATH.with_suffix(".csv")
    if bar_records:
        rows = []
        for r in bar_records:
            row = {
                "date":             r["bar_date"],
                "final_score":      r["final_score"],
                "direction":        r["direction"],
                "nucleus_mult":     r["nucleus_multiplier"],
                "confidence":       r["confidence"],
                "next_day_return":  r["next_day_return"],
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"✅ Report saved → {REPORT_PATH}")
    print(f"   Bar-level CSV → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ICT2 Convergence Ground Truth")
    parser.add_argument("--symbol",  default="SPY", help="Symbol to evaluate (default: SPY)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and recompute")
    args = parser.parse_args()

    print(f"\n{'='*68}")
    print(f"  ICT2 Convergence Ground Truth — {args.symbol}")
    print(f"{'='*68}")

    print("\nLoading bars …")
    df = _load_bars(args.symbol)
    df = _ensure_datetime_index(df)
    print(f"  {len(df)} bars loaded  ({df.index[0].date()} → {df.index[-1].date()})")

    print("\nRunning bar-by-bar evaluation …")
    records = _run_bar_by_bar(args.symbol, df, refresh=args.refresh)
    print(f"  {len(records)} bar records computed")

    print("\nBuilding report …")
    report = build_report(args.symbol, records)

    print_report(report)
    save_report(report)

    sys.exit(0 if report.overall_gate_pass else 1)


if __name__ == "__main__":
    main()
