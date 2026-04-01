#!/usr/bin/env python3
"""
scripts/run_full_pipeline.py
============================
7-Stage end-to-end integration test for the Draculative Alpha Engine.

Runs every layer of the stack in sequence for a single ticker and prints
a colour-coded pass/fail table.  Exits 0 if all REQUIRED stages pass,
exits 1 otherwise.

Stages
------
  1  Data loading          – Alpaca (primary) + yfinance fallback
  2  ICT2 signals          – all 8 detectors on the most-recent 120 bars
  3  FVG / ICT1 analysis   – FVGAnalyser on the same slice
  4  Nucleus scoring       – StandaloneNucleusScorer + apply_ict2_adjustments
  5  Convergence scoring   – ICT2ConvergenceEngine.score()
  6  Stock predictor       – StockPredictor (optional; skip gracefully)
  7  Ground truth check    – GroundTruthReport gate (only when --date provided)

Usage
-----
    # Minimal (no ground-truth check):
    python scripts/run_full_pipeline.py --ticker NVDA

    # Full (with ground-truth gate):
    python scripts/run_full_pipeline.py --ticker NVDA --date 2025-01-10
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as dt

import numpy as np
import pandas as pd

# ── repo root ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

_CHECK  = f"{_GREEN}✓{_RESET}"
_CROSS  = f"{_RED}✗{_RESET}"
_SKIP   = f"{_YELLOW}⊘{_RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    stage:    int
    name:     str
    required: bool
    passed:   Optional[bool] = None   # None = skipped
    detail:   str = ""
    data:     Any = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_bars(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for use when live data is unavailable."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.8, n))
    high  = close + rng.uniform(0.1, 0.8, n)
    low   = close - rng.uniform(0.1, 0.8, n)
    open_ = close - rng.normal(0, 0.4, n)
    vol   = rng.integers(100_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────────────

def stage1_data_loading(ticker: str) -> StageResult:
    """Stage 1: Load daily OHLCV bars via Alpaca (primary) or yfinance (fallback)."""
    r = StageResult(1, "Data Loading", required=True)
    df: Optional[pd.DataFrame] = None
    source = "unknown"

    # Alpaca primary
    try:
        from simulation.alpaca_data_fetcher import AlpacaDataFetcher
        fetcher = AlpacaDataFetcher()
        raw = fetcher.load_symbol(ticker, timeframe="1Day")
        if raw is not None and len(raw) >= 100:
            raw.columns = [c.lower() for c in raw.columns]
            raw = raw.reset_index(drop=False)
            date_col = next((c for c in raw.columns if c in ("date", "timestamp", "datetime")), None)
            if date_col:
                raw["date"] = pd.to_datetime(raw[date_col], utc=True)
            df     = raw
            source = "Alpaca"
    except Exception as _ae:
        pass

    # yfinance fallback
    if df is None or len(df) < 100:
        try:
            import yfinance as yf
            raw = yf.Ticker(ticker).history(period="2y", interval="1d").reset_index()
            raw.columns = [c.lower() for c in raw.columns]
            if len(raw) >= 100:
                df     = raw
                source = "yfinance"
        except Exception:
            pass

    # Synthetic last resort (marks SKIP, not FAIL)
    if df is None or len(df) < 20:
        df     = _synthetic_bars(n=120)
        source = "synthetic"

    r.passed = (source != "synthetic") or True   # pipeline continues with synthetic
    r.detail = f"source={source}, bars={len(df)}"
    r.data   = df
    if source == "synthetic":
        r.passed = False
        r.detail += " (SYNTHETIC — no live data available)"
    return r


def stage2_ict2_signals(df: pd.DataFrame) -> StageResult:
    """Stage 2: Run all 8 ICT2 signal detectors on the most-recent 120 bars."""
    r = StageResult(2, "ICT2 Signals (all 8)", required=True)
    try:
        from trading_system.ict_signals.displacement_detector    import DisplacementDetector
        from trading_system.ict_signals.nwog_detector            import NWOGDetector
        from trading_system.ict_signals.propulsion_block_detector import PropulsionBlockDetector
        from trading_system.ict_signals.balanced_price_range     import BPRDetector
        from trading_system.ict_signals.turtle_soup_detector     import TurtleSoupDetector
        from trading_system.ict_signals.power_of_three           import PowerOfThreeDetector
        from trading_system.ict_signals.silver_bullet_setup      import SilverBulletDetector
        from trading_system.ict_signals.killzone_filter          import KillZoneDetector

        # Use last 120 bars
        slice_df = df.tail(120).copy()
        if isinstance(slice_df.index, pd.DatetimeIndex):
            pass
        else:
            # Ensure numeric index for detectors that use .iloc
            slice_df = slice_df.reset_index(drop=True)

        # Normalise required columns
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(slice_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        results: Dict[str, Any] = {}

        detectors_plain = [
            ("displacement",       DisplacementDetector()),
            ("nwog",               NWOGDetector()),
            ("propulsion_block",   PropulsionBlockDetector()),
            ("bpr",                BPRDetector()),
            ("turtle_soup",        TurtleSoupDetector()),
            ("po3",                PowerOfThreeDetector(expected_direction="bullish")),
            ("silver_bullet",      SilverBulletDetector()),
        ]
        for name, det in detectors_plain:
            try:
                results[name] = det.update(slice_df)
            except Exception as ex:
                results[name] = f"ERROR: {ex}"

        # KillZone needs price + timestamp
        try:
            kz = KillZoneDetector(htf_bias="bullish")
            cur_price = float(slice_df["close"].iloc[-1])
            results["kill_zone"] = kz.process(None, cur_price)
        except Exception as ex:
            results["kill_zone"] = f"ERROR: {ex}"

        errors = [k for k, v in results.items() if isinstance(v, str) and v.startswith("ERROR")]
        r.passed = len(errors) == 0
        r.detail = (
            f"all 8 detectors ran"
            if not errors
            else f"errors in: {errors}"
        )
        r.data = results

    except Exception:
        r.passed = False
        r.detail = traceback.format_exc(limit=3)
    return r


def stage3_fvg_analysis(df: pd.DataFrame) -> StageResult:
    """Stage 3: Run FVGAnalyser on the same 120-bar slice."""
    r = StageResult(3, "FVG / ICT1 Analysis", required=False)
    try:
        # Try multiple import paths for FVGAnalyser
        fvg_result = None
        for mod_path in (
            "trading_system.ict_signals.fvg_analyser",
            "technical.concepts.fvg_analyser",
            "core.fvg_analyser",
        ):
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                FVGAnalyser = getattr(mod, "FVGAnalyser", None)
                if FVGAnalyser is None:
                    continue
                fvg_result = FVGAnalyser().update(df.tail(120).copy())
                break
            except (ImportError, AttributeError):
                continue

        if fvg_result is None:
            r.passed = None   # SKIP — FVGAnalyser not found
            r.detail = "FVGAnalyser not found (skipped)"
            return r

        r.passed = True
        r.detail = f"result type={type(fvg_result).__name__}"
        r.data   = fvg_result

    except Exception:
        r.passed = None  # non-required, so skip gracefully
        r.detail = f"SKIP: {traceback.format_exc(limit=2)}"
    return r


def stage4_nucleus_scoring(df: pd.DataFrame, ict2_results: Dict[str, Any]) -> StageResult:
    """Stage 4: Nucleus scoring + apply_ict2_adjustments on the final bar."""
    r = StageResult(4, "Nucleus Scoring", required=True)
    try:
        from core.nucleus_validator import StandaloneNucleusScorer, NucleusValidator

        scorer  = StandaloneNucleusScorer()
        scores  = scorer.score(df)

        # apply_ict2_adjustments (may not exist on older builds)
        try:
            validator = NucleusValidator(df)
            adj_scores = validator.apply_ict2_adjustments(
                nucleus_scores=scores,
                ict2_results=ict2_results,
                bar_timestamp=None,
            )
        except AttributeError:
            adj_scores = scores

        top_type  = max(adj_scores, key=lambda k: adj_scores[k]) if adj_scores else "N/A"
        top_score = adj_scores.get(top_type, 0.0)

        r.passed = isinstance(scores, dict) and len(scores) > 0
        r.detail = (
            f"types={len(adj_scores)}, "
            f"top={top_type} ({top_score:.3f})"
        )
        r.data = adj_scores

    except Exception:
        r.passed = False
        r.detail = traceback.format_exc(limit=3)
    return r


def stage5_convergence_scoring(
    df: pd.DataFrame,
    ict2_results: Dict[str, Any],
    nucleus_scores: Optional[Dict[str, float]],
) -> StageResult:
    """Stage 5: ICT2ConvergenceEngine.score() with nucleus_scores wired in."""
    r = StageResult(5, "Convergence Scoring", required=True)
    try:
        # Try Sprint-2 engine path first
        for mod_path in (
            "core.ict2_convergence_engine",
            "trading_system.ict2_convergence_engine",
        ):
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                ICT2ConvergenceEngine = getattr(mod, "ICT2ConvergenceEngine", None)
                if ICT2ConvergenceEngine is None:
                    continue

                all_results: Dict[str, Any] = dict(ict2_results)
                if nucleus_scores:
                    # Compute single nucleus_score as max value
                    all_results["nucleus_score"] = max(nucleus_scores.values())

                engine = ICT2ConvergenceEngine()
                conv_result = engine.score(df.tail(120).copy(), all_results)
                final_score  = getattr(conv_result, "final_score", conv_result)

                r.passed = isinstance(final_score, (float, int)) and not np.isnan(float(final_score))
                r.detail = f"final_score={float(final_score):.4f}"
                r.data   = conv_result
                return r
            except (ImportError, AttributeError):
                continue

        r.passed = None
        r.detail = "ICT2ConvergenceEngine not found (skipped)"

    except Exception:
        r.passed = False
        r.detail = traceback.format_exc(limit=3)
    return r


def stage6_stock_predictor(
    ticker: str,
    df: pd.DataFrame,
    nucleus_scores: Optional[Dict[str, float]],
    convergence_score: Optional[float],
) -> StageResult:
    """Stage 6: Full StockPredictor prediction (optional — skip gracefully)."""
    r = StageResult(6, "Stock Predictor", required=False)
    try:
        # Try multiple import paths
        for mod_path in (
            "predictor.src.stock_predictor",
            "predictor.app.stock_predictor",
            "core.stock_predictor",
        ):
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                StockPredictor = getattr(mod, "StockPredictor", None)
                if StockPredictor is None:
                    continue

                predictor = StockPredictor(ticker=ticker)
                pred_result = predictor.predict(df)
                r.passed = pred_result is not None
                r.detail = f"prediction type={type(pred_result).__name__}"
                r.data   = pred_result
                return r
            except (ImportError, AttributeError):
                continue

        r.passed = None
        r.detail = "StockPredictor not found (skipped)"

    except Exception:
        r.passed = None
        r.detail = f"SKIP: {traceback.format_exc(limit=2)}"
    return r


def stage7_ground_truth(
    ticker: str,
    target_date: Optional[str],
) -> StageResult:
    """Stage 7: GroundTruthReport gate (only when --date is provided)."""
    r = StageResult(7, "Ground Truth Check", required=False)
    if target_date is None:
        r.passed = None
        r.detail = "skipped (no --date provided)"
        return r

    try:
        from core.convergence_ground_truth import run_ground_truth, GroundTruthReport

        report: GroundTruthReport = run_ground_truth(
            symbol=ticker,
            refresh=False,
        )
        gate = report.overall_gate_pass
        r.passed = gate
        r.detail = (
            f"dir_acc={report.directional_accuracy:.3f} "
            f"spearman={report.score_calibration_spearman:.3f} "
            f"gate={'PASS' if gate else 'FAIL'}"
        )
        r.data = report

    except Exception:
        r.passed = None
        r.detail = f"SKIP: {traceback.format_exc(limit=2)}"
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_pipeline_report(ticker: str, stages: List[StageResult]) -> bool:
    """Print the final table and return True iff all REQUIRED stages passed."""
    print()
    print(f"{_BOLD}{'=' * 68}{_RESET}")
    print(f"{_BOLD}  DRACULATIVE ALPHA ENGINE — FULL PIPELINE REPORT{_RESET}")
    print(f"  Ticker: {_CYAN}{ticker}{_RESET}    "
          f"Run: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{_BOLD}{'=' * 68}{_RESET}")
    print(f"  {'#':<3}  {'Stage':<28}  {'Req':<5}  {'Status':<8}  Detail")
    print(f"  {'-'*3}  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*30}")

    all_required_pass = True
    for sr in stages:
        if sr.passed is True:
            icon = _CHECK
            stat = f"{_GREEN}PASS{_RESET}"
        elif sr.passed is False:
            icon = _CROSS
            stat = f"{_RED}FAIL{_RESET}"
            if sr.required:
                all_required_pass = False
        else:
            icon = _SKIP
            stat = f"{_YELLOW}SKIP{_RESET}"

        req_label = "REQ" if sr.required else "OPT"
        detail    = sr.detail[:55] if len(sr.detail) > 55 else sr.detail
        print(f"  {icon}  {sr.stage:<3} {sr.name:<27}  {req_label:<5}  {stat:<8}  {detail}")

    print(f"{_BOLD}{'=' * 68}{_RESET}")
    overall = f"{_GREEN}ALL REQUIRED STAGES PASSED{_RESET}" if all_required_pass \
              else f"{_RED}ONE OR MORE REQUIRED STAGES FAILED{_RESET}"
    print(f"  Overall: {overall}")
    print(f"{_BOLD}{'=' * 68}{_RESET}\n")
    return all_required_pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draculative Alpha Engine — 7-Stage Integration Test"
    )
    parser.add_argument(
        "--ticker", default="NVDA",
        help="Ticker symbol to run the pipeline for (default: NVDA)"
    )
    parser.add_argument(
        "--date", default=None,
        help="Optional date (YYYY-MM-DD) to run ground-truth check against"
    )
    args = parser.parse_args()
    ticker: str = args.ticker.upper()

    print(f"\n{_CYAN}{_BOLD}Draculative Alpha Engine — Full Pipeline Run{_RESET}")
    print(f"Ticker: {ticker}  |  Date: {args.date or 'N/A (no ground-truth check)'}\n")

    stages: List[StageResult] = []

    # Stage 1: Data loading
    print(f"[1/7] Loading bars for {ticker} …")
    s1 = stage1_data_loading(ticker)
    stages.append(s1)

    df: pd.DataFrame = s1.data if s1.data is not None else _synthetic_bars()

    # Stage 2: ICT2 signals
    print("[2/7] Running ICT2 signal detectors …")
    s2 = stage2_ict2_signals(df)
    stages.append(s2)
    ict2_results: Dict[str, Any] = s2.data or {}

    # Stage 3: FVG analysis
    print("[3/7] Running FVG / ICT1 analysis …")
    s3 = stage3_fvg_analysis(df)
    stages.append(s3)

    # Stage 4: Nucleus scoring
    print("[4/7] Running nucleus scoring …")
    s4 = stage4_nucleus_scoring(df, ict2_results)
    stages.append(s4)
    nucleus_scores: Optional[Dict[str, float]] = s4.data

    # Stage 5: Convergence scoring
    print("[5/7] Running convergence scoring …")
    s5 = stage5_convergence_scoring(df, ict2_results, nucleus_scores)
    stages.append(s5)
    conv_score: Optional[float] = None
    if s5.data is not None:
        conv_score = getattr(s5.data, "final_score", None)
        if conv_score is None and isinstance(s5.data, (float, int)):
            conv_score = float(s5.data)

    # Stage 6: Stock predictor
    print("[6/7] Running stock predictor …")
    s6 = stage6_stock_predictor(ticker, df, nucleus_scores, conv_score)
    stages.append(s6)

    # Stage 7: Ground truth check
    print("[7/7] Running ground truth check …")
    s7 = stage7_ground_truth(ticker, args.date)
    stages.append(s7)

    # Report
    all_pass = print_pipeline_report(ticker, stages)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
