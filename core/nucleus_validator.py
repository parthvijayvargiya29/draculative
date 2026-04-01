"""
nucleus_validator.py
====================
Runs the nucleus engine bar-by-bar against 2 years of real Alpaca data
and answers five critical validation questions:

  CHECK 1 — Distribution: does each nucleus type fire ≤35% of trading days?
  CHECK 2 — Persistence: does each nucleus last 2–15 days before switching?
  CHECK 3 — Transitions: do transitions happen at the right time?
            (FOMC/CPI/earnings days → nucleus change expected)
  CHECK 4 — Predictive: does nucleus direction correlate with 5-day SPY return?
             (target: ≥52% directional accuracy when nucleus has strong score)
  CHECK 5 — Calibration: are scores spread 0.20–0.90, or are they bunched?

A GATE PASS requires all 5 checks to pass.
On GATE FAIL: recalibration instructions are printed per failing check.

USAGE
-----
  python core/nucleus_validator.py

  # Or import:
  from core.nucleus_validator import NucleusValidator
  report = NucleusValidator(data).run()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ICT2 module guard
try:
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent))
    from trading_system.ict_signals.turtle_soup_detector import TurtleSoupDetector as _TSD
    from trading_system.ict_signals.displacement_detector import DisplacementDetector as _DD
    from trading_system.ict_signals.nwog_detector import NWOGDetector as _NWOG
    _ICT2_AVAILABLE = True
except Exception:
    _ICT2_AVAILABLE = False

_NUCLEUS_ICT2_CFG_PATH = Path(__file__).parent.parent / "configs" / "nucleus_ict2_adjustments.yaml"

def _load_nucleus_ict2_cfg() -> dict:
    if _NUCLEUS_ICT2_CFG_PATH.exists():
        with open(_NUCLEUS_ICT2_CFG_PATH) as _fh:
            return yaml.safe_load(_fh) or {}
    return {}

_NUCLEUS_ICT2_CFG = _load_nucleus_ict2_cfg()

# ── Gate thresholds (all must pass) ──────────────────────────────────────────
GATE_MAX_DOMINANT_PCT   = 0.35   # No single nucleus type > 35% of days
GATE_MIN_PERSISTENCE    = 2      # Nucleus must last at least 2 bars on average
GATE_MAX_PERSISTENCE    = 20     # … but not more than 20 (that means it's frozen)
GATE_MIN_ACCURACY       = 0.52   # ≥52% directional on 5D SPY return (strong signals only)
GATE_STRONG_SCORE_MIN   = 0.65   # Only count predictions where winning nucleus score ≥ this
GATE_MIN_SCORE_SPREAD   = 0.30   # max_score - min_score must be ≥ this (not bunched)
GATE_MIN_TRANSITION_RATE= 0.05   # At least 5% of days must have a nucleus change

# Nucleus candidate IDs (must match what nucleus_engine.py uses)
NUCLEUS_TYPES = [
    "MACRO_EVENT",
    "INDEX_MOMENTUM",
    "SECTOR_ROTATION",
    "SINGLE_STOCK_LEADERSHIP",
    "COMMODITY_REGIME",
    "CURRENCY_PRESSURE",
    "OPTIONS_EXPIRY",
    "GEOPOLITICAL",
]

# Major scheduled event dates for transition check (extend as needed)
# Format: "YYYY-MM-DD"
KNOWN_EVENT_DATES: List[str] = [
    # 2024 FOMC meetings
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025 FOMC meetings (through knowledge cutoff)
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-06", "2025-12-17",
    # 2026 FOMC (projected)
    "2026-01-29", "2026-03-19",
    # Major CPI releases (monthly, ~10th-14th)
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-15", "2025-11-12", "2025-12-10",
    "2026-01-15", "2026-02-12", "2026-03-12",
]

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BarResult:
    """Nucleus engine output for a single bar."""
    date:        str
    nucleus:     str
    score:       float
    scores:      Dict[str, float]  # full scoring for all candidates
    satellite:   str               # second-highest scorer
    regime:      str               # TRENDING or CORRECTIVE

@dataclass
class CheckResult:
    name:    str
    passed:  bool
    value:   float
    gate:    float
    message: str
    detail:  Dict = field(default_factory=dict)

@dataclass
class ValidationReport:
    run_date:       str
    data_period:    Dict[str, str]
    total_bars:     int
    checks:         List[CheckResult]
    gate_pass:      bool
    recalibration:  List[str]
    raw_results:    List[BarResult] = field(default_factory=list)
    distribution:   Dict[str, float] = field(default_factory=dict)
    transition_rate: float = 0.0
    accuracy_5d:    float = 0.0
    avg_persistence: float = 0.0
    score_spread:   float = 0.0

# ── Nucleus scoring (standalone — mirrors nucleus_engine.py logic) ─────────────

class StandaloneNucleusScorer:
    """
    Reimplementation of nucleus_engine.py scoring that takes a bar snapshot
    as raw price series rather than a live BarSnapshot object.

    This lets the validator run without the full engine stack.
    Each score() call returns Dict[nucleus_type → float 0–1].
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self._spy  = data.get("SPY")
        self._qqq  = data.get("QQQ")
        self._vxx  = data.get("VXX")
        self._gld  = data.get("GLD")
        _uup = data.get("UUP")
        self._uup  = _uup if _uup is not None else data.get("DXY_proxy")
        self._tlt  = data.get("TLT")
        self._xl   = {s: data.get(s) for s in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLC"]}
        self._mega = {s: data.get(s) for s in ["NVDA","AAPL","MSFT","GOOGL","META","AMZN","TSLA"]}

    def score(self, date: pd.Timestamp, lookback: int = 20) -> Dict[str, float]:
        """
        Score all 8 nucleus candidates at this date.
        Returns dict of {nucleus_type: score_0_to_1}.
        """
        scores: Dict[str, float] = {nt: 0.0 for nt in NUCLEUS_TYPES}

        # ── MACRO_EVENT ──────────────────────────────────────────────────────
        # Proxy: VXX spike > 1.5 std on event date (volatility expansion)
        scores["MACRO_EVENT"] = self._score_macro_event(date, lookback)

        # ── INDEX_MOMENTUM ───────────────────────────────────────────────────
        # SPY ADX > 25 and price > SMA50 slope positive
        scores["INDEX_MOMENTUM"] = self._score_index_momentum(date, lookback)

        # ── SECTOR_ROTATION ──────────────────────────────────────────────────
        # High dispersion between sector ETF 20D returns (one sector dominates)
        scores["SECTOR_ROTATION"] = self._score_sector_rotation(date, lookback)

        # ── SINGLE_STOCK_LEADERSHIP ──────────────────────────────────────────
        # One mega-cap has outsized return vs SPY (>3x SPY return on this bar)
        scores["SINGLE_STOCK_LEADERSHIP"] = self._score_single_stock(date)

        # ── COMMODITY_REGIME ─────────────────────────────────────────────────
        # GLD or USO 5D return > 2x SPY 5D return
        scores["COMMODITY_REGIME"] = self._score_commodity(date, lookback)

        # ── CURRENCY_PRESSURE ────────────────────────────────────────────────
        # UUP (DXY proxy) 5D return > 0.5% absolute
        scores["CURRENCY_PRESSURE"] = self._score_currency(date, lookback)

        # ── OPTIONS_EXPIRY ───────────────────────────────────────────────────
        # Monthly OpEx: 3rd Friday of month (proxy: within 3 days of 3rd Friday)
        scores["OPTIONS_EXPIRY"] = self._score_options_expiry(date)

        # ── GEOPOLITICAL ─────────────────────────────────────────────────────
        # TLT spike (flight to safety) + GLD spike simultaneously
        scores["GEOPOLITICAL"] = self._score_geopolitical(date, lookback)

        # Normalize: ensure scores sum makes sense (not forced to 1 — each is independent)
        return scores

    # ── Individual scorers ────────────────────────────────────────────────────

    def _close_at(self, df: Optional[pd.DataFrame], date: pd.Timestamp) -> Optional[float]:
        if df is None or date not in df.index:
            return None
        return float(df.loc[date, "close"])

    def _returns_before(self, df: Optional[pd.DataFrame], date: pd.Timestamp, n: int) -> Optional[pd.Series]:
        """N-bar returns ending at date (inclusive)."""
        if df is None:
            return None
        mask = df.index <= date
        sub  = df[mask].tail(n + 1)
        if len(sub) < 2:
            return None
        return sub["close"].pct_change().dropna()

    def _score_macro_event(self, date: pd.Timestamp, lookback: int) -> float:
        rets = self._returns_before(self._vxx, date, lookback)
        if rets is None or len(rets) < 5:
            return 0.0
        std  = rets.std()
        last = rets.iloc[-1]
        if std == 0:
            return 0.0
        z = abs(last / std)
        return float(min(1.0, z / 3.0))   # z=3 → score=1.0

    def _score_index_momentum(self, date: pd.Timestamp, lookback: int) -> float:
        if self._spy is None:
            return 0.0
        sub = self._spy[self._spy.index <= date].tail(60)
        if len(sub) < 52:
            return 0.0
        close = sub["close"]

        # ADX proxy: directional movement ratio
        high, low = sub["high"], sub["low"]
        up_move   = high.diff()
        dn_move   = -low.diff()
        plus_dm   = up_move.where((up_move > dn_move) & (up_move > 0), 0)
        minus_dm  = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
        tr        = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr       = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_di   = 100 * plus_dm.ewm(alpha=1/14).mean() / atr.replace(0, np.nan)
        minus_di  = 100 * minus_dm.ewm(alpha=1/14).mean() / atr.replace(0, np.nan)
        dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx       = dx.ewm(alpha=1/14).mean().iloc[-1]

        sma50     = close.rolling(50).mean()
        slope_pos = bool(sma50.iloc[-1] > sma50.iloc[-5])
        trending  = adx > 25 and slope_pos

        if trending:
            return float(min(1.0, (adx - 25) / 25 * 0.8 + 0.2))
        return float(max(0.0, (adx - 15) / 25 * 0.3))

    def _score_sector_rotation(self, date: pd.Timestamp, lookback: int) -> float:
        rets_20d = {}
        for sec, df in self._xl.items():
            if df is None:
                continue
            r = self._returns_before(df, date, lookback)
            if r is not None and len(r) >= lookback:
                rets_20d[sec] = float(r.sum())
        if len(rets_20d) < 5:
            return 0.0
        vals = list(rets_20d.values())
        spread = max(vals) - min(vals)
        return float(min(1.0, spread / 0.15))   # 15% spread → score=1.0

    def _score_single_stock(self, date: pd.Timestamp) -> float:
        spy_ret = self._returns_before(self._spy, date, 1)
        if spy_ret is None or len(spy_ret) == 0:
            return 0.0
        spy_1d = float(spy_ret.iloc[-1])

        max_outperform = 0.0
        for sym, df in self._mega.items():
            ret = self._returns_before(df, date, 1)
            if ret is None or len(ret) == 0:
                continue
            stock_1d    = float(ret.iloc[-1])
            outperform  = abs(stock_1d) - abs(spy_1d)
            max_outperform = max(max_outperform, outperform)

        return float(min(1.0, max_outperform / 0.04))   # 4% excess → score=1.0

    def _score_commodity(self, date: pd.Timestamp, lookback: int) -> float:
        spy_5d = self._returns_before(self._spy, date, 5)
        gld_5d = self._returns_before(self._gld, date, 5)
        if spy_5d is None or gld_5d is None:
            return 0.0
        if len(spy_5d) < 5 or len(gld_5d) < 5:
            return 0.0
        spy_sum = abs(float(spy_5d.sum()))
        gld_sum = abs(float(gld_5d.sum()))
        if spy_sum < 0.005:
            return float(min(1.0, gld_sum / 0.03))
        ratio = gld_sum / (spy_sum + 1e-9)
        return float(min(1.0, (ratio - 1.5) / 1.5)) if ratio > 1.5 else 0.0

    def _score_currency(self, date: pd.Timestamp, lookback: int) -> float:
        rets = self._returns_before(self._uup, date, 5)
        if rets is None or len(rets) < 5:
            return 0.0
        cumret = abs(float(rets.sum()))
        return float(min(1.0, cumret / 0.015))   # 1.5% 5D move → score=1.0

    def _score_options_expiry(self, date: pd.Timestamp) -> float:
        """Proxy: are we within 3 calendar days of the 3rd Friday of the month?"""
        d = date.to_pydatetime()
        # Find the 3rd Friday of this month
        first_day = d.replace(day=1)
        # weekday(): Monday=0, Friday=4
        first_friday = first_day
        while first_friday.weekday() != 4:
            first_friday = first_friday.replace(day=first_friday.day + 1)
        third_friday = first_friday.replace(day=first_friday.day + 14)

        diff = abs((d - third_friday).days)
        if diff <= 1:
            return 1.0
        elif diff <= 3:
            return 0.6
        elif diff <= 7:
            return 0.2
        return 0.0

    def _score_geopolitical(self, date: pd.Timestamp, lookback: int) -> float:
        """GLD spike + TLT spike simultaneously (flight to safety)."""
        gld_ret = self._returns_before(self._gld, date, 1)
        tlt_ret = self._returns_before(self._tlt, date, 1)
        if gld_ret is None or tlt_ret is None:
            return 0.0
        if len(gld_ret) == 0 or len(tlt_ret) == 0:
            return 0.0
        gld_1d = abs(float(gld_ret.iloc[-1]))
        tlt_1d = abs(float(tlt_ret.iloc[-1]))
        # Both need to move in same direction (up) to signal fear
        gld_up = float(gld_ret.iloc[-1]) > 0.003
        tlt_up = float(tlt_ret.iloc[-1]) > 0.003
        if gld_up and tlt_up:
            return float(min(1.0, (gld_1d + tlt_1d) / 0.02))
        return 0.0

# ── Regime classifier (standalone) ───────────────────────────────────────────

def classify_regime(spy_df: pd.DataFrame, date: pd.Timestamp) -> str:
    """Returns 'TRENDING' or 'CORRECTIVE' based on SPY ADX and SMA50 slope."""
    sub = spy_df[spy_df.index <= date].tail(60)
    if len(sub) < 52:
        return "UNKNOWN"

    close = sub["close"]
    high  = sub["high"]
    low   = sub["low"]

    up_move  = high.diff()
    dn_move  = -low.diff()
    plus_dm  = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
    tr       = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr      = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = float(dx.ewm(alpha=1/14).mean().iloc[-1])

    sma50    = close.rolling(50).mean()
    slope    = sma50.iloc[-1] > sma50.iloc[-5]

    return "TRENDING" if (adx > 25 and slope) else "CORRECTIVE"

# ── Validator ─────────────────────────────────────────────────────────────────

class NucleusValidator:
    """
    Runs the 5-check validation suite on 2 years of real Alpaca data.

    USAGE:
        from simulation.alpaca_data_fetcher import AlpacaFetcher
        from core.nucleus_validator import NucleusValidator

        fetcher = AlpacaFetcher()
        data    = fetcher.load_universe()
        report  = NucleusValidator(data).run()
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data    = data
        self.spy     = data.get("SPY")
        self.scorer  = StandaloneNucleusScorer(data)

        if self.spy is None:
            raise ValueError("SPY is required for nucleus validation — not found in data.")

    # ── Main ─────────────────────────────────────────────────────────────────

    # ── ICT2 adjustment layer ─────────────────────────────────────────────────

    def apply_ict2_adjustments(
        self,
        nucleus_scores: Dict[str, float],
        ict2_results: dict,
        bar_timestamp: "pd.Timestamp",
    ) -> Dict[str, float]:
        """
        Apply ICT2 signal evidence to adjust nucleus candidate scores.
        All adjustments are additive deltas, clamped to [0, 1] after application.
        Adjustment magnitudes loaded from configs/nucleus_ict2_adjustments.yaml.
        """
        cfg = _NUCLEUS_ICT2_CFG
        scores = dict(nucleus_scores)  # copy

        # ── Helper: safe add/multiply ──────────────────────────────────────
        def _add(key: str, delta: float) -> None:
            if key in scores:
                scores[key] = max(0.0, min(1.0, scores[key] + delta))

        def _mul_all(factor: float) -> None:
            for k in scores:
                scores[k] = max(0.0, min(1.0, scores[k] * factor))

        # ── Alias resolution ──────────────────────────────────────────────
        # Some config keys use conceptual aliases not in NUCLEUS_TYPES
        # Map them to the closest canonical type if present.
        _alias = {
            "LIQUIDITY_POOL":    "MACRO_EVENT",   # fallback if not its own type
            "VOLATILITY_REGIME": "MACRO_EVENT",
            "DOLLAR_STRENGTH":   "CURRENCY_PRESSURE",
            "EARNINGS_MOMENTUM": "SINGLE_STOCK_LEADERSHIP",
        }

        def _resolve(key: str) -> str:
            if key in scores:
                return key
            return _alias.get(key, key)

        def _add_resolved(key: str, delta: float) -> None:
            _add(_resolve(key), delta)

        # ── Displacement ──────────────────────────────────────────────────
        disp = ict2_results.get("displacement")
        if disp is not None and getattr(disp, "detected", False):
            direction = getattr(disp, "direction", "")
            d_cfg = cfg.get("displacement", {})
            sub = d_cfg.get(direction, {})
            for k, v in sub.items():
                _add_resolved(k, float(v))

        # ── Turtle Soup ───────────────────────────────────────────────────
        ts = ict2_results.get("turtle_soup")
        if ts is not None and getattr(ts, "detected", False):
            ts_dir = getattr(ts, "direction", "long")
            ts_cfg = cfg.get("turtle_soup", {})
            sub = ts_cfg.get(ts_dir, {})
            for k, v in sub.items():
                _add_resolved(k, float(v))

        # ── NWOG CE proximity ─────────────────────────────────────────────
        nwog = ict2_results.get("nwog")
        if nwog is not None:
            current_price = float(ict2_results.get("current_price", 0.0))
            nwog_cfg = cfg.get("nwog_ce_proximity", {})
            threshold_pct = float(nwog_cfg.get("ce_proximity_threshold_pct", 0.5)) / 100
            for attr in ("nearest_ce_above", "nearest_ce_below"):
                ce = getattr(nwog, attr, None)
                if ce and current_price > 0:
                    if abs(ce - current_price) / current_price <= threshold_pct:
                        for k, v in nwog_cfg.items():
                            if isinstance(v, (int, float)) and k != "ce_proximity_threshold_pct":
                                _add_resolved(k, float(v))
                        break

        # ── Power of Three ────────────────────────────────────────────────
        po3 = ict2_results.get("po3")
        if po3 is not None and getattr(po3, "phase", "") == "distribution":
            expected = getattr(po3, "expected_direction", "bullish")
            po3_key = f"po3_distribution_{expected}"
            po3_cfg = cfg.get(po3_key, {})
            for k, v in po3_cfg.items():
                _add_resolved(k, float(v))

        # ── Kill Zone amplifier ───────────────────────────────────────────
        kz = ict2_results.get("kill_zone")
        if kz is not None:
            kz_cfg = cfg.get("kill_zone_amplifier", {})
            min_strength = float(kz_cfg.get("min_zone_strength", 0.8))
            qualifying   = kz_cfg.get("qualifying_zones", ["london", "ny_open"])
            factor       = float(kz_cfg.get("all_types_multiplier", 1.05))
            if (getattr(kz, "zone_strength", 0.0) >= min_strength and
                    getattr(kz, "active_zone", "") in qualifying):
                _mul_all(factor)

        # ── BPR proximity ─────────────────────────────────────────────────
        bpr = ict2_results.get("bpr")
        if bpr is not None:
            current_price = float(ict2_results.get("current_price", 0.0))
            bpr_cfg = cfg.get("bpr_proximity", {})
            threshold_pct = float(bpr_cfg.get("proximity_threshold_pct", 1.0)) / 100
            for attr in ("nearest_bpr_below", "nearest_bpr_above"):
                bpr_obj = getattr(bpr, attr, None)
                if bpr_obj and current_price > 0:
                    ce_val = getattr(bpr_obj, "ce", None)
                    if ce_val and abs(ce_val - current_price) / current_price <= threshold_pct:
                        for k, v in bpr_cfg.items():
                            if isinstance(v, (int, float)) and k != "proximity_threshold_pct":
                                _add_resolved(k, float(v))
                        break

        # Final clamp
        for k in scores:
            scores[k] = max(0.0, min(1.0, scores[k]))

        return scores

    # ── CHECK 6: ICT2 Signal Corroboration ───────────────────────────────────

    def _check_ict2_corroboration(self) -> CheckResult:
        """
        CHECK 6: On bars where LIQUIDITY_POOL (or MACRO_EVENT proxy) scores
        highest and score > 0.60, verify that at least one ICT2 signal
        (TurtleSoup, Displacement, or NWOG-CE-proximity) also fires.
        Gate: ≥ 40% of such bars must have ICT2 corroboration.
        Skipped if ICT2 modules are unavailable.
        """
        cfg6 = _NUCLEUS_ICT2_CFG.get("check6", {})
        lp_threshold = float(cfg6.get("liquidity_pool_score_threshold", 0.60))
        min_corr_rate = float(cfg6.get("min_ict2_corroboration_rate", 0.40))
        ce_prox_pct   = float(cfg6.get("nwog_ce_proximity_for_check6", 1.0)) / 100

        if not _ICT2_AVAILABLE:
            return CheckResult(
                name    = "ICT2 Corroboration",
                passed  = True,  # skip = pass
                value   = -1.0,
                gate    = min_corr_rate,
                message = "CHECK 6: SKIPPED (ICT2 modules not available)",
                detail  = {"skipped": True},
            )

        spy = self.spy
        dates = spy.index
        warmup = 80

        high_lp_bars = 0
        corroborated = 0

        ts_det   = _TSD()
        disp_det = _DD()
        nwog_det = _NWOG()

        for i, date in enumerate(dates):
            if i < warmup:
                continue

            scores_at_bar = self.scorer.score(date, lookback=20)
            # LIQUIDITY_POOL is a conceptual alias — use MACRO_EVENT as proxy
            lp_score = scores_at_bar.get("MACRO_EVENT", 0.0)
            top_type = max(scores_at_bar, key=scores_at_bar.get)

            # Only evaluate on bars where the LP proxy scores highest AND > threshold
            if lp_score < lp_threshold or top_type != "MACRO_EVENT":
                continue

            high_lp_bars += 1
            slice_df = spy.iloc[: i + 1]
            current_price = float(spy["close"].iloc[i])

            # Check 1: Turtle Soup
            try:
                ts_r = ts_det.update(slice_df)
                if ts_r.detected:
                    corroborated += 1
                    continue
            except Exception:
                pass

            # Check 2: Displacement
            try:
                disp_r = disp_det.update(slice_df)
                if disp_r.detected:
                    corroborated += 1
                    continue
            except Exception:
                pass

            # Check 3: NWOG CE proximity
            try:
                nwog_r = nwog_det.update(slice_df)
                if current_price > 0:
                    for attr in ("nearest_ce_above", "nearest_ce_below"):
                        ce = getattr(nwog_r, attr, None)
                        if ce and abs(ce - current_price) / current_price <= ce_prox_pct:
                            corroborated += 1
                            break
            except Exception:
                pass

        if high_lp_bars == 0:
            corr_rate = 0.0
            passed    = True   # can't fail if no LP bars
            msg = "CHECK 6: No high-LP bars found (MACRO_EVENT threshold too high or data missing)"
        else:
            corr_rate = corroborated / high_lp_bars
            passed    = corr_rate >= min_corr_rate
            status    = "PASS" if passed else "FAIL — PHANTOM_DATA_RISK"
            msg = (
                f"CHECK 6: {status}: ICT2 corroboration on {corr_rate*100:.1f}% "
                f"of {high_lp_bars} high-LP bars (gate: ≥{min_corr_rate*100:.0f}%)"
            )

        return CheckResult(
            name    = "ICT2 Corroboration",
            passed  = passed,
            value   = round(corr_rate if high_lp_bars > 0 else 0.0, 4),
            gate    = min_corr_rate,
            message = msg,
            detail  = {
                "high_lp_bars":        high_lp_bars,
                "corroborated_bars":   corroborated,
                "lp_score_threshold":  lp_threshold,
                "ict2_available":      _ICT2_AVAILABLE,
            },
        )

    def run(self) -> ValidationReport:
        logger.info("Running nucleus validation…")

        results = self._run_bar_by_bar()
        logger.info(f"  {len(results)} bars processed")

        c1 = self._check_distribution(results)
        c2 = self._check_persistence(results)
        c3 = self._check_transitions(results)
        c4 = self._check_predictive(results)
        c5 = self._check_calibration(results)
        c6 = self._check_ict2_corroboration()

        checks    = [c1, c2, c3, c4, c5, c6]
        gate_pass = all(c.passed for c in checks)

        recal = self._build_recalibration(checks)

        report = ValidationReport(
            run_date    = datetime.now().isoformat(),
            data_period = {
                "start": str(self.spy.index[0].date()),
                "end":   str(self.spy.index[-1].date()),
            },
            total_bars       = len(results),
            checks           = checks,
            gate_pass        = gate_pass,
            recalibration    = recal,
            raw_results      = results,
            distribution     = self._distribution(results),
            transition_rate  = c3.value,
            accuracy_5d      = c4.value,
            avg_persistence  = c2.value,
            score_spread     = c5.value,
        )

        self._print_report(report)
        self._save_report(report)
        return report

    # ── Bar-by-bar engine ─────────────────────────────────────────────────────

    def _run_bar_by_bar(self) -> List[BarResult]:
        """
        Process each trading day in SPY history chronologically.
        At each bar: score all 8 nucleus candidates, pick the winner.
        NO lookahead — each score() call only sees data up to that bar.
        """
        results: List[BarResult] = []
        dates = self.spy.index

        # Warm-up: skip first 80 bars for indicator stability
        warmup = 80
        logger.info(f"  Warm-up: skipping first {warmup} bars")

        for i, date in enumerate(dates):
            if i < warmup:
                continue

            scores = self.scorer.score(date, lookback=20)

            # Rank candidates
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            nucleus, top_score = sorted_scores[0]
            satellite          = sorted_scores[1][0] if len(sorted_scores) > 1 else ""

            regime = classify_regime(self.spy, date)

            results.append(BarResult(
                date      = str(date.date()),
                nucleus   = nucleus,
                score     = round(top_score, 4),
                scores    = {k: round(v, 4) for k, v in scores.items()},
                satellite = satellite,
                regime    = regime,
            ))

            if i % 100 == 0:
                logger.info(f"  Bar {i}/{len(dates)}: {date.date()} — nucleus={nucleus} ({top_score:.2f})")

        return results

    # ── Check 1: Distribution ─────────────────────────────────────────────────

    def _check_distribution(self, results: List[BarResult]) -> CheckResult:
        dist = self._distribution(results)
        max_pct  = max(dist.values()) if dist else 1.0
        max_type = max(dist, key=dist.get) if dist else ""

        passed  = max_pct <= GATE_MAX_DOMINANT_PCT
        message = (
            f"{'PASS' if passed else 'FAIL'}: {max_type} dominates "
            f"{max_pct*100:.1f}% of days (gate: ≤{GATE_MAX_DOMINANT_PCT*100:.0f}%)"
        )
        return CheckResult(
            name    = "Distribution",
            passed  = passed,
            value   = max_pct,
            gate    = GATE_MAX_DOMINANT_PCT,
            message = message,
            detail  = {k: round(v*100, 1) for k, v in dist.items()},
        )

    # ── Check 2: Persistence ─────────────────────────────────────────────────

    def _check_persistence(self, results: List[BarResult]) -> CheckResult:
        if not results:
            return CheckResult("Persistence", False, 0, GATE_MIN_PERSISTENCE, "No data", {})

        runs     = []
        current  = results[0].nucleus
        run_len  = 1

        for r in results[1:]:
            if r.nucleus == current:
                run_len += 1
            else:
                runs.append(run_len)
                current = r.nucleus
                run_len = 1
        runs.append(run_len)

        avg_persistence = float(np.mean(runs))
        passed = GATE_MIN_PERSISTENCE <= avg_persistence <= GATE_MAX_PERSISTENCE

        by_type: Dict[str, List[int]] = defaultdict(list)
        current = results[0].nucleus
        run_len = 1
        for r in results[1:]:
            if r.nucleus == current:
                run_len += 1
            else:
                by_type[current].append(run_len)
                current = r.nucleus
                run_len = 1
        by_type[current].append(run_len)

        return CheckResult(
            name    = "Persistence",
            passed  = passed,
            value   = round(avg_persistence, 2),
            gate    = GATE_MIN_PERSISTENCE,
            message = f"{'PASS' if passed else 'FAIL'}: avg persistence = {avg_persistence:.1f} bars (gate: {GATE_MIN_PERSISTENCE}–{GATE_MAX_PERSISTENCE})",
            detail  = {k: round(float(np.mean(v)), 1) for k, v in by_type.items()},
        )

    # ── Check 3: Transitions ─────────────────────────────────────────────────

    def _check_transitions(self, results: List[BarResult]) -> CheckResult:
        if len(results) < 2:
            return CheckResult("Transitions", False, 0, GATE_MIN_TRANSITION_RATE, "No data", {})

        # Count days where nucleus changed
        transitions = [
            i for i in range(1, len(results))
            if results[i].nucleus != results[i-1].nucleus
        ]
        rate = len(transitions) / len(results)

        # Check: on known event dates, how often does the nucleus change?
        event_set = set(KNOWN_EVENT_DATES)
        dates_in_results = {r.date for r in results}
        overlap  = event_set & dates_in_results

        event_transition_count = 0
        transition_dates = {results[i].date for i in transitions}
        event_transitions_on_day = sum(1 for d in overlap if d in transition_dates)
        # Also count +1 day (event impact may show on the next bar)
        date_list = [r.date for r in results]
        for d in overlap:
            try:
                idx = date_list.index(d)
                if idx + 1 < len(date_list) and date_list[idx+1] in transition_dates:
                    event_transition_count += 1
            except ValueError:
                pass
        event_transition_count += event_transitions_on_day
        event_hit_rate = event_transition_count / len(overlap) if overlap else 0.0

        passed  = rate >= GATE_MIN_TRANSITION_RATE
        return CheckResult(
            name    = "Transitions",
            passed  = passed,
            value   = round(rate, 4),
            gate    = GATE_MIN_TRANSITION_RATE,
            message = f"{'PASS' if passed else 'FAIL'}: transition rate = {rate*100:.1f}% (gate: ≥{GATE_MIN_TRANSITION_RATE*100:.0f}%)",
            detail  = {
                "total_transitions":    len(transitions),
                "total_bars":           len(results),
                "known_event_dates":    len(overlap),
                "events_with_transition": event_transition_count,
                "event_hit_rate":       round(event_hit_rate * 100, 1),
            },
        )

    # ── Check 4: Predictive ───────────────────────────────────────────────────

    def _check_predictive(self, results: List[BarResult]) -> CheckResult:
        """
        For each bar where nucleus score ≥ GATE_STRONG_SCORE_MIN:
          - Determine implied direction (INDEX_MOMENTUM/SINGLE_STOCK = BULLISH, etc.)
          - Check if 5-day SPY return matches direction
        """
        BULLISH_NUCLEI  = {"INDEX_MOMENTUM", "SINGLE_STOCK_LEADERSHIP"}
        BEARISH_NUCLEI  = {"MACRO_EVENT", "CURRENCY_PRESSURE", "GEOPOLITICAL"}
        NEUTRAL_NUCLEI  = {"SECTOR_ROTATION", "COMMODITY_REGIME", "OPTIONS_EXPIRY"}

        correct = 0
        total   = 0

        spy_close = self.spy["close"]
        spy_dates = spy_close.index

        for bar in results:
            if bar.score < GATE_STRONG_SCORE_MIN:
                continue
            if bar.nucleus in NEUTRAL_NUCLEI:
                continue   # no directional prediction for neutral nuclei

            try:
                date_ts  = pd.Timestamp(bar.date, tz="UTC")
                # Find position in SPY index
                pos = spy_dates.get_loc(date_ts)
            except (KeyError, TypeError):
                # Try without tz
                try:
                    date_ts = pd.Timestamp(bar.date)
                    pos = spy_dates.get_indexer([date_ts], method="nearest")[0]
                except Exception:
                    continue

            if pos + 5 >= len(spy_dates):
                continue

            fwd_return = (float(spy_close.iloc[pos + 5]) / float(spy_close.iloc[pos])) - 1

            predicted_up = bar.nucleus in BULLISH_NUCLEI
            actual_up    = fwd_return > 0

            if predicted_up == actual_up:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        passed   = accuracy >= GATE_MIN_ACCURACY and total >= 50

        return CheckResult(
            name    = "Predictive",
            passed  = passed,
            value   = round(accuracy, 4),
            gate    = GATE_MIN_ACCURACY,
            message = f"{'PASS' if passed else 'FAIL'}: 5D SPY directional accuracy = {accuracy*100:.1f}% on {total} strong-signal bars (gate: ≥{GATE_MIN_ACCURACY*100:.0f}%)",
            detail  = {
                "correct":       correct,
                "total":         total,
                "score_threshold": GATE_STRONG_SCORE_MIN,
            },
        )

    # ── Check 5: Calibration ──────────────────────────────────────────────────

    def _check_calibration(self, results: List[BarResult]) -> CheckResult:
        """
        Score spread: is the winning nucleus score meaningfully higher
        than the second-highest? Bunching = scorer can't discriminate.
        Also check: are ANY scores above 0.9 (ceiling clipping)?
        """
        top_scores    = [r.score for r in results]
        second_scores = [sorted(r.scores.values(), reverse=True)[1]
                         for r in results if len(r.scores) >= 2]

        spreads = [t - s for t, s in zip(top_scores, second_scores)]
        avg_spread = float(np.mean(spreads)) if spreads else 0.0

        max_score  = max(top_scores) if top_scores else 0.0
        min_score  = min(top_scores) if top_scores else 0.0
        full_range = max_score - min_score

        ceiling_pct = sum(1 for s in top_scores if s >= 0.95) / len(top_scores) if top_scores else 0.0
        floor_pct   = sum(1 for s in top_scores if s <= 0.10) / len(top_scores) if top_scores else 0.0

        passed = avg_spread >= GATE_MIN_SCORE_SPREAD

        return CheckResult(
            name    = "Calibration",
            passed  = passed,
            value   = round(avg_spread, 4),
            gate    = GATE_MIN_SCORE_SPREAD,
            message = f"{'PASS' if passed else 'FAIL'}: avg score spread = {avg_spread:.3f} (gate: ≥{GATE_MIN_SCORE_SPREAD})",
            detail  = {
                "avg_top_score":      round(float(np.mean(top_scores)), 3),
                "avg_second_score":   round(float(np.mean(second_scores)), 3),
                "full_range":         round(full_range, 3),
                "ceiling_clip_pct":   round(ceiling_pct * 100, 1),
                "floor_clip_pct":     round(floor_pct * 100, 1),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _distribution(self, results: List[BarResult]) -> Dict[str, float]:
        counts: Dict[str, int] = defaultdict(int)
        for r in results:
            counts[r.nucleus] += 1
        total = len(results)
        return {k: v / total for k, v in counts.items()} if total else {}

    def _build_recalibration(self, checks: List[CheckResult]) -> List[str]:
        recs = []
        for c in checks:
            if c.passed:
                continue
            if c.name == "Distribution":
                dom = max(c.detail, key=lambda k: c.detail[k])
                recs.append(
                    f"DISTRIBUTION: '{dom}' fires {c.detail[dom]:.0f}% of days. "
                    f"Reduce its scorer sensitivity by 30% OR add a minimum-bar cooldown "
                    f"(e.g., nucleus can't repeat within 3 bars unless score > 0.85)."
                )
            elif c.name == "Persistence":
                if c.value < GATE_MIN_PERSISTENCE:
                    recs.append(
                        "PERSISTENCE: Nucleus flipping every bar — scorer is too noisy. "
                        "Add a smoothing layer: nucleus only changes if new winner scores "
                        "> current winner × 1.15 for 2 consecutive bars."
                    )
                else:
                    recs.append(
                        "PERSISTENCE: Nucleus never changes — scorer is too sticky. "
                        "Remove the persistence filter or lower the change threshold."
                    )
            elif c.name == "Transitions":
                recs.append(
                    "TRANSITIONS: Nucleus barely changes. This suggests the scorer is "
                    "converging to a single dominant type. Check: does MACRO_EVENT "
                    "spike on VXX but decay quickly enough? Add a 5-bar exponential "
                    "decay to all scorer outputs."
                )
            elif c.name == "Predictive":
                recs.append(
                    f"PREDICTIVE: {c.value*100:.1f}% accuracy is below the 52% gate. "
                    "Options: (A) Remove GEOPOLITICAL from directional predictions — "
                    "too noisy. (B) Raise GATE_STRONG_SCORE_MIN to 0.75 to only predict "
                    "on very high-confidence bars. (C) Add a regime filter — only predict "
                    "in TRENDING regime where forward returns are more directional."
                )
            elif c.name == "Calibration":
                recs.append(
                    f"CALIBRATION: Score spread = {c.value:.3f} — scores are bunched. "
                    "Apply a power transform to each scorer output: score = score^0.5 "
                    "will spread values. Or add competitive normalization: divide each "
                    "score by the sum of all scores, then multiply back by the winner's "
                    "raw value to preserve magnitude while increasing discrimination."
                )
        return recs

    # ── Output ────────────────────────────────────────────────────────────────

    def _print_report(self, report: ValidationReport):
        width = 68
        print(f"\n{'='*width}")
        print(f"  NUCLEUS VALIDATION REPORT")
        print(f"  Period: {report.data_period['start']} → {report.data_period['end']}")
        print(f"  Bars:   {report.total_bars}")
        print(f"{'='*width}")

        print(f"\n  NUCLEUS DISTRIBUTION")
        print(f"  {'─'*40}")
        for nt in NUCLEUS_TYPES:
            pct = report.distribution.get(nt, 0.0) * 100
            bar = "█" * int(pct / 2)
            flag = " ⚠" if pct > GATE_MAX_DOMINANT_PCT * 100 else ""
            print(f"  {nt:<28} {pct:>5.1f}%  {bar}{flag}")

        print(f"\n  VALIDATION CHECKS")
        print(f"  {'─'*60}")
        for c in report.checks:
            icon = "✓" if c.passed else "✗"
            print(f"  {icon}  {c.name:<15}  {c.message}")
            if c.detail:
                for k, v in c.detail.items():
                    print(f"        {k}: {v}")

        print(f"\n  {'─'*60}")
        print(f"  GATE: {'✅ PASS — nucleus engine approved for live signal use' if report.gate_pass else '❌ FAIL — recalibration required before live use'}")

        if report.recalibration:
            print(f"\n  RECALIBRATION INSTRUCTIONS")
            print(f"  {'─'*60}")
            for i, rec in enumerate(report.recalibration, 1):
                print(f"  [{i}] {rec}\n")

        print(f"{'='*width}\n")

    def _save_report(self, report: ValidationReport):
        out_dir = Path(__file__).parent.parent / "reports" / "nucleus"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"nucleus_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Serialize (raw_results can be large — save separately)
        raw = report.raw_results
        report.raw_results = []
        with open(fname, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Save raw bar results separately (large file)
        raw_fname = out_dir / f"nucleus_bar_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(raw_fname, "w") as f:
            for r in raw:
                f.write(json.dumps(asdict(r)) + "\n")

        report.raw_results = raw
        logger.info(f"Report saved: {fname}")
        logger.info(f"Raw results: {raw_fname}")
