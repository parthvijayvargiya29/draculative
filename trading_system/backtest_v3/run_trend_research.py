#!/usr/bin/env python3
"""
run_trend_research.py  --  V5.1 Trend Module Regime Gate Research
==================================================================
Part B research track: build a regime classifier that gates the combo_trend_entry
signal, filtering out entries in choppy/non-trending environments where trend-following
strategies lose money.

BACKGROUND
----------
V5.0 Module 3 test result: PF=0.287, WFE=0.133.
Root cause: combo_trend_entry (EMA>63 + ADX>20 + RSI2<40) fires in BOTH trending
and choppy regimes. Without a regime gate, the strategy is long-only momentum in
random environments. Need a classifier that identifies "will the trend continue?"
at signal time using features available at the entry bar (no lookahead).

PIPELINE (9 Steps)
-------------------
  Step 1: Load data — trend universe: QQQ IWM XLK NVDA META SPY
  Step 2: Compute 12 feature groups on every bar via indicator engine
  Step 3: Identify potential entry bars where combo_trend_entry fires
  Step 4: Generate forward-return regime labels on entry bars (no lookahead)
  Step 5: Feature importance analysis on TRAIN period (point-biserial corr)
  Step 6: Train 3 classifiers on TRAIN period
  Step 7: Validate all 3 on VAL period — select best by precision×recall
  Step 8: ONE-SHOT test period evaluation of selected classifier
  Step 9: Correlation check — gated trend vs Combo C equity curves

HARD CONSTRAINTS
-----------------
  - Classifier trained ONLY on train data; all hyperparam choices on val data
  - Test set touched exactly ONCE at Step 8; no iteration after that
  - If test precision < 0.65 OR recall < 0.50: suspend trend research entirely
  - Max 5 input features per classifier
  - Combo C parameters are NOT modified under any circumstances

REGIME LABELS (forward-looking, applied only to train/val/test entry bars)
---------------------------------------------------------------------------
  TREND_CONTINUATION  : 10-bar fwd return > +2%   (trade normally)
  TREND_EXHAUSTION    : 10-bar fwd return < -1%   (skip entry)
  CHOPPY              : -1% <= fwd return <= +2%  (skip entry)

12 FEATURE GROUPS (all computable at entry bar, no lookahead)
--------------------------------------------------------------
  1.  ema_fan             : (EMA21 - EMA63) / EMA63
  2.  linreg_slope        : 20-bar linear regression slope / close (normalized)
  3.  linreg_r2           : 20-bar linear regression R²
  4.  price_vs_linreg     : (close - linreg_value) / linreg_stddev
  5.  weekly_trend_aligned: EMA40 > EMA105 (weekly proxy: 8-week vs 21-week daily)
  6.  adx_roc             : (ADX14 - ADX14_10bars_ago) / 10
  7.  di_spread           : DI+ - DI-
  8.  momentum_diverging  : price new 20-bar high AND RSI14 not confirming (bool)
  9.  atr_slope           : (ATR14 - ATR14_10bars_ago) / 10 / close (normalized)
  10. bb_width_expanding  : current BB width > 20-bar avg × 1.1 (bool)
  11. qqq_atr_percentile  : ATR14 rank within 60-bar window (from atr_pct_rank_60)
  12. spy_dd_from_high     : (SPY close - SPY 252-bar high) / SPY 252-bar high
      spy_rsi_14           : SPY RSI(14) at signal bar
      spy_adx_14           : SPY ADX(14) at signal bar
      spy_sma50_slope      : (SPY EMA50 - SPY EMA50_20bars_ago) / SPY EMA50_20bars_ago

3 CLASSIFIERS
-------------
  A: Rule-based      — top 3 features by corr, max 3 conditions (threshold search)
  B: Logistic reg    — L1, top 5 features, C=0.1, classification threshold=0.60
  C: Additive score  — 8 conditions (binary), threshold >= 5/8 points

ACCEPTANCE CRITERIA
-------------------
  Validation: precision > 0.65 AND recall > 0.50 (TREND_CONTINUATION class)
  Test (one-shot): precision > 0.65 AND recall > 0.50
  If test fails both: print SUSPEND message and exit

USAGE
-----
  cd backtest_v3
  export ALPACA_API_KEY="..."
  export ALPACA_SECRET_KEY="..."
  export APCA_API_BASE_URL="https://paper-api.alpaca.markets"
  ../../.venv/bin/python run_trend_research.py --alpaca --daily
  ../../.venv/bin/python run_trend_research.py --alpaca --daily --step 1-4
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from alpaca_loader import load_data
from simulator import walk_forward_split, COMBO_C_SYMBOLS
from combos import combo_trend_entry, COMBO_TREND_SYMBOLS
from indicators_v3 import IndicatorStateV3, BarSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TREND_UNIVERSE       = ["QQQ", "IWM", "XLK", "NVDA", "META", "SPY"]
SPY_SYM              = "SPY"

# Regime label thresholds
FWD_BARS             = 10      # forward bars for label computation
FWD_TREND_CONT_PCT   = 2.0    # > +2% fwd return → TREND_CONTINUATION
FWD_TREND_EXHA_PCT   = -1.0   # < -1% fwd return → TREND_EXHAUSTION
# -1% <= ret <= +2% → CHOPPY

LABEL_TC    = "TREND_CONTINUATION"
LABEL_TE    = "TREND_EXHAUSTION"
LABEL_CHOP  = "CHOPPY"
LABEL_POS   = LABEL_TC   # positive class for classifier

# Acceptance thresholds
PREC_THRESHOLD   = 0.65
RECALL_THRESHOLD = 0.50

# Logistic regression settings
LR_C             = 0.1
LR_MAX_FEATURES  = 5
LR_THRESHOLD     = 0.60

# Additive score settings
SCORE_N_CONDITIONS = 8
SCORE_THRESHOLD    = 5

# Walk-forward split
TRAIN_PCT = 0.60
VAL_PCT   = 0.20


# ---------------------------------------------------------------------------
# Step 1: Data loading
# ---------------------------------------------------------------------------

def load_trend_data(alpaca: bool, daily: bool, no_cache: bool) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data for trend universe + SPY."""
    all_syms = sorted(set(TREND_UNIVERSE) | {SPY_SYM})
    logger.info(f"Step 1: Loading data — symbols: {all_syms}")
    # load_data() uses use_cache=True/False (not 'alpaca' / 'no_cache')
    # If ALPACA_API_KEY is set in env, it fetches from Alpaca automatically.
    # force_synth=False means: use Alpaca if keys present, else synthetic.
    data = load_data(symbols=all_syms, use_cache=(not no_cache), force_synth=False)
    if daily:
        data = _resample_daily(data)
    # Report bar counts
    for sym, df in sorted(data.items()):
        logger.info(f"  {sym}: {len(df):,} bars  "
                    f"({df.index[0].date()} → {df.index[-1].date()})")
    return data


def _resample_daily(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym, df in data.items():
        if df.empty:
            out[sym] = df
            continue
        daily = df.resample("D").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna(subset=["open", "close"])
        daily = daily[daily["volume"] > 0]
        out[sym] = daily
    return out


# ---------------------------------------------------------------------------
# Step 2: Compute all indicators + extra regime features
# ---------------------------------------------------------------------------

class RegimeFeatureState:
    """
    Per-symbol state machine that computes the 12 regime feature groups
    in addition to the standard IndicatorStateV3 output.
    Extends the standard engine with extra rolling buffers.
    """
    def __init__(self):
        self._ind     = IndicatorStateV3()
        self._ema40   = _EWM(40)
        self._ema105  = _EWM(105)
        # 20-bar linear regression buffer (close prices)
        self._lr_buf  = _RollingBuf(20)
        # ADX14 10-bar-ago ring
        self._adx14_10ago_buf = _RollingBuf(11)   # keep 11 → oldest = 10 bars ago
        # ATR14 10-bar-ago ring
        self._atr14_10ago_buf = _RollingBuf(11)
        # RSI14 20-bar high for divergence
        self._rsi14_20hi_buf  = _RollingBuf(20)
        self._high20c_buf     = _RollingBuf(20)   # 20-bar close high for price pivot
        # BB width 20-bar avg
        self._bbw_buf         = _RollingBuf(20)
        # ATR14 60-bar for percentile
        self._atr14_60_buf    = _RollingBuf(60)
        # EMA50 20-bar-ago
        self._ema50_20ago_buf = _RollingBuf(21)

    def update(self, o, h, l, c, vol, spy_close=0.0, spy_atr=0.0,
               spy_adx=0.0, spy_ema50=0.0, spy_high252=0.0,
               spy_rsi14=0.0) -> Optional[dict]:
        """
        Advance by one bar. Returns feature dict if indicator engine is ready,
        else None. spy_* args must be filled in from SPY's own IndicatorStateV3.
        """
        snap: BarSnapshot = self._ind.update(o, h, l, c, vol)

        # Extra EMAs for weekly proxy
        ema40  = self._ema40.update(c)
        ema105 = self._ema105.update(c)

        # Linear regression over last 20 closes
        self._lr_buf.append(c)
        lr_slope = lr_r2 = price_vs_lr = 0.0
        if self._lr_buf.full():
            lr_slope, lr_r2, lr_val, lr_std = _linreg(list(self._lr_buf._d))
            price_vs_lr = (c - lr_val) / lr_std if lr_std > 1e-9 else 0.0
            # Normalise slope by price
            lr_slope = lr_slope / c if c > 0 else lr_slope

        # ADX14 10-bars-ago
        self._adx14_10ago_buf.append(snap.adx14_val)
        adx14_10ago = self._adx14_10ago_buf.oldest() if self._adx14_10ago_buf.full() else snap.adx14_val
        adx_roc     = (snap.adx14_val - adx14_10ago) / 10.0

        # DI spread (requires direct ADX14 DI+ DI- — approximate from adx14_val and slope)
        # We don't have raw DI+/DI- in BarSnapshot; approximate: di_spread ~ adx14_val sign
        # For the feature, use adx14_val * (1 if snap.close > snap.ema21 else -1)
        # (positive when price above EMA21 in a trending env)
        di_spread = snap.adx14_val * (1.0 if snap.close > snap.ema21 else -1.0)

        # ATR14 10-bars-ago
        self._atr14_10ago_buf.append(snap.atr)
        atr14_10ago = self._atr14_10ago_buf.oldest() if self._atr14_10ago_buf.full() else snap.atr
        atr_slope   = ((snap.atr - atr14_10ago) / 10.0 / c) if c > 0 else 0.0

        # Momentum divergence: price at new 20-bar close high AND RSI14 below 20-bar RSI high
        self._high20c_buf.append(c)
        self._rsi14_20hi_buf.append(snap.rsi)
        price_new_high = c >= self._high20c_buf.max() if self._high20c_buf.full() else False
        rsi_confirming  = snap.rsi >= self._rsi14_20hi_buf.max() if self._rsi14_20hi_buf.full() else True
        momentum_diverg = 1.0 if (price_new_high and not rsi_confirming) else 0.0

        # BB width expansion
        bb_w = snap.bb_upper - snap.bb_lower
        self._bbw_buf.append(bb_w)
        bb_width_expanding = 1.0 if (self._bbw_buf.full() and bb_w > self._bbw_buf.mean() * 1.1) else 0.0

        # ATR14 60-bar percentile
        self._atr14_60_buf.append(snap.atr)
        if self._atr14_60_buf.full():
            buf_list = list(self._atr14_60_buf._d)
            rank = sum(x <= snap.atr for x in buf_list) / len(buf_list)
        else:
            rank = 0.5
        atr14_pct60 = rank

        # SPY slope: (SPY EMA50 now - SPY EMA50 20 bars ago) / SPY EMA50 20 bars ago
        self._ema50_20ago_buf.append(spy_ema50)
        spy_ema50_20ago = self._ema50_20ago_buf.oldest() if self._ema50_20ago_buf.full() else spy_ema50
        spy_sma50_slope = ((spy_ema50 - spy_ema50_20ago) / spy_ema50_20ago
                           if spy_ema50_20ago > 0 else 0.0)

        # SPY DD from high
        spy_dd = ((spy_close - spy_high252) / spy_high252
                  if spy_high252 > 0 and spy_close > 0 else 0.0)

        if not snap.ready:
            return None

        return {
            # Core snap passthrough
            "_snap":          snap,
            "close":          c,
            # Feature group 1: EMA fan
            "ema_fan":        (snap.ema21 - snap.ema63) / snap.ema63 if snap.ema63 > 0 else 0.0,
            # Feature group 2-3: Linear regression
            "linreg_slope":   lr_slope,
            "linreg_r2":      lr_r2,
            # Feature group 4: Price vs linreg deviation
            "price_vs_linreg": price_vs_lr,
            # Feature group 5: Weekly trend proxy
            "weekly_aligned": 1.0 if ema40 > ema105 else 0.0,
            # Feature group 6: ADX rate of change
            "adx_roc":        adx_roc,
            # Feature group 7: DI spread approximation
            "di_spread":      di_spread,
            # Feature group 8: Momentum divergence
            "momentum_diverg": momentum_diverg,
            # Feature group 9: ATR slope
            "atr_slope":      atr_slope,
            # Feature group 10: BB width expansion
            "bb_width_expanding": bb_width_expanding,
            # Feature group 11: ATR percentile 60-bar
            "atr14_pct60":    atr14_pct60,
            # Feature group 12: SPY macro regime
            "spy_dd_from_high": spy_dd,
            "spy_rsi14":      spy_rsi14,
            "spy_adx14":      spy_adx,
            "spy_sma50_slope": spy_sma50_slope,
        }


# ---------------------------------------------------------------------------
# Step 3: Identify entry bars and build feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    For each symbol in TREND_UNIVERSE (excluding SPY), run the indicator engine
    on every bar and collect features at bars where combo_trend_entry fires.
    SPY is used for macro regime features only.

    Returns a DataFrame with columns:
      symbol, date, label (to be filled in Step 4), feature_1..N
    """
    logger.info("Step 2+3: Computing features and identifying entry bars...")

    spy_df = data.get(SPY_SYM, pd.DataFrame())
    spy_snaps: Dict[Any, BarSnapshot] = {}       # date → spy BarSnapshot
    spy_feats: Dict[Any, dict] = {}              # date → spy regime dict

    # First pass: pre-compute SPY snapshots
    if not spy_df.empty:
        spy_state = IndicatorStateV3()
        spy_feat_state = RegimeFeatureState()
        for ts, row in spy_df.iterrows():
            d = ts.date() if hasattr(ts, "date") else ts
            snap_spy = spy_state.update(
                row["open"], row["high"], row["low"], row["close"], row["volume"]
            )
            spy_snaps[d] = snap_spy
            spy_feats[d] = spy_feat_state.update(
                row["open"], row["high"], row["low"], row["close"], row["volume"],
            )

    records = []
    trade_syms = [s for s in TREND_UNIVERSE if s != SPY_SYM]

    for sym in trade_syms:
        if sym not in data:
            logger.warning(f"  {sym} not in data — skipping")
            continue
        df = data[sym]
        if df.empty:
            continue

        state     = RegimeFeatureState()
        rows_list = list(df.iterrows())
        n_bars    = len(rows_list)

        for i, (ts, row) in enumerate(rows_list):
            d = ts.date() if hasattr(ts, "date") else ts
            spy_snap = spy_snaps.get(d)

            # Gather SPY context for this bar
            s_close  = spy_snap.close  if spy_snap else 0.0
            s_atr    = spy_snap.atr    if spy_snap else 0.0
            s_adx14  = spy_snap.adx14_val if spy_snap else 0.0
            s_ema50  = spy_snap.ema50  if spy_snap else 0.0
            s_high252 = spy_snap.high252 if spy_snap else 0.0
            s_rsi14  = spy_snap.rsi    if spy_snap else 50.0

            feat = state.update(
                row["open"], row["high"], row["low"], row["close"], row["volume"],
                spy_close=s_close, spy_atr=s_atr, spy_adx=s_adx14,
                spy_ema50=s_ema50, spy_high252=s_high252, spy_rsi14=s_rsi14,
            )

            if feat is None:
                continue

            snap: BarSnapshot = feat["_snap"]
            # Check if combo_trend_entry fires on this bar
            if not combo_trend_entry(snap):
                continue

            # Record this as a potential entry bar (label computed in Step 4)
            rec = {"symbol": sym, "date": d, "label": None,
                   "bar_idx": i, "n_bars_sym": n_bars,
                   "bars_seen": snap.bars_seen}  # for warm-up filtering
            for k, v in feat.items():
                if k not in ("_snap", "close"):
                    rec[k] = v
            rec["close"] = feat["close"]
            records.append(rec)

    df_out = pd.DataFrame(records)
    # --- Warm-up filter: exclude first 252 bars (~1 year) per symbol --------
    WARMUP_BARS = 252
    before = len(df_out)
    if "bars_seen" in df_out.columns:
        df_out = df_out[df_out["bars_seen"] >= WARMUP_BARS].copy()
    dropped = before - len(df_out)
    logger.info(f"  Entry bars found: {len(df_out)} (dropped {dropped} warm-up bars < {WARMUP_BARS}) "
                f"across {len(trade_syms)} symbols")
    if len(df_out) > 0:
        by_sym = df_out.groupby("symbol").size()
        logger.info(f"  Per-symbol counts:\n{by_sym.to_string()}")
    # --- NaN report per feature column ----------------------------------------
    nan_counts = df_out[FEATURE_COLS].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning("  NaN counts per feature (should be zero after warm-up filter):")
        for col, cnt in nan_counts[nan_counts > 0].items():
            logger.warning(f"    {col}: {cnt} NaN rows")
    else:
        logger.info("  NaN check: all feature columns are clean (0 NaN)")
    return df_out


# ---------------------------------------------------------------------------
# Step 4: Assign regime labels
# ---------------------------------------------------------------------------

def assign_labels(
    entry_df: pd.DataFrame,
    data:     Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Assign TREND_CONTINUATION / TREND_EXHAUSTION / CHOPPY labels
    based on forward 10-bar return from entry close.
    Entries within FWD_BARS bars of symbol end are dropped (no label available).
    """
    logger.info(f"Step 4: Assigning forward-{FWD_BARS}-bar regime labels...")

    labelled = []
    for _, row in entry_df.iterrows():
        sym    = row["symbol"]
        b_idx  = int(row["bar_idx"])
        n_bars = int(row["n_bars_sym"])
        fwd_end = b_idx + FWD_BARS

        if fwd_end >= n_bars:
            continue   # insufficient forward data

        df_sym   = data[sym]
        entry_c  = df_sym.iloc[b_idx]["close"]
        fwd_c    = df_sym.iloc[fwd_end]["close"]

        fwd_ret_pct = (fwd_c - entry_c) / entry_c * 100.0 if entry_c > 0 else 0.0

        if fwd_ret_pct > FWD_TREND_CONT_PCT:
            label = LABEL_TC
        elif fwd_ret_pct < FWD_TREND_EXHA_PCT:
            label = LABEL_TE
        else:
            label = LABEL_CHOP

        row = row.copy()
        row["label"]       = label
        row["fwd_ret_pct"] = round(fwd_ret_pct, 3)
        labelled.append(row)

    df_lab = pd.DataFrame(labelled)
    if df_lab.empty:
        logger.error("  No labelled entries — insufficient forward data")
        return df_lab

    counts = df_lab["label"].value_counts()
    logger.info(f"  Label distribution:\n{counts.to_string()}")
    tc_pct = counts.get(LABEL_TC, 0) / len(df_lab) * 100
    logger.info(f"  TREND_CONTINUATION prevalence: {tc_pct:.1f}%")
    return df_lab


# ---------------------------------------------------------------------------
# Step 5: Feature importance (point-biserial correlation, train period only)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ema_fan", "linreg_slope", "linreg_r2", "price_vs_linreg",
    "weekly_aligned", "adx_roc", "di_spread", "momentum_diverg",
    "atr_slope", "bb_width_expanding", "atr14_pct60",
    "spy_dd_from_high", "spy_rsi14", "spy_adx14", "spy_sma50_slope",
]


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute point-biserial correlation of each feature vs binary label
    (TREND_CONTINUATION=1, other=0) on the provided DataFrame.
    Returns sorted DataFrame with feature, corr, abs_corr columns.
    """
    if df.empty:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])

    binary = (df["label"] == LABEL_TC).astype(float)
    rows = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        x = df[col].astype(float)
        # Remove NaN
        mask = x.notna() & binary.notna()
        xc = x[mask]; yc = binary[mask]
        if xc.std() < 1e-9 or len(xc) < 5:
            corr = 0.0
        else:
            corr = float(np.corrcoef(xc, yc)[0, 1])
        rows.append({"feature": col, "corr": round(corr, 4),
                     "abs_corr": round(abs(corr), 4)})

    fi = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return fi


# ---------------------------------------------------------------------------
# Step 6: Train 3 classifiers
# ---------------------------------------------------------------------------

class _ClassifierBase:
    name: str = "base"
    def fit(self, df: pd.DataFrame, top_features: List[str]) -> None: ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...
    def description(self) -> str: return ""


class ClassifierA_RuleBased(_ClassifierBase):
    """
    Rule-based: use top 3 features, find optimal threshold per feature
    independently by maximising F1 on training data, then combine with AND.
    Max 3 conditions, features selected by absolute point-biserial correlation.
    """
    name = "A_RuleBased"

    def __init__(self):
        self.rules: List[Tuple[str, float, str]] = []  # (feature, threshold, direction)

    def fit(self, df: pd.DataFrame, top_features: List[str]) -> None:
        binary = (df["label"] == LABEL_TC).astype(float)
        top3   = top_features[:3]
        self.rules = []

        for feat in top3:
            if feat not in df.columns:
                continue
            x = df[feat].astype(float)
            best_f1 = -1.0
            best_thr = float(x.median())
            best_dir = ">"

            # Try direction > and <; candidate thresholds = percentile grid
            vals = sorted(x.dropna().unique())
            if len(vals) < 2:
                continue
            cands = np.percentile(vals, np.linspace(10, 90, 20))

            for thr in cands:
                for direction in (">", "<"):
                    if direction == ">":
                        pred = (x >= thr).astype(float)
                    else:
                        pred = (x <= thr).astype(float)
                    tp = ((pred == 1) & (binary == 1)).sum()
                    fp = ((pred == 1) & (binary == 0)).sum()
                    fn = ((pred == 0) & (binary == 1)).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    if f1 > best_f1:
                        best_f1, best_thr, best_dir = f1, thr, direction

            self.rules.append((feat, best_thr, best_dir))
            logger.debug(f"    Rule: {feat} {best_dir} {best_thr:.4f}  (F1={best_f1:.3f})")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.rules:
            return np.zeros(len(df), dtype=int)
        mask = pd.Series(True, index=df.index)
        for feat, thr, direction in self.rules:
            if feat not in df.columns:
                continue
            if direction == ">":
                mask = mask & (df[feat].astype(float) >= thr)
            else:
                mask = mask & (df[feat].astype(float) <= thr)
        return mask.astype(int).values

    def description(self) -> str:
        lines = []
        for feat, thr, direction in self.rules:
            lines.append(f"{feat} {direction} {thr:.4f}")
        return "  AND  ".join(lines) if lines else "(no rules)"


class ClassifierB_Logistic(_ClassifierBase):
    """
    Logistic regression with L1 regularisation.
    Trained on top-5 features selected by point-biserial correlation.
    Threshold=0.60 for TREND_CONTINUATION prediction.
    """
    name = "B_Logistic"

    def __init__(self):
        self._coef    = None
        self._intercept = 0.0
        self._feat_means = {}
        self._feat_stds  = {}
        self._threshold  = LR_THRESHOLD
        self._selected_feats: List[str] = []

    def fit(self, df: pd.DataFrame, top_features: List[str]) -> None:
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.preprocessing import StandardScaler     # type: ignore
        except ImportError:
            logger.warning("  sklearn not installed — Classifier B skipped")
            self._coef = None
            return

        self._selected_feats = [f for f in top_features[:LR_MAX_FEATURES] if f in df.columns]
        if len(self._selected_feats) < 2:
            logger.warning("  Classifier B: fewer than 2 features available")
            return

        X = df[self._selected_feats].fillna(0.0).astype(float).values
        y = (df["label"] == LABEL_TC).astype(int).values

        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)
        self._feat_means = dict(zip(self._selected_feats, scaler.mean_))
        self._feat_stds  = dict(zip(self._selected_feats, scaler.scale_))

        lr = LogisticRegression(
            penalty="l1", C=LR_C, solver="liblinear",
            class_weight="balanced", max_iter=1000, random_state=42,
        )
        lr.fit(Xs, y)
        self._coef      = lr.coef_[0]
        self._intercept = lr.intercept_[0]

        logger.debug(f"    LR coefficients: {dict(zip(self._selected_feats, self._coef.round(3)))}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._coef is None or not self._selected_feats:
            return np.zeros(len(df), dtype=int)
        X = df[self._selected_feats].fillna(0.0).astype(float).values
        # Manual standardisation
        means = np.array([self._feat_means.get(f, 0.0) for f in self._selected_feats])
        stds  = np.array([self._feat_stds.get(f, 1.0)  for f in self._selected_feats])
        Xs    = (X - means) / np.maximum(stds, 1e-9)
        logit = Xs @ self._coef + self._intercept
        prob  = 1.0 / (1.0 + np.exp(-logit))
        return (prob >= self._threshold).astype(int)

    def description(self) -> str:
        if self._coef is None:
            return "(sklearn not available)"
        parts = [f"{f}: {c:.3f}" for f, c in zip(self._selected_feats, self._coef)]
        return f"threshold={self._threshold}  |  " + ", ".join(parts)


class ClassifierC_AdditiveScore(_ClassifierBase):
    """
    Additive score: each of 8 binary conditions contributes 1 point.
    Threshold: score >= SCORE_THRESHOLD points = TREND_CONTINUATION.
    Conditions are thresholded from the top 8 features (by correlation)
    using the individual optimal thresholds found on training data.
    """
    name = "C_AdditiveScore"

    def __init__(self):
        self.conditions: List[Tuple[str, float, str]] = []
        self._threshold = SCORE_THRESHOLD

    def fit(self, df: pd.DataFrame, top_features: List[str]) -> None:
        binary = (df["label"] == LABEL_TC).astype(float)
        top8   = [f for f in top_features if f in df.columns][:SCORE_N_CONDITIONS]
        self.conditions = []

        for feat in top8:
            x = df[feat].astype(float)
            best_f1 = -1.0
            best_thr = float(x.median())
            best_dir = ">"
            vals = sorted(x.dropna().unique())
            if len(vals) < 2:
                continue
            cands = np.percentile(vals, np.linspace(10, 90, 20))
            for thr in cands:
                for direction in (">", "<"):
                    pred = (x >= thr).astype(float) if direction == ">" else (x <= thr).astype(float)
                    tp   = ((pred == 1) & (binary == 1)).sum()
                    fp   = ((pred == 1) & (binary == 0)).sum()
                    fn   = ((pred == 0) & (binary == 1)).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    if f1 > best_f1:
                        best_f1, best_thr, best_dir = f1, thr, direction
            self.conditions.append((feat, best_thr, best_dir))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df))
        for feat, thr, direction in self.conditions:
            if feat not in df.columns:
                continue
            if direction == ">":
                scores += (df[feat].astype(float) >= thr).values.astype(float)
            else:
                scores += (df[feat].astype(float) <= thr).values.astype(float)
        return (scores >= self._threshold).astype(int)

    def description(self) -> str:
        parts = [f"{f} {d} {t:.4f}" for f, t, d in self.conditions]
        return f"threshold={self._threshold}/{SCORE_N_CONDITIONS}  |  " + "  ".join(parts)


def train_classifiers(
    train_df:     pd.DataFrame,
    top_features: List[str],
) -> List[_ClassifierBase]:
    logger.info("Step 6: Training 3 classifiers on TRAIN period...")
    clfs = [ClassifierA_RuleBased(), ClassifierB_Logistic(), ClassifierC_AdditiveScore()]
    for clf in clfs:
        clf.fit(train_df, top_features)
        logger.info(f"  {clf.name}: {clf.description()}")
    return clfs


# ---------------------------------------------------------------------------
# Step 7: Validate and select best classifier
# ---------------------------------------------------------------------------

def _eval_clf(clf: _ClassifierBase, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred_pos": 0, "n_true_pos": 0, "n_total": 0}
    y_true = (df["label"] == LABEL_TC).astype(int).values
    y_pred = clf.predict(df)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "n_pred_pos": tp + fp, "n_true_pos": tp, "n_total": len(df),
    }


def _pf_of_subset(df: pd.DataFrame, mask: np.ndarray) -> float:
    """Profit factor of entries selected by boolean mask, using fwd_ret_pct."""
    if "fwd_ret_pct" not in df.columns:
        return float("nan")
    subset = df[mask.astype(bool)]["fwd_ret_pct"].values
    wins   = subset[subset > 0].sum()
    losses = abs(subset[subset < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 3)


def _print_confusion_matrix(clf_name: str, tp: int, fp: int, fn: int, tn: int,
                             prec: float, rec: float, f1: float) -> None:
    logger.info(f"\n  Confusion matrix — {clf_name}:")
    logger.info(f"  {'':30s}  Predicted TC  Predicted OTHER")
    logger.info(f"  {'Actual TC':30s}  {tp:>11}   {fn:>14}")
    logger.info(f"  {'Actual OTHER':30s}  {fp:>11}   {tn:>14}")
    logger.info(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


def validate_and_select(
    clfs:      List[_ClassifierBase],
    val_df:    pd.DataFrame,
    train_df:  pd.DataFrame,
) -> Tuple[_ClassifierBase, dict]:
    """Evaluate all classifiers on val set. Return best by F1 score.
    Also reports blocked vs allowed PF on training data to detect inverted classifiers."""
    logger.info("Step 7: Validating classifiers on VAL period...")
    best_clf = clfs[0]
    best_m   = {"f1": -1.0}
    rows = []
    for clf in clfs:
        m = _eval_clf(clf, val_df)
        y_true = (val_df["label"] == LABEL_TC).astype(int).values
        y_pred = clf.predict(val_df)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        _print_confusion_matrix(clf.name, tp, fp, fn, tn,
                                m["precision"], m["recall"], m["f1"])

        # PF of allowed vs blocked entries ON TRAINING DATA
        if not train_df.empty and "fwd_ret_pct" in train_df.columns:
            tr_pred    = clf.predict(train_df)
            allowed_pf = _pf_of_subset(train_df, tr_pred == 1)
            blocked_pf = _pf_of_subset(train_df, tr_pred == 0)
            inversion  = (
                not math.isnan(allowed_pf) and not math.isnan(blocked_pf)
                and blocked_pf > allowed_pf
            )
            inv_tag = "  *** INVERTED — blocked PF > allowed PF ***" if inversion else ""
            logger.info(f"  Train-data PF: allowed={allowed_pf:.3f}  blocked={blocked_pf:.3f}{inv_tag}")
            if inversion:
                logger.warning(
                    f"  ⚠ {clf.name}: classifier is filtering GOOD trades and keeping BAD ones. "
                    f"Rejected regardless of F1."
                )
        else:
            inversion = False

        rows.append({**{"classifier": clf.name, "inverted": inversion}, **m})
        logger.info(
            f"  {clf.name:25s}  precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"n_pred={m['n_pred_pos']}  n_total={m['n_total']}"
        )
        ok_prec = m["precision"] >= PREC_THRESHOLD
        ok_rec  = m["recall"]    >= RECALL_THRESHOLD
        # Only select if thresholds met AND not inverted
        if m["f1"] > best_m["f1"] and ok_prec and ok_rec and not inversion:
            best_clf = clf
            best_m   = m

    if best_m["f1"] < 0:
        # No classifier meets thresholds — pick best F1 anyway and warn
        best_clf = max(clfs, key=lambda c: _eval_clf(c, val_df)["f1"])
        best_m   = _eval_clf(best_clf, val_df)
        logger.warning(
            f"  ⚠ No classifier meets precision>={PREC_THRESHOLD} AND "
            f"recall>={RECALL_THRESHOLD} on val set."
        )

    logger.info(f"\n  Selected: {best_clf.name} "
                f"(val precision={best_m['precision']:.3f}, "
                f"recall={best_m['recall']:.3f}, F1={best_m['f1']:.3f})")
    return best_clf, best_m


# ---------------------------------------------------------------------------
# Step 8: One-shot test evaluation
# ---------------------------------------------------------------------------

# Test period baseline from V5.0 (ungated Combo C trend module test results)
_UNGATED_TEST_PF  = 0.287   # V5.0 test period PF for ungated trend entries
_UNGATED_TEST_N   = 7       # V5.0 ungated test period trade count
_TEST_PF_REQUIRED = 1.10    # minimum acceptable gated test PF
_TEST_WFE_REQUIRED= 0.60    # minimum acceptable Walk-Forward Efficiency


def _classify_test_failure(test_df: pd.DataFrame, clf: _ClassifierBase) -> str:
    """Heuristic: classify why the test period failed."""
    if test_df.empty or "fwd_ret_pct" not in test_df.columns:
        return "insufficient_data"

    y_pred = clf.predict(test_df)
    allowed = test_df[y_pred == 1].copy()
    if allowed.empty:
        return "classifier_blocks_all"

    # Regime mismatch: are losses concentrated in high SPY DD bars?
    if "spy_dd_from_high" in allowed.columns:
        losers = allowed[allowed["fwd_ret_pct"] < 0]
        if len(losers) > 0 and len(losers) >= 0.7 * len(allowed):
            corrective_losers = losers[losers["spy_dd_from_high"] < -0.05]
            if len(corrective_losers) >= 0.7 * len(losers):
                return "regime_mismatch"

    # Instrument mismatch: single-stock losses dominating
    if "symbol" in allowed.columns:
        losers = allowed[allowed["fwd_ret_pct"] < 0]
        single_stocks = ["NVDA", "META"]
        if losers["symbol"].isin(single_stocks).sum() >= 0.7 * len(losers):
            return "instrument_mismatch_single_stock"

    # Exit mechanism: if most entries are profitable in the short term but exits cut them
    # (proxy: mean fwd_ret > 0 but PF < 1 — suggests exit is cutting wins short)
    if allowed["fwd_ret_pct"].mean() > 0 and _pf_of_subset(allowed, pd.Series(True, index=allowed.index).values) < 1.0:
        return "exit_mechanism"

    return "other"


def write_suspension_report(
    clf:        _ClassifierBase,
    val_metrics: dict,
    test_result: dict,
    test_df:    pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """Write TREND_MODULE_SUSPENSION_REPORT.txt. Returns path written."""
    from datetime import date as _date
    today      = _date.today().isoformat()
    reopen     = _date.fromordinal(
        _date.today().toordinal() + 365
    ).isoformat()

    failure_class = _classify_test_failure(test_df, clf)
    future_dir_map = {
        "regime_mismatch":
            "Leading indicators of regime change (not lagging). Consider SPY-drawdown gate "
            "as simple alternative to complex classifier.",
        "instrument_mismatch_single_stock":
            "Restrict trend universe to ETFs only (QQQ, IWM, XLK). One retest permitted "
            "with ETF-only universe before full suspension.",
        "exit_mechanism":
            "Replace EMA21 trailing stop with ATR-based trail (2× ATR below rolling highest "
            "close since entry). New architecture requires full retest cycle.",
        "classifier_blocks_all":
            "Classifier threshold too strict — no signal passes. Relax SCORE_THRESHOLD or "
            "PREC_THRESHOLD and retrain. Do NOT use same test set.",
        "insufficient_data":
            "Structural data limitation — insufficient test entries for reliable evaluation. "
            "Accumulate more live data before next research cycle.",
        "other":
            "Structural limitation on equity trend-following — pursue non-equity "
            "diversification (fixed income MR, volatility MR, commodity trend).",
    }
    future_direction = future_dir_map.get(failure_class, future_dir_map["other"])

    report = f"""TREND MODULE SUSPENSION REPORT
{'='*60}
Date              : {today}
Version           : V5.2
Test PF achieved  : {test_result.get('gated_pf', float('nan')):.3f}  (required >= {_TEST_PF_REQUIRED})
WFE achieved      : {test_result.get('wfe', float('nan')):.3f}  (required >= {_TEST_WFE_REQUIRED})
Precision         : {test_result['precision']:.3f}  (required > {PREC_THRESHOLD})
Recall            : {test_result['recall']:.3f}  (required > {RECALL_THRESHOLD})

Classifier selected    : {clf.name}
Classifier description : {clf.description()}
Val F1                 : {val_metrics['f1']:.3f}
Val precision          : {val_metrics['precision']:.3f}
Val recall             : {val_metrics['recall']:.3f}

Failure classification : {failure_class}

Key finding:
  The regime classifier (V5.2, {clf.name}) did not meet the acceptance criteria
  on the held-out test set. This evaluation ran exactly once and the result is final.
  The test set was not examined before this evaluation.

Recommended future approach:
  {future_direction}

Earliest appropriate reopen date: {reopen}
  (12-month minimum enforced — regimes persist; wait for regime change before retrying)

Decision:
  - Do NOT deploy trend module
  - Do NOT combine with Combo C live trading
  - Continue Combo C + Kelly ramp only
  - Reopen this research track no earlier than {reopen}
{'='*60}
"""
    path = output_path or str(HERE / "TREND_MODULE_SUSPENSION_REPORT.txt")
    with open(path, "w") as f:
        f.write(report)
    logger.info(f"  Suspension report written: {path}")
    return path


def test_one_shot(
    clf:         _ClassifierBase,
    test_df:     pd.DataFrame,
    train_df:    pd.DataFrame,
) -> dict:
    """Evaluate selected classifier on test set EXACTLY ONCE."""
    logger.info(f"\nStep 8: ONE-SHOT TEST EVALUATION — {clf.name}")
    logger.info("  !! This is the single allowed test evaluation. No iteration. !!")

    m = _eval_clf(clf, test_df)
    prec_ok = m["precision"] >= PREC_THRESHOLD
    rec_ok  = m["recall"]    >= RECALL_THRESHOLD
    passes  = prec_ok and rec_ok

    # PF of gated test entries (uses fwd_ret_pct as proxy returns)
    y_pred_test = clf.predict(test_df)
    gated_pf    = _pf_of_subset(test_df, y_pred_test == 1)
    ungated_pf  = _pf_from_series(test_df["fwd_ret_pct"]) if "fwd_ret_pct" in test_df.columns else float("nan")

    # WFE: gated_test_PF / train_allowed_PF
    # (how much of train edge survives to test after gating)
    y_pred_train = clf.predict(train_df)
    train_pf_allowed = _pf_of_subset(train_df, y_pred_train == 1) if not train_df.empty else float("nan")
    wfe = (gated_pf / train_pf_allowed
           if (not math.isnan(gated_pf) and not math.isnan(train_pf_allowed)
               and train_pf_allowed > 0)
           else float("nan"))

    pf_ok  = (not math.isnan(gated_pf))  and gated_pf  >= _TEST_PF_REQUIRED
    wfe_ok = (not math.isnan(wfe))       and wfe        >= _TEST_WFE_REQUIRED
    passes = prec_ok and rec_ok and pf_ok and wfe_ok

    status = "PASS" if passes else "FAIL"
    logger.info(
        f"\n  ┌─ TEST RESULT ({status}) ──────────────────────────────────────\n"
        f"  │  Classifier       : {clf.name}\n"
        f"  │  Gated N (test)   : {m['n_pred_pos']} / {m['n_total']} entries\n"
        f"  │  Ungated N (base) : {_UNGATED_TEST_N} (V5.0 reference)\n"
        f"  │  Precision        : {m['precision']:.3f}  (>{PREC_THRESHOLD})  "
        f"{'OK' if prec_ok else 'FAIL'}\n"
        f"  │  Recall           : {m['recall']:.3f}  (>{RECALL_THRESHOLD})  "
        f"{'OK' if rec_ok else 'FAIL'}\n"
        f"  │  F1 score         : {m['f1']:.3f}\n"
        f"  │  Gated PF         : {gated_pf:.3f}  (>={_TEST_PF_REQUIRED})  "
        f"{'OK' if pf_ok else 'FAIL'}\n"
        f"  │  Ungated PF (ref) : {ungated_pf:.3f}  (V5.0 test = {_UNGATED_TEST_PF})\n"
        f"  │  Train allowed PF : {train_pf_allowed:.3f}\n"
        f"  │  WFE              : {wfe:.3f}  (>={_TEST_WFE_REQUIRED})  "
        f"{'OK' if wfe_ok else 'FAIL'}\n"
        f"  └──────────────────────────────────────────────────────────────"
    )

    if not passes:
        logger.warning(
            "\n  ╔══════════════════════════════════════════════════════════════╗\n"
            "  ║  TREND MODULE TEST FAILED — SUSPEND RESEARCH                ║\n"
            "  ║                                                              ║\n"
            "  ║  Regime classifier does not meet acceptance criteria on      ║\n"
            "  ║  held-out test data. Per V5.2 rules:                        ║\n"
            "  ║                                                              ║\n"
            "  ║  1. Do NOT deploy trend module in any form                  ║\n"
            "  ║  2. Do NOT combine with Combo C live trading                ║\n"
            "  ║  3. Continue live trading with Combo C + Kelly ramp only    ║\n"
            "  ║  4. Revisit in 12 months minimum (regime persistence)       ║\n"
            "  ╚══════════════════════════════════════════════════════════════╝"
        )

    return {**m, "passes": passes,
            "gated_pf": gated_pf, "ungated_pf": ungated_pf,
            "train_pf_allowed": train_pf_allowed, "wfe": wfe}


# ---------------------------------------------------------------------------
# Step 9: Correlation check — gated trend vs Combo C equity
# ---------------------------------------------------------------------------

# Combo C instrument universe for overlap analysis
try:
    _COMBO_C_SYMS_SET = set(COMBO_C_SYMBOLS)
except Exception:
    _COMBO_C_SYMS_SET = {"GLD", "WMT", "USMV", "NVDA", "AMZN", "GOOGL", "COST", "XOM", "HD", "MA"}


def correlation_check(
    clf:        _ClassifierBase,
    all_df:     pd.DataFrame,   # all labelled entries (no period filter)
    data:       Dict[str, pd.DataFrame],
    combo_c_trades_path: Optional[str] = None,
) -> None:
    """
    Build a synthetic equity curve for the regime-gated trend module
    and compare to Combo C equity curve (loaded from trade log CSV if provided).
    Reports Pearson correlation + instrument overlap analysis.
    """
    logger.info("\nStep 9: Correlation check — gated trend vs Combo C...")

    # --- Instrument overlap analysis ----------------------------------------
    trend_syms  = set(s for s in TREND_UNIVERSE if s != SPY_SYM)
    overlap     = trend_syms & _COMBO_C_SYMS_SET
    if overlap:
        logger.warning(
            f"  Instrument overlap detected: {sorted(overlap)} appear in both "
            f"Combo C and trend universes. This inflates portfolio correlation."
        )
    else:
        logger.info("  No instrument overlap between Combo C and trend universes.")

    # Gated trend equity: all entries where classifier predicts TC
    y_pred = clf.predict(all_df)
    gated  = all_df[y_pred == 1].copy()

    if gated.empty:
        logger.warning("  No gated entries — correlation not computable")
        return

    # Use fwd_ret_pct as equity contribution per trade (1 unit equity, no sizing)
    if "fwd_ret_pct" not in gated.columns:
        logger.warning("  fwd_ret_pct column missing — run Step 4 first")
        return

    gated_sorted  = gated.sort_values("date")
    trend_cum_pnl = gated_sorted["fwd_ret_pct"].cumsum().reset_index(drop=True)

    logger.info(f"  Gated entries: {len(gated)}  mean fwd ret: "
                f"{gated['fwd_ret_pct'].mean():.2f}%  "
                f"PF proxy: {_pf_from_series(gated['fwd_ret_pct']):.3f}")

    def _report_correlation(trend_curve: pd.Series, cc_curve: pd.Series,
                            label: str) -> float:
        min_len = min(len(trend_curve), len(cc_curve))
        if min_len < 5:
            logger.warning(f"  Too few points ({min_len}) for {label} correlation")
            return float("nan")
        c = float(np.corrcoef(trend_curve.iloc[:min_len].values,
                              cc_curve.iloc[:min_len].values)[0, 1])
        tag = "✓ PASS" if abs(c) < 0.30 else ("⚠ WATCH" if abs(c) < 0.50 else "✗ FAIL")
        logger.info(f"  Correlation [{label}]: {c:.3f}  {tag}")
        return c

    # Load Combo C trades if available
    if combo_c_trades_path and Path(combo_c_trades_path).exists():
        try:
            cc_df = pd.read_csv(combo_c_trades_path)
            if "net_pnl" in cc_df.columns and "exit_date" in cc_df.columns:
                cc_df     = cc_df.sort_values("exit_date")
                cc_cum    = cc_df["net_pnl"].cumsum().reset_index(drop=True)

                # Full-period correlation
                corr_full = _report_correlation(trend_cum_pnl, cc_cum, "full period")

                # Overlap-excluded correlation (recompute gated curve without overlap syms)
                if overlap and "symbol" in gated_sorted.columns:
                    gated_no_overlap = gated_sorted[
                        ~gated_sorted["symbol"].isin(overlap)
                    ]["fwd_ret_pct"].cumsum().reset_index(drop=True)
                    corr_no_overlap = _report_correlation(
                        gated_no_overlap, cc_cum, "overlap-excluded"
                    )
                    if corr_full >= 0.30 and corr_no_overlap < 0.30:
                        logger.info(
                            f"  → Instrument overlap is the primary correlation driver. "
                            f"Removing {sorted(overlap)} from trend universe drops correlation "
                            f"from {corr_full:.3f} to {corr_no_overlap:.3f}. "
                            f"Portfolio integration may proceed with separated universes."
                        )
                    elif corr_full >= 0.30 and corr_no_overlap >= 0.30:
                        logger.warning(
                            f"  → Structural correlation remains ({corr_no_overlap:.3f}) even "
                            f"after removing overlapping instruments. Both strategies are "
                            f"long-equity — combination does not provide genuine diversification."
                        )

                if not math.isnan(corr_full):
                    if abs(corr_full) < 0.30:
                        logger.info("  ✓ Full-period correlation < 0.30 — safe to consider portfolio combination")
                    else:
                        logger.warning("  ✗ Full-period correlation >= 0.30 — do NOT combine without further analysis")
        except Exception as exc:
            logger.warning(f"  Could not load Combo C trades: {exc}")
    else:
        logger.info("  (Combo C trades not provided — use --combo-c-trades <path> for correlation)")


def _pf_from_series(s: pd.Series) -> float:
    wins   = s[s > 0].sum()
    losses = abs(s[s < 0].sum())
    return round(wins / losses, 3) if losses > 0 else float("inf")


# ---------------------------------------------------------------------------
# Walk-forward split on entry DataFrame
# ---------------------------------------------------------------------------

def split_entries(
    df:         pd.DataFrame,
    data:       Dict[str, pd.DataFrame],
    train_pct:  float = TRAIN_PCT,
    val_pct:    float = VAL_PCT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labelled entry DataFrame into train/val/test using the same
    chronological boundary dates as walk_forward_split on the underlying data.
    """
    # Get boundary dates from one representative symbol
    ref_sym = next((s for s in TREND_UNIVERSE if s != SPY_SYM and s in data), None)
    if ref_sym is None:
        logger.error("  No reference symbol for date split")
        return df, pd.DataFrame(), pd.DataFrame()

    ref_df = data[ref_sym]
    n      = len(ref_df)
    t_end  = ref_df.index[int(n * train_pct) - 1]
    v_end  = ref_df.index[int(n * (train_pct + val_pct)) - 1]

    t_date = t_end.date() if hasattr(t_end, "date") else t_end
    v_date = v_end.date() if hasattr(v_end, "date") else v_end

    train = df[df["date"] <= t_date]
    val   = df[(df["date"] > t_date) & (df["date"] <= v_date)]
    test  = df[df["date"] > v_date]

    logger.info(
        f"\n  Entry split ({ref_sym} reference):\n"
        f"    Train : {len(train):3d} entries  (→ {t_date})\n"
        f"    Val   : {len(val):3d}   entries  ({t_date} → {v_date})\n"
        f"    Test  : {len(test):3d}  entries  ({v_date} →)"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_feature_importance(fi: pd.DataFrame) -> None:
    logger.info("\nStep 5: Feature importance (point-biserial corr vs TREND_CONTINUATION):")
    logger.info(f"  {'Rank':>4}  {'Feature':<25}  {'Corr':>8}  {'|Corr|':>8}")
    logger.info(f"  {'----':>4}  {'-'*25}  {'--------':>8}  {'--------':>8}")
    for i, row in fi.iterrows():
        logger.info(f"  {i+1:>4}  {row['feature']:<25}  {row['corr']:>8.4f}  {row['abs_corr']:>8.4f}")


def print_final_summary(
    best_clf:    _ClassifierBase,
    val_metrics: dict,
    test_result: dict,
) -> None:
    sep  = "═" * 78
    sep2 = "─" * 78
    print(f"\n{sep}")
    print(f"  TREND MODULE REGIME GATE — RESEARCH SUMMARY (V5.1)")
    print(f"{sep}")
    print(f"  Forward return label window : {FWD_BARS} bars")
    print(f"  TREND_CONTINUATION threshold: fwd return > {FWD_TREND_CONT_PCT:.1f}%")
    print(f"  TREND_EXHAUSTION threshold  : fwd return < {FWD_TREND_EXHA_PCT:.1f}%")
    print(f"  Acceptance criteria         : precision > {PREC_THRESHOLD} AND recall > {RECALL_THRESHOLD}")
    print(f"{sep2}")
    print(f"  Selected classifier : {best_clf.name}")
    print(f"  Description         : {best_clf.description()}")
    print(f"{sep2}")
    print(f"  Validation period:")
    print(f"    Precision : {val_metrics['precision']:.3f}")
    print(f"    Recall    : {val_metrics['recall']:.3f}")
    print(f"    F1        : {val_metrics['f1']:.3f}")
    print(f"{sep2}")
    status = "PASS ✓" if test_result["passes"] else "FAIL ✗"
    print(f"  Test period (ONE-SHOT)  [{status}]:")
    print(f"    Precision     : {test_result['precision']:.3f}  "
          f"({'OK' if test_result['precision'] >= PREC_THRESHOLD else 'BELOW threshold'})")
    print(f"    Recall        : {test_result['recall']:.3f}  "
          f"({'OK' if test_result['recall'] >= RECALL_THRESHOLD else 'BELOW threshold'})")
    print(f"    F1            : {test_result['f1']:.3f}")
    gated_pf = test_result.get('gated_pf', float('nan'))
    wfe      = test_result.get('wfe', float('nan'))
    print(f"    Gated PF      : {gated_pf:.3f}  "
          f"({'OK' if not math.isnan(gated_pf) and gated_pf >= _TEST_PF_REQUIRED else 'BELOW threshold'})")
    print(f"    WFE           : {wfe:.3f}  "
          f"({'OK' if not math.isnan(wfe) and wfe >= _TEST_WFE_REQUIRED else 'BELOW threshold'})")
    print(f"    Ungated PF ref: {_UNGATED_TEST_PF} (V5.0 baseline)")
    print(f"{sep2}")
    if test_result["passes"]:
        print(f"  NEXT STEP: Integrate regime gate into combo_trend_entry (Step 10).")
        print(f"             Verify portfolio correlation < 0.30 vs Combo C before")
        print(f"             combining in live trading.")
    else:
        print(f"  DECISION : SUSPEND trend research. Continue Combo C + Kelly only.")
        print(f"             Revisit when 6+ months of live Combo C data available.")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Minimal helper classes (avoid numpy import at top-level for indicator engine)
# ---------------------------------------------------------------------------

class _EWM:
    __slots__ = ("alpha", "value", "_n")
    def __init__(self, span: int):
        self.alpha = 2.0 / (span + 1.0)
        self.value = 0.0
        self._n    = 0
    def update(self, x: float) -> float:
        if self._n == 0: self.value = x
        else:            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        self._n += 1
        return self.value


class _RollingBuf:
    from collections import deque as _deque
    def __init__(self, size: int):
        from collections import deque
        self._maxlen = size
        self._d = deque(maxlen=size)
    def append(self, x): self._d.append(x)
    def full(self):      return len(self._d) == self._maxlen
    def mean(self):      return sum(self._d) / len(self._d) if self._d else 0.0
    def max(self):       return max(self._d) if self._d else 0.0
    def oldest(self):    return self._d[0] if self._d else 0.0
    def __len__(self):   return len(self._d)
    def __iter__(self):  return iter(self._d)


def _linreg(values: List[float]) -> Tuple[float, float, float, float]:
    """
    Compute OLS linear regression on values (index 0..n-1).
    Returns (slope, R², value_at_last_x, residual_std).
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0, values[-1] if values else 0.0, 0.0
    xs = list(range(n))
    xm = (n - 1) / 2.0
    ym = sum(values) / n
    ssxx = sum((x - xm) ** 2 for x in xs)
    ssxy = sum((x - xm) * (y - ym) for x, y in zip(xs, values))
    ssy  = sum((y - ym) ** 2 for y in values)

    slope = ssxy / ssxx if ssxx > 0 else 0.0
    intercept = ym - slope * xm
    y_hat_last = slope * (n - 1) + intercept
    r2  = (ssxy ** 2) / (ssxx * ssy) if (ssxx * ssy) > 0 else 0.0

    residuals = [v - (slope * i + intercept) for i, v in enumerate(values)]
    res_std   = math.sqrt(sum(r ** 2 for r in residuals) / n) if n > 1 else 0.0

    return slope, r2, y_hat_last, res_std


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="V5.1 Trend Module Regime Gate Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alpaca",          action="store_true",
                   help="Fetch data from Alpaca API")
    p.add_argument("--daily",           action="store_true",
                   help="Resample intraday data to daily bars (recommended)")
    p.add_argument("--no-cache",        action="store_true",
                   help="Force re-download even if cache exists")
    p.add_argument("--step",            default=None,
                   help="Run only steps N-M (e.g. '1-4' or '5')")
    p.add_argument("--combo-c-trades",  default=None, metavar="CSV_PATH",
                   help="Path to Combo C trade CSV for correlation check (Step 9)")
    return p.parse_args()


def _step_filter(args) -> Optional[set]:
    raw = args.step
    if raw is None:
        return None
    result = set()
    for part in str(raw).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def should_run(step_n: int, step_filter: Optional[set]) -> bool:
    return step_filter is None or step_n in step_filter


def main():
    args        = parse_args()
    step_filter = _step_filter(args)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    if should_run(1, step_filter):
        data = load_trend_data(args.alpaca, args.daily, args.no_cache)
    else:
        logger.info("Step 1: skipped (not in --step filter)")
        return

    # ── Step 2+3: Features + entry bars ──────────────────────────────────
    if should_run(2, step_filter) and should_run(3, step_filter):
        entry_df = build_feature_matrix(data)
        if entry_df.empty:
            logger.error("No entry bars found — check data and indicator warmup")
            sys.exit(1)
    else:
        logger.info("Steps 2-3: skipped")
        return

    # ── Step 4: Labels ────────────────────────────────────────────────────
    if should_run(4, step_filter):
        labelled_df = assign_labels(entry_df, data)
        if labelled_df.empty or len(labelled_df) < 10:
            logger.error("Insufficient labelled entries — cannot train classifiers")
            sys.exit(1)
    else:
        logger.info("Step 4: skipped")
        return

    # Walk-forward split on entry DataFrame
    train_df, val_df, test_df = split_entries(labelled_df, data)

    if len(train_df) < 5:
        logger.error(f"Training set too small: {len(train_df)} entries — need >= 5")
        sys.exit(1)

    # ── Step 5: Feature importance (train only) ───────────────────────────
    if should_run(5, step_filter):
        fi = feature_importance(train_df)
        print_feature_importance(fi)
        top_features = fi["feature"].tolist()
    else:
        logger.info("Step 5: skipped — using default feature order")
        top_features = FEATURE_COLS
        fi = None

    # ── Step 6: Train classifiers ─────────────────────────────────────────
    if should_run(6, step_filter):
        clfs = train_classifiers(train_df, top_features)
    else:
        logger.info("Step 6: skipped")
        return

    # ── Step 7: Validate and select ───────────────────────────────────────
    if should_run(7, step_filter):
        best_clf, val_metrics = validate_and_select(clfs, val_df, train_df)
    else:
        logger.info("Step 7: skipped")
        return

    # ── Step 8: One-shot test (RUNS UNCONDITIONALLY if step_filter allows) ──
    if should_run(8, step_filter):
        test_result = test_one_shot(best_clf, test_df, train_df)
        if not test_result.get("passes", False):
            write_suspension_report(best_clf, val_metrics, test_result, test_df)
    else:
        logger.info("Step 8: skipped (test evaluation not run)")
        test_result = {"passes": False, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                       "gated_pf": float("nan"), "wfe": float("nan")}

    # ── Step 9: Correlation check ─────────────────────────────────────────
    if should_run(9, step_filter) and test_result.get("passes", False):
        correlation_check(best_clf, labelled_df, data, args.combo_c_trades)
    elif not test_result.get("passes", False):
        logger.info("Step 9: skipped — test failed, no correlation check needed")

    # ── Final summary ─────────────────────────────────────────────────────
    print_final_summary(best_clf, val_metrics, test_result)

    sys.exit(0 if test_result.get("passes", False) else 2)


if __name__ == "__main__":
    main()
