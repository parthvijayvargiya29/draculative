"""
walk_forward.py — Walk-Forward Validation Engine.

Splits data 60/20/20 (train / validate / test), runs the live simulator
on each slice, and computes WFE (Walk-Forward Efficiency).

Gate criteria (Section 3.4):
  PF_test  ≥  0.90 × PF_train   (degradation ≤ 10%)
  WFE      ≥  0.60               (robust generalisation)

WFE = PF_test / PF_train
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from technical.bar_snapshot import WalkForwardResult, ApprovalStatus, SimulationResult
from simulation.metrics_engine import MetricsEngine, TradeRecord

logger = logging.getLogger(__name__)

TRAIN_FRAC    = 0.60
VALIDATE_FRAC = 0.20
TEST_FRAC     = 0.20

WFE_MIN       = 0.60
PF_DEGRADATION_MAX = 0.10  # PF_test must be ≥ (1 - 0.10) × PF_train


def _profit_factor(trades: List[TradeRecord]) -> float:
    wins   = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    return wins / losses if losses > 0 else float("inf")


@dataclass
class SliceResult:
    label:  str    # "train" | "validate" | "test"
    start:  str
    end:    str
    trades: List[TradeRecord]
    pf:     float
    win_rate: float
    total_trades: int


class WalkForwardValidator:
    """
    Parameters
    ----------
    simulator : LiveSimulator instance
    concepts  : list of concept instances (same ones passed to simulator)
    symbol    : str
    df        : enriched OHLCV DataFrame (already loaded + enriched)
    concept_name : str  (for labelling results)
    """

    def __init__(self, simulator, concepts: list, symbol: str,
                 df: pd.DataFrame, concept_name: str):
        self.simulator    = simulator
        self.concepts     = concepts
        self.symbol       = symbol
        self.df           = df
        self.concept_name = concept_name

    def run(self) -> WalkForwardResult:
        n = len(self.df)
        train_end    = int(n * TRAIN_FRAC)
        validate_end = int(n * (TRAIN_FRAC + VALIDATE_FRAC))

        slices = {
            "train":    (0,          train_end),
            "validate": (train_end,  validate_end),
            "test":     (validate_end, n),
        }

        slice_results: Dict[str, SliceResult] = {}
        for label, (start_i, end_i) in slices.items():
            df_slice = self.df.iloc[start_i:end_i].copy()
            if len(df_slice) < 50:
                logger.warning("Slice '%s' too small (%d bars) — skipping", label, len(df_slice))
                slice_results[label] = SliceResult(
                    label=label,
                    start=str(df_slice.index[0]) if len(df_slice) else "",
                    end=str(df_slice.index[-1]) if len(df_slice) else "",
                    trades=[], pf=0.0, win_rate=0.0, total_trades=0,
                )
                continue

            trades, _ = self.simulator.run(self.symbol, df_slice)
            pf = _profit_factor(trades)
            wr = len([t for t in trades if t.pnl > 0]) / len(trades) if trades else 0.0
            slice_results[label] = SliceResult(
                label  = label,
                start  = str(df_slice.index[0]),
                end    = str(df_slice.index[-1]),
                trades = trades,
                pf     = pf,
                win_rate = wr,
                total_trades = len(trades),
            )
            logger.info("Walk-forward [%s] | %d trades | PF=%.2f | WR=%.1f%%",
                        label, len(trades), pf, wr * 100)

        train_pf = slice_results["train"].pf
        test_pf  = slice_results["test"].pf
        wfe      = test_pf / train_pf if train_pf > 0 and train_pf != float("inf") else 0.0

        passes_gate = (
            test_pf  >= (1 - PF_DEGRADATION_MAX) * train_pf and
            wfe      >= WFE_MIN and
            train_pf > 1.0 and
            test_pf  > 1.0
        )

        all_trades: List[TradeRecord] = []
        for sr in slice_results.values():
            all_trades.extend(sr.trades)

        sim_result = MetricsEngine.compute(
            trades       = all_trades,
            concept_name = self.concept_name,
            data_period  = {
                "train_start": slice_results["train"].start,
                "test_end"   : slice_results["test"].end,
            },
            universe = [self.symbol],
        )
        sim_result.wfe = wfe

        result = WalkForwardResult(
            concept         = self.concept_name,
            symbol          = self.symbol,
            pf_train        = train_pf,
            pf_validate     = slice_results["validate"].pf,
            pf_test         = test_pf,
            wfe             = wfe,
            passes_gate     = passes_gate,
            slice_results   = {k: v.__dict__ for k, v in slice_results.items()},
        )

        if passes_gate:
            sim_result.approval_status  = ApprovalStatus.APPROVED
        else:
            sim_result.approval_status  = ApprovalStatus.REJECTED
            reasons = []
            if test_pf < (1 - PF_DEGRADATION_MAX) * train_pf:
                reasons.append(f"PF degraded too much (train={train_pf:.2f}, test={test_pf:.2f})")
            if wfe < WFE_MIN:
                reasons.append(f"WFE={wfe:.2f} < {WFE_MIN}")
            if train_pf <= 1.0:
                reasons.append(f"Train PF={train_pf:.2f} ≤ 1.0 (not profitable in-sample)")
            sim_result.rejection_reason = "; ".join(reasons)

        result.simulation_result = sim_result
        return result
