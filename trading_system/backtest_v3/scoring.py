"""
scoring.py -- V5.0 Module 1: Entry Quality Scoring System
==========================================================
Computes a composite quality score (0–100) for every Combo C entry.
Score is computed from data available at signal bar close — no lookahead.

Five components:
  C1  RSI(2) extremity          0–25 pts
  C2  BB penetration depth      0–25 pts  (measured in BB std-dev units)
  C3  Mean-reversion distance   0–20 pts  (entry-to-target in ATR units)
  C4  Instrument beta           0–15 pts  (lower beta = stronger reversion)
  C5  Broad market context      0–15 pts  (SPY RSI14 zone)

Score → position size multiplier:
  80–100  1.5× standard   (high conviction)
  60–79   1.0× standard   (standard entry)
  40–59   0.75× standard  (marginal)
  < 40    0.5× standard   (low conviction)

Cap: 10% of equity regardless of multiplier.

Validation hypothesis:
  High-score (≥60) entries should show PF at least 0.15 higher than
  low-score (<60) entries over 30+ live trades. If no meaningful PF
  differential exists, discard scoring and revert to fixed sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from indicators_v3 import BarSnapshot


# ---------------------------------------------------------------------------
# Scoring breakpoints
# ---------------------------------------------------------------------------

def score_rsi2(rsi2: float) -> int:
    """Component 1: RSI(2) extremity (0–25 pts)."""
    if rsi2 < 3.0:
        return 25
    if rsi2 < 5.0:
        return 20
    if rsi2 < 8.0:
        return 15
    if rsi2 < 11.0:
        return 10
    if rsi2 <= 15.0:
        return 5
    return 0   # above qualifying threshold — should not reach here for valid entry


def score_bb_penetration(penetration_sd: float) -> int:
    """
    Component 2: BB penetration depth (0–25 pts).
    penetration_sd = (bb_lower - close) / bb_std_dev
    Positive = close is below lower band; negative = above (should not happen at entry).
    """
    if penetration_sd > 0.5:
        return 25
    if penetration_sd > 0.3:
        return 18
    if penetration_sd > 0.1:
        return 10
    if penetration_sd >= 0.0:
        return 5   # barely below band
    return 0       # above lower band (degenerate — entry should not fire)


def score_target_distance(target_dist_atr: float) -> int:
    """
    Component 3: Mean-reversion distance (0–20 pts).
    target_dist_atr = (bb_mid - close) / ATR(10)
    """
    if target_dist_atr > 3.0:
        return 20
    if target_dist_atr > 2.0:
        return 15
    if target_dist_atr > 1.5:
        return 10
    if target_dist_atr > 1.0:
        return 5
    return 0   # < 1.0× ATR: small target — flag but don't auto-skip


def score_beta(beta_60: float) -> int:
    """Component 4: Instrument beta at signal time (0–15 pts)."""
    if beta_60 < 0.3:
        return 15
    if beta_60 < 0.5:
        return 12
    if beta_60 < 0.7:
        return 8
    if beta_60 < 0.8:
        return 4
    return 0   # ≥ 0.8: ineligible for Combo C anyway


def score_spy_context(spy_rsi14: float) -> int:
    """
    Component 5: Broad market context via SPY RSI(14) (0–15 pts).
    Mean-reversion edge is strongest when the market is neutral (40–60).
    """
    if spy_rsi14 > 75:
        return 5    # overbought market: individual reversion less reliable
    if spy_rsi14 > 60:
        return 10   # market strong: instrument weakness is likely temporary
    if spy_rsi14 >= 40:
        return 15   # neutral market: best mean-reversion conditions
    return 5        # market oversold: broad decline may persist; score low


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

@dataclass
class EntryScore:
    """Full score breakdown for one entry signal."""
    c1_rsi2:           int = 0
    c2_bb_penetration: int = 0
    c3_target_dist:    int = 0
    c4_beta:           int = 0
    c5_spy_context:    int = 0
    total:             int = 0
    size_multiplier:   float = 1.0
    flag_small_target: bool  = False   # target_dist_atr < 1.0× — log separately

    @property
    def band(self) -> str:
        if self.total >= 80:
            return "HIGH"
        if self.total >= 60:
            return "STANDARD"
        if self.total >= 40:
            return "MARGINAL"
        return "LOW"


def compute_quality_score(snap: BarSnapshot, spy_rsi14: float = 50.0) -> EntryScore:
    """
    Compute the full composite quality score for a Combo C entry signal.

    Parameters
    ----------
    snap       : BarSnapshot at signal bar close (all indicators current)
    spy_rsi14  : SPY RSI(14) at signal bar close (fetched from spy_snap_map
                 in the simulator — passed in by the orchestrator)

    Returns
    -------
    EntryScore with all component scores and the size multiplier.

    Note: snap.bb_penetration_sd and snap.target_dist_atr are computed in
    IndicatorStateV3.update() as of V5.0. If snap.bb_std == 0 (warmup period),
    those fields are 0 — score safely degrades to minimum points.
    """
    c1 = score_rsi2(snap.rsi2)
    c2 = score_bb_penetration(snap.bb_penetration_sd)
    c3 = score_target_distance(snap.target_dist_atr)
    c4 = score_beta(snap.beta_60)
    c5 = score_spy_context(spy_rsi14)

    total = c1 + c2 + c3 + c4 + c5

    # Size multiplier mapping
    if total >= 80:
        multiplier = 1.5
    elif total >= 60:
        multiplier = 1.0
    elif total >= 40:
        multiplier = 0.75
    else:
        multiplier = 0.5

    return EntryScore(
        c1_rsi2           = c1,
        c2_bb_penetration = c2,
        c3_target_dist    = c3,
        c4_beta           = c4,
        c5_spy_context    = c5,
        total             = total,
        size_multiplier   = multiplier,
        flag_small_target = snap.target_dist_atr < 1.0 and snap.target_dist_atr > 0,
    )


# ---------------------------------------------------------------------------
# Score band PF comparison (validation gate)
# ---------------------------------------------------------------------------

def compare_score_bands(trades: list) -> dict:
    """
    Given a list of trade dicts or TradeRecord objects with 'quality_score'
    and 'net_pnl' / 'won' fields, compute PF for high (≥60) vs low (<60) bands.

    Acceptance criterion: PF_high - PF_low >= 0.15

    Returns dict with metrics and accept/reject decision.
    """
    def _pf(trades_subset):
        wins  = sum(float(getattr(t, "net_pnl", t.get("net_pnl", 0)) if hasattr(t, "get") else t.net_pnl)
                    for t in trades_subset if (getattr(t, "won", t.get("won", False)) if hasattr(t, "get") else t.won))
        loss  = abs(sum(float(getattr(t, "net_pnl", t.get("net_pnl", 0)) if hasattr(t, "get") else t.net_pnl)
                        for t in trades_subset if not (getattr(t, "won", t.get("won", False)) if hasattr(t, "get") else t.won)))
        return round(wins / loss, 3) if loss > 0 else float("inf")

    def _score(t):
        return (t.get("quality_score", 0) if hasattr(t, "get")
                else getattr(t, "quality_score", 0))

    high = [t for t in trades if _score(t) >= 60]
    low  = [t for t in trades if _score(t) < 60]

    pf_high = _pf(high)
    pf_low  = _pf(low)
    pf_diff = (pf_high - pf_low) if pf_high != float("inf") and pf_low != 0 else None

    accept = (pf_diff is not None and pf_diff >= 0.15 and len(high) >= 10 and len(low) >= 10)

    return {
        "n_high":    len(high),
        "n_low":     len(low),
        "pf_high":   pf_high,
        "pf_low":    pf_low,
        "pf_diff":   round(pf_diff, 3) if pf_diff is not None else None,
        "accept":    accept,
        "decision":  "ACCEPT" if accept else "REJECT — no meaningful PF differential (revert to fixed sizing)",
        "criterion": "PF_high - PF_low >= 0.15 with N ≥ 10 in each band",
    }


# ---------------------------------------------------------------------------
# Pseudocode documentation (for reference)
# ---------------------------------------------------------------------------

PSEUDOCODE = """
MODULE 1 — Entry Quality Scoring System
========================================
At every Combo C signal bar (close < bb_lower AND rsi2 < 15):

  # Component 1: RSI(2) extremity
  if rsi2 < 3:    c1 = 25
  elif rsi2 < 5:  c1 = 20
  elif rsi2 < 8:  c1 = 15
  elif rsi2 < 11: c1 = 10
  else:           c1 = 5

  # Component 2: BB penetration depth
  pen = (bb_lower - close) / bb_std_dev
  if pen > 0.5:   c2 = 25
  elif pen > 0.3: c2 = 18
  elif pen > 0.1: c2 = 10
  else:           c2 = 5

  # Component 3: Mean-reversion distance
  dist = (bb_mid - close) / atr10
  if dist > 3.0:   c3 = 20
  elif dist > 2.0: c3 = 15
  elif dist > 1.5: c3 = 10
  elif dist > 1.0: c3 = 5
  else:            c3 = 0   # flag: small target, consider skipping

  # Component 4: Instrument beta
  if beta < 0.3:   c4 = 15
  elif beta < 0.5: c4 = 12
  elif beta < 0.7: c4 = 8
  elif beta < 0.8: c4 = 4
  else:            c4 = 0

  # Component 5: SPY RSI(14)
  if spy_rsi14 > 75:  c5 = 5
  elif spy_rsi14 > 60: c5 = 10
  elif spy_rsi14 >= 40: c5 = 15
  else:               c5 = 5

  score = c1 + c2 + c3 + c4 + c5

  # Size multiplier
  if score >= 80:   multiplier = 1.5
  elif score >= 60: multiplier = 1.0
  elif score >= 40: multiplier = 0.75
  else:             multiplier = 0.5

  position_size = floor(equity * 0.005 / atr10) * multiplier
  position_size = min(position_size, equity * 0.10 / entry_price)  # 10% cap

State variables to log per trade:
  quality_score, c1_rsi2, c2_bb_pen, c3_target_dist, c4_beta, c5_spy,
  size_multiplier, flag_small_target

Validation gate (after 30+ live trades):
  Compute PF for score >= 60 vs score < 60 bands.
  Accept scoring if PF_high - PF_low >= 0.15 with N >= 10 in each band.
  Reject (revert to fixed sizing) if no meaningful differential.
"""
