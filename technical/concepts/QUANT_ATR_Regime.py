"""
CONCEPT: QUANT ATR Regime Classifier
SOURCE: Internal quantitative extension — informed by ICT time/regime concept
CATEGORY: ANOMALY
TIMEFRAME: 1D, 4H
INSTRUMENTS: ANY
DESCRIPTION:
    Classifies the current market volatility regime using ATR percentile.
    LOW_VOL: current ATR < 25th percentile of rolling ATR window
    NORMAL_VOL: 25th–75th percentile
    HIGH_VOL: > 75th percentile

    This is used as a filter/multiplier by other concepts:
    - In HIGH_VOL regime: widen stops, reduce position size
    - In LOW_VOL regime: tighter stops possible, watch for breakout
    - TRENDING regime = HIGH_VOL + directional momentum
EDGE:
    Volatility clustering is a well-documented market phenomenon. Being aware
    of the current volatility regime prevents using the wrong stop size.
KNOWN_LIMITATIONS:
    ATR percentile is backward-looking. Regime changes can be abrupt.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
ATR_LOOKBACK      = 14    # bars for ATR computation
PERCENTILE_WINDOW = 100   # rolling window for ATR percentile rank
LOW_VOL_PCT       = 25    # percentile below which = LOW_VOL
HIGH_VOL_PCT      = 75    # percentile above which = HIGH_VOL


class QUANT_ATR_Regime:
    """Classifies volatility regime and provides ATR multipliers for other concepts."""

    def __init__(self, params: dict = None):
        p = params or {}
        self.atr_lookback      = p.get("atr_lookback",      ATR_LOOKBACK)
        self.percentile_window = p.get("percentile_window", PERCENTILE_WINDOW)
        self.low_vol_pct       = p.get("low_vol_pct",       LOW_VOL_PCT)
        self.high_vol_pct      = p.get("high_vol_pct",      HIGH_VOL_PCT)
        self._atr_history:     List[float] = []

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        """No directional signal — regime module only. Returns None."""
        return None

    def classify(self, snapshot: BarSnapshot) -> str:
        """Returns 'LOW_VOL', 'NORMAL_VOL', or 'HIGH_VOL'."""
        if snapshot.atr is None:
            return "NORMAL_VOL"
        self._atr_history.append(snapshot.atr)
        if len(self._atr_history) < self.percentile_window:
            return "NORMAL_VOL"
        window = self._atr_history[-self.percentile_window:]
        pct = np.percentile(window, [self.low_vol_pct, self.high_vol_pct])
        if snapshot.atr < pct[0]:
            return "LOW_VOL"
        if snapshot.atr > pct[1]:
            return "HIGH_VOL"
        return "NORMAL_VOL"

    def stop_multiplier(self, snapshot: BarSnapshot) -> float:
        """Returns suggested stop ATR multiplier for current regime."""
        regime = self.classify(snapshot)
        return {"LOW_VOL": 0.75, "NORMAL_VOL": 1.0, "HIGH_VOL": 1.50}.get(regime, 1.0)

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= self.percentile_window else 0
        return ValidationResult("QUANT_ATR_Regime", 1, passed, 1 - passed)

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[QUANT_ATR] %s at %s | %s", event_type, bar, details)
