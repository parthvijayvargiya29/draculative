"""
regime_classifier.py — Tags each trade entry with TRENDING or CORRECTIVE.

TRENDING  → ADX > 25  AND  SMA50 slope positive (for longs) / negative (for shorts)
CORRECTIVE → everything else

Used by LiveSimulator to populate TradeRecord.regime for per-regime attribution.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ADX_TRENDING_THRESHOLD = 25.0
SLOPE_LOOKBACK         = 5   # bars for SMA50 slope


class RegimeClassifier:
    """
    Stateless classifier. Given a DataFrame of OHLCV bars enriched with
    indicators (adx, sma_50 columns expected), returns a string regime tag
    for the most recent bar.
    """

    def tag(self, df: pd.DataFrame, direction: Optional[str] = None) -> str:
        """
        Parameters
        ----------
        df        : DataFrame enriched with enrich_dataframe(); uses last row.
        direction : "long" | "short" | None — used to determine SMA slope sign.

        Returns
        -------
        "TRENDING" or "CORRECTIVE"
        """
        if df is None or len(df) < SLOPE_LOOKBACK + 1:
            return "CORRECTIVE"

        row = df.iloc[-1]

        # ── ADX filter ──────────────────────────────────────────────────
        adx = float(row.get("adx", 0))
        if adx < ADX_TRENDING_THRESHOLD:
            return "CORRECTIVE"

        # ── SMA50 slope ─────────────────────────────────────────────────
        if "sma_50" not in df.columns:
            return "TRENDING" if adx >= ADX_TRENDING_THRESHOLD else "CORRECTIVE"

        recent = df["sma_50"].dropna()
        if len(recent) < SLOPE_LOOKBACK:
            return "CORRECTIVE"

        slope = float(recent.iloc[-1]) - float(recent.iloc[-SLOPE_LOOKBACK])

        if direction is None:
            # No direction specified — just check ADX
            return "TRENDING"

        direction_lc = direction.lower()
        if direction_lc == "long" and slope > 0:
            return "TRENDING"
        if direction_lc == "short" and slope < 0:
            return "TRENDING"

        return "CORRECTIVE"

    def classify_bar(self, df: pd.DataFrame) -> str:
        """Tag current market regime without direction context."""
        if df is None or len(df) < 2:
            return "CORRECTIVE"
        row = df.iloc[-1]
        adx = float(row.get("adx", 0))
        return "TRENDING" if adx >= ADX_TRENDING_THRESHOLD else "CORRECTIVE"
