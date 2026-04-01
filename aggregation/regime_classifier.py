"""
aggregation/regime_classifier.py
=================================
THE NUCLEUS — Macro Regime Classification Engine

PHILOSOPHY
----------
Markets behave like atomic structures. Everything orbits the NUCLEUS (macro regime).
The regime is the gravitational center that determines which signals are amplified
and which are suppressed.

REGIMES (4 states)
------------------
1. TRENDING      — Strong directional movement, ADX > 25, clear SMA slope
2. CORRECTIVE    — Mean reversion environment, choppy, low ADX
3. HIGH_VOL      — VIX > 25, erratic price action, reduce all position sizes
4. RANGING       — Sideways consolidation, ATR < 0.5 * 20-day ATR

REGIME → TC ROUTING
-------------------
TRENDING     → Activate: TC-01 (Supertrend), TC-07 (ADX Gate), TC-08 (MA Cross), TC-11 (ChoCH)
CORRECTIVE   → Activate: TC-02 (BB+RSI2), TC-06 (VWAP Mean Rev), TC-10 (Liquidity Sweep), TC-13 (Stoch RSI)
HIGH_VOL     → Reduce all position sizes by 50%, widen stops by 1.5x
RANGING      → Activate only mean reversion TCs, disable breakout TCs

INPUTS
------
- SPY 1D bars: ADX, SMA50 slope, SMA200, close
- VIX 1D bars: close
- Breadth indicator (optional): advance/decline ratio

OUTPUTS
-------
RegimeState dataclass:
  - regime: str (TRENDING | CORRECTIVE | HIGH_VOL | RANGING)
  - confidence: float (0.0 to 1.0)
  - active_tc_ids: List[str] (e.g. ["TC-01", "TC-07", "TC-08"])
  - position_size_multiplier: float (1.0 = full, 0.5 = half)
  - metadata: dict (SPY ADX, VIX, slopes, etc.)

Usage
-----
    from aggregation.regime_classifier import RegimeClassifier, fetch_spy_vix_data
    
    classifier = RegimeClassifier()
    spy_df, vix_df = fetch_spy_vix_data()
    regime = classifier.classify(spy_df, vix_df)
    
    print(regime.regime)               # "TRENDING"
    print(regime.active_tc_ids)        # ["TC-01", "TC-07", "TC-08", "TC-11"]
    print(regime.position_size_multiplier)  # 1.0
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from technical.indicators_v4 import compute_adx, compute_sma, compute_atr

logger = logging.getLogger(__name__)

# ── Regime thresholds (tunable via config) ────────────────────────────────────
ADX_TRENDING_THRESHOLD  = 25.0
VIX_HIGH_VOL_THRESHOLD  = 25.0
ATR_RANGING_THRESHOLD   = 0.5   # ratio vs 20-day ATR
SMA_SLOPE_LOOKBACK      = 5     # bars for slope calculation


# ── TC → Regime mapping ───────────────────────────────────────────────────────
REGIME_TC_MAP: Dict[str, List[str]] = {
    "TRENDING": ["TC-01", "TC-07", "TC-08", "TC-11"],
    "CORRECTIVE": ["TC-02", "TC-06", "TC-10", "TC-13"],
    "HIGH_VOL": ["TC-02", "TC-06"],  # Only safest mean reversion
    "RANGING": ["TC-02", "TC-06", "TC-13"],
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """Current macro regime state with routing instructions."""
    regime:                  str              # TRENDING | CORRECTIVE | HIGH_VOL | RANGING
    confidence:              float            # 0.0 to 1.0
    timestamp:               datetime
    
    # Routing instructions
    active_tc_ids:           List[str]        # TCs to activate in this regime
    position_size_multiplier: float = 1.0     # Scaling factor for position sizing
    stop_multiplier:         float = 1.0      # Scaling factor for stop distances
    
    # Market context
    spy_adx:                 Optional[float] = None
    spy_sma50_slope:         Optional[float] = None
    spy_price_vs_sma200:     Optional[float] = None  # % above/below SMA200
    vix_close:               Optional[float] = None
    atr_ratio:               Optional[float] = None  # current ATR / 20-day avg ATR
    
    # Metadata
    metadata:                Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "active_tc_ids": self.active_tc_ids,
            "position_size_multiplier": self.position_size_multiplier,
            "stop_multiplier": self.stop_multiplier,
            "spy_adx": self.spy_adx,
            "spy_sma50_slope": self.spy_sma50_slope,
            "spy_price_vs_sma200": self.spy_price_vs_sma200,
            "vix_close": self.vix_close,
            "atr_ratio": self.atr_ratio,
            "metadata": self.metadata,
        }


# ── Regime Classifier ─────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Macro regime classification engine. Determines which market environment
    we're in and routes signals accordingly.
    
    This is the NUCLEUS of the atomic trading system.
    """
    
    def __init__(
        self,
        adx_threshold: float = ADX_TRENDING_THRESHOLD,
        vix_threshold: float = VIX_HIGH_VOL_THRESHOLD,
        atr_ratio_threshold: float = ATR_RANGING_THRESHOLD,
    ):
        self.adx_threshold = adx_threshold
        self.vix_threshold = vix_threshold
        self.atr_ratio_threshold = atr_ratio_threshold
    
    def classify(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> RegimeState:
        """
        Classify the current macro regime based on SPY and VIX data.
        
        Parameters
        ----------
        spy_df : pd.DataFrame
            SPY 1D OHLCV data. Must have at least 200 bars.
            Expected columns: open, high, low, close, volume
        vix_df : pd.DataFrame, optional
            VIX 1D close data. If None, HIGH_VOL regime won't be triggered.
        
        Returns
        -------
        RegimeState
            Current regime classification with routing instructions.
        """
        if len(spy_df) < 200:
            logger.warning(f"Insufficient SPY data ({len(spy_df)} bars, need 200). Defaulting to CORRECTIVE.")
            return self._default_regime()
        
        # ── Enrich SPY with indicators ────────────────────────────────────────
        spy = spy_df.copy()
        spy["adx"] = compute_adx(spy, period=14)
        spy["sma_50"] = compute_sma(spy, period=50)
        spy["sma_200"] = compute_sma(spy, period=200)
        spy["atr"] = compute_atr(spy, period=14)
        
        last_row = spy.iloc[-1]
        timestamp = last_row.name if isinstance(last_row.name, datetime) else pd.Timestamp(last_row.name).to_pydatetime()
        
        # ── Extract metrics ───────────────────────────────────────────────────
        adx = float(last_row["adx"]) if pd.notna(last_row["adx"]) else 0.0
        close = float(last_row["close"])
        sma_50 = float(last_row["sma_50"]) if pd.notna(last_row["sma_50"]) else close
        sma_200 = float(last_row["sma_200"]) if pd.notna(last_row["sma_200"]) else close
        atr_current = float(last_row["atr"]) if pd.notna(last_row["atr"]) else 0.0
        
        # SMA50 slope (last 5 days)
        sma50_recent = spy["sma_50"].dropna().tail(SMA_SLOPE_LOOKBACK + 1)
        if len(sma50_recent) >= 2:
            sma50_slope = float(sma50_recent.iloc[-1] - sma50_recent.iloc[-SMA_SLOPE_LOOKBACK])
        else:
            sma50_slope = 0.0
        
        # Price vs SMA200
        price_vs_sma200 = ((close - sma_200) / sma_200 * 100) if sma_200 > 0 else 0.0
        
        # ATR ratio (current vs 20-day average)
        atr_20day_avg = float(spy["atr"].tail(20).mean()) if len(spy["atr"].dropna()) >= 20 else atr_current
        atr_ratio = (atr_current / atr_20day_avg) if atr_20day_avg > 0 else 1.0
        
        # VIX
        vix_close = None
        if vix_df is not None and len(vix_df) > 0:
            vix_last = vix_df.iloc[-1]
            vix_close = float(vix_last["close"]) if "close" in vix_df.columns else float(vix_last.iloc[0])
        
        # ── Regime decision logic ─────────────────────────────────────────────
        
        # Priority 1: HIGH_VOL (overrides everything)
        if vix_close is not None and vix_close > self.vix_threshold:
            regime = "HIGH_VOL"
            confidence = min(1.0, (vix_close - self.vix_threshold) / 10.0 + 0.6)
            pos_mult = 0.5
            stop_mult = 1.5
        
        # Priority 2: RANGING (low ATR, choppy)
        elif atr_ratio < self.atr_ratio_threshold:
            regime = "RANGING"
            confidence = 1.0 - atr_ratio
            pos_mult = 0.75
            stop_mult = 1.0
        
        # Priority 3: TRENDING (high ADX + SMA slope aligned)
        elif adx > self.adx_threshold and abs(sma50_slope) > 0.5:
            regime = "TRENDING"
            confidence = min(1.0, (adx - self.adx_threshold) / 20.0 + 0.5)
            pos_mult = 1.0
            stop_mult = 1.0
        
        # Default: CORRECTIVE
        else:
            regime = "CORRECTIVE"
            confidence = 0.6
            pos_mult = 0.85
            stop_mult = 1.0
        
        # ── Map regime → active TCs ───────────────────────────────────────────
        active_tc_ids = REGIME_TC_MAP.get(regime, [])
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            timestamp=timestamp,
            active_tc_ids=active_tc_ids,
            position_size_multiplier=pos_mult,
            stop_multiplier=stop_mult,
            spy_adx=adx,
            spy_sma50_slope=sma50_slope,
            spy_price_vs_sma200=price_vs_sma200,
            vix_close=vix_close,
            atr_ratio=atr_ratio,
            metadata={
                "adx_threshold": self.adx_threshold,
                "vix_threshold": self.vix_threshold,
                "atr_ratio_threshold": self.atr_ratio_threshold,
            },
        )
    
    def _default_regime(self) -> RegimeState:
        """Fallback regime when data is insufficient."""
        return RegimeState(
            regime="CORRECTIVE",
            confidence=0.3,
            timestamp=datetime.now(),
            active_tc_ids=REGIME_TC_MAP["CORRECTIVE"],
            position_size_multiplier=0.5,
            stop_multiplier=1.0,
        )


# ── Data Fetcher (uses existing Alpaca loader) ───────────────────────────────

def fetch_spy_vix_data(
    lookback_days: int = 400,
    timeframe: str = "1Day",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Fetch SPY and VIX historical data for regime classification.
    
    Parameters
    ----------
    lookback_days : int
        Number of calendar days of history to fetch
    timeframe : str
        Alpaca timeframe, e.g. "1Day", "1Hour"
    
    Returns
    -------
    (spy_df, vix_df)
        spy_df: OHLCV DataFrame for SPY
        vix_df: Close DataFrame for VIX (or None if unavailable)
    """
    try:
        from simulation.alpaca_loader import AlpacaDataLoader
        loader = AlpacaDataLoader()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch SPY
        spy_df = loader.fetch_bars("SPY", start_date, end_date, timeframe)
        
        # Fetch VIX (may not be available on all Alpaca accounts)
        try:
            vix_df = loader.fetch_bars("VIX", start_date, end_date, timeframe)
        except Exception as e:
            logger.warning(f"VIX data unavailable: {e}")
            vix_df = None
        
        return spy_df, vix_df
    
    except ImportError:
        logger.error("AlpacaDataLoader not available. Cannot fetch SPY/VIX data.")
        return pd.DataFrame(), None
    except Exception as e:
        logger.error(f"Failed to fetch SPY/VIX data: {e}")
        return pd.DataFrame(), None


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(description="Classify current macro regime (NUCLEUS)")
    parser.add_argument("--lookback", type=int, default=400, help="Days of SPY/VIX history")
    args = parser.parse_args()
    
    print("=" * 80)
    print("  DRACULATIVE NUCLEUS — REGIME CLASSIFIER")
    print("=" * 80)
    
    print(f"\n📡 Fetching SPY/VIX data (last {args.lookback} days) …")
    spy_df, vix_df = fetch_spy_vix_data(lookback_days=args.lookback)
    
    if spy_df.empty:
        print("❌ No SPY data available. Exiting.")
        sys.exit(1)
    
    print(f"✅ SPY: {len(spy_df)} bars")
    if vix_df is not None and not vix_df.empty:
        print(f"✅ VIX: {len(vix_df)} bars")
    else:
        print("⚠️  VIX: unavailable (HIGH_VOL regime detection disabled)")
    
    print("\n🧬 Classifying regime …")
    classifier = RegimeClassifier()
    regime = classifier.classify(spy_df, vix_df)
    
    print("\n" + "=" * 80)
    print(f"  REGIME: {regime.regime}  (confidence: {regime.confidence:.1%})")
    print("=" * 80)
    print(f"  Timestamp              : {regime.timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Position Size Multiplier: {regime.position_size_multiplier:.2f}x")
    print(f"  Stop Multiplier        : {regime.stop_multiplier:.2f}x")
    print(f"\n  Active TC IDs          : {', '.join(regime.active_tc_ids)}")
    print(f"\n  Market Metrics:")
    print(f"    SPY ADX              : {regime.spy_adx:.2f}" if regime.spy_adx else "    SPY ADX              : N/A")
    print(f"    SPY SMA50 Slope      : {regime.spy_sma50_slope:+.2f}" if regime.spy_sma50_slope else "    SPY SMA50 Slope      : N/A")
    print(f"    SPY vs SMA200        : {regime.spy_price_vs_sma200:+.2f}%" if regime.spy_price_vs_sma200 else "    SPY vs SMA200        : N/A")
    print(f"    VIX                  : {regime.vix_close:.2f}" if regime.vix_close else "    VIX                  : N/A")
    print(f"    ATR Ratio            : {regime.atr_ratio:.2f}" if regime.atr_ratio else "    ATR Ratio            : N/A")
    print("=" * 80)
