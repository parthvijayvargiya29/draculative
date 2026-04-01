"""
aggregation/combined_predictor.py
==================================
THE ATOMIC CORE — Combined prediction engine that integrates ALL system layers.

PHILOSOPHY
----------
The market is an atomic system. Everything orbits the NUCLEUS (macro regime).
This module is the FINAL ARBITER that combines:
  1. Macro Regime (from RegimeClassifier)
  2. Technical Signals (from SignalRouter → TCs)
  3. Fundamental Direction (from PHFundamentalModel)
  4. News Convergence Score (from ph_news_alignment)

LAYERED DECISION PROCESS
-------------------------
Layer 1: REGIME (nucleus)
  - Determines which TCs are active
  - Adjusts position sizing and stops
  - Provides base confidence modifier

Layer 2: TECHNICAL SIGNALS (electron cloud)
  - Multiple TCs fire independently
  - SignalRouter weights and combines based on regime
  - Output: RoutedSignal (direction, score, confidence)

Layer 3: FUNDAMENTAL DIRECTION (gravitational field)
  - PHFundamentalModel predicts market direction from transcripts
  - If technicals ALIGN with fundamental → confidence boost (+20%)
  - If technicals CONFLICT with fundamental → confidence penalty (-30%)

Layer 4: NEWS CONVERGENCE (real-time validation)
  - News alignment score measures how well live news confirms transcripts
  - High alignment (>70%) → confidence boost (+10%)
  - Low alignment (<30%) → confidence penalty (-15%)

FINAL OUTPUT
------------
CombinedPrediction:
  - final_direction: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
  - final_confidence: 0.0 to 1.0
  - position_size_pct: % of portfolio to allocate (regime-adjusted)
  - stop_loss, take_profit: price levels
  - reasoning: dict with breakdown of all layers

EXECUTION GATES
---------------
Only execute if:
  - final_confidence ≥ 0.50 (minimum threshold)
  - position_size_pct > 0.0
  - final_direction is not HOLD

Usage
-----
    from aggregation.combined_predictor import CombinedPredictor
    
    predictor = CombinedPredictor()
    prediction = predictor.predict(
        symbol="SPY",
        spy_df=spy_data,
        vix_df=vix_data,
        tc_signals=[(tc_id, signal), ...],
        news_items=news_list,
    )
    
    if prediction.should_execute:
        # place order
        pass
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from technical.bar_snapshot import Signal, Direction, SignalStrength
    from aggregation.regime_classifier import RegimeClassifier, RegimeState
    from aggregation.signal_router import SignalRouter, RoutedSignal
    from fundamental.ph_fundamental_model import PHFundamentalModel, get_model, DirectionalPrediction
    from fundamental.ph_news_alignment import compute_alignment, AlignmentReport
    from fundamental.news_fetcher import NewsItem
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Allow module to load for documentation/inspection
    RegimeClassifier = None  # type: ignore
    SignalRouter = None  # type: ignore


# ── Configuration ─────────────────────────────────────────────────────────────
MIN_EXECUTION_CONFIDENCE = 0.50   # Must exceed this to trade
ALIGNMENT_CONFIDENCE_BONUS = {
    "strong": 0.10,    # Similarity ≥ 70%
    "moderate": 0.05,  # Similarity 45-70%
    "low": -0.15,      # Similarity < 45%
}
FUNDAMENTAL_ALIGNMENT_BONUS = 0.20   # When technicals match fundamental
FUNDAMENTAL_CONFLICT_PENALTY = -0.30 # When technicals oppose fundamental


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class CombinedPrediction:
    """
    Final prediction output with all layers integrated.
    """
    # Core prediction
    final_direction:     SignalStrength
    final_confidence:    float              # 0.0 to 1.0
    position_size_pct:   float              # % of portfolio (0.0 to 1.0)
    
    # Entry/exit levels
    stop_loss:           float
    take_profit:         float
    
    # Timestamp
    timestamp:           datetime
    symbol:              str
    
    # Layer breakdowns
    regime:              str                # TRENDING | CORRECTIVE | HIGH_VOL | RANGING
    regime_confidence:   float
    
    technical_direction: Direction
    technical_score:     float
    technical_confidence: float
    contributing_tcs:    List[str]
    
    fundamental_direction: str              # BULLISH | BEARISH | NEUTRAL
    fundamental_confidence: float
    fundamental_aligned:   bool             # True if fundamental matches technical
    
    news_alignment_rate:   float            # 0.0 to 1.0
    news_alignment_grade:  str              # STRONG | MODERATE | LOW
    
    # Reasoning (for audit trail)
    reasoning:           Dict[str, any] = field(default_factory=dict)
    
    @property
    def should_execute(self) -> bool:
        """Gate: only execute if confidence and position size meet thresholds."""
        return (
            self.final_confidence >= MIN_EXECUTION_CONFIDENCE
            and self.position_size_pct > 0.0
            and self.final_direction not in (SignalStrength.HOLD,)
        )
    
    def to_dict(self) -> Dict:
        return {
            "final_direction": self.final_direction.value,
            "final_confidence": round(self.final_confidence, 4),
            "position_size_pct": round(self.position_size_pct, 4),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "technical_direction": self.technical_direction.value,
            "technical_score": round(self.technical_score, 4),
            "technical_confidence": round(self.technical_confidence, 4),
            "contributing_tcs": self.contributing_tcs,
            "fundamental_direction": self.fundamental_direction,
            "fundamental_confidence": round(self.fundamental_confidence, 4),
            "fundamental_aligned": self.fundamental_aligned,
            "news_alignment_rate": round(self.news_alignment_rate, 4),
            "news_alignment_grade": self.news_alignment_grade,
            "should_execute": self.should_execute,
            "reasoning": self.reasoning,
        }


# ── Combined Predictor ────────────────────────────────────────────────────────

class CombinedPredictor:
    """
    The atomic core. Integrates all system layers into one final prediction.
    """
    
    def __init__(
        self,
        regime_classifier: Optional[RegimeClassifier] = None,
        signal_router: Optional[SignalRouter] = None,
        fundamental_model: Optional[PHFundamentalModel] = None,
    ):
        """
        Initialize with optional custom components.
        
        Parameters
        ----------
        regime_classifier : RegimeClassifier, optional
        signal_router : SignalRouter, optional
        fundamental_model : PHFundamentalModel, optional
        """
        self.regime_classifier = regime_classifier or RegimeClassifier()
        self.signal_router = signal_router or SignalRouter()
        self.fundamental_model = fundamental_model or get_model()
    
    def predict(
        self,
        symbol: str,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        tc_signals: Optional[List[Tuple[str, Signal]]] = None,
        news_items: Optional[List[NewsItem]] = None,
        news_lookback_days: int = 1,
    ) -> CombinedPrediction:
        """
        Generate combined prediction integrating all system layers.
        
        Parameters
        ----------
        symbol : str
            Symbol to predict (e.g. "SPY", "QQQ")
        spy_df : pd.DataFrame
            SPY OHLCV data (minimum 200 bars)
        vix_df : pd.DataFrame, optional
            VIX close data
        tc_signals : List[Tuple[str, Signal]], optional
            List of (tc_id, signal) from active TCs.
            If None, technical layer will be FLAT.
        news_items : List[NewsItem], optional
            Today's news items for convergence scoring.
            If None, will attempt to fetch via NewsFetcher.
        news_lookback_days : int
            Days ahead for news fetch (default 1 = tomorrow's calendar)
        
        Returns
        -------
        CombinedPrediction
            Final integrated prediction with all layers
        """
        timestamp = datetime.now()
        
        # ── LAYER 1: REGIME (nucleus) ─────────────────────────────────────────
        regime = self.regime_classifier.classify(spy_df, vix_df)
        logger.info(f"Regime: {regime.regime} (confidence={regime.confidence:.2%})")
        
        # ── LAYER 2: TECHNICAL SIGNALS (electron cloud) ───────────────────────
        if tc_signals:
            routed = self.signal_router.route(tc_signals, regime)
            tech_direction = routed.direction
            tech_score = routed.score
            tech_confidence = routed.confidence
            contributing_tcs = routed.contributing_tcs
        else:
            tech_direction = Direction.FLAT
            tech_score = 0.0
            tech_confidence = 0.0
            contributing_tcs = []
        
        logger.info(f"Technical: {tech_direction.value} (score={tech_score:.3f}, conf={tech_confidence:.2%})")
        
        # ── LAYER 3: FUNDAMENTAL DIRECTION (gravitational field) ──────────────
        # Predict on a generic market summary (or latest news headline if available)
        if news_items and len(news_items) > 0:
            sample_text = " ".join([n.headline for n in news_items[:3]])
        else:
            sample_text = f"{symbol} market analysis for {timestamp.strftime('%Y-%m-%d')}"
        
        try:
            fund_pred = self.fundamental_model.predict(sample_text)
            fund_direction = fund_pred.direction
            fund_confidence = fund_pred.confidence
        except Exception as e:
            logger.warning(f"Fundamental model prediction failed: {e}")
            fund_direction = "NEUTRAL"
            fund_confidence = 0.5
        
        logger.info(f"Fundamental: {fund_direction} (conf={fund_confidence:.2%})")
        
        # Check alignment
        fund_aligned = self._check_fundamental_alignment(tech_direction, fund_direction)
        
        # ── LAYER 4: NEWS CONVERGENCE (real-time validation) ──────────────────
        try:
            if news_items is None:
                from fundamental.news_fetcher import NewsFetcher
                fetcher = NewsFetcher()
                news_items = fetcher.get_upcoming(days_ahead=news_lookback_days)
            
            alignment_report = compute_alignment(
                days_ahead=news_lookback_days,
                news_items=news_items,
            )
            news_alignment_rate = alignment_report.overall_similarity_rate
            news_alignment_grade = self._alignment_grade(news_alignment_rate)
        except Exception as e:
            logger.warning(f"News alignment computation failed: {e}")
            news_alignment_rate = 0.5
            news_alignment_grade = "MODERATE"
        
        logger.info(f"News Alignment: {news_alignment_rate:.1%} ({news_alignment_grade})")
        
        # ── COMBINE LAYERS → FINAL PREDICTION ─────────────────────────────────
        final_direction, final_confidence = self._combine(
            tech_direction=tech_direction,
            tech_confidence=tech_confidence,
            regime_confidence=regime.confidence,
            fund_aligned=fund_aligned,
            news_alignment_grade=news_alignment_grade,
        )
        
        # Position sizing (regime-adjusted)
        position_size_pct = self._calculate_position_size(
            final_confidence,
            regime.position_size_multiplier,
        )
        
        # Stop/target (use routed values if available, else default)
        if tc_signals and tech_direction != Direction.FLAT:
            stop_loss = routed.stop_loss
            take_profit = routed.take_profit
        else:
            # No technical signal → use default SPY levels (example)
            last_close = float(spy_df.iloc[-1]["close"])
            stop_loss = last_close * 0.98 if tech_direction == Direction.LONG else last_close * 1.02
            take_profit = last_close * 1.03 if tech_direction == Direction.LONG else last_close * 0.97
        
        # Reasoning
        reasoning = {
            "regime_weight": regime.position_size_multiplier,
            "fundamental_alignment_bonus": FUNDAMENTAL_ALIGNMENT_BONUS if fund_aligned else FUNDAMENTAL_CONFLICT_PENALTY,
            "news_convergence_bonus": ALIGNMENT_CONFIDENCE_BONUS.get(news_alignment_grade.lower(), 0.0),
            "base_tech_confidence": tech_confidence,
            "regime_active_tcs": regime.active_tc_ids,
            "tc_scores": routed.tc_scores if tc_signals else {},
        }
        
        return CombinedPrediction(
            final_direction=final_direction,
            final_confidence=final_confidence,
            position_size_pct=position_size_pct,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=timestamp,
            symbol=symbol,
            regime=regime.regime,
            regime_confidence=regime.confidence,
            technical_direction=tech_direction,
            technical_score=tech_score,
            technical_confidence=tech_confidence,
            contributing_tcs=contributing_tcs,
            fundamental_direction=fund_direction,
            fundamental_confidence=fund_confidence,
            fundamental_aligned=fund_aligned,
            news_alignment_rate=news_alignment_rate,
            news_alignment_grade=news_alignment_grade,
            reasoning=reasoning,
        )
    
    # ── Helper Methods ────────────────────────────────────────────────────────
    
    @staticmethod
    def _check_fundamental_alignment(tech_dir: Direction, fund_dir: str) -> bool:
        """Check if technical and fundamental directions align."""
        if tech_dir == Direction.FLAT:
            return True  # No conflict
        
        fund_dir_upper = fund_dir.upper()
        
        if tech_dir == Direction.LONG:
            return fund_dir_upper == "BULLISH"
        elif tech_dir == Direction.SHORT:
            return fund_dir_upper == "BEARISH"
        
        return False
    
    @staticmethod
    def _alignment_grade(rate: float) -> str:
        """Convert similarity rate to grade."""
        if rate >= 0.70:
            return "STRONG"
        elif rate >= 0.45:
            return "MODERATE"
        else:
            return "LOW"
    
    def _combine(
        self,
        tech_direction: Direction,
        tech_confidence: float,
        regime_confidence: float,
        fund_aligned: bool,
        news_alignment_grade: str,
    ) -> Tuple[SignalStrength, float]:
        """
        Combine all layers into final direction + confidence.
        
        Returns
        -------
        (final_direction, final_confidence)
        """
        if tech_direction == Direction.FLAT:
            return SignalStrength.HOLD, 0.0
        
        # Start with technical confidence
        confidence = tech_confidence
        
        # Boost/penalty from fundamental alignment
        if fund_aligned:
            confidence += FUNDAMENTAL_ALIGNMENT_BONUS
        else:
            confidence += FUNDAMENTAL_CONFLICT_PENALTY
        
        # Boost/penalty from news convergence
        confidence += ALIGNMENT_CONFIDENCE_BONUS.get(news_alignment_grade.lower(), 0.0)
        
        # Regime multiplier (confidence reflects regime strength)
        confidence *= regime_confidence
        
        # Clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        
        # Map to SignalStrength
        if tech_direction == Direction.LONG:
            if confidence >= 0.75:
                direction = SignalStrength.STRONG_BUY
            elif confidence >= 0.50:
                direction = SignalStrength.BUY
            else:
                direction = SignalStrength.HOLD
        else:  # SHORT
            if confidence >= 0.75:
                direction = SignalStrength.STRONG_SELL
            elif confidence >= 0.50:
                direction = SignalStrength.SELL
            else:
                direction = SignalStrength.HOLD
        
        return direction, confidence
    
    @staticmethod
    def _calculate_position_size(confidence: float, regime_multiplier: float) -> float:
        """
        Calculate position size as % of portfolio.
        
        Base allocation: 1% per 10% confidence (so 100% confidence = 10% position)
        Regime-adjusted.
        """
        base_pct = confidence * 0.10  # Max 10% at full confidence
        adjusted_pct = base_pct * regime_multiplier
        return max(0.0, min(0.10, adjusted_pct))  # Cap at 10%


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Combined prediction engine (atomic core)")
    parser.add_argument("--symbol", default="SPY", help="Symbol to predict")
    parser.add_argument("--lookback", type=int, default=400, help="Days of SPY history")
    args = parser.parse_args()
    
    print("=" * 80)
    print("  DRACULATIVE ATOMIC CORE — COMBINED PREDICTOR")
    print("=" * 80)
    
    # Fetch data
    from aggregation.regime_classifier import fetch_spy_vix_data
    
    print(f"\n📡 Fetching {args.symbol} data …")
    spy_df, vix_df = fetch_spy_vix_data(lookback_days=args.lookback)
    
    if spy_df.empty:
        print("❌ No data available. Exiting.")
        sys.exit(1)
    
    print(f"✅ SPY: {len(spy_df)} bars")
    
    # Mock TC signals (in production, these come from actual TC modules)
    print("\n🔧 Mock TC signals (replace with real TCs in production)")
    
    # Predict
    predictor = CombinedPredictor()
    prediction = predictor.predict(
        symbol=args.symbol,
        spy_df=spy_df,
        vix_df=vix_df,
        tc_signals=None,  # No TCs active (mock)
        news_items=None,  # Will fetch automatically
    )
    
    # Display
    print("\n" + "=" * 80)
    print(f"  FINAL PREDICTION: {prediction.final_direction.value}")
    print("=" * 80)
    print(f"  Confidence         : {prediction.final_confidence:.1%}")
    print(f"  Position Size      : {prediction.position_size_pct:.2%} of portfolio")
    print(f"  Stop / Target      : {prediction.stop_loss:.2f} / {prediction.take_profit:.2f}")
    print(f"\n  Should Execute     : {prediction.should_execute}")
    print(f"\n  REGIME             : {prediction.regime} (conf={prediction.regime_confidence:.1%})")
    print(f"  Technical          : {prediction.technical_direction.value} @ {prediction.technical_score:.3f}")
    print(f"  Fundamental        : {prediction.fundamental_direction} (aligned={prediction.fundamental_aligned})")
    print(f"  News Alignment     : {prediction.news_alignment_rate:.1%} ({prediction.news_alignment_grade})")
    print(f"\n  Contributing TCs   : {', '.join(prediction.contributing_tcs) if prediction.contributing_tcs else 'None (mock run)'}")
    print("=" * 80)
    
    # Export
    import json
    output_path = _ROOT / "reports" / "combined_prediction_latest.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(prediction.to_dict(), indent=2))
    print(f"\n💾 Saved → {output_path.relative_to(_ROOT)}")
