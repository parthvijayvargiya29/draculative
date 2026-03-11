#!/usr/bin/env python3
"""
Stock Predictor - Combines Technical, Fundamental, and News Signals

Three independent signal streams:
1. Technical Signal - Based on price action, indicators, options flow
2. Fundamental Signal - Based on valuation, quality, growth metrics
3. News/Event Signal - Based on news sentiment and historical event impact

Final prediction combines all three with configurable weights.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Import our modules
from realtime_data import RealTimeDataFetcher, LiveMarketData
from news_tracker import NewsTracker, NewsAnalysis


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-1
    confidence: float  # 0-1
    
    # Component scores
    trend_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    options_score: float
    
    # Key reasons
    bullish_factors: List[str]
    bearish_factors: List[str]
    
    # Targets
    support: float
    resistance: float
    stop_loss: float
    target_1: float
    target_2: float


@dataclass
class FundamentalSignal:
    """Fundamental analysis signal."""
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-1
    confidence: float  # 0-1
    
    # Component scores
    value_score: float
    quality_score: float
    growth_score: float
    sentiment_score: float
    
    # Key reasons
    bullish_factors: List[str]
    bearish_factors: List[str]
    
    # Fair value estimate
    fair_value: float
    upside_downside: float


@dataclass
class NewsSignal:
    """News/Event-based signal."""
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-1
    confidence: float  # 0-1
    
    # Component scores
    sentiment_score: float
    event_score: float
    momentum_score: float
    
    # Event details
    detected_events: List[str]
    predicted_impact: Dict
    
    # Key reasons
    bullish_factors: List[str]
    bearish_factors: List[str]


@dataclass
class CombinedPrediction:
    """Final combined prediction."""
    ticker: str
    timestamp: str
    
    # Overall prediction
    direction: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence: float  # 0-1
    
    # Individual signals
    technical: TechnicalSignal
    fundamental: FundamentalSignal
    news: NewsSignal
    
    # Price targets
    current_price: float
    target_price_bull: float
    target_price_bear: float
    stop_loss: float
    
    # Risk metrics
    risk_reward_ratio: float
    volatility: float
    
    # Summary
    summary: str
    action_items: List[str]


class StockPredictor:
    """
    Main predictor combining all signal sources.
    """
    
    def __init__(self, ticker: str, weights: Dict[str, float] = None):
        self.ticker = ticker.upper()
        self.weights = weights or {
            'technical': 0.40,
            'fundamental': 0.30,
            'news': 0.30
        }
        
        self.live_data: Optional[LiveMarketData] = None
        self.news_analysis: Optional[NewsAnalysis] = None
        
    def fetch_data(self):
        """Fetch all required data."""
        print(f"\n{'='*60}")
        print(f"STOCK PREDICTOR: {self.ticker}")
        print(f"{'='*60}")
        
        # Fetch live market data
        print("\n📊 Fetching live market data...")
        fetcher = RealTimeDataFetcher(self.ticker)
        self.live_data = fetcher.fetch_all()
        
        # Fetch and analyze news
        print("\n📰 Analyzing news and events...")
        tracker = NewsTracker(self.ticker)
        self.news_analysis = tracker.analyze()
        
    def generate_technical_signal(self) -> TechnicalSignal:
        """Generate technical analysis signal."""
        t = self.live_data.technicals
        o = self.live_data.options
        
        bullish = []
        bearish = []
        
        # === TREND ANALYSIS ===
        trend_score = 0
        
        # Moving average alignment
        if t.price_vs_sma_20 == "above" and t.price_vs_sma_50 == "above" and t.price_vs_sma_200 == "above":
            trend_score += 0.3
            bullish.append("Price above all major MAs (20/50/200)")
        elif t.price_vs_sma_20 == "below" and t.price_vs_sma_50 == "below" and t.price_vs_sma_200 == "below":
            trend_score -= 0.3
            bearish.append("Price below all major MAs (20/50/200)")
        
        # Golden/Death cross
        if t.golden_cross:
            trend_score += 0.15
            bullish.append("Golden Cross (SMA50 > SMA200)")
        else:
            trend_score -= 0.1
            bearish.append("Death Cross (SMA50 < SMA200)")
        
        # ADX trend strength
        if t.adx > 25:
            if t.trend_direction == "bullish":
                trend_score += 0.2
                bullish.append(f"Strong bullish trend (ADX={t.adx})")
            else:
                trend_score -= 0.2
                bearish.append(f"Strong bearish trend (ADX={t.adx})")
        
        # === MOMENTUM ANALYSIS ===
        momentum_score = 0
        
        # RSI
        if t.rsi_14 < 30:
            momentum_score += 0.25
            bullish.append(f"RSI oversold ({t.rsi_14})")
        elif t.rsi_14 > 70:
            momentum_score -= 0.25
            bearish.append(f"RSI overbought ({t.rsi_14})")
        elif 40 <= t.rsi_14 <= 60:
            pass  # Neutral
        elif t.rsi_14 < 40:
            momentum_score -= 0.1
        else:
            momentum_score += 0.1
        
        # MACD
        if t.macd_cross == "bullish":
            momentum_score += 0.2
            bullish.append("MACD bullish crossover")
        elif t.macd_cross == "bearish":
            momentum_score -= 0.2
            bearish.append("MACD bearish crossover")
        
        if t.macd > 0 and t.macd_histogram > 0:
            momentum_score += 0.1
        elif t.macd < 0 and t.macd_histogram < 0:
            momentum_score -= 0.1
        
        # Stochastic
        if t.stochastic_k < 20:
            momentum_score += 0.15
            bullish.append(f"Stochastic oversold ({t.stochastic_k})")
        elif t.stochastic_k > 80:
            momentum_score -= 0.15
            bearish.append(f"Stochastic overbought ({t.stochastic_k})")
        
        # === VOLATILITY ANALYSIS ===
        volatility_score = 0
        
        # Bollinger position
        if t.bollinger_position < 0.2:
            volatility_score += 0.2
            bullish.append("Price near lower Bollinger Band")
        elif t.bollinger_position > 0.8:
            volatility_score -= 0.2
            bearish.append("Price near upper Bollinger Band")
        
        # VWAP
        if t.vwap_distance < -2:
            volatility_score += 0.15
            bullish.append(f"Price below VWAP ({t.vwap_distance:.1f}%)")
        elif t.vwap_distance > 2:
            volatility_score -= 0.1
            bearish.append(f"Price above VWAP ({t.vwap_distance:.1f}%)")
        
        # === VOLUME ANALYSIS ===
        volume_score = 0
        
        if t.volume_ratio > 1.5 and t.change_1d > 0:
            volume_score += 0.2
            bullish.append(f"High volume rally ({t.volume_ratio:.1f}x avg)")
        elif t.volume_ratio > 1.5 and t.change_1d < 0:
            volume_score -= 0.2
            bearish.append(f"High volume selloff ({t.volume_ratio:.1f}x avg)")
        
        if t.obv_trend == "up":
            volume_score += 0.1
            bullish.append("OBV trending up (accumulation)")
        else:
            volume_score -= 0.1
            bearish.append("OBV trending down (distribution)")
        
        # === OPTIONS ANALYSIS ===
        options_score = 0
        
        if o:
            # Put/Call ratio
            if o.put_call_ratio_oi > 1.5:
                options_score -= 0.15
                bearish.append(f"High put/call OI ratio ({o.put_call_ratio_oi})")
            elif o.put_call_ratio_oi < 0.7:
                options_score += 0.15
                bullish.append(f"Low put/call OI ratio ({o.put_call_ratio_oi})")
            
            # Max pain
            distance_to_max_pain = (t.price - o.max_pain) / t.price * 100
            if distance_to_max_pain < -3:
                options_score += 0.1
                bullish.append(f"Price below max pain (${o.max_pain})")
            elif distance_to_max_pain > 3:
                options_score -= 0.1
                bearish.append(f"Price above max pain (${o.max_pain})")
            
            # IV Skew (negative = puts expensive = bearish hedging)
            if o.iv_skew > 5:
                options_score -= 0.1
                bearish.append("High IV skew (bearish hedging)")
            elif o.iv_skew < -5:
                options_score += 0.1
                bullish.append("Negative IV skew (bullish sentiment)")
            
            # Unusual activity
            if o.unusual_activity:
                for activity in o.unusual_activity[:2]:
                    if activity['type'] == 'call':
                        options_score += 0.05
                        bullish.append(f"Unusual call activity: ${activity['strike']}")
                    else:
                        options_score -= 0.05
                        bearish.append(f"Unusual put activity: ${activity['strike']}")
        
        # === COMBINE SCORES ===
        total_score = (
            trend_score * 0.25 +
            momentum_score * 0.30 +
            volatility_score * 0.20 +
            volume_score * 0.15 +
            options_score * 0.10
        )
        
        # Direction
        if total_score > 0.15:
            direction = "BULLISH"
        elif total_score < -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        # Strength (0-1)
        strength = min(1.0, abs(total_score) * 2)
        
        # Confidence based on signal agreement
        signals = [trend_score, momentum_score, volatility_score, volume_score]
        agreement = sum(1 for s in signals if (s > 0) == (total_score > 0)) / len(signals)
        confidence = agreement * 0.5 + 0.3  # Base confidence of 0.3
        
        # === PRICE TARGETS ===
        atr = t.atr_14
        
        support = min(t.sma_20, t.bollinger_lower)
        resistance = max(t.sma_50, t.bollinger_upper)
        stop_loss = t.price - (2 * atr) if direction == "BULLISH" else t.price + (2 * atr)
        target_1 = t.price + (1.5 * atr) if direction == "BULLISH" else t.price - (1.5 * atr)
        target_2 = t.price + (3 * atr) if direction == "BULLISH" else t.price - (3 * atr)
        
        return TechnicalSignal(
            direction=direction,
            strength=round(strength, 2),
            confidence=round(confidence, 2),
            trend_score=round(trend_score, 2),
            momentum_score=round(momentum_score, 2),
            volatility_score=round(volatility_score, 2),
            volume_score=round(volume_score, 2),
            options_score=round(options_score, 2),
            bullish_factors=bullish[:5],
            bearish_factors=bearish[:5],
            support=round(support, 2),
            resistance=round(resistance, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2)
        )
    
    def generate_fundamental_signal(self) -> FundamentalSignal:
        """Generate fundamental analysis signal."""
        f = self.live_data.fundamentals
        
        bullish = []
        bearish = []
        
        # === VALUE ANALYSIS ===
        value_score = 0
        
        # P/E analysis
        if f.pe_ratio > 0:
            if f.pe_ratio < 15:
                value_score += 0.25
                bullish.append(f"Low P/E ratio ({f.pe_ratio:.1f})")
            elif f.pe_ratio > 50:
                value_score -= 0.2
                bearish.append(f"High P/E ratio ({f.pe_ratio:.1f})")
            elif f.pe_ratio > 30:
                value_score -= 0.1
        
        # Forward P/E vs trailing (earnings growth expected?)
        if f.forward_pe > 0 and f.pe_ratio > 0:
            if f.forward_pe < f.pe_ratio * 0.8:
                value_score += 0.15
                bullish.append("Forward P/E significantly lower (growth expected)")
            elif f.forward_pe > f.pe_ratio * 1.2:
                value_score -= 0.15
                bearish.append("Forward P/E higher (earnings decline expected)")
        
        # PEG ratio
        if f.peg_ratio > 0:
            if f.peg_ratio < 1:
                value_score += 0.2
                bullish.append(f"PEG ratio < 1 ({f.peg_ratio:.2f})")
            elif f.peg_ratio > 2:
                value_score -= 0.15
                bearish.append(f"High PEG ratio ({f.peg_ratio:.2f})")
        
        # === QUALITY ANALYSIS ===
        quality_score = 0
        
        # ROE
        if f.roe > 20:
            quality_score += 0.2
            bullish.append(f"High ROE ({f.roe}%)")
        elif f.roe < 5:
            quality_score -= 0.15
            bearish.append(f"Low ROE ({f.roe}%)")
        
        # Profit margins
        if f.profit_margin > 20:
            quality_score += 0.15
            bullish.append(f"High profit margin ({f.profit_margin}%)")
        elif f.profit_margin < 0:
            quality_score -= 0.25
            bearish.append(f"Negative profit margin ({f.profit_margin}%)")
        elif f.profit_margin < 5:
            quality_score -= 0.1
        
        # Debt
        if f.debt_to_equity > 0:
            if f.debt_to_equity < 50:
                quality_score += 0.1
                bullish.append(f"Low debt ({f.debt_to_equity}% D/E)")
            elif f.debt_to_equity > 150:
                quality_score -= 0.15
                bearish.append(f"High debt ({f.debt_to_equity}% D/E)")
        
        # === GROWTH ANALYSIS ===
        growth_score = 0
        
        if f.revenue_growth > 20:
            growth_score += 0.25
            bullish.append(f"Strong revenue growth ({f.revenue_growth}%)")
        elif f.revenue_growth < 0:
            growth_score -= 0.2
            bearish.append(f"Revenue declining ({f.revenue_growth}%)")
        elif f.revenue_growth > 10:
            growth_score += 0.1
        
        if f.earnings_growth > 20:
            growth_score += 0.2
            bullish.append(f"Strong earnings growth ({f.earnings_growth}%)")
        elif f.earnings_growth < 0:
            growth_score -= 0.15
            bearish.append(f"Earnings declining ({f.earnings_growth}%)")
        
        # === SENTIMENT ANALYSIS ===
        sentiment_score = 0
        
        # Analyst rating
        if f.analyst_rating in ["Strong Buy", "Buy"]:
            sentiment_score += 0.2
            bullish.append(f"Analyst rating: {f.analyst_rating}")
        elif f.analyst_rating in ["Sell", "Strong Sell"]:
            sentiment_score -= 0.2
            bearish.append(f"Analyst rating: {f.analyst_rating}")
        
        # Upside potential
        if f.upside_potential > 30:
            sentiment_score += 0.15
            bullish.append(f"High analyst upside ({f.upside_potential:.0f}%)")
        elif f.upside_potential < -10:
            sentiment_score -= 0.15
            bearish.append(f"Analyst downside ({f.upside_potential:.0f}%)")
        
        # === COMBINE SCORES ===
        total_score = (
            value_score * 0.30 +
            quality_score * 0.30 +
            growth_score * 0.25 +
            sentiment_score * 0.15
        )
        
        # Direction
        if total_score > 0.1:
            direction = "BULLISH"
        elif total_score < -0.1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        strength = min(1.0, abs(total_score) * 2)
        
        # Confidence based on data availability
        data_points = sum(1 for v in [f.pe_ratio, f.roe, f.revenue_growth, f.earnings_growth] if v != 0)
        confidence = 0.3 + (data_points / 4) * 0.4
        
        # Fair value estimate (simplified)
        if f.pe_ratio > 0 and f.earnings_growth > 0:
            # PEG-based fair value
            fair_pe = max(10, min(40, f.earnings_growth * 1.5))
            current_price = self.live_data.technicals.price
            fair_value = current_price * (fair_pe / f.pe_ratio)
        else:
            fair_value = f.target_price
        
        upside_downside = (fair_value / self.live_data.technicals.price - 1) * 100
        
        return FundamentalSignal(
            direction=direction,
            strength=round(strength, 2),
            confidence=round(confidence, 2),
            value_score=round(value_score, 2),
            quality_score=round(quality_score, 2),
            growth_score=round(growth_score, 2),
            sentiment_score=round(sentiment_score, 2),
            bullish_factors=bullish[:5],
            bearish_factors=bearish[:5],
            fair_value=round(fair_value, 2),
            upside_downside=round(upside_downside, 2)
        )
    
    def generate_news_signal(self) -> NewsSignal:
        """Generate news/event-based signal."""
        n = self.news_analysis
        
        bullish = []
        bearish = []
        
        # === SENTIMENT ANALYSIS ===
        sentiment_score = n.overall_sentiment  # Already -1 to 1
        
        if sentiment_score > 0.3:
            bullish.append(f"Positive news sentiment ({sentiment_score:+.2f})")
        elif sentiment_score < -0.3:
            bearish.append(f"Negative news sentiment ({sentiment_score:+.2f})")
        
        # === EVENT ANALYSIS ===
        event_score = 0
        detected_events = []
        
        for event in n.event_signals:
            detected_events.append(event['event_type'])
            impact = event.get('predicted_impact', {})
            
            if impact:
                avg_spy = impact.get('avg_spy_1d', 0)
                
                if avg_spy > 1:
                    event_score += 0.3
                    bullish.append(f"Event: {event['event_type']} (historically bullish)")
                elif avg_spy < -1:
                    event_score -= 0.3
                    bearish.append(f"Event: {event['event_type']} (historically bearish)")
        
        # === MOMENTUM (news volume/intensity) ===
        momentum_score = 0
        
        # More news = more attention = higher volatility expected
        if n.news_count > 10:
            momentum_score = 0.1 if sentiment_score > 0 else -0.1
        
        # === COMBINE SCORES ===
        total_score = (
            sentiment_score * 0.50 +
            event_score * 0.35 +
            momentum_score * 0.15
        )
        
        # Direction
        if total_score > 0.1:
            direction = "BULLISH"
        elif total_score < -0.1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        strength = min(1.0, abs(total_score) * 1.5)
        
        # Confidence based on news volume and event detection
        confidence = 0.3
        if n.news_count > 5:
            confidence += 0.2
        if detected_events:
            confidence += 0.2
        confidence = min(0.8, confidence)
        
        return NewsSignal(
            direction=direction,
            strength=round(strength, 2),
            confidence=round(confidence, 2),
            sentiment_score=round(sentiment_score, 2),
            event_score=round(event_score, 2),
            momentum_score=round(momentum_score, 2),
            detected_events=detected_events,
            predicted_impact=n.predicted_impact,
            bullish_factors=bullish[:5],
            bearish_factors=bearish[:5]
        )
    
    def generate_prediction(self) -> CombinedPrediction:
        """Generate final combined prediction."""
        self.fetch_data()
        
        # Generate individual signals
        print("\n🔮 Generating predictions...")
        
        technical = self.generate_technical_signal()
        fundamental = self.generate_fundamental_signal()
        news = self.generate_news_signal()
        
        # === COMBINE SIGNALS ===
        direction_scores = {
            'BULLISH': 1,
            'NEUTRAL': 0,
            'BEARISH': -1
        }
        
        tech_score = direction_scores[technical.direction] * technical.strength * technical.confidence
        fund_score = direction_scores[fundamental.direction] * fundamental.strength * fundamental.confidence
        news_score = direction_scores[news.direction] * news.strength * news.confidence
        
        combined_score = (
            tech_score * self.weights['technical'] +
            fund_score * self.weights['fundamental'] +
            news_score * self.weights['news']
        )
        
        # Final direction
        if combined_score > 0.3:
            direction = "STRONG_BUY"
        elif combined_score > 0.1:
            direction = "BUY"
        elif combined_score > -0.1:
            direction = "HOLD"
        elif combined_score > -0.3:
            direction = "SELL"
        else:
            direction = "STRONG_SELL"
        
        # Combined confidence
        confidence = (
            technical.confidence * self.weights['technical'] +
            fundamental.confidence * self.weights['fundamental'] +
            news.confidence * self.weights['news']
        )
        
        # === PRICE TARGETS ===
        current_price = self.live_data.technicals.price
        atr = self.live_data.technicals.atr_14
        
        if direction in ["STRONG_BUY", "BUY"]:
            target_bull = current_price + 3 * atr
            target_bear = current_price - 1.5 * atr
            stop_loss = current_price - 2 * atr
        elif direction in ["STRONG_SELL", "SELL"]:
            target_bull = current_price + 1.5 * atr
            target_bear = current_price - 3 * atr
            stop_loss = current_price + 2 * atr
        else:
            target_bull = current_price + 2 * atr
            target_bear = current_price - 2 * atr
            stop_loss = current_price - 1.5 * atr
        
        # Risk/Reward
        potential_gain = abs(target_bull - current_price) if direction in ["STRONG_BUY", "BUY"] else abs(current_price - target_bear)
        potential_loss = abs(current_price - stop_loss)
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 1
        
        # === SUMMARY ===
        all_bullish = technical.bullish_factors + fundamental.bullish_factors + news.bullish_factors
        all_bearish = technical.bearish_factors + fundamental.bearish_factors + news.bearish_factors
        
        summary_parts = [
            f"{self.ticker} is rated {direction} with {confidence:.0%} confidence.",
            f"Technical: {technical.direction} | Fundamental: {fundamental.direction} | News: {news.direction}."
        ]
        
        if all_bullish:
            summary_parts.append(f"Key bullish: {all_bullish[0]}")
        if all_bearish:
            summary_parts.append(f"Key bearish: {all_bearish[0]}")
        
        summary = " ".join(summary_parts)
        
        # === ACTION ITEMS ===
        actions = []
        if direction == "STRONG_BUY":
            actions.append(f"Consider buying with stop at ${stop_loss:.2f}")
            actions.append(f"Target 1: ${target_bull:.2f} (+{(target_bull/current_price-1)*100:.1f}%)")
        elif direction == "BUY":
            actions.append(f"Consider scaling into position")
            actions.append(f"Set stop loss at ${stop_loss:.2f}")
        elif direction == "HOLD":
            actions.append("Maintain current position")
            actions.append("Wait for clearer signals")
        elif direction == "SELL":
            actions.append("Consider reducing position")
            actions.append(f"Stop loss for shorts at ${stop_loss:.2f}")
        else:
            actions.append("Consider short position or exit longs")
            actions.append(f"Target: ${target_bear:.2f}")
        
        return CombinedPrediction(
            ticker=self.ticker,
            timestamp=datetime.now().isoformat(),
            direction=direction,
            confidence=round(confidence, 2),
            technical=technical,
            fundamental=fundamental,
            news=news,
            current_price=round(current_price, 2),
            target_price_bull=round(target_bull, 2),
            target_price_bear=round(target_bear, 2),
            stop_loss=round(stop_loss, 2),
            risk_reward_ratio=round(risk_reward, 2),
            volatility=round(self.live_data.technicals.atr_percent, 2),
            summary=summary,
            action_items=actions
        )
    
    def print_prediction(self, pred: CombinedPrediction):
        """Pretty print the prediction."""
        
        direction_colors = {
            "STRONG_BUY": "🟢🟢",
            "BUY": "🟢",
            "HOLD": "🟡",
            "SELL": "🔴",
            "STRONG_SELL": "🔴🔴"
        }
        
        print("\n" + "="*70)
        print(f"STOCK PREDICTION: {pred.ticker}")
        print("="*70)
        
        print(f"\n{direction_colors[pred.direction]} RECOMMENDATION: {pred.direction}")
        print(f"   Confidence: {pred.confidence:.0%}")
        print(f"   Current Price: ${pred.current_price}")
        
        print(f"\n📊 TECHNICAL SIGNAL: {pred.technical.direction} ({pred.technical.strength:.0%} strength)")
        print(f"   Components: Trend={pred.technical.trend_score:+.2f}, Momentum={pred.technical.momentum_score:+.2f}")
        print(f"              Volume={pred.technical.volume_score:+.2f}, Options={pred.technical.options_score:+.2f}")
        print(f"   Bullish: {', '.join(pred.technical.bullish_factors[:2]) if pred.technical.bullish_factors else 'None'}")
        print(f"   Bearish: {', '.join(pred.technical.bearish_factors[:2]) if pred.technical.bearish_factors else 'None'}")
        
        print(f"\n💰 FUNDAMENTAL SIGNAL: {pred.fundamental.direction} ({pred.fundamental.strength:.0%} strength)")
        print(f"   Components: Value={pred.fundamental.value_score:+.2f}, Quality={pred.fundamental.quality_score:+.2f}")
        print(f"              Growth={pred.fundamental.growth_score:+.2f}, Sentiment={pred.fundamental.sentiment_score:+.2f}")
        print(f"   Fair Value: ${pred.fundamental.fair_value} ({pred.fundamental.upside_downside:+.1f}%)")
        print(f"   Bullish: {', '.join(pred.fundamental.bullish_factors[:2]) if pred.fundamental.bullish_factors else 'None'}")
        print(f"   Bearish: {', '.join(pred.fundamental.bearish_factors[:2]) if pred.fundamental.bearish_factors else 'None'}")
        
        print(f"\n📰 NEWS SIGNAL: {pred.news.direction} ({pred.news.strength:.0%} strength)")
        print(f"   Sentiment: {pred.news.sentiment_score:+.2f}")
        if pred.news.detected_events:
            print(f"   Events Detected: {', '.join(pred.news.detected_events)}")
        print(f"   Bullish: {', '.join(pred.news.bullish_factors[:2]) if pred.news.bullish_factors else 'None'}")
        print(f"   Bearish: {', '.join(pred.news.bearish_factors[:2]) if pred.news.bearish_factors else 'None'}")
        
        print(f"\n🎯 PRICE TARGETS")
        print(f"   Bull Target: ${pred.target_price_bull} (+{(pred.target_price_bull/pred.current_price-1)*100:.1f}%)")
        print(f"   Bear Target: ${pred.target_price_bear} ({(pred.target_price_bear/pred.current_price-1)*100:.1f}%)")
        print(f"   Stop Loss:   ${pred.stop_loss}")
        print(f"   Risk/Reward: {pred.risk_reward_ratio:.2f}")
        print(f"   Volatility:  {pred.volatility}% (ATR %)")
        
        print(f"\n📋 ACTION ITEMS:")
        for action in pred.action_items:
            print(f"   • {action}")
        
        print(f"\n📝 SUMMARY:")
        print(f"   {pred.summary}")
        
        print("\n" + "="*70)
    
    def to_dict(self, pred: CombinedPrediction) -> Dict:
        """Convert prediction to dictionary."""
        return {
            'ticker': pred.ticker,
            'timestamp': pred.timestamp,
            'direction': pred.direction,
            'confidence': pred.confidence,
            'technical': asdict(pred.technical),
            'fundamental': asdict(pred.fundamental),
            'news': asdict(pred.news),
            'current_price': pred.current_price,
            'target_price_bull': pred.target_price_bull,
            'target_price_bear': pred.target_price_bear,
            'stop_loss': pred.stop_loss,
            'risk_reward_ratio': pred.risk_reward_ratio,
            'volatility': pred.volatility,
            'summary': pred.summary,
            'action_items': pred.action_items
        }


def predict(ticker: str) -> CombinedPrediction:
    """Convenience function to generate prediction for a ticker."""
    predictor = StockPredictor(ticker)
    return predictor.generate_prediction()


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else "IONQ"
    
    predictor = StockPredictor(ticker)
    prediction = predictor.generate_prediction()
    predictor.print_prediction(prediction)
    
    # Save to file
    output_dir = Path("predictor/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{ticker}_prediction.json"
    with open(output_file, 'w') as f:
        json.dump(predictor.to_dict(prediction), f, indent=2)
    
    print(f"\nPrediction saved to {output_file}")
