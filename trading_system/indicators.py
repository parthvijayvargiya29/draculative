"""Layer 2: Indicator Calculation - MACD, Bollinger Bands, Stochastic."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MACDValues:
    """MACD indicator values."""
    macd_line: float
    signal_line: float
    histogram: float
    is_bullish: bool  # histogram > 0 and increasing
    momentum: str  # 'strong_up', 'weak_up', 'weak_down', 'strong_down'


@dataclass
class BollingerBandsValues:
    """Bollinger Bands values."""
    upper_band: float
    middle_band: float
    lower_band: float
    bandwidth: float  # (upper - lower) / middle
    position: float  # (close - lower) / (upper - lower), 0-1
    is_bullish: bool  # close near upper band
    is_squeeze: bool  # bandwidth < 20th percentile


@dataclass
class StochasticValues:
    """Stochastic oscillator values."""
    k: float  # %K (0-100)
    d: float  # %D (SMA of %K, 0-100)
    is_bullish: bool  # K > D and K < 80
    is_overbought: bool  # K > 80
    is_oversold: bool  # K < 20


class MACDCalculator:
    """Calculate MACD (momentum indicator)."""
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(data) < period:
            return np.full_like(data, np.nan)
        ema_vals = np.empty_like(data)
        ema_vals[:period] = np.nan
        multiplier = 2.0 / (period + 1)
        ema_vals[period] = np.mean(data[:period])
        for i in range(period + 1, len(data)):
            ema_vals[i] = data[i] * multiplier + ema_vals[i - 1] * (1 - multiplier)
        return ema_vals
    
    @staticmethod
    def calculate(close_prices: np.ndarray) -> Optional[MACDValues]:
        """Calculate MACD (12/26/9 periods)."""
        if len(close_prices) < 26:
            return None
        
        ema12 = MACDCalculator.ema(close_prices, 12)
        ema26 = MACDCalculator.ema(close_prices, 26)
        
        macd_line = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        signal = MACDCalculator.ema(macd_line, 9)
        
        # Histogram
        histogram = macd_line - signal
        
        # Get latest values
        macd_val = macd_line[-1]
        signal_val = signal[-1]
        hist_val = histogram[-1]
        
        # Determine bullish/bearish
        is_bullish = hist_val > 0
        
        # Momentum strength
        if len(histogram) >= 2:
            hist_prev = histogram[-2]
            if is_bullish and hist_val > hist_prev:
                momentum = 'strong_up'
            elif is_bullish and hist_val <= hist_prev:
                momentum = 'weak_up'
            elif not is_bullish and hist_val < hist_prev:
                momentum = 'strong_down'
            else:
                momentum = 'weak_down'
        else:
            momentum = 'neutral'
        
        return MACDValues(
            macd_line=float(macd_val),
            signal_line=float(signal_val),
            histogram=float(hist_val),
            is_bullish=bool(is_bullish),
            momentum=momentum
        )


class BollingerBandsCalculator:
    """Calculate Bollinger Bands (volatility)."""
    
    @staticmethod
    def calculate(close_prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> Optional[BollingerBandsValues]:
        """Calculate Bollinger Bands (20-period, 2 std dev)."""
        if len(close_prices) < period:
            return None
        
        close = close_prices[-period:]
        middle = np.mean(close)
        std = np.std(close)
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        bandwidth = (upper - lower) / middle if middle != 0 else 0
        
        close_val = close_prices[-1]
        position = (close_val - lower) / (upper - lower) if upper != lower else 0.5
        
        is_bullish = position > 0.7  # Near upper band
        is_squeeze = bandwidth < 0.05  # Low volatility
        
        return BollingerBandsValues(
            upper_band=float(upper),
            middle_band=float(middle),
            lower_band=float(lower),
            bandwidth=float(bandwidth),
            position=float(position),
            is_bullish=bool(is_bullish),
            is_squeeze=bool(is_squeeze)
        )


class StochasticCalculator:
    """Calculate Stochastic Oscillator (momentum + overbought/oversold)."""
    
    @staticmethod
    def calculate(close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray,
                  period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Optional[StochasticValues]:
        """Calculate Stochastic %K and %D."""
        if len(close_prices) < period:
            return None
        
        # Get current period's high and low
        period_high = np.max(high_prices[-period:])
        period_low = np.min(low_prices[-period:])
        
        close_val = close_prices[-1]
        
        # Raw %K
        if period_high == period_low:
            raw_k = 50.0
        else:
            raw_k = 100.0 * (close_val - period_low) / (period_high - period_low)
        
        # Smooth %K (moving average of raw K)
        # For simplicity, use current value as approximation
        k_val = raw_k
        
        # %D is SMA of %K (for simplicity, use K-smooth approximation)
        d_val = raw_k  # Ideally would maintain rolling K values
        
        # Determine signals
        is_bullish = k_val > d_val and k_val < 80
        is_overbought = k_val > 80
        is_oversold = k_val < 20
        
        return StochasticValues(
            k=float(k_val),
            d=float(d_val),
            is_bullish=bool(is_bullish),
            is_overbought=bool(is_overbought),
            is_oversold=bool(is_oversold)
        )


class IndicatorEngine:
    """Unified engine for calculating all indicators."""
    
    def __init__(self):
        self.macd_calc = MACDCalculator()
        self.bb_calc = BollingerBandsCalculator()
        self.stoch_calc = StochasticCalculator()
    
    def calculate_all(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict:
        """Calculate all three indicators in one call."""
        results = {
            'macd': self.macd_calc.calculate(close),
            'bollinger': self.bb_calc.calculate(close),
            'stochastic': self.stoch_calc.calculate(close, high, low)
        }
        return results
    
    def get_signal_strengths(self, indicators: dict) -> dict:
        """Convert indicators to buy (1), neutral (0), sell (-1) scores."""
        scores = {
            'macd': 0,
            'bollinger': 0,
            'stochastic': 0
        }
        
        if indicators['macd']:
            macd = indicators['macd']
            if macd.momentum == 'strong_up':
                scores['macd'] = 1
            elif macd.momentum == 'weak_up':
                scores['macd'] = 0.5
            elif macd.momentum == 'weak_down':
                scores['macd'] = -0.5
            else:
                scores['macd'] = -1
        
        if indicators['bollinger']:
            bb = indicators['bollinger']
            if bb.is_bullish and not bb.is_squeeze:
                scores['bollinger'] = 1
            elif bb.is_squeeze:
                scores['bollinger'] = 0  # Neutral in squeeze
            else:
                scores['bollinger'] = -1
        
        if indicators['stochastic']:
            stoch = indicators['stochastic']
            if stoch.is_overbought:
                scores['stochastic'] = -1
            elif stoch.is_oversold:
                scores['stochastic'] = 1
            elif stoch.is_bullish:
                scores['stochastic'] = 0.5
            else:
                scores['stochastic'] = -0.5
        
        return scores


# Demo function
def demo_indicators():
    """Demo: Calculate indicators on sample data."""
    # Generate sample price data (200 candles)
    np.random.seed(42)
    close = 132.50 + np.cumsum(np.random.randn(200) * 0.5)
    high = close + np.abs(np.random.randn(200) * 0.3)
    low = close - np.abs(np.random.randn(200) * 0.3)
    
    engine = IndicatorEngine()
    indicators = engine.calculate_all(close, high, low)
    scores = engine.get_signal_strengths(indicators)
    
    print("\n📊 INDICATOR CALCULATION DEMO")
    print(f"Latest Close: ${close[-1]:.2f}")
    print(f"\nMACD: {indicators['macd']}")
    print(f"Bollinger Bands: {indicators['bollinger']}")
    print(f"Stochastic: {indicators['stochastic']}")
    print(f"\nSignal Scores: {scores}")
    print(f"Total Score: {sum(scores.values()):.2f}/3.0")


if __name__ == '__main__':
    demo_indicators()
