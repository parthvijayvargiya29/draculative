"""Technical Indicators Calculator.

Implements all standard oscillators, moving averages, and pivot points
extracted from TradingView's technical analysis methodology.

Usage:
    from indicators import TechnicalIndicators
    ti = TechnicalIndicators(df)  # df must have OHLCV columns
    ti.compute_all()
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class TechnicalIndicators:
    """Compute technical indicators on OHLCV data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with OHLCV DataFrame.
        
        Expected columns: open, high, low, close, volume (case-insensitive)
        """
        self.df = df.copy()
        # Normalize column names
        self.df.columns = [c.lower() for c in self.df.columns]
        self._validate_columns()
    
    def _validate_columns(self):
        required = {'open', 'high', 'low', 'close'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # ========== OSCILLATORS ==========
    
    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI).
        
        Overbought > 70, Oversold < 30.
        Bullish divergence: price makes lower low, RSI makes higher low.
        """
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        self.df[f'rsi_{period}'] = rsi
        return rsi
    
    def stochastic(self, k_period: int = 14, d_period: int = 3, smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K, %D).
        
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K
        
        Signals: %K crossing above %D = bullish, below = bearish.
        Overbought > 80, Oversold < 20.
        """
        low_min = self.df['low'].rolling(k_period).min()
        high_max = self.df['high'].rolling(k_period).max()
        
        stoch_k = 100 * (self.df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        stoch_k = stoch_k.rolling(smooth).mean()  # Smoothed %K
        stoch_d = stoch_k.rolling(d_period).mean()
        
        self.df[f'stoch_k_{k_period}'] = stoch_k
        self.df[f'stoch_d_{k_period}'] = stoch_d
        return stoch_k, stoch_d
    
    def stochastic_rsi(self, rsi_period: int = 14, stoch_period: int = 14, 
                       k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic RSI - RSI of RSI with stochastic formula."""
        rsi = self.rsi(rsi_period)
        
        rsi_low = rsi.rolling(stoch_period).min()
        rsi_high = rsi.rolling(stoch_period).max()
        
        stoch_rsi_k = 100 * (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
        stoch_rsi_k = stoch_rsi_k.rolling(k_period).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(d_period).mean()
        
        self.df['stoch_rsi_k'] = stoch_rsi_k
        self.df['stoch_rsi_d'] = stoch_rsi_d
        return stoch_rsi_k, stoch_rsi_d
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence.
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA of MACD Line
        Histogram = MACD Line - Signal Line
        
        Bullish: MACD crosses above signal / histogram turns positive.
        Bearish: MACD crosses below signal / histogram turns negative.
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        self.df['macd'] = macd_line
        self.df['macd_signal'] = signal_line
        self.df['macd_histogram'] = histogram
        return macd_line, signal_line, histogram
    
    def cci(self, period: int = 20) -> pd.Series:
        """Commodity Channel Index (CCI).
        
        CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        Overbought > 100, Oversold < -100.
        """
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma) / (0.015 * mad)
        self.df[f'cci_{period}'] = cci
        return cci
    
    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R.
        
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        Overbought > -20, Oversold < -80.
        """
        high_max = self.df['high'].rolling(period).max()
        low_min = self.df['low'].rolling(period).min()
        
        wr = -100 * (high_max - self.df['close']) / (high_max - low_min).replace(0, np.nan)
        self.df[f'williams_r_{period}'] = wr
        return wr
    
    def adx(self, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index (ADX) with +DI and -DI.
        
        ADX > 25 = strong trend, < 20 = weak/ranging.
        +DI > -DI = bullish, +DI < -DI = bearish.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        
        self.df['adx'] = adx
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        return adx, plus_di, minus_di
    
    def awesome_oscillator(self) -> pd.Series:
        """Awesome Oscillator (AO).
        
        AO = SMA(5) of Median Price - SMA(34) of Median Price
        Green bar = AO > previous, Red = AO < previous.
        """
        median_price = (self.df['high'] + self.df['low']) / 2
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        self.df['awesome_oscillator'] = ao
        return ao
    
    def momentum(self, period: int = 10) -> pd.Series:
        """Momentum indicator.
        
        Momentum = Close - Close(n periods ago)
        """
        mom = self.df['close'] - self.df['close'].shift(period)
        self.df[f'momentum_{period}'] = mom
        return mom
    
    def ultimate_oscillator(self, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
        """Ultimate Oscillator.
        
        Combines short, medium, and long-term momentum.
        Overbought > 70, Oversold < 30.
        """
        close = self.df['close']
        low = self.df['low']
        high = self.df['high']
        
        prev_close = close.shift(1)
        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
        
        avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
        avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
        avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
        
        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        self.df['ultimate_oscillator'] = uo
        return uo
    
    def bull_bear_power(self, period: int = 13) -> Tuple[pd.Series, pd.Series]:
        """Bull/Bear Power (Elder Ray Index).
        
        Bull Power = High - EMA
        Bear Power = Low - EMA
        """
        ema = self.df['close'].ewm(span=period, adjust=False).mean()
        bull = self.df['high'] - ema
        bear = self.df['low'] - ema
        
        self.df['bull_power'] = bull
        self.df['bear_power'] = bear
        return bull, bear
    
    # ========== MOVING AVERAGES ==========
    
    def sma(self, period: int) -> pd.Series:
        """Simple Moving Average."""
        sma = self.df['close'].rolling(period).mean()
        self.df[f'sma_{period}'] = sma
        return sma
    
    def ema(self, period: int) -> pd.Series:
        """Exponential Moving Average."""
        ema = self.df['close'].ewm(span=period, adjust=False).mean()
        self.df[f'ema_{period}'] = ema
        return ema
    
    def wma(self, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        wma = self.df['close'].rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        self.df[f'wma_{period}'] = wma
        return wma
    
    def vwma(self, period: int = 20) -> pd.Series:
        """Volume Weighted Moving Average."""
        if 'volume' not in self.df.columns:
            return pd.Series(dtype=float)
        
        vwma = (self.df['close'] * self.df['volume']).rolling(period).sum() / \
               self.df['volume'].rolling(period).sum()
        self.df[f'vwma_{period}'] = vwma
        return vwma
    
    def hull_ma(self, period: int = 9) -> pd.Series:
        """Hull Moving Average - Reduces lag."""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self.wma(half_period)
        wma_full = self.wma(period)
        
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(sqrt_period).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_period + 1)) / np.arange(1, sqrt_period + 1).sum(), raw=True
        )
        self.df[f'hull_ma_{period}'] = hma
        return hma
    
    def ichimoku(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, 
                 displacement: int = 26) -> dict:
        """Ichimoku Cloud.
        
        - Tenkan-sen (Conversion): (9-period high + low) / 2
        - Kijun-sen (Base): (26-period high + low) / 2
        - Senkou Span A: (Tenkan + Kijun) / 2, shifted forward
        - Senkou Span B: (52-period high + low) / 2, shifted forward
        - Chikou Span: Close shifted backward
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(displacement)
        chikou_span = close.shift(-displacement)
        
        self.df['ichimoku_tenkan'] = tenkan_sen
        self.df['ichimoku_kijun'] = kijun_sen
        self.df['ichimoku_senkou_a'] = senkou_span_a
        self.df['ichimoku_senkou_b'] = senkou_span_b
        self.df['ichimoku_chikou'] = chikou_span
        
        return {
            'tenkan': tenkan_sen,
            'kijun': kijun_sen,
            'senkou_a': senkou_span_a,
            'senkou_b': senkou_span_b,
            'chikou': chikou_span
        }
    
    def all_moving_averages(self, periods: list = [10, 20, 30, 50, 100, 200]) -> dict:
        """Compute SMA and EMA for multiple periods."""
        result = {}
        for p in periods:
            result[f'sma_{p}'] = self.sma(p)
            result[f'ema_{p}'] = self.ema(p)
        return result
    
    # ========== PIVOT POINTS ==========
    
    def pivot_classic(self) -> dict:
        """Classic Pivot Points.
        
        P = (H + L + C) / 3
        R1 = 2P - L, S1 = 2P - H
        R2 = P + (H - L), S2 = P - (H - L)
        R3 = H + 2(P - L), S3 = L - 2(H - P)
        """
        h, l, c = self.df['high'].iloc[-1], self.df['low'].iloc[-1], self.df['close'].iloc[-1]
        
        p = (h + l + c) / 3
        r1 = 2 * p - l
        s1 = 2 * p - h
        r2 = p + (h - l)
        s2 = p - (h - l)
        r3 = h + 2 * (p - l)
        s3 = l - 2 * (h - p)
        
        return {'P': p, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}
    
    def pivot_fibonacci(self) -> dict:
        """Fibonacci Pivot Points.
        
        Uses Fibonacci ratios (0.382, 0.618, 1.0) for support/resistance.
        """
        h, l, c = self.df['high'].iloc[-1], self.df['low'].iloc[-1], self.df['close'].iloc[-1]
        
        p = (h + l + c) / 3
        r = h - l
        
        r1 = p + 0.382 * r
        r2 = p + 0.618 * r
        r3 = p + 1.0 * r
        s1 = p - 0.382 * r
        s2 = p - 0.618 * r
        s3 = p - 1.0 * r
        
        return {'P': p, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}
    
    def pivot_camarilla(self) -> dict:
        """Camarilla Pivot Points.
        
        Tighter levels, good for intraday trading.
        """
        h, l, c = self.df['high'].iloc[-1], self.df['low'].iloc[-1], self.df['close'].iloc[-1]
        
        r = h - l
        r1 = c + r * 1.1 / 12
        r2 = c + r * 1.1 / 6
        r3 = c + r * 1.1 / 4
        r4 = c + r * 1.1 / 2
        s1 = c - r * 1.1 / 12
        s2 = c - r * 1.1 / 6
        s3 = c - r * 1.1 / 4
        s4 = c - r * 1.1 / 2
        
        return {'R1': r1, 'R2': r2, 'R3': r3, 'R4': r4, 'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4}
    
    def pivot_woodie(self) -> dict:
        """Woodie Pivot Points.
        
        Gives more weight to the close price.
        """
        h, l, c = self.df['high'].iloc[-1], self.df['low'].iloc[-1], self.df['close'].iloc[-1]
        
        p = (h + l + 2 * c) / 4
        r1 = 2 * p - l
        r2 = p + h - l
        s1 = 2 * p - h
        s2 = p - h + l
        
        return {'P': p, 'R1': r1, 'R2': r2, 'S1': s1, 'S2': s2}
    
    def pivot_demark(self) -> dict:
        """DeMark Pivot Points.
        
        Uses relationship between open and close.
        """
        h, l, c = self.df['high'].iloc[-1], self.df['low'].iloc[-1], self.df['close'].iloc[-1]
        o = self.df['open'].iloc[-1]
        
        if c < o:
            x = h + 2 * l + c
        elif c > o:
            x = 2 * h + l + c
        else:
            x = h + l + 2 * c
        
        p = x / 4
        r1 = x / 2 - l
        s1 = x / 2 - h
        
        return {'P': p, 'R1': r1, 'S1': s1}
    
    # ========== VOLATILITY ==========
    
    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        self.df[f'atr_{period}'] = atr
        return atr
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.
        
        Middle = SMA, Upper/Lower = SMA ± (std_dev * StdDev)
        """
        middle = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower
        return upper, middle, lower
    
    def keltner_channel(self, period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channel.
        
        Middle = EMA, Upper/Lower = EMA ± (atr_mult * ATR)
        """
        middle = self.df['close'].ewm(span=period, adjust=False).mean()
        atr = self.atr(period)
        
        upper = middle + atr_mult * atr
        lower = middle - atr_mult * atr
        
        self.df['kc_upper'] = upper
        self.df['kc_middle'] = middle
        self.df['kc_lower'] = lower
        return upper, middle, lower
    
    # ========== VOLUME ==========
    
    def obv(self) -> pd.Series:
        """On-Balance Volume."""
        if 'volume' not in self.df.columns:
            return pd.Series(dtype=float)
        
        sign = np.sign(self.df['close'].diff())
        obv = (sign * self.df['volume']).cumsum()
        self.df['obv'] = obv
        return obv
    
    def vwap(self) -> pd.Series:
        """Volume Weighted Average Price (intraday)."""
        if 'volume' not in self.df.columns:
            return pd.Series(dtype=float)
        
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (tp * self.df['volume']).cumsum() / self.df['volume'].cumsum()
        self.df['vwap'] = vwap
        return vwap
    
    # ========== SIGNALS ==========
    
    def generate_signals(self) -> dict:
        """Generate buy/sell signals based on indicators."""
        signals = {}
        
        # RSI signals
        if 'rsi_14' in self.df.columns:
            rsi = self.df['rsi_14'].iloc[-1]
            if rsi < 30:
                signals['rsi'] = 'BUY'
            elif rsi > 70:
                signals['rsi'] = 'SELL'
            else:
                signals['rsi'] = 'NEUTRAL'
        
        # MACD signals
        if 'macd' in self.df.columns and 'macd_signal' in self.df.columns:
            macd = self.df['macd'].iloc[-1]
            signal = self.df['macd_signal'].iloc[-1]
            if macd > signal:
                signals['macd'] = 'BUY'
            else:
                signals['macd'] = 'SELL'
        
        # Moving average crossovers
        if 'sma_50' in self.df.columns and 'sma_200' in self.df.columns:
            if self.df['sma_50'].iloc[-1] > self.df['sma_200'].iloc[-1]:
                signals['golden_cross'] = 'BUY'  # Golden Cross
            else:
                signals['golden_cross'] = 'SELL'  # Death Cross
        
        # ADX trend strength
        if 'adx' in self.df.columns:
            adx = self.df['adx'].iloc[-1]
            plus_di = self.df['plus_di'].iloc[-1]
            minus_di = self.df['minus_di'].iloc[-1]
            
            if adx > 25:
                signals['adx_trend'] = 'STRONG'
                if plus_di > minus_di:
                    signals['adx_direction'] = 'BUY'
                else:
                    signals['adx_direction'] = 'SELL'
            else:
                signals['adx_trend'] = 'WEAK'
                signals['adx_direction'] = 'NEUTRAL'
        
        return signals
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all standard indicators."""
        # Oscillators
        self.rsi()
        self.stochastic()
        self.stochastic_rsi()
        self.macd()
        self.cci()
        self.williams_r()
        self.adx()
        self.awesome_oscillator()
        self.momentum()
        self.ultimate_oscillator()
        self.bull_bear_power()
        
        # Moving averages
        self.all_moving_averages()
        self.vwma()
        self.hull_ma()
        self.ichimoku()
        
        # Volatility
        self.atr()
        self.bollinger_bands()
        self.keltner_channel()
        
        # Volume
        self.obv()
        self.vwap()
        
        return self.df


# ========== TRADING STRATEGIES ==========

class TradingStrategies:
    """Common trading strategies based on technical indicators."""
    
    @staticmethod
    def rsi_divergence(df: pd.DataFrame, rsi_col: str = 'rsi_14', 
                       lookback: int = 14) -> pd.Series:
        """Detect RSI divergence (bullish/bearish).
        
        Bullish: Price makes lower low, RSI makes higher low.
        Bearish: Price makes higher high, RSI makes lower high.
        """
        price = df['close']
        rsi = df[rsi_col]
        
        divergence = pd.Series(0, index=df.index)
        
        for i in range(lookback, len(df)):
            # Check for bullish divergence
            price_low_idx = price.iloc[i-lookback:i+1].idxmin()
            rsi_at_price_low = rsi.loc[price_low_idx]
            
            prev_price_low = price.iloc[i-lookback:i].min()
            prev_rsi_low = rsi.iloc[i-lookback:i].min()
            
            if price.iloc[i] < prev_price_low and rsi.iloc[i] > prev_rsi_low:
                divergence.iloc[i] = 1  # Bullish divergence
            
            # Check for bearish divergence
            if price.iloc[i] > price.iloc[i-lookback:i].max() and rsi.iloc[i] < rsi.iloc[i-lookback:i].max():
                divergence.iloc[i] = -1  # Bearish divergence
        
        return divergence
    
    @staticmethod
    def ma_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.Series:
        """Moving average crossover strategy.
        
        Buy when fast MA crosses above slow MA.
        Sell when fast MA crosses below slow MA.
        """
        fast_ma = df['close'].ewm(span=fast, adjust=False).mean()
        slow_ma = df['close'].ewm(span=slow, adjust=False).mean()
        
        signal = pd.Series(0, index=df.index)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        
        # Generate crossover signals
        crossover = signal.diff()
        return crossover
    
    @staticmethod
    def bollinger_squeeze(df: pd.DataFrame, bb_period: int = 20, 
                          kc_period: int = 20) -> pd.Series:
        """Bollinger Band squeeze detection.
        
        Squeeze occurs when BB is inside KC (low volatility).
        Breakout expected when squeeze ends.
        """
        ti = TechnicalIndicators(df)
        ti.bollinger_bands(bb_period)
        ti.keltner_channel(kc_period)
        
        squeeze = (ti.df['bb_lower'] > ti.df['kc_lower']) & (ti.df['bb_upper'] < ti.df['kc_upper'])
        return squeeze.astype(int)


if __name__ == '__main__':
    # Demo with sample data
    import yfinance as yf
    
    ticker = yf.Ticker('NVDA')
    df = ticker.history(period='1y')
    
    ti = TechnicalIndicators(df)
    ti.compute_all()
    
    print(f"Computed {len(ti.df.columns)} columns")
    print("\nLatest values:")
    print(ti.df[['close', 'rsi_14', 'macd', 'sma_50', 'sma_200', 'adx']].tail())
    
    print("\nPivot Points (Classic):")
    print(ti.pivot_classic())
    
    print("\nSignals:")
    print(ti.generate_signals())
