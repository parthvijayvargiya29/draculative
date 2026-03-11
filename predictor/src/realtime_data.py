#!/usr/bin/env python3
"""
Real-Time Data Fetcher

Dynamically fetches live market data including:
- OHLCV price data
- Options chain (IV, Greeks, OI, Max Pain)
- Technical indicators (RSI, MACD, VWAP, etc.)
- Fundamental metrics
- Market microstructure data
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class OptionsData:
    """Container for options chain data."""
    atm_iv_call: float
    atm_iv_put: float
    put_call_ratio_volume: float
    put_call_ratio_oi: float
    max_pain: float
    iv_percentile: float
    iv_skew: float
    total_call_oi: int
    total_put_oi: int
    total_call_volume: int
    total_put_volume: int
    nearest_expiry: str
    unusual_activity: List[Dict]


@dataclass
class TechnicalData:
    """Container for technical indicators."""
    # Price
    price: float
    change_1d: float
    change_5d: float
    change_20d: float
    
    # VWAP
    vwap: float
    vwap_distance: float
    
    # Moving Averages
    sma_9: float
    sma_20: float
    sma_50: float
    sma_100: float
    sma_200: float
    ema_12: float
    ema_26: float
    price_vs_sma_20: str  # above/below
    price_vs_sma_50: str
    price_vs_sma_200: str
    golden_cross: bool  # SMA50 > SMA200
    
    # Oscillators
    rsi_14: float
    rsi_signal: str  # overbought/oversold/neutral
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_cross: str  # bullish/bearish/none
    macd_histogram_trend: str  # expanding/contracting/flat
    stochastic_k: float
    stochastic_d: float
    stoch_rsi_k: float   # Stochastic RSI %K (0-100)
    stoch_rsi_d: float   # Stochastic RSI %D (0-100)
    stoch_rsi_signal: str  # overbought/oversold/neutral
    ppo: float           # Percentage Price Oscillator line
    ppo_signal: float    # PPO signal line
    ppo_histogram: float # PPO histogram
    ppo_cross: str       # bullish/bearish/none
    pvo: float           # Percentage Volume Oscillator
    pvo_signal: float    # PVO signal line
    pvo_cross: str       # bullish/bearish/none
    
    # Volatility
    atr_14: float
    atr_percent: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_position: float    # 0-1 position within bands (%B)
    bollinger_bandwidth: float   # Band width as % of middle
    bollinger_squeeze: bool      # True when bandwidth < 20-period low
    
    # Pivot Points (Standard / Classic)
    pivot_p: float
    pivot_r1: float
    pivot_r2: float
    pivot_r3: float
    pivot_s1: float
    pivot_s2: float
    pivot_s3: float
    price_vs_pivot: str  # above_r1/above_p/between_s1_p/below_s1/etc.
    
    # Fibonacci Retracement (swing high → swing low over last 50 bars)
    fib_swing_high: float
    fib_swing_low: float
    fib_ret_236: float
    fib_ret_382: float
    fib_ret_500: float
    fib_ret_618: float
    fib_ret_786: float
    fib_nearest_level: str   # label of nearest fib level
    fib_nearest_distance: float  # % distance to nearest level
    
    # VIX
    vix: float
    vix_signal: str  # extreme_fear/fear/neutral/complacency
    
    # Volume
    volume: int
    volume_sma_20: float
    volume_ratio: float  # current vs average
    obv_trend: str  # up/down/flat
    
    # Trend
    adx: float
    trend_strength: str  # strong/weak/none
    plus_di: float
    minus_di: float
    trend_direction: str  # bullish/bearish


@dataclass
class FundamentalData:
    """Container for fundamental data."""
    market_cap: float
    pe_ratio: float
    forward_pe: float
    peg_ratio: float
    pb_ratio: float
    ps_ratio: float
    ev_ebitda: float
    
    # Profitability
    roe: float
    roa: float
    gross_margin: float
    operating_margin: float
    profit_margin: float
    
    # Growth
    revenue_growth: float
    earnings_growth: float
    
    # Financial Health
    current_ratio: float
    debt_to_equity: float
    
    # Analyst
    analyst_rating: str
    target_price: float
    upside_potential: float
    num_analysts: int


@dataclass
class LiveMarketData:
    """Complete live market snapshot."""
    ticker: str
    timestamp: str
    technicals: TechnicalData
    options: Optional[OptionsData]
    fundamentals: FundamentalData


class RealTimeDataFetcher:
    """Fetch and compute real-time market data."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._price_data = None
        self._info = None
        
    def fetch_all(self) -> LiveMarketData:
        """Fetch all live data for the ticker."""
        print(f"Fetching live data for {self.ticker}...")
        
        # Fetch price data (last 200 days for indicator calculation)
        self._price_data = self._fetch_price_data()
        self._info = self.stock.info
        
        # Compute all components
        technicals = self._compute_technicals()
        options = self._fetch_options()
        fundamentals = self._fetch_fundamentals()
        
        return LiveMarketData(
            ticker=self.ticker,
            timestamp=datetime.now().isoformat(),
            technicals=technicals,
            options=options,
            fundamentals=fundamentals
        )
    
    def _fetch_price_data(self) -> pd.DataFrame:
        """Fetch historical price data."""
        df = self.stock.history(period="1y", interval="1d")
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        return df
    
    def _compute_technicals(self) -> TechnicalData:
        """Compute all technical indicators."""
        df = self._price_data
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        current_price = close.iloc[-1]
        
        # Price changes
        change_1d = (current_price / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0
        change_5d = (current_price / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
        change_20d = (current_price / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
        
        # VWAP (intraday approximation using daily data)
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).sum() / volume.sum()
        vwap_distance = (current_price - vwap) / vwap * 100
        
        # Moving Averages — SMA 9, 20, 50, 100, 200
        sma_9  = close.rolling(9).mean().iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_100 = close.rolling(100).mean().iloc[-1] if len(close) >= 100 else close.mean()
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = rsi_series.iloc[-1]
        rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        # MACD (12, 26, 9)
        macd_series = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        macd_hist_series = macd_series - signal_series
        macd_line = macd_series.iloc[-1]
        macd_signal_line = signal_series.iloc[-1]
        macd_hist = macd_hist_series.iloc[-1]
        
        if macd_series.iloc[-1] > signal_series.iloc[-1] and macd_series.iloc[-2] <= signal_series.iloc[-2]:
            macd_cross = "bullish"
        elif macd_series.iloc[-1] < signal_series.iloc[-1] and macd_series.iloc[-2] >= signal_series.iloc[-2]:
            macd_cross = "bearish"
        else:
            macd_cross = "none"
        
        # MACD histogram momentum (expanding = histogram growing in same direction)
        if len(macd_hist_series) >= 3:
            h1, h2, h3 = macd_hist_series.iloc[-3], macd_hist_series.iloc[-2], macd_hist_series.iloc[-1]
            if h3 > h2 > h1:
                macd_histogram_trend = "expanding_bullish"
            elif h3 < h2 < h1:
                macd_histogram_trend = "expanding_bearish"
            elif abs(h3) < abs(h2):
                macd_histogram_trend = "contracting"
            else:
                macd_histogram_trend = "flat"
        else:
            macd_histogram_trend = "flat"
        
        # Stochastic (14, 3, 3)
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        raw_k = (close - low_14) / (high_14 - low_14).replace(0, np.nan) * 100
        stoch_k_series = raw_k.rolling(3).mean()   # smoothed %K
        stoch_d_series = stoch_k_series.rolling(3).mean()
        stoch_k = stoch_k_series.iloc[-1]
        stoch_d = stoch_d_series.iloc[-1]
        
        # Stochastic RSI (14, 14, 3, 3)
        rsi_low  = rsi_series.rolling(14).min()
        rsi_high = rsi_series.rolling(14).max()
        stoch_rsi_raw = (rsi_series - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan) * 100
        stoch_rsi_k_series = stoch_rsi_raw.rolling(3).mean()
        stoch_rsi_d_series = stoch_rsi_k_series.rolling(3).mean()
        stoch_rsi_k = stoch_rsi_k_series.iloc[-1]
        stoch_rsi_d = stoch_rsi_d_series.iloc[-1]
        stoch_rsi_signal = (
            "overbought" if stoch_rsi_k > 80 else
            "oversold"   if stoch_rsi_k < 20 else
            "neutral"
        )
        
        # Percentage Price Oscillator (PPO) — like MACD but as % of slow EMA
        ema_12_series = close.ewm(span=12, adjust=False).mean()
        ema_26_series = close.ewm(span=26, adjust=False).mean()
        ppo_series    = (ema_12_series - ema_26_series) / ema_26_series.replace(0, np.nan) * 100
        ppo_sig_series = ppo_series.ewm(span=9, adjust=False).mean()
        ppo_hist_series = ppo_series - ppo_sig_series
        ppo  = ppo_series.iloc[-1]
        ppo_sig  = ppo_sig_series.iloc[-1]
        ppo_hist = ppo_hist_series.iloc[-1]
        if ppo_series.iloc[-1] > ppo_sig_series.iloc[-1] and ppo_series.iloc[-2] <= ppo_sig_series.iloc[-2]:
            ppo_cross = "bullish"
        elif ppo_series.iloc[-1] < ppo_sig_series.iloc[-1] and ppo_series.iloc[-2] >= ppo_sig_series.iloc[-2]:
            ppo_cross = "bearish"
        else:
            ppo_cross = "none"
        
        # Percentage Volume Oscillator (PVO) — fast/slow EMAs on volume
        vol_ema_fast = volume.ewm(span=12, adjust=False).mean()
        vol_ema_slow = volume.ewm(span=26, adjust=False).mean()
        pvo_series   = (vol_ema_fast - vol_ema_slow) / vol_ema_slow.replace(0, np.nan) * 100
        pvo_sig_series = pvo_series.ewm(span=9, adjust=False).mean()
        pvo  = pvo_series.iloc[-1]
        pvo_sig = pvo_sig_series.iloc[-1]
        if pvo_series.iloc[-1] > pvo_sig_series.iloc[-1] and pvo_series.iloc[-2] <= pvo_sig_series.iloc[-2]:
            pvo_cross = "bullish"
        elif pvo_series.iloc[-1] < pvo_sig_series.iloc[-1] and pvo_series.iloc[-2] >= pvo_sig_series.iloc[-2]:
            pvo_cross = "bearish"
        else:
            pvo_cross = "none"
        
        # ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
        atr_percent = atr / current_price * 100
        
        # Bollinger Bands (20, 2)
        bb_middle_series = close.rolling(20).mean()
        bb_std_series    = close.rolling(20).std()
        bb_upper_series  = bb_middle_series + 2 * bb_std_series
        bb_lower_series  = bb_middle_series - 2 * bb_std_series
        bb_upper_val  = bb_upper_series.iloc[-1]
        bb_middle_val = bb_middle_series.iloc[-1]
        bb_lower_val  = bb_lower_series.iloc[-1]
        bb_position   = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) if (bb_upper_val - bb_lower_val) != 0 else 0.5
        bb_bandwidth  = (bb_upper_val - bb_lower_val) / bb_middle_val * 100 if bb_middle_val != 0 else 0
        # Bollinger squeeze: current bandwidth < 20-bar rolling min of bandwidth
        bw_series = (bb_upper_series - bb_lower_series) / bb_middle_series.replace(0, np.nan) * 100
        bw_min_20 = bw_series.rolling(20).min()
        bb_squeeze = bool(bw_series.iloc[-1] <= bw_min_20.iloc[-1] * 1.05)  # within 5% of 20-bar low
        
        # Pivot Points (Standard / Classic) — based on previous day's H/L/C
        prev_h = high.iloc[-2] if len(high) >= 2 else high.iloc[-1]
        prev_l = low.iloc[-2]  if len(low)  >= 2 else low.iloc[-1]
        prev_c = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
        pivot_p  = (prev_h + prev_l + prev_c) / 3
        pivot_r1 = 2 * pivot_p - prev_l
        pivot_r2 = pivot_p + (prev_h - prev_l)
        pivot_r3 = prev_h + 2 * (pivot_p - prev_l)
        pivot_s1 = 2 * pivot_p - prev_h
        pivot_s2 = pivot_p - (prev_h - prev_l)
        pivot_s3 = prev_l - 2 * (prev_h - pivot_p)
        
        if current_price >= pivot_r2:
            price_vs_pivot = "above_r2"
        elif current_price >= pivot_r1:
            price_vs_pivot = "above_r1"
        elif current_price >= pivot_p:
            price_vs_pivot = "above_pivot"
        elif current_price >= pivot_s1:
            price_vs_pivot = "below_pivot"
        elif current_price >= pivot_s2:
            price_vs_pivot = "below_s1"
        else:
            price_vs_pivot = "below_s2"
        
        # Fibonacci Retracement — swing high / swing low over last 50 bars
        lookback = min(50, len(close))
        swing_high = high.iloc[-lookback:].max()
        swing_low  = low.iloc[-lookback:].min()
        fib_range  = swing_high - swing_low
        fib_levels = {
            "0.0":   swing_low,
            "23.6":  swing_low + 0.236 * fib_range,
            "38.2":  swing_low + 0.382 * fib_range,
            "50.0":  swing_low + 0.500 * fib_range,
            "61.8":  swing_low + 0.618 * fib_range,
            "78.6":  swing_low + 0.786 * fib_range,
            "100.0": swing_high,
        }
        # Find the nearest fib level to current price
        nearest_label = min(fib_levels, key=lambda k: abs(fib_levels[k] - current_price))
        nearest_price = fib_levels[nearest_label]
        fib_nearest_dist = (current_price - nearest_price) / current_price * 100
        
        # VIX — fetch live ^VIX close
        try:
            vix_data = yf.Ticker("^VIX").history(period="5d", interval="1d")
            vix_val = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20.0
        except Exception:
            vix_val = 20.0
        vix_signal = (
            "extreme_fear"  if vix_val > 40 else
            "fear"          if vix_val > 25 else
            "neutral"       if vix_val > 15 else
            "complacency"
        )
        
        # Volume
        current_volume = volume.iloc[-1]
        volume_sma = volume.rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / volume_sma
        
        # OBV trend
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_sma = obv.rolling(20).mean()
        obv_trend = "up" if obv.iloc[-1] > obv_sma.iloc[-1] else "down"
        
        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr_14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/14).mean() / atr_14
        minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / atr_14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/14).mean().iloc[-1]
        
        trend_strength = "strong" if adx > 25 else "weak" if adx > 20 else "none"
        trend_direction = "bullish" if plus_di.iloc[-1] > minus_di.iloc[-1] else "bearish"
        
        return TechnicalData(
            price=round(current_price, 2),
            change_1d=round(change_1d, 2),
            change_5d=round(change_5d, 2),
            change_20d=round(change_20d, 2),
            vwap=round(vwap, 2),
            vwap_distance=round(vwap_distance, 2),
            sma_9=round(sma_9, 2),
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            sma_100=round(sma_100, 2),
            sma_200=round(sma_200, 2),
            ema_12=round(ema_12, 2),
            ema_26=round(ema_26, 2),
            price_vs_sma_20="above" if current_price > sma_20 else "below",
            price_vs_sma_50="above" if current_price > sma_50 else "below",
            price_vs_sma_200="above" if current_price > sma_200 else "below",
            golden_cross=sma_50 > sma_200,
            rsi_14=round(rsi, 2),
            rsi_signal=rsi_signal,
            macd=round(macd_line, 4),
            macd_signal=round(macd_signal_line, 4),
            macd_histogram=round(macd_hist, 4),
            macd_cross=macd_cross,
            macd_histogram_trend=macd_histogram_trend,
            stochastic_k=round(stoch_k, 2),
            stochastic_d=round(stoch_d, 2),
            stoch_rsi_k=round(stoch_rsi_k, 2),
            stoch_rsi_d=round(stoch_rsi_d, 2),
            stoch_rsi_signal=stoch_rsi_signal,
            ppo=round(ppo, 4),
            ppo_signal=round(ppo_sig, 4),
            ppo_histogram=round(ppo_hist, 4),
            ppo_cross=ppo_cross,
            pvo=round(pvo, 4),
            pvo_signal=round(pvo_sig, 4),
            pvo_cross=pvo_cross,
            atr_14=round(atr, 2),
            atr_percent=round(atr_percent, 2),
            bollinger_upper=round(bb_upper_val, 2),
            bollinger_middle=round(bb_middle_val, 2),
            bollinger_lower=round(bb_lower_val, 2),
            bollinger_position=round(bb_position, 4),
            bollinger_bandwidth=round(bb_bandwidth, 2),
            bollinger_squeeze=bb_squeeze,
            pivot_p=round(pivot_p, 2),
            pivot_r1=round(pivot_r1, 2),
            pivot_r2=round(pivot_r2, 2),
            pivot_r3=round(pivot_r3, 2),
            pivot_s1=round(pivot_s1, 2),
            pivot_s2=round(pivot_s2, 2),
            pivot_s3=round(pivot_s3, 2),
            price_vs_pivot=price_vs_pivot,
            fib_swing_high=round(swing_high, 2),
            fib_swing_low=round(swing_low, 2),
            fib_ret_236=round(fib_levels["23.6"], 2),
            fib_ret_382=round(fib_levels["38.2"], 2),
            fib_ret_500=round(fib_levels["50.0"], 2),
            fib_ret_618=round(fib_levels["61.8"], 2),
            fib_ret_786=round(fib_levels["78.6"], 2),
            fib_nearest_level=nearest_label,
            fib_nearest_distance=round(fib_nearest_dist, 2),
            vix=round(vix_val, 2),
            vix_signal=vix_signal,
            volume=int(current_volume),
            volume_sma_20=round(volume_sma, 0),
            volume_ratio=round(volume_ratio, 2),
            obv_trend=obv_trend,
            adx=round(adx, 2),
            trend_strength=trend_strength,
            plus_di=round(plus_di.iloc[-1], 2),
            minus_di=round(minus_di.iloc[-1], 2),
            trend_direction=trend_direction
        )
    
    def _fetch_options(self) -> Optional[OptionsData]:
        """Fetch and analyze options chain."""
        try:
            expirations = self.stock.options
            if not expirations:
                return None
            
            current_price = self._price_data['close'].iloc[-1]
            
            all_calls = []
            all_puts = []
            
            # Get first 3 expirations
            for exp in expirations[:3]:
                try:
                    chain = self.stock.option_chain(exp)
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    calls['expiration'] = exp
                    puts['expiration'] = exp
                    all_calls.append(calls)
                    all_puts.append(puts)
                except:
                    continue
            
            if not all_calls:
                return None
            
            calls_df = pd.concat(all_calls, ignore_index=True)
            puts_df = pd.concat(all_puts, ignore_index=True)
            
            # ATM IV
            atm_calls = calls_df.iloc[(calls_df['strike'] - current_price).abs().argsort()[:3]]
            atm_puts = puts_df.iloc[(puts_df['strike'] - current_price).abs().argsort()[:3]]
            
            atm_iv_call = atm_calls['impliedVolatility'].mean() if 'impliedVolatility' in atm_calls.columns else 0
            atm_iv_put = atm_puts['impliedVolatility'].mean() if 'impliedVolatility' in atm_puts.columns else 0
            
            # Put/Call ratios
            total_call_vol = calls_df['volume'].sum() if 'volume' in calls_df.columns else 0
            total_put_vol = puts_df['volume'].sum() if 'volume' in puts_df.columns else 0
            total_call_oi = calls_df['openInterest'].sum() if 'openInterest' in calls_df.columns else 0
            total_put_oi = puts_df['openInterest'].sum() if 'openInterest' in puts_df.columns else 0
            
            pc_ratio_vol = total_put_vol / (total_call_vol + 1)
            pc_ratio_oi = total_put_oi / (total_call_oi + 1)
            
            # IV Skew (OTM puts vs OTM calls)
            otm_puts = puts_df[puts_df['strike'] < current_price * 0.95]
            otm_calls = calls_df[calls_df['strike'] > current_price * 1.05]
            
            iv_skew = 0
            if len(otm_puts) > 0 and len(otm_calls) > 0:
                otm_put_iv = otm_puts['impliedVolatility'].mean() if 'impliedVolatility' in otm_puts.columns else 0
                otm_call_iv = otm_calls['impliedVolatility'].mean() if 'impliedVolatility' in otm_calls.columns else 0
                iv_skew = otm_put_iv - otm_call_iv
            
            # Max Pain
            strikes = calls_df['strike'].unique()
            pain = {}
            for strike in strikes:
                itm_calls = calls_df[calls_df['strike'] < strike]
                itm_puts = puts_df[puts_df['strike'] > strike]
                
                call_pain = (itm_calls['openInterest'] * (strike - itm_calls['strike'])).sum() if len(itm_calls) > 0 else 0
                put_pain = (itm_puts['openInterest'] * (itm_puts['strike'] - strike)).sum() if len(itm_puts) > 0 else 0
                pain[strike] = call_pain + put_pain
            
            max_pain = min(pain, key=pain.get) if pain else current_price
            
            # IV Percentile (vs last year)
            # Simplified: compare current ATM IV to historical realized vol
            returns = self._price_data['close'].pct_change()
            historical_vol = returns.std() * np.sqrt(252)
            current_iv = (atm_iv_call + atm_iv_put) / 2
            iv_percentile = min(100, (current_iv / (historical_vol + 0.01)) * 50)  # Rough approximation
            
            # Unusual activity
            unusual = []
            for _, row in calls_df.iterrows():
                if 'volume' in row and 'openInterest' in row:
                    if row['volume'] > row['openInterest'] * 2 and row['volume'] > 1000:
                        unusual.append({
                            'type': 'call',
                            'strike': row['strike'],
                            'expiration': row.get('expiration', ''),
                            'volume': int(row['volume']),
                            'oi': int(row['openInterest']),
                            'ratio': round(row['volume'] / (row['openInterest'] + 1), 2)
                        })
            
            for _, row in puts_df.iterrows():
                if 'volume' in row and 'openInterest' in row:
                    if row['volume'] > row['openInterest'] * 2 and row['volume'] > 1000:
                        unusual.append({
                            'type': 'put',
                            'strike': row['strike'],
                            'expiration': row.get('expiration', ''),
                            'volume': int(row['volume']),
                            'oi': int(row['openInterest']),
                            'ratio': round(row['volume'] / (row['openInterest'] + 1), 2)
                        })
            
            return OptionsData(
                atm_iv_call=round(atm_iv_call * 100, 2),
                atm_iv_put=round(atm_iv_put * 100, 2),
                put_call_ratio_volume=round(pc_ratio_vol, 2),
                put_call_ratio_oi=round(pc_ratio_oi, 2),
                max_pain=round(max_pain, 2),
                iv_percentile=round(iv_percentile, 1),
                iv_skew=round(iv_skew * 100, 2),
                total_call_oi=int(total_call_oi),
                total_put_oi=int(total_put_oi),
                total_call_volume=int(total_call_vol),
                total_put_volume=int(total_put_vol),
                nearest_expiry=expirations[0] if expirations else "",
                unusual_activity=unusual[:5]  # Top 5
            )
            
        except Exception as e:
            print(f"  Options error: {e}")
            return None
    
    def _fetch_fundamentals(self) -> FundamentalData:
        """Fetch fundamental data."""
        info = self._info or {}
        
        current_price = self._price_data['close'].iloc[-1] if self._price_data is not None else 0
        target_price = info.get('targetMeanPrice', current_price)
        upside = (target_price / current_price - 1) * 100 if current_price > 0 else 0
        
        # Analyst rating interpretation
        rating_score = info.get('recommendationMean', 3)
        if rating_score <= 1.5:
            analyst_rating = "Strong Buy"
        elif rating_score <= 2.5:
            analyst_rating = "Buy"
        elif rating_score <= 3.5:
            analyst_rating = "Hold"
        elif rating_score <= 4.5:
            analyst_rating = "Sell"
        else:
            analyst_rating = "Strong Sell"
        
        return FundamentalData(
            market_cap=info.get('marketCap', 0),
            pe_ratio=info.get('trailingPE', 0) or 0,
            forward_pe=info.get('forwardPE', 0) or 0,
            peg_ratio=info.get('pegRatio', 0) or 0,
            pb_ratio=info.get('priceToBook', 0) or 0,
            ps_ratio=info.get('priceToSalesTrailing12Months', 0) or 0,
            ev_ebitda=info.get('enterpriseToEbitda', 0) or 0,
            roe=round((info.get('returnOnEquity', 0) or 0) * 100, 2),
            roa=round((info.get('returnOnAssets', 0) or 0) * 100, 2),
            gross_margin=round((info.get('grossMargins', 0) or 0) * 100, 2),
            operating_margin=round((info.get('operatingMargins', 0) or 0) * 100, 2),
            profit_margin=round((info.get('profitMargins', 0) or 0) * 100, 2),
            revenue_growth=round((info.get('revenueGrowth', 0) or 0) * 100, 2),
            earnings_growth=round((info.get('earningsGrowth', 0) or 0) * 100, 2),
            current_ratio=info.get('currentRatio', 0) or 0,
            debt_to_equity=info.get('debtToEquity', 0) or 0,
            analyst_rating=analyst_rating,
            target_price=round(target_price, 2),
            upside_potential=round(upside, 2),
            num_analysts=info.get('numberOfAnalystOpinions', 0) or 0
        )
    
    def to_dict(self, data: LiveMarketData) -> dict:
        """Convert LiveMarketData to dictionary."""
        result = {
            'ticker': data.ticker,
            'timestamp': data.timestamp,
            'technicals': asdict(data.technicals),
            'fundamentals': asdict(data.fundamentals)
        }
        if data.options:
            result['options'] = asdict(data.options)
        return result
    
    def to_json(self, data: LiveMarketData) -> str:
        """Convert LiveMarketData to JSON string."""
        return json.dumps(self.to_dict(data), indent=2)


def fetch_live_data(ticker: str) -> LiveMarketData:
    """Convenience function to fetch live data for a ticker."""
    fetcher = RealTimeDataFetcher(ticker)
    return fetcher.fetch_all()


def fib_label_price(t: 'TechnicalData') -> float:
    """Return the price value of the nearest fibonacci retracement level."""
    fib_map = {
        "23.6": t.fib_ret_236,
        "38.2": t.fib_ret_382,
        "50.0": t.fib_ret_500,
        "61.8": t.fib_ret_618,
        "78.6": t.fib_ret_786,
        "0.0":  t.fib_swing_low,
        "100.0": t.fib_swing_high,
    }
    return fib_map.get(t.fib_nearest_level, 0.0)


if __name__ == '__main__':
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "IONQ"
    
    fetcher = RealTimeDataFetcher(ticker)
    data = fetcher.fetch_all()
    
    print("\n" + "="*60)
    print(f"LIVE MARKET DATA: {ticker}")
    print("="*60)
    
    t = data.technicals
    print(f"\n📊 PRICE & TECHNICALS")
    print(f"  Price: ${t.price} ({t.change_1d:+.2f}% 1D, {t.change_5d:+.2f}% 5D, {t.change_20d:+.2f}% 20D)")
    print(f"  VWAP: ${t.vwap} (Distance: {t.vwap_distance:+.2f}%)")
    
    print(f"\n📈 MOVING AVERAGES")
    print(f"  SMA9:   ${t.sma_9}")
    print(f"  SMA20:  ${t.sma_20} [{t.price_vs_sma_20}]")
    print(f"  SMA50:  ${t.sma_50} [{t.price_vs_sma_50}]")
    print(f"  SMA100: ${t.sma_100}")
    print(f"  SMA200: ${t.sma_200} [{t.price_vs_sma_200}]")
    print(f"  Golden Cross (SMA50 > SMA200): {'✅ YES' if t.golden_cross else '❌ NO'}")
    
    print(f"\n📉 OSCILLATORS")
    print(f"  RSI(14): {t.rsi_14} [{t.rsi_signal.upper()}]")
    print(f"  MACD: {t.macd:.4f} | Signal: {t.macd_signal:.4f} | Hist: {t.macd_histogram:.4f} [{t.macd_cross.upper()}] [hist: {t.macd_histogram_trend}]")
    print(f"  PPO: {t.ppo:.4f}% | Signal: {t.ppo_signal:.4f}% | Hist: {t.ppo_histogram:.4f}% [{t.ppo_cross.upper()}]")
    print(f"  Stochastic: K={t.stochastic_k:.1f}, D={t.stochastic_d:.1f}")
    print(f"  StochRSI:   K={t.stoch_rsi_k:.1f}, D={t.stoch_rsi_d:.1f} [{t.stoch_rsi_signal.upper()}]")
    
    print(f"\n📦 VOLUME OSCILLATOR")
    print(f"  PVO: {t.pvo:.4f}% | Signal: {t.pvo_signal:.4f}% [{t.pvo_cross.upper()}]")
    print(f"  Volume: {t.volume:,} ({t.volume_ratio:.2f}x 20-day avg) | OBV Trend: {t.obv_trend.upper()}")
    
    print(f"\n🎯 VOLATILITY")
    print(f"  ATR(14): ${t.atr_14} ({t.atr_percent:.2f}% of price)")
    print(f"  Bollinger Bands: Lower=${t.bollinger_lower} | Mid=${t.bollinger_middle} | Upper=${t.bollinger_upper}")
    print(f"  BB %B: {t.bollinger_position:.4f} (0=lower band, 1=upper band) | Bandwidth: {t.bollinger_bandwidth:.2f}% | Squeeze: {'🔴 YES' if t.bollinger_squeeze else '🟢 NO'}")
    
    print(f"\n📐 PIVOT POINTS (Standard)")
    print(f"  R3=${t.pivot_r3:.2f}  R2=${t.pivot_r2:.2f}  R1=${t.pivot_r1:.2f}")
    print(f"  Pivot (P)=${t.pivot_p:.2f}  ← Price is [{t.price_vs_pivot.upper()}]")
    print(f"  S1=${t.pivot_s1:.2f}  S2=${t.pivot_s2:.2f}  S3=${t.pivot_s3:.2f}")
    
    print(f"\n🌀 FIBONACCI RETRACEMENT (last 50 bars: High=${t.fib_swing_high} → Low=${t.fib_swing_low})")
    print(f"  78.6% = ${t.fib_ret_786}")
    print(f"  61.8% = ${t.fib_ret_618}  ← OTE zone top (ICT)")
    print(f"  50.0% = ${t.fib_ret_500}")
    print(f"  38.2% = ${t.fib_ret_382}  ← OTE zone bottom (ICT)")
    print(f"  23.6% = ${t.fib_ret_236}")
    print(f"  Nearest level: {t.fib_nearest_level}% (${fib_label_price(t)}) | Distance: {t.fib_nearest_distance:+.2f}%")
    
    print(f"\n🌡️  MARKET SENTIMENT")
    print(f"  VIX: {t.vix} [{t.vix_signal.upper().replace('_',' ')}]")
    print(f"  ADX: {t.adx} [{t.trend_strength.upper()} {t.trend_direction.upper()} TREND]")
    print(f"  +DI: {t.plus_di}  -DI: {t.minus_di}")
    
    if data.options:
        o = data.options
        print(f"\n📋 OPTIONS DATA")
        print(f"  ATM IV: Call={o.atm_iv_call}%, Put={o.atm_iv_put}%")
        print(f"  IV Skew: {o.iv_skew}%")
        print(f"  Put/Call Volume: {o.put_call_ratio_volume}")
        print(f"  Put/Call OI: {o.put_call_ratio_oi}")
        print(f"  Max Pain: ${o.max_pain}")
        print(f"  Total Call OI: {o.total_call_oi:,}")
        print(f"  Total Put OI: {o.total_put_oi:,}")
        if o.unusual_activity:
            print(f"  Unusual Activity:")
            for act in o.unusual_activity[:3]:
                print(f"    - {act['type'].upper()} ${act['strike']} exp:{act['expiration']} vol:{act['volume']:,} (OI ratio: {act['ratio']}x)")
    
    f = data.fundamentals
    print(f"\n💰 FUNDAMENTALS")
    print(f"  Market Cap: ${f.market_cap/1e9:.2f}B")
    print(f"  P/E: {f.pe_ratio:.1f} (Forward: {f.forward_pe:.1f})")
    print(f"  P/S: {f.ps_ratio:.1f}, P/B: {f.pb_ratio:.1f}")
    print(f"  ROE: {f.roe}%, ROA: {f.roa}%")
    print(f"  Revenue Growth: {f.revenue_growth}%")
    print(f"  Analyst Rating: {f.analyst_rating} (Target: ${f.target_price}, Upside: {f.upside_potential:+.1f}%)")
    
    print("\n" + "="*60)
