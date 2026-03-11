#!/usr/bin/env python3
"""
Advanced Quantitative Analysis Framework

A Jane Street-style approach to technical and fundamental analysis:
- Microstructure signals (order flow, bid-ask dynamics)
- Cross-asset correlations and regime detection
- Options-implied metrics (skew, term structure, vol surface)
- Factor exposures and risk decomposition
- Statistical arbitrage signals
- Machine learning ensemble with proper cross-validation
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

from indicators import TechnicalIndicators


# =============================================================================
# 1. MICROSTRUCTURE ANALYSIS
# =============================================================================

class MicrostructureAnalyzer:
    """
    Analyze market microstructure signals that institutional traders watch.
    These are the "hidden" signals retail doesn't see.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all microstructure features."""
        self.volume_profile()
        self.price_efficiency()
        self.volatility_clustering()
        self.order_flow_imbalance()
        self.liquidity_metrics()
        self.smart_money_indicators()
        return self.df
    
    def volume_profile(self):
        """Volume distribution analysis - where is size trading?"""
        close = self.df['close']
        volume = self.df['volume']
        high = self.df['high']
        low = self.df['low']
        
        # VWAP (Volume Weighted Average Price) - institutional benchmark
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        self.df['vwap'] = cumulative_tp_vol / cumulative_vol
        
        # Distance from VWAP (institutional traders watch this)
        self.df['vwap_distance'] = (close - self.df['vwap']) / self.df['vwap'] * 100
        
        # Volume-at-price concentration (simplified)
        # High concentration = potential support/resistance
        self.df['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        
        # Accumulation/Distribution based on close location in range
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        self.df['ad_line'] = (clv * volume).cumsum()
        
        # Chaikin Money Flow - measures buying/selling pressure
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        self.df['cmf_20'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
    def price_efficiency(self):
        """
        Measure how efficiently price moves - trending vs mean-reverting.
        Efficiency Ratio (Kaufman) - used in adaptive moving averages.
        """
        close = self.df['close']
        
        # Efficiency Ratio = Direction / Volatility
        for period in [10, 20, 50]:
            direction = abs(close - close.shift(period))
            volatility = abs(close.diff()).rolling(period).sum()
            self.df[f'efficiency_ratio_{period}'] = direction / (volatility + 1e-10)
        
        # Fractal Dimension - market complexity measure
        # Low FD = trending, High FD = choppy
        n = 20
        max_price = self.df['high'].rolling(n).max()
        min_price = self.df['low'].rolling(n).min()
        price_range = max_price - min_price
        
        # Approximate fractal dimension using range
        self.df['fractal_dimension'] = np.log(n) / (np.log(n) + np.log(price_range / (self.df['atr_14'] * n + 1e-10) + 1e-10))
        
        # Hurst Exponent approximation (simplified)
        # H > 0.5 = trending, H < 0.5 = mean reverting, H = 0.5 = random walk
        returns = close.pct_change()
        
        def rolling_hurst(series, window=100):
            """Simplified Hurst exponent estimation."""
            result = pd.Series(index=series.index, dtype=float)
            for i in range(window, len(series)):
                subset = series.iloc[i-window:i].dropna()
                if len(subset) < 20:
                    continue
                # R/S analysis
                mean_val = subset.mean()
                std_val = subset.std()
                if std_val == 0:
                    continue
                cumdev = (subset - mean_val).cumsum()
                r = cumdev.max() - cumdev.min()
                s = std_val
                if s > 0 and r > 0:
                    # Approximate H
                    result.iloc[i] = np.log(r / s) / np.log(window)
            return result
        
        self.df['hurst_approx'] = rolling_hurst(returns, 50)
        
    def volatility_clustering(self):
        """
        Volatility tends to cluster - use GARCH-like features.
        High vol predicts high vol, low vol predicts low vol.
        """
        returns = self.df['close'].pct_change()
        
        # Realized volatility at different horizons
        for period in [5, 10, 20, 60]:
            self.df[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Volatility of volatility (vol-of-vol)
        self.df['vol_of_vol'] = self.df['realized_vol_20'].rolling(20).std()
        
        # Volatility term structure (short vs long vol)
        self.df['vol_term_structure'] = self.df['realized_vol_5'] / (self.df['realized_vol_60'] + 1e-10)
        
        # Parkinson volatility (uses high-low range, more efficient estimator)
        hl_ratio = np.log(self.df['high'] / self.df['low'])
        self.df['parkinson_vol'] = np.sqrt(hl_ratio ** 2 / (4 * np.log(2))).rolling(20).mean() * np.sqrt(252)
        
        # Garman-Klass volatility (uses OHLC, even more efficient)
        log_hl = np.log(self.df['high'] / self.df['low']) ** 2
        log_co = np.log(self.df['close'] / self.df['open']) ** 2
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        self.df['gk_volatility'] = np.sqrt(gk_var.rolling(20).mean() * 252)
        
        # Yang-Zhang volatility (most efficient, handles overnight gaps)
        log_oc = np.log(self.df['open'] / self.df['close'].shift(1))
        log_co = np.log(self.df['close'] / self.df['open'])
        log_ho = np.log(self.df['high'] / self.df['open'])
        log_lo = np.log(self.df['low'] / self.df['open'])
        
        overnight_var = log_oc.rolling(20).var()
        open_to_close_var = log_co.rolling(20).var()
        rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(20).mean()
        
        k = 0.34 / (1.34 + (21) / (20 - 1))
        self.df['yz_volatility'] = np.sqrt((overnight_var + k * open_to_close_var + (1 - k) * rs_var) * 252)
        
    def order_flow_imbalance(self):
        """
        Approximate order flow from price/volume data.
        Real order flow requires tick data, but we can estimate.
        """
        close = self.df['close']
        volume = self.df['volume']
        high = self.df['high']
        low = self.df['low']
        
        # Buy/Sell volume estimation (tick rule approximation)
        # If close > previous close, assume more buying
        price_change = close.diff()
        
        # Estimated buy volume (simplified)
        buy_ratio = (close - low) / (high - low + 1e-10)
        self.df['est_buy_volume'] = volume * buy_ratio
        self.df['est_sell_volume'] = volume * (1 - buy_ratio)
        
        # Order flow imbalance
        self.df['ofi'] = (self.df['est_buy_volume'] - self.df['est_sell_volume']) / volume
        self.df['ofi_ma'] = self.df['ofi'].rolling(10).mean()
        
        # Cumulative delta (simplified)
        self.df['cumulative_delta'] = (self.df['est_buy_volume'] - self.df['est_sell_volume']).cumsum()
        
        # Delta divergence from price
        price_norm = (close - close.rolling(50).min()) / (close.rolling(50).max() - close.rolling(50).min() + 1e-10)
        delta_norm = (self.df['cumulative_delta'] - self.df['cumulative_delta'].rolling(50).min()) / \
                     (self.df['cumulative_delta'].rolling(50).max() - self.df['cumulative_delta'].rolling(50).min() + 1e-10)
        self.df['delta_divergence'] = price_norm - delta_norm
        
    def liquidity_metrics(self):
        """
        Liquidity proxies - crucial for execution.
        """
        close = self.df['close']
        volume = self.df['volume']
        high = self.df['high']
        low = self.df['low']
        
        # Amihud illiquidity ratio (price impact per dollar volume)
        returns = close.pct_change().abs()
        dollar_volume = close * volume
        self.df['amihud_illiquidity'] = (returns / (dollar_volume + 1e-10)).rolling(20).mean() * 1e6
        
        # Roll spread estimator (bid-ask spread proxy)
        # Based on serial covariance of returns
        ret = close.pct_change()
        cov = ret.rolling(20).apply(lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0)
        self.df['roll_spread'] = 2 * np.sqrt(-np.minimum(cov, 0))
        
        # Kyle's Lambda approximation (price impact coefficient)
        # Higher = more price impact per trade
        signed_volume = np.sign(close.diff()) * volume
        self.df['kyle_lambda'] = abs(close.diff()) / (signed_volume.abs() + 1e-10)
        self.df['kyle_lambda_ma'] = self.df['kyle_lambda'].rolling(20).mean()
        
        # Volume concentration (Herfindahl index of volume across price levels)
        # Approximated using daily range distribution
        range_pct = (high - low) / close
        self.df['volume_concentration'] = 1 / (range_pct.rolling(20).std() + 1e-10)
        
    def smart_money_indicators(self):
        """
        Indicators that attempt to track institutional activity.
        """
        close = self.df['close']
        volume = self.df['volume']
        high = self.df['high']
        low = self.df['low']
        
        # Smart Money Index (SMI)
        # Institutions trade at close, retail at open
        open_price = self.df['open']
        overnight_move = open_price - close.shift(1)
        intraday_move = close - open_price
        
        self.df['smi'] = (intraday_move - overnight_move).cumsum()
        self.df['smi_signal'] = self.df['smi'] - self.df['smi'].rolling(20).mean()
        
        # Force Index (volume confirms price moves)
        self.df['force_index'] = close.diff() * volume
        self.df['force_index_13'] = self.df['force_index'].ewm(span=13).mean()
        
        # Elder Ray - Bull Power / Bear Power (already in TechnicalIndicators)
        # But we add normalized version
        ema_13 = close.ewm(span=13).mean()
        self.df['bull_power_norm'] = (high - ema_13) / close * 100
        self.df['bear_power_norm'] = (low - ema_13) / close * 100


# =============================================================================
# 2. CROSS-ASSET & REGIME ANALYSIS
# =============================================================================

class CrossAssetAnalyzer:
    """
    Analyze relationships across asset classes.
    Key for understanding macro regimes and correlations.
    """
    
    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker = ticker
        self.df = df.copy()
        
    def fetch_cross_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch related asset data for correlation analysis."""
        related_assets = {
            # Market indices
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq 100',
            'IWM': 'Russell 2000',
            'VIX': 'Volatility Index',
            
            # Rates & Bonds
            'TLT': '20+ Year Treasury',
            'IEF': '7-10 Year Treasury',
            'HYG': 'High Yield Bonds',
            
            # Sectors
            'XLK': 'Tech Sector',
            'XLF': 'Financial Sector',
            'XLE': 'Energy Sector',
            
            # Commodities
            'GLD': 'Gold',
            'USO': 'Oil',
            
            # Currencies
            'UUP': 'Dollar Index',
        }
        
        asset_data = {}
        print("  Fetching cross-asset data...")
        
        for symbol, name in related_assets.items():
            try:
                data = yf.Ticker(symbol).history(period="2y", interval="1d")
                if len(data) > 100:
                    data = data.reset_index()
                    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
                    asset_data[symbol] = data
            except Exception as e:
                pass
        
        return asset_data
    
    def compute_correlations(self, asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute rolling correlations with related assets."""
        # Align all returns to same index
        returns_df = pd.DataFrame()
        returns_df[self.ticker] = self.df['close'].pct_change()
        
        for symbol, data in asset_data.items():
            if 'close' in data.columns:
                # Align by date
                returns = data.set_index('date')['close'].pct_change()
                returns_df[symbol] = returns
        
        # Forward fill missing values
        returns_df = returns_df.ffill().dropna()
        
        # Rolling correlations
        for symbol in asset_data.keys():
            if symbol in returns_df.columns:
                # 20-day rolling correlation
                corr_20 = returns_df[self.ticker].rolling(20).corr(returns_df[symbol])
                self.df[f'corr_{symbol}_20'] = corr_20.reindex(self.df.index).ffill()
                
                # 60-day rolling correlation
                corr_60 = returns_df[self.ticker].rolling(60).corr(returns_df[symbol])
                self.df[f'corr_{symbol}_60'] = corr_60.reindex(self.df.index).ffill()
        
        return returns_df
    
    def detect_regime(self, returns_df: pd.DataFrame):
        """
        Detect market regime using clustering.
        Regimes: Risk-on, Risk-off, High-vol, Low-vol, etc.
        """
        # Use key relationships for regime detection
        features_for_regime = []
        
        if 'SPY' in returns_df.columns:
            # Stock-bond correlation (risk-on/off indicator)
            if 'TLT' in returns_df.columns:
                sb_corr = returns_df['SPY'].rolling(20).corr(returns_df['TLT'])
                features_for_regime.append(sb_corr)
            
            # VIX relationship
            if 'VIX' in returns_df.columns:
                vix_level = returns_df['VIX'].rolling(20).mean()
                features_for_regime.append(vix_level)
        
        # Market volatility
        market_vol = returns_df[self.ticker].rolling(20).std() * np.sqrt(252)
        features_for_regime.append(market_vol)
        
        # Combine features
        if features_for_regime:
            regime_features = pd.concat(features_for_regime, axis=1).dropna()
            
            if len(regime_features) > 50:
                # Standardize
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(regime_features)
                
                # K-means clustering for regime detection
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                regimes = kmeans.fit_predict(scaled_features)
                
                # Map back to dataframe
                regime_series = pd.Series(regimes, index=regime_features.index)
                self.df['market_regime'] = regime_series.reindex(self.df.index).ffill()
        
    def compute_beta_factors(self, returns_df: pd.DataFrame):
        """
        Compute factor exposures (beta to market, sector, etc.)
        """
        if self.ticker not in returns_df.columns:
            return
            
        stock_returns = returns_df[self.ticker]
        
        # Market beta (rolling)
        if 'SPY' in returns_df.columns:
            market_returns = returns_df['SPY']
            
            for window in [20, 60, 120]:
                cov = stock_returns.rolling(window).cov(market_returns)
                var = market_returns.rolling(window).var()
                beta = cov / (var + 1e-10)
                self.df[f'beta_market_{window}'] = beta.reindex(self.df.index).ffill()
            
            # Alpha (excess return vs market)
            self.df['alpha_20'] = (stock_returns.rolling(20).mean() - 
                                   self.df['beta_market_20'] * market_returns.rolling(20).mean()) * 252
        
        # Sector beta (if tech stock)
        if 'XLK' in returns_df.columns:
            sector_returns = returns_df['XLK']
            cov = stock_returns.rolling(60).cov(sector_returns)
            var = sector_returns.rolling(60).var()
            self.df['beta_tech_sector'] = (cov / (var + 1e-10)).reindex(self.df.index).ffill()
        
        # VIX sensitivity (how much does stock move with vol?)
        if 'VIX' in returns_df.columns:
            vix_returns = returns_df['VIX']
            corr = stock_returns.rolling(60).corr(vix_returns)
            self.df['vix_sensitivity'] = corr.reindex(self.df.index).ffill()


# =============================================================================
# 3. OPTIONS-IMPLIED METRICS
# =============================================================================

class OptionsImpliedAnalyzer:
    """
    Extract forward-looking signals from options market.
    Options traders are often more informed than equity traders.
    """
    
    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker = ticker
        self.df = df.copy()
    
    def fetch_options_data(self) -> Optional[pd.DataFrame]:
        """Fetch current options chain data."""
        try:
            stock = yf.Ticker(self.ticker)
            expirations = stock.options
            
            if not expirations:
                return None
            
            all_options = []
            # Get first 4 expirations
            for exp in expirations[:4]:
                try:
                    chain = stock.option_chain(exp)
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    calls['type'] = 'call'
                    puts['type'] = 'put'
                    calls['expiration'] = exp
                    puts['expiration'] = exp
                    all_options.append(calls)
                    all_options.append(puts)
                except:
                    continue
            
            if all_options:
                return pd.concat(all_options, ignore_index=True)
            return None
        except:
            return None
    
    def compute_implied_metrics(self, options_df: Optional[pd.DataFrame]):
        """Compute options-implied metrics."""
        if options_df is None or len(options_df) == 0:
            # Use defaults/NaN
            self.df['iv_rank'] = np.nan
            self.df['iv_percentile'] = np.nan
            self.df['put_call_skew'] = np.nan
            self.df['term_structure_slope'] = np.nan
            return
        
        current_price = self.df['close'].iloc[-1]
        
        # ATM implied volatility
        calls = options_df[options_df['type'] == 'call']
        puts = options_df[options_df['type'] == 'put']
        
        # Find ATM options
        atm_calls = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:5]]
        atm_puts = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:5]]
        
        if 'impliedVolatility' in atm_calls.columns:
            atm_iv_call = atm_calls['impliedVolatility'].mean()
            atm_iv_put = atm_puts['impliedVolatility'].mean()
            
            # Store in last row (current snapshot)
            self.df.loc[self.df.index[-1], 'atm_iv_call'] = atm_iv_call
            self.df.loc[self.df.index[-1], 'atm_iv_put'] = atm_iv_put
            self.df.loc[self.df.index[-1], 'atm_iv_avg'] = (atm_iv_call + atm_iv_put) / 2
            
            # Put-Call IV skew (puts more expensive = bearish)
            self.df.loc[self.df.index[-1], 'put_call_iv_skew'] = atm_iv_put - atm_iv_call
        
        # 25-delta skew (OTM puts vs OTM calls)
        otm_puts = puts[puts['strike'] < current_price * 0.95]
        otm_calls = calls[calls['strike'] > current_price * 1.05]
        
        if len(otm_puts) > 0 and len(otm_calls) > 0 and 'impliedVolatility' in otm_puts.columns:
            otm_put_iv = otm_puts['impliedVolatility'].mean()
            otm_call_iv = otm_calls['impliedVolatility'].mean()
            self.df.loc[self.df.index[-1], 'skew_25d'] = otm_put_iv - otm_call_iv
        
        # IV term structure
        if 'expiration' in options_df.columns:
            expirations = options_df['expiration'].unique()
            if len(expirations) >= 2:
                # Near-term vs far-term IV
                near_exp = min(expirations)
                far_exp = max(expirations)
                
                near_iv = options_df[options_df['expiration'] == near_exp]['impliedVolatility'].mean()
                far_iv = options_df[options_df['expiration'] == far_exp]['impliedVolatility'].mean()
                
                # Contango (far > near) vs Backwardation (near > far)
                self.df.loc[self.df.index[-1], 'iv_term_structure'] = far_iv - near_iv
        
        # Open interest analysis
        if 'openInterest' in options_df.columns:
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            self.df.loc[self.df.index[-1], 'put_call_oi_ratio'] = total_put_oi / (total_call_oi + 1)
            
            # Max pain calculation
            strikes = options_df['strike'].unique()
            pain = {}
            for strike in strikes:
                call_pain = calls[calls['strike'] < strike]['openInterest'].sum() * (strike - calls[calls['strike'] < strike]['strike']).sum()
                put_pain = puts[puts['strike'] > strike]['openInterest'].sum() * (puts[puts['strike'] > strike]['strike'] - strike).sum()
                pain[strike] = call_pain + put_pain
            
            if pain:
                max_pain_strike = min(pain, key=pain.get)
                self.df.loc[self.df.index[-1], 'max_pain'] = max_pain_strike
                self.df.loc[self.df.index[-1], 'dist_to_max_pain'] = (current_price - max_pain_strike) / current_price * 100


# =============================================================================
# 4. FUNDAMENTAL FACTOR ANALYSIS  
# =============================================================================

class FundamentalFactorAnalyzer:
    """
    Compute fundamental factors that quants use for stock selection.
    Based on academic research (Fama-French, quality factors, etc.)
    """
    
    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker = ticker
        self.df = df.copy()
        self.fundamentals = {}
    
    def fetch_fundamentals(self):
        """Fetch fundamental data from Yahoo Finance."""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get various fundamental data
            self.fundamentals['info'] = stock.info
            self.fundamentals['financials'] = stock.financials
            self.fundamentals['balance_sheet'] = stock.balance_sheet
            self.fundamentals['cashflow'] = stock.cashflow
            
            return True
        except Exception as e:
            print(f"  Error fetching fundamentals: {e}")
            return False
    
    def compute_value_factors(self):
        """
        Value factors - is the stock cheap?
        """
        info = self.fundamentals.get('info', {})
        
        # Price ratios
        self.df.loc[self.df.index[-1], 'pe_ratio'] = info.get('trailingPE', np.nan)
        self.df.loc[self.df.index[-1], 'forward_pe'] = info.get('forwardPE', np.nan)
        self.df.loc[self.df.index[-1], 'peg_ratio'] = info.get('pegRatio', np.nan)
        self.df.loc[self.df.index[-1], 'pb_ratio'] = info.get('priceToBook', np.nan)
        self.df.loc[self.df.index[-1], 'ps_ratio'] = info.get('priceToSalesTrailing12Months', np.nan)
        
        # Enterprise value ratios
        self.df.loc[self.df.index[-1], 'ev_ebitda'] = info.get('enterpriseToEbitda', np.nan)
        self.df.loc[self.df.index[-1], 'ev_revenue'] = info.get('enterpriseToRevenue', np.nan)
        
        # Earnings yield (inverse of P/E, comparable to bond yields)
        pe = info.get('trailingPE', np.nan)
        if pe and pe > 0:
            self.df.loc[self.df.index[-1], 'earnings_yield'] = 100 / pe
        
        # FCF yield
        fcf = info.get('freeCashflow', 0)
        market_cap = info.get('marketCap', 1)
        if market_cap > 0:
            self.df.loc[self.df.index[-1], 'fcf_yield'] = fcf / market_cap * 100
    
    def compute_quality_factors(self):
        """
        Quality factors - is the company fundamentally strong?
        """
        info = self.fundamentals.get('info', {})
        
        # Profitability
        self.df.loc[self.df.index[-1], 'roe'] = info.get('returnOnEquity', np.nan)
        self.df.loc[self.df.index[-1], 'roa'] = info.get('returnOnAssets', np.nan)
        self.df.loc[self.df.index[-1], 'gross_margin'] = info.get('grossMargins', np.nan)
        self.df.loc[self.df.index[-1], 'operating_margin'] = info.get('operatingMargins', np.nan)
        self.df.loc[self.df.index[-1], 'profit_margin'] = info.get('profitMargins', np.nan)
        
        # Financial health
        self.df.loc[self.df.index[-1], 'current_ratio'] = info.get('currentRatio', np.nan)
        self.df.loc[self.df.index[-1], 'debt_to_equity'] = info.get('debtToEquity', np.nan)
        
        # Cash position
        total_cash = info.get('totalCash', 0)
        total_debt = info.get('totalDebt', 0)
        market_cap = info.get('marketCap', 1)
        
        self.df.loc[self.df.index[-1], 'net_cash_ratio'] = (total_cash - total_debt) / market_cap * 100 if market_cap > 0 else np.nan
        
    def compute_growth_factors(self):
        """
        Growth factors - is the company growing?
        """
        info = self.fundamentals.get('info', {})
        
        # Revenue growth
        self.df.loc[self.df.index[-1], 'revenue_growth'] = info.get('revenueGrowth', np.nan)
        
        # Earnings growth
        self.df.loc[self.df.index[-1], 'earnings_growth'] = info.get('earningsGrowth', np.nan)
        
        # Analyst estimates
        self.df.loc[self.df.index[-1], 'earnings_estimate_growth'] = info.get('earningsQuarterlyGrowth', np.nan)
        
        # Target price vs current (analyst sentiment)
        target = info.get('targetMeanPrice', np.nan)
        current = info.get('currentPrice', np.nan)
        if target and current and current > 0:
            self.df.loc[self.df.index[-1], 'upside_to_target'] = (target - current) / current * 100
        
        # Analyst recommendations
        self.df.loc[self.df.index[-1], 'analyst_rating'] = info.get('recommendationMean', np.nan)
        
    def compute_momentum_factors(self):
        """
        Price/earnings momentum factors.
        """
        close = self.df['close']
        
        # Price momentum at various horizons
        for months in [1, 3, 6, 12]:
            days = months * 21
            if len(close) > days:
                self.df[f'momentum_{months}m'] = close / close.shift(days) - 1
        
        # 52-week high/low position
        high_52w = close.rolling(252).max()
        low_52w = close.rolling(252).min()
        self.df['position_52w'] = (close - low_52w) / (high_52w - low_52w + 1e-10)
        
        # Earnings revision momentum (would need estimates data)
        # Placeholder for when you have earnings estimates
        
    def compute_size_factors(self):
        """
        Size factors - small cap premium.
        """
        info = self.fundamentals.get('info', {})
        
        market_cap = info.get('marketCap', np.nan)
        self.df.loc[self.df.index[-1], 'market_cap'] = market_cap
        
        if market_cap:
            self.df.loc[self.df.index[-1], 'log_market_cap'] = np.log(market_cap)
            
            # Size category
            if market_cap < 2e9:
                size_cat = 'small'
            elif market_cap < 10e9:
                size_cat = 'mid'
            else:
                size_cat = 'large'
            # Can't store string, so encode
            self.df.loc[self.df.index[-1], 'size_category'] = {'small': 1, 'mid': 2, 'large': 3}.get(size_cat, 2)


# =============================================================================
# 5. STATISTICAL ARBITRAGE SIGNALS
# =============================================================================

class StatArbAnalyzer:
    """
    Statistical arbitrage signals - mean reversion and relative value.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all stat arb signals."""
        self.mean_reversion_signals()
        self.z_score_analysis()
        self.pair_trading_prep()
        return self.df
    
    def mean_reversion_signals(self):
        """
        Signals for mean reversion strategies.
        """
        close = self.df['close']
        
        # Deviation from various moving averages
        for period in [10, 20, 50, 100]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            
            # Z-score from MA
            self.df[f'zscore_ma_{period}'] = (close - ma) / (std + 1e-10)
            
            # Bollinger band position (-1 to +1 scale)
            upper = ma + 2 * std
            lower = ma - 2 * std
            self.df[f'bb_position_{period}'] = (close - lower) / (upper - lower + 1e-10) * 2 - 1
        
        # RSI divergence
        rsi = self.df.get('rsi_14', pd.Series())
        if len(rsi) > 0:
            # Price making higher highs but RSI making lower highs = bearish divergence
            price_hh = close > close.rolling(20).max().shift(1)
            rsi_lh = rsi < rsi.rolling(20).max().shift(1)
            self.df['bearish_divergence'] = (price_hh & rsi_lh).astype(int)
            
            # Price making lower lows but RSI making higher lows = bullish divergence
            price_ll = close < close.rolling(20).min().shift(1)
            rsi_hl = rsi > rsi.rolling(20).min().shift(1)
            self.df['bullish_divergence'] = (price_ll & rsi_hl).astype(int)
    
    def z_score_analysis(self):
        """
        Z-score based trading signals.
        """
        close = self.df['close']
        returns = close.pct_change()
        
        # Return z-score (how extreme is today's move?)
        ret_mean = returns.rolling(60).mean()
        ret_std = returns.rolling(60).std()
        self.df['return_zscore'] = (returns - ret_mean) / (ret_std + 1e-10)
        
        # Volume z-score
        volume = self.df['volume']
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        self.df['volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-10)
        
        # Combined signal (extreme move on high volume)
        self.df['extreme_move'] = (abs(self.df['return_zscore']) > 2) & (self.df['volume_zscore'] > 1.5)
        
    def pair_trading_prep(self):
        """
        Prepare features useful for pair trading.
        """
        close = self.df['close']
        
        # Log price for cointegration
        self.df['log_price'] = np.log(close)
        
        # Returns at different frequencies
        for period in [1, 5, 10, 20]:
            self.df[f'return_{period}d'] = close.pct_change(period)
        
        # Autocorrelation (mean-reverting assets have negative autocorr)
        returns = close.pct_change()
        self.df['autocorr_1'] = returns.rolling(60).apply(lambda x: x.autocorr(lag=1))
        self.df['autocorr_5'] = returns.rolling(60).apply(lambda x: x.autocorr(lag=5))


# =============================================================================
# 6. ADVANCED ML ENSEMBLE
# =============================================================================

class MLEnsemblePredictor:
    """
    Ensemble of ML models with proper time-series cross-validation.
    """
    
    def __init__(self, df: pd.DataFrame, features: List[str], target: str):
        self.df = df.copy()
        self.features = features
        self.target = target
        self.models = {}
        self.importances = {}
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target."""
        X = self.df[self.features].copy()
        y = self.df[self.target].copy()
        
        # Drop features with too many NaN (>30%)
        nan_pct = X.isna().mean()
        good_features = nan_pct[nan_pct < 0.3].index.tolist()
        X = X[good_features]
        self.features = good_features
        
        # Forward fill then backward fill remaining NaN
        X = X.ffill().bfill().fillna(0)
        
        # Remove rows where target is NaN
        valid = ~y.isna()
        X = X[valid]
        y = y[valid]
        
        # Replace any inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X, y
    
    def train_ensemble(self, n_splits: int = 5):
        """Train ensemble with time-series cross-validation."""
        X, y = self.prepare_data()
        
        if len(X) < 200:
            print("  Not enough data for robust training")
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Time-series split (no future data leakage)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                    min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='accuracy')
        rf.fit(X_scaled, y)
        self.models['random_forest'] = rf
        self.importances['random_forest'] = dict(zip(self.features, rf.feature_importances_))
        results['random_forest'] = {'accuracy': rf_scores.mean(), 'std': rf_scores.std()}
        
        # 2. Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                        learning_rate=0.1, random_state=42)
        gb_scores = cross_val_score(gb, X_scaled, y, cv=tscv, scoring='accuracy')
        gb.fit(X_scaled, y)
        self.models['gradient_boosting'] = gb
        self.importances['gradient_boosting'] = dict(zip(self.features, gb.feature_importances_))
        results['gradient_boosting'] = {'accuracy': gb_scores.mean(), 'std': gb_scores.std()}
        
        # 3. Ridge Classifier (linear model for interpretability)
        ridge = Ridge(alpha=1.0)
        ridge_scores = cross_val_score(ridge, X_scaled, y, cv=tscv, scoring='r2')
        ridge.fit(X_scaled, y)
        self.models['ridge'] = ridge
        self.importances['ridge'] = dict(zip(self.features, np.abs(ridge.coef_)))
        results['ridge'] = {'r2': ridge_scores.mean(), 'std': ridge_scores.std()}
        
        # 4. Elastic Net (sparse linear model)
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        enet.fit(X_scaled, y)
        self.models['elastic_net'] = enet
        self.importances['elastic_net'] = dict(zip(self.features, np.abs(enet.coef_)))
        
        return results
    
    def get_combined_importance(self) -> Dict[str, float]:
        """Combine importances from all models."""
        combined = {}
        
        for feat in self.features:
            scores = []
            for model_name, imp in self.importances.items():
                if feat in imp:
                    # Normalize by max importance in that model
                    max_imp = max(imp.values()) if imp.values() else 1
                    scores.append(imp[feat] / max_imp)
            
            if scores:
                combined[feat] = np.mean(scores)
        
        # Sort by importance
        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features."""
        combined = self.get_combined_importance()
        return list(combined.keys())[:n]


# =============================================================================
# 7. RISK ANALYSIS
# =============================================================================

class RiskAnalyzer:
    """
    Risk metrics that professional traders monitor.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all risk metrics."""
        self.drawdown_analysis()
        self.var_analysis()
        self.tail_risk()
        self.risk_adjusted_returns()
        return self.df
    
    def drawdown_analysis(self):
        """Compute drawdown metrics."""
        close = self.df['close']
        
        # Rolling max
        rolling_max = close.expanding().max()
        drawdown = (close - rolling_max) / rolling_max * 100
        
        self.df['drawdown'] = drawdown
        self.df['drawdown_20d_max'] = drawdown.rolling(20).min()
        self.df['drawdown_60d_max'] = drawdown.rolling(60).min()
        
        # Drawdown duration (days since last high)
        is_at_high = close >= rolling_max
        self.df['days_since_high'] = (~is_at_high).cumsum() - (~is_at_high).cumsum().where(is_at_high).ffill().fillna(0)
        
    def var_analysis(self):
        """Value at Risk analysis."""
        returns = self.df['close'].pct_change()
        
        # Historical VaR (95% and 99%)
        self.df['var_95_20d'] = returns.rolling(20).quantile(0.05) * 100
        self.df['var_99_20d'] = returns.rolling(20).quantile(0.01) * 100
        
        # Parametric VaR (assuming normal distribution)
        ret_mean = returns.rolling(60).mean()
        ret_std = returns.rolling(60).std()
        self.df['var_95_parametric'] = (ret_mean - 1.645 * ret_std) * 100
        
        # Expected Shortfall (CVaR) - expected loss given VaR breach
        def cvar(x, alpha=0.05):
            var = np.percentile(x, alpha * 100)
            return x[x <= var].mean()
        
        self.df['cvar_95_20d'] = returns.rolling(60).apply(lambda x: cvar(x, 0.05)) * 100
        
    def tail_risk(self):
        """Analyze tail risk."""
        returns = self.df['close'].pct_change()
        
        # Skewness (negative = left tail, crash risk)
        self.df['skewness_60d'] = returns.rolling(60).skew()
        
        # Kurtosis (high = fat tails)
        self.df['kurtosis_60d'] = returns.rolling(60).kurt()
        
        # Tail ratio (upside vs downside)
        def tail_ratio(x, threshold=0.05):
            upper = np.percentile(x, (1 - threshold) * 100)
            lower = np.percentile(x, threshold * 100)
            return abs(upper / lower) if lower != 0 else 1
        
        self.df['tail_ratio'] = returns.rolling(60).apply(lambda x: tail_ratio(x))
        
    def risk_adjusted_returns(self):
        """Calculate risk-adjusted performance metrics."""
        returns = self.df['close'].pct_change()
        
        # Sharpe Ratio (annualized, assuming 0 risk-free rate)
        ret_mean = returns.rolling(60).mean() * 252
        ret_std = returns.rolling(60).std() * np.sqrt(252)
        self.df['sharpe_60d'] = ret_mean / (ret_std + 1e-10)
        
        # Sortino Ratio (only downside vol)
        downside_returns = returns.where(returns < 0, 0)
        downside_std = downside_returns.rolling(60).std() * np.sqrt(252)
        self.df['sortino_60d'] = ret_mean / (downside_std + 1e-10)
        
        # Calmar Ratio (return / max drawdown)
        drawdown = self.df['drawdown'] if 'drawdown' in self.df.columns else 0
        max_dd = abs(drawdown).rolling(252).max()
        self.df['calmar_ratio'] = ret_mean / (max_dd + 1e-10)
        
        # Information Ratio (would need benchmark)
        # Placeholder - compare to rolling mean return
        excess_ret = returns - returns.rolling(60).mean()
        tracking_error = excess_ret.rolling(60).std() * np.sqrt(252)
        self.df['info_ratio'] = (excess_ret.rolling(60).mean() * 252) / (tracking_error + 1e-10)


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_advanced_analysis(ticker: str, output_dir: Path) -> dict:
    """Run complete advanced quant analysis."""
    print(f"\n{'='*70}")
    print(f"ADVANCED QUANT ANALYSIS: {ticker}")
    print(f"{'='*70}")
    
    # Fetch historical data
    print("\n1. Fetching historical data...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1d")
    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  Loaded {len(df)} trading days")
    
    # Standard technical indicators
    print("\n2. Computing technical indicators...")
    ti = TechnicalIndicators(df)
    ti.rsi(14)
    ti.macd()
    ti.adx()
    ti.bollinger_bands()
    ti.atr()
    ti.ichimoku()
    ti.stochastic()
    ti.cci(20)
    ti.williams_r()
    ti.all_moving_averages()
    df = ti.df
    
    # Microstructure analysis
    print("\n3. Analyzing market microstructure...")
    micro = MicrostructureAnalyzer(df)
    df = micro.compute_all()
    
    # Cross-asset analysis
    print("\n4. Cross-asset correlation analysis...")
    cross = CrossAssetAnalyzer(ticker, df)
    asset_data = cross.fetch_cross_asset_data()
    returns_df = cross.compute_correlations(asset_data)
    cross.detect_regime(returns_df)
    cross.compute_beta_factors(returns_df)
    df = cross.df
    
    # Options analysis
    print("\n5. Options-implied metrics...")
    options = OptionsImpliedAnalyzer(ticker, df)
    options_df = options.fetch_options_data()
    options.compute_implied_metrics(options_df)
    df = options.df
    
    # Fundamental analysis
    print("\n6. Fundamental factor analysis...")
    fund = FundamentalFactorAnalyzer(ticker, df)
    if fund.fetch_fundamentals():
        fund.compute_value_factors()
        fund.compute_quality_factors()
        fund.compute_growth_factors()
        fund.compute_momentum_factors()
        fund.compute_size_factors()
    df = fund.df
    
    # Stat arb signals
    print("\n7. Statistical arbitrage signals...")
    statarb = StatArbAnalyzer(df)
    df = statarb.compute_all()
    
    # Risk analysis
    print("\n8. Risk analysis...")
    risk = RiskAnalyzer(df)
    df = risk.compute_all()
    
    # Create target variable
    print("\n9. Creating prediction targets...")
    close = df['close']
    for horizon in [1, 5, 10]:
        future_return = close.shift(-horizon) / close - 1
        df[f'target_return_{horizon}d'] = future_return
        df[f'target_direction_{horizon}d'] = (future_return > 0).astype(int)
    
    # Get feature columns
    exclude_cols = {'datetime', 'date', 'open', 'high', 'low', 'close', 'volume', 
                   'dividends', 'stock_splits', 'adj_close', 'capital_gains'}
    exclude_prefixes = ['target_']
    
    features = [col for col in df.columns 
                if col not in exclude_cols 
                and not any(col.startswith(p) for p in exclude_prefixes)
                and df[col].dtype in [np.float64, np.int64, float, int]]
    
    print(f"  Total features: {len(features)}")
    
    # ML ensemble
    print("\n10. Training ML ensemble...")
    ml = MLEnsemblePredictor(df, features, 'target_direction_5d')
    model_results = ml.train_ensemble()
    
    print("\n  Model Performance (Time-Series CV):")
    for model, metrics in model_results.items():
        if 'accuracy' in metrics:
            print(f"    {model}: {metrics['accuracy']:.1%} (+/- {metrics['std']:.1%})")
        elif 'r2' in metrics:
            print(f"    {model}: R² = {metrics['r2']:.3f}")
    
    # Feature importance
    combined_importance = ml.get_combined_importance()
    top_features = ml.get_top_features(30)
    
    print("\n" + "="*70)
    print("TOP 30 MOST PREDICTIVE FEATURES")
    print("="*70)
    for i, feat in enumerate(top_features, 1):
        imp = combined_importance[feat]
        print(f"  {i:2d}. {feat:40s}: {imp:.4f}")
    
    # Compile results
    results = {
        'ticker': ticker,
        'analysis_date': datetime.now().isoformat(),
        'n_trading_days': len(df),
        'n_features': len(features),
        'model_performance': model_results,
        'feature_importance': combined_importance,
        'top_30_features': top_features,
        'feature_categories': categorize_features(top_features)
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{ticker}_advanced_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    features_file = output_dir / f"{ticker}_advanced_features.parquet"
    df.to_parquet(features_file, index=False)
    
    print(f"\nSaved results to {results_file}")
    print(f"Saved features to {features_file}")
    
    return results


def categorize_features(features: List[str]) -> Dict[str, List[str]]:
    """Categorize features by type for analysis."""
    categories = {
        'microstructure': [],
        'cross_asset': [],
        'options': [],
        'fundamental': [],
        'technical': [],
        'risk': [],
        'stat_arb': [],
        'other': []
    }
    
    microstructure_patterns = ['vwap', 'ofi', 'cmf', 'kyle', 'amihud', 'roll_spread', 
                              'efficiency', 'hurst', 'gk_vol', 'yz_vol', 'parkinson',
                              'smi', 'force_index', 'delta', 'liquidity']
    
    cross_asset_patterns = ['corr_', 'beta_', 'regime', 'alpha_']
    
    options_patterns = ['iv_', 'skew', 'put_call', 'max_pain', 'atm_iv']
    
    fundamental_patterns = ['pe_', 'pb_', 'ps_', 'ev_', 'roe', 'roa', 'margin', 
                           'growth', 'yield', 'debt', 'cash', 'market_cap', 'analyst']
    
    risk_patterns = ['var_', 'cvar', 'drawdown', 'sharpe', 'sortino', 'calmar',
                    'skewness', 'kurtosis', 'tail']
    
    stat_arb_patterns = ['zscore', 'divergence', 'autocorr', 'bb_position']
    
    for feat in features:
        feat_lower = feat.lower()
        
        if any(p in feat_lower for p in microstructure_patterns):
            categories['microstructure'].append(feat)
        elif any(p in feat_lower for p in cross_asset_patterns):
            categories['cross_asset'].append(feat)
        elif any(p in feat_lower for p in options_patterns):
            categories['options'].append(feat)
        elif any(p in feat_lower for p in fundamental_patterns):
            categories['fundamental'].append(feat)
        elif any(p in feat_lower for p in risk_patterns):
            categories['risk'].append(feat)
        elif any(p in feat_lower for p in stat_arb_patterns):
            categories['stat_arb'].append(feat)
        else:
            categories['technical'].append(feat)
    
    return {k: v for k, v in categories.items() if v}


def main():
    parser = argparse.ArgumentParser(description='Advanced Quant Analysis')
    parser.add_argument('--tickers', nargs='+', default=['IONQ', 'NVDA'])
    parser.add_argument('--output-dir', type=str, default='technicals/data/advanced')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    all_results = []
    for ticker in args.tickers:
        try:
            results = run_advanced_analysis(ticker, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: WHAT REALLY MATTERS FOR PREDICTION")
    print("="*70)
    
    if all_results:
        # Aggregate top features across tickers
        all_top_features = {}
        for res in all_results:
            for i, feat in enumerate(res['top_30_features']):
                if feat not in all_top_features:
                    all_top_features[feat] = []
                all_top_features[feat].append(30 - i)  # Higher rank = more important
        
        # Sort by average rank
        avg_ranks = {f: np.mean(ranks) for f, ranks in all_top_features.items()}
        sorted_features = sorted(avg_ranks.items(), key=lambda x: x[1], reverse=True)
        
        print("\nCONSISTENTLY IMPORTANT FEATURES (across all tickers):")
        print("-" * 50)
        for feat, score in sorted_features[:20]:
            print(f"  {feat:40s}: avg_rank = {score:.1f}")
        
        # Category breakdown
        print("\nFEATURE CATEGORY IMPORTANCE:")
        print("-" * 50)
        for res in all_results:
            print(f"\n{res['ticker']}:")
            for cat, feats in res.get('feature_categories', {}).items():
                print(f"  {cat:20s}: {len(feats)} features")
    
    print("\n✅ Advanced quant analysis complete!")


if __name__ == '__main__':
    main()
