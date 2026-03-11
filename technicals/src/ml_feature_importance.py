#!/usr/bin/env python3
"""
ML Feature Importance Analysis for Technical Indicators

Backpropagates through historical price data to determine which technical
indicators have the most predictive power for price movements.

Uses:
- Random Forest feature importance
- XGBoost gain importance
- SHAP values for interpretability
- Correlation analysis at different lags
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Import our indicators class
from indicators import TechnicalIndicators


def fetch_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical OHLCV data for analysis."""
    print(f"Fetching {period} of historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval="1d")
    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Rename for consistency
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'datetime' not in df.columns:
        df['datetime'] = df.index
    
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  Loaded {len(df)} trading days")
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for each day using TechnicalIndicators class."""
    # Use the TechnicalIndicators class
    ti = TechnicalIndicators(df)
    
    # === Oscillators ===
    ti.rsi(14)
    ti.rsi(7)
    ti.rsi(21)
    ti.stochastic(14, 3, 3)
    ti.stochastic_rsi(14, 14, 3, 3)
    ti.macd(12, 26, 9)
    ti.cci(20)
    ti.cci(14)
    ti.williams_r(14)
    ti.adx(14)
    ti.awesome_oscillator()
    ti.momentum(10)
    ti.momentum(20)
    ti.ultimate_oscillator(7, 14, 28)
    ti.bull_bear_power(13)
    
    # === Moving Averages ===
    for period in [5, 10, 20, 30, 50, 100, 200]:
        ti.sma(period)
        ti.ema(period)
    ti.wma(20)
    ti.hull_ma(9)
    ti.vwma(20)
    
    # === Ichimoku ===
    ti.ichimoku()
    
    # === Volatility ===
    ti.atr(14)
    ti.bollinger_bands(20, 2)
    
    # Get the result DataFrame with all indicators
    result = ti.df.copy()
    
    # === Additional derived features ===
    close = result['close'].values
    high = result['high'].values
    low = result['low'].values
    volume = result['volume'].values if 'volume' in result.columns else np.ones(len(close))
    open_price = result['open'].values
    
    # Distance from MAs (normalized)
    for period in [5, 10, 20, 50, 100, 200]:
        if f'sma_{period}' in result.columns:
            result[f'dist_sma_{period}'] = (close - result[f'sma_{period}'].values) / close * 100
        if f'ema_{period}' in result.columns:
            result[f'dist_ema_{period}'] = (close - result[f'ema_{period}'].values) / close * 100
    
    # MA Crossovers (as binary signals)
    if 'sma_5' in result.columns and 'sma_20' in result.columns:
        result['sma_5_20_cross'] = (result['sma_5'] > result['sma_20']).astype(int)
    if 'sma_20' in result.columns and 'sma_50' in result.columns:
        result['sma_20_50_cross'] = (result['sma_20'] > result['sma_50']).astype(int)
    if 'sma_50' in result.columns and 'sma_200' in result.columns:
        result['sma_50_200_cross'] = (result['sma_50'] > result['sma_200']).astype(int)
    if 'ema_10' in result.columns and 'ema_20' in result.columns:
        result['ema_10_20_cross'] = (result['ema_10'] > result['ema_20']).astype(int)
    
    # RSI zones
    if 'rsi_14' in result.columns:
        result['rsi_oversold'] = (result['rsi_14'] < 30).astype(int)
        result['rsi_overbought'] = (result['rsi_14'] > 70).astype(int)
    
    # Stochastic cross
    if 'stoch_k_14' in result.columns and 'stoch_d_14' in result.columns:
        result['stoch_cross'] = (result['stoch_k_14'] > result['stoch_d_14']).astype(int)
    
    # DI cross
    if 'plus_di' in result.columns and 'minus_di' in result.columns:
        result['di_cross'] = (result['plus_di'] > result['minus_di']).astype(int)
    
    # ADX strong trend
    if 'adx' in result.columns:
        result['adx_strong_trend'] = (result['adx'] > 25).astype(int)
    
    # MACD cross
    if 'macd' in result.columns and 'macd_signal' in result.columns:
        result['macd_cross'] = (result['macd'] > result['macd_signal']).astype(int)
        result['macd_positive'] = (result['macd'] > 0).astype(int)
    
    # Bollinger Band features
    if 'bb_upper' in result.columns and 'bb_lower' in result.columns:
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle'] * 100
        result['bb_position'] = (close - result['bb_lower'].values) / (result['bb_upper'].values - result['bb_lower'].values + 1e-10)
    
    # ATR percentage
    if 'atr_14' in result.columns:
        result['atr_pct'] = result['atr_14'] / close * 100
    
    # Ichimoku cloud signals
    if 'ichimoku_senkou_a' in result.columns and 'ichimoku_senkou_b' in result.columns:
        result['above_cloud'] = ((close > result['ichimoku_senkou_a'].values) & 
                                  (close > result['ichimoku_senkou_b'].values)).astype(int)
        result['below_cloud'] = ((close < result['ichimoku_senkou_a'].values) & 
                                  (close < result['ichimoku_senkou_b'].values)).astype(int)
    if 'ichimoku_tenkan' in result.columns and 'ichimoku_kijun' in result.columns:
        result['ichimoku_tk_cross'] = (result['ichimoku_tenkan'] > result['ichimoku_kijun']).astype(int)
    
    # MFI (Money Flow Index) - manual calculation
    if 'volume' in result.columns:
        tp = (result['high'] + result['low'] + result['close']) / 3
        mf = tp * result['volume']
        pos_mf = mf.where(tp > tp.shift(1), 0)
        neg_mf = mf.where(tp < tp.shift(1), 0)
        mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
        result['mfi_14'] = 100 - (100 / (1 + mfr))
        result['mfi_oversold'] = (result['mfi_14'] < 20).astype(int)
        result['mfi_overbought'] = (result['mfi_14'] > 80).astype(int)
    
    # OBV (On Balance Volume)
    if 'volume' in result.columns:
        obv = (np.sign(result['close'].diff()) * result['volume']).fillna(0).cumsum()
        result['obv'] = obv
        result['obv_sma_20'] = obv.rolling(20).mean()
        result['obv_trend'] = (result['obv'] > result['obv_sma_20']).astype(int)
    
    # Volume relative to average
    if 'volume' in result.columns:
        vol_sma = result['volume'].rolling(20).mean()
        result['volume_ratio'] = result['volume'] / (vol_sma + 1e-10)
        result['high_volume'] = (result['volume_ratio'] > 1.5).astype(int)
    
    # === Price Action Features ===
    result['daily_return'] = result['close'].pct_change() * 100
    result['daily_range'] = (high - low) / close * 100
    result['body_size'] = abs(close - open_price) / close * 100
    result['upper_wick'] = (high - np.maximum(close, open_price)) / close * 100
    result['lower_wick'] = (np.minimum(close, open_price) - low) / close * 100
    
    # Candle patterns (simplified)
    result['bullish_candle'] = (close > open_price).astype(int)
    result['doji'] = (result['body_size'] < 0.1).astype(int)
    
    # Consecutive days
    result['consec_up'] = 0
    result['consec_down'] = 0
    
    consec_up = 0
    consec_down = 0
    for i in range(1, len(result)):
        if close[i] > close[i-1]:
            consec_up += 1
            consec_down = 0
        elif close[i] < close[i-1]:
            consec_down += 1
            consec_up = 0
        else:
            consec_up = 0
            consec_down = 0
        result.iloc[i, result.columns.get_loc('consec_up')] = consec_up
        result.iloc[i, result.columns.get_loc('consec_down')] = consec_down
    
    # Distance from 52-week high/low
    rolling_high_252 = pd.Series(high).rolling(252, min_periods=20).max()
    rolling_low_252 = pd.Series(low).rolling(252, min_periods=20).min()
    result['dist_52w_high'] = (close - rolling_high_252.values) / close * 100
    result['dist_52w_low'] = (close - rolling_low_252.values) / close * 100
    
    # Recent highs/lows
    rolling_high_20 = pd.Series(high).rolling(20).max()
    rolling_low_20 = pd.Series(low).rolling(20).min()
    result['at_20d_high'] = (close >= rolling_high_20.values * 0.99).astype(int)
    result['at_20d_low'] = (close <= rolling_low_20.values * 1.01).astype(int)
    
    return result


def create_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for prediction."""
    close = df['close'].values
    
    # Future returns at different horizons
    for horizon in [1, 2, 3, 5, 10, 20]:
        future_close = np.roll(close, -horizon)
        future_return = (future_close - close) / close * 100
        future_return[-horizon:] = np.nan
        df[f'return_{horizon}d'] = future_return
        df[f'direction_{horizon}d'] = (future_return > 0).astype(int)
        df.loc[df[f'return_{horizon}d'].isna(), f'direction_{horizon}d'] = np.nan
    
    # Significant moves (>2% in 5 days)
    df['big_up_5d'] = (df['return_5d'] > 2).astype(int)
    df['big_down_5d'] = (df['return_5d'] < -2).astype(int)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns (excluding targets and metadata)."""
    exclude_prefixes = ['return_', 'direction_', 'big_up', 'big_down']
    exclude_cols = ['datetime', 'date', 'open', 'high', 'low', 'close', 'volume', 
                   'dividends', 'stock_splits', 'adj_close', 'capital_gains']
    
    features = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            features.append(col)
    
    return features


def train_feature_importance_model(df: pd.DataFrame, target: str, features: list) -> dict:
    """Train Random Forest and extract feature importances."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    
    # Prepare data
    X = df[features].copy()
    y = df[target].copy()
    
    # Remove NaN rows
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 100:
        print(f"  Not enough data for {target}: {len(X)} samples")
        return None
    
    # Fill any remaining NaN with 0
    X = X.fillna(0)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = dict(zip(features, rf.feature_importances_))
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    
    # Gradient Boosting for comparison
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X, y)
    gb_importance = dict(zip(features, gb.feature_importances_))
    
    return {
        'rf_importance': rf_importance,
        'gb_importance': gb_importance,
        'cv_accuracy': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'n_samples': len(X)
    }


def compute_correlation_analysis(df: pd.DataFrame, features: list, target: str) -> dict:
    """Compute correlation between features and future returns."""
    correlations = {}
    
    y = df[target].dropna()
    
    for feat in features:
        x = df.loc[y.index, feat]
        valid = ~(x.isna() | y.isna())
        if valid.sum() > 50:
            corr = np.corrcoef(x[valid], y[valid])[0, 1]
            correlations[feat] = float(corr) if not np.isnan(corr) else 0.0
        else:
            correlations[feat] = 0.0
    
    return correlations


def compute_signal_analysis(df: pd.DataFrame, features: list) -> dict:
    """
    Analyze how well each indicator predicts price direction.
    For each indicator, compute win rate when signal is triggered.
    """
    results = {}
    
    # Binary/threshold signals
    signal_mappings = {
        'rsi_oversold': {'condition': 'rsi_14 < 30', 'expect': 'up'},
        'rsi_overbought': {'condition': 'rsi_14 > 70', 'expect': 'down'},
        'macd_cross': {'condition': 'macd > macd_signal', 'expect': 'up'},
        'sma_50_200_cross': {'condition': 'sma_50 > sma_200', 'expect': 'up'},
        'di_cross': {'condition': 'plus_di > minus_di', 'expect': 'up'},
        'stoch_cross': {'condition': 'stoch_k > stoch_d', 'expect': 'up'},
        'above_cloud': {'condition': 'close > ichimoku_cloud', 'expect': 'up'},
        'mfi_oversold': {'condition': 'mfi_14 < 20', 'expect': 'up'},
        'mfi_overbought': {'condition': 'mfi_14 > 80', 'expect': 'down'},
    }
    
    for horizon in [1, 5, 10]:
        return_col = f'return_{horizon}d'
        if return_col not in df.columns:
            continue
            
        horizon_results = {}
        
        for signal_name, config in signal_mappings.items():
            if signal_name not in df.columns:
                continue
            
            signal_on = df[signal_name] == 1
            signal_off = df[signal_name] == 0
            
            returns_when_on = df.loc[signal_on, return_col].dropna()
            returns_when_off = df.loc[signal_off, return_col].dropna()
            
            if len(returns_when_on) < 10:
                continue
            
            expect_up = config['expect'] == 'up'
            
            if expect_up:
                win_rate = (returns_when_on > 0).mean()
                avg_return = returns_when_on.mean()
            else:
                win_rate = (returns_when_on < 0).mean()
                avg_return = -returns_when_on.mean()
            
            horizon_results[signal_name] = {
                'win_rate': float(win_rate),
                'avg_return': float(avg_return),
                'n_signals': int(len(returns_when_on)),
                'expected_direction': config['expect']
            }
        
        results[f'{horizon}d'] = horizon_results
    
    return results


def run_shap_analysis(df: pd.DataFrame, target: str, features: list, top_n: int = 20) -> dict:
    """Run SHAP analysis for feature importance interpretation."""
    try:
        import shap
    except ImportError:
        print("  SHAP not installed, skipping SHAP analysis")
        return {}
    
    from sklearn.ensemble import RandomForestClassifier
    
    X = df[features].copy()
    y = df[target].copy()
    
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].fillna(0)
    y = y[valid_mask]
    
    if len(X) < 100:
        return {}
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # SHAP values (use subset for speed)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values is a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (positive direction)
        
        # Handle 3D arrays from newer SHAP versions
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # Take class 1
        
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure mean_shap is 1D
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.flatten()[:len(features)]
        
        shap_importance = {}
        for i, feat in enumerate(features):
            if i < len(mean_shap):
                val = mean_shap[i]
                shap_importance[feat] = float(val) if np.isscalar(val) else float(val.item() if hasattr(val, 'item') else np.mean(val))
        
        # Sort and return top N
        sorted_shap = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        return sorted_shap
    except Exception as e:
        print(f"    SHAP error: {e}")
        return {}


def analyze_indicator_regimes(df: pd.DataFrame, features: list) -> dict:
    """
    Analyze how indicators perform in different market regimes.
    Regimes: trending up, trending down, sideways, high volatility, low volatility
    """
    results = {}
    
    # Define regimes
    df = df.copy()
    
    # Trend regime based on 50-day SMA slope
    sma_50_pct_change = df['sma_50'].pct_change(20) * 100
    df['regime_trend_up'] = sma_50_pct_change > 2
    df['regime_trend_down'] = sma_50_pct_change < -2
    df['regime_sideways'] = (sma_50_pct_change >= -2) & (sma_50_pct_change <= 2)
    
    # Volatility regime based on ATR percentile
    atr_pct_rank = df['atr_pct'].rank(pct=True)
    df['regime_high_vol'] = atr_pct_rank > 0.7
    df['regime_low_vol'] = atr_pct_rank < 0.3
    
    regimes = ['regime_trend_up', 'regime_trend_down', 'regime_sideways', 'regime_high_vol', 'regime_low_vol']
    
    for regime in regimes:
        regime_mask = df[regime] == True
        if regime_mask.sum() < 30:
            continue
        
        regime_df = df[regime_mask]
        
        # Calculate which indicators had best predictive power in this regime
        regime_corrs = {}
        for feat in features[:30]:  # Top 30 features
            if feat in df.columns and 'return_5d' in df.columns:
                valid = ~(regime_df[feat].isna() | regime_df['return_5d'].isna())
                if valid.sum() > 20:
                    corr = np.corrcoef(regime_df.loc[valid.values, feat], 
                                      regime_df.loc[valid.values, 'return_5d'])[0, 1]
                    if not np.isnan(corr):
                        regime_corrs[feat] = float(corr)
        
        # Sort by absolute correlation
        sorted_corrs = dict(sorted(regime_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        results[regime.replace('regime_', '')] = {
            'n_days': int(regime_mask.sum()),
            'top_indicators': sorted_corrs
        }
    
    return results


def generate_importance_report(results: dict, ticker: str) -> str:
    """Generate a human-readable report of findings."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"TECHNICAL INDICATOR IMPORTANCE ANALYSIS: {ticker}")
    lines.append("=" * 70)
    lines.append("")
    
    # Model Performance
    if 'model_results' in results:
        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 40)
        for target, model_res in results['model_results'].items():
            if model_res:
                acc = model_res['cv_accuracy']
                std = model_res['cv_std']
                lines.append(f"  {target}: Accuracy = {acc:.1%} (+/- {std:.1%})")
        lines.append("")
    
    # Top Features by Random Forest
    if 'combined_importance' in results:
        lines.append("TOP 20 MOST IMPORTANT INDICATORS")
        lines.append("-" * 40)
        sorted_imp = sorted(results['combined_importance'].items(), key=lambda x: x[1], reverse=True)[:20]
        for i, (feat, imp) in enumerate(sorted_imp, 1):
            lines.append(f"  {i:2d}. {feat:30s} : {imp:.4f}")
        lines.append("")
    
    # Signal Win Rates
    if 'signal_analysis' in results:
        lines.append("SIGNAL WIN RATES (5-day horizon)")
        lines.append("-" * 40)
        if '5d' in results['signal_analysis']:
            signals = results['signal_analysis']['5d']
            sorted_signals = sorted(signals.items(), key=lambda x: x[1]['win_rate'], reverse=True)
            for sig_name, sig_data in sorted_signals:
                wr = sig_data['win_rate']
                ar = sig_data['avg_return']
                n = sig_data['n_signals']
                direction = sig_data['expected_direction']
                lines.append(f"  {sig_name:25s}: Win Rate={wr:.1%}, Avg Return={ar:+.2f}%, N={n} ({direction})")
        lines.append("")
    
    # Regime Analysis
    if 'regime_analysis' in results:
        lines.append("BEST INDICATORS BY MARKET REGIME")
        lines.append("-" * 40)
        for regime, regime_data in results['regime_analysis'].items():
            n_days = regime_data['n_days']
            lines.append(f"\n  {regime.upper()} ({n_days} days):")
            for ind, corr in list(regime_data['top_indicators'].items())[:5]:
                lines.append(f"    - {ind}: correlation = {corr:+.3f}")
        lines.append("")
    
    # Correlation with Future Returns
    if 'correlations' in results:
        lines.append("STRONGEST CORRELATIONS WITH 5-DAY RETURNS")
        lines.append("-" * 40)
        corrs = results['correlations'].get('return_5d', {})
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        for feat, corr in sorted_corrs:
            direction = "↑" if corr > 0 else "↓"
            lines.append(f"  {feat:35s}: {corr:+.3f} {direction}")
        lines.append("")
    
    # Key Insights
    lines.append("KEY INSIGHTS")
    lines.append("-" * 40)
    
    if 'combined_importance' in results:
        top_5 = list(sorted(results['combined_importance'].items(), key=lambda x: x[1], reverse=True)[:5])
        lines.append(f"  1. Most predictive indicators: {', '.join([x[0] for x in top_5])}")
    
    if 'signal_analysis' in results and '5d' in results['signal_analysis']:
        signals = results['signal_analysis']['5d']
        best_signal = max(signals.items(), key=lambda x: x[1]['win_rate']) if signals else None
        if best_signal:
            lines.append(f"  2. Best signal: {best_signal[0]} with {best_signal[1]['win_rate']:.1%} win rate")
    
    if 'correlations' in results:
        corrs = results['correlations'].get('return_5d', {})
        if corrs:
            strongest = max(corrs.items(), key=lambda x: abs(x[1]))
            lines.append(f"  3. Strongest correlation: {strongest[0]} ({strongest[1]:+.3f})")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def analyze_ticker(ticker: str, output_dir: Path) -> dict:
    """Run full analysis for a single ticker."""
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}")
    print(f"{'='*60}")
    
    # Fetch data
    df = fetch_historical_data(ticker, period="2y")
    
    # Compute indicators
    print("Computing technical indicators...")
    df = compute_all_indicators(df)
    
    # Create targets
    print("Creating target labels...")
    df = create_target_labels(df)
    
    # Get feature columns
    features = get_feature_columns(df)
    print(f"  {len(features)} features extracted")
    
    results = {
        'ticker': ticker,
        'analysis_date': datetime.now().isoformat(),
        'n_trading_days': len(df),
        'date_range': {
            'start': df['datetime'].min().strftime('%Y-%m-%d'),
            'end': df['datetime'].max().strftime('%Y-%m-%d')
        }
    }
    
    # Train models for different targets
    print("Training feature importance models...")
    model_results = {}
    combined_importance = {f: 0.0 for f in features}
    
    for target in ['direction_1d', 'direction_5d', 'direction_10d']:
        print(f"  Target: {target}")
        model_res = train_feature_importance_model(df, target, features)
        if model_res:
            model_results[target] = model_res
            # Accumulate importance
            for f, imp in model_res['rf_importance'].items():
                combined_importance[f] += imp
            for f, imp in model_res['gb_importance'].items():
                combined_importance[f] += imp
    
    # Normalize combined importance
    total_imp = sum(combined_importance.values())
    if total_imp > 0:
        combined_importance = {k: v/total_imp for k, v in combined_importance.items()}
    
    results['model_results'] = model_results
    results['combined_importance'] = combined_importance
    
    # Correlation analysis
    print("Computing correlation analysis...")
    correlations = {}
    for target in ['return_1d', 'return_5d', 'return_10d']:
        correlations[target] = compute_correlation_analysis(df, features, target)
    results['correlations'] = correlations
    
    # Signal analysis
    print("Analyzing signal win rates...")
    results['signal_analysis'] = compute_signal_analysis(df, features)
    
    # Regime analysis
    print("Analyzing market regimes...")
    results['regime_analysis'] = analyze_indicator_regimes(df, features)
    
    # SHAP analysis (if available)
    print("Running SHAP analysis...")
    results['shap_importance'] = run_shap_analysis(df, 'direction_5d', features)
    
    # Generate report
    report = generate_importance_report(results, ticker)
    print(report)
    
    # Save results
    output_file = output_dir / f"{ticker}_indicator_importance.json"
    with open(output_file, 'w') as f:
        # Convert any remaining numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\nSaved results to {output_file}")
    
    # Save report
    report_file = output_dir / f"{ticker}_importance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_file}")
    
    # Save feature dataset for further analysis
    feature_file = output_dir / f"{ticker}_ml_features.parquet"
    df.to_parquet(feature_file, index=False)
    print(f"Saved feature dataset to {feature_file}")
    
    return results


def aggregate_multi_ticker_results(results_list: list) -> dict:
    """Aggregate results across multiple tickers."""
    if not results_list:
        return {}
    
    # Combine feature importances
    all_features = set()
    for res in results_list:
        all_features.update(res.get('combined_importance', {}).keys())
    
    avg_importance = {f: 0.0 for f in all_features}
    for res in results_list:
        imp = res.get('combined_importance', {})
        for f in all_features:
            avg_importance[f] += imp.get(f, 0.0)
    
    n = len(results_list)
    avg_importance = {k: v/n for k, v in avg_importance.items()}
    
    # Sort by importance
    sorted_importance = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'n_tickers': n,
        'tickers': [r['ticker'] for r in results_list],
        'average_importance': sorted_importance,
        'top_20_indicators': dict(list(sorted_importance.items())[:20])
    }


def main():
    parser = argparse.ArgumentParser(description='ML Feature Importance Analysis for Technical Indicators')
    parser.add_argument('--tickers', nargs='+', default=['IONQ', 'NVDA'], 
                       help='Tickers to analyze')
    parser.add_argument('--output-dir', type=str, default='technicals/data/ml',
                       help='Output directory for results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for ticker in args.tickers:
        try:
            results = analyze_ticker(ticker, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate results
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS ACROSS ALL TICKERS")
        print("=" * 70)
        
        agg = aggregate_multi_ticker_results(all_results)
        
        print(f"\nAnalyzed {agg['n_tickers']} tickers: {', '.join(agg['tickers'])}")
        print("\nTOP 20 MOST IMPORTANT INDICATORS (AVERAGED):")
        print("-" * 50)
        for i, (feat, imp) in enumerate(agg['top_20_indicators'].items(), 1):
            print(f"  {i:2d}. {feat:35s}: {imp:.4f}")
        
        # Save aggregated results
        agg_file = output_dir / "aggregated_importance.json"
        with open(agg_file, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved aggregated results to {agg_file}")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
