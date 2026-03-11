"""Technical Analysis Pipeline.

Orchestrates the full technical analysis workflow:
1. Fetch OHLCV data
2. Compute all technical indicators
3. Generate signals and pivots
4. Analyze options flow
5. Output comprehensive report

Usage:
    python analyze_technicals.py --ticker NVDA
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd

from ingest_ohlcv import fetch_ohlcv, save_ohlcv
from indicators import TechnicalIndicators, TradingStrategies
from ingest_options import fetch_options_chain, compute_greeks_summary, detect_unusual_activity, suggest_strategies


def analyze_ticker(ticker: str, period: str = '1y', interval: str = '1d',
                   include_options: bool = True) -> dict:
    """Run full technical analysis on a ticker."""
    
    results = {
        'ticker': ticker,
        'analyzed_at': datetime.now().isoformat(),
    }
    
    # 1. Fetch OHLCV
    print(f"[1/4] Fetching OHLCV data for {ticker}...")
    df = fetch_ohlcv(ticker, period, interval)
    results['price_data'] = {
        'start': str(df['datetime'].min()),
        'end': str(df['datetime'].max()),
        'records': len(df),
        'latest_close': float(df['close'].iloc[-1]),
        'latest_volume': int(df['volume'].iloc[-1]) if 'volume' in df.columns else None,
    }
    
    # 2. Compute indicators
    print(f"[2/4] Computing technical indicators...")
    ti = TechnicalIndicators(df)
    ti.compute_all()
    
    # Latest indicator values
    latest = ti.df.iloc[-1]
    results['indicators'] = {
        # Price
        'close': float(latest['close']),
        'change_1d': float(df['close'].pct_change().iloc[-1] * 100),
        'change_5d': float(df['close'].pct_change(5).iloc[-1] * 100) if len(df) > 5 else None,
        'change_20d': float(df['close'].pct_change(20).iloc[-1] * 100) if len(df) > 20 else None,
        
        # Oscillators
        'rsi_14': float(latest.get('rsi_14', 0)),
        'stoch_k_14': float(latest.get('stoch_k_14', 0)),
        'stoch_d_14': float(latest.get('stoch_d_14', 0)),
        'macd': float(latest.get('macd', 0)),
        'macd_signal': float(latest.get('macd_signal', 0)),
        'macd_histogram': float(latest.get('macd_histogram', 0)),
        'cci_20': float(latest.get('cci_20', 0)),
        'williams_r_14': float(latest.get('williams_r_14', 0)),
        'adx': float(latest.get('adx', 0)),
        'plus_di': float(latest.get('plus_di', 0)),
        'minus_di': float(latest.get('minus_di', 0)),
        'awesome_oscillator': float(latest.get('awesome_oscillator', 0)),
        'momentum_10': float(latest.get('momentum_10', 0)),
        
        # Moving Averages
        'sma_10': float(latest.get('sma_10', 0)),
        'sma_20': float(latest.get('sma_20', 0)),
        'sma_50': float(latest.get('sma_50', 0)),
        'sma_100': float(latest.get('sma_100', 0)),
        'sma_200': float(latest.get('sma_200', 0)),
        'ema_10': float(latest.get('ema_10', 0)),
        'ema_20': float(latest.get('ema_20', 0)),
        'ema_50': float(latest.get('ema_50', 0)),
        'ema_100': float(latest.get('ema_100', 0)),
        'ema_200': float(latest.get('ema_200', 0)),
        'vwma_20': float(latest.get('vwma_20', 0)) if 'vwma_20' in latest else None,
        'hull_ma_9': float(latest.get('hull_ma_9', 0)) if 'hull_ma_9' in latest else None,
        
        # Ichimoku
        'ichimoku_tenkan': float(latest.get('ichimoku_tenkan', 0)),
        'ichimoku_kijun': float(latest.get('ichimoku_kijun', 0)),
        
        # Volatility
        'atr_14': float(latest.get('atr_14', 0)),
        'bb_upper': float(latest.get('bb_upper', 0)),
        'bb_middle': float(latest.get('bb_middle', 0)),
        'bb_lower': float(latest.get('bb_lower', 0)),
        
        # Volume
        'obv': float(latest.get('obv', 0)) if 'obv' in latest else None,
    }
    
    # 3. Generate signals and pivots
    print(f"[3/4] Generating signals and pivot points...")
    
    # Signals
    signals = ti.generate_signals()
    results['signals'] = signals
    
    # Pivot points (all types)
    results['pivots'] = {
        'classic': ti.pivot_classic(),
        'fibonacci': ti.pivot_fibonacci(),
        'camarilla': ti.pivot_camarilla(),
        'woodie': ti.pivot_woodie(),
        'demark': ti.pivot_demark(),
    }
    
    # MA signals
    ma_signals = {}
    close = float(latest['close'])
    for period in [10, 20, 50, 100, 200]:
        sma = latest.get(f'sma_{period}', 0)
        ema = latest.get(f'ema_{period}', 0)
        if sma:
            ma_signals[f'sma_{period}'] = 'BUY' if close > sma else 'SELL'
        if ema:
            ma_signals[f'ema_{period}'] = 'BUY' if close > ema else 'SELL'
    results['ma_signals'] = ma_signals
    
    # Oscillator signals
    osc_signals = {}
    rsi = latest.get('rsi_14', 50)
    if rsi < 30:
        osc_signals['rsi'] = 'BUY'
    elif rsi > 70:
        osc_signals['rsi'] = 'SELL'
    else:
        osc_signals['rsi'] = 'NEUTRAL'
    
    stoch_k = latest.get('stoch_k_14', 50)
    stoch_d = latest.get('stoch_d_14', 50)
    if stoch_k < 20 and stoch_k > stoch_d:
        osc_signals['stochastic'] = 'BUY'
    elif stoch_k > 80 and stoch_k < stoch_d:
        osc_signals['stochastic'] = 'SELL'
    else:
        osc_signals['stochastic'] = 'NEUTRAL'
    
    cci = latest.get('cci_20', 0)
    if cci < -100:
        osc_signals['cci'] = 'BUY'
    elif cci > 100:
        osc_signals['cci'] = 'SELL'
    else:
        osc_signals['cci'] = 'NEUTRAL'
    
    results['oscillator_signals'] = osc_signals
    
    # Summary rating
    buy_count = sum(1 for v in {**ma_signals, **osc_signals}.values() if v == 'BUY')
    sell_count = sum(1 for v in {**ma_signals, **osc_signals}.values() if v == 'SELL')
    total = buy_count + sell_count
    
    if total > 0:
        buy_pct = buy_count / total
        if buy_pct > 0.7:
            results['summary_rating'] = 'STRONG BUY'
        elif buy_pct > 0.5:
            results['summary_rating'] = 'BUY'
        elif buy_pct < 0.3:
            results['summary_rating'] = 'STRONG SELL'
        elif buy_pct < 0.5:
            results['summary_rating'] = 'SELL'
        else:
            results['summary_rating'] = 'NEUTRAL'
    else:
        results['summary_rating'] = 'NEUTRAL'
    
    # 4. Options analysis (if requested)
    if include_options:
        print(f"[4/4] Analyzing options flow...")
        try:
            opt_data = fetch_options_chain(ticker)
            if 'chain' in opt_data:
                chain = opt_data['chain']
                current_price = opt_data['current_price']
                
                results['options'] = {
                    'current_price': current_price,
                    'expirations_count': len(opt_data['expirations']),
                    'total_contracts': len(chain),
                    'greeks_summary': compute_greeks_summary(chain, current_price),
                    'unusual_activity_count': len(detect_unusual_activity(chain)),
                }
                
                # Strategy suggestions based on signals
                if results['summary_rating'] in ['STRONG BUY', 'BUY']:
                    outlook = 'bullish'
                elif results['summary_rating'] in ['STRONG SELL', 'SELL']:
                    outlook = 'bearish'
                else:
                    outlook = 'neutral'
                
                results['options']['suggested_strategies'] = suggest_strategies(current_price, chain, outlook)
        except Exception as e:
            print(f"  Options analysis failed: {e}")
            results['options'] = {'error': str(e)}
    else:
        print(f"[4/4] Skipping options analysis")
    
    # Save indicator data
    save_ohlcv(ticker, ti.df, interval)
    
    return results


def save_analysis(ticker: str, results: dict, base_path: str = 'technicals/data'):
    """Save analysis results to JSON."""
    os.makedirs(base_path, exist_ok=True)
    
    # Convert any non-serializable objects
    def serialize(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if hasattr(obj, 'item'):  # numpy types
            return obj.item()
        return obj
    
    # Deep copy with serialization
    import copy
    serialized = json.loads(json.dumps(results, default=serialize))
    
    path = os.path.join(base_path, f'{ticker}_technical_analysis.json')
    with open(path, 'w') as f:
        json.dump(serialized, f, indent=2)
    
    print(f"\nSaved analysis: {path}")
    return path


def print_report(results: dict):
    """Print formatted analysis report."""
    ticker = results['ticker']
    
    print("\n" + "=" * 60)
    print(f"  TECHNICAL ANALYSIS REPORT: {ticker}")
    print("=" * 60)
    
    # Price
    ind = results['indicators']
    print(f"\n📈 PRICE")
    print(f"  Close: ${ind['close']:.2f}")
    print(f"  1D Change: {ind['change_1d']:+.2f}%")
    if ind.get('change_5d'):
        print(f"  5D Change: {ind['change_5d']:+.2f}%")
    if ind.get('change_20d'):
        print(f"  20D Change: {ind['change_20d']:+.2f}%")
    
    # Summary
    print(f"\n⭐ SUMMARY: {results['summary_rating']}")
    
    # Oscillators
    print(f"\n📊 OSCILLATORS")
    print(f"  RSI(14): {ind['rsi_14']:.2f} [{results['oscillator_signals'].get('rsi', 'N/A')}]")
    print(f"  Stochastic: K={ind['stoch_k_14']:.2f}, D={ind['stoch_d_14']:.2f} [{results['oscillator_signals'].get('stochastic', 'N/A')}]")
    print(f"  MACD: {ind['macd']:.4f} (Signal: {ind['macd_signal']:.4f})")
    print(f"  CCI(20): {ind['cci_20']:.2f} [{results['oscillator_signals'].get('cci', 'N/A')}]")
    print(f"  ADX: {ind['adx']:.2f} (+DI: {ind['plus_di']:.2f}, -DI: {ind['minus_di']:.2f})")
    
    # Moving Averages
    print(f"\n📈 MOVING AVERAGES")
    for period in [20, 50, 200]:
        sma = ind.get(f'sma_{period}')
        ema = ind.get(f'ema_{period}')
        if sma:
            sig = results['ma_signals'].get(f'sma_{period}', 'N/A')
            print(f"  SMA({period}): ${sma:.2f} [{sig}]")
        if ema:
            sig = results['ma_signals'].get(f'ema_{period}', 'N/A')
            print(f"  EMA({period}): ${ema:.2f} [{sig}]")
    
    # Pivot Points
    print(f"\n🎯 PIVOT POINTS (Classic)")
    pivots = results['pivots']['classic']
    print(f"  R3: ${pivots['R3']:.2f}")
    print(f"  R2: ${pivots['R2']:.2f}")
    print(f"  R1: ${pivots['R1']:.2f}")
    print(f"  P:  ${pivots['P']:.2f}")
    print(f"  S1: ${pivots['S1']:.2f}")
    print(f"  S2: ${pivots['S2']:.2f}")
    print(f"  S3: ${pivots['S3']:.2f}")
    
    # Options
    if 'options' in results and 'greeks_summary' in results.get('options', {}):
        opt = results['options']
        gs = opt['greeks_summary']
        print(f"\n📋 OPTIONS FLOW")
        print(f"  Put/Call Volume Ratio: {gs.get('put_call_volume_ratio', 'N/A'):.2f}" if gs.get('put_call_volume_ratio') else "  Put/Call Volume Ratio: N/A")
        print(f"  Put/Call OI Ratio: {gs.get('put_call_oi_ratio', 'N/A'):.2f}" if gs.get('put_call_oi_ratio') else "  Put/Call OI Ratio: N/A")
        print(f"  Max Pain Strike: ${gs.get('max_pain_strike', 'N/A')}" if gs.get('max_pain_strike') else "  Max Pain Strike: N/A")
        print(f"  IV Skew: {gs.get('iv_skew', 'N/A'):.4f}" if gs.get('iv_skew') else "  IV Skew: N/A")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Full technical analysis pipeline')
    parser.add_argument('--ticker', '-t', required=True, help='Stock ticker symbol')
    parser.add_argument('--period', '-p', default='1y', help='Data period')
    parser.add_argument('--interval', '-i', default='1d', help='Data interval')
    parser.add_argument('--no-options', action='store_true', help='Skip options analysis')
    args = parser.parse_args()
    
    results = analyze_ticker(
        args.ticker, 
        args.period, 
        args.interval,
        include_options=not args.no_options
    )
    
    print_report(results)
    save_analysis(args.ticker, results)


if __name__ == '__main__':
    main()
