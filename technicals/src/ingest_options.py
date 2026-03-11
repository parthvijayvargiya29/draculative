"""Options chain data ingestion and analysis.

Fetches options data from yfinance and computes Greeks, IV analysis,
unusual activity detection, and strategy recommendations.

Usage:
    python ingest_options.py --ticker NVDA
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


def fetch_options_chain(ticker: str) -> Dict:
    """Fetch options chain data from yfinance."""
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    
    # Get all available expiration dates
    expirations = stock.options
    
    # Current stock price
    info = stock.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
    
    all_chains = []
    
    for exp in expirations[:12]:  # Limit to next 12 expirations
        try:
            opt = stock.option_chain(exp)
            
            # Process calls
            calls = opt.calls.copy()
            calls['type'] = 'call'
            calls['expiration'] = exp
            
            # Process puts
            puts = opt.puts.copy()
            puts['type'] = 'put'
            puts['expiration'] = exp
            
            all_chains.append(calls)
            all_chains.append(puts)
        except Exception as e:
            print(f"Error fetching {exp}: {e}")
            continue
    
    if not all_chains:
        return {'ticker': ticker, 'error': 'No options data available'}
    
    df = pd.concat(all_chains, ignore_index=True)
    
    # Clean up column names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'expirations': list(expirations),
        'fetched_at': datetime.now().isoformat(),
        'chain': df
    }


def compute_greeks_summary(chain_df: pd.DataFrame, current_price: float) -> Dict:
    """Compute aggregate Greeks and flow analysis."""
    
    # Split by type
    calls = chain_df[chain_df['type'] == 'call']
    puts = chain_df[chain_df['type'] == 'put']
    
    # ATM strikes (within 5% of current price)
    atm_range = (current_price * 0.95, current_price * 1.05)
    
    atm_calls = calls[(calls['strike'] >= atm_range[0]) & (calls['strike'] <= atm_range[1])]
    atm_puts = puts[(puts['strike'] >= atm_range[0]) & (puts['strike'] <= atm_range[1])]
    
    # Volume-weighted IV
    def weighted_iv(df):
        if df.empty or 'impliedvolatility' not in df.columns:
            return None
        vol = df['volume'].fillna(0)
        iv = df['impliedvolatility'].fillna(0)
        if vol.sum() == 0:
            return iv.mean()
        return (iv * vol).sum() / vol.sum()
    
    # Put/Call ratios
    total_call_volume = calls['volume'].fillna(0).sum()
    total_put_volume = puts['volume'].fillna(0).sum()
    pc_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
    
    total_call_oi = calls['openinterest'].fillna(0).sum()
    total_put_oi = puts['openinterest'].fillna(0).sum()
    pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
    
    # Aggregate Greeks (if available)
    delta_cols = ['delta'] if 'delta' in chain_df.columns else []
    gamma_cols = ['gamma'] if 'gamma' in chain_df.columns else []
    
    total_gamma = chain_df['gamma'].fillna(0).sum() if 'gamma' in chain_df.columns else None
    
    # Max pain calculation (strike with max combined OI)
    strike_oi = chain_df.groupby('strike')['openinterest'].sum()
    max_pain_strike = strike_oi.idxmax() if not strike_oi.empty else None
    
    return {
        'atm_call_iv': weighted_iv(atm_calls),
        'atm_put_iv': weighted_iv(atm_puts),
        'iv_skew': (weighted_iv(atm_puts) or 0) - (weighted_iv(atm_calls) or 0),
        'total_call_volume': int(total_call_volume),
        'total_put_volume': int(total_put_volume),
        'put_call_volume_ratio': pc_volume_ratio,
        'total_call_oi': int(total_call_oi),
        'total_put_oi': int(total_put_oi),
        'put_call_oi_ratio': pc_oi_ratio,
        'max_pain_strike': max_pain_strike,
        'total_gamma': total_gamma,
    }


def detect_unusual_activity(chain_df: pd.DataFrame, 
                            volume_threshold: float = 2.0,
                            oi_threshold: float = 0.5) -> pd.DataFrame:
    """Detect unusual options activity.
    
    Criteria:
    - Volume > threshold * Open Interest (high turnover)
    - Volume significantly above average
    - Large OI changes
    """
    df = chain_df.copy()
    
    # Volume to OI ratio
    df['vol_oi_ratio'] = df['volume'] / df['openinterest'].replace(0, np.nan)
    
    # Flag unusual
    df['unusual'] = (
        (df['vol_oi_ratio'] > volume_threshold) |
        (df['volume'] > df['volume'].quantile(0.95))
    )
    
    unusual = df[df['unusual']].sort_values('volume', ascending=False)
    
    return unusual[['expiration', 'strike', 'type', 'lastprice', 'bid', 'ask', 
                    'volume', 'openinterest', 'impliedvolatility', 'vol_oi_ratio']]


def suggest_strategies(current_price: float, chain_df: pd.DataFrame, 
                       outlook: str = 'neutral') -> List[Dict]:
    """Suggest options strategies based on outlook.
    
    Outlooks: bullish, bearish, neutral, volatile
    """
    strategies = []
    
    # Get nearest expiration with good liquidity
    exp_volume = chain_df.groupby('expiration')['volume'].sum()
    best_exp = exp_volume.idxmax() if not exp_volume.empty else None
    
    if best_exp is None:
        return strategies
    
    exp_chain = chain_df[chain_df['expiration'] == best_exp]
    calls = exp_chain[exp_chain['type'] == 'call']
    puts = exp_chain[exp_chain['type'] == 'put']
    
    # Find ATM strike
    atm_strike = calls.iloc[(calls['strike'] - current_price).abs().argmin()]['strike'] if len(calls) > 0 else None
    
    if atm_strike is None:
        return strategies
    
    if outlook == 'bullish':
        # Bull call spread
        long_strike = atm_strike
        short_strike = atm_strike * 1.05  # 5% OTM
        
        long_call = calls[calls['strike'] == long_strike]
        short_call = calls[calls['strike'] >= short_strike].head(1)
        
        if len(long_call) > 0 and len(short_call) > 0:
            strategies.append({
                'name': 'Bull Call Spread',
                'legs': [
                    {'action': 'BUY', 'type': 'call', 'strike': float(long_strike)},
                    {'action': 'SELL', 'type': 'call', 'strike': float(short_call.iloc[0]['strike'])}
                ],
                'expiration': best_exp,
                'max_profit': float(short_call.iloc[0]['strike'] - long_strike - 
                                   (long_call.iloc[0]['lastprice'] - short_call.iloc[0]['lastprice'])),
                'max_loss': float(long_call.iloc[0]['lastprice'] - short_call.iloc[0]['lastprice'])
            })
        
        # Long call
        strategies.append({
            'name': 'Long Call',
            'legs': [{'action': 'BUY', 'type': 'call', 'strike': float(atm_strike)}],
            'expiration': best_exp,
            'premium': float(long_call.iloc[0]['lastprice']) if len(long_call) > 0 else None
        })
    
    elif outlook == 'bearish':
        # Bear put spread
        long_strike = atm_strike
        short_strike = atm_strike * 0.95  # 5% OTM
        
        long_put = puts[puts['strike'] == long_strike]
        short_put = puts[puts['strike'] <= short_strike].tail(1)
        
        if len(long_put) > 0 and len(short_put) > 0:
            strategies.append({
                'name': 'Bear Put Spread',
                'legs': [
                    {'action': 'BUY', 'type': 'put', 'strike': float(long_strike)},
                    {'action': 'SELL', 'type': 'put', 'strike': float(short_put.iloc[0]['strike'])}
                ],
                'expiration': best_exp
            })
    
    elif outlook == 'neutral':
        # Iron condor
        strategies.append({
            'name': 'Iron Condor',
            'description': 'Profit from low volatility, range-bound movement',
            'legs': [
                {'action': 'SELL', 'type': 'put', 'strike': float(atm_strike * 0.95)},
                {'action': 'BUY', 'type': 'put', 'strike': float(atm_strike * 0.90)},
                {'action': 'SELL', 'type': 'call', 'strike': float(atm_strike * 1.05)},
                {'action': 'BUY', 'type': 'call', 'strike': float(atm_strike * 1.10)}
            ],
            'expiration': best_exp
        })
    
    elif outlook == 'volatile':
        # Long straddle
        atm_call = calls[calls['strike'] == atm_strike]
        atm_put = puts[puts['strike'] == atm_strike]
        
        if len(atm_call) > 0 and len(atm_put) > 0:
            strategies.append({
                'name': 'Long Straddle',
                'description': 'Profit from large move in either direction',
                'legs': [
                    {'action': 'BUY', 'type': 'call', 'strike': float(atm_strike)},
                    {'action': 'BUY', 'type': 'put', 'strike': float(atm_strike)}
                ],
                'expiration': best_exp,
                'total_premium': float(atm_call.iloc[0]['lastprice'] + atm_put.iloc[0]['lastprice']),
                'breakeven_up': float(atm_strike + atm_call.iloc[0]['lastprice'] + atm_put.iloc[0]['lastprice']),
                'breakeven_down': float(atm_strike - atm_call.iloc[0]['lastprice'] - atm_put.iloc[0]['lastprice'])
            })
    
    return strategies


def save_options_data(ticker: str, data: Dict, base_path: str = 'technicals/data'):
    """Save options data to files."""
    os.makedirs(base_path, exist_ok=True)
    
    # Save chain as parquet
    if 'chain' in data and isinstance(data['chain'], pd.DataFrame):
        chain = data['chain']
        chain.to_parquet(os.path.join(base_path, f'{ticker}_options_chain.parquet'), index=False)
        
        # Convert for JSON
        data_json = {k: v for k, v in data.items() if k != 'chain'}
        data_json['chain_summary'] = {
            'total_contracts': len(chain),
            'expirations': chain['expiration'].nunique(),
            'strikes': chain['strike'].nunique()
        }
    else:
        data_json = data
    
    # Save metadata as JSON
    with open(os.path.join(base_path, f'{ticker}_options_meta.json'), 'w') as f:
        json.dump(data_json, f, indent=2, default=str)
    
    print(f"Saved: {base_path}/{ticker}_options_chain.parquet")
    print(f"Saved: {base_path}/{ticker}_options_meta.json")


def main():
    parser = argparse.ArgumentParser(description='Fetch and analyze options chain data')
    parser.add_argument('--ticker', '-t', required=True, help='Stock ticker symbol')
    parser.add_argument('--outlook', '-o', default='neutral', 
                        choices=['bullish', 'bearish', 'neutral', 'volatile'],
                        help='Market outlook for strategy suggestions')
    args = parser.parse_args()
    
    print(f"Fetching options data for {args.ticker}...")
    
    data = fetch_options_chain(args.ticker)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
        return
    
    chain = data['chain']
    current_price = data['current_price']
    
    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"Expirations: {len(data['expirations'])}")
    print(f"Total Contracts: {len(chain)}")
    
    # Greeks summary
    print("\n--- Greeks & Flow Summary ---")
    summary = compute_greeks_summary(chain, current_price)
    for k, v in summary.items():
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Unusual activity
    print("\n--- Unusual Activity (Top 10) ---")
    unusual = detect_unusual_activity(chain)
    if len(unusual) > 0:
        print(unusual.head(10).to_string())
    else:
        print("No unusual activity detected")
    
    # Strategy suggestions
    print(f"\n--- Strategy Suggestions ({args.outlook}) ---")
    strategies = suggest_strategies(current_price, chain, args.outlook)
    for strat in strategies:
        print(f"\n{strat['name']}:")
        if 'description' in strat:
            print(f"  {strat['description']}")
        print(f"  Expiration: {strat.get('expiration')}")
        if 'legs' in strat:
            for leg in strat['legs']:
                print(f"    {leg['action']} {leg['type'].upper()} @ {leg['strike']:.2f}")
    
    # Save data
    save_options_data(args.ticker, data)


if __name__ == '__main__':
    main()
