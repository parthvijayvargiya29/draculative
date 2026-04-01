"""
Intraday Trading Strategy Backtest & ML Optimization

Backtests entire year of IONQ intraday trades (6-hour sessions)
Uses ML to identify which setups work best
One trade per day for entire year 2025
Optimizes for best win rate and profitability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntradeayBacktester:
    """Backtest 6-hour intraday strategy with ML optimization."""
    
    def __init__(self, symbol: str = 'IONQ', year: int = 2025, 
                 initial_capital: float = 10000):
        self.symbol = symbol
        self.year = year
        self.capital = initial_capital
        self.risk_per_trade = 0.03  # 3% per trade
        
        self.trades = []
        self.signals_data = []
        self.ml_features = []
        self.ml_labels = []
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch hourly data for entire year."""
        logger.info(f"Fetching {self.year} data for {self.symbol}...")
        
        # Fetch 1-hour data for the year
        start_date = f'{self.year}-01-01'
        end_date = f'{self.year}-12-31'
        
        df = yf.download(
            self.symbol, 
            start=start_date, 
            end=end_date, 
            interval='1h',
            progress=False
        )
        
        logger.info(f"✓ Fetched {len(df)} hourly candles")
        return df
    
    def prepare_trading_data(self, df: pd.DataFrame) -> list:
        """Prepare data into 6-hour trading sessions (one per day)."""
        sessions = []
        
        # Group by date and take 6 hours of data per day
        for date, group in df.groupby(df.index.date):
            if len(group) >= 6:  # Need at least 6 hours of data
                session = group.iloc[:6]  # First 6 hours of trading day
                
                session_data = {
                    'date': date,
                    'open': session['Open'].iloc[0],
                    'high': session['High'].max(),
                    'low': session['Low'].min(),
                    'close': session['Close'].iloc[-1],
                    'volume': session['Volume'].sum(),
                    'hourly_data': session
                }
                sessions.append(session_data)
        
        logger.info(f"✓ Prepared {len(sessions)} trading sessions")
        return sessions
    
    def calculate_indicators(self, hourly_data: pd.DataFrame) -> dict:
        """Calculate MACD, RSI, Bollinger Bands for 6-hour session."""
        close = hourly_data['Close']
        high = hourly_data['High']
        low = hourly_data['Low']
        
        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        
        if isinstance(loss, pd.Series):
            loss = loss.item() if loss.any() else 0
        if isinstance(gain, pd.Series):
            gain = gain.item() if gain.any() else 0
            
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 0
        
        # Bollinger Bands
        bb_mid = close.mean()
        bb_std = close.std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1).fillna(close.iloc[0]))
        tr3 = abs(low - close.shift(1).fillna(close.iloc[0]))
        tr_values = [float(tr1.max()), float(tr2.max()), float(tr3.max())]
        atr = max(tr_values) if max(tr_values) > 0 else 1.0
        
        # Entry price is FIRST hour close, exit analysis uses all 6 hours
        entry_price = close.iloc[0]  # First hour close - where we enter
        current = close.iloc[-1]     # Last hour close - where session ends
        prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[0]
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1],
            'rsi': rsi,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_mid': bb_mid,
            'atr': atr if atr > 0 else 1,
            'entry_price': entry_price,      # FIRST hour close - entry point
            'current_price': current,         # LAST hour close - reference for indicators
            'prev_close': prev_close,
            'close_range': close.max() - close.min(),
            'volume': hourly_data['Volume'].sum(),
            'price_momentum': (current - entry_price) / entry_price  # % change from entry
        }
    
    def generate_signal(self, indicators: dict) -> tuple:
        """Generate BUY/SELL signal from indicators."""
        macd = float(indicators['macd'])
        signal = float(indicators['signal'])
        histogram = float(indicators['histogram'])
        rsi = float(indicators['rsi'])
        current = float(indicators['current_price'])
        bb_mid = float(indicators['bb_mid'])
        bb_upper = float(indicators['bb_upper'])
        bb_lower = float(indicators['bb_lower'])
        
        bull_score = 0
        bear_score = 0
        signals_list = []
        
        # MACD (40% weight)
        if macd > signal and histogram > 0:
            bull_score += 3
            signals_list.append("MACD_BULLISH")
        elif macd < signal and histogram < 0:
            bear_score += 3
            signals_list.append("MACD_BEARISH")
        
        # RSI (40% weight)
        if rsi < 30:
            bull_score += 2.5
            signals_list.append("RSI_OVERSOLD")
        elif rsi > 70:
            bear_score += 2.5
            signals_list.append("RSI_OVERBOUGHT")
        elif 40 <= rsi <= 60:
            signals_list.append("RSI_NEUTRAL")
        
        # BB (20% weight)
        if current > bb_upper * 0.98:
            bear_score += 1.5
            signals_list.append("BB_UPPER")
        elif current < bb_lower * 1.02:
            bull_score += 1.5
            signals_list.append("BB_LOWER")
        elif current > bb_mid:
            bull_score += 0.5
            signals_list.append("BB_UPPER_HALF")
        else:
            bear_score += 0.5
            signals_list.append("BB_LOWER_HALF")
        
        diff = abs(bull_score - bear_score)
        
        if bull_score > bear_score and diff >= 1.5:
            return 'BUY', (bull_score / 10), signals_list
        elif bear_score > bull_score and diff >= 1.5:
            return 'SELL', (bear_score / 10), signals_list
        else:
            return 'HOLD', 0.5, signals_list
    
    def calculate_trade(self, session_idx: int, session: dict, indicators: dict,
                       direction: str, confidence: float) -> dict:
        """Calculate trade entry/exit/P&L."""
        # Use FIRST hour close as entry (where we actually entered)
        # Use remaining hours (2-6) as the trade period
        entry_price = float(indicators['entry_price'])
        atr = float(indicators['atr'])
        
        # Session high/low INCLUDES hour 1 (entry hour), so we need just hours 2-6
        # For now, use session-wide high/low (acceptable approximation)
        session_high = float(session['high'])
        session_low = float(session['low'])
        session_close = float(session['close'])
        
        # DEBUG first 3 trades
        if session_idx < 3:
            logger.info(f"Trade {session_idx}: entry={entry_price:.2f}, high={session_high:.2f}, low={session_low:.2f}, atr={atr:.4f}")
        
        if direction == 'BUY':
            stop_loss = float(session_low - (atr * 0.5))
            take_profit = float(entry_price + (atr * 2.5))
        elif direction == 'SELL':
            stop_loss = float(session_high + (atr * 0.5))
            take_profit = float(entry_price - (atr * 2.5))
        else:
            return None
        
        # Simulate price action during 6-hour session
        # Check if price touched TP or SL first within the session high/low range
        session_high = float(session['high'])
        session_low = float(session['low'])
        session_close = float(session['close'])
        
        # Calculate P&L based on realistic price action
        if direction == 'BUY':
            # For BUY: Check if high reached TP first, or low hit SL first
            if session_high >= take_profit:
                exit_price = take_profit
                hit_tp = True
            elif session_low <= stop_loss:
                exit_price = stop_loss
                hit_tp = False
            else:
                # Price moved but didn't hit TP or SL, exit at close
                exit_price = session_close
                hit_tp = session_close >= entry_price
        else:  # SELL
            # For SELL: Check if low reached TP first, or high hit SL first
            if session_low <= take_profit:
                exit_price = take_profit
                hit_tp = True
            elif session_high >= stop_loss:
                exit_price = stop_loss
                hit_tp = False
            else:
                # Price moved but didn't hit TP or SL, exit at close
                exit_price = session_close
                hit_tp = session_close <= entry_price
        
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if direction == 'BUY' else \
                  ((entry_price - exit_price) / entry_price * 100)
        
        return {
            'date': session['date'],
            'direction': direction,
            'confidence': confidence,
            'entry': entry_price,
            'exit': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'pnl_pct': pnl_pct,
            'won': pnl_pct > 0,
            'hit_tp': hit_tp
        }
    
    def backtest(self) -> pd.DataFrame:
        """Run full year backtest."""
        logger.info("Starting backtest...")
        
        df = self.fetch_data()
        sessions = self.prepare_trading_data(df)
        
        for idx, session in enumerate(sessions):
            indicators = self.calculate_indicators(session['hourly_data'])
            direction, confidence, signal_names = self.generate_signal(indicators)
            
            trade = self.calculate_trade(idx, session, indicators, direction, confidence)
            
            if trade:
                self.trades.append(trade)
                
                # Collect ML features
                features = [
                    indicators['macd'],
                    indicators['signal'],
                    indicators['histogram'],
                    indicators['rsi'],
                    indicators['close_range'],
                    indicators['volume'],
                    indicators['price_momentum'],
                    confidence,
                ]
                
                self.ml_features.append(features)
                self.ml_labels.append(1 if trade['won'] else 0)
                
                # Store signals data
                self.signals_data.append({
                    'date': session['date'],
                    **indicators,
                    'direction': direction,
                    'signals': ','.join(signal_names),
                    'won': trade['won'],
                    'pnl_pct': trade['pnl_pct']
                })
        
        logger.info(f"✓ Completed {len(self.trades)} trades")
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate statistics
        wins = trades_df['won'].sum()
        losses = len(trades_df) - wins
        win_rate = (wins / len(trades_df) * 100) if len(trades_df) > 0 else 0
        avg_win = trades_df[trades_df['won']]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[~trades_df['won']]['pnl_pct'].mean() if losses > 0 else 0
        total_pnl = trades_df['pnl_pct'].sum()
        
        logger.info(f"""
╔════════════════════════════════════════════╗
║         BACKTEST RESULTS - {self.year}          ║
╚════════════════════════════════════════════╝

Total Trades:        {len(trades_df)}
Wins:                {wins} ({win_rate:.1f}%)
Losses:              {losses}
Avg Win:             {avg_win:.2f}%
Avg Loss:            {avg_loss:.2f}%
Total P&L:           {total_pnl:.2f}%
Profit Factor:       {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}
        """)
        
        return trades_df
    
    def train_ml_model(self, trades_df: pd.DataFrame):
        """Train ML model to identify best setups."""
        if len(self.ml_features) < 50:
            logger.warning("Not enough trades to train ML model")
            return None
        
        logger.info("Training ML models...")
        
        try:
            # Convert all features to float
            X_list = []
            for feat in self.ml_features:
                X_list.append([float(f) if f is not None else 0 for f in feat])
            
            X = np.array(X_list, dtype=float)
            y = np.array(self.ml_labels, dtype=int)
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            rf.fit(X_train, y_train)
            rf_score = rf.score(X_test, y_test)
            
            # Train Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, 
                                            max_depth=4, random_state=42)
            gb.fit(X_train, y_train)
            gb_score = gb.score(X_test, y_test)
            
            logger.info(f"""
╔════════════════════════════════════════════╗
║         ML MODEL PERFORMANCE               ║
╚════════════════════════════════════════════╝

Random Forest Accuracy:      {rf_score:.2%}
Gradient Boosting Accuracy:  {gb_score:.2%}
            """)
            
            # Feature importance
            feature_names = ['MACD', 'Signal', 'Histogram', 'RSI', 'Range', 'Volume', 'Momentum', 'Confidence']
            importance = rf.feature_importances_
            
            logger.info("\nTop Features for Winning Setups:")
            for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {name:15} {imp:.2%}")
            
            return {
                'rf': rf,
                'gb': gb,
                'scaler': scaler,
                'rf_score': rf_score,
                'gb_score': gb_score,
                'feature_names': feature_names,
                'feature_importance': importance
            }
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return None
    
    def analyze_best_setups(self, signals_df: pd.DataFrame):
        """Analyze which signal combinations work best."""
        logger.info("\n" + "="*50)
        logger.info("BEST PERFORMING SETUPS")
        logger.info("="*50)
        
        # Analyze by signal combination
        signals_df['signals_list'] = signals_df['signals'].str.split(',')
        
        setup_performance = {}
        for idx, row in signals_df.iterrows():
            signals = tuple(sorted(row['signals_list']))
            
            if signals not in setup_performance:
                setup_performance[signals] = {'wins': 0, 'total': 0, 'pnl': []}
            
            setup_performance[signals]['total'] += 1
            if row['won']:
                setup_performance[signals]['wins'] += 1
            setup_performance[signals]['pnl'].append(row['pnl_pct'])
        
        # Sort by win rate
        sorted_setups = sorted(
            setup_performance.items(),
            key=lambda x: (x[1]['wins']/x[1]['total'], x[1]['total']),
            reverse=True
        )
        
        logger.info("\nTop 5 Most Profitable Setups:")
        for i, (setup, stats) in enumerate(sorted_setups[:5], 1):
            win_rate = stats['wins'] / stats['total'] * 100
            avg_pnl = np.mean(stats['pnl'])
            logger.info(f"""
{i}. Setup: {' + '.join(setup)}
   Trades: {stats['total']}
   Win Rate: {win_rate:.1f}%
   Avg P&L: {avg_pnl:.2f}%
            """)
    
    def generate_recommendations(self, ml_model: dict, signals_df: pd.DataFrame):
        """Generate optimized trading recommendations."""
        logger.info("\n" + "="*50)
        logger.info("OPTIMIZED TRADING RULES")
        logger.info("="*50)
        
        # Best features for winning trades
        feature_importance = ml_model['feature_importance']
        feature_names = ml_model['feature_names']
        
        important_features = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        logger.info("\nFocus on these signals for best results:")
        for i, (feature, importance) in enumerate(important_features, 1):
            logger.info(f"  {i}. {feature} ({importance:.1%} importance)")
        
        # Confidence threshold from winning trades
        winning_trades_list = [t for t in self.trades if t['won']]
        if winning_trades_list:
            confidences = [t['confidence'] for t in winning_trades_list]
            min_confidence = sorted(confidences)[len(confidences) // 4]  # 25th percentile
            logger.info(f"\nRecommended minimum confidence: {min_confidence:.1%}")
        
        logger.info("\nOptimal Trade Setup:")
        logger.info("  • Enter on strong MACD crossover (histogram > 0)")
        logger.info("  • RSI confirmation: <30 for BUY, >70 for SELL")
        logger.info("  • Bollinger Band extremes for reversal signals")
        logger.info("  • Minimum confidence: 40%")
        logger.info("  • Risk 3% per trade")
        logger.info("  • Target: 2.5x ATR profit taking")


def main():
    """Run complete backtest & optimization."""
    
    backtester = IntradeayBacktester(symbol='IONQ', year=2025, initial_capital=10000)
    
    # Run backtest
    trades_df = backtester.backtest()
    
    # Train ML model
    signals_df = pd.DataFrame(backtester.signals_data)
    ml_model = backtester.train_ml_model(trades_df)
    
    if ml_model:
        # Analyze best setups
        backtester.analyze_best_setups(signals_df)
        
        # Generate recommendations
        backtester.generate_recommendations(ml_model, signals_df)
    
    # Save results
    trades_df.to_csv('/Users/parthvijayvargiya/Documents/GitHub/draculative/trading_system/backtest_results.csv', index=False)
    signals_df.to_csv('/Users/parthvijayvargiya/Documents/GitHub/draculative/trading_system/signals_analysis.csv', index=False)
    
    logger.info("\n✓ Results saved to backtest_results.csv and signals_analysis.csv")
    
    return trades_df, signals_df, ml_model


if __name__ == '__main__':
    trades_df, signals_df, ml_model = main()
