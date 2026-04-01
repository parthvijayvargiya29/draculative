"""Historical Backtester - Test trading strategies on historical data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Dict

from indicators import IndicatorEngine
from trade_logic import SignalAggregator, Signal, SignalType, TradingEngine
from monitoring import TradeLogger, PerformanceMetrics


class HistoricalBacktester:
    """Backtest trading strategy on historical OHLCV data."""
    
    def __init__(self, symbol: str = 'NVDA', initial_capital: float = 10000):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.data: pd.DataFrame = None
        
        self.indicator_engine = IndicatorEngine()
        self.signal_aggregator = SignalAggregator(convergence_threshold=3, total_indicators=3)
        self.trading_engine = TradingEngine(initial_capital=initial_capital, risk_per_trade=0.01)
        self.metrics = PerformanceMetrics(initial_capital=initial_capital)
        
        self.trades: List[Dict] = []
        self.bar_count = 0
    
    def load_data(self, data_source: str = 'yfinance', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load historical data for backtesting."""
        try:
            import yfinance as yf
        except ImportError:
            print("Install yfinance: pip install yfinance")
            return None
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📥 Loading {self.symbol} data ({start_date} to {end_date})...")
        
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start_date, end=end_date, interval='15m')  # 15-min bars
        
        if df.empty:
            print(f"⚠️ No data found for {self.symbol}")
            return None
        
        df = df.reset_index()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.data = df
        print(f"✓ Loaded {len(df)} bars ({df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']})")
        
        return df
    
    def backtest(self) -> Dict:
        """Run backtest on loaded data."""
        if self.data is None or self.data.empty:
            print("⚠️ No data loaded. Call load_data() first.")
            return None
        
        print(f"\n📊 Backtesting {self.symbol} ({len(self.data)} bars)...")
        
        for idx in range(26, len(self.data)):  # Start at 26 for MACD
            bar = self.data.iloc[idx]
            
            # Get historical data up to this point
            hist = self.data.iloc[:idx+1]
            close = hist['close'].values
            high = hist['high'].values
            low = hist['low'].values
            
            # Calculate indicators
            indicators = self.indicator_engine.calculate_all(close, high, low)
            scores = self.indicator_engine.get_signal_strengths(indicators)
            
            # Aggregate signals
            for indicator_name, score in scores.items():
                signal_type = SignalType.BUY if score > 0 else SignalType.SELL if score < 0 else SignalType.NEUTRAL
                signal = Signal(source=indicator_name, type=signal_type, strength=abs(score))
                self.signal_aggregator.add_signal(signal)
            
            agg_signal = self.signal_aggregator.aggregate()
            current_price = bar['close']
            
            # Entry logic
            if self.trading_engine.state.value == 'idle' and agg_signal.should_enter_long:
                position = self.trading_engine.enter_long(current_price, agg_signal)
            
            # Exit logic
            if self.trading_engine.position:
                exit_reason = self.trading_engine.should_exit_position(current_price)
                if exit_reason:
                    trade = self.trading_engine.exit_long(current_price, exit_reason)
                    if trade:
                        self.trades.append(trade)
                        self.metrics.add_trade(trade)
            
            self.bar_count = idx
            
            # Progress
            if (idx - 26) % 100 == 0 and idx > 26:
                metrics = self.metrics.calculate(self.trading_engine.current_capital)
                print(f"  Bar {idx}: {len(self.trades)} trades, WR: {metrics['win_rate_pct']:.1f}%")
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """Get backtest results."""
        metrics = self.metrics.calculate(self.trading_engine.current_capital)
        
        return {
            'symbol': self.symbol,
            'bars_tested': self.bar_count,
            'period': f"{self.data.iloc[0]['timestamp']} to {self.data.iloc[-1]['timestamp']}" if self.data is not None else "N/A",
            'metrics': metrics,
            'trades': self.trades
        }
    
    def print_results(self):
        """Print backtest results in readable format."""
        results = self.get_results()
        metrics = results['metrics']
        
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS - {results['symbol']}")
        print(f"{'='*70}")
        print(f"Period:              {results['period']}")
        print(f"Bars Tested:         {results['bars_tested']}")
        print(f"\nTRADES:")
        print(f"  Total:             {metrics['total_trades']}")
        print(f"  Wins:              {metrics['wins']} ({metrics['win_rate_pct']:.1f}%)")
        print(f"  Losses:            {metrics['losses']}")
        print(f"\nPROFITABILITY:")
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}x")
        print(f"  Gross Profit:      ${metrics['gross_profit']:,.2f}")
        print(f"  Gross Loss:        ${metrics['gross_loss']:,.2f}")
        print(f"  Total P&L:         ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return:      {metrics['total_pnl_pct']:.2f}%")
        print(f"\nRISK METRICS:")
        print(f"  Avg Win:           ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss:          ${metrics['avg_loss']:,.2f}")
        print(f"  Expectancy:        ${metrics['expectancy_per_trade']:,.2f}")
        print(f"  Max Drawdown:      ${metrics['max_drawdown']:,.2f}")
        print(f"  Max DD %:          {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"\nCAPITAL:")
        print(f"  Initial:           ${self.initial_capital:,.2f}")
        print(f"  Final:             ${metrics['current_capital']:,.2f}")
        print(f"  Avg Trade:         ${metrics['avg_win'] + metrics['avg_loss']:,.2f}")
        print(f"  Avg Hold:          {metrics['avg_hold_minutes']:.1f} minutes")
        print(f"{'='*70}\n")


async def run_backtest_demo():
    """Run demo backtest."""
    backtest = HistoricalBacktester(symbol='NVDA', initial_capital=10000)
    
    # Load data (last 6 months)
    backtest.load_data(
        start_date=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Run backtest
    backtest.backtest()
    backtest.print_results()


if __name__ == '__main__':
    import asyncio
    asyncio.run(run_backtest_demo())
