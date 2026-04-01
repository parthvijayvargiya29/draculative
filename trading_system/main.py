"""Main Trading System Orchestrator - Integrates all 5 layers."""

import asyncio
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
import numpy as np

from trading_system.data_ingestion import MarketDataFetcher, DataBuffer
from trading_system.indicators import IndicatorEngine
from trading_system.trade_logic import SignalAggregator, Signal, SignalType, TradingEngine
from trading_system.order_execution import BrokerInterface, RiskManager, OrderGenerator
from trading_system.monitoring import TradeLogger, PerformanceMetrics, SessionRecovery


class TradingSystem:
    """Complete real-time trading system (5 layers)."""
    
    def __init__(self, symbol: str = 'NVDA', config: dict = None):
        """
        Initialize trading system.
        
        Args:
            symbol: Stock ticker to trade
            config: Configuration dict with trading parameters
        """
        self.symbol = symbol
        
        # Default config
        self.config = {
            'initial_capital': 10000,
            'max_daily_loss_pct': 0.03,
            'max_risk_per_trade': 0.01,
            'convergence_threshold': 3,
            'broker_type': 'alpaca',
            'paper_trading': True,
            'log_dir': 'trading_system/data'
        }
        if config:
            self.config.update(config)
        
        # Layer 1: Data Ingestion
        self.data_fetcher = MarketDataFetcher(symbol, interval_seconds=900)
        
        # Layer 2: Indicators
        self.indicator_engine = IndicatorEngine()
        
        # Layer 3: Trade Logic
        self.signal_aggregator = SignalAggregator(
            convergence_threshold=self.config['convergence_threshold'],
            total_indicators=3
        )
        self.trading_engine = TradingEngine(
            initial_capital=self.config['initial_capital'],
            risk_per_trade=self.config['max_risk_per_trade']
        )
        
        # Layer 4: Order Execution
        self.broker = BrokerInterface(
            broker_type=self.config['broker_type'],
            paper_trading=self.config['paper_trading']
        )
        self.risk_manager = RiskManager(
            max_risk_per_trade=self.config['max_risk_per_trade'],
            max_daily_loss_pct=self.config['max_daily_loss_pct'],
            account_balance=self.config['initial_capital']
        )
        self.order_generator = OrderGenerator(self.broker, self.risk_manager)
        
        # Layer 5: Monitoring
        self.logger = TradeLogger(log_dir=self.config['log_dir'])
        self.metrics = PerformanceMetrics(initial_capital=self.config['initial_capital'])
        self.session_recovery = SessionRecovery(self.logger)
        
        # Runtime state
        self.is_running = False
        self.iteration = 0
    
    async def market_hours_check(self) -> bool:
        """Check if market is open (simplified - assumes 9:30-16:00 ET)."""
        now = datetime.now()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        return market_open <= now.time() < market_close and now.weekday() < 5
    
    async def process_bar(self):
        """Process one candle (15-min bar) of data."""
        buffer = self.data_fetcher.get_buffer()
        
        if len(buffer) < 26:  # Need min 26 candles for MACD
            return None
        
        # Get OHLC arrays
        df = buffer.to_dataframe()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Layer 2: Calculate indicators
        indicators = self.indicator_engine.calculate_all(close, high, low)
        scores = self.indicator_engine.get_signal_strengths(indicators)
        
        # Layer 3: Aggregate signals
        for indicator_name, score in scores.items():
            signal_type = SignalType.BUY if score > 0 else SignalType.SELL if score < 0 else SignalType.NEUTRAL
            signal = Signal(
                source=indicator_name,
                type=signal_type,
                strength=abs(score)
            )
            self.signal_aggregator.add_signal(signal)
        
        agg_signal = self.signal_aggregator.aggregate()
        current_price = close[-1]
        
        # Layer 3: Check entry conditions
        if self.trading_engine.state.value == 'idle' and agg_signal.should_enter_long:
            position = self.trading_engine.enter_long(current_price, agg_signal)
            if position:
                print(f"\n✅ ENTRY: Long {position.quantity} @ ${current_price:.2f}")
                print(f"   Stop: ${position.stop_loss:.2f}, TP1: ${position.take_profit_1:.2f}")
                print(f"   Signal: {agg_signal}")
                
                # Layer 4: Generate orders
                bracket = self.order_generator.generate_bracket_order(
                    self.symbol, current_price, position.quantity
                )
        
        # Layer 3: Check exit conditions
        if self.trading_engine.position:
            hold_minutes = (datetime.now() - self.trading_engine.position.entry_time).total_seconds() / 60
            exit_reason = self.trading_engine.should_exit_position(current_price)
            
            if exit_reason:
                trade = self.trading_engine.exit_long(current_price, exit_reason)
                if trade:
                    print(f"\n❌ EXIT: {exit_reason}")
                    print(f"   Exit Price: ${current_price:.2f}, PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
                    
                    # Layer 5: Log trade
                    self.logger.log_trade(trade)
                    self.metrics.add_trade(trade)
                    self.risk_manager.record_trade_pnl(trade['pnl'])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'iteration': self.iteration,
            'price': current_price,
            'signal': agg_signal,
            'indicators': {
                'macd': indicators['macd'],
                'bollinger': indicators['bollinger'],
                'stochastic': indicators['stochastic']
            },
            'position': self.trading_engine.position,
            'metrics': self.metrics.calculate(self.trading_engine.current_capital)
        }
    
    async def run_live(self, max_bars: int = None):
        """Run live trading system."""
        print(f"\n🚀 Starting Trading System - {self.symbol}")
        print(f"   Capital: ${self.config['initial_capital']:,.0f}")
        print(f"   Risk/Trade: {self.config['max_risk_per_trade']*100:.1f}%")
        print(f"   Daily Loss Limit: {self.config['max_daily_loss_pct']*100:.1f}%")
        print(f"   Mode: {'Paper Trading' if self.config['paper_trading'] else 'LIVE'}")
        
        self.is_running = True
        
        try:
            # Start data stream
            async def data_stream():
                await self.data_fetcher.stream_candles(max_iterations=max_bars)
            
            # Start processing bars
            async def bar_processor():
                while self.is_running and (max_bars is None or self.iteration < max_bars):
                    await asyncio.sleep(2)  # Wait for new candle
                    
                    result = await self.process_bar()
                    if result:
                        self.iteration += 1
                        if self.iteration % 5 == 0:
                            metrics = result['metrics']
                            print(f"\n📊 [Bar {self.iteration}] {result['price']:.2f} | "
                                  f"Trades: {metrics['total_trades']} | "
                                  f"WinRate: {metrics['win_rate_pct']:.1f}% | "
                                  f"PnL: ${metrics['total_pnl']:.2f}")
            
            # Run concurrently
            await asyncio.gather(data_stream(), bar_processor())
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Trading system interrupted by user")
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
        finally:
            self.is_running = False
            self.save_state()
            self.print_session_summary()
    
    def save_state(self):
        """Save system state before shutdown."""
        self.session_recovery.save_state(self.trading_engine, self.broker, self.risk_manager)
        print("✓ State saved")
    
    def print_session_summary(self):
        """Print trading session summary."""
        metrics = self.metrics.calculate(self.trading_engine.current_capital)
        
        print(f"\n{'='*60}")
        print(f"TRADING SESSION SUMMARY - {self.symbol}")
        print(f"{'='*60}")
        print(f"Total Trades:        {metrics['total_trades']}")
        print(f"Wins / Losses:       {metrics['wins']} / {metrics['losses']}")
        print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
        print(f"Profit Factor:       {metrics['profit_factor']:.2f}x")
        print(f"Total P&L:           ${metrics['total_pnl']:.2f} ({metrics['total_pnl_pct']:.2f}%)")
        print(f"Avg Trade:           ${metrics['avg_win']:.2f} / ${metrics['avg_loss']:.2f}")
        print(f"Max Drawdown:        ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"Final Capital:       ${metrics['current_capital']:,.2f}")
        print(f"{'='*60}\n")


async def main():
    """Run trading system demo."""
    # Create system
    system = TradingSystem(
        symbol='NVDA',
        config={
            'initial_capital': 10000,
            'max_risk_per_trade': 0.01,
            'convergence_threshold': 3,
            'paper_trading': True
        }
    )
    
    # Run with max 15 candles for demo
    await system.run_live(max_bars=15)


if __name__ == '__main__':
    asyncio.run(main())
