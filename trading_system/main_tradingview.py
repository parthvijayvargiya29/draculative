"""
Enhanced Trading System Main - TradingView Real-Time Integration

This is the updated main.py that includes TradingView webhook support.
It can run in three modes:
1. Paper Trading (simulation, no broker)
2. Paper + TradingView (simulation with real TradingView data)
3. Live + TradingView (live trading with real TradingView data)

Usage:
    # Paper trading (existing data sources)
    python3 main.py --mode paper

    # Paper trading with TradingView
    python3 main.py --mode paper --tradingview --webhook-port 5000

    # Live trading with TradingView
    python3 main.py --mode live --tradingview --webhook-port 5000

    # Backtest
    python3 backtest.py
"""

import asyncio
import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import trading system modules
from trading_system.main import TradingSystem
from trading_system.config import load_config
from trading_system.tradingview_webhook import create_tradingview_server
from trading_system.tradingview_integration import (
    TradingViewDataAdapter,
    TradingSystemWithTradingView
)


class EnhancedTradingSystem:
    """Trading system with TradingView integration."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_tradingview: bool = False,
        webhook_port: int = 5000,
        webhook_auth_token: Optional[str] = None,
        symbols: Optional[list] = None,
        mode: str = 'paper'
    ):
        """
        Initialize enhanced trading system.
        
        Args:
            config_path: Path to config.yml
            use_tradingview: Enable TradingView webhook
            webhook_port: Port for webhook server
            webhook_auth_token: Optional auth token
            symbols: List of symbols to trade
            mode: 'paper' or 'live'
        """
        self.config_path = config_path or 'trading_system/config.yml'
        self.use_tradingview = use_tradingview
        self.webhook_port = webhook_port
        self.webhook_auth_token = webhook_auth_token
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT']
        self.mode = mode
        
        # Load configuration
        logger.info(f"Loading config from {self.config_path}")
        self.config = load_config(self.config_path)
        self.config['mode'] = mode
        
        # Create trading system
        self.trading_system = TradingSystem(self.config)
        
        # Setup TradingView if enabled
        self.webhook_server = None
        self.tradingview_adapter = None
        self.enhanced_system = None
        
        if use_tradingview:
            self._setup_tradingview()
    
    def _setup_tradingview(self):
        """Setup TradingView webhook integration."""
        logger.info("Setting up TradingView integration...")
        
        # Create webhook server
        self.webhook_server = create_tradingview_server(
            port=self.webhook_port,
            auth_token=self.webhook_auth_token
        )
        
        # Create adapter
        self.tradingview_adapter = TradingViewDataAdapter(
            webhook_server=self.webhook_server,
            symbols=self.symbols
        )
        
        # Create enhanced system
        self.enhanced_system = TradingSystemWithTradingView(
            trading_system=self.trading_system,
            tradingview_adapter=self.tradingview_adapter
        )
        
        logger.info(f"✓ TradingView integration ready on port {self.webhook_port}")
        logger.info(f"  Webhook URL: http://localhost:{self.webhook_port}/webhook/tradingview")
        logger.info(f"  Monitoring symbols: {', '.join(self.symbols)}")
    
    async def run_paper(self):
        """Run paper trading (simulation mode)."""
        logger.info("🟡 Starting Paper Trading Mode")
        logger.info(f"   Mode: {self.mode.upper()}")
        logger.info(f"   Account: Simulated")
        logger.info(f"   Risk: None (no real money)")
        
        try:
            await self.trading_system.run_live()
        except KeyboardInterrupt:
            logger.info("\nTrading stopped by user")
        finally:
            self.trading_system.print_session_summary()
    
    async def run_with_tradingview(self):
        """Run trading system with TradingView data."""
        if not self.enhanced_system:
            logger.error("TradingView not initialized")
            return
        
        logger.info(f"🟢 Starting {self.mode.upper()} Trading with TradingView")
        logger.info(f"   Data Source: TradingView Webhooks")
        logger.info(f"   Webhook Port: {self.webhook_port}")
        logger.info(f"   Symbols: {', '.join(self.symbols)}")
        logger.info(f"   Mode: {self.mode}")
        
        # Start webhook server
        self.webhook_server.start()
        await asyncio.sleep(1)  # Give server time to start
        
        try:
            await self.enhanced_system.run_with_tradingview()
        except KeyboardInterrupt:
            logger.info("\nTrading stopped by user")
        finally:
            logger.info("\nShutting down...")
            self.trading_system.print_session_summary()
    
    async def run(self):
        """Run the system (paper or live)."""
        if self.use_tradingview:
            await self.run_with_tradingview()
        else:
            await self.run_paper()
    
    def print_info(self):
        """Print system information."""
        print("\n" + "="*80)
        print("ENHANCED TRADING SYSTEM - TradingView Integration")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Mode:              {self.mode.upper()}")
        print(f"  Config File:       {self.config_path}")
        print(f"  TradingView:       {'✓ Enabled' if self.use_tradingview else '✗ Disabled'}")
        
        if self.use_tradingview:
            print(f"\nTradingView Setup:")
            print(f"  Webhook Port:      {self.webhook_port}")
            print(f"  Webhook URL:       http://localhost:{self.webhook_port}/webhook/tradingview")
            print(f"  Symbols:           {', '.join(self.symbols)}")
            print(f"  Auth Token:        {'✓ Enabled' if self.webhook_auth_token else '✗ Disabled'}")
        
        print(f"\nTrading Configuration:")
        print(f"  Initial Capital:   ${self.config.get('initial_capital', 10000):,.2f}")
        print(f"  Max Risk/Trade:    {self.config.get('max_risk_percent', 1.0):.2f}%")
        print(f"  Daily Loss Limit:  {self.config.get('daily_loss_limit_percent', 3.0):.2f}%")
        print(f"  Max Positions:     {self.config.get('max_concurrent_positions', 2)}")
        
        print(f"\nIndicators:")
        for indicator in ['macd', 'bollinger_bands', 'stochastic']:
            if indicator in self.config.get('indicators', {}):
                print(f"  {indicator:20} ✓")
        
        print("\n" + "="*80 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Trading System with TradingView Integration'
    )
    parser.add_argument(
        '--mode',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode: paper (simulation) or live (real money)'
    )
    parser.add_argument(
        '--config',
        default='trading_system/config.yml',
        help='Path to config.yml'
    )
    parser.add_argument(
        '--tradingview',
        action='store_true',
        help='Enable TradingView webhook integration'
    )
    parser.add_argument(
        '--webhook-port',
        type=int,
        default=5000,
        help='Webhook server port (default: 5000)'
    )
    parser.add_argument(
        '--webhook-token',
        default=None,
        help='Optional authentication token for webhook'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'GOOGL', 'MSFT'],
        help='Symbols to trade (default: AAPL GOOGL MSFT)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode (sends test alerts)'
    )
    
    args = parser.parse_args()
    
    # Check mode
    if args.mode == 'live':
        confirm = input("⚠️  LIVE TRADING MODE - Real money will be traded. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return
    
    # Create system
    system = EnhancedTradingSystem(
        config_path=args.config,
        use_tradingview=args.tradingview,
        webhook_port=args.webhook_port,
        webhook_auth_token=args.webhook_token,
        symbols=args.symbols,
        mode=args.mode
    )
    
    # Print info
    system.print_info()
    
    # Run demo if requested
    if args.demo and args.tradingview:
        logger.info("Running demo mode - sending test alerts...")
        system.webhook_server.start()
        await asyncio.sleep(2)
        
        # Send test alerts
        import requests
        from datetime import datetime, timezone
        
        test_alerts = [
            {
                'symbol': symbol,
                'close': 100 + i*10,
                'high': 101 + i*10,
                'low': 99 + i*10,
                'open': 100.5 + i*10,
                'volume': 1000000,
                'time': datetime.now(timezone.utc).isoformat(),
                'interval': '5min'
            }
            for i, symbol in enumerate(args.symbols)
        ]
        
        for alert in test_alerts:
            try:
                requests.post(
                    f'http://localhost:{args.webhook_port}/webhook/tradingview',
                    json=alert
                )
                logger.info(f"✓ Test alert sent: {alert['symbol']}")
            except Exception as e:
                logger.error(f"Failed to send test alert: {e}")
        
        logger.info("Demo complete. Use Ctrl+C to exit.")
    
    # Run system
    try:
        await system.run()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())
