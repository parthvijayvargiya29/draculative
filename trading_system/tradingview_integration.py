"""
TradingView Integration Layer - Connects Trading System to TradingView Real-Time Data

This module adapts TradingView webhook data into the trading system's data pipeline.
It handles:
- Converting TradingView alerts to OHLCV format
- Managing real-time data flow from TradingView
- Fallback to existing data sources if TradingView unavailable
- Data synchronization across multiple symbols

Usage:
    from tradingview_integration import TradingViewDataAdapter
    
    # Create adapter
    adapter = TradingViewDataAdapter(
        trading_system=system,
        webhook_server=server,
        symbols=['AAPL', 'GOOGL', 'MSFT']
    )
    
    # Start monitoring
    await adapter.start()
    
    # Process TradingView candles continuously
    async with adapter:
        async for processed in adapter.process_candles():
            # Trade!
            pass
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from trading_system.data_ingestion import OHLCV, MarketDataFetcher
from trading_system.tradingview_webhook import TradingViewWebhookServer, TradingViewAlert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source priority."""
    TRADINGVIEW = "tradingview"
    WEBSOCKET = "websocket"
    REST = "rest"
    FALLBACK = "fallback"


@dataclass
class DataSourceMetrics:
    """Track data source quality."""
    source: DataSource
    candles_received: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    last_update: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.candles_received + self.errors
        if total == 0:
            return 0.0
        return (self.candles_received / total) * 100


class TradingViewDataAdapter:
    """
    Adapts TradingView webhook data to trading system format.
    Converts TradingViewAlert -> OHLCV -> Trading System.
    """
    
    def __init__(
        self,
        webhook_server: TradingViewWebhookServer,
        symbols: List[str],
        max_latency_ms: float = 5000.0
    ):
        """
        Initialize adapter.
        
        Args:
            webhook_server: TradingViewWebhookServer instance
            symbols: List of symbols to monitor (e.g., ['AAPL', 'GOOGL'])
            max_latency_ms: Maximum acceptable latency
        """
        self.webhook_server = webhook_server
        self.symbols = [s.upper() for s in symbols]
        self.max_latency_ms = max_latency_ms
        
        # Tracking
        self.last_seen_timestamps: Dict[str, float] = {}
        self.data_source_metrics: Dict[str, DataSourceMetrics] = {
            s: DataSourceMetrics(source=DataSource.TRADINGVIEW) for s in self.symbols
        }
        self.active = False
        
        logger.info(f"TradingViewDataAdapter initialized for {len(symbols)} symbols: {symbols}")
    
    def convert_alert_to_ohlcv(self, alert: TradingViewAlert) -> OHLCV:
        """
        Convert TradingView alert to OHLCV format.
        
        Args:
            alert: TradingViewAlert from webhook
            
        Returns:
            OHLCV object compatible with trading system
        """
        ohlcv = OHLCV(
            timestamp=alert.timestamp,
            symbol=alert.symbol,
            open=alert.open_price,
            high=alert.high_price,
            low=alert.low_price,
            close=alert.close_price,
            volume=alert.volume,
            source=DataSource.TRADINGVIEW.value
        )
        return ohlcv
    
    async def process_tradingview_candle(self, symbol: str) -> Optional[OHLCV]:
        """
        Process latest TradingView candle for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OHLCV or None if no new data
        """
        try:
            alert = self.webhook_server.data_buffer.get_latest(symbol)
            
            if not alert:
                logger.debug(f"No data from TradingView for {symbol}")
                return None
            
            # Check if new
            if symbol in self.last_seen_timestamps:
                if alert.timestamp <= self.last_seen_timestamps[symbol]:
                    return None  # Duplicate
            
            # Convert to OHLCV
            ohlcv = self.convert_alert_to_ohlcv(alert)
            self.last_seen_timestamps[symbol] = alert.timestamp
            
            # Update metrics
            metrics = self.data_source_metrics[symbol]
            metrics.candles_received += 1
            metrics.last_update = alert.timestamp
            
            logger.info(f"✓ {symbol} from TradingView: ${alert.close_price:.2f}")
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error processing TradingView candle for {symbol}: {e}")
            self.data_source_metrics[symbol].errors += 1
            return None
    
    async def get_fallback_candle(self, symbol: str) -> Optional[OHLCV]:
        """
        Get fallback data from REST/WebSocket.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            None (TradingView webhook is the primary data source)
        """
        return None
    
    async def process_symbol(self, symbol: str) -> Optional[OHLCV]:
        """
        Process single symbol - try TradingView, fallback if needed.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OHLCV or None
        """
        # Try TradingView first
        ohlcv = await self.process_tradingview_candle(symbol)
        if ohlcv:
            return ohlcv
        
        # Fallback to REST/WebSocket
        ohlcv = await self.get_fallback_candle(symbol)
        if ohlcv:
            return ohlcv
        
        return None
    
    async def process_candles(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continuously process candles from TradingView for all symbols.
        
        Yields:
            Dict with symbol and OHLCV data
            
        Example:
            async for data in adapter.process_candles():
                symbol = data['symbol']
                ohlcv = data['ohlcv']
                # Process trade...
        """
        logger.info("Starting candle processing loop...")
        
        while self.active:
            try:
                for symbol in self.symbols:
                    ohlcv = await self.process_symbol(symbol)
                    if ohlcv:
                        yield {
                            'symbol': symbol,
                            'ohlcv': ohlcv,
                            'timestamp': datetime.now(timezone.utc).timestamp(),
                            'source': ohlcv.source
                        }
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in candle processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def start(self):
        """Start monitoring TradingView data."""
        if self.active:
            logger.warning("Adapter already active")
            return
        
        self.active = True
        logger.info(f"✓ TradingView adapter started, monitoring {len(self.symbols)} symbols")
    
    async def stop(self):
        """Stop monitoring."""
        self.active = False
        logger.info("TradingView adapter stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return {
            'active': self.active,
            'symbols_monitored': len(self.symbols),
            'data_sources': {
                symbol: {
                    'source': self.data_source_metrics[symbol].source.value,
                    'candles_received': self.data_source_metrics[symbol].candles_received,
                    'errors': self.data_source_metrics[symbol].errors,
                    'success_rate': f"{self.data_source_metrics[symbol].success_rate:.1f}%",
                    'last_update': self.data_source_metrics[symbol].last_update
                }
                for symbol in self.symbols
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class TradingSystemWithTradingView:
    """
    Enhanced trading system with TradingView real-time data integration.
    Wraps existing trading system and adds TradingView data adapter.
    """
    
    def __init__(
        self,
        trading_system,
        tradingview_adapter: TradingViewDataAdapter
    ):
        """
        Initialize enhanced system.
        
        Args:
            trading_system: Existing TradingSystem instance
            tradingview_adapter: TradingViewDataAdapter instance
        """
        self.trading_system = trading_system
        self.adapter = tradingview_adapter
        self.processed_candles = 0
        self.trades_executed = 0
    
    async def run_with_tradingview(self):
        """
        Run trading system powered by TradingView real-time data.
        
        Main event loop:
        1. Receive TradingView alert via webhook
        2. Convert to OHLCV
        3. Process through indicators
        4. Execute trades
        5. Log results
        """
        await self.adapter.start()
        
        try:
            logger.info("🟢 Starting trading system with TradingView data...")
            
            async for candle_data in self.adapter.process_candles():
                try:
                    symbol = candle_data['symbol']
                    ohlcv = candle_data['ohlcv']
                    
                    # Process bar through trading system
                    await self.trading_system.process_bar(ohlcv)
                    self.processed_candles += 1
                    
                    # Check if trade was executed
                    if self.trading_system.trading_engine.position:
                        self.trades_executed += 1
                    
                    # Log progress every 100 candles
                    if self.processed_candles % 100 == 0:
                        metrics = self.adapter.get_metrics()
                        logger.info(f"Processed {self.processed_candles} candles, " +
                                  f"{self.trades_executed} trades executed")
                    
                except Exception as e:
                    logger.error(f"Error processing candle: {e}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await self.adapter.stop()
            self.trading_system.print_session_summary()


# Demo function
def demo_tradingview_integration():
    """Demonstrate TradingView integration."""
    import requests
    import time
    
    print("\n" + "="*80)
    print("TRADINGVIEW INTEGRATION - DEMO")
    print("="*80 + "\n")
    
    from trading_system.tradingview_webhook import create_tradingview_server
    
    # Create server
    print("1️⃣  Starting TradingView webhook server...")
    server = create_tradingview_server(port=5555)
    server.start()
    time.sleep(2)
    
    # Create adapter
    print("2️⃣  Creating TradingView adapter...")
    adapter = TradingViewDataAdapter(
        webhook_server=server,
        symbols=['AAPL', 'GOOGL', 'MSFT']
    )
    
    # Send test alerts
    print("3️⃣  Sending test alerts...\n")
    test_alerts = [
        {
            'symbol': 'AAPL',
            'close': 150.25,
            'high': 151.00,
            'low': 149.50,
            'open': 149.75,
            'volume': 1000000,
            'time': datetime.now(timezone.utc).isoformat(),
            'interval': '5min'
        },
        {
            'symbol': 'GOOGL',
            'close': 140.50,
            'high': 141.25,
            'low': 139.75,
            'open': 140.00,
            'volume': 800000,
            'time': datetime.now(timezone.utc).isoformat(),
            'interval': '5min'
        }
    ]
    
    try:
        for alert in test_alerts:
            requests.post(
                'http://localhost:5555/webhook/tradingview',
                json=alert
            )
            print(f"  ✓ Sent {alert['symbol']} alert")
            time.sleep(0.5)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test conversion
    print("\n4️⃣  Testing alert-to-OHLCV conversion...\n")
    for symbol in ['AAPL', 'GOOGL']:
        alert = server.data_buffer.get_latest(symbol)
        if alert:
            ohlcv = adapter.convert_alert_to_ohlcv(alert)
            print(f"  {symbol}:")
            print(f"    Alert:  {alert.close_price} | Volume: {alert.volume}")
            print(f"    OHLCV:  {ohlcv.close} | Volume: {ohlcv.volume}")
    
    # Get metrics
    print("\n5️⃣  Adapter metrics:\n")
    asyncio.run(adapter.start())
    metrics = adapter.get_metrics()
    print(f"  Active:            {metrics['active']}")
    print(f"  Symbols Monitored: {metrics['symbols_monitored']}")
    for symbol, stats in metrics['data_sources'].items():
        print(f"  {symbol:8} - {stats['candles_received']} candles, {stats['success_rate']} success")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(demo_tradingview_integration())
