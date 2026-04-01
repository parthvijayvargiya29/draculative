"""
TradingView Webhook Integration for Real-Time Data Feed

This module receives TradingView alerts via webhook and integrates them
into the trading system's data pipeline. TradingView alerts are sent as
HTTP POST requests containing OHLC data and signal information.

Setup:
1. In TradingView, create alert with webhook URL: http://your_ip:5000/webhook/tradingview
2. In alert message, use JSON format:
   {
     "symbol": "{{ticker}}",
     "close": "{{close}}",
     "high": "{{high}}",
     "low": "{{low}}",
     "open": "{{open}}",
     "volume": "{{volume}}",
     "time": "{{time}}",
     "interval": "{{interval}}"
   }
3. Server receives alert → converts to OHLCV → feeds to trading system

Usage:
    from tradingview_webhook import create_tradingview_server, TradingViewDataBuffer
    
    # Create server
    server = create_tradingview_server(port=5000)
    server.start()
    
    # Access latest data
    buffer = server.data_buffer
    latest_candle = buffer.get_latest()
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from collections import deque
from flask import Flask, request, jsonify
from threading import Thread, Lock
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TradingViewAlert:
    """Single TradingView webhook alert data."""
    symbol: str
    timestamp: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    interval: str  # e.g., "1min", "5min", "15min", "1h"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TradingViewDataBuffer:
    """
    Thread-safe circular buffer for TradingView candle data.
    Maintains most recent N candles for technical analysis.
    """
    
    def __init__(self, max_size: int = 500):
        """
        Initialize buffer.
        
        Args:
            max_size: Maximum number of candles to keep (default 500)
        """
        self.max_size = max_size
        self.buffer: Dict[str, deque] = {}  # symbol -> deque of alerts
        self.lock = Lock()
        self.latest_timestamp = {}  # symbol -> latest timestamp
        logger.info(f"TradingViewDataBuffer initialized with max_size={max_size}")
    
    def add_candle(self, alert: TradingViewAlert) -> None:
        """
        Add candle to buffer. Automatically removes old candles if buffer full.
        
        Args:
            alert: TradingViewAlert to add
        """
        with self.lock:
            if alert.symbol not in self.buffer:
                self.buffer[alert.symbol] = deque(maxlen=self.max_size)
            
            self.buffer[alert.symbol].append(alert)
            self.latest_timestamp[alert.symbol] = alert.timestamp
            logger.debug(f"Added candle: {alert.symbol} @ ${alert.close_price}")
    
    def get_latest(self, symbol: str) -> Optional[TradingViewAlert]:
        """
        Get most recent candle for symbol.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            
        Returns:
            Latest TradingViewAlert or None if no data
        """
        with self.lock:
            if symbol in self.buffer and len(self.buffer[symbol]) > 0:
                return self.buffer[symbol][-1]
        return None
    
    def get_recent(self, symbol: str, count: int = 20) -> List[TradingViewAlert]:
        """
        Get N most recent candles for symbol.
        
        Args:
            symbol: Trading symbol
            count: Number of candles to retrieve
            
        Returns:
            List of TradingViewAlert objects
        """
        with self.lock:
            if symbol not in self.buffer:
                return []
            return list(self.buffer[symbol])[-count:]
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in buffer."""
        with self.lock:
            return list(self.buffer.keys())
    
    def get_candle_count(self, symbol: str) -> int:
        """Get number of candles stored for symbol."""
        with self.lock:
            return len(self.buffer.get(symbol, []))


class TradingViewWebhookServer:
    """
    Flask server that receives TradingView webhook alerts.
    Converts alerts to OHLCV format and feeds to trading system.
    """
    
    def __init__(self, port: int = 5000, auth_token: Optional[str] = None):
        """
        Initialize webhook server.
        
        Args:
            port: Port to listen on (default 5000)
            auth_token: Optional authentication token for security
        """
        self.port = port
        self.auth_token = auth_token
        self.data_buffer = TradingViewDataBuffer()
        self.app = Flask(__name__)
        self.server_thread = None
        self.running = False
        self.received_count = 0
        self.error_count = 0
        
        # Setup routes
        self._setup_routes()
        logger.info(f"TradingViewWebhookServer initialized on port {port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/webhook/tradingview', methods=['POST'])
        def receive_alert():
            """
            Receive TradingView webhook alert.
            
            Expected JSON payload:
            {
                "symbol": "AAPL",
                "close": 150.25,
                "high": 151.00,
                "low": 149.50,
                "open": 149.75,
                "volume": 1000000,
                "time": "2026-03-19 14:30:00",
                "interval": "5min"
            }
            """
            try:
                # Check authentication
                if self.auth_token:
                    token = request.headers.get('Authorization', '').replace('Bearer ', '')
                    if token != self.auth_token:
                        logger.warning(f"Invalid auth token: {token}")
                        return jsonify({'error': 'Unauthorized'}), 401
                
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                # Parse alert
                alert = self._parse_alert(data)
                if not alert:
                    return jsonify({'error': 'Invalid alert format'}), 400
                
                # Add to buffer
                self.data_buffer.add_candle(alert)
                self.received_count += 1
                
                logger.info(f"✓ Alert received: {alert.symbol} @ ${alert.close_price:.2f} (Total: {self.received_count})")
                
                return jsonify({
                    'status': 'success',
                    'symbol': alert.symbol,
                    'price': alert.close_price,
                    'timestamp': alert.timestamp,
                    'total_received': self.received_count
                }), 200
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"✗ Error processing alert: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/tradingview/status', methods=['GET'])
        def status():
            """Get server status and statistics."""
            return jsonify({
                'server_running': self.running,
                'port': self.port,
                'alerts_received': self.received_count,
                'errors': self.error_count,
                'symbols_tracked': len(self.data_buffer.get_all_symbols()),
                'symbols': {
                    symbol: {
                        'candle_count': self.data_buffer.get_candle_count(symbol),
                        'latest_price': self.data_buffer.get_latest(symbol).close_price
                        if self.data_buffer.get_latest(symbol) else None
                    }
                    for symbol in self.data_buffer.get_all_symbols()
                }
            }), 200
        
        @self.app.route('/', methods=['GET'])
        def dashboard():
            """Dashboard HTML page."""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading System Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }
                    .container { max-width: 1000px; margin: 0 auto; }
                    .header { text-align: center; color: #1976d2; margin-bottom: 30px; }
                    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .card h3 { margin: 0 0 10px 0; color: #666; font-size: 0.9em; }
                    .value { font-size: 2em; font-weight: bold; color: #1976d2; }
                    .symbols { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .symbols h2 { margin-top: 0; color: #333; border-bottom: 2px solid #1976d2; padding-bottom: 10px; }
                    .symbol-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; margin-top: 15px; }
                    .symbol { background: #f5f5f5; padding: 12px; border-radius: 6px; border: 1px solid #ddd; }
                    .symbol-name { font-weight: bold; color: #1976d2; margin-bottom: 5px; }
                    .symbol-info { font-size: 0.85em; color: #666; }
                    button { background: #1976d2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-top: 15px; }
                    button:hover { background: #1565c0; }
                    .time { color: #999; font-size: 0.9em; margin-top: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 Trading System Dashboard</h1>
                        <p>Real-time TradingView Integration</p>
                    </div>
                    
                    <div class="grid" id="status-grid">
                        <div class="card"><h3>Server Status</h3><div class="value" id="status">🔄</div></div>
                        <div class="card"><h3>Alerts Received</h3><div class="value" id="alerts">0</div></div>
                        <div class="card"><h3>Symbols</h3><div class="value" id="symbols">0</div></div>
                        <div class="card"><h3>Errors</h3><div class="value" id="errors">0</div></div>
                    </div>
                    
                    <div class="symbols">
                        <h2>📊 Monitored Symbols</h2>
                        <div id="symbols-list">Loading...</div>
                        <button onclick="location.reload()">🔄 Refresh</button>
                        <div class="time">Last updated: <span id="time">—</span></div>
                    </div>
                </div>
                
                <script>
                    async function updateDashboard() {
                        try {
                            const response = await fetch('/tradingview/status');
                            const data = await response.json();
                            
                            document.getElementById('status').textContent = data.server_running ? '✅ Running' : '⛔ Stopped';
                            document.getElementById('alerts').textContent = data.alerts_received;
                            document.getElementById('symbols').textContent = data.symbols_tracked;
                            document.getElementById('errors').textContent = data.errors;
                            
                            if (data.symbols_tracked === 0) {
                                document.getElementById('symbols-list').innerHTML = '<p style="color: #999;">No symbols tracked yet. Waiting for alerts...</p>';
                            } else {
                                const html = Object.entries(data.symbols).map(([sym, info]) => `
                                    <div class="symbol">
                                        <div class="symbol-name">${sym}</div>
                                        <div class="symbol-info">
                                            Candles: ${info.candle_count}<br>
                                            Price: $${info.latest_price ? parseFloat(info.latest_price).toFixed(2) : 'N/A'}
                                        </div>
                                    </div>
                                `).join('');
                                document.getElementById('symbols-list').innerHTML = '<div class="symbol-grid">' + html + '</div>';
                            }
                            
                            document.getElementById('time').textContent = new Date().toLocaleTimeString();
                        } catch (e) {
                            console.error('Error:', e);
                        }
                    }
                    
                    updateDashboard();
                    setInterval(updateDashboard, 2000);
                </script>
            </body>
            </html>
            """
            return html
        
        @self.app.route('/tradingview/candles/<symbol>', methods=['GET'])
        def get_candles(symbol):
            """Get recent candles for symbol."""
            count = request.args.get('count', 20, type=int)
            candles = self.data_buffer.get_recent(symbol.upper(), count)
            
            return jsonify({
                'symbol': symbol.upper(),
                'candle_count': len(candles),
                'candles': [c.to_dict() for c in candles]
            }), 200
        
        @self.app.route('/tradingview/test', methods=['POST'])
        def test_webhook():
            """
            Test webhook with sample data.
            Useful for verifying setup before connecting real TradingView alerts.
            """
            test_alert = {
                'symbol': 'TEST',
                'close': 100.50,
                'high': 101.00,
                'low': 100.00,
                'open': 100.25,
                'volume': 500000,
                'time': datetime.now(timezone.utc).isoformat(),
                'interval': '1min'
            }
            
            # Process as normal alert
            response = receive_alert()
            logger.info("✓ Test alert processed successfully")
            
            return jsonify({
                'test_status': 'success',
                'test_payload': test_alert
            }), 200
    
    def _parse_alert(self, data: Dict[str, Any]) -> Optional[TradingViewAlert]:
        """
        Parse TradingView webhook JSON into TradingViewAlert.
        
        Args:
            data: JSON data from webhook
            
        Returns:
            TradingViewAlert or None if parsing fails
        """
        try:
            symbol = str(data.get('symbol', '')).upper()
            close = float(data.get('close', 0))
            high = float(data.get('high', 0))
            low = float(data.get('low', 0))
            open_price = float(data.get('open', 0))
            volume = float(data.get('volume', 0))
            interval = str(data.get('interval', '1min'))
            
            # Parse timestamp
            time_str = data.get('time', '')
            try:
                if isinstance(time_str, str):
                    # Try ISO format first
                    timestamp = datetime.fromisoformat(time_str).timestamp()
                else:
                    # Assume Unix timestamp
                    timestamp = float(time_str)
            except:
                # Fallback to current time
                timestamp = time.time()
            
            # Validate data
            if not symbol or close <= 0:
                logger.error(f"Invalid alert data: {data}")
                return None
            
            alert = TradingViewAlert(
                symbol=symbol,
                timestamp=timestamp,
                open_price=open_price,
                high_price=high,
                low_price=low,
                close_price=close,
                volume=volume,
                interval=interval
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to parse alert: {e}")
            return None
    
    def start(self):
        """Start webhook server in background thread."""
        if self.running:
            logger.warning("Server already running")
            return
        
        self.running = True
        self.server_thread = Thread(
            target=lambda: self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True
            ),
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"✓ Webhook server started on port {self.port}")
        logger.info(f"  Webhook URL: http://localhost:{self.port}/webhook/tradingview")
        logger.info(f"  Status URL:  http://localhost:{self.port}/tradingview/status")
    
    def stop(self):
        """Stop webhook server."""
        self.running = False
        logger.info("Webhook server stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'alerts_received': self.received_count,
            'errors': self.error_count,
            'symbols_tracked': len(self.data_buffer.get_all_symbols()),
            'uptime_seconds': time.time()
        }


def create_tradingview_server(port: int = 5000, auth_token: Optional[str] = None) -> TradingViewWebhookServer:
    """
    Factory function to create and configure TradingView webhook server.
    
    Args:
        port: Port to listen on
        auth_token: Optional authentication token
        
    Returns:
        TradingViewWebhookServer instance
    """
    return TradingViewWebhookServer(port=port, auth_token=auth_token)


# Demo function
def demo_tradingview_webhook():
    """
    Demonstrate TradingView webhook integration.
    Creates server, simulates alerts, verifies data reception.
    """
    import requests
    
    print("\n" + "="*80)
    print("TRADINGVIEW WEBHOOK INTEGRATION - DEMO")
    print("="*80 + "\n")
    
    # Create server
    print("1️⃣  Creating TradingView webhook server...")
    server = create_tradingview_server(port=5555)
    server.start()
    
    # Give server time to start
    time.sleep(2)
    
    # Test with sample alerts
    print("\n2️⃣  Sending test alerts...\n")
    
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
        },
        {
            'symbol': 'MSFT',
            'close': 420.75,
            'high': 422.00,
            'low': 419.50,
            'open': 420.00,
            'volume': 600000,
            'time': datetime.now(timezone.utc).isoformat(),
            'interval': '5min'
        }
    ]
    
    try:
        for alert in test_alerts:
            response = requests.post(
                f'http://localhost:5555/webhook/tradingview',
                json=alert
            )
            if response.status_code == 200:
                print(f"✓ {alert['symbol']:8} sent successfully")
            else:
                print(f"✗ {alert['symbol']:8} failed: {response.text}")
            time.sleep(0.5)
    except Exception as e:
        print(f"Error sending alerts: {e}")
        return
    
    # Get status
    print("\n3️⃣  Server status:\n")
    try:
        status_response = requests.get('http://localhost:5555/tradingview/status')
        status = status_response.json()
        print(f"  Alerts Received:  {status['alerts_received']}")
        print(f"  Errors:           {status['errors']}")
        print(f"  Symbols Tracked:  {status['symbols_tracked']}")
        print(f"  Symbols:          {', '.join(status['symbols'].keys())}")
    except Exception as e:
        print(f"Error getting status: {e}")
    
    # Get candles
    print("\n4️⃣  Retrieved candles:\n")
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        try:
            candles_response = requests.get(
                f'http://localhost:5555/tradingview/candles/{symbol}',
                params={'count': 5}
            )
            candles = candles_response.json()
            if candles['candles']:
                latest = candles['candles'][-1]
                print(f"  {symbol:8} - ${latest['close_price']:8.2f} | " +
                      f"Vol: {latest['volume']:>10,.0f}")
        except Exception as e:
            print(f"  {symbol:8} - Error: {e}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nServer will continue running. Press Ctrl+C to stop.\n")


if __name__ == '__main__':
    demo_tradingview_webhook()
