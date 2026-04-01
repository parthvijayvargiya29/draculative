# TradingView Real-Time Integration - Quick Start

You now have a complete trading system that integrates with **TradingView Plus** for real-time data streaming and automated trading execution.

## 🎯 What You Have

### New Files Created:
```
trading_system/
├── tradingview_webhook.py          # Webhook server receives TradingView alerts
├── tradingview_integration.py      # Adapts TradingView data to trading system
├── main_tradingview.py             # Enhanced main with TradingView support
└── requirements.txt                # Updated with flask, requests
```

### New Documentation:
```
TRADINGVIEW_SETUP.md                # Complete setup guide with examples
```

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
cd trading_system
pip3 install flask requests aiohttp websockets
```

### Step 2: Start Webhook Server
```bash
python3 main_tradingview.py --tradingview --webhook-port 5000 --mode paper
```

Output:
```
🟢 Starting PAPER Trading with TradingView
   Data Source: TradingView Webhooks
   Webhook Port: 5000
   Webhook URL: http://localhost:5000/webhook/tradingview
```

### Step 3: Configure TradingView Alert

In TradingView:

1. **Create Alert** on any chart
2. **Set Webhook URL:**
   ```
   http://YOUR_IP:5000/webhook/tradingview
   ```
3. **Alert Message (copy exactly):**
   ```json
   {
     "symbol": "{{ticker}}",
     "close": {{close}},
     "high": {{high}},
     "low": {{low}},
     "open": {{open}},
     "volume": {{volume}},
     "time": "{{time}}",
     "interval": "{{interval}}"
   }
   ```

### Step 4: Test Connection

In another terminal:
```bash
# Check server status
curl http://localhost:5000/tradingview/status

# Send test alert
curl -X POST http://localhost:5000/webhook/tradingview \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","close":150.25,"high":151,"low":149.5,"open":149.75,"volume":1000000,"time":"2026-03-19","interval":"5min"}'
```

Response:
```json
{
  "status": "success",
  "symbol": "AAPL",
  "price": 150.25,
  "timestamp": 1711000000.0,
  "total_received": 1
}
```

## 📊 Architecture

```
TradingView
    ↓ (HTTP POST with JSON)
    ↓
Webhook Server (tradingview_webhook.py)
    ↓ (receives alert)
    ↓
TradingViewAlert (parsed)
    ↓
TradingViewDataAdapter (tradingview_integration.py)
    ↓ (converts to OHLCV)
    ↓
TradingSystem (main.py)
    ↓ (processes indicators)
    ↓
IndicatorEngine
    ↓ (calculates MACD, BB, Stoch)
    ↓
SignalAggregator (checks 3/3 convergence)
    ↓
TradingEngine (entry/exit logic)
    ↓
OrderGenerator (bracket orders)
    ↓
Broker Interface (paper/live)
    ↓
Trade Execution & Logging
```

## 🎮 Running Modes

### Mode 1: Paper Trading (Simulation)
```bash
python3 main_tradingview.py --mode paper --tradingview --webhook-port 5000
```
- Simulates trades without real money
- Perfect for testing your alerts and strategy
- Recommended: Run for 1-2 weeks before going live

### Mode 2: Paper + Local Testing
```bash
python3 main_tradingview.py --mode paper --tradingview --webhook-port 5000 --demo
```
- Sends test alerts automatically
- Verify system works without TradingView alerts
- Great for debugging

### Mode 3: Live Trading
```bash
python3 main_tradingview.py --mode live --tradingview --webhook-port 5000
```
- ⚠️ **WARNING:** Real money at risk
- Requires broker credentials (Alpaca, IB)
- Only after successful 2+ weeks of paper trading

## 🛠️ Configuration

### Webhook Server Options

```bash
# Custom port
python3 main_tradingview.py --webhook-port 8080 --tradingview

# Custom symbols to monitor
python3 main_tradingview.py --symbols AAPL GOOGL MSFT TSLA --tradingview

# With authentication token
python3 main_tradingview.py --webhook-token "your-secret-key" --tradingview

# Custom config file
python3 main_tradingview.py --config custom_config.yml --tradingview

# All together
python3 main_tradingview.py \
  --mode paper \
  --tradingview \
  --webhook-port 5000 \
  --webhook-token "secret123" \
  --symbols AAPL GOOGL MSFT \
  --config trading_system/config.yml
```

## 📈 Data Flow Example

### 1. You Set Alert in TradingView
```
Chart: AAPL 5min
Condition: close > SMA(20)
Webhook: http://your_ip:5000/webhook/tradingview
Message: {...JSON...}
```

### 2. When Alert Triggers
TradingView sends:
```json
POST /webhook/tradingview
{
  "symbol": "AAPL",
  "close": 150.25,
  "high": 151.00,
  "low": 149.50,
  "open": 149.75,
  "volume": 2500000,
  "time": "2026-03-19 14:30:00",
  "interval": "5min"
}
```

### 3. Server Receives & Processes
```
✓ Alert received: AAPL @ $150.25
✓ Converted to OHLCV
✓ Passed to IndicatorEngine
✓ MACD: +0.85 (bullish)
✓ Bollinger: +0.90 (above MA)
✓ Stochastic: +0.75 (overbought)
✓ Convergence: 3/3 = 100% ✓ ENTRY SIGNAL
✓ Order generated: BUY 10 shares
✓ Position opened: Entry $150.25
✓ Stop: $148.88 | TP1: $151.25 | TP2: $151.88 | TP3: $152.50
```

### 4. Exit Management
```
Monitoring position...
  [+0.50] Price: $150.75 → No action
  [+1.00] Price: $151.25 → Sell 50% (TP1 hit)
  [+1.50] Price: $151.88 → Sell 30% (TP2 hit)
  [+2.00] Price: $152.50 → Sell 20% (TP3 hit)
Trade closed: +$21.50 profit
```

## 📊 Monitoring Dashboard

### Check Server Status
```bash
curl http://localhost:5000/tradingview/status | python3 -m json.tool
```

Output:
```json
{
  "server_running": true,
  "port": 5000,
  "alerts_received": 42,
  "errors": 0,
  "symbols_tracked": 3,
  "symbols": {
    "AAPL": {
      "candle_count": 10,
      "latest_price": 150.25
    },
    "GOOGL": {
      "candle_count": 8,
      "latest_price": 140.50
    },
    "MSFT": {
      "candle_count": 6,
      "latest_price": 420.75
    }
  }
}
```

### Get Recent Candles
```bash
curl http://localhost:5000/tradingview/candles/AAPL?count=5
```

### View Trading System Logs
```bash
tail -f trading_system.log
```

Look for:
```
✓ Alert received: AAPL @ $150.25
✓ Indicator: MACD bullish
✓ Signal Convergence: 3/3 (100%)
✓ Entry signal: LONG
✓ Order submitted: BUY 10 @ $150.25
```

## 🔒 Security

### Option 1: Local Testing (ngrok)
```bash
# In another terminal
ngrok http 5000

# Copy ngrok URL (e.g., https://abc123.ngrok.io)
# Use in TradingView: https://abc123.ngrok.io/webhook/tradingview
```

### Option 2: Production (AWS EC2)
```bash
# Find your IP
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Use in TradingView: http://YOUR_IP:5000/webhook/tradingview
```

### Option 3: With Auth Token
```bash
# Start server with token
python3 main_tradingview.py --tradingview --webhook-token "your-secret-key"

# Send alert with token
curl -X POST http://localhost:5000/webhook/tradingview \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## 🚨 Troubleshooting

### Alerts Not Received?

1. **Check server running:**
   ```bash
   ps aux | grep main_tradingview
   ```

2. **Verify port open:**
   ```bash
   lsof -i :5000
   ```

3. **Test manually:**
   ```bash
   curl -X POST http://localhost:5000/webhook/tradingview \
     -H "Content-Type: application/json" \
     -d '{"symbol":"TEST","close":100,"high":101,"low":99,"open":100,"volume":1000000,"time":"2026-03-19","interval":"5min"}'
   ```

4. **Check firewall:**
   ```bash
   # On Mac
   sudo lsof -i :5000
   
   # On Linux
   sudo ufw allow 5000
   ```

### Connection Refused?

1. **Server not running - start it:**
   ```bash
   python3 main_tradingview.py --tradingview
   ```

2. **Wrong IP - find yours:**
   ```bash
   # Get local IP
   ifconfig | grep "inet "
   
   # Get public IP (if remote)
   curl ifconfig.me
   ```

3. **Wrong port - check config:**
   ```bash
   python3 main_tradingview.py --webhook-port 5000 --tradingview
   ```

### High Latency?

1. **Reduce symbols:**
   ```bash
   python3 main_tradingview.py --symbols AAPL --tradingview
   ```

2. **Increase buffer size:**
   ```python
   # In main_tradingview.py
   adapter = TradingViewDataAdapter(
       webhook_server=server,
       symbols=['AAPL'],
       max_latency_ms=2000.0  # Stricter latency
   )
   ```

## 📚 Complete Documentation

See **TRADINGVIEW_SETUP.md** for:
- Detailed step-by-step setup
- TradingView Pine Script examples
- Production deployment options (Docker, systemd, AWS)
- Advanced custom alert formats
- Full troubleshooting guide

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Read this file
2. ✅ Run: `pip3 install flask requests`
3. ✅ Start server: `python3 main_tradingview.py --tradingview --mode paper`
4. ✅ Create test alert in TradingView
5. ✅ Verify alert received

### This Week
1. Create production alert for your favorite stock
2. Paper trade for 2-3 days
3. Monitor logs for false signals
4. Adjust indicator settings if needed

### Next Week
1. Add more stocks (2-3 symbols)
2. Paper trade for 1-2 weeks
3. Monitor win rate, profit factor
4. Calculate realistic expected returns

### Before Going Live
1. ✅ 2+ weeks successful paper trading
2. ✅ Win rate > 45%
3. ✅ Positive expectancy
4. ✅ Fund account with broker
5. ✅ Test live with 1 share (penny scale)
6. ✅ Monitor for 3+ days
7. ✅ Scale up gradually

## 📞 Commands Cheat Sheet

```bash
# Install dependencies
pip3 install flask requests aiohttp websockets

# Start paper trading with TradingView
python3 main_tradingview.py --tradingview --mode paper

# Test with demo alerts
python3 main_tradingview.py --tradingview --mode paper --demo

# Check server status
curl http://localhost:5000/tradingview/status

# View logs
tail -f trading_system.log

# Backtest strategy
python3 backtest.py

# Get recent candles
curl http://localhost:5000/tradingview/candles/AAPL?count=20
```

## 💡 Example: Full Setup for AAPL Trading

### 1. Start Server
```bash
python3 main_tradingview.py --mode paper --tradingview --symbols AAPL
```

### 2. TradingView Alert

**Chart:** AAPL, 1h  
**Condition:** `close > ta.sma(close, 20)`  
**Webhook:** `http://YOUR_IP:5000/webhook/tradingview`  
**Message:**
```json
{"symbol":"{{ticker}}","close":{{close}},"high":{{high}},"low":{{low}},"open":{{open}},"volume":{{volume}},"time":"{{time}}","interval":"{{interval}}"}
```

### 3. Monitor Trades
```bash
tail -f trading_system.log

# Look for:
# ✓ Alert received: AAPL @ $XXX.XX
# ✓ Entry signal: LONG
# ✓ Order submitted
# ✓ Trade closed: +$XX profit
```

### 4. Check Results
```bash
curl http://localhost:5000/tradingview/status
```

---

**Status:** ✅ Production Ready  
**Created:** March 19, 2026  
**Version:** 1.0
