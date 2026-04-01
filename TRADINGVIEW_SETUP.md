# TradingView Integration Guide

Complete setup instructions for connecting your TradingView account to the trading system for real-time data streaming and automated trading.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [TradingView Alert Setup](#tradingview-alert-setup)
3. [Webhook Server Configuration](#webhook-server-configuration)
4. [Testing the Integration](#testing-the-integration)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Requirements

- **TradingView Account:** Plus plan or higher (required for webhooks)
- **Python 3.9+** with the trading system installed
- **Public IP or ngrok tunnel:** For TradingView to reach your webhook server
- **Firewall:** Port 5000 (or custom) open for inbound connections

### Verify Installation

```bash
cd trading_system
python3 -c "from tradingview_webhook import create_tradingview_server; print('✓ TradingView module ready')"
```

---

## TradingView Alert Setup

### Step 1: Create TradingView Alert

1. Go to **TradingView.com** → Log in to your account
2. Open any chart (e.g., AAPL, 5-minute timeframe)
3. Click **Alerts** icon (bell) → **Create Alert**

### Step 2: Configure Alert

**Alert Conditions:**
```
Symbol: Any (e.g., AAPL)
Timeframe: 5min (recommended), 1h, 15min, etc.
Trigger: Custom > (your indicator/strategy)
```

**Example Trigger:**
```pine
close > open  // Simple: Buy when close > open
// OR
close > ta.sma(close, 20)  // SMA crossover
// OR
condition = 1  // Always trigger (sends every candle)
```

### Step 3: Webhook Configuration

In the alert dialog, set **Webhook URL**:

```
http://YOUR_IP:5000/webhook/tradingview
```

**For Testing (Local):** Use ngrok to tunnel:
```bash
ngrok http 5000
# Then use: https://YOUR_NGROK_ID.ngrok.io/webhook/tradingview
```

### Step 4: Alert Message (CRITICAL)

In the **Message** field, paste this JSON payload:

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

**TradingView Variables Explained:**
| Variable | Meaning | Example |
|----------|---------|---------|
| `{{ticker}}` | Stock symbol | AAPL |
| `{{close}}` | Close price | 150.25 |
| `{{high}}` | High price | 151.00 |
| `{{low}}` | Low price | 149.50 |
| `{{open}}` | Open price | 149.75 |
| `{{volume}}` | Volume | 1000000 |
| `{{time}}` | Candle time | 2026-03-19 14:30 |
| `{{interval}}` | Timeframe | 5min, 1h, etc. |

### Step 5: Enable and Test

- Check **Show notification**
- Set **Frequency** to: "On every alert trigger"
- Click **Create Alert**

---

## Webhook Server Configuration

### Start Server Locally

```bash
cd trading_system

# Basic start
python3 -c "
from tradingview_webhook import create_tradingview_server
server = create_tradingview_server(port=5000)
server.start()
import time; time.sleep(9999)
"

# OR with auth token
python3 -c "
from tradingview_webhook import create_tradingview_server
server = create_tradingview_server(port=5000, auth_token='your-secret-token')
server.start()
import time; time.sleep(9999)
"
```

### Server Endpoints

**Receive Alerts:**
```
POST http://your_ip:5000/webhook/tradingview
Content-Type: application/json

{
  "symbol": "AAPL",
  "close": 150.25,
  ...
}
```

**Check Status:**
```bash
curl http://localhost:5000/tradingview/status
```

**Response:**
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
    }
  }
}
```

**Get Recent Candles:**
```bash
curl http://localhost:5000/tradingview/candles/AAPL?count=20
```

**Test Webhook:**
```bash
curl -X POST http://localhost:5000/tradingview/test
```

---

## Testing the Integration

### Test 1: Local Server

```bash
python3 trading_system/tradingview_webhook.py
```

Output should show:
```
✓ Webhook server started on port 5000
✓ Alert received: TEST @ $100.50
```

### Test 2: Send Alert from TradingView

1. Go to your alert settings
2. Find the alert you created
3. Click the menu (three dots) → **Test Alert**

Watch your server output:
```
✓ Alert received: AAPL @ $150.25
```

### Test 3: Verify Data Reception

```bash
curl http://localhost:5000/tradingview/status | python3 -m json.tool
```

Check `alerts_received` counter is increasing.

### Test 4: Full Integration Demo

```bash
python3 trading_system/tradingview_integration.py
```

---

## Production Deployment

### Option 1: AWS EC2 (Recommended)

**Setup:**
```bash
# 1. Launch EC2 instance (t3.micro, Ubuntu 22.04)
# 2. SSH into instance
# 3. Install Python
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 4. Clone repository
git clone https://github.com/parthvijayvargiya29/draculative.git
cd draculative/trading_system

# 5. Setup environment
pip3 install -r requirements.txt
pip3 install flask aiohttp websockets

# 6. Start server
python3 -c "
from tradingview_webhook import create_tradingview_server
from tradingview_integration import TradingSystemWithTradingView
import asyncio

server = create_tradingview_server(port=5000)
server.start()

# Your trading system setup here...
print('✓ Server running on EC2')

import time
time.sleep(9999)
"
```

**Security Group Rules:**
- Inbound: Port 5000 from 0.0.0.0/0 (TradingView IPs)
- Outbound: All traffic

**Find Your Public IP:**
```bash
curl http://169.254.169.254/latest/meta-data/public-ipv4
# Then use in TradingView: http://YOUR_EC2_IP:5000/webhook/tradingview
```

### Option 2: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install flask aiohttp websockets

COPY trading_system/ .

EXPOSE 5000

CMD ["python3", "tradingview_webhook.py"]
```

Build and run:
```bash
docker build -t trading-system .
docker run -p 5000:5000 trading-system
```

### Option 3: Systemd Service (Linux)

Create `/etc/systemd/system/tradingview-webhook.service`:
```ini
[Unit]
Description=TradingView Trading System Webhook
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/draculative/trading_system
ExecStart=/usr/bin/python3 -c "from tradingview_webhook import create_tradingview_server; server = create_tradingview_server(port=5000); server.start(); import time; time.sleep(999999)"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable tradingview-webhook
sudo systemctl start tradingview-webhook
sudo systemctl status tradingview-webhook
```

---

## Troubleshooting

### Issue 1: "Connection refused" when TradingView tries to send alert

**Causes:**
- Server not running
- Wrong IP/port in TradingView alert
- Firewall blocking port 5000

**Fix:**
```bash
# 1. Verify server running
ps aux | grep tradingview

# 2. Check port listening
lsof -i :5000

# 3. Test locally
curl http://localhost:5000/tradingview/status

# 4. If remote, verify IP
curl http://YOUR_IP:5000/tradingview/status
```

### Issue 2: Alerts not being received

**Causes:**
- JSON format incorrect in TradingView alert message
- Network latency too high
- Alert condition not triggering

**Fix:**
```bash
# Test webhook endpoint
curl -X POST http://localhost:5000/webhook/tradingview \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "close": 150.25,
    "high": 151.00,
    "low": 149.50,
    "open": 149.75,
    "volume": 1000000,
    "time": "2026-03-19 14:30:00",
    "interval": "5min"
  }'

# Should return: {"status": "success", "symbol": "AAPL", ...}
```

### Issue 3: High latency / slow data reception

**Causes:**
- Network bandwidth limited
- Server overloaded
- Too many simultaneous alerts

**Fix:**
```python
# Reduce processing overhead
from tradingview_integration import TradingViewDataAdapter

adapter = TradingViewDataAdapter(
    webhook_server=server,
    symbols=['AAPL', 'GOOGL'],  # Limit symbols
    max_latency_ms=1000.0  # Reject old data
)
```

### Issue 4: "Not triggering" on specific condition

**TradingView Pine Script Example:**

Wrong:
```pine
// This will never trigger (condition never true)
alert_condition = close > 1000000
```

Right:
```pine
// This triggers every time candle closes
alert_condition = close > open

// Or your specific condition
alert_condition = close > ta.sma(close, 20) and volume > ta.sma(volume, 20)
```

Test:
```
Go to the chart
Wait for condition to trigger
You should see "Alert created" notification
```

---

## Advanced: Custom Alert Format

If you need different data, modify TradingView message:

```json
{
  "symbol": "{{ticker}}",
  "close": {{close}},
  "high": {{high}},
  "low": {{low}},
  "open": {{open}},
  "volume": {{volume}},
  "time": "{{time}}",
  "interval": "{{interval}}",
  "rsi": {{ta.rsi(close, 14)}},
  "macd": {{ta.macd(close, 12, 26, 9)[0]}},
  "custom_signal": "{{strategy.order.action}}"
}
```

Then update `tradingview_webhook.py` to parse extra fields:

```python
def _parse_alert(self, data: Dict[str, Any]) -> Optional[TradingViewAlert]:
    alert = TradingViewAlert(...)
    
    # Custom fields
    alert.rsi = data.get('rsi')
    alert.macd = data.get('macd')
    alert.custom_signal = data.get('custom_signal')
    
    return alert
```

---

## Real-World Example: Complete Setup

### TradingView Alert Setup (Copy-Paste Ready)

**Chart:** AAPL, 5min

**Alert Name:** "AAPL 5M Trading System"

**Condition:**
```pine
// Enter when price crosses above SMA
close > ta.sma(close, 20) and close[1] <= ta.sma(close[1], 20)
```

**Webhook URL:**
```
http://YOUR_EC2_IP:5000/webhook/tradingview
```

**Alert Message:**
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

### Python Server Setup

```python
from trading_system.tradingview_webhook import create_tradingview_server
from trading_system.tradingview_integration import TradingViewDataAdapter
from trading_system.main import TradingSystem
import asyncio

# 1. Start webhook server
server = create_tradingview_server(port=5000)
server.start()

# 2. Create trading system
system = TradingSystem(config={...})

# 3. Create adapter
adapter = TradingViewDataAdapter(
    webhook_server=server,
    symbols=['AAPL'],
)

# 4. Run
async def main():
    async with adapter:
        async for candle_data in adapter.process_candles():
            ohlcv = candle_data['ohlcv']
            await system.process_bar(ohlcv)

asyncio.run(main())
```

---

## Support & Resources

- **TradingView Docs:** https://www.tradingview.com/pine-script-docs/
- **Webhook Testing:** https://webhook.site/ (receive test alerts)
- **Python Asyncio:** https://docs.python.org/3/library/asyncio.html

---

**Last Updated:** March 2026  
**Status:** Production Ready ✅
