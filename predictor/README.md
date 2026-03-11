# Stock Predictor System

A comprehensive stock prediction system that combines three independent signal streams to generate trading recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       STOCK PREDICTOR                                │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   TECHNICAL  │  │ FUNDAMENTAL  │  │    NEWS      │              │
│  │   SIGNAL     │  │   SIGNAL     │  │   SIGNAL     │              │
│  │    (40%)     │  │    (30%)     │  │    (30%)     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                  │                  │                      │
│         └────────────┬─────┴─────────────────┘                      │
│                      │                                               │
│              ┌───────▼───────┐                                       │
│              │   COMBINED    │                                       │
│              │  PREDICTION   │                                       │
│              └───────────────┘                                       │
│                                                                      │
│  Output: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL              │
└─────────────────────────────────────────────────────────────────────┘
```

## Signal Components

### 1. Technical Signal (`realtime_data.py`)

Fetches live market data and computes:

| Category | Indicators | Weight |
|----------|------------|--------|
| **Trend** | SMA 20/50/200, Golden/Death Cross, ADX | 25% |
| **Momentum** | RSI, MACD, Stochastic, CCI | 30% |
| **Volatility** | Bollinger Bands, VWAP distance, ATR | 20% |
| **Volume** | Volume ratio, OBV trend | 15% |
| **Options** | Put/Call ratio, Max pain, IV skew, unusual activity | 10% |

**Data Sources:**
- OHLCV: yfinance real-time
- Options Chain: yfinance options data
- All indicators computed on-the-fly

### 2. Fundamental Signal

Analyzes company fundamentals:

| Category | Metrics | Weight |
|----------|---------|--------|
| **Value** | P/E, Forward P/E, PEG ratio | 30% |
| **Quality** | ROE, Profit margin, Debt/Equity | 30% |
| **Growth** | Revenue growth, Earnings growth | 25% |
| **Sentiment** | Analyst rating, Target price upside | 15% |

**Fair Value Calculation:**
- PEG-based fair value estimation
- Comparison to analyst targets

### 3. News Signal (`news_tracker.py`)

Tracks news and events:

| Category | Analysis | Weight |
|----------|----------|--------|
| **Sentiment** | Keyword-based sentiment scoring | 50% |
| **Events** | Historical event impact correlation | 35% |
| **Momentum** | News volume and intensity | 15% |

**Historical Event Database:**
- Fed rate decisions
- Earnings announcements
- Geopolitical events
- Economic data releases
- Sector-specific events

## Usage

```bash
# Single ticker prediction
python3 predictor/src/stock_predictor.py IONQ

# Multiple tickers
for ticker in IONQ NVDA AAPL; do
    python3 predictor/src/stock_predictor.py $ticker
done
```

## Output

### Console Output

```
======================================================================
STOCK PREDICTION: IONQ
======================================================================

🟡 RECOMMENDATION: HOLD
   Confidence: 57%
   Current Price: $34.14

📊 TECHNICAL SIGNAL: NEUTRAL (15% strength)
   Components: Trend=-0.40, Momentum=+0.00
              Volume=-0.10, Options=+0.10
   
💰 FUNDAMENTAL SIGNAL: NEUTRAL (14% strength)
   Components: Value=+0.00, Quality=-0.15
              Growth=+0.25, Sentiment=+0.35
   Fair Value: $65.29 (+91.2%)

📰 NEWS SIGNAL: NEUTRAL (0% strength)
   Sentiment: +0.00

🎯 PRICE TARGETS
   Bull Target: $40.08 (+17.4%)
   Bear Target: $28.2 (-17.4%)
   Stop Loss:   $29.68
   Risk/Reward: 1.33
```

### JSON Output

Predictions saved to `predictor/data/{TICKER}_prediction.json`:

```json
{
  "ticker": "IONQ",
  "timestamp": "2025-01-10T...",
  "direction": "HOLD",
  "confidence": 0.57,
  "technical": {
    "direction": "NEUTRAL",
    "strength": 0.15,
    "bullish_factors": [...],
    "bearish_factors": [...]
  },
  "fundamental": {...},
  "news": {...},
  "current_price": 34.14,
  "target_price_bull": 40.08,
  "target_price_bear": 28.2,
  "stop_loss": 29.68,
  "risk_reward_ratio": 1.33
}
```

## Latest Predictions

### IONQ - HOLD (57% confidence)

| Signal | Direction | Strength | Key Factor |
|--------|-----------|----------|------------|
| Technical | NEUTRAL | 15% | Price below all MAs |
| Fundamental | NEUTRAL | 14% | Strong revenue growth (+428%) |
| News | NEUTRAL | 0% | No significant events |

**Action:** Wait for clearer signals

### NVDA - BUY (58% confidence)

| Signal | Direction | Strength | Key Factor |
|--------|-----------|----------|------------|
| Technical | NEUTRAL | 0% | Golden Cross active |
| Fundamental | BULLISH | 63% | High ROE (101%), strong growth |
| News | NEUTRAL | 0% | No significant events |

**Action:** Consider scaling into position with stop at $173.48

## File Structure

```
predictor/
├── src/
│   ├── realtime_data.py    # Live data fetcher
│   │   └── RealTimeDataFetcher class
│   │       ├── get_ohlcv()
│   │       ├── get_options_data()
│   │       ├── get_technical_indicators()
│   │       └── get_live_fundamentals()
│   │
│   ├── news_tracker.py     # News analysis
│   │   └── NewsTracker class
│   │       ├── fetch_news()
│   │       ├── analyze_sentiment()
│   │       ├── detect_events()
│   │       └── predict_event_impact()
│   │
│   └── stock_predictor.py  # Main orchestrator
│       └── StockPredictor class
│           ├── generate_technical_signal()
│           ├── generate_fundamental_signal()
│           ├── generate_news_signal()
│           └── generate_prediction()
│
└── data/
    ├── IONQ_prediction.json
    └── NVDA_prediction.json
```

## Signal Weights

Default weights (configurable):

```python
weights = {
    'technical': 0.40,    # Short-term price action
    'fundamental': 0.30,  # Long-term value
    'news': 0.30          # Event-driven
}
```

Adjust for different strategies:
- Day trading: technical=0.6, fundamental=0.1, news=0.3
- Value investing: technical=0.2, fundamental=0.6, news=0.2
- Event-driven: technical=0.3, fundamental=0.2, news=0.5

## Dependencies

```
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
```

## Integration with Existing Analysis

This predictor builds on:
- `technicals/src/indicators.py` - Core indicator library
- `technicals/src/advanced_quant_analysis.py` - Institutional metrics
- `technicals/src/ml_feature_importance.py` - ML-validated features

Key ML findings incorporated:
- OBV weighted heavily (highest feature importance)
- RSI oversold = 65% win rate (used for momentum)
- ATR for volatility and price targets
