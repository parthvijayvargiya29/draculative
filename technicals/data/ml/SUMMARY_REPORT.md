# Technical Indicator Importance Analysis

## Executive Summary

This analysis backpropagates through 2 years of historical price data to identify
which technical indicators have the most predictive power for future price movements.

**Tickers Analyzed:** IONQ, NVDA

### Top 10 Most Important Indicators

| Rank | Indicator | Importance Score |
|------|-----------|------------------|
| 1 | `obv` | 0.0666 |
| 2 | `obv_sma_20` | 0.0357 |
| 3 | `ichimoku_senkou_b` | 0.0286 |
| 4 | `ichimoku_chikou` | 0.0284 |
| 5 | `sma_100` | 0.0283 |
| 6 | `atr_14` | 0.0262 |
| 7 | `macd_histogram` | 0.0240 |
| 8 | `bb_upper` | 0.0237 |
| 9 | `bb_width` | 0.0224 |
| 10 | `williams_r_14` | 0.0194 |

### Signal Win Rates (5-day horizon)

These signals show the probability of price moving in the expected direction:

| Signal | Avg Win Rate | Interpretation |
|--------|-------------|----------------|
| `rsi_oversold` | 65.0% | ✅ Reliable (up) |
| `macd_cross` | 56.1% | ✅ Reliable (up) |
| `above_cloud` | 54.1% | ⚠️ Moderate (up) |
| `stoch_cross` | 53.9% | ⚠️ Moderate (up) |
| `di_cross` | 53.3% | ⚠️ Moderate (up) |
| `sma_50_200_cross` | 50.8% | ⚠️ Moderate (up) |
| `mfi_oversold` | 43.8% | ❌ Weak (up) |
| `mfi_overbought` | 41.7% | ❌ Weak (down) |
| `rsi_overbought` | 32.1% | ❌ Weak (down) |

## Key Insights

### Volume-Based Indicators Dominate
- **OBV (On-Balance Volume)** is consistently the most predictive indicator
- Volume confirms price moves - divergences often precede reversals

### Ichimoku Cloud is Highly Effective
- Senkou Span B and Chikou Span rank in top 5
- Cloud breakouts/breakdowns provide reliable directional signals

### Volatility Indicators Matter
- ATR and Bollinger Band width help predict move magnitude
- High volatility regimes require different indicator weighting

### Signal Recommendations

- **IONQ**: Best signal is `rsi_oversold` with 65.0% win rate
- **NVDA**: Best signal is `sma_50_200_cross` with 54.3% win rate

## Regime-Specific Recommendations

### In Trending Markets (Up or Down)
- Prioritize moving average crossovers (SMA 50/200)
- ADX > 25 confirms trend strength
- Follow momentum indicators (MACD)

### In Sideways/Ranging Markets
- Use oscillators: Stochastic, RSI, CCI
- Mean reversion strategies work better
- Bollinger Band extremes provide entry points

### In High Volatility
- Widen stops based on ATR
- DI crossovers become more significant
- Reduce position sizes

### In Low Volatility
- Stochastic RSI and Williams %R work well
- Watch for volatility expansion signals
- Prepare for breakout moves

## Model Performance

| Ticker | 1-Day Accuracy | 5-Day Accuracy | 10-Day Accuracy |
|--------|---------------|----------------|-----------------|
| IONQ | 51.6% | 39.6% | 39.6% |
| NVDA | 54.5% | 48.4% | 55.0% |

---

*Note: This analysis is for research purposes. Past performance does not guarantee future results.*