
CODEBASE INTERACTION MAP
═══════════════════════════════════════════════════════════════════════════════

DATA SOURCES (External)
─────────────────────────────────────────
  yfinance API ──────────────────────────────────────────────────────────────┐
  Yahoo Finance RSS (news) ──────────────────────────────────────────────────┤
                                                                              ▼

══════════════════════════════════════════════════════════════════════════════
 LAYER 1 — RAW DATA INGESTION (runs independently, writes to disk)
══════════════════════════════════════════════════════════════════════════════

  companies/src/ingest_yfinance.py          technicals/src/ingest_ohlcv.py
  ┌───────────────────────────────┐          ┌────────────────────────────┐
  │ fetch income stmt, balance    │          │ fetch daily OHLCV          │
  │ sheet, cash flow per ticker   │          │ 1y / 2y history            │
  │ Saves → companies/data/*.parq │          │ Saves → technicals/data/   │
  └───────────────┬───────────────┘          └────────────┬───────────────┘
                  │                                        │
                  ▼                                        ▼
  companies/src/feature_engineering.py     technicals/src/ingest_options.py
  ┌───────────────────────────────┐          ┌────────────────────────────┐
  │ 224–231 fundamental features  │          │ options chain per ticker   │
  │ per quarter (IONQ, NVDA)      │          │ Greeks, OI, IV, max pain   │
  │ Saves → companies/data/*.parq │          │ Saves → technicals/data/   │
  └───────────────┬───────────────┘          └────────────────────────────┘
                  │
                  ▼
  companies/src/evaluator.py
  ┌───────────────────────────────┐
  │ Scores: Value / Quality /     │
  │ Growth / Sentiment quadrants  │
  │ Outputs YAML scorecard        │
  └───────────────────────────────┘
       NOTE: companies/ folder is STANDALONE.
       It does NOT import from technicals/ or predictor/.
       Results sit on disk as parquet/JSON.

══════════════════════════════════════════════════════════════════════════════
 LAYER 2 — TECHNICAL ANALYSIS (runs independently, writes to disk)
══════════════════════════════════════════════════════════════════════════════

  technicals/src/indicators.py
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Class: TechnicalIndicators(df)                                         │
  │ Methods: rsi(), macd(), stochastic(), bollinger(), ichimoku(),          │
  │          adx(), atr(), obv(), pivot_points(), vwap()                   │
  │ Pure computation library — no I/O, no side effects                     │
  └──────────────┬─────────────────────────────────────────────────────────┘
                 │ imported by ↓
     ┌───────────┼────────────────────────────────┐
     ▼           ▼                                ▼
  analyze_technicals.py      ml_feature_importance.py    advanced_quant_analysis.py
  ┌───────────────────┐      ┌────────────────────────┐  ┌──────────────────────────┐
  │ Computes signals  │      │ RandomForest + XGBoost │  │ 170 features across:     │
  │ BUY/SELL per      │      │ SHAP analysis          │  │ - Microstructure (VWAP)  │
  │ ticker using      │      │ 100 features × 502 days│  │ - Vol (Parkinson, YZ)    │
  │ TechnicalIndicators│     │ Finds: OBV, Ichimoku,  │  │ - Risk (VaR, Sortino)    │
  │ + options data    │      │ ATR most predictive    │  │ - Cross-asset corr       │
  │ Saves → data/JSON │      │ Saves → data/ml/*.json │  │ - Stat-arb               │
  └───────────────────┘      └────────────────────────┘  └──────────────────────────┘
       NOTE: technicals/ folder is STANDALONE.
       indicators.py is the only shared dependency (used internally).
       ml results on disk ARE NOT yet piped into predictor/ automatically.

══════════════════════════════════════════════════════════════════════════════
 LAYER 3 — LIVE PREDICTOR (the active pipeline, runs stock_predictor.py)
══════════════════════════════════════════════════════════════════════════════

  yfinance (live)
       │
       ├──────────────────────────────────────────────────────────────────┐
       ▼                                                                  ▼
  realtime_data.py                                               fvg_analysis.py
  ┌────────────────────────────────────┐              ┌──────────────────────────────┐
  │ RealTimeDataFetcher(ticker)        │              │ FVGAnalyser(df, ticker)      │
  │                                    │              │                              │
  │ .fetch_all() → LiveMarketData      │              │ .analyse() → FVGAnalysisResult│
  │   ├── TechnicalData                │              │   ├── MarketStructure        │
  │   │     price, RSI, MACD,          │              │   │     HH/HL/LH/LL          │
  │   │     Stochastic, ADX,           │              │   │     BOS / ChoCH          │
  │   │     Bollinger, VWAP,           │              │   │     swing_high / low     │
  │   │     OBV, SMA/EMA,              │              │   ├── FairValueGaps[]        │
  │   │     volume_ratio, ATR          │              │   │     top/bottom/mitigated │
  │   ├── OptionsData                  │              │   ├── OrderBlocks[]          │
  │   │     IV, P/C ratio,             │              │   │     bullish / bearish    │
  │   │     max_pain, IV_skew,         │              │   ├── BreakerBlocks[]        │
  │   │     unusual_activity           │              │   ├── LiquiditySweeps[]      │
  │   └── FundamentalData              │              │   │     BSL / SSL sweeps     │
  │         P/E, ROE, margins,         │              │   ├── PremiumDiscount        │
  │         growth, analyst rating     │              │   │     OTE 61.8%–78.6%      │
  └────────────────────────────────────┘              │   └── bullish/bearish_targets│
                                                       └──────────────────────────────┘

  Yahoo Finance RSS
       │
       ▼
  news_tracker.py
  ┌────────────────────────────────────┐
  │ NewsTracker(ticker)                │
  │                                    │
  │ .analyze() → NewsAnalysis          │
  │   ├── overall_sentiment (-1 → +1)  │
  │   ├── event_signals[]              │
  │   │     Fed, earnings,             │
  │   │     geopolitical events        │
  │   │     matched to hist DB         │
  │   ├── predicted_impact{}           │
  │   └── news_count                   │
  └────────────────────────────────────┘

  ALL THREE ↓ feed into ↓

  stock_predictor.py  ←──────────── THE BRAIN
  ┌────────────────────────────────────────────────────────────────────────┐
  │ StockPredictor(ticker)                                                 │
  │                                                                        │
  │ .generate_prediction()                                                 │
  │   │                                                                    │
  │   ├── generate_technical_signal()  ──────── weight: 40%               │
  │   │     SCORING (each ± 0.0–0.3):                                     │
  │   │       trend_score    (25% of tech):                                │
  │   │         +0.3  price above SMA 20/50/200                           │
  │   │         +0.15 Golden Cross (SMA50 > SMA200)                       │
  │   │         +0.2  strong bullish ADX > 25                             │
  │   │       momentum_score (30% of tech):                               │
  │   │         +0.25 RSI < 30 (oversold)                                 │
  │   │         +0.20 MACD bullish crossover                              │
  │   │         +0.15 Stochastic < 20                                     │
  │   │       volatility_score (15% of tech):                             │
  │   │         +0.2  price at lower Bollinger Band                       │
  │   │         +0.15 price below VWAP                                    │
  │   │       volume_score (10% of tech):                                 │
  │   │         +0.2  high volume rally (>1.5x avg)                       │
  │   │         +0.1  OBV trending up                                     │
  │   │       options_score (10% of tech):                                │
  │   │         +0.15 low P/C ratio (< 0.7)                               │
  │   │         +0.1  price below max pain                                │
  │   │         +0.05 unusual call activity                               │
  │   │       fvg_score (20% of tech):       ← NEW                       │
  │   │         +0.3  ICT bias bullish/strong                             │
  │   │         +0.2  Bullish ChoCH detected                              │
  │   │         +0.15 Bullish BOS detected                                │
  │   │         +0.2  price sitting on Bullish FVG                        │
  │   │         +0.1  price in Discount zone                              │
  │   │         +0.1  SSL sweep + rejection                               │
  │   │                                                                    │
  │   │     DIRECTION: total_score > 0.15 → BULLISH                      │
  │   │                total_score < -0.15 → BEARISH else NEUTRAL         │
  │   │     STRENGTH:  min(1.0, |total_score| × 2)                       │
  │   │     CONFIDENCE: signal agreement ratio × 0.5 + 0.3               │
  │   │                                                                    │
  │   ├── generate_fundamental_signal()  ─────── weight: 30%              │
  │   │     value_score   (30%): P/E, Forward P/E, PEG                   │
  │   │     quality_score (30%): ROE, profit margin, debt/equity          │
  │   │     growth_score  (25%): revenue growth, earnings growth          │
  │   │     sentiment_score(15%): analyst rating, target upside           │
  │   │                                                                    │
  │   │     DIRECTION: total_score > 0.1 → BULLISH else BEARISH          │
  │   │     FAIR VALUE: PEG-based (growth × 1.5 = fair P/E)              │
  │   │                                                                    │
  │   ├── generate_news_signal()  ─────────────── weight: 30%             │
  │   │     sentiment_score (50%): keyword scoring on headlines           │
  │   │     event_score    (35%): match event → historical DB             │
  │   │                           avg SPY return after similar events     │
  │   │     momentum_score (15%): news volume × sentiment direction       │
  │   │                                                                    │
  │   └── COMBINE → combined_score                                        │
  │         tech_score  = direction × strength × confidence               │
  │         fund_score  = direction × strength × confidence               │
  │         news_score  = direction × strength × confidence               │
  │                                                                        │
  │         combined_score = tech×0.40 + fund×0.30 + news×0.30           │
  │                                                                        │
  │         > +0.3  → STRONG_BUY                                         │
  │         > +0.1  → BUY                                                 │
  │         > -0.1  → HOLD                                                │
  │         > -0.3  → SELL                                                │
  │         ≤ -0.3  → STRONG_SELL                                         │
  │                                                                        │
  │  PRICE TARGETS (priority order):                                      │
  │    1. FVG fill levels  (nearest unmitigated gap)                      │
  │    2. Order Block edges (mitigation targets)                          │
  │    3. Liquidity pools  (equal highs/lows clusters)                    │
  │    4. OTE Fibonacci    (61.8%–78.6% of swing range)                  │
  │    5. Swing high/low   (fallback)                                     │
  │    6. ATR-based        (last resort if no ICT levels)                 │
  │                                                                        │
  │  STOP LOSS:                                                           │
  │    → Just beyond nearest Order Block                                  │
  │    → Falls back to 2× ATR if no OB found                             │
  │                                                                        │
  │  OUTPUT: CombinedPrediction JSON + console report                     │
  └────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════
 CONNECTIVITY BETWEEN FOLDERS
══════════════════════════════════════════════════════════════════════════════

  companies/ ──────── NO direct code link to technicals/ or predictor/
                       (shares the yfinance data source, not code)

  technicals/ ─────── NO direct code link to companies/ or predictor/
    indicators.py is used INTERNALLY within technicals/ only

  predictor/ ─────────── SELF-CONTAINED live pipeline
    realtime_data.py  → standalone (yfinance only)
    fvg_analysis.py   → standalone (pandas/numpy only)
    news_tracker.py   → standalone (yfinance RSS only)
    stock_predictor.py → imports the above 3, nothing from companies/ or technicals/

  ML FINDINGS (from technicals/ml_feature_importance.py) inform the
  MANUAL WEIGHT DESIGN in stock_predictor.py but are NOT piped in at runtime.
  Example: OBV = highest feature importance → OBV trend included in volume_score.
           RSI oversold 65% win rate → RSI < 30 gets highest momentum weight.

══════════════════════════════════════════════════════════════════════════════
 SIGNAL → BUY/SELL DECISION TRACE  (concrete example: IONQ HOLD)
══════════════════════════════════════════════════════════════════════════════

  technical_signal:
    trend_score     = -0.40  (below all MAs, Death Cross)
    momentum_score  =  0.00  (RSI neutral, no MACD cross)
    volatility_score=  0.00  (not at BB edges)
    volume_score    = -0.10  (OBV down)
    options_score   = +0.05  (negative IV skew)
    fvg_score       = +0.0   (ranging structure, no FVGs)
    total_score     = (-0.40×0.20) + (0.00×0.25) + (0.00×0.15)
                    + (-0.10×0.10) + (0.05×0.10) + (0.0×0.20)
                    = -0.08 + 0 + 0 - 0.01 + 0.005 = -0.085
    → NEUTRAL (threshold: |score| > 0.15)
    strength  = min(1.0, 0.085 × 2) = 0.17 → ~0% displayed as 1%
    confidence = 0.57

  fundamental_signal:
    value_score     =  0.00
    quality_score   = -0.15
    growth_score    = +0.25  (revenue +428%)
    sentiment_score = +0.35  (analyst Buy rating)
    total_score     = 0.00×0.30 + -0.15×0.30 + 0.25×0.25 + 0.35×0.15
                    = 0 - 0.045 + 0.0625 + 0.0525 = +0.07
    → NEUTRAL (threshold: |score| > 0.1)

  news_signal:
    sentiment = 0.0  →  NEUTRAL

  COMBINE:
    tech_score  = 0 (NEUTRAL) × 0.17 × 0.57 = 0
    fund_score  = 0 (NEUTRAL) × 0.14 × 0.57 = 0
    news_score  = 0 (NEUTRAL) × 0.0  × 0.30 = 0
    combined    = 0×0.40 + 0×0.30 + 0×0.30  = 0.0
    → HOLD  (0.0 sits between -0.1 and +0.1)
