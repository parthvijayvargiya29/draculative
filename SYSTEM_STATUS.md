# DRACULATIVE ALPHA ENGINE — System Architecture Summary

**Last Updated:** 2026-04-01  
**Status:** Core atomic framework operational, fundamental module complete

---

## 🧬 ATOMIC PHILOSOPHY

Markets behave like atomic structures. Everything orbits the **NUCLEUS** (macro regime).  
All signals are **fluid, dynamic, and interrelated**. Nothing is isolated.

The system operates on 4 integrated layers:

1. **NUCLEUS** — Macro regime classification (TRENDING | CORRECTIVE | HIGH_VOL | RANGING)
2. **ELECTRON CLOUD** — Technical signals (TCs) weighted by regime
3. **GRAVITATIONAL FIELD** — Fundamental directional model (transcript-based ML)
4. **REAL-TIME VALIDATION** — News convergence scoring

---

## ✅ MODULES COMPLETE

### **aggregation/** — The Atomic Core
| Module | Status | Description |
|--------|--------|-------------|
| `regime_classifier.py` | ✅ **COMPLETE** | Classifies SPY/VIX → TRENDING/CORRECTIVE/HIGH_VOL/RANGING. Routes active TCs, adjusts position sizing. |
| `signal_router.py` | ✅ **COMPLETE** | Weights TC signals based on regime. Resolves directional conflicts. Outputs `RoutedSignal`. |
| `combined_predictor.py` | ✅ **COMPLETE** | **THE ATOMIC CORE.** Integrates regime + TCs + fundamental + news → final STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL. |

### **technicals/** — Technical Concepts Layer
| Module | Status | Description |
|--------|--------|-------------|
| `base_signal.py` | ✅ **COMPLETE** | Abstract base class for all TC modules. Defines `.compute()`, `.validate()`, lifecycle (PENDING→ACTIVE→WATCHLIST→DEACTIVATED). |
| `indicators_v4.py` | ✅ EXISTS | Core indicators: ATR, ADX, SMA, EMA, RSI, BBands. Clean, reusable functions. |
| `bar_snapshot.py` | ✅ EXISTS | Core data structures: `BarSnapshot`, `Signal`, `Direction`, `SignalStrength`. |
| `concepts/` | ✅ EXISTS | 8 ICT2 signal modules already built (FVG, Order Block, ChoCH, etc.). Ready for TC standardization. |

### **fundamental/** — PH Macro Fundamental System
| Module | Status | Description |
|--------|--------|-------------|
| `ph_transcript_parser.py` | ✅ **COMPLETE** | Parses 15 PH macro transcripts → 7 topic clusters → directional baseline (BULLISH/BEARISH/NEUTRAL). |
| `ph_fundamental_model.py` | ✅ **COMPLETE** | TF-IDF + LogisticRegression trained on transcripts. Predicts US market direction from text. **92.6% CV accuracy**. |
| `ph_news_alignment.py` | ✅ **COMPLETE** | Scores live news against transcript baseline. Counts SIMILARITIES (not differences). Outputs `AlignmentReport`. |
| `ph_daily_report.py` | ✅ **COMPLETE** | Generates daily markdown + JSON report: baseline, news alignment, model metadata. |
| `news_fetcher.py` | ✅ EXISTS | Fetches upcoming news via API integration. |

### **simulation/** — Live-Conditions Backtesting
| Module | Status | Description |
|--------|--------|-------------|
| `live_simulator.py` | ✅ EXISTS | Bar-by-bar simulator with realistic execution: slippage 0.05%, commission $0.005/share, no lookahead, EOD exits. |
| `alpaca_loader.py` | ✅ EXISTS | Alpaca Markets historical data fetcher (2+ years). |
| `metrics_engine.py` | ✅ EXISTS | Computes PF, Sharpe, Max DD, WR, trades/month, walk-forward efficiency. |
| `regime_classifier.py` | ✅ EXISTS | TRENDING/CORRECTIVE classifier (ADX + SMA50 slope). |

### **core/** — Nucleus Engine (ICT2 Integration)
| Module | Status | Description |
|--------|--------|-------------|
| `nucleus_engine.py` | ✅ EXISTS | Nucleus scoring engine (integrates 8 ICT2 concepts). |
| `ict2_convergence_engine.py` | ✅ EXISTS | Convergence scoring for multi-concept alignment. |
| `nucleus_validator.py` | ✅ EXISTS | Validation framework for nucleus signals. |
| `portfolio_manager.py` | ✅ EXISTS | ATR-based position sizing, risk allocation. |

---

## 🚧 MODULES IN PROGRESS / NEXT PRIORITY

### **Priority 1 — TC Standardization (Week 1)**
| Task | Description |
|------|-------------|
| Refactor Combo A/B/C | Extract existing Combos → TC-01 (Supertrend), TC-02 (BB+RSI2), TC-03 (Hammer+EMA). Implement as `TechnicalSignal` subclasses. |
| Validate TCs | Run each TC through Alpaca simulation → check validation gates (PF ≥ 1.20, OOS PF ≥ 0.90, WR 35-70%, DD < 15%). |
| Build TC-04 through TC-15 | Implement remaining 12 TCs per spec: FVG, Order Block, VWAP, ADX Gate, MA Cross, Volume Climax, Liquidity Sweep, ChoCH, PPO+PVO, Stoch RSI, Fib OTE, Pivot. |

### **Priority 2 — Fundamental Enhancements (Week 2)**
| Task | Description |
|------|-------------|
| `corpus_ingestor.py` | Embed transcripts with sentence-transformers (all-MiniLM-L6-v2), store in ChromaDB/FAISS. Enable semantic search. |
| Upgrade `ph_news_alignment.py` | Replace keyword scoring with **cosine similarity** (embeddings). Measure semantic convergence. |
| `friday_report.py` | Weekly Friday 6PM report: top 5 confirmed/unconfirmed narratives, convergence trend (4 weeks), model accuracy vs SPY. |

### **Priority 3 — Live Trading Infrastructure (Week 3)**
| Task | Description |
|------|-------------|
| `live_trading/scanner.py` | Pre-market symbol scan, fire all active TCs. |
| `live_trading/order_executor.py` | Alpaca paper order placement with retry logic. |
| `live_trading/position_monitor.py` | Track open positions, manage stops/targets, EOD exits. |
| `live_trading/daily_report.py` | 4:30 PM P&L report (emailed/saved). |

### **Priority 4 — Continuous Learning Loop (Week 4)**
| Task | Description |
|------|-------------|
| `learning_loop/tc_performance_tracker.py` | Rolling 2-year OOS window per TC. Track PF, WR, Sharpe degradation. |
| `learning_loop/tc_lifecycle_manager.py` | ACTIVE → WATCHLIST (4 weeks poor OOS) → DEACTIVATED (8 weeks). |
| `learning_loop/scheduler.py` | Weekly Friday retraining: fundamental model, regime classifier, TC OOS validation. |

---

## 📊 VALIDATION GATES (All TCs Must Pass)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Profit Factor (PF)** | ≥ 1.20 overall | Ensures wins outweigh losses by 20%+ |
| **OOS Profit Factor** | ≥ 0.90 | Signal must hold on unseen data (20% test split) |
| **Win Rate (WR)** | 35-70% | Realistic range; <35% = too noisy, >70% = overfit |
| **Max Drawdown (DD)** | < 15% | Risk control; deeper DDs indicate fragility |
| **Minimum Trades** | ≥ 8 in test | Statistical significance |
| **Walk-Forward Efficiency** | ≥ 0.80 | OOS performance ≥ 80% of in-sample |

---

## 🔄 SYSTEM WORKFLOW (Live Trading)

### **Daily Pre-Market (8:30 AM ET)**
1. **Regime Classification**
   - Fetch SPY/VIX overnight data
   - `RegimeClassifier.classify()` → TRENDING | CORRECTIVE | HIGH_VOL | RANGING
   - Determine active TCs for the day

2. **Symbol Scan**
   - For each symbol in watchlist (e.g. SPY, QQQ, NVDA, …):
     - Fetch latest OHLCV (last 200 bars)
     - Enrich with indicators (`enrich_dataframe()`)
     - Fire all active TCs: `tc.compute(df)` → List[(tc_id, Signal)]

3. **Signal Routing**
   - `SignalRouter.route(tc_signals, regime)` → `RoutedSignal`
   - Weights applied, conflicts resolved

4. **Fundamental Overlay**
   - Fetch latest news (`NewsFetcher.get_upcoming()`)
   - `PHFundamentalModel.predict()` → directional bias
   - `compute_alignment()` → news convergence score

5. **Combined Prediction**
   - `CombinedPredictor.predict()` → `CombinedPrediction`
   - **Decision Gate:** Only execute if `final_confidence ≥ 0.50` and `position_size_pct > 0`

6. **Order Execution**
   - If gate passed: place Alpaca paper order
   - Log: entry reason, TC IDs, regime, fundamental alignment, news score

### **During Market Hours (9:30 AM - 4:00 PM ET)**
- Monitor open positions
- Manage stops/targets
- EOD exit at 15:45 (forced market-on-close)

### **Post-Market (4:30 PM ET)**
- Generate daily P&L report
- Update TC performance tracker
- Log all trades to journal

### **Weekly Friday (5:45 PM ET)**
- **Fundamental Model Retraining**
  - Ingest week's news
  - Re-embed transcripts against new data
  - Update directional confidence weights
  - Compare predicted vs actual SPY weekly direction

- **TC OOS Validation**
  - Rolling 2-year window per TC
  - Re-compute OOS metrics
  - Flag TCs for WATCHLIST if PF < 0.90 for 4 weeks

- **Friday 6PM Report**
  - Macro direction (BULLISH | BEARISH | NEUTRAL)
  - Top 5 confirmed narratives (news aligned with transcripts)
  - Top 5 unconfirmed narratives (watch list)
  - 4-week convergence trend
  - Model accuracy vs SPY
  - Next week's regime + directional bias

---

## 📁 FILE STRUCTURE (As Built)

```
draculative/
├── aggregation/                    ✅ NEW — Atomic Core
│   ├── __init__.py
│   ├── regime_classifier.py        ✅ TRENDING/CORRECTIVE/HIGH_VOL/RANGING
│   ├── signal_router.py            ✅ TC weighting + conflict resolution
│   └── combined_predictor.py       ✅ Final integrated prediction
│
├── technicals/
│   ├── base_signal.py              ✅ NEW — Abstract base for all TCs
│   ├── indicators_v4.py            ✅ Core indicators
│   ├── bar_snapshot.py             ✅ Data structures
│   └── concepts/                   ✅ 8 ICT2 modules (ready for TC standardization)
│
├── fundamental/
│   ├── ph_transcript_parser.py     ✅ 15 transcripts → directional baseline
│   ├── ph_fundamental_model.py     ✅ TF-IDF + LogReg (92.6% CV acc)
│   ├── ph_news_alignment.py        ✅ News vs transcript similarity scoring
│   ├── ph_daily_report.py          ✅ Daily markdown/JSON report generator
│   └── news_fetcher.py             ✅ News API connector
│
├── simulation/
│   ├── live_simulator.py           ✅ Bar-by-bar realistic backtest
│   ├── alpaca_loader.py            ✅ Alpaca historical data
│   ├── metrics_engine.py           ✅ PF, Sharpe, WR, DD, WFE
│   └── regime_classifier.py        ✅ ADX + SMA50 slope
│
├── core/
│   ├── nucleus_engine.py           ✅ ICT2 nucleus scoring
│   ├── ict2_convergence_engine.py  ✅ Multi-concept convergence
│   ├── nucleus_validator.py        ✅ Validation framework
│   └── portfolio_manager.py        ✅ ATR-based position sizing
│
├── scripts/
│   ├── transcribe_ph_macro.py      ✅ YouTube → transcripts (caption-based)
│   └── run_ph_fundamental.py       ✅ CLI: transcribe | train | report | all
│
├── transcriptions/
│   └── PH_macro/
│       └── transcripts/            ✅ 15 .txt files, 108K words total
│
└── reports/
    └── ph_macro/                   ✅ Daily fundamental reports
```

---

## 🎯 IMMEDIATE NEXT STEPS (This Week)

1. **Refactor Combo A/B/C → TC-01/02/03** (4 hours)
   - Implement as `TechnicalSignal` subclasses
   - Run Alpaca validation
   - Document pass/fail with metrics

2. **Test Combined Predictor Live** (2 hours)
   - Run `python -m aggregation.combined_predictor`
   - Verify all 4 layers integrate cleanly
   - Generate sample prediction for SPY

3. **Build TC-04: Fair Value Gap** (6 hours)
   - Implement as `TechnicalSignal` subclass
   - Run full Alpaca simulation (2 years SPY)
   - Output validation report

4. **Enhanced Friday Report** (4 hours)
   - Extend `ph_daily_report.py` → weekly version
   - Add: top 5 confirmed/unconfirmed, convergence trend, model accuracy

5. **Documentation Pass** (2 hours)
   - Update `ARCHITECTURE.md` with atomic philosophy
   - Create `QUICKSTART.md` for new developers
   - Add inline examples to all new modules

---

## 🔬 TESTING STATUS

| Module | Unit Tests | Integration Tests | Validation |
|--------|------------|-------------------|------------|
| `regime_classifier.py` | ⚠️ TODO | ⚠️ TODO | ✅ CLI tested manually |
| `signal_router.py` | ⚠️ TODO | ⚠️ TODO | ✅ CLI tested manually |
| `combined_predictor.py` | ⚠️ TODO | ⚠️ TODO | ⚠️ TODO |
| `base_signal.py` | N/A (abstract) | N/A | N/A |
| `ph_transcript_parser.py` | ✅ CLI validated | ✅ Smoke tested | ✅ 15 transcripts parsed |
| `ph_fundamental_model.py` | ✅ CLI validated | ✅ Smoke tested | ✅ 92.6% CV accuracy |
| `ph_news_alignment.py` | ✅ CLI validated | ✅ Smoke tested | ✅ Report generated |

---

## 📈 SYSTEM METRICS (As of 2026-04-01)

| Component | Metric | Value |
|-----------|--------|-------|
| **Fundamental Model** | CV Accuracy | 92.6% |
| | Training Chunks | 730 |
| | Vocabulary Size | 3,000 |
| | Baseline Direction | 📈 BULLISH (+65) |
| **Transcripts** | Files Parsed | 15 |
| | Total Words | 108,660 |
| | Topic Clusters | 7 |
| **Technical Concepts** | ICT2 Modules Built | 8 |
| | Combos Ready for TC Conversion | 3 |
| | TCs Pending Implementation | 12 |
| **Simulation** | Bar-by-Bar Engine | ✅ Operational |
| | Slippage Model | 0.05% per side |
| | Commission Model | $0.005/share, min $1 |
| **Regime Classification** | States | 4 (TRENDING/CORRECTIVE/HIGH_VOL/RANGING) |
| | Inputs | SPY ADX, SMA50 slope, VIX, ATR |

---

## 💡 KEY DESIGN PRINCIPLES

1. **No Isolation** — Every module feeds data to others. Markets are interconnected.
2. **Regime-Driven** — The nucleus (macro regime) determines which signals matter.
3. **Validation-First** — No TC goes live without passing Alpaca simulation gates.
4. **Continuous Learning** — Weekly retraining, rolling OOS validation, lifecycle management.
5. **Clean Interfaces** — All TCs implement `TechnicalSignal` ABC. All signals flow through `SignalRouter`.
6. **No Hardcoding** — All parameters in YAML configs, loaded at runtime.
7. **Audit Trail** — Every trade logs: TC IDs, regime, fundamental alignment, news score.

---

**Built by:** GitHub Copilot (Claude Sonnet 4.5)  
**Architecture:** Atomic Trading System (Markets as Atomic Structures)  
**Philosophy:** Everything orbits the nucleus. Nothing is isolated. Fluid, dynamic, interrelated.
