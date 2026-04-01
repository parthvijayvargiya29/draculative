# DRACULATIVE QUICKSTART GUIDE

**System:** Institutional-grade algorithmic trading system  
**Philosophy:** Atomic model — markets orbit a macro regime nucleus  
**Status:** Core framework operational, ready for TC implementation

---

## 🚀 QUICK START (5 Minutes)

### 1. Environment Setup

```bash
# Navigate to repo
cd /path/to/draculative

# Activate virtual environment
source .venv/bin/activate

# Verify dependencies
pip list | grep -E "scikit-learn|pandas|numpy"
```

### 2. Test Fundamental System (Already Working)

```bash
# Generate today's fundamental report
python scripts/run_ph_fundamental.py report

# View output
cat reports/ph_macro/$(date +%Y-%m-%d)_ph_fundamental_report.md
```

**Expected Output:**
- Transcript baseline: BULLISH (92.6% CV accuracy)
- News alignment score
- Topic-level convergence breakdown

### 3. Test Regime Classifier

```bash
python -m aggregation.regime_classifier
```

**Note:** Requires Alpaca API keys (see Configuration below)

### 4. Test Signal Router (Mock Data)

```bash
python -m aggregation.signal_router
```

**Expected Output:**
- Example of weighted signal aggregation
- Conflict resolution demonstration

---

## ⚙️ CONFIGURATION

### Alpaca API Setup (Required for Live Data)

1. **Get API Keys**
   - Sign up at https://alpaca.markets
   - Generate Paper Trading API keys

2. **Set Environment Variables**

```bash
# Add to ~/.zshrc or ~/.bashrc
export APCA_API_KEY_ID="your_key_id"
export APCA_API_SECRET_KEY="your_secret_key"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
```

3. **Verify Connection**

```bash
python -c "
from simulation.alpaca_loader import AlpacaDataLoader
loader = AlpacaDataLoader()
print('✅ Alpaca connected')
"
```

### News API Setup (Optional)

```bash
# If using NewsAPI.org
export NEWS_API_KEY="your_news_api_key"
```

---

## 🧬 ATOMIC SYSTEM ARCHITECTURE

### The 4 Layers (All Integrated in `combined_predictor.py`)

```
┌─────────────────────────────────────────────────────────┐
│                    LAYER 1: NUCLEUS                     │
│           (Macro Regime — The Gravitational Core)       │
│                                                          │
│  RegimeClassifier → TRENDING | CORRECTIVE | HIGH_VOL    │
│  Determines active TCs, position sizing, stop widths    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              LAYER 2: ELECTRON CLOUD                     │
│         (Technical Signals — Weighted by Regime)         │
│                                                          │
│  TC-01, TC-02, ... TC-15 → SignalRouter                 │
│  Weights applied, conflicts resolved → RoutedSignal     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           LAYER 3: GRAVITATIONAL FIELD                   │
│      (Fundamental Direction — Transcript-Based ML)       │
│                                                          │
│  PHFundamentalModel (92.6% CV accuracy)                 │
│  BULLISH/BEARISH/NEUTRAL → confidence boost/penalty     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         LAYER 4: REAL-TIME VALIDATION                    │
│      (News Convergence — Transcript Confirmation)        │
│                                                          │
│  Daily news vs transcript baseline                       │
│  High alignment → confidence boost, Low → penalty        │
└─────────────────────────────────────────────────────────┘
                            ↓
                   ┌─────────────┐
                   │   EXECUTE   │ if confidence ≥ 50%
                   └─────────────┘
```

---

## 📋 MODULE INVENTORY

### ✅ **COMPLETE** (Ready for Use)

| Module | Purpose | CLI Test |
|--------|---------|----------|
| `technicals/base_signal.py` | Abstract base for all TCs | N/A (abstract) |
| `aggregation/regime_classifier.py` | Macro regime detection | `python -m aggregation.regime_classifier` |
| `aggregation/signal_router.py` | TC weighting + routing | `python -m aggregation.signal_router` |
| `aggregation/combined_predictor.py` | Atomic core integration | `python -m aggregation.combined_predictor` |
| `fundamental/ph_transcript_parser.py` | Transcript → baseline | `python -m fundamental.ph_transcript_parser` |
| `fundamental/ph_fundamental_model.py` | Directional ML model | `python -m fundamental.ph_fundamental_model` |
| `fundamental/ph_news_alignment.py` | News convergence scoring | `python -m fundamental.ph_news_alignment` |
| `fundamental/ph_daily_report.py` | Daily report generator | `python -m fundamental.ph_daily_report` |

### 🚧 **NEXT PRIORITY** (This Week)

1. **TC-01: Supertrend** (refactor from Combo B)
2. **TC-02: BB + RSI2** (refactor from Combo C)
3. **TC-03: Hammer + EMA** (refactor from Combo A)
4. **TC-04: Fair Value Gap** (new implementation)
5. **Enhanced Friday Report** (weekly version with trends)

---

## 🎯 TYPICAL WORKFLOWS

### Workflow 1: Daily Pre-Market Analysis

```bash
# 1. Update transcripts (if new videos available)
python scripts/transcribe_ph_macro.py --refresh

# 2. Retrain fundamental model (if transcripts changed)
python scripts/run_ph_fundamental.py train

# 3. Generate daily report
python scripts/run_ph_fundamental.py report

# 4. Run combined prediction
python -m aggregation.combined_predictor --symbol SPY
```

### Workflow 2: Build & Validate New TC

```python
# technicals/tc_04_fvg.py

from technicals.base_signal import TechnicalSignal, ValidationResult
from technical.bar_snapshot import Signal, Direction, SignalStrength

class TC04_FairValueGap(TechnicalSignal):
    TC_ID = "TC-04"
    TC_NAME = "Fair Value Gap"
    TC_CATEGORY = TCCategory.STRUCTURE
    MIN_LOOKBACK = 50
    
    def compute(self, df: pd.DataFrame) -> Signal:
        # Implementation
        pass
    
    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        # Run Alpaca simulation
        pass

# Then run validation
if __name__ == "__main__":
    tc = TC04_FairValueGap()
    result = tc.validate(historical_df)
    print(f"PF: {result.metrics.profit_factor:.2f}")
    print(f"Approved: {result.approved}")
```

### Workflow 3: Friday Weekly Retraining

```bash
# Automated via cron (Friday 5:45 PM):
# 0 17 * * 5 cd /path/to/draculative && .venv/bin/python scripts/weekly_retrain.py

# Manual run:
python scripts/weekly_retrain.py
```

---

## 🧪 VALIDATION CHECKLIST (All TCs)

Before a TC can go ACTIVE, it must pass:

- [ ] Profit Factor (PF) ≥ 1.20 overall
- [ ] Out-of-Sample PF ≥ 0.90
- [ ] Win Rate between 35-70%
- [ ] Max Drawdown < 15%
- [ ] Minimum 8 trades in test period
- [ ] Walk-Forward Efficiency ≥ 0.80

**Validation Script:**
```bash
python -m simulation.validate_tc --tc-id TC-04 --symbol SPY --years 2
```

---

## 📊 CURRENT SYSTEM METRICS

| Metric | Value |
|--------|-------|
| **Fundamental Model CV Accuracy** | 92.6% |
| **Transcript Baseline Direction** | BULLISH (+65) |
| **Transcripts Parsed** | 15 (108,660 words) |
| **Topic Clusters** | 7 (RATES, DOLLAR, EQUITIES, etc.) |
| **ICT2 Concepts Built** | 8 (ready for TC standardization) |
| **Validation Gates** | 6 (PF, OOS PF, WR, DD, trades, WFE) |
| **Regime States** | 4 (TRENDING, CORRECTIVE, HIGH_VOL, RANGING) |

---

## 🔍 DEBUGGING TIPS

### Issue: "No SPY data available"
**Cause:** Alpaca API keys not configured  
**Fix:**
```bash
export APCA_API_KEY_ID="your_key"
export APCA_API_SECRET_KEY="your_secret"
```

### Issue: "Import error: fundamental module"
**Cause:** Not running from repo root  
**Fix:**
```bash
cd /path/to/draculative
python -m fundamental.ph_daily_report  # Use -m flag
```

### Issue: "Fundamental model not trained"
**Cause:** Model cache missing  
**Fix:**
```bash
python scripts/run_ph_fundamental.py train
```

---

## 📚 DOCUMENTATION INDEX

- **SYSTEM_STATUS.md** — Complete system overview, metrics, next steps
- **ARCHITECTURE.md** — Original ICT2 nucleus architecture (pre-atomic upgrade)
- **TRADING_SYSTEM_INTEGRATION.md** — TradingView webhook integration
- **DOCUMENTATION_INDEX.md** — Full module reference

---

## 🆘 SUPPORT & DEVELOPMENT

**Current Status:** Core atomic framework operational, 40% complete  
**Next Milestone:** TC-01 through TC-04 validated and ACTIVE  
**Timeline:** 2-3 weeks to full live paper trading

**Questions?** Review `SYSTEM_STATUS.md` for complete technical breakdown.

---

**Built:** 2026-04-01  
**Engine:** Draculative Alpha (Atomic Trading System)  
**Philosophy:** Markets are atomic. Everything orbits the nucleus. Nothing is isolated.
