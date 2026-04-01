"""
Signal Generator — Lookahead-Free

Receives an IndicatorSnapshot (which itself only contains closed-bar data)
and emits a Signal: BUY, SELL, or FLAT.

Rules:
- Signal is FLAT until MIN_WARMUP_BARS bars have been seen (enforced by snapshot.is_valid)
- No parameters are tuned on the test set — this file defines the strategy logic
- Entry signal fires at bar CLOSE. Actual execution happens at NEXT bar OPEN.
"""

from dataclasses import dataclass
from typing import Optional
from indicators import IndicatorSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Signal constants
# ─────────────────────────────────────────────────────────────────────────────

class Direction:
    BUY  = 'BUY'
    SELL = 'SELL'
    FLAT = 'FLAT'


@dataclass
class Signal:
    """Output of the signal generator at one bar close."""
    direction:    str     = Direction.FLAT  # BUY / SELL / FLAT
    confidence:   float   = 0.0            # 0.0 – 1.0
    reason:       str     = ""             # Human-readable explanation
    atr:          float   = 0.0            # ATR at signal time (used for SL sizing)
    close:        float   = 0.0            # Price at signal bar close
    is_valid:     bool    = False          # False if warmup not met


# ─────────────────────────────────────────────────────────────────────────────
# Strategy parameters (defined once, not tuned on test data)
# ─────────────────────────────────────────────────────────────────────────────

# RSI thresholds
RSI_OVERSOLD    = 35      # BUY condition
RSI_OVERBOUGHT  = 65      # SELL condition

# MACD histogram must have the same sign for 2 consecutive bars (reduces noise)
MACD_CONFIRM_BARS = 2     # Not used directly here — crossover check is sufficient

# Bollinger Band position
BB_LOWER_THRESHOLD = 0.20   # bb_pct_b < 0.20 → near lower band
BB_UPPER_THRESHOLD = 0.80   # bb_pct_b > 0.80 → near upper band

# Minimum score to emit a non-FLAT signal
BUY_SCORE_THRESHOLD  = 2.0
SELL_SCORE_THRESHOLD = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Signal generator function
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(snap: IndicatorSnapshot) -> Signal:
    """
    Evaluate a single IndicatorSnapshot and return a Signal.

    Strategy: Multi-factor scoring
    ─────────────────────────────
    Three independent indicators each contribute a directional score.
    A signal fires only when total score exceeds the threshold.

    Factor 1 — MACD histogram crossover (weight: up to 2.0)
        BUY:  histogram crossed from negative to positive (prev < 0, curr > 0)
        SELL: histogram crossed from positive to negative (prev > 0, curr < 0)
        Also: continuing in direction adds 1.0

    Factor 2 — RSI extreme (weight: 1.5)
        BUY:  RSI < RSI_OVERSOLD
        SELL: RSI > RSI_OVERBOUGHT

    Factor 3 — Bollinger Band position (weight: 1.0)
        BUY:  price near lower band (pct_b < 0.20)
        SELL: price near upper band (pct_b > 0.80)

    This gives a maximum possible score of 4.5 in one direction.
    The threshold of 2.0 requires AT LEAST 2 factors to agree.
    """

    sig = Signal(
        direction  = Direction.FLAT,
        confidence = 0.0,
        atr        = snap.atr,
        close      = snap.close,
        is_valid   = snap.is_valid,
    )

    # Hard gate: no signal until warmup complete
    if not snap.is_valid:
        sig.reason = f"Warmup ({snap.bars_seen} bars)"
        return sig

    bull_score = 0.0
    bear_score = 0.0
    reasons_bull = []
    reasons_bear = []

    # ── Factor 1: MACD histogram ─────────────────────────────────────────────
    hist      = snap.macd_hist
    prev_hist = snap.prev_macd_hist

    if prev_hist < 0 and hist > 0:
        # Bullish crossover — strongest MACD signal
        bull_score += 2.0
        reasons_bull.append(f"MACD✗↑ hist={hist:+.4f}")
    elif hist > 0 and prev_hist > 0:
        # Continuing bullish momentum
        bull_score += 1.0
        reasons_bull.append(f"MACD↑ hist={hist:+.4f}")
    elif prev_hist > 0 and hist < 0:
        # Bearish crossover — strongest MACD signal
        bear_score += 2.0
        reasons_bear.append(f"MACD✗↓ hist={hist:+.4f}")
    elif hist < 0 and prev_hist < 0:
        # Continuing bearish momentum
        bear_score += 1.0
        reasons_bear.append(f"MACD↓ hist={hist:+.4f}")

    # ── Factor 2: RSI extreme ─────────────────────────────────────────────────
    rsi = snap.rsi
    if rsi < RSI_OVERSOLD:
        bull_score += 1.5
        reasons_bull.append(f"RSI oversold={rsi:.1f}")
    elif rsi > RSI_OVERBOUGHT:
        bear_score += 1.5
        reasons_bear.append(f"RSI overbought={rsi:.1f}")

    # ── Factor 3: Bollinger Band position ─────────────────────────────────────
    pct_b = snap.bb_pct_b
    if pct_b < BB_LOWER_THRESHOLD:
        bull_score += 1.0
        reasons_bull.append(f"BB lower pct_b={pct_b:.2f}")
    elif pct_b > BB_UPPER_THRESHOLD:
        bear_score += 1.0
        reasons_bear.append(f"BB upper pct_b={pct_b:.2f}")

    # ── Determine direction ───────────────────────────────────────────────────
    if bull_score >= BUY_SCORE_THRESHOLD and bull_score > bear_score:
        max_possible = 4.5
        confidence = min(bull_score / max_possible, 1.0)
        sig.direction  = Direction.BUY
        sig.confidence = round(confidence, 3)
        sig.reason     = " | ".join(reasons_bull)

    elif bear_score >= SELL_SCORE_THRESHOLD and bear_score > bull_score:
        max_possible = 4.5
        confidence = min(bear_score / max_possible, 1.0)
        sig.direction  = Direction.SELL
        sig.confidence = round(confidence, 3)
        sig.reason     = " | ".join(reasons_bear)

    else:
        sig.reason = (
            f"No signal (bull={bull_score:.1f}, bear={bear_score:.1f})"
        )

    return sig
