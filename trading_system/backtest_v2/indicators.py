"""
Lookahead-Free Indicator Engine

Calculates MACD, RSI, Bollinger Bands, ATR using a rolling state machine.
Each indicator update receives ONLY bars that have closed — no future data.

Design principles:
- IndicatorState is a struct of running state for ONE symbol
- Each call to update(bar) advances state by EXACTLY ONE bar
- The caller (simulation engine) controls the feed — no internal buffering tricks
- Minimum warmup bars are enforced: signals are INVALID until warmup is met
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
RSI_PERIOD  = 14
BB_PERIOD   = 20
BB_STD      = 2.0
ATR_PERIOD  = 14

# Minimum bars before any signal is valid.
# MACD needs slow_period + signal_period = 26 + 9 = 35 bars minimum.
# We use 50 to ensure all indicators are stable.
MIN_WARMUP_BARS = 50


# ─────────────────────────────────────────────────────────────────────────────
# EWM helper (online, single-pass)
# ─────────────────────────────────────────────────────────────────────────────

class EWM:
    """
    Online exponential weighted moving average.
    Uses the standard alpha = 2 / (span + 1) formula.
    Equivalent to pandas ewm(span=N, adjust=False).
    """
    def __init__(self, span: int):
        self.alpha = 2.0 / (span + 1)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# ─────────────────────────────────────────────────────────────────────────────
# Rolling window helper
# ─────────────────────────────────────────────────────────────────────────────

class RollingWindow:
    """Fixed-size deque for rolling statistics (std, mean)."""
    def __init__(self, size: int):
        self.size = size
        self._buf: deque = deque(maxlen=size)

    def append(self, x: float):
        self._buf.append(x)

    def full(self) -> bool:
        return len(self._buf) == self.size

    def mean(self) -> float:
        return float(np.mean(self._buf))

    def std(self) -> float:
        return float(np.std(self._buf, ddof=1)) if len(self._buf) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Indicator values (snapshot at one bar)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    """Values of all indicators at the current bar close. Immutable snapshot."""

    # Core prices
    close:  float = 0.0
    high:   float = 0.0
    low:    float = 0.0
    open:   float = 0.0
    volume: float = 0.0

    # MACD
    macd_line:   float = 0.0    # FAST_EWM - SLOW_EWM
    macd_signal: float = 0.0    # EWM(macd_line, SIGNAL_PERIOD)
    macd_hist:   float = 0.0    # macd_line - macd_signal
    prev_macd_hist: float = 0.0  # histogram one bar ago (for crossover detection)

    # RSI
    rsi: float = 50.0           # 0–100

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_mid:   float = 0.0
    bb_lower: float = 0.0
    bb_pct_b: float = 0.5       # (price - lower) / (upper - lower), 0–1

    # ATR (rolling mean of true range)
    atr: float = 0.0

    # Momentum
    momentum_pct: float = 0.0   # (close - close[N bars ago]) / close[N bars ago]

    # Bars seen so far
    bars_seen: int = 0
    is_valid:  bool = False     # False until MIN_WARMUP_BARS met


# ─────────────────────────────────────────────────────────────────────────────
# Indicator state machine for ONE symbol
# ─────────────────────────────────────────────────────────────────────────────

class IndicatorState:
    """
    Online indicator state for a single symbol.

    Call update(open, high, low, close, volume) on every closed bar.
    Returns an IndicatorSnapshot with values computed from closed data only.

    LOOKAHEAD PREVENTION:
    - Each update() call receives one historical bar at a time
    - The simulation engine feeds bars strictly in chronological order
    - No internal look-ahead: state only accumulates backward
    """

    def __init__(self):
        # MACD
        self._ema_fast   = EWM(MACD_FAST)
        self._ema_slow   = EWM(MACD_SLOW)
        self._ema_signal = EWM(MACD_SIGNAL)

        # RSI — Wilder smoothing (RMA)
        self._rsi_alpha    = 1.0 / RSI_PERIOD
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._rsi_seed_gains: list = []   # Collect first RSI_PERIOD changes for seeding

        # Bollinger Bands
        self._bb_window = RollingWindow(BB_PERIOD)

        # ATR
        self._atr_alpha = 1.0 / ATR_PERIOD
        self._atr_value: Optional[float] = None
        self._atr_prev_close: Optional[float] = None

        # Momentum (20-bar lookback)
        self._momentum_window = RollingWindow(21)

        # MACD histogram history for crossover detection
        self._prev_macd_hist: float = 0.0

        self._bars_seen = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, o: float, h: float, l: float,
               c: float, vol: float) -> IndicatorSnapshot:
        """
        Advance state by ONE closed bar. Returns indicator snapshot.
        This is the ONLY entry point — no batch processing allowed.
        """
        self._bars_seen += 1

        snap = IndicatorSnapshot(
            close=c, high=h, low=l, open=o, volume=vol,
            bars_seen=self._bars_seen,
        )

        # ── MACD ────────────────────────────────────────────────────────────
        ema_fast   = self._ema_fast.update(c)
        ema_slow   = self._ema_slow.update(c)
        macd_line  = ema_fast - ema_slow
        macd_sig   = self._ema_signal.update(macd_line)
        macd_hist  = macd_line - macd_sig

        snap.macd_line       = macd_line
        snap.macd_signal     = macd_sig
        snap.macd_hist       = macd_hist
        snap.prev_macd_hist  = self._prev_macd_hist
        self._prev_macd_hist = macd_hist

        # ── RSI (Wilder's smoothing) ─────────────────────────────────────────
        snap.rsi = self._update_rsi(c)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        self._bb_window.append(c)
        if self._bb_window.full():
            mid    = self._bb_window.mean()
            std    = self._bb_window.std()
            upper  = mid + BB_STD * std
            lower  = mid - BB_STD * std
            band_w = upper - lower

            snap.bb_upper = upper
            snap.bb_mid   = mid
            snap.bb_lower = lower
            snap.bb_pct_b = ((c - lower) / band_w) if band_w > 0 else 0.5
        else:
            snap.bb_upper = c
            snap.bb_mid   = c
            snap.bb_lower = c
            snap.bb_pct_b = 0.5

        # ── ATR (Wilder's RMA) ────────────────────────────────────────────────
        snap.atr = self._update_atr(h, l, c)

        # ── Momentum ──────────────────────────────────────────────────────────
        self._momentum_window.append(c)
        if self._momentum_window.full():
            oldest = self._momentum_window._buf[0]
            snap.momentum_pct = (c - oldest) / oldest if oldest > 0 else 0.0

        # ── Validity ──────────────────────────────────────────────────────────
        snap.is_valid = (self._bars_seen >= MIN_WARMUP_BARS)

        return snap

    # ── Private helpers ──────────────────────────────────────────────────────

    def _update_rsi(self, close: float) -> float:
        """Wilder RSI. Uses simple average seed, then RMA smoothing."""
        if self._prev_close is None:
            self._prev_close = close
            return 50.0

        change = close - self._prev_close
        self._prev_close = close

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        # Seed phase: collect RSI_PERIOD price changes before starting RMA
        if self._avg_gain is None:
            self._rsi_seed_gains.append((gain, loss))
            if len(self._rsi_seed_gains) >= RSI_PERIOD:
                gains = [g for g, _ in self._rsi_seed_gains]
                losses = [l for _, l in self._rsi_seed_gains]
                self._avg_gain = float(np.mean(gains))
                self._avg_loss = float(np.mean(losses))
            return 50.0

        # RMA smoothing (Wilder)
        self._avg_gain = (self._avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
        self._avg_loss = (self._avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD

        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _update_atr(self, high: float, low: float, close: float) -> float:
        """Wilder ATR using RMA smoothing (same as TradingView default)."""
        if self._atr_prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._atr_prev_close),
                abs(low  - self._atr_prev_close),
            )

        self._atr_prev_close = close

        if self._atr_value is None:
            self._atr_value = tr
        else:
            self._atr_value = (self._atr_value * (ATR_PERIOD - 1) + tr) / ATR_PERIOD

        return self._atr_value
