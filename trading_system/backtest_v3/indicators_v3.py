"""
Indicators V3.3 — Online Rolling Indicator Engine
══════════════════════════════════════════════════
One IndicatorStateV3 instance per symbol.
Call .update(o, h, l, c, vol, qqq_close) once per closed bar -> BarSnapshot.

New in V3.3 vs V3.2:
  rsi2           -- RSI(2) -- 2-period RSI for Combo C deeply oversold check
                   (V3.2 used RSI(14) with threshold 30, which produced only 1 trade;
                    RSI(2) is the fast oscillator designed for mean-reversion setups)

Retained from V3.2:
  BUGFIX: prev_st_dir captured BEFORE computing new ST direction
  atr_10, atr_10_5bars_ago  -- kept for compatibility (no longer used in Combo C entry)
  adx14_val, adx14_slope    -- ADX(14) for Combo A trend filter
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMA_FAST    = 8
EMA_RIBBON  = 21
EMA_MED     = 50
EMA_LONG    = 200
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIG    = 9
VOL_SMA_PERIOD = 20
ATR_PERIOD  = 14
ST_PERIOD   = 10
ST_MULT     = 2.5
RSI_PERIOD  = 14
RSI2_PERIOD = 2
BB_PERIOD   = 20
BB_STD_MULT = 2.0
CCI_PERIOD  = 20
MOM_PERIOD  = 10
ADX_PERIOD  = 7
ADX14_PERIOD = 14
BETA_PERIOD = 60
OBV_SLOPE_BARS = 10
ATR10_PERIOD = 10
ATR10_LOOKBACK = 5    # bars ago for ATR contraction check
MIN_WARMUP_BARS = 80

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class _EWM:
    __slots__ = ("alpha", "value", "_n")
    def __init__(self, span):
        self.alpha = 2.0 / (span + 1.0)
        self.value = 0.0
        self._n    = 0
    def update(self, x):
        if self._n == 0:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        self._n += 1
        return self.value


class _Wilder:
    __slots__ = ("period", "alpha", "value", "_n")
    def __init__(self, period):
        self.period = period
        self.alpha  = 1.0 / period
        self.value  = 0.0
        self._n     = 0
    def update(self, x):
        if self._n == 0:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        self._n += 1
        return self.value


class _RollingBuf:
    __slots__ = ("_d", "_maxlen")
    def __init__(self, size):
        self._maxlen = size
        self._d = deque(maxlen=size)
    def append(self, x):        self._d.append(x)
    def full(self):             return len(self._d) == self._maxlen
    def mean(self):             return sum(self._d) / len(self._d) if self._d else 0.0
    def std(self):
        if len(self._d) < 2: return 0.0
        m = self.mean()
        return math.sqrt(sum((x - m)**2 for x in self._d) / (len(self._d) - 1))
    def sum(self):              return sum(self._d)
    def last(self):             return self._d[-1] if self._d else 0.0
    def oldest(self):           return self._d[0]  if self._d else 0.0
    def min(self):              return min(self._d) if self._d else 0.0
    def max(self):              return max(self._d) if self._d else 0.0
    def __len__(self):          return len(self._d)

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class BarSnapshot:
    """All indicator values at one bar close."""
    open:   float = 0.0
    high:   float = 0.0
    low:    float = 0.0
    close:  float = 0.0
    volume: float = 0.0

    # EMA Ribbon
    ema8:   float = 0.0
    ema21:  float = 0.0
    ema50:  float = 0.0
    ema200: float = 0.0

    # MACD
    macd_line:           float = 0.0
    macd_signal:         float = 0.0
    macd_hist:           float = 0.0
    prev_macd_hist:      float = 0.0
    macd_cross_bars_ago: int   = 999

    # Volume
    vol_sma:   float = 0.0
    vol_ratio: float = 0.0

    # ATR
    atr:               float = 0.0
    atr_10:            float = 0.0   # ATR(10) for Combo C volatility contraction
    atr_10_5bars_ago:  float = 0.0   # ATR(10) from 5 bars ago

    # SuperTrend
    supertrend:       float = 0.0
    supertrend_line:  float = 0.0
    supertrend_dir:   int   = 0
    prev_st_dir:      int   = 0

    # ADX
    adx_val:    float = 0.0
    adx_slope:  float = 0.0
    adx14_val:  float = 0.0   # ADX(14) for Combo A trend confirmation
    adx14_slope: float = 0.0

    # RSI
    rsi:  float = 50.0   # RSI(14) -- used by Combo B gate
    rsi2: float = 50.0   # RSI(2)  -- used by Combo C deep oversold check

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_mid:   float = 0.0
    bb_lower: float = 0.0
    bb_pct_b: float = 0.5

    # CCI (backward compat)
    cci: float = 0.0

    # OBV
    obv:           float = 0.0
    obv_slope:     float = 0.0
    obv_new_low:   bool  = False
    price_new_low: bool  = False

    # Momentum
    momentum: float = 0.0

    # Beta vs QQQ
    beta_60: float = 1.0

    # Breakout -- Combo A
    high20: float = 0.0  # 20-bar highest HIGH (excluding current bar) for breakout detection

    # Regime classification (SMA50 value, for slope computation in run_v3)
    sma50_val: float = 0.0  # raw EMA50 value aliased as SMA50 for regime use

    # ── V5.0 additions ─────────────────────────────────────────────────────
    # Module 1 scoring
    bb_std:           float = 0.0   # BB standard deviation (for penetration depth score)
    bb_penetration_sd: float = 0.0  # (bb_lower - close) / bb_std  (Component 2)
    target_dist_atr:  float = 0.0   # (bb_mid - close) / atr_10    (Component 3)
    spy_rsi14:        float = 50.0  # SPY RSI(14) at signal bar     (Component 5)
    # Module 2 exits
    atr_pct_rank_60:  float = 50.0  # ATR(10) percentile rank vs rolling 60-bar window
    # Module 3 trend signal
    ema63:            float = 0.0   # EMA(63) for trend module medium-term alignment
    high252:          float = 0.0   # rolling 252-bar highest HIGH for 52-week proximity
    highest_close_10: float = 0.0   # highest close of last 10 bars (pullback check)
    adx14_3bar_ago:   float = 0.0   # ADX(14) value 3 bars ago (slope acceleration check)

    # Meta
    bars_seen: int  = 0
    ready:     bool = False


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class IndicatorStateV3:
    """
    Per-symbol rolling indicator engine V3.1.

    Call .update(o, h, l, c, vol, qqq_close=0.0) once per closed bar.
    qqq_close=0.0 disables beta calculation (beta_60 stays 1.0).
    """

    def __init__(self):
        # EMA ribbon
        self._ema8   = _EWM(EMA_FAST)
        self._ema21  = _EWM(EMA_RIBBON)
        self._ema50  = _EWM(EMA_MED)
        self._ema200 = _EWM(EMA_LONG)

        # MACD
        self._macd_fast         = _EWM(MACD_FAST)
        self._macd_slow         = _EWM(MACD_SLOW)
        self._macd_sig          = _EWM(MACD_SIG)
        self._prev_hist         = 0.0
        self._macd_prev_line    = 0.0
        self._macd_prev_signal  = 0.0
        self._macd_cross_ctr    = 999

        # Volume
        self._vol_buf = _RollingBuf(VOL_SMA_PERIOD)

        # ATR(14)
        self._atr_wilder         = _Wilder(ATR_PERIOD)
        self._prev_close_atr     = None

        # ATR(10) -- for Combo C volatility contraction
        self._atr10_wilder       = _Wilder(ATR10_PERIOD)
        self._prev_close_atr10   = None
        self._atr10_buf          = _RollingBuf(ATR10_LOOKBACK + 1)  # keep 6 values

        # SuperTrend
        self._st_atr_wilder      = _Wilder(ST_PERIOD)
        self._st_prev_close      = None
        self._st_ub              = None   # None until first bar computed
        self._st_lb              = None   # None until first bar computed
        self._st_dir             = 1
        self._prev_st_dir        = 1

        # ADX (7-bar)
        self._adx_prev_high      = None
        self._adx_prev_low       = None
        self._adx_prev_close     = None
        self._adx_tr_wilder      = _Wilder(ADX_PERIOD)
        self._adx_pdm_wilder     = _Wilder(ADX_PERIOD)
        self._adx_ndm_wilder     = _Wilder(ADX_PERIOD)
        self._adx_wilder         = _Wilder(ADX_PERIOD)
        self._prev_adx_val       = 0.0

        # ADX (14-bar) -- for Combo A
        self._adx14_prev_high    = None
        self._adx14_prev_low     = None
        self._adx14_prev_close   = None
        self._adx14_tr_wilder    = _Wilder(ADX14_PERIOD)
        self._adx14_pdm_wilder   = _Wilder(ADX14_PERIOD)
        self._adx14_ndm_wilder   = _Wilder(ADX14_PERIOD)
        self._adx14_wilder       = _Wilder(ADX14_PERIOD)
        self._prev_adx14_val     = 0.0

        # RSI(14)
        self._rsi_gain           = _Wilder(RSI_PERIOD)
        self._rsi_loss           = _Wilder(RSI_PERIOD)
        self._rsi_seed_buf       = []
        self._rsi_seeded         = False
        self._prev_close_rsi     = None

        # RSI(2) -- fast oscillator for Combo C deep oversold
        self._rsi2_gain          = _Wilder(RSI2_PERIOD)
        self._rsi2_loss          = _Wilder(RSI2_PERIOD)
        self._rsi2_seed_buf      = []
        self._rsi2_seeded        = False
        self._prev_close_rsi2    = None

        # Bollinger Bands
        self._bb_buf = _RollingBuf(BB_PERIOD)

        # CCI
        self._cci_buf = _RollingBuf(CCI_PERIOD)

        # OBV
        self._obv                = 0.0
        self._obv_buf            = _RollingBuf(OBV_SLOPE_BARS)
        self._prev_obv           = 0.0
        self._prev_close_obv     = None

        # Momentum
        self._mom_buf = _RollingBuf(MOM_PERIOD + 1)

        # Beta vs QQQ
        self._sym_ret_buf        = _RollingBuf(BETA_PERIOD)
        self._qqq_ret_buf        = _RollingBuf(BETA_PERIOD)
        self._prev_close_beta    = None
        self._prev_qqq_close     = None

        # 20-bar highest high (for Combo A breakout)
        self._high20_buf = _RollingBuf(20)

        # ── V5.0 additions ─────────────────────────────────────────────────
        # EMA(63) for Module 3 trend medium-term alignment
        self._ema63 = _EWM(63)

        # 252-bar highest HIGH for 52-week proximity (Module 3)
        self._high252_buf = _RollingBuf(252)

        # 10-bar highest CLOSE for pullback check (Module 3)
        self._high10c_buf = _RollingBuf(10)

        # ATR(10) 60-bar rolling window for percentile rank (Module 2)
        self._atr60_buf = _RollingBuf(60)

        # ADX14 3-bar-ago ring buffer (for slope acceleration check)
        self._adx14_hist = _RollingBuf(4)   # keep last 4 ADX14 values

        self._bars_seen = 0

    def update(self, o, h, l, c, vol, qqq_close=0.0):
        """
        Advance by one closed bar. Returns BarSnapshot.
        qqq_close: pass QQQ close for beta; 0.0 = skip.
        """
        self._bars_seen += 1
        snap = BarSnapshot(open=o, high=h, low=l, close=c, volume=vol,
                           bars_seen=self._bars_seen)

        # EMA ribbon
        snap.ema8   = self._ema8.update(c)
        snap.ema21  = self._ema21.update(c)
        snap.ema50  = self._ema50.update(c)
        snap.ema200 = self._ema200.update(c)
        snap.sma50_val = snap.ema50   # alias for regime SMA50 slope computation

        # MACD
        fast = self._macd_fast.update(c)
        slow = self._macd_slow.update(c)
        line = fast - slow
        sig  = self._macd_sig.update(line)
        hist = line - sig

        prev_above = self._macd_prev_line > self._macd_prev_signal
        curr_above = line > sig
        if prev_above != curr_above:
            self._macd_cross_ctr = 0
        else:
            self._macd_cross_ctr = min(self._macd_cross_ctr + 1, 999)

        snap.macd_line           = line
        snap.macd_signal         = sig
        snap.macd_hist           = hist
        snap.prev_macd_hist      = self._prev_hist
        snap.macd_cross_bars_ago = self._macd_cross_ctr
        self._prev_hist          = hist
        self._macd_prev_line     = line
        self._macd_prev_signal   = sig

        # Volume
        self._vol_buf.append(vol)
        snap.vol_sma   = self._vol_buf.mean()
        snap.vol_ratio = vol / snap.vol_sma if snap.vol_sma > 0 else 1.0

        # ATR(14)
        snap.atr = self._update_atr(h, l, c)

        # ATR(10) -- Combo C volatility contraction
        snap.atr_10, snap.atr_10_5bars_ago = self._update_atr10(h, l, c)

        # SuperTrend
        st_line, st_dir, prev_st_dir = self._update_supertrend(h, l, c)
        snap.supertrend_line = st_line
        snap.supertrend      = st_line
        snap.supertrend_dir  = st_dir
        snap.prev_st_dir     = prev_st_dir

        # ADX(7)
        snap.adx_val, snap.adx_slope = self._update_adx(h, l, c)

        # ADX(14) -- Combo A trend filter
        snap.adx14_val, snap.adx14_slope = self._update_adx14(h, l, c)

        # RSI(14)
        snap.rsi = self._update_rsi(c)

        # RSI(2) -- fast oscillator
        snap.rsi2 = self._update_rsi2(c)

        # Bollinger Bands
        self._bb_buf.append(c)
        if self._bb_buf.full():
            mid   = self._bb_buf.mean()
            std   = self._bb_buf.std()
            upper = mid + BB_STD_MULT * std
            lower = mid - BB_STD_MULT * std
            bw    = upper - lower
            snap.bb_upper = upper
            snap.bb_mid   = mid
            snap.bb_lower = lower
            snap.bb_pct_b = (c - lower) / bw if bw > 0 else 0.5
        else:
            snap.bb_upper = snap.bb_lower = snap.bb_mid = c
            snap.bb_pct_b = 0.5

        # CCI
        typical = (h + l + c) / 3.0
        self._cci_buf.append(typical)
        if self._cci_buf.full():
            tp_mean = self._cci_buf.mean()
            md = sum(abs(x - tp_mean) for x in self._cci_buf._d) / len(self._cci_buf)
            snap.cci = (typical - tp_mean) / (0.015 * md) if md > 0 else 0.0

        # OBV + divergence
        prev_c_obv = self._prev_close_obv if self._prev_close_obv is not None else c
        if self._bars_seen > 1:
            if c > prev_c_obv:
                self._obv += vol
            elif c < prev_c_obv:
                self._obv -= vol
        snap.obv       = self._obv
        self._obv_buf.append(self._obv)
        snap.obv_slope = (self._obv - self._obv_buf.oldest()
                          if self._obv_buf.full() else 0.0)
        snap.obv_new_low   = (self._bars_seen > 1) and (self._obv < self._prev_obv)
        snap.price_new_low = (self._prev_close_obv is not None) and (c < prev_c_obv)
        self._prev_obv       = self._obv
        self._prev_close_obv = c

        # Momentum
        self._mom_buf.append(c)
        if self._mom_buf.full():
            old = self._mom_buf.oldest()
            snap.momentum = (c - old) / old if old > 0 else 0.0

        # 20-bar highest high (previous 20 bars, excluding current bar)
        if self._high20_buf.full():
            snap.high20 = self._high20_buf.max()
        self._high20_buf.append(h)

        # Beta
        snap.beta_60 = self._update_beta(c, qqq_close)

        # ── V5.0 additions ──────────────────────────────────────────────────
        # EMA(63)
        snap.ema63 = self._ema63.update(c)

        # 252-bar highest HIGH (for 52-week proximity)
        if self._high252_buf.full():
            snap.high252 = self._high252_buf.max()
        self._high252_buf.append(h)

        # 10-bar highest CLOSE (for 2% pullback check)
        if self._high10c_buf.full():
            snap.highest_close_10 = self._high10c_buf.max()
        self._high10c_buf.append(c)

        # BB std dev and penetration depth (Component 2, Module 1)
        if self._bb_buf.full():
            snap.bb_std = self._bb_buf.std()
            if snap.bb_std > 0 and snap.bb_lower > 0:
                snap.bb_penetration_sd = max(0.0, (snap.bb_lower - c) / snap.bb_std)
            # target distance in ATR units (Component 3, Module 1)
            atr10_v = snap.atr_10 if snap.atr_10 > 0 else snap.atr
            if atr10_v > 0 and snap.bb_mid > 0:
                snap.target_dist_atr = max(0.0, (snap.bb_mid - c) / atr10_v)

        # ATR(10) percentile rank vs 60-bar window (Module 2 vol-adj time stop)
        self._atr60_buf.append(snap.atr_10)
        if len(self._atr60_buf) >= 10:
            vals = list(self._atr60_buf._d)
            cur  = vals[-1]
            snap.atr_pct_rank_60 = (sum(1 for v in vals if v < cur) / len(vals)) * 100.0

        # ADX14 3-bar-ago (for slope acceleration: ADX14[0] > ADX14[3])
        self._adx14_hist.append(snap.adx14_val)
        if len(self._adx14_hist) >= 4:
            snap.adx14_3bar_ago = list(self._adx14_hist._d)[0]

        snap.ready = self._bars_seen >= MIN_WARMUP_BARS
        return snap

    # ---- Private helpers --------------------------------------------------

    def _update_atr(self, h, l, c):
        if self._prev_close_atr is None:
            tr = h - l
        else:
            tr = max(h - l,
                     abs(h - self._prev_close_atr),
                     abs(l - self._prev_close_atr))
        self._prev_close_atr = c
        return self._atr_wilder.update(tr)

    def _update_atr10(self, h, l, c):
        """ATR(10) with 5-bar lookback buffer for contraction check."""
        if self._prev_close_atr10 is None:
            tr = h - l
        else:
            tr = max(h - l,
                     abs(h - self._prev_close_atr10),
                     abs(l - self._prev_close_atr10))
        self._prev_close_atr10 = c
        atr10 = self._atr10_wilder.update(tr)
        self._atr10_buf.append(atr10)
        # atr_10_5bars_ago: oldest in buffer if full (6 values = current + 5 prior)
        if len(self._atr10_buf) >= ATR10_LOOKBACK + 1:
            atr_5ago = self._atr10_buf.oldest()
        else:
            atr_5ago = atr10  # not enough history: treat as equal (no contraction)
        return atr10, atr_5ago

    def _update_supertrend(self, h, l, c):
        if self._st_prev_close is None:
            tr = h - l
        else:
            tr = max(h - l,
                     abs(h - self._st_prev_close),
                     abs(l - self._st_prev_close))
        self._st_prev_close = c
        atr    = self._st_atr_wilder.update(tr)
        hl2    = (h + l) / 2.0
        raw_ub = hl2 + ST_MULT * atr
        raw_lb = hl2 - ST_MULT * atr

        # Direction-dependent SuperTrend band pinning:
        # When BULLISH  (dir=1):  lower band is the active trailing stop — ratchet UP only
        #                         upper band resets freely (not the active line)
        # When BEARISH  (dir=-1): upper band is the active trailing stop — ratchet DOWN only
        #                         lower band resets freely (not the active line)
        if self._st_ub is None:
            # First bar: initialize both bands directly
            final_ub = raw_ub
            final_lb = raw_lb
        else:
            prev_ub = self._st_ub
            prev_lb = self._st_lb
            if self._st_dir == 1:
                # Bullish: lower band is the active support — it can only ratchet UP
                final_lb = max(raw_lb, prev_lb)
                final_ub = raw_ub   # upper band resets freely (not active in bullish mode)
            else:
                # Bearish: upper band is the active resistance — it can only ratchet DOWN
                final_ub = min(raw_ub, prev_ub)
                final_lb = raw_lb   # lower band resets freely (not active in bearish mode)

        # V3.2 FIX: capture prev BEFORE computing new direction
        # self._st_dir still holds the PREVIOUS bar's direction here
        prev_dir = self._st_dir          # save prior bar's direction
        if self._st_dir == 1:
            # Bullish: flip bearish only if close breaks BELOW lower band (support)
            new_dir = -1 if c < final_lb else 1
        else:
            # Bearish: flip bullish only if close breaks ABOVE upper band (resistance)
            new_dir = 1 if c > final_ub else -1
        self._prev_st_dir = prev_dir     # store for snapshot
        self._st_dir      = new_dir      # advance state
        self._st_ub       = final_ub
        self._st_lb       = final_lb
        band_val = final_lb if new_dir == 1 else final_ub
        return band_val, new_dir, prev_dir

    def _update_adx(self, h, l, c):
        if self._adx_prev_high is None:
            self._adx_prev_high  = h
            self._adx_prev_low   = l
            self._adx_prev_close = c
            return 0.0, 0.0
        tr = max(h - l,
                 abs(h - self._adx_prev_close),
                 abs(l - self._adx_prev_close))
        up_move   = h - self._adx_prev_high
        down_move = self._adx_prev_low - l
        if up_move > down_move and up_move > 0:
            pdm, ndm = up_move, 0.0
        elif down_move > up_move and down_move > 0:
            pdm, ndm = 0.0, down_move
        else:
            pdm, ndm = 0.0, 0.0
        self._adx_prev_high  = h
        self._adx_prev_low   = l
        self._adx_prev_close = c
        atr14  = self._adx_tr_wilder.update(tr)
        pdm14  = self._adx_pdm_wilder.update(pdm)
        ndm14  = self._adx_ndm_wilder.update(ndm)
        if atr14 > 0:
            pdi = (pdm14 / atr14) * 100.0
            ndi = (ndm14 / atr14) * 100.0
        else:
            pdi = ndi = 0.0
        di_sum  = pdi + ndi
        di_diff = abs(pdi - ndi)
        dx = (di_diff / di_sum) * 100.0 if di_sum > 0 else 0.0
        adx_val  = self._adx_wilder.update(dx)
        slope    = adx_val - self._prev_adx_val
        self._prev_adx_val = adx_val
        return adx_val, slope

    def _update_adx14(self, h, l, c):
        """ADX(14) — same algorithm as ADX(7) but with 14-bar Wilder smoothing."""
        if self._adx14_prev_high is None:
            self._adx14_prev_high  = h
            self._adx14_prev_low   = l
            self._adx14_prev_close = c
            return 0.0, 0.0
        tr = max(h - l,
                 abs(h - self._adx14_prev_close),
                 abs(l - self._adx14_prev_close))
        up_move   = h - self._adx14_prev_high
        down_move = self._adx14_prev_low - l
        if up_move > down_move and up_move > 0:
            pdm, ndm = up_move, 0.0
        elif down_move > up_move and down_move > 0:
            pdm, ndm = 0.0, down_move
        else:
            pdm, ndm = 0.0, 0.0
        self._adx14_prev_high  = h
        self._adx14_prev_low   = l
        self._adx14_prev_close = c
        atr14  = self._adx14_tr_wilder.update(tr)
        pdm14  = self._adx14_pdm_wilder.update(pdm)
        ndm14  = self._adx14_ndm_wilder.update(ndm)
        if atr14 > 0:
            pdi = (pdm14 / atr14) * 100.0
            ndi = (ndm14 / atr14) * 100.0
        else:
            pdi = ndi = 0.0
        di_sum  = pdi + ndi
        di_diff = abs(pdi - ndi)
        dx = (di_diff / di_sum) * 100.0 if di_sum > 0 else 0.0
        adx14_val  = self._adx14_wilder.update(dx)
        slope      = adx14_val - self._prev_adx14_val
        self._prev_adx14_val = adx14_val
        return adx14_val, slope

    def _update_rsi(self, c):
        if self._prev_close_rsi is None:
            self._prev_close_rsi = c
            return 50.0
        change = c - self._prev_close_rsi
        self._prev_close_rsi = c
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if not self._rsi_seeded:
            self._rsi_seed_buf.append((gain, loss))
            if len(self._rsi_seed_buf) >= RSI_PERIOD:
                avg_g = sum(g for g, _ in self._rsi_seed_buf) / RSI_PERIOD
                avg_l = sum(lo for _, lo in self._rsi_seed_buf) / RSI_PERIOD
                self._rsi_gain.value = avg_g
                self._rsi_loss.value = avg_l
                self._rsi_seeded = True
            return 50.0
        avg_g = self._rsi_gain.update(gain)
        avg_l = self._rsi_loss.update(loss)
        if avg_l == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    def _update_rsi2(self, c):
        """RSI(2) — 2-period Wilder RSI for fast mean-reversion oversold detection."""
        if self._prev_close_rsi2 is None:
            self._prev_close_rsi2 = c
            return 50.0
        change = c - self._prev_close_rsi2
        self._prev_close_rsi2 = c
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if not self._rsi2_seeded:
            self._rsi2_seed_buf.append((gain, loss))
            if len(self._rsi2_seed_buf) >= RSI2_PERIOD:
                avg_g = sum(g for g, _ in self._rsi2_seed_buf) / RSI2_PERIOD
                avg_l = sum(lo for _, lo in self._rsi2_seed_buf) / RSI2_PERIOD
                self._rsi2_gain.value = avg_g
                self._rsi2_loss.value = avg_l
                self._rsi2_seeded = True
            return 50.0
        avg_g = self._rsi2_gain.update(gain)
        avg_l = self._rsi2_loss.update(loss)
        if avg_l == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    def _update_beta(self, c, qqq_close):
        sym_ret = 0.0
        qqq_ret = 0.0
        if self._prev_close_beta is not None and self._prev_close_beta > 0:
            sym_ret = (c - self._prev_close_beta) / self._prev_close_beta
        if (qqq_close > 0 and
                self._prev_qqq_close is not None and
                self._prev_qqq_close > 0):
            qqq_ret = (qqq_close - self._prev_qqq_close) / self._prev_qqq_close
        self._prev_close_beta = c
        if qqq_close > 0:
            self._prev_qqq_close = qqq_close
        if qqq_close > 0 and self._prev_close_beta is not None:
            self._sym_ret_buf.append(sym_ret)
            self._qqq_ret_buf.append(qqq_ret)
        if not self._sym_ret_buf.full() or not self._qqq_ret_buf.full():
            return 1.0
        sym_rets = list(self._sym_ret_buf._d)
        qqq_rets = list(self._qqq_ret_buf._d)
        n = len(sym_rets)
        sym_mean = sum(sym_rets) / n
        qqq_mean = sum(qqq_rets) / n
        cov = sum((s - sym_mean) * (q - qqq_mean)
                  for s, q in zip(sym_rets, qqq_rets)) / (n - 1)
        var = sum((q - qqq_mean)**2 for q in qqq_rets) / (n - 1)
        if var < 1e-10:
            return 1.0
        return max(-3.0, min(5.0, cov / var))
