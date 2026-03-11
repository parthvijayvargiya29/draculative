#!/usr/bin/env python3
"""
Fair Value Gap (FVG) & Advanced Price Action Analysis

Implements ICT (Inner Circle Trader) / SMC (Smart Money Concepts):
- Fair Value Gaps (Bullish / Bearish FVGs)
- Order Blocks (Bullish / Bearish)
- Breaker Blocks
- Liquidity Sweeps (Buy-side / Sell-side)
- Change of Character (ChoCH) & Break of Structure (BOS)
- Premium / Discount Zones
- Optimal Trade Entry (OTE) via Fibonacci
- Imbalance / Inefficiency detection
- Mitigation levels (where price is likely to return)
- Inducement levels
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class FairValueGap:
    kind: str           # 'bullish' | 'bearish'
    top: float          # upper edge of gap
    bottom: float       # lower edge of gap
    midpoint: float
    size: float         # gap size in $
    size_pct: float     # gap size as % of price
    date: str           # candle date that created the gap
    mitigated: bool     # has price already filled it?
    partial_fill: float # 0-1, how much has been filled
    strength: str       # 'strong' | 'medium' | 'weak' based on volume


@dataclass
class OrderBlock:
    kind: str           # 'bullish' | 'bearish'
    top: float
    bottom: float
    midpoint: float
    date: str
    mitigated: bool
    respected: int      # how many times price bounced from it
    volume: float       # volume on the OB candle
    strength: str       # 'strong' | 'medium' | 'weak'


@dataclass
class BreakerBlock:
    """Failed Order Block that becomes opposite-direction resistance/support."""
    kind: str           # 'bullish' | 'bearish'  (direction it NOW acts as)
    top: float
    bottom: float
    date: str
    origin: str         # was originally 'bullish' | 'bearish' OB


@dataclass
class LiquiditySweep:
    kind: str           # 'bsl_sweep' (buy-side) | 'ssl_sweep' (sell-side)
    level: float        # the liquidity level swept
    date: str
    rejection: bool     # did price reject back quickly?
    volume_spike: bool


@dataclass
class MarketStructure:
    trend: str                      # 'bullish' | 'bearish' | 'ranging'
    last_bos: Optional[str]         # 'bullish' | 'bearish'
    last_bos_level: float
    last_choch: Optional[str]       # 'bullish' | 'bearish'
    last_choch_level: float
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool
    swing_high: float
    swing_low: float


@dataclass
class PremiumDiscount:
    equilibrium: float      # 50% of the range
    premium_zone: float     # 75% - top of range (sell area)
    discount_zone: float    # 25% - bottom of range (buy area)
    current_position: str   # 'premium' | 'discount' | 'equilibrium'
    position_pct: float     # 0-100 where price sits in range
    ote_buy: Tuple[float, float]    # 0.618–0.786 from swing low
    ote_sell: Tuple[float, float]   # 0.618–0.786 from swing high


@dataclass
class FVGAnalysisResult:
    ticker: str
    current_price: float
    timestamp: str

    # Structure
    structure: MarketStructure

    # Gaps
    bullish_fvgs: List[FairValueGap]
    bearish_fvgs: List[FairValueGap]
    nearest_bullish_fvg: Optional[FairValueGap]   # closest below price
    nearest_bearish_fvg: Optional[FairValueGap]   # closest above price

    # Order Blocks
    bullish_obs: List[OrderBlock]
    bearish_obs: List[OrderBlock]
    nearest_bullish_ob: Optional[OrderBlock]
    nearest_bearish_ob: Optional[OrderBlock]

    # Breaker Blocks
    breaker_blocks: List[BreakerBlock]

    # Liquidity
    liquidity_sweeps: List[LiquiditySweep]
    buy_side_liquidity: List[float]    # equal highs, prev swing highs
    sell_side_liquidity: List[float]   # equal lows, prev swing lows

    # Zones
    premium_discount: PremiumDiscount

    # Price targets derived from all of the above
    bullish_targets: List[dict]    # [{price, reason, confidence}]
    bearish_targets: List[dict]

    # Summary
    bias: str           # 'bullish' | 'bearish' | 'neutral'
    bias_strength: str  # 'strong' | 'moderate' | 'weak'
    key_levels: List[dict]


# ─────────────────────────────────────────────
# MAIN ANALYSER
# ─────────────────────────────────────────────

class FVGAnalyser:
    """
    Detect Fair Value Gaps, Order Blocks, Market Structure, and Liquidity
    using ICT / SMC methodology on daily OHLCV data.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = ""):
        """
        df must have columns: date/datetime, open, high, low, close, volume
        """
        self.ticker = ticker
        self.df = df.copy()
        self.df.columns = [c.lower().replace(' ', '_') for c in self.df.columns]

        # Normalise date column
        date_col = next((c for c in self.df.columns if 'date' in c or 'time' in c), None)
        if date_col and date_col != 'date':
            self.df.rename(columns={date_col: 'date'}, inplace=True)
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date']).dt.strftime('%Y-%m-%d')

        self.close  = self.df['close']
        self.high   = self.df['high']
        self.low    = self.df['low']
        self.open   = self.df['open']
        self.volume = self.df['volume']
        self.n      = len(self.df)

    # ──────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────

    def analyse(self) -> FVGAnalysisResult:
        from datetime import datetime

        price = self.close.iloc[-1]

        structure   = self._market_structure()
        bull_fvgs   = self._detect_fvgs('bullish')
        bear_fvgs   = self._detect_fvgs('bearish')
        bull_obs    = self._detect_order_blocks('bullish')
        bear_obs    = self._detect_order_blocks('bearish')
        breakers    = self._detect_breaker_blocks(bull_obs, bear_obs)
        sweeps      = self._detect_liquidity_sweeps()
        bsl, ssl    = self._map_liquidity_pools()
        pd_zones    = self._premium_discount(structure)

        # nearest levels
        nearest_bull_fvg = self._nearest_below(bull_fvgs, price)
        nearest_bear_fvg = self._nearest_above(bear_fvgs, price)
        nearest_bull_ob  = self._nearest_ob_below(bull_obs, price)
        nearest_bear_ob  = self._nearest_ob_above(bear_obs, price)

        bull_targets, bear_targets = self._derive_targets(
            price, structure, bull_fvgs, bear_fvgs,
            bull_obs, bear_obs, bsl, ssl, pd_zones
        )

        key_levels = self._key_levels(
            price, structure, bull_fvgs, bear_fvgs,
            bull_obs, bear_obs, bsl, ssl, pd_zones
        )

        bias, bias_strength = self._overall_bias(structure, bull_fvgs, bear_fvgs, sweeps)

        return FVGAnalysisResult(
            ticker=self.ticker,
            current_price=round(price, 2),
            timestamp=datetime.now().isoformat(),
            structure=structure,
            bullish_fvgs=bull_fvgs,
            bearish_fvgs=bear_fvgs,
            nearest_bullish_fvg=nearest_bull_fvg,
            nearest_bearish_fvg=nearest_bear_fvg,
            bullish_obs=bull_obs,
            bearish_obs=bear_obs,
            nearest_bullish_ob=nearest_bull_ob,
            nearest_bearish_ob=nearest_bear_ob,
            breaker_blocks=breakers,
            liquidity_sweeps=sweeps,
            buy_side_liquidity=bsl,
            sell_side_liquidity=ssl,
            premium_discount=pd_zones,
            bullish_targets=bull_targets,
            bearish_targets=bear_targets,
            bias=bias,
            bias_strength=bias_strength,
            key_levels=key_levels,
        )

    # ──────────────────────────────────────────
    # FAIR VALUE GAPS
    # ──────────────────────────────────────────

    def _detect_fvgs(self, kind: str, lookback: int = 60) -> List[FairValueGap]:
        """
        Bullish FVG : candle[i-1].high < candle[i+1].low  (gap up)
        Bearish FVG : candle[i-1].low  > candle[i+1].high (gap down)
        """
        fvgs = []
        start = max(1, self.n - lookback)

        for i in range(start, self.n - 1):
            date = self.df['date'].iloc[i] if 'date' in self.df.columns else str(i)
            vol  = self.volume.iloc[i]
            avg_vol = self.volume.iloc[max(0, i-20):i].mean()

            if kind == 'bullish':
                gap_bottom = self.high.iloc[i - 1]
                gap_top    = self.low.iloc[i + 1]
                if gap_top > gap_bottom:
                    size     = gap_top - gap_bottom
                    size_pct = size / self.close.iloc[i] * 100
                    mid      = (gap_top + gap_bottom) / 2

                    # Check mitigation (has price traded into gap since?)
                    future_lows = self.low.iloc[i + 1:]
                    mitigated   = bool((future_lows <= gap_top).any())
                    # Partial fill: how deep into gap did price go?
                    if mitigated:
                        deepest = future_lows[future_lows <= gap_top].min()
                        filled  = min(1.0, (gap_top - deepest) / (size + 1e-9))
                    else:
                        filled = 0.0

                    strength = 'strong' if vol > avg_vol * 1.5 else 'medium' if vol > avg_vol else 'weak'
                    fvgs.append(FairValueGap(
                        kind='bullish', top=round(gap_top, 2), bottom=round(gap_bottom, 2),
                        midpoint=round(mid, 2), size=round(size, 2),
                        size_pct=round(size_pct, 2), date=date,
                        mitigated=mitigated, partial_fill=round(filled, 2),
                        strength=strength
                    ))

            else:  # bearish
                gap_top    = self.low.iloc[i - 1]
                gap_bottom = self.high.iloc[i + 1]
                if gap_top > gap_bottom:
                    size     = gap_top - gap_bottom
                    size_pct = size / self.close.iloc[i] * 100
                    mid      = (gap_top + gap_bottom) / 2

                    future_highs = self.high.iloc[i + 1:]
                    mitigated    = bool((future_highs >= gap_bottom).any())
                    if mitigated:
                        highest = future_highs[future_highs >= gap_bottom].max()
                        filled  = min(1.0, (highest - gap_bottom) / (size + 1e-9))
                    else:
                        filled = 0.0

                    strength = 'strong' if vol > avg_vol * 1.5 else 'medium' if vol > avg_vol else 'weak'
                    fvgs.append(FairValueGap(
                        kind='bearish', top=round(gap_top, 2), bottom=round(gap_bottom, 2),
                        midpoint=round(mid, 2), size=round(size, 2),
                        size_pct=round(size_pct, 2), date=date,
                        mitigated=mitigated, partial_fill=round(filled, 2),
                        strength=strength
                    ))

        # Sort by proximity to current price, most recent unmitigated first
        price = self.close.iloc[-1]
        unmitigated = [f for f in fvgs if not f.mitigated]
        unmitigated.sort(key=lambda f: abs(f.midpoint - price))
        mitigated   = [f for f in fvgs if f.mitigated]
        return unmitigated + mitigated

    # ──────────────────────────────────────────
    # ORDER BLOCKS
    # ──────────────────────────────────────────

    def _detect_order_blocks(self, kind: str, lookback: int = 60) -> List[OrderBlock]:
        """
        Bullish OB  : last bearish (red) candle before a strong bullish impulse move.
        Bearish OB  : last bullish (green) candle before a strong bearish impulse move.
        Impulse defined as the next candle body > 1.5× ATR.
        """
        obs = []
        atr = self._atr(14)
        start = max(1, self.n - lookback)

        for i in range(start, self.n - 1):
            date   = self.df['date'].iloc[i] if 'date' in self.df.columns else str(i)
            candle_range = self.high.iloc[i] - self.low.iloc[i]
            next_body    = abs(self.close.iloc[i + 1] - self.open.iloc[i + 1])
            current_atr  = atr.iloc[i] if not np.isnan(atr.iloc[i]) else candle_range

            is_impulse = next_body > 1.5 * current_atr

            if not is_impulse:
                continue

            vol     = self.volume.iloc[i]
            avg_vol = self.volume.iloc[max(0, i - 20):i].mean()
            strength = 'strong' if vol > avg_vol * 1.5 else 'medium' if vol > avg_vol else 'weak'

            if kind == 'bullish':
                # Last red candle before bullish impulse
                if (self.close.iloc[i] < self.open.iloc[i] and
                        self.close.iloc[i + 1] > self.open.iloc[i + 1]):
                    top    = self.high.iloc[i]
                    bottom = self.low.iloc[i]
                    mid    = (top + bottom) / 2

                    # Mitigation: price trades into the OB from above → it's "used"
                    future_lows = self.low.iloc[i + 1:]
                    mitigated   = bool((future_lows <= top).any())
                    respected   = int((future_lows[future_lows > bottom]).count()) if not mitigated else 0

                    obs.append(OrderBlock(
                        kind='bullish', top=round(top, 2), bottom=round(bottom, 2),
                        midpoint=round(mid, 2), date=date,
                        mitigated=mitigated, respected=respected,
                        volume=round(vol, 0), strength=strength
                    ))

            else:  # bearish
                # Last green candle before bearish impulse
                if (self.close.iloc[i] > self.open.iloc[i] and
                        self.close.iloc[i + 1] < self.open.iloc[i + 1]):
                    top    = self.high.iloc[i]
                    bottom = self.low.iloc[i]
                    mid    = (top + bottom) / 2

                    future_highs = self.high.iloc[i + 1:]
                    mitigated    = bool((future_highs >= bottom).any())
                    respected    = int((future_highs[future_highs < top]).count()) if not mitigated else 0

                    obs.append(OrderBlock(
                        kind='bearish', top=round(top, 2), bottom=round(bottom, 2),
                        midpoint=round(mid, 2), date=date,
                        mitigated=mitigated, respected=respected,
                        volume=round(vol, 0), strength=strength
                    ))

        price = self.close.iloc[-1]
        unmitigated = [o for o in obs if not o.mitigated]
        unmitigated.sort(key=lambda o: abs(o.midpoint - price))
        return unmitigated + [o for o in obs if o.mitigated]

    # ──────────────────────────────────────────
    # BREAKER BLOCKS
    # ──────────────────────────────────────────

    def _detect_breaker_blocks(
        self, bull_obs: List[OrderBlock], bear_obs: List[OrderBlock]
    ) -> List[BreakerBlock]:
        """
        A mitigated bullish OB that failed (price broke below it) becomes a
        bearish Breaker Block (now acts as resistance).
        Vice-versa for mitigated bearish OBs.
        """
        breakers = []
        price = self.close.iloc[-1]

        for ob in bull_obs:
            if ob.mitigated and price < ob.bottom:
                breakers.append(BreakerBlock(
                    kind='bearish', top=ob.top, bottom=ob.bottom,
                    date=ob.date, origin='bullish'
                ))

        for ob in bear_obs:
            if ob.mitigated and price > ob.top:
                breakers.append(BreakerBlock(
                    kind='bullish', top=ob.top, bottom=ob.bottom,
                    date=ob.date, origin='bearish'
                ))

        return breakers

    # ──────────────────────────────────────────
    # MARKET STRUCTURE  (BOS / ChoCH)
    # ──────────────────────────────────────────

    def _market_structure(self, swing_period: int = 5) -> MarketStructure:
        """
        Identify swing highs / lows and determine:
        - HH / HL  → bullish structure
        - LH / LL  → bearish structure
        - BOS      → confirmed break of structure
        - ChoCH    → first sign of potential reversal
        """
        swing_highs = []
        swing_lows  = []

        for i in range(swing_period, self.n - swing_period):
            if self.high.iloc[i] == self.high.iloc[i - swing_period:i + swing_period + 1].max():
                swing_highs.append((i, self.high.iloc[i]))
            if self.low.iloc[i] == self.low.iloc[i - swing_period:i + swing_period + 1].min():
                swing_lows.append((i, self.low.iloc[i]))

        # Need at least 2 swing highs and lows for structure analysis
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketStructure(
                trend='ranging', last_bos=None, last_bos_level=0,
                last_choch=None, last_choch_level=0,
                higher_highs=False, higher_lows=False,
                lower_highs=False, lower_lows=False,
                swing_high=self.high.max(),
                swing_low=self.low.min()
            )

        last_sh = swing_highs[-1][1]
        prev_sh = swing_highs[-2][1]
        last_sl = swing_lows[-1][1]
        prev_sl = swing_lows[-2][1]

        higher_highs = last_sh > prev_sh
        higher_lows  = last_sl > prev_sl
        lower_highs  = last_sh < prev_sh
        lower_lows   = last_sl < prev_sl

        # Determine trend
        if higher_highs and higher_lows:
            trend = 'bullish'
        elif lower_highs and lower_lows:
            trend = 'bearish'
        else:
            trend = 'ranging'

        # BOS: price closes beyond last swing high/low
        price = self.close.iloc[-1]
        last_bos = None
        last_bos_level = 0.0
        if price > last_sh:
            last_bos = 'bullish'
            last_bos_level = last_sh
        elif price < last_sl:
            last_bos = 'bearish'
            last_bos_level = last_sl

        # ChoCH: in a downtrend a higher high forms; in uptrend a lower low forms
        last_choch = None
        last_choch_level = 0.0
        if trend == 'bearish' and higher_highs:
            last_choch = 'bullish'
            last_choch_level = last_sh
        elif trend == 'bullish' and lower_lows:
            last_choch = 'bearish'
            last_choch_level = last_sl

        return MarketStructure(
            trend=trend,
            last_bos=last_bos,
            last_bos_level=round(last_bos_level, 2),
            last_choch=last_choch,
            last_choch_level=round(last_choch_level, 2),
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            swing_high=round(float(last_sh), 2),
            swing_low=round(float(last_sl), 2),
        )

    # ──────────────────────────────────────────
    # LIQUIDITY SWEEPS
    # ──────────────────────────────────────────

    def _detect_liquidity_sweeps(self, lookback: int = 40) -> List[LiquiditySweep]:
        """
        A liquidity sweep occurs when price briefly trades beyond a recent
        swing high/low (stop-hunt) before reversing sharply.
        """
        sweeps = []
        start  = max(10, self.n - lookback)

        for i in range(start, self.n):
            date = self.df['date'].iloc[i] if 'date' in self.df.columns else str(i)

            # Look at prior swing high/low within 20 bars
            window_high = self.high.iloc[max(0, i - 20):i].max()
            window_low  = self.low.iloc[max(0, i - 20):i].min()

            vol     = self.volume.iloc[i]
            avg_vol = self.volume.iloc[max(0, i - 20):i].mean()
            vol_spike = bool(vol > avg_vol * 1.5)

            # Buy-side liquidity sweep: wick above prior swing high, close back below
            if self.high.iloc[i] > window_high and self.close.iloc[i] < window_high:
                sweeps.append(LiquiditySweep(
                    kind='bsl_sweep', level=round(window_high, 2),
                    date=date, rejection=True, volume_spike=vol_spike
                ))

            # Sell-side liquidity sweep: wick below prior swing low, close back above
            if self.low.iloc[i] < window_low and self.close.iloc[i] > window_low:
                sweeps.append(LiquiditySweep(
                    kind='ssl_sweep', level=round(window_low, 2),
                    date=date, rejection=True, volume_spike=vol_spike
                ))

        return sweeps[-10:]  # last 10 sweeps

    # ──────────────────────────────────────────
    # LIQUIDITY POOLS (Equal Highs / Lows)
    # ──────────────────────────────────────────

    def _map_liquidity_pools(self, lookback: int = 60, tolerance: float = 0.003
                             ) -> Tuple[List[float], List[float]]:
        """
        Identify clusters of equal highs (buy-side liquidity above market)
        and equal lows (sell-side liquidity below market).
        Clusters within `tolerance` % of each other are merged.
        """
        price  = self.close.iloc[-1]
        recent = self.df.iloc[-lookback:]

        raw_highs = recent['high'].tolist()
        raw_lows  = recent['low'].tolist()

        def cluster(vals, above_price: bool):
            vals = sorted(set(vals))
            clusters = []
            buf = [vals[0]] if vals else []
            for v in vals[1:]:
                if abs(v - buf[-1]) / buf[-1] < tolerance:
                    buf.append(v)
                else:
                    clusters.append(round(np.mean(buf), 2))
                    buf = [v]
            if buf:
                clusters.append(round(np.mean(buf), 2))
            return [c for c in clusters if (c > price) == above_price]

        bsl = cluster(raw_highs, above_price=True)   # buy-side above price
        ssl = cluster(raw_lows,  above_price=False)  # sell-side below price
        return sorted(bsl), sorted(ssl, reverse=True)

    # ──────────────────────────────────────────
    # PREMIUM / DISCOUNT  +  OTE
    # ──────────────────────────────────────────

    def _premium_discount(self, structure: MarketStructure) -> PremiumDiscount:
        """
        Range = swing_low → swing_high.
        Discount zone  : below 50 % (buy opportunities)
        Premium zone   : above 50 % (sell opportunities)
        OTE (Optimal Trade Entry):
          - Long  : 61.8 % – 78.6 % retracement from swing_low (Fibonacci)
          - Short : 61.8 % – 78.6 % retracement from swing_high
        """
        sh    = structure.swing_high
        sl    = structure.swing_low
        rng   = sh - sl if sh > sl else 1
        price = self.close.iloc[-1]

        eq           = sl + rng * 0.50
        premium_line = sl + rng * 0.75
        discount_line = sl + rng * 0.25

        pos_pct = (price - sl) / rng * 100

        if price > premium_line:
            position = 'premium'
        elif price < discount_line:
            position = 'discount'
        else:
            position = 'equilibrium'

        # OTE long (buy): deep retracement into discount – 61.8% to 78.6% retrace
        ote_buy_top    = sh - rng * 0.618
        ote_buy_bottom = sh - rng * 0.786

        # OTE short (sell): shallow retracement into premium – 61.8% to 78.6% of rally
        ote_sell_bottom = sl + rng * 0.618
        ote_sell_top    = sl + rng * 0.786

        return PremiumDiscount(
            equilibrium=round(eq, 2),
            premium_zone=round(premium_line, 2),
            discount_zone=round(discount_line, 2),
            current_position=position,
            position_pct=round(pos_pct, 1),
            ote_buy=(round(ote_buy_bottom, 2), round(ote_buy_top, 2)),
            ote_sell=(round(ote_sell_bottom, 2), round(ote_sell_top, 2)),
        )

    # ──────────────────────────────────────────
    # PRICE TARGET DERIVATION
    # ──────────────────────────────────────────

    def _derive_targets(
        self, price, structure, bull_fvgs, bear_fvgs,
        bull_obs, bear_obs, bsl, ssl, pd_zones
    ):
        bull_targets = []
        bear_targets = []

        # ---- BULLISH TARGETS (price goes up) ----

        # 1. Nearest unmitigated Bearish FVG above (price will fill it)
        for fvg in [f for f in bear_fvgs if not f.mitigated and f.bottom > price]:
            bull_targets.append({
                'price': fvg.bottom,
                'reason': f'Bearish FVG fill (gap {fvg.bottom}–{fvg.top})',
                'confidence': 'high' if fvg.strength == 'strong' else 'medium',
                'type': 'fvg'
            })
            break  # nearest only

        # 2. Buy-side liquidity pools above
        for level in sorted(bsl)[:3]:
            if level > price:
                bull_targets.append({
                    'price': level,
                    'reason': f'Buy-side liquidity sweep target',
                    'confidence': 'medium',
                    'type': 'liquidity'
                })

        # 3. Nearest Bearish OB above (will act as magnet)
        for ob in [o for o in bear_obs if not o.mitigated and o.bottom > price]:
            bull_targets.append({
                'price': ob.bottom,
                'reason': f'Bearish OB above (mitigated target)',
                'confidence': 'high' if ob.strength == 'strong' else 'medium',
                'type': 'order_block'
            })
            break

        # 4. OTE sell zone (price exhaustion / reversal area)
        if pd_zones.ote_sell[1] > price:
            bull_targets.append({
                'price': pd_zones.ote_sell[1],
                'reason': f'OTE sell zone top (premium exhaustion)',
                'confidence': 'medium',
                'type': 'ote'
            })

        # 5. Swing high
        if structure.swing_high > price:
            bull_targets.append({
                'price': structure.swing_high,
                'reason': 'Prior swing high (liquidity magnet)',
                'confidence': 'medium',
                'type': 'structure'
            })

        # ---- BEARISH TARGETS (price goes down) ----

        # 1. Nearest unmitigated Bullish FVG below (price will fill it)
        for fvg in sorted([f for f in bull_fvgs if not f.mitigated and f.top < price],
                          key=lambda f: f.top, reverse=True):
            bear_targets.append({
                'price': fvg.top,
                'reason': f'Bullish FVG fill (gap {fvg.bottom}–{fvg.top})',
                'confidence': 'high' if fvg.strength == 'strong' else 'medium',
                'type': 'fvg'
            })
            break

        # 2. Sell-side liquidity pools below
        for level in sorted(ssl, reverse=True)[:3]:
            if level < price:
                bear_targets.append({
                    'price': level,
                    'reason': f'Sell-side liquidity sweep target',
                    'confidence': 'medium',
                    'type': 'liquidity'
                })

        # 3. Nearest Bullish OB below
        for ob in sorted([o for o in bull_obs if not o.mitigated and o.top < price],
                         key=lambda o: o.top, reverse=True):
            bear_targets.append({
                'price': ob.top,
                'reason': f'Bullish OB below (support target)',
                'confidence': 'high' if ob.strength == 'strong' else 'medium',
                'type': 'order_block'
            })
            break

        # 4. OTE buy zone
        if pd_zones.ote_buy[0] < price:
            bear_targets.append({
                'price': pd_zones.ote_buy[0],
                'reason': 'OTE buy zone (discount / reversal area)',
                'confidence': 'medium',
                'type': 'ote'
            })

        # 5. Swing low
        if structure.swing_low < price:
            bear_targets.append({
                'price': structure.swing_low,
                'reason': 'Prior swing low (liquidity below)',
                'confidence': 'medium',
                'type': 'structure'
            })

        # Sort by price
        bull_targets.sort(key=lambda x: x['price'])
        bear_targets.sort(key=lambda x: x['price'], reverse=True)
        return bull_targets[:5], bear_targets[:5]

    # ──────────────────────────────────────────
    # KEY LEVELS SUMMARY
    # ──────────────────────────────────────────

    def _key_levels(self, price, structure, bull_fvgs, bear_fvgs,
                    bull_obs, bear_obs, bsl, ssl, pd_zones) -> List[dict]:
        levels = []

        def add(p, label, kind):
            levels.append({'price': round(p, 2), 'label': label, 'kind': kind,
                           'distance_pct': round((p / price - 1) * 100, 2)})

        add(pd_zones.equilibrium,    '50% Equilibrium',           'zone')
        add(pd_zones.premium_zone,   'Premium Zone (75%)',         'zone')
        add(pd_zones.discount_zone,  'Discount Zone (25%)',        'zone')
        add(pd_zones.ote_buy[0],     'OTE Buy Lower (78.6%)',      'fibonacci')
        add(pd_zones.ote_buy[1],     'OTE Buy Upper (61.8%)',      'fibonacci')
        add(pd_zones.ote_sell[0],    'OTE Sell Lower (61.8%)',     'fibonacci')
        add(pd_zones.ote_sell[1],    'OTE Sell Upper (78.6%)',     'fibonacci')
        add(structure.swing_high,    'Swing High',                 'structure')
        add(structure.swing_low,     'Swing Low',                  'structure')

        for fvg in bull_fvgs[:2]:
            if not fvg.mitigated:
                add(fvg.midpoint, f'Bullish FVG Mid ({fvg.date})', 'fvg')
        for fvg in bear_fvgs[:2]:
            if not fvg.mitigated:
                add(fvg.midpoint, f'Bearish FVG Mid ({fvg.date})', 'fvg')

        for ob in bull_obs[:2]:
            if not ob.mitigated:
                add(ob.midpoint, f'Bullish OB ({ob.date})', 'order_block')
        for ob in bear_obs[:2]:
            if not ob.mitigated:
                add(ob.midpoint, f'Bearish OB ({ob.date})', 'order_block')

        for liq in bsl[:2]:
            add(liq, 'Buy-Side Liquidity', 'liquidity')
        for liq in ssl[:2]:
            add(liq, 'Sell-Side Liquidity', 'liquidity')

        levels.sort(key=lambda x: x['price'])
        return levels

    # ──────────────────────────────────────────
    # OVERALL BIAS
    # ──────────────────────────────────────────

    def _overall_bias(self, structure, bull_fvgs, bear_fvgs, sweeps):
        score = 0

        # Structure
        if structure.trend == 'bullish':
            score += 2
        elif structure.trend == 'bearish':
            score -= 2

        if structure.last_bos == 'bullish':
            score += 1
        elif structure.last_bos == 'bearish':
            score -= 1

        if structure.last_choch == 'bullish':
            score += 1
        elif structure.last_choch == 'bearish':
            score -= 1

        # FVG proximity (unmitigated gaps nearby)
        price = self.close.iloc[-1]
        near_bull_fvg = any(f for f in bull_fvgs if not f.mitigated and abs(f.midpoint - price) / price < 0.05)
        near_bear_fvg = any(f for f in bear_fvgs if not f.mitigated and abs(f.midpoint - price) / price < 0.05)
        if near_bull_fvg:
            score += 1
        if near_bear_fvg:
            score -= 1

        # Recent liquidity sweeps
        for sw in sweeps[-3:]:
            if sw.kind == 'ssl_sweep' and sw.rejection:
                score += 1   # swept lows and rejected → bullish
            elif sw.kind == 'bsl_sweep' and sw.rejection:
                score -= 1   # swept highs and rejected → bearish

        if score >= 3:
            return 'bullish', 'strong'
        elif score >= 1:
            return 'bullish', 'moderate'
        elif score <= -3:
            return 'bearish', 'strong'
        elif score <= -1:
            return 'bearish', 'moderate'
        else:
            return 'neutral', 'weak'

    # ──────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────

    def _atr(self, period: int = 14) -> pd.Series:
        tr = pd.concat([
            self.high - self.low,
            (self.high - self.close.shift()).abs(),
            (self.low  - self.close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period).mean()

    def _nearest_below(self, fvgs: List[FairValueGap], price: float) -> Optional[FairValueGap]:
        candidates = [f for f in fvgs if not f.mitigated and f.top < price]
        return max(candidates, key=lambda f: f.top) if candidates else None

    def _nearest_above(self, fvgs: List[FairValueGap], price: float) -> Optional[FairValueGap]:
        candidates = [f for f in fvgs if not f.mitigated and f.bottom > price]
        return min(candidates, key=lambda f: f.bottom) if candidates else None

    def _nearest_ob_below(self, obs: List[OrderBlock], price: float) -> Optional[OrderBlock]:
        candidates = [o for o in obs if not o.mitigated and o.top < price]
        return max(candidates, key=lambda o: o.top) if candidates else None

    def _nearest_ob_above(self, obs: List[OrderBlock], price: float) -> Optional[OrderBlock]:
        candidates = [o for o in obs if not o.mitigated and o.bottom > price]
        return min(candidates, key=lambda o: o.bottom) if candidates else None


# ─────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────

def print_fvg_report(result: FVGAnalysisResult):
    p = result.current_price
    s = result.structure
    pd_z = result.premium_discount

    print(f"\n{'='*70}")
    print(f"  ICT / FVG ANALYSIS  —  {result.ticker}  @  ${p}")
    print(f"{'='*70}")

    # Market Structure
    trend_icon = "📈" if s.trend == 'bullish' else "📉" if s.trend == 'bearish' else "↔️"
    print(f"\n🏗  MARKET STRUCTURE  [{trend_icon} {s.trend.upper()}]")
    print(f"   Swing High : ${s.swing_high}   Swing Low : ${s.swing_low}")
    print(f"   HH: {s.higher_highs}  HL: {s.higher_lows}  LH: {s.lower_highs}  LL: {s.lower_lows}")
    if s.last_bos:
        print(f"   BOS        : {s.last_bos.upper()} @ ${s.last_bos_level}")
    if s.last_choch:
        print(f"   ChoCH      : {s.last_choch.upper()} @ ${s.last_choch_level}")

    # Premium / Discount
    pos_icon = "🔴" if pd_z.current_position == 'premium' else "🟢" if pd_z.current_position == 'discount' else "🟡"
    print(f"\n📐 PREMIUM / DISCOUNT  [{pos_icon} {pd_z.current_position.upper()} — {pd_z.position_pct:.1f}% of range]")
    print(f"   Premium Zone  : ${pd_z.premium_zone}+")
    print(f"   Equilibrium   : ${pd_z.equilibrium}")
    print(f"   Discount Zone : <${pd_z.discount_zone}")
    print(f"   OTE Buy       : ${pd_z.ote_buy[0]} – ${pd_z.ote_buy[1]}")
    print(f"   OTE Sell      : ${pd_z.ote_sell[0]} – ${pd_z.ote_sell[1]}")

    # FVGs
    unm_bull = [f for f in result.bullish_fvgs if not f.mitigated]
    unm_bear = [f for f in result.bearish_fvgs if not f.mitigated]
    print(f"\n⬜ FAIR VALUE GAPS  (unmitigated: 🟢{len(unm_bull)} bullish  🔴{len(unm_bear)} bearish)")
    if result.nearest_bullish_fvg:
        f = result.nearest_bullish_fvg
        print(f"   Nearest Bullish FVG : ${f.bottom} – ${f.top}  (mid ${f.midpoint})  [{f.strength}]  {f.date}")
    if result.nearest_bearish_fvg:
        f = result.nearest_bearish_fvg
        print(f"   Nearest Bearish FVG : ${f.bottom} – ${f.top}  (mid ${f.midpoint})  [{f.strength}]  {f.date}")

    # Order Blocks
    unm_bull_ob = [o for o in result.bullish_obs if not o.mitigated]
    unm_bear_ob = [o for o in result.bearish_obs if not o.mitigated]
    print(f"\n📦 ORDER BLOCKS  (unmitigated: 🟢{len(unm_bull_ob)} bullish  🔴{len(unm_bear_ob)} bearish)")
    if result.nearest_bullish_ob:
        o = result.nearest_bullish_ob
        print(f"   Nearest Bullish OB  : ${o.bottom} – ${o.top}  [{o.strength}]  {o.date}")
    if result.nearest_bearish_ob:
        o = result.nearest_bearish_ob
        print(f"   Nearest Bearish OB  : ${o.bottom} – ${o.top}  [{o.strength}]  {o.date}")

    # Breakers
    if result.breaker_blocks:
        print(f"\n💥 BREAKER BLOCKS  ({len(result.breaker_blocks)})")
        for b in result.breaker_blocks[:3]:
            print(f"   {b.kind.upper()} breaker : ${b.bottom} – ${b.top}  (was {b.origin} OB)  {b.date}")

    # Liquidity
    print(f"\n💧 LIQUIDITY POOLS")
    print(f"   Buy-Side  (above) : {[str(l) for l in result.buy_side_liquidity[:4]]}")
    print(f"   Sell-Side (below) : {[str(l) for l in result.sell_side_liquidity[:4]]}")
    if result.liquidity_sweeps:
        last = result.liquidity_sweeps[-1]
        print(f"   Last Sweep        : {last.kind.upper()} @ ${last.level}  {last.date}  vol-spike={last.volume_spike}")

    # Price Targets
    bias_icon = "🟢" if result.bias == 'bullish' else "🔴" if result.bias == 'bearish' else "🟡"
    print(f"\n🎯 PRICE TARGETS  [{bias_icon} {result.bias.upper()} — {result.bias_strength.upper()} BIAS]")
    print(f"   Bullish Targets ↑")
    for t in result.bullish_targets:
        dist = (t['price'] / p - 1) * 100
        print(f"     ${t['price']:>10}  ({dist:+.1f}%)  [{t['confidence']}]  {t['reason']}")
    print(f"   Bearish Targets ↓")
    for t in result.bearish_targets:
        dist = (t['price'] / p - 1) * 100
        print(f"     ${t['price']:>10}  ({dist:+.1f}%)  [{t['confidence']}]  {t['reason']}")

    # Key levels
    print(f"\n🗺  KEY LEVELS")
    above = [l for l in result.key_levels if l['price'] > p]
    below = [l for l in result.key_levels if l['price'] <= p]
    print(f"   ABOVE price:")
    for l in sorted(above, key=lambda x: x['price'])[:6]:
        print(f"     ${l['price']:>10}  ({l['distance_pct']:+.1f}%)  {l['label']}")
    print(f"   BELOW price:")
    for l in sorted(below, key=lambda x: x['price'], reverse=True)[:6]:
        print(f"     ${l['price']:>10}  ({l['distance_pct']:+.1f}%)  {l['label']}")

    print(f"\n{'='*70}")


# ─────────────────────────────────────────────
# STANDALONE RUNNER
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import json
    import yfinance as yf

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'IONQ'
    print(f"Fetching data for {ticker}...")

    stock = yf.Ticker(ticker)
    df    = stock.history(period='1y', interval='1d').reset_index()

    analyser = FVGAnalyser(df, ticker=ticker)
    result   = analyser.analyse()
    print_fvg_report(result)

    # Save JSON
    from pathlib import Path
    from dataclasses import asdict
    out_dir = Path('predictor/data')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{ticker}_fvg_analysis.json'

    def _serial(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        raise TypeError(f'Not serializable: {type(obj)}')

    with open(out_file, 'w') as fh:
        json.dump(asdict(result), fh, indent=2, default=_serial)
    print(f"\nSaved → {out_file}")
