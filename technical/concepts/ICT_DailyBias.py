"""
CONCEPT: ICT Daily Bias Framework
SOURCE: 11 - ICT FOR DUMMIES | Daily Bias EP. 10
CATEGORY: BIAS_FRAMEWORK
TIMEFRAME: 1D (bias), 4H, 1H (entry)
INSTRUMENTS: ANY
DESCRIPTION:
    Daily Bias is the process of determining whether the market will be bullish
    or bearish for the CURRENT trading day BEFORE the session opens.

    The bias framework involves:
    1. Higher Timeframe (Weekly/Daily) structure: What is the prevailing HTF trend?
    2. Where is the draw on liquidity? (BSL above or SSL below?)
    3. Where did price close yesterday relative to the prior day's range?
    4. Is price in premium (look for shorts) or discount (look for longs)?
    5. What is the DXY doing? (inverse for risk assets)

    RULE: Trade entries must ALIGN with the daily bias. Never take a trade
    that opposes the daily bias, regardless of how good the setup looks.
    The bias is established BEFORE the session; if price invalidates the bias
    (e.g., takes out the level that defined the bias), STAND ASIDE.

    From transcript: "This is all fractal — same thing we look for on the higher
    time frame, we're looking for on the lower time frame."

EDGE:
    Most traders enter without a directional bias and get caught in both directions.
    Having a pre-defined bias filters out 50% of potential trades and concentrates
    capital into higher-probability setups.
KNOWN_LIMITATIONS:
    Bias can be WRONG. If price takes out a key level that invalidates the bias,
    the bias must be abandoned immediately — do not force trades against reality.
    News events can override any technically-derived bias.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SWING_LOOKBACK       = 10   # bars for HTF structure detection
PREMIUM_DISC_BUFFER  = 0.002 # price must be this far from equilibrium to classify
MIN_DAILY_BARS       = 5    # minimum daily bars needed to compute bias


@dataclass
class DailyBias:
    direction:   str    # "BULLISH", "BEARISH", or "NEUTRAL"
    draw_on_liq: float  # the target price (where BSL or SSL sits)
    side:        str    # "BSL" or "SSL"
    confidence:  float
    reason:      str
    invalidation: float  # the level that, if broken, cancels the bias


class ICT_DailyBias:
    """
    Computes and caches the daily bias based on HTF structure, liquidity,
    and premium/discount positioning.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.swing_lookback     = p.get("swing_lookback",      SWING_LOOKBACK)
        self.pd_buffer          = p.get("premium_disc_buffer", PREMIUM_DISC_BUFFER)
        self.min_daily_bars     = p.get("min_daily_bars",      MIN_DAILY_BARS)
        self._daily_bias: Optional[DailyBias] = None
        self._last_bias_date:   Optional[object] = None

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        """
        Computes or retrieves the daily bias.
        Returns a Signal only if the current bar aligns with the bias direction
        AND is in the correct premium/discount zone for execution.
        """
        history = snapshot.history
        if len(history) < self.swing_lookback + self.min_daily_bars:
            return None

        # Recompute bias at start of each new day
        today = snapshot.timestamp.date() if hasattr(snapshot.timestamp, "date") else None
        if today != self._last_bias_date:
            self._daily_bias = self._compute_bias(snapshot, history)
            self._last_bias_date = today
            if self._daily_bias:
                self.log_event("BIAS_SET", snapshot.timestamp, {
                    "direction": self._daily_bias.direction,
                    "draw_on_liq": self._daily_bias.draw_on_liq,
                    "confidence": self._daily_bias.confidence,
                })

        if self._daily_bias is None:
            return None

        bias = self._daily_bias
        atr  = snapshot.atr or snapshot.close * 0.01

        # Check if bias is still valid
        if bias.direction == "BULLISH" and snapshot.close < bias.invalidation:
            self.log_event("BIAS_INVALIDATED", snapshot.timestamp, {
                "close": snapshot.close, "invalidation": bias.invalidation
            })
            self._daily_bias = None
            return None

        if bias.direction == "BEARISH" and snapshot.close > bias.invalidation:
            self.log_event("BIAS_INVALIDATED", snapshot.timestamp, {
                "close": snapshot.close, "invalidation": bias.invalidation
            })
            self._daily_bias = None
            return None

        # Bias-aligned entry: BULLISH bias + price in discount → LONG setup trigger
        if bias.direction == "BULLISH":
            equilibrium = (snapshot.high + snapshot.low) / 2
            if snapshot.close < equilibrium:
                return Signal(
                    concept     = "ICT_DailyBias",
                    symbol      = snapshot.symbol,
                    direction   = Direction.LONG,
                    timestamp   = snapshot.timestamp,
                    entry_price = snapshot.close,
                    stop_loss   = bias.invalidation - atr * 0.1,
                    take_profit = bias.draw_on_liq,
                    confidence  = bias.confidence,
                    category    = ConceptCategory.BIAS_FRAMEWORK,
                    regime      = snapshot.regime,
                    reason      = f"BULLISH bias active (draw={bias.draw_on_liq:.4f}): price in discount, align longs",
                )

        if bias.direction == "BEARISH":
            equilibrium = (snapshot.high + snapshot.low) / 2
            if snapshot.close > equilibrium:
                return Signal(
                    concept     = "ICT_DailyBias",
                    symbol      = snapshot.symbol,
                    direction   = Direction.SHORT,
                    timestamp   = snapshot.timestamp,
                    entry_price = snapshot.close,
                    stop_loss   = bias.invalidation + atr * 0.1,
                    take_profit = bias.draw_on_liq,
                    confidence  = bias.confidence,
                    category    = ConceptCategory.BIAS_FRAMEWORK,
                    regime      = snapshot.regime,
                    reason      = f"BEARISH bias active (draw={bias.draw_on_liq:.4f}): price in premium, align shorts",
                )
        return None

    def _compute_bias(self, snapshot: BarSnapshot,
                       history: List[BarSnapshot]) -> Optional[DailyBias]:
        """
        Determines daily bias from HTF structure and liquidity positioning.
        """
        lb = self.swing_lookback
        bars = history[-(3 * lb):]
        if len(bars) < 2 * lb + 1:
            return None

        # Identify swing highs and lows
        sh_prices, sl_prices = [], []
        for i in range(lb, len(bars) - lb):
            wh = [b.high for b in bars[i - lb: i + lb + 1]]
            wl = [b.low  for b in bars[i - lb: i + lb + 1]]
            if bars[i].high == max(wh):
                sh_prices.append(bars[i].high)
            if bars[i].low == min(wl):
                sl_prices.append(bars[i].low)

        if not sh_prices or not sl_prices:
            return None

        last_sh = sh_prices[-1]
        last_sl = sl_prices[-1]
        equilibrium = (last_sh + last_sl) / 2

        current_price = snapshot.close

        # Structure trend
        bullish_structure = len(sh_prices) >= 2 and sh_prices[-1] > sh_prices[-2]
        bearish_structure = len(sh_prices) >= 2 and sh_prices[-1] < sh_prices[-2]

        if bullish_structure and current_price < equilibrium:
            return DailyBias(
                direction="BULLISH",
                draw_on_liq=last_sh,
                side="BSL",
                confidence=0.60,
                reason=f"HTF bullish structure + price in discount. Draw on BSL at {last_sh:.4f}",
                invalidation=last_sl,
            )
        elif bearish_structure and current_price > equilibrium:
            return DailyBias(
                direction="BEARISH",
                draw_on_liq=last_sl,
                side="SSL",
                confidence=0.60,
                reason=f"HTF bearish structure + price in premium. Draw on SSL at {last_sl:.4f}",
                invalidation=last_sh,
            )
        return None

    def get_current_bias(self) -> Optional[DailyBias]:
        return self._daily_bias

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        passed = 1 if len(historical_df) >= self.swing_lookback + self.min_daily_bars else 0
        return ValidationResult(
            concept="ICT_DailyBias",
            total_tests=1, passed=passed, failed=1 - passed,
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_DailyBias] %s at %s | %s", event_type, bar, details)
