"""
TC-05: Order Block Entry
=========================

LOGIC:
------
1. Detect Order Block (OB): Last bullish/bearish candle before strong impulse move
   - Impulse = candle body > 1.5x ATR(14)
   - OB = previous candle's high/low zone
2. Wait for price to return to OB zone (within 0.1% buffer)
3. Entry on rejection (wick through OB + close back inside)
4. SL beyond OB far edge, TP = 2x risk

REGIME FIT: TRENDING (1.0x), CORRECTIVE (0.7x)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path

from technicals.base_signal import *
from technical.indicators_v4 import compute_atr


@dataclass
class OrderBlock:
    direction: str
    ob_high: float
    ob_low: float
    impulse_bar_idx: int


class TC05_OrderBlock(TechnicalSignal):
    TC_ID = "TC-05"
    TC_NAME = "Order Block Entry"
    TC_CATEGORY = TCCategory.ORDERFLOW
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        
        self.params = all_params['tc_05']
        self.impulse_atr_mult = self.params['impulse_atr_mult']
        self.lookback_bars = self.params['lookback_bars']
        self.buffer_pct = self.params['buffer_pct']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
        
        self.active_obs = []
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        df['atr'] = compute_atr(df, period=14)
        df['body'] = abs(df['close'] - df['open'])
        
        # Detect new OBs
        if len(df) >= 2:
            last_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            # Bullish impulse
            if last_bar['close'] > last_bar['open'] and last_bar['body'] > self.impulse_atr_mult * last_bar['atr']:
                ob = OrderBlock(
                    direction="BULLISH",
                    ob_high=prev_bar['high'],
                    ob_low=prev_bar['low'],
                    impulse_bar_idx=len(df) - 1
                )
                self.active_obs.append(ob)
            
            # Bearish impulse
            elif last_bar['close'] < last_bar['open'] and last_bar['body'] > self.impulse_atr_mult * last_bar['atr']:
                ob = OrderBlock(
                    direction="BEARISH",
                    ob_high=prev_bar['high'],
                    ob_low=prev_bar['low'],
                    impulse_bar_idx=len(df) - 1
                )
                self.active_obs.append(ob)
        
        # Check for entry
        current = df.iloc[-1]
        
        for ob in self.active_obs[:]:
            bars_since = len(df) - 1 - ob.impulse_bar_idx
            if bars_since > self.lookback_bars:
                self.active_obs.remove(ob)
                continue
            
            if ob.direction == "BULLISH":
                ob_mid = (ob.ob_high + ob.ob_low) / 2
                touched_ob = current['low'] <= ob.ob_high * (1 + self.buffer_pct)
                closed_above = current['close'] > ob_mid
                
                if touched_ob and closed_above:
                    sl = ob.ob_low - (ob.ob_high - ob.ob_low) * 0.2
                    risk = current['close'] - sl
                    tp = current['close'] + 2 * risk
                    
                    signal = Signal(
                        direction=Direction.LONG,
                        strength=SignalStrength.MODERATE,
                        confidence=0.68,
                        entry_price=current['close'],
                        stop_loss=sl,
                        take_profit=tp,
                        metadata={'tc_id': self.TC_ID, 'ob_type': 'BULLISH'}
                    )
                    self.active_obs.remove(ob)
                    return signal
            
            else:  # BEARISH
                ob_mid = (ob.ob_high + ob.ob_low) / 2
                touched_ob = current['high'] >= ob.ob_low * (1 - self.buffer_pct)
                closed_below = current['close'] < ob_mid
                
                if touched_ob and closed_below:
                    sl = ob.ob_high + (ob.ob_high - ob.ob_low) * 0.2
                    risk = sl - current['close']
                    tp = current['close'] - 2 * risk
                    
                    signal = Signal(
                        direction=Direction.SHORT,
                        strength=SignalStrength.MODERATE,
                        confidence=0.68,
                        entry_price=current['close'],
                        stop_loss=sl,
                        take_profit=tp,
                        metadata={'tc_id': self.TC_ID, 'ob_type': 'BEARISH'}
                    )
                    self.active_obs.remove(ob)
                    return signal
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(
            tc_id=self.TC_ID,
            validation_date=datetime.now(),
            metrics={},
            gates_passed=[],
            gates_failed=[],
            approved=True,  # Placeholder
            notes="Validation placeholder"
        )


if __name__ == "__main__":
    print("TC-05 Order Block validation placeholder")
