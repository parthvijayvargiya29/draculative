"""
TC-06: VWAP Deviation Mean Reversion
=====================================

LOGIC:
------
1. Entry when price > 2% above VWAP (SHORT) or < 2% below VWAP (LONG)
2. Gates: RSI < 40 for LONG, RSI > 60 for SHORT, ADX < 30 (not trending)
3. Exit: Return to VWAP or 1.5x ATR SL

REGIME FIT: CORRECTIVE (1.0x), RANGING (1.0x)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path

from technicals.base_signal import *
from technical.indicators_v4 import compute_rsi, compute_adx, compute_atr


class TC06_VWAPReversion(TechnicalSignal):
    TC_ID = "TC-06"
    TC_NAME = "VWAP Deviation Mean Reversion"
    TC_CATEGORY = TCCategory.MEAN_REVERSION
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        
        self.params = all_params['tc_06']
        self.vwap_deviation_pct = self.params['vwap_deviation_pct']
        self.rsi_long_max = self.params['rsi_long_max']
        self.rsi_short_min = self.params['rsi_short_min']
        self.adx_max = self.params['adx_max']
        self.sl_atr_mult = self.params['sl_atr_mult']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        
        # Calculate VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['rsi'] = compute_rsi(df, period=14)
        adx_df = compute_adx(df, period=14)
        df['adx'] = adx_df['adx']
        df['atr'] = compute_atr(df, period=14)
        
        current = df.iloc[-1]
        
        if pd.isna([current['vwap'], current['rsi'], current['adx'], current['atr']]).any():
            return None
        
        deviation_pct = (current['close'] - current['vwap']) / current['vwap']
        
        # LONG: Price < VWAP - 2%
        if deviation_pct < -self.vwap_deviation_pct:
            if current['rsi'] < self.rsi_long_max and current['adx'] < self.adx_max:
                sl = current['close'] - self.sl_atr_mult * current['atr']
                tp = current['vwap']
                
                return Signal(
                    direction=Direction.LONG,
                    strength=SignalStrength.MODERATE,
                    confidence=0.70,
                    entry_price=current['close'],
                    stop_loss=sl,
                    take_profit=tp,
                    metadata={'tc_id': self.TC_ID, 'deviation_pct': deviation_pct}
                )
        
        # SHORT: Price > VWAP + 2%
        elif deviation_pct > self.vwap_deviation_pct:
            if current['rsi'] > self.rsi_short_min and current['adx'] < self.adx_max:
                sl = current['close'] + self.sl_atr_mult * current['atr']
                tp = current['vwap']
                
                return Signal(
                    direction=Direction.SHORT,
                    strength=SignalStrength.MODERATE,
                    confidence=0.70,
                    entry_price=current['close'],
                    stop_loss=sl,
                    take_profit=tp,
                    metadata={'tc_id': self.TC_ID, 'deviation_pct': deviation_pct}
                )
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(
            tc_id=self.TC_ID,
            validation_date=datetime.now(),
            metrics={},
            gates_passed=[],
            gates_failed=[],
            approved=True,
            notes="Validation placeholder"
        )


if __name__ == "__main__":
    print("TC-06 VWAP Reversion validation placeholder")
