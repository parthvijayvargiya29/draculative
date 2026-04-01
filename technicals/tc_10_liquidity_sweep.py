"""TC-10: Liquidity Sweep + Rejection"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_atr

class TC10_LiquiditySweep(TechnicalSignal):
    TC_ID = "TC-10"
    TC_NAME = "Liquidity Sweep + Rejection"
    TC_CATEGORY = TCCategory.ORDERFLOW
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_10']
        self.swing_lookback = self.params['swing_lookback']
        self.wick_atr_mult = self.params['wick_atr_mult']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        df['atr'] = compute_atr(df, period=14)
        
        # Find swing high/low
        lookback_window = df.iloc[-(self.swing_lookback+1):-1]
        swing_high = lookback_window['high'].max()
        swing_low = lookback_window['low'].min()
        
        current = df.iloc[-1]
        
        # Bullish sweep: Wick extends below swing low, close back above
        lower_wick = current['low'] - min(current['open'], current['close'])
        if current['low'] < swing_low and lower_wick > self.wick_atr_mult * current['atr']:
            if current['close'] > swing_low:
                sl = current['low'] - 0.5 * current['atr']
                risk = current['close'] - sl
                tp = current['close'] + 2 * risk
                return Signal(Direction.LONG, SignalStrength.MODERATE, 0.68, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'sweep_type': 'BULLISH'})
        
        # Bearish sweep: Wick extends above swing high, close back below
        upper_wick = current['high'] - max(current['open'], current['close'])
        if current['high'] > swing_high and upper_wick > self.wick_atr_mult * current['atr']:
            if current['close'] < swing_high:
                sl = current['high'] + 0.5 * current['atr']
                risk = sl - current['close']
                tp = current['close'] - 2 * risk
                return Signal(Direction.SHORT, SignalStrength.MODERATE, 0.68, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'sweep_type': 'BEARISH'})
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-10 Liquidity Sweep validation placeholder")
