"""TC-11: Change of Character (ChoCH)"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_atr

class TC11_ChoCH(TechnicalSignal):
    TC_ID = "TC-11"
    TC_NAME = "Change of Character"
    TC_CATEGORY = TCCategory.STRUCTURE
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_11']
        self.fractal_bars = self.params['fractal_bars']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        df['atr'] = compute_atr(df, period=14)
        
        # Detect swing highs/lows using fractals
        swings = self._detect_swings(df)
        
        if len(swings) < 3:
            return None
        
        # ChoCH: Break of previous swing (structure shift)
        # Bullish ChoCH: Price breaks above previous swing high after lower lows
        # Bearish ChoCH: Price breaks below previous swing low after higher highs
        
        last_3_swings = swings[-3:]
        current = df.iloc[-1]
        
        # Check for bullish ChoCH
        if all(s['type'] == 'low' for s in last_3_swings[:-1]):
            if last_3_swings[-1]['type'] == 'high' and current['close'] > last_3_swings[-1]['price']:
                sl = last_3_swings[-2]['price'] - 0.5 * current['atr']
                risk = current['close'] - sl
                tp = current['close'] + 2 * risk
                return Signal(Direction.LONG, SignalStrength.MODERATE, 0.67, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'choch_type': 'BULLISH'})
        
        # Check for bearish ChoCH
        if all(s['type'] == 'high' for s in last_3_swings[:-1]):
            if last_3_swings[-1]['type'] == 'low' and current['close'] < last_3_swings[-1]['price']:
                sl = last_3_swings[-2]['price'] + 0.5 * current['atr']
                risk = sl - current['close']
                tp = current['close'] - 2 * risk
                return Signal(Direction.SHORT, SignalStrength.MODERATE, 0.67, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'choch_type': 'BEARISH'})
        
        return None
    
    def _detect_swings(self, df: pd.DataFrame):
        """Simple fractal-based swing detection"""
        swings = []
        for i in range(self.fractal_bars, len(df) - self.fractal_bars):
            window = df.iloc[i-self.fractal_bars:i+self.fractal_bars+1]
            if df.iloc[i]['high'] == window['high'].max():
                swings.append({'type': 'high', 'price': df.iloc[i]['high'], 'idx': i})
            elif df.iloc[i]['low'] == window['low'].min():
                swings.append({'type': 'low', 'price': df.iloc[i]['low'], 'idx': i})
        return swings
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-11 ChoCH validation placeholder")
