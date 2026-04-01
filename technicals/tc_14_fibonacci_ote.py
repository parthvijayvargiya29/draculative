"""TC-14: Fibonacci OTE Zone Entry"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_atr

class TC14_FibonacciOTE(TechnicalSignal):
    TC_ID = "TC-14"
    TC_NAME = "Fibonacci OTE Zone Entry"
    TC_CATEGORY = TCCategory.RETRACEMENT
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_14']
        self.ote_min = self.params['ote_min']  # 0.618
        self.ote_max = self.params['ote_max']  # 0.786
        self.invalidation_level = self.params['invalidation_level']  # 0.886
        self.fractal_bars = self.params['fractal_bars']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        df['atr'] = compute_atr(df, period=14)
        
        # Detect last swing (simplified: last 20 bars high/low)
        lookback = min(20, len(df) - 1)
        recent = df.iloc[-lookback:]
        
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        swing_high_idx = recent['high'].idxmax()
        swing_low_idx = recent['low'].idxmin()
        
        current = df.iloc[-1]
        
        # Bullish setup: Price retraces from swing high back toward swing low
        if swing_low_idx < swing_high_idx:  # Low came before high (uptrend)
            swing_range = swing_high - swing_low
            ote_low = swing_low + self.ote_min * swing_range
            ote_high = swing_low + self.ote_max * swing_range
            
            # Price in OTE zone
            if ote_low <= current['close'] <= ote_high:
                sl = swing_low - self.invalidation_level * swing_range
                tp = swing_high + 0.618 * swing_range  # Extension target
                return Signal(Direction.LONG, SignalStrength.MODERATE, 0.70, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'ote_zone': f"{self.ote_min}-{self.ote_max}"})
        
        # Bearish setup: Price retraces from swing low back toward swing high
        elif swing_high_idx < swing_low_idx:  # High came before low (downtrend)
            swing_range = swing_high - swing_low
            ote_low = swing_high - self.ote_max * swing_range
            ote_high = swing_high - self.ote_min * swing_range
            
            if ote_low <= current['close'] <= ote_high:
                sl = swing_high + self.invalidation_level * swing_range
                tp = swing_low - 0.618 * swing_range
                return Signal(Direction.SHORT, SignalStrength.MODERATE, 0.70, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'ote_zone': f"{self.ote_min}-{self.ote_max}"})
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-14 Fibonacci OTE validation placeholder")
