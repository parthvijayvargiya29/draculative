"""TC-08: Golden/Death Cross Filter - MA crossover detection"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *

class TC08_GoldenCross(TechnicalSignal):
    TC_ID = "TC-08"
    TC_NAME = "Golden/Death Cross Filter"
    TC_CATEGORY = TCCategory.FILTER
    MIN_LOOKBACK = 210
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_08']
        self.fast_ma = self.params['fast_ma']
        self.slow_ma = self.params['slow_ma']
        self.fresh_cross_bars = self.params['fresh_cross_bars']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        return None  # Filter module
    
    def get_cross_state(self, df: pd.DataFrame) -> str:
        """Returns GOLDEN (bullish), DEATH (bearish), or NEUTRAL"""
        if len(df) < self.MIN_LOOKBACK:
            return "NEUTRAL"
        df = df.copy()
        df['ma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # Check for recent cross
        for i in range(1, min(self.fresh_cross_bars + 1, len(df))):
            prev_diff = df['ma_fast'].iloc[-(i+1)] - df['ma_slow'].iloc[-(i+1)]
            curr_diff = df['ma_fast'].iloc[-i] - df['ma_slow'].iloc[-i]
            
            if prev_diff <= 0 and curr_diff > 0:
                return "GOLDEN"
            elif prev_diff >= 0 and curr_diff < 0:
                return "DEATH"
        
        # No recent cross, check current state
        if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
            return "GOLDEN"
        else:
            return "DEATH"
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Filter module")

if __name__ == "__main__":
    print("TC-08 Golden Cross - Filter module")
