"""TC-07: ADX Trend Strength Gate - Returns regime state based on ADX levels"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_adx

class TC07_ADXGate(TechnicalSignal):
    TC_ID = "TC-07"
    TC_NAME = "ADX Trend Strength Gate"
    TC_CATEGORY = TCCategory.FILTER
    MIN_LOOKBACK = 30
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_07']
        self.adx_strong = self.params['adx_strong']
        self.adx_moderate = self.params['adx_moderate']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        # This is a gate, not a signal generator
        return None
    
    def get_trend_state(self, df: pd.DataFrame) -> str:
        """Returns TRENDING_STRONG, TRENDING_MODERATE, or RANGING"""
        if len(df) < self.MIN_LOOKBACK:
            return "UNKNOWN"
        df = df.copy()
        adx_df = compute_adx(df, period=14)
        adx = adx_df['adx'].iloc[-1]
        if adx >= self.adx_strong:
            return "TRENDING_STRONG"
        elif adx >= self.adx_moderate:
            return "TRENDING_MODERATE"
        else:
            return "RANGING"
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Gate module")

if __name__ == "__main__":
    print("TC-07 ADX Gate - Gate module, no standalone validation")
