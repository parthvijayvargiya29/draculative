"""TC-09: Volume Climax Reversal"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_atr

class TC09_VolumeClimax(TechnicalSignal):
    TC_ID = "TC-09"
    TC_NAME = "Volume Climax Reversal"
    TC_CATEGORY = TCCategory.REVERSAL
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_09']
        self.volume_mult = self.params['volume_mult']
        self.trend_lookback = self.params['trend_lookback']
        self.wick_body_ratio = self.params['wick_body_ratio']
        self.sl_buffer_pct = self.params['sl_buffer_pct']
        self.tp_atr_mult = self.params['tp_atr_mult']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['atr'] = compute_atr(df, period=14)
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        current = df.iloc[-1]
        
        # Volume spike
        if current['volume'] < self.volume_mult * current['volume_ma']:
            return None
        
        # Determine trend direction
        trend_bars = df.iloc[-(self.trend_lookback+1):-1]
        trend_up = (trend_bars['close'] > trend_bars['open']).sum() > self.trend_lookback / 2
        
        # Bullish climax (after downtrend)
        if not trend_up and current['lower_wick'] > self.wick_body_ratio * current['body']:
            sl = current['low'] - self.sl_buffer_pct * current['close']
            tp = current['close'] + self.tp_atr_mult * current['atr']
            return Signal(Direction.LONG, SignalStrength.MODERATE, 0.65, current['close'], sl, tp,
                         {'tc_id': self.TC_ID, 'climax_type': 'BULLISH'})
        
        # Bearish climax (after uptrend)
        elif trend_up and current['upper_wick'] > self.wick_body_ratio * current['body']:
            sl = current['high'] + self.sl_buffer_pct * current['close']
            tp = current['close'] - self.tp_atr_mult * current['atr']
            return Signal(Direction.SHORT, SignalStrength.MODERATE, 0.65, current['close'], sl, tp,
                         {'tc_id': self.TC_ID, 'climax_type': 'BEARISH'})
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-09 Volume Climax validation placeholder")
