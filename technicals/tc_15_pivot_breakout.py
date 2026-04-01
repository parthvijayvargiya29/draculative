"""TC-15: Pivot Point Breakout/Breakdown"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_atr

class TC15_PivotBreakout(TechnicalSignal):
    TC_ID = "TC-15"
    TC_NAME = "Pivot Point Breakout/Breakdown"
    TC_CATEGORY = TCCategory.BREAKOUT
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_15']
        self.volume_mult = self.params['volume_mult']
        self.use_retest = self.params['use_retest']
        self.use_direct = self.params['use_direct']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        
        # Calculate standard pivot points (previous day H/L/C)
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        
        df['pivot'] = (df['prev_high'] + df['prev_low'] + df['prev_close']) / 3
        df['r1'] = 2 * df['pivot'] - df['prev_low']
        df['s1'] = 2 * df['pivot'] - df['prev_high']
        
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['atr'] = compute_atr(df, period=14)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Volume confirmation
        volume_ok = current['volume'] > self.volume_mult * current['volume_ma']
        
        # Direct breakout mode
        if self.use_direct:
            # Break above R1
            if prev['close'] <= prev['r1'] and current['close'] > current['r1'] and volume_ok:
                sl = current['pivot']
                risk = current['close'] - sl
                tp = current['close'] + 2 * risk
                return Signal(Direction.LONG, SignalStrength.MODERATE, 0.65, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'mode': 'direct', 'level': 'R1'})
            
            # Break below S1
            if prev['close'] >= prev['s1'] and current['close'] < current['s1'] and volume_ok:
                sl = current['pivot']
                risk = sl - current['close']
                tp = current['close'] - 2 * risk
                return Signal(Direction.SHORT, SignalStrength.MODERATE, 0.65, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'mode': 'direct', 'level': 'S1'})
        
        # Retest mode (higher win rate, fewer trades)
        if self.use_retest:
            # Check if we broke R1 recently (within last 5 bars) and are now retesting
            lookback = df.iloc[-5:]
            broke_r1 = any(lookback['close'] > lookback['r1'])
            
            if broke_r1 and current['low'] <= current['r1'] <= current['high'] and current['close'] > current['r1']:
                sl = current['r1'] - 0.5 * current['atr']
                risk = current['close'] - sl
                tp = current['close'] + 2 * risk
                return Signal(Direction.LONG, SignalStrength.STRONG, 0.72, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'mode': 'retest', 'level': 'R1'})
            
            # Retest S1
            broke_s1 = any(lookback['close'] < lookback['s1'])
            if broke_s1 and current['low'] <= current['s1'] <= current['high'] and current['close'] < current['s1']:
                sl = current['s1'] + 0.5 * current['atr']
                risk = sl - current['close']
                tp = current['close'] - 2 * risk
                return Signal(Direction.SHORT, SignalStrength.STRONG, 0.72, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'mode': 'retest', 'level': 'S1'})
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-15 Pivot Breakout validation placeholder")
