"""TC-13: Stochastic RSI Oversold Crossup"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_rsi, compute_atr

class TC13_StochRSI(TechnicalSignal):
    TC_ID = "TC-13"
    TC_NAME = "Stochastic RSI Oversold Crossup"
    TC_CATEGORY = TCCategory.MEAN_REVERSION
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_13']
        self.rsi_period = self.params['rsi_period']
        self.stoch_period = self.params['stoch_period']
        self.k_smooth = self.params['k_smooth']
        self.d_smooth = self.params['d_smooth']
        self.k_oversold = self.params['k_oversold']
        self.k_overbought = self.params['k_overbought']
        self.rsi_long_max = self.params['rsi_long_max']
        self.rsi_short_min = self.params['rsi_short_min']
        self.sl_atr_mult = self.params['sl_atr_mult']
        self.target_k_level = self.params['target_k_level']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        
        # Calculate RSI
        df['rsi'] = compute_rsi(df, period=self.rsi_period)
        
        # Calculate Stochastic of RSI
        rsi_rolling = df['rsi'].rolling(window=self.stoch_period)
        rsi_min = rsi_rolling.min()
        rsi_max = rsi_rolling.max()
        df['stoch_rsi'] = 100 * (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
        
        # Smooth %K
        df['k'] = df['stoch_rsi'].rolling(window=self.k_smooth).mean()
        # %D
        df['d'] = df['k'].rolling(window=self.d_smooth).mean()
        
        df['atr'] = compute_atr(df, period=14)
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # LONG: %K crosses above %D from oversold
        if prev['k'] <= prev['d'] and current['k'] > current['d']:
            if current['k'] < self.k_oversold and current['rsi'] < self.rsi_long_max:
                sl = current['close'] - self.sl_atr_mult * current['atr']
                # TP when StochRSI reaches 50 (neutralization) - use 2x risk as proxy
                risk = current['close'] - sl
                tp = current['close'] + 2 * risk
                return Signal(Direction.LONG, SignalStrength.MODERATE, 0.68, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'k': current['k'], 'rsi': current['rsi']})
        
        # SHORT: %K crosses below %D from overbought
        if prev['k'] >= prev['d'] and current['k'] < current['d']:
            if current['k'] > self.k_overbought and current['rsi'] > self.rsi_short_min:
                sl = current['close'] + self.sl_atr_mult * current['atr']
                risk = sl - current['close']
                tp = current['close'] - 2 * risk
                return Signal(Direction.SHORT, SignalStrength.MODERATE, 0.68, current['close'], sl, tp,
                             {'tc_id': self.TC_ID, 'k': current['k'], 'rsi': current['rsi']})
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-13 StochRSI validation placeholder")
