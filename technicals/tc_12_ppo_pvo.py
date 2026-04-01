"""TC-12: PPO + PVO Dual Momentum"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import yaml
from pathlib import Path
from technicals.base_signal import *
from technical.indicators_v4 import compute_atr

class TC12_PPOPVO(TechnicalSignal):
    TC_ID = "TC-12"
    TC_NAME = "PPO + PVO Dual Momentum"
    TC_CATEGORY = TCCategory.MOMENTUM
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        self.params = all_params['tc_12']
        self.fast_ema = self.params['fast_ema']
        self.slow_ema = self.params['slow_ema']
        self.signal_ema = self.params['signal_ema']
        self.sync_window = self.params['sync_window']
        self.tp_atr_mult = self.params['tp_atr_mult']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = df.copy()
        
        # PPO (Price Percentage Oscillator)
        ema_fast = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
        df['ppo_signal'] = df['ppo'].ewm(span=self.signal_ema, adjust=False).mean()
        
        # PVO (Percentage Volume Oscillator)
        vol_ema_fast = df['volume'].ewm(span=self.fast_ema, adjust=False).mean()
        vol_ema_slow = df['volume'].ewm(span=self.slow_ema, adjust=False).mean()
        df['pvo'] = ((vol_ema_fast - vol_ema_slow) / vol_ema_slow) * 100
        df['pvo_signal'] = df['pvo'].ewm(span=self.signal_ema, adjust=False).mean()
        
        df['atr'] = compute_atr(df, period=14)
        
        # Check for synchronized crossovers within sync_window
        ppo_cross_up = None
        pvo_cross_up = None
        
        for i in range(1, min(self.sync_window + 1, len(df))):
            idx = -i
            if df['ppo'].iloc[idx-1] <= df['ppo_signal'].iloc[idx-1] and df['ppo'].iloc[idx] > df['ppo_signal'].iloc[idx]:
                ppo_cross_up = i
            if df['pvo'].iloc[idx-1] <= df['pvo_signal'].iloc[idx-1] and df['pvo'].iloc[idx] > df['pvo_signal'].iloc[idx]:
                pvo_cross_up = i
        
        current = df.iloc[-1]
        
        # Both crossed up within sync window = LONG
        if ppo_cross_up and pvo_cross_up:
            sl = current['close'] - 1.5 * current['atr']
            tp = current['close'] + self.tp_atr_mult * current['atr']
            return Signal(Direction.LONG, SignalStrength.STRONG, 0.72, current['close'], sl, tp,
                         {'tc_id': self.TC_ID, 'ppo': current['ppo'], 'pvo': current['pvo']})
        
        # Both crossed down = SHORT
        ppo_cross_down = None
        pvo_cross_down = None
        
        for i in range(1, min(self.sync_window + 1, len(df))):
            idx = -i
            if df['ppo'].iloc[idx-1] >= df['ppo_signal'].iloc[idx-1] and df['ppo'].iloc[idx] < df['ppo_signal'].iloc[idx]:
                ppo_cross_down = i
            if df['pvo'].iloc[idx-1] >= df['pvo_signal'].iloc[idx-1] and df['pvo'].iloc[idx] < df['pvo_signal'].iloc[idx]:
                pvo_cross_down = i
        
        if ppo_cross_down and pvo_cross_down:
            sl = current['close'] + 1.5 * current['atr']
            tp = current['close'] - self.tp_atr_mult * current['atr']
            return Signal(Direction.SHORT, SignalStrength.STRONG, 0.72, current['close'], sl, tp,
                         {'tc_id': self.TC_ID, 'ppo': current['ppo'], 'pvo': current['pvo']})
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        return ValidationResult(self.TC_ID, datetime.now(), {}, [], [], True, "Validation placeholder")

if __name__ == "__main__":
    print("TC-12 PPO/PVO validation placeholder")
