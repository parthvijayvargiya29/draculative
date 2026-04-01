"""
TC-04: Fair Value Gap (FVG) Entry
==================================

LOGIC:
------
1. Detect FVG: 3-candle pattern where candle[i-1] creates a gap
   - Bullish FVG: candle[i-1].low > candle[i-2].high (gap down that gets filled up)
   - Bearish FVG: candle[i-1].high < candle[i-2].low (gap up that gets filled down)
2. Gap must be >= 0.3% of price (min_gap_pct)
3. Volume on gap candle > 1.2x 20-bar average (volume confirmation)
4. Entry: Price enters FVG zone (50% fill of gap)
5. Exit: SL beyond FVG far edge, TP at gap full fill + 1x gap size

REGIME FIT: TRENDING (1.0x), CORRECTIVE (0.8x)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path

from technicals.base_signal import (
    TechnicalSignal,
    Signal,
    Direction,
    SignalStrength,
    TCCategory,
    ValidationResult,
    ValidationGate
)
from technical.indicators_v4 import compute_atr


@dataclass
class FVG:
    """Fair Value Gap structure"""
    direction: str  # "BULLISH" or "BEARISH"
    gap_high: float
    gap_low: float
    gap_size: float
    gap_bar_idx: int
    volume_ratio: float


class TC04_FairValueGap(TechnicalSignal):
    """Fair Value Gap Entry Technical Concept (TC-04)"""
    
    TC_ID = "TC-04"
    TC_NAME = "Fair Value Gap Entry"
    TC_CATEGORY = TCCategory.ORDERFLOW
    MIN_LOOKBACK = 50
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        
        self.params = all_params['tc_04']
        self.min_gap_pct = self.params['min_gap_pct']
        self.volume_mult = self.params['volume_mult']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
        
        self.active_fvgs: List[FVG] = []  # Track unfiltered FVGs
    
    def compute(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        df = self._enrich_indicators(df.copy())
        
        # Detect new FVGs
        new_fvg = self._detect_fvg(df)
        if new_fvg:
            self.active_fvgs.append(new_fvg)
        
        # Check if price is entering any active FVG
        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        for fvg in self.active_fvgs[:]:
            # Check if FVG is filled (remove from active list)
            if fvg.direction == "BULLISH" and current_low <= fvg.gap_low:
                self.active_fvgs.remove(fvg)
                continue
            if fvg.direction == "BEARISH" and current_high >= fvg.gap_high:
                self.active_fvgs.remove(fvg)
                continue
            
            # Check for entry (50% fill of gap)
            fvg_mid = (fvg.gap_high + fvg.gap_low) / 2
            
            if fvg.direction == "BULLISH":
                if current_low <= fvg_mid <= current_high:
                    # Entry signal
                    sl_price = fvg.gap_low - (fvg.gap_size * 0.2)  # SL 20% below gap
                    tp_price = fvg.gap_high + fvg.gap_size  # TP = full fill + 1x gap
                    
                    signal = Signal(
                        direction=Direction.LONG,
                        strength=SignalStrength.MODERATE,
                        confidence=0.65,
                        entry_price=current_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        metadata={
                            'tc_id': self.TC_ID,
                            'fvg_type': 'BULLISH',
                            'gap_size_pct': fvg.gap_size / current_price,
                            'volume_ratio': fvg.volume_ratio
                        }
                    )
                    self.active_fvgs.remove(fvg)
                    return signal
            
            else:  # BEARISH
                if current_low <= fvg_mid <= current_high:
                    sl_price = fvg.gap_high + (fvg.gap_size * 0.2)
                    tp_price = fvg.gap_low - fvg.gap_size
                    
                    signal = Signal(
                        direction=Direction.SHORT,
                        strength=SignalStrength.MODERATE,
                        confidence=0.65,
                        entry_price=current_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        metadata={
                            'tc_id': self.TC_ID,
                            'fvg_type': 'BEARISH',
                            'gap_size_pct': fvg.gap_size / current_price,
                            'volume_ratio': fvg.volume_ratio
                        }
                    )
                    self.active_fvgs.remove(fvg)
                    return signal
        
        return None
    
    def _enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['atr'] = compute_atr(df, period=14)
        return df
    
    def _detect_fvg(self, df: pd.DataFrame) -> Optional[FVG]:
        """Detect FVG on last 3 bars"""
        if len(df) < 3:
            return None
        
        bars = df.iloc[-3:].copy()
        
        # Bullish FVG: bars[0].high < bars[2].low (gap created by bars[1])
        if bars.iloc[0]['high'] < bars.iloc[2]['low']:
            gap_low = bars.iloc[0]['high']
            gap_high = bars.iloc[2]['low']
            gap_size = gap_high - gap_low
            gap_pct = gap_size / bars.iloc[1]['close']
            
            volume_ratio = bars.iloc[1]['volume'] / bars.iloc[1]['volume_ma20']
            
            if gap_pct >= self.min_gap_pct and volume_ratio >= self.volume_mult:
                return FVG(
                    direction="BULLISH",
                    gap_high=gap_high,
                    gap_low=gap_low,
                    gap_size=gap_size,
                    gap_bar_idx=len(df) - 2,
                    volume_ratio=volume_ratio
                )
        
        # Bearish FVG: bars[0].low > bars[2].high
        if bars.iloc[0]['low'] > bars.iloc[2]['high']:
            gap_high = bars.iloc[0]['low']
            gap_low = bars.iloc[2]['high']
            gap_size = gap_high - gap_low
            gap_pct = gap_size / bars.iloc[1]['close']
            
            volume_ratio = bars.iloc[1]['volume'] / bars.iloc[1]['volume_ma20']
            
            if gap_pct >= self.min_gap_pct and volume_ratio >= self.volume_mult:
                return FVG(
                    direction="BEARISH",
                    gap_high=gap_high,
                    gap_low=gap_low,
                    gap_size=gap_size,
                    gap_bar_idx=len(df) - 2,
                    volume_ratio=volume_ratio
                )
        
        return None
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Run validation (same pattern as TC-01/02/03)"""
        from simulation.alpaca_data_fetcher import AlpacaDataFetcher
        from simulation.walk_forward import WalkForwardValidator
        from simulation.metrics_engine import MetricsEngine
        
        if validation_config is None:
            validation_config = {}
        
        start_date = validation_config.get('start_date', '2024-01-01')
        end_date = validation_config.get('end_date', '2026-01-01')
        initial_capital = validation_config.get('initial_capital', 100000)
        symbols = validation_config.get('symbols', self.symbols[:5])
        
        print(f"\n{'='*70}")
        print(f"TC-04 Fair Value Gap — VALIDATION")
        print(f"{'='*70}")
        print(f"Period: {start_date} → {end_date}")
        print(f"Symbols: {', '.join(symbols)}\n")
        
        fetcher = AlpacaDataFetcher()
        all_trades = []
        
        for symbol in symbols:
            print(f"Simulating {symbol}...")
            df = fetcher.fetch_bars(symbol, start_date, end_date, timeframe='1Day')
            
            if df is None or len(df) < self.MIN_LOOKBACK:
                print(f"  ⚠️  Insufficient data, skipping")
                continue
            
            df = self._enrich_indicators(df)
            trades = self._simulate_symbol(df, symbol, initial_capital)
            all_trades.extend(trades)
            print(f"  → {len(trades)} trades")
        
        if len(all_trades) == 0:
            return ValidationResult(
                tc_id=self.TC_ID,
                validation_date=datetime.now(),
                metrics={},
                gates_passed=[],
                gates_failed=[ValidationGate("MIN_TRADES", 8, 0, False)],
                approved=False,
                notes="No trades generated"
            )
        
        metrics_engine = MetricsEngine()
        metrics = metrics_engine.calculate_all_metrics(all_trades, initial_capital)
        
        wfv = WalkForwardValidator()
        wf_results = wfv.validate(all_trades, train_pct=0.6, val_pct=0.2, test_pct=0.2)
        
        self._print_validation_report(metrics, wf_results, all_trades)
        gates_passed, gates_failed = self._check_validation_gates(metrics, wf_results, all_trades)
        
        return ValidationResult(
            tc_id=self.TC_ID,
            validation_date=datetime.now(),
            metrics=metrics,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            approved=len(gates_failed) == 0,
            notes=f"Simulated on {len(symbols)} symbols"
        )
    
    def _simulate_symbol(self, df: pd.DataFrame, symbol: str, initial_capital: float) -> list:
        trades = []
        position = None
        self.active_fvgs = []
        
        for i in range(self.MIN_LOOKBACK, len(df)):
            window = df.iloc[:i+1]
            
            if position is not None:
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check TP
                if position['direction'] == Direction.LONG and current_high >= position['take_profit']:
                    pnl = (position['take_profit'] - position['entry_price']) * position['shares']
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i],
                        'direction': 'LONG',
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'shares': position['shares'],
                        'pnl': pnl,
                        'exit_reason': 'TP_HIT',
                        'bars_held': position['bars_held']
                    })
                    position = None
                    continue
                
                elif position['direction'] == Direction.SHORT and current_low <= position['take_profit']:
                    pnl = (position['entry_price'] - position['take_profit']) * position['shares']
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i],
                        'direction': 'SHORT',
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'shares': position['shares'],
                        'pnl': pnl,
                        'exit_reason': 'TP_HIT',
                        'bars_held': position['bars_held']
                    })
                    position = None
                    continue
                
                # Check SL
                if position['direction'] == Direction.LONG and current_low <= position['stop_loss']:
                    pnl = (position['stop_loss'] - position['entry_price']) * position['shares']
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i],
                        'direction': 'LONG',
                        'entry_price': position['entry_price'],
                        'exit_price': position['stop_loss'],
                        'shares': position['shares'],
                        'pnl': pnl,
                        'exit_reason': 'SL_HIT',
                        'bars_held': position['bars_held']
                    })
                    position = None
                    continue
                
                elif position['direction'] == Direction.SHORT and current_high >= position['stop_loss']:
                    pnl = (position['entry_price'] - position['stop_loss']) * position['shares']
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i],
                        'direction': 'SHORT',
                        'entry_price': position['entry_price'],
                        'exit_price': position['stop_loss'],
                        'shares': position['shares'],
                        'pnl': pnl,
                        'exit_reason': 'SL_HIT',
                        'bars_held': position['bars_held']
                    })
                    position = None
                    continue
                
                position['bars_held'] += 1
            
            if position is None:
                signal = self.compute(window)
                if signal:
                    risk_amt = initial_capital * self.risk_per_trade
                    stop_dist = abs(signal.entry_price - signal.stop_loss)
                    shares = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                    
                    if shares > 0:
                        position = {
                            'entry_date': df.index[i],
                            'entry_price': signal.entry_price,
                            'direction': signal.direction,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'shares': shares,
                            'bars_held': 0
                        }
        
        return trades
    
    def _check_validation_gates(self, metrics: Dict, wf_results: Dict, trades: list) -> Tuple[list, list]:
        passed, failed = [], []
        
        pf = metrics.get('profit_factor', 0)
        gate = ValidationGate("PROFIT_FACTOR", 1.20, pf, pf >= 1.20)
        (passed if gate.passed else failed).append(gate)
        
        oos_pf = wf_results.get('test_profit_factor', 0)
        gate = ValidationGate("OOS_PROFIT_FACTOR", 0.90, oos_pf, oos_pf >= 0.90)
        (passed if gate.passed else failed).append(gate)
        
        wr = metrics.get('win_rate', 0)
        gate = ValidationGate("WIN_RATE", "35-70%", wr, 0.35 <= wr <= 0.70)
        (passed if gate.passed else failed).append(gate)
        
        dd = metrics.get('max_drawdown_pct', 0)
        gate = ValidationGate("MAX_DRAWDOWN", 0.15, dd, dd < 0.15)
        (passed if gate.passed else failed).append(gate)
        
        gate = ValidationGate("MIN_TRADES", 8, len(trades), len(trades) >= 8)
        (passed if gate.passed else failed).append(gate)
        
        wfe = wf_results.get('walk_forward_efficiency', 0)
        gate = ValidationGate("WALK_FORWARD_EFFICIENCY", 0.80, wfe, wfe >= 0.80)
        (passed if gate.passed else failed).append(gate)
        
        return passed, failed
    
    def _print_validation_report(self, metrics: Dict, wf_results: Dict, trades: list):
        print(f"\n{'─'*70}")
        print("METRICS")
        print(f"{'─'*70}")
        print(f"Total Trades:        {len(trades)}")
        print(f"Win Rate:            {metrics.get('win_rate', 0):.1%}")
        print(f"Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.1%}")


if __name__ == "__main__":
    print("\n🚀 TC-04 Fair Value Gap — Validation Mode\n")
    tc = TC04_FairValueGap()
    result = tc.validate({'symbols': ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT']})
    
    print(f"\n{'='*70}")
    print("VALIDATION GATES")
    print(f"{'='*70}\n")
    
    for gate in result.gates_passed:
        print(f"  ✅ {gate.name}")
    for gate in result.gates_failed:
        print(f"  ❌ {gate.name}")
    
    print(f"\n{'─'*70}")
    print("✅ APPROVED → TC-04 ACTIVE" if result.approved else "❌ REJECTED → TC-04 PENDING")
    print(f"{'─'*70}\n")
