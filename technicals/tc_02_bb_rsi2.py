"""
TC-02: Bollinger Band + RSI2 Mean Reversion
============================================

Refactored from Combo C (backtest_v3/combos.py).

LOGIC:
------
1. Entry: close < BB_lower AND RSI(2) < 15
2. Exit priority:
   a) Acceleration SL: BB_lower - 1x ATR(10) (stop tightens as price moves)
   b) BB midline reversion (profit target)
   c) 10-bar time stop

REGIME FIT: CORRECTIVE (1.0x), RANGING (1.0x), TRENDING (0.3x)

UNIVERSE: Low-beta instruments (β < 0.8 enforced via symbol list)

VALIDATION: 2-year Alpaca simulation, 6 gates
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
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
from technical.indicators_v4 import (
    compute_bollinger_bands,
    compute_rsi,
    compute_atr
)


@dataclass
class BarSnapshot:
    """Minimal BarSnapshot for Combo C logic"""
    close: float
    high: float
    low: float
    volume: float
    
    bb_lower: float
    bb_mid: float
    bb_upper: float
    
    rsi2: float
    atr_10: float
    
    ready: bool = True


class TC02_BollingerRSI2(TechnicalSignal):
    """Bollinger Band + RSI2 Mean Reversion Technical Concept (TC-02)"""
    
    TC_ID = "TC-02"
    TC_NAME = "Bollinger RSI2 Mean Reversion"
    TC_CATEGORY = TCCategory.MEAN_REVERSION
    MIN_LOOKBACK = 50  # Need 20 for BB, 10 for ATR(10), warmup buffer
    
    def __init__(self, config_path: Optional[str] = None):
        """Load parameters from tc_params.yaml"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        
        self.params = all_params['tc_02']
        
        # Unpack parameters
        self.bb_period = self.params['bb_period']
        self.bb_std = self.params['bb_std']
        self.rsi2_max = self.params['rsi2_max']
        self.sl_atr_mult = self.params['sl_atr_mult']
        self.time_bars = self.params['time_bars']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
    
    def compute(self, df: pd.DataFrame) -> Signal:
        """
        Core signal generation from enriched OHLCV DataFrame.
        
        Args:
            df: Must contain ['open','high','low','close','volume']
        
        Returns:
            Signal object or None
        """
        if len(df) < self.MIN_LOOKBACK:
            return None
        
        # 1. Enrich with indicators
        df = self._enrich_indicators(df.copy())
        
        # 2. Build BarSnapshot from last row
        snap = self._build_snapshot(df)
        
        if not snap.ready:
            return None
        
        # 3. Check entry condition (port of combo_c_entry)
        entry_signal = self._check_entry(snap)
        
        if entry_signal is not None:
            direction = Direction.LONG if entry_signal == "LONG" else Direction.SHORT
            
            # Calculate stops
            sl_price = self._calculate_stop_loss(snap, direction)
            tp_price = snap.bb_mid  # BB midline reversion target
            
            signal = Signal(
                direction=direction,
                strength=SignalStrength.MODERATE,
                confidence=0.70,  # Mean reversion has higher base confidence
                entry_price=snap.close,
                stop_loss=sl_price,
                take_profit=tp_price,
                metadata={
                    'tc_id': self.TC_ID,
                    'rsi2': snap.rsi2,
                    'bb_lower': snap.bb_lower,
                    'bb_mid': snap.bb_mid,
                    'distance_to_lower_pct': (snap.close - snap.bb_lower) / snap.close
                }
            )
            
            return signal
        
        return None
    
    def _enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to DataFrame"""
        # Bollinger Bands (20, 2.0)
        bb_df = compute_bollinger_bands(df, period=self.bb_period, std=self.bb_std)
        df['bb_upper'] = bb_df['upper']
        df['bb_mid'] = bb_df['middle']
        df['bb_lower'] = bb_df['lower']
        
        # RSI(2)
        df['rsi2'] = compute_rsi(df, period=2)
        
        # ATR(10) for acceleration SL
        df['atr_10'] = compute_atr(df, period=10)
        
        return df
    
    def _build_snapshot(self, df: pd.DataFrame) -> BarSnapshot:
        """Build BarSnapshot from last row of enriched DataFrame"""
        row = df.iloc[-1]
        
        # Check if all indicators are ready
        ready = not pd.isna([
            row['bb_lower'],
            row['bb_mid'],
            row['bb_upper'],
            row['rsi2'],
            row['atr_10']
        ]).any()
        
        return BarSnapshot(
            close=row['close'],
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            bb_lower=row['bb_lower'],
            bb_mid=row['bb_mid'],
            bb_upper=row['bb_upper'],
            rsi2=row['rsi2'],
            atr_10=row['atr_10'],
            ready=ready
        )
    
    def _check_entry(self, snap: BarSnapshot) -> Optional[str]:
        """
        Check entry condition (port of combo_c_entry).
        
        Entry: close < BB_lower AND RSI(2) < 15
        
        Returns:
            "LONG" if condition met, None otherwise
        """
        below_lower = snap.close < snap.bb_lower
        rsi2_oversold = snap.rsi2 < self.rsi2_max
        
        if below_lower and rsi2_oversold:
            return "LONG"
        
        # Could add SHORT condition: close > BB_upper AND RSI(2) > 85
        # But Combo C was LONG-only, so keeping it consistent
        
        return None
    
    def _calculate_stop_loss(self, snap: BarSnapshot, direction: Direction) -> float:
        """
        Calculate acceleration stop (port of Combo C V2A exit logic).
        
        Combo C uses: bb_lower - 1x ATR(10)
        This creates a "trailing" effect as BB_lower adjusts with volatility.
        """
        sl_dist = self.sl_atr_mult * snap.atr_10
        
        if direction == Direction.LONG:
            return snap.bb_lower - sl_dist
        else:
            return snap.bb_upper + sl_dist
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Run 2-year Alpaca simulation and validate against 6 gates.
        
        Args:
            validation_config: Optional override config
        
        Returns:
            ValidationResult with metrics and gate pass/fail
        """
        from simulation.alpaca_data_fetcher import AlpacaDataFetcher
        from simulation.walk_forward import WalkForwardValidator
        from simulation.metrics_engine import MetricsEngine
        
        # Use default config if not provided
        if validation_config is None:
            validation_config = {}
        
        start_date = validation_config.get('start_date', '2024-01-01')
        end_date = validation_config.get('end_date', '2026-01-01')
        initial_capital = validation_config.get('initial_capital', 100000)
        symbols = validation_config.get('symbols', self.symbols[:5])  # Limit to 5 for speed
        
        print(f"\n{'='*70}")
        print(f"TC-02 Bollinger RSI2 Mean Reversion — VALIDATION")
        print(f"{'='*70}")
        print(f"Period: {start_date} → {end_date}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Initial Capital: ${initial_capital:,.0f}\n")
        
        # Fetch data
        fetcher = AlpacaDataFetcher()
        all_trades = []
        
        for symbol in symbols:
            print(f"Simulating {symbol}...")
            df = fetcher.fetch_bars(symbol, start_date, end_date, timeframe='1Day')
            
            if df is None or len(df) < self.MIN_LOOKBACK:
                print(f"  ⚠️  Insufficient data for {symbol}, skipping")
                continue
            
            # Enrich
            df = self._enrich_indicators(df)
            
            # Simulate
            trades = self._simulate_symbol(df, symbol, initial_capital)
            all_trades.extend(trades)
            print(f"  → {len(trades)} trades")
        
        if len(all_trades) == 0:
            print("\n❌ NO TRADES GENERATED — Cannot validate")
            return ValidationResult(
                tc_id=self.TC_ID,
                validation_date=datetime.now(),
                metrics={},
                gates_passed=[],
                gates_failed=[
                    ValidationGate(
                        name="MIN_TRADES",
                        threshold=8,
                        actual=0,
                        passed=False
                    )
                ],
                approved=False,
                notes="No trades generated during simulation period"
            )
        
        # Calculate metrics
        metrics_engine = MetricsEngine()
        metrics = metrics_engine.calculate_all_metrics(all_trades, initial_capital)
        
        # Walk-forward validation
        wfv = WalkForwardValidator()
        wf_results = wfv.validate(all_trades, train_pct=0.6, val_pct=0.2, test_pct=0.2)
        
        # Print results
        self._print_validation_report(metrics, wf_results, all_trades)
        
        # Check 6 gates
        gates_passed, gates_failed = self._check_validation_gates(metrics, wf_results, all_trades)
        
        approved = len(gates_failed) == 0
        
        return ValidationResult(
            tc_id=self.TC_ID,
            validation_date=datetime.now(),
            metrics=metrics,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            approved=approved,
            notes=f"Simulated on {len(symbols)} symbols, {len(all_trades)} total trades"
        )
    
    def _simulate_symbol(self, df: pd.DataFrame, symbol: str, initial_capital: float) -> list:
        """Bar-by-bar simulation for one symbol"""
        trades = []
        position = None
        
        for i in range(self.MIN_LOOKBACK, len(df)):
            window = df.iloc[:i+1]
            snap = self._build_snapshot(window)
            
            if not snap.ready:
                continue
            
            # If in position, check exit
            if position is not None:
                exit_reason, exit_price = self._check_exit(position, snap, window)
                if exit_reason is not None:
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    if position['direction'] == Direction.SHORT:
                        pnl = -pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i],
                        'direction': position['direction'].value,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'bars_held': position['bars_held']
                    })
                    
                    position = None
                else:
                    position['bars_held'] += 1
            
            # If flat, check for entry
            if position is None:
                signal = self.compute(window)
                if signal is not None:
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
    
    def _check_exit(self, position: Dict, snap: BarSnapshot, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """
        Check exit conditions (port of combo_c_exit).
        
        Priority:
        1. Acceleration SL hit (bb_lower - 1x ATR)
        2. BB midline reversion (profit target)
        3. 10-bar time stop
        """
        # 1. Acceleration SL
        current_sl = snap.bb_lower - (self.sl_atr_mult * snap.atr_10)
        if position['direction'] == Direction.LONG:
            if snap.low <= current_sl:
                return ("ACCEL_SL", current_sl)
        else:
            current_sl = snap.bb_upper + (self.sl_atr_mult * snap.atr_10)
            if snap.high >= current_sl:
                return ("ACCEL_SL", current_sl)
        
        # 2. BB midline reversion
        if position['direction'] == Direction.LONG:
            if snap.close >= snap.bb_mid:
                return ("BB_MID_TARGET", snap.close)
        else:
            if snap.close <= snap.bb_mid:
                return ("BB_MID_TARGET", snap.close)
        
        # 3. Time stop
        if position['bars_held'] >= self.time_bars:
            return ("TIME_STOP", snap.close)
        
        return (None, None)
    
    def _check_validation_gates(self, metrics: Dict, wf_results: Dict, trades: list) -> Tuple[list, list]:
        """Check 6 validation gates"""
        passed = []
        failed = []
        
        # Gate 1: PF ≥ 1.20
        pf = metrics.get('profit_factor', 0)
        gate = ValidationGate(name="PROFIT_FACTOR", threshold=1.20, actual=pf, passed=(pf >= 1.20))
        (passed if gate.passed else failed).append(gate)
        
        # Gate 2: OOS PF ≥ 0.90
        oos_pf = wf_results.get('test_profit_factor', 0)
        gate = ValidationGate(name="OOS_PROFIT_FACTOR", threshold=0.90, actual=oos_pf, passed=(oos_pf >= 0.90))
        (passed if gate.passed else failed).append(gate)
        
        # Gate 3: WR 35-70%
        wr = metrics.get('win_rate', 0)
        gate = ValidationGate(name="WIN_RATE", threshold="35-70%", actual=wr, passed=(0.35 <= wr <= 0.70))
        (passed if gate.passed else failed).append(gate)
        
        # Gate 4: DD < 15%
        dd = metrics.get('max_drawdown_pct', 0)
        gate = ValidationGate(name="MAX_DRAWDOWN", threshold=0.15, actual=dd, passed=(dd < 0.15))
        (passed if gate.passed else failed).append(gate)
        
        # Gate 5: Min 8 trades
        num_trades = len(trades)
        gate = ValidationGate(name="MIN_TRADES", threshold=8, actual=num_trades, passed=(num_trades >= 8))
        (passed if gate.passed else failed).append(gate)
        
        # Gate 6: WFE ≥ 0.80
        wfe = wf_results.get('walk_forward_efficiency', 0)
        gate = ValidationGate(name="WALK_FORWARD_EFFICIENCY", threshold=0.80, actual=wfe, passed=(wfe >= 0.80))
        (passed if gate.passed else failed).append(gate)
        
        return passed, failed
    
    def _print_validation_report(self, metrics: Dict, wf_results: Dict, trades: list):
        """Print detailed validation report"""
        print(f"\n{'─'*70}")
        print("METRICS")
        print(f"{'─'*70}")
        print(f"Total Trades:        {len(trades)}")
        print(f"Win Rate:            {metrics.get('win_rate', 0):.1%}")
        print(f"Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.1%}")
        print(f"Avg Win:             ${metrics.get('avg_win', 0):,.2f}")
        print(f"Avg Loss:            ${metrics.get('avg_loss', 0):,.2f}")
        print(f"Expectancy:          ${metrics.get('expectancy', 0):,.2f}")
        
        print(f"\n{'─'*70}")
        print("WALK-FORWARD VALIDATION (60/20/20)")
        print(f"{'─'*70}")
        print(f"Train PF:            {wf_results.get('train_profit_factor', 0):.2f}")
        print(f"Val PF:              {wf_results.get('val_profit_factor', 0):.2f}")
        print(f"Test PF (OOS):       {wf_results.get('test_profit_factor', 0):.2f}")
        print(f"Walk-Forward Eff:    {wf_results.get('walk_forward_efficiency', 0):.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI VALIDATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run validation from command line:
        python -m technicals.tc_02_bb_rsi2
    """
    print("\n🚀 TC-02 Bollinger RSI2 Mean Reversion — Validation Mode\n")
    
    tc = TC02_BollingerRSI2()
    
    validation_result = tc.validate(validation_config={
        'start_date': '2024-01-01',
        'end_date': '2026-01-01',
        'initial_capital': 100000,
        'symbols': ['GLD', 'WMT', 'USMV', 'COST', 'XOM']
    })
    
    # Print gate results
    print(f"\n{'='*70}")
    print("VALIDATION GATES")
    print(f"{'='*70}\n")
    
    for gate in validation_result.gates_passed:
        print(f"  ✅ {gate.name:25s} ≥ {gate.threshold:8} → {gate.actual:.3f} PASS")
    
    for gate in validation_result.gates_failed:
        print(f"  ❌ {gate.name:25s} ≥ {gate.threshold:8} → {gate.actual:.3f} FAIL")
    
    print(f"\n{'─'*70}")
    if validation_result.approved:
        print("✅ STATUS: APPROVED → TC-02 ACTIVE")
    else:
        print("❌ STATUS: REJECTED → TC-02 remains PENDING")
    print(f"{'─'*70}\n")
