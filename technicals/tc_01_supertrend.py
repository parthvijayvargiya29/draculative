"""
TC-01: SuperTrend Pullback
==========================

Refactored from Combo B (backtest_v3/combos.py).

LOGIC:
------
1. Detect SuperTrend flip (UP→DOWN or DOWN→UP)
2. Enter "armed" state for 15 bars
3. Watch for pullback: low touches ST line ±8%, close holds above/below
4. Entry gates: RSI > 50 (LONG) / < 50 (SHORT), ADX(14) > 20
5. Exit priority: Hard SL (1.5x ATR) → ST re-flip → 15-bar time stop

REGIME FIT: TRENDING (1.0x), CORRECTIVE (0.3x)

VALIDATION: 2-year Alpaca simulation, 6 gates (PF≥1.20, OOS PF≥0.90, WR 35-70%, DD<15%, min 8 trades, WFE≥0.80)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    compute_supertrend,
    compute_rsi,
    compute_adx,
    compute_atr
)


@dataclass
class BarSnapshot:
    """
    Minimal BarSnapshot compatible with Combo B logic.
    Built from enriched DataFrame row.
    """
    close: float
    high: float
    low: float
    volume: float
    
    supertrend_line: float
    supertrend_dir: int         # 1 = UP, -1 = DOWN
    prev_st_dir: int
    
    rsi: float
    adx14_val: float
    atr: float
    
    ready: bool = True


class TC01_Supertrend(TechnicalSignal):
    """SuperTrend Pullback Technical Concept (TC-01)"""
    
    TC_ID = "TC-01"
    TC_NAME = "SuperTrend Pullback"
    TC_CATEGORY = TCCategory.TREND_FOLLOWING
    MIN_LOOKBACK = 80  # Need 50+ for ADX, 20+ for ST warmup
    
    def __init__(self, config_path: Optional[str] = None):
        """Load parameters from tc_params.yaml"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tc_params.yaml"
        
        with open(config_path, 'r') as f:
            all_params = yaml.safe_load(f)
        
        self.params = all_params['tc_01']
        
        # Unpack for readability
        self.st_period = self.params['st_period']
        self.st_multiplier = self.params['st_multiplier']
        self.pullback_pct = self.params['pullback_pct']
        self.rsi_min = self.params['rsi_min']
        self.adx_min = self.params['adx_min']
        self.armed_window_bars = self.params['armed_window_bars']
        self.sl_atr_mult = self.params['sl_atr_mult']
        self.time_bars = self.params['time_bars']
        self.risk_per_trade = self.params['risk_per_trade']
        self.symbols = self.params['symbols']
        
        # State tracking for armed window
        self.armed_state: Optional[Dict[str, Any]] = None  # {flip_dir, flip_bar, flip_st_line}
    
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
        
        # 3. Detect flip
        flip_signal = self._detect_flip(snap)
        if flip_signal is not None:
            # Enter armed state
            self.armed_state = {
                'flip_dir': flip_signal,
                'flip_bar': len(df) - 1,
                'flip_st_line': snap.supertrend_line
            }
        
        # 4. Check if armed window expired
        if self.armed_state is not None:
            bars_since_flip = (len(df) - 1) - self.armed_state['flip_bar']
            if bars_since_flip > self.armed_window_bars:
                self.armed_state = None  # Window expired
        
        # 5. If armed, check for pullback
        if self.armed_state is not None:
            entry_signal = self._check_pullback_entry(snap)
            if entry_signal is not None:
                # Entry gates passed
                direction = Direction.LONG if entry_signal == "LONG" else Direction.SHORT
                
                # Calculate stops
                sl_price = self._calculate_stop_loss(snap, direction)
                tp_price = None  # ST flip is primary exit, no fixed TP
                
                signal = Signal(
                    direction=direction,
                    strength=SignalStrength.MODERATE,  # Pullback is medium conviction
                    confidence=0.65,
                    entry_price=snap.close,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    metadata={
                        'tc_id': self.TC_ID,
                        'flip_bar': self.armed_state['flip_bar'],
                        'bars_since_flip': (len(df) - 1) - self.armed_state['flip_bar'],
                        'rsi': snap.rsi,
                        'adx': snap.adx14_val,
                        'supertrend_line': snap.supertrend_line
                    }
                )
                
                # Reset armed state after entry
                self.armed_state = None
                return signal
        
        return None
    
    def _enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to DataFrame"""
        # SuperTrend
        st_df = compute_supertrend(df, period=self.st_period, multiplier=self.st_multiplier)
        df['supertrend_line'] = st_df['supertrend']
        df['supertrend_dir'] = st_df['direction']
        
        # RSI(14)
        df['rsi'] = compute_rsi(df, period=14)
        
        # ADX(14)
        adx_df = compute_adx(df, period=14)
        df['adx'] = adx_df['adx']
        
        # ATR(14)
        df['atr'] = compute_atr(df, period=14)
        
        # Previous ST direction (shifted)
        df['prev_st_dir'] = df['supertrend_dir'].shift(1)
        
        return df
    
    def _build_snapshot(self, df: pd.DataFrame) -> BarSnapshot:
        """Build BarSnapshot from last row of enriched DataFrame"""
        row = df.iloc[-1]
        
        # Check if all indicators are ready
        ready = not pd.isna([
            row['supertrend_line'],
            row['supertrend_dir'],
            row['prev_st_dir'],
            row['rsi'],
            row['adx'],
            row['atr']
        ]).any()
        
        return BarSnapshot(
            close=row['close'],
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            supertrend_line=row['supertrend_line'],
            supertrend_dir=int(row['supertrend_dir']),
            prev_st_dir=int(row['prev_st_dir']),
            rsi=row['rsi'],
            adx14_val=row['adx'],
            atr=row['atr'],
            ready=ready
        )
    
    def _detect_flip(self, snap: BarSnapshot) -> Optional[str]:
        """
        Detect SuperTrend flip (port of combo_b_flip_detect).
        
        Returns:
            "LONG" if UP flip, "SHORT" if DOWN flip, None otherwise
        """
        if snap.supertrend_dir == 1 and snap.prev_st_dir == -1:
            return "LONG"
        elif snap.supertrend_dir == -1 and snap.prev_st_dir == 1:
            return "SHORT"
        return None
    
    def _check_pullback_entry(self, snap: BarSnapshot) -> Optional[str]:
        """
        Check if pullback + entry gates are satisfied (port of combo_b_pullback_check + combo_b_entry_gates).
        
        Returns:
            "LONG" or "SHORT" if entry confirmed, None otherwise
        """
        flip_dir = self.armed_state['flip_dir']
        st_line = snap.supertrend_line
        
        # Pullback check
        if flip_dir == "LONG":
            low_touched = snap.low <= st_line * (1.0 + self.pullback_pct)
            close_above = snap.close > st_line
            pullback_ok = low_touched and close_above
        else:  # SHORT
            high_touched = snap.high >= st_line * (1.0 - self.pullback_pct)
            close_below = snap.close < st_line
            pullback_ok = high_touched and close_below
        
        if not pullback_ok:
            return None
        
        # Entry gates
        if flip_dir == "LONG":
            rsi_ok = snap.rsi > self.rsi_min
        else:
            rsi_ok = snap.rsi < (100 - self.rsi_min)
        
        adx_ok = snap.adx14_val > self.adx_min
        
        if rsi_ok and adx_ok:
            return flip_dir
        
        return None
    
    def _calculate_stop_loss(self, snap: BarSnapshot, direction: Direction) -> float:
        """
        Calculate hard stop (1.5x ATR from entry).
        
        Port of Combo B exit logic:
            sl_dist = COMBO_B_SL_ATR_MULT * snap.atr
        """
        sl_dist = self.sl_atr_mult * snap.atr
        
        if direction == Direction.LONG:
            return snap.close - sl_dist
        else:
            return snap.close + sl_dist
    
    def validate(self, validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Run 2-year Alpaca simulation and validate against 6 gates.
        
        Args:
            validation_config: Optional override config
                {
                    'start_date': '2024-01-01',
                    'end_date': '2026-01-01',
                    'initial_capital': 100000,
                    'symbols': [...],  # Override self.symbols
                }
        
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
        print(f"TC-01 SuperTrend Pullback — VALIDATION")
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
            
            # Simulate (walk forward bar-by-bar)
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
        
        # Walk-forward validation (60/20/20 split)
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
        """
        Bar-by-bar simulation for one symbol.
        
        Returns:
            List of trade dicts
        """
        trades = []
        position = None  # {entry_bar, entry_price, direction, stop_loss, bars_held}
        
        for i in range(self.MIN_LOOKBACK, len(df)):
            # Build rolling window
            window = df.iloc[:i+1]
            snap = self._build_snapshot(window)
            
            if not snap.ready:
                continue
            
            # If in position, check exit
            if position is not None:
                exit_reason, exit_price = self._check_exit(position, snap, window)
                if exit_reason is not None:
                    # Close position
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
            
            # If flat, check for entry signal
            if position is None:
                signal = self.compute(window)
                if signal is not None:
                    # Calculate position size (1% risk per trade)
                    risk_amt = initial_capital * self.risk_per_trade
                    stop_dist = abs(signal.entry_price - signal.stop_loss)
                    shares = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                    
                    if shares > 0:
                        position = {
                            'entry_date': df.index[i],
                            'entry_price': signal.entry_price,
                            'direction': signal.direction,
                            'stop_loss': signal.stop_loss,
                            'shares': shares,
                            'bars_held': 0
                        }
        
        return trades
    
    def _check_exit(self, position: Dict, snap: BarSnapshot, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """
        Check exit conditions (port of combo_b_exit priority logic).
        
        Priority:
        1. Hard SL hit
        2. SuperTrend re-flip
        3. Time stop (15 bars)
        
        Returns:
            (exit_reason, exit_price) or (None, None)
        """
        # 1. Hard SL
        if position['direction'] == Direction.LONG:
            if snap.low <= position['stop_loss']:
                return ("SL_HIT", position['stop_loss'])
        else:
            if snap.high >= position['stop_loss']:
                return ("SL_HIT", position['stop_loss'])
        
        # 2. ST re-flip
        flip = self._detect_flip(snap)
        if flip is not None:
            if position['direction'] == Direction.LONG and flip == "SHORT":
                return ("ST_REFLIP", snap.close)
            elif position['direction'] == Direction.SHORT and flip == "LONG":
                return ("ST_REFLIP", snap.close)
        
        # 3. Time stop
        if position['bars_held'] >= self.time_bars:
            return ("TIME_STOP", snap.close)
        
        return (None, None)
    
    def _check_validation_gates(self, metrics: Dict, wf_results: Dict, trades: list) -> Tuple[list, list]:
        """
        Check 6 validation gates.
        
        Returns:
            (gates_passed, gates_failed) as lists of ValidationGate objects
        """
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
        python -m technicals.tc_01_supertrend
    """
    print("\n🚀 TC-01 SuperTrend Pullback — Validation Mode\n")
    
    tc = TC01_Supertrend()
    
    validation_result = tc.validate(validation_config={
        'start_date': '2024-01-01',
        'end_date': '2026-01-01',
        'initial_capital': 100000,
        'symbols': ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT']  # Fast test on 5 symbols
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
        print("✅ STATUS: APPROVED → TC-01 ACTIVE")
    else:
        print("❌ STATUS: REJECTED → TC-01 remains PENDING")
    print(f"{'─'*70}\n")
