"""Layer 3: Trade Logic - Signal aggregation, convergence scoring, state machine."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict
import numpy as np


class TradeState(Enum):
    """Trading state machine."""
    IDLE = "idle"
    LONG = "long"
    SHORT = "short"
    CLOSING = "closing"


class SignalType(Enum):
    """Signal types from indicators."""
    BUY = 1
    NEUTRAL = 0
    SELL = -1


@dataclass
class Signal:
    """Individual indicator signal."""
    source: str  # 'macd', 'bollinger', 'stochastic'
    type: SignalType
    strength: float  # 0.0-1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedSignal:
    """Combined signal from multiple indicators."""
    timestamp: datetime
    buy_count: int  # Number of indicators voting BUY
    total_count: int  # Total indicators
    convergence_score: float  # 0-1.0
    should_enter_long: bool  # convergence_score >= threshold
    signals: List[Signal] = field(default_factory=list)
    
    def __str__(self) -> str:
        signals_str = ", ".join([f"{s.source}={s.type.name}" for s in self.signals])
        return f"Convergence: {self.buy_count}/{self.total_count} (score={self.convergence_score:.2f}) [{signals_str}]"


class SignalAggregator:
    """Combines signals from multiple indicators with convergence threshold."""
    
    def __init__(self, convergence_threshold: int = 3, total_indicators: int = 3):
        """
        Args:
            convergence_threshold: Minimum # of indicators that must agree (default 3/3 = 100%)
            total_indicators: Total indicators in system (default 3)
        """
        self.convergence_threshold = convergence_threshold
        self.total_indicators = total_indicators
        self.recent_signals: List[Signal] = []
        self.max_history = 100
    
    def add_signal(self, signal: Signal):
        """Add individual indicator signal."""
        self.recent_signals.append(signal)
        # Keep only recent history
        if len(self.recent_signals) > self.max_history:
            self.recent_signals = self.recent_signals[-self.max_history:]
    
    def aggregate(self) -> AggregatedSignal:
        """Aggregate recent signals and score convergence."""
        now = datetime.now()
        
        # Count signals by type from most recent window (last 5 minutes)
        window_start = now - timedelta(minutes=5)
        recent = [s for s in self.recent_signals if s.timestamp >= window_start]
        
        if not recent:
            return AggregatedSignal(
                timestamp=now,
                buy_count=0,
                total_count=self.total_indicators,
                convergence_score=0.0,
                should_enter_long=False
            )
        
        # Get latest signal from each source
        latest_by_source = {}
        for signal in recent:
            latest_by_source[signal.source] = signal
        
        buy_count = sum(1 for s in latest_by_source.values() if s.type == SignalType.BUY)
        total_available = len(latest_by_source)
        
        # Convergence score: higher is better
        convergence_score = buy_count / self.total_indicators if self.total_indicators > 0 else 0
        
        # Should we enter? Check threshold
        should_enter = buy_count >= self.convergence_threshold
        
        return AggregatedSignal(
            timestamp=now,
            buy_count=buy_count,
            total_count=self.total_indicators,
            convergence_score=convergence_score,
            should_enter_long=should_enter,
            signals=list(latest_by_source.values())
        )
    
    def get_signal_history(self, lookback_minutes: int = 60) -> List[Signal]:
        """Get signals from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        return [s for s in self.recent_signals if s.timestamp >= cutoff]


@dataclass
class Position:
    """Active trade position."""
    entry_price: float
    entry_time: datetime
    quantity: int
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    entry_signal_strength: float
    
    def pnl(self, current_price: float) -> float:
        """Unrealized P&L in dollars."""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def pnl_pct(self, current_price: float) -> float:
        """Unrealized P&L in percent."""
        if self.entry_price == 0:
            return 0
        if self.side == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


class TradingEngine:
    """State machine that manages entry/exit logic."""
    
    def __init__(self, initial_capital: int = 10000, risk_per_trade: float = 0.01):
        """
        Args:
            initial_capital: Starting account balance
            risk_per_trade: Max % of capital risked per trade (1% = 0.01)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        self.state = TradeState.IDLE
        self.position: Optional[Position] = None
        
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.max_daily_loss = initial_capital * 0.03  # 3% daily stop
    
    def calculate_position_size(self, current_price: float, stop_loss: float) -> int:
        """Calculate shares to buy based on risk management."""
        risk_amount = self.current_capital * self.risk_per_trade
        risk_per_share = abs(current_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        shares = int(risk_amount / risk_per_share)
        # Don't risk more than 5% of account per trade
        max_shares = int(self.current_capital * 0.05 / current_price)
        return min(shares, max_shares)
    
    def should_enter_long(self, aggregated_signal: AggregatedSignal) -> bool:
        """Check if we should enter a long trade."""
        if self.state != TradeState.IDLE:
            return False
        return aggregated_signal.should_enter_long
    
    def should_exit_position(self, current_price: float) -> Optional[str]:
        """Check exit conditions. Returns reason or None."""
        if self.position is None:
            return None
        
        # Stop loss hit
        if current_price <= self.position.stop_loss:
            return f"Stop loss hit (${self.position.stop_loss:.2f})"
        
        # Take profits
        if current_price >= self.position.take_profit_1:
            return "TP1 reached"
        if current_price >= self.position.take_profit_2:
            return "TP2 reached"
        if current_price >= self.position.take_profit_3:
            return "TP3 reached"
        
        # Time-based exit (45 minutes max hold)
        hold_duration = datetime.now() - self.position.entry_time
        if hold_duration > timedelta(minutes=45):
            return f"Time exit ({hold_duration.total_seconds()/60:.0f} min hold)"
        
        # Daily loss limit
        if self.daily_pnl + self.position.pnl(current_price) < -self.max_daily_loss:
            return "Daily loss limit"
        
        return None
    
    def enter_long(self, current_price: float, aggregated_signal: AggregatedSignal) -> Optional[Position]:
        """Enter a long position."""
        if not self.should_enter_long(aggregated_signal):
            return None
        
        # Calculate position parameters
        stop_loss = current_price * 0.985  # 1.5% stop loss
        tp1 = current_price * 1.010  # 1.0% first target
        tp2 = current_price * 1.015  # 1.5% second target
        tp3 = current_price * 1.020  # 2.0% third target
        
        quantity = self.calculate_position_size(current_price, stop_loss)
        if quantity <= 0:
            return None
        
        self.position = Position(
            entry_price=current_price,
            entry_time=datetime.now(),
            quantity=quantity,
            side='long',
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            entry_signal_strength=aggregated_signal.convergence_score
        )
        
        self.state = TradeState.LONG
        return self.position
    
    def exit_long(self, current_price: float, reason: str = "Manual exit") -> Optional[Dict]:
        """Close a long position."""
        if self.position is None or self.position.side != 'long':
            return None
        
        pnl = self.position.pnl(current_price)
        pnl_pct = self.position.pnl_pct(current_price)
        
        trade_record = {
            'entry_price': self.position.entry_price,
            'exit_price': current_price,
            'quantity': self.position.quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': self.position.entry_time,
            'exit_time': datetime.now(),
            'hold_minutes': (datetime.now() - self.position.entry_time).total_seconds() / 60,
            'reason': reason,
            'signal_strength': self.position.entry_signal_strength
        }
        
        self.trade_history.append(trade_record)
        self.current_capital += pnl
        self.daily_pnl += pnl
        
        self.position = None
        self.state = TradeState.IDLE
        
        return trade_record
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0
            }
        
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': len(wins) / len(self.trade_history) * 100 if self.trade_history else 0,
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'current_capital': self.current_capital
        }


# Demo function
def demo_trade_logic():
    """Demo: Simulate trade entry and exit."""
    engine = TradingEngine(initial_capital=10000, risk_per_trade=0.01)
    aggregator = SignalAggregator(convergence_threshold=3, total_indicators=3)
    
    # Simulate signals
    aggregator.add_signal(Signal('macd', SignalType.BUY, 0.9))
    aggregator.add_signal(Signal('bollinger', SignalType.BUY, 0.8))
    aggregator.add_signal(Signal('stochastic', SignalType.BUY, 0.7))
    
    agg_signal = aggregator.aggregate()
    print(f"Aggregated Signal: {agg_signal}")
    
    # Enter trade
    position = engine.enter_long(132.50, agg_signal)
    print(f"\n✓ Entered Long: {position}")
    
    # Simulate price movement
    prices = [132.50, 132.75, 133.00, 133.20, 133.50]
    for price in prices[1:]:
        exit_reason = engine.should_exit_position(price)
        if exit_reason:
            trade = engine.exit_long(price, exit_reason)
            print(f"✓ Exited: {trade}")
            break
        else:
            pnl = position.pnl(price)
            print(f"  Price: ${price:.2f}, PnL: ${pnl:.2f}")
    
    # Print stats
    stats = engine.get_performance_stats()
    print(f"\nPerformance Stats: {stats}")


if __name__ == '__main__':
    demo_trade_logic()
