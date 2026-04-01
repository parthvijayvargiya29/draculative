"""Layer 5: Monitoring & Persistence - Trade logging, metrics, database."""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
import statistics


class TradeLogger:
    """Log trades to CSV and JSON for persistence and analysis."""
    
    def __init__(self, log_dir: str = "trading_system/data"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.log_dir / "trades.csv"
        self.json_file = self.log_dir / "trades.json"
        self.session_file = self.log_dir / "session.json"
        
        # Initialize CSV if needed
        if not self.csv_file.exists():
            self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV with headers."""
        headers = [
            'timestamp', 'entry_time', 'exit_time', 'symbol', 'side', 'quantity',
            'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'hold_minutes',
            'exit_reason', 'signal_strength', 'win'
        ]
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def log_trade(self, trade: Dict):
        """Log a closed trade."""
        trade['timestamp'] = datetime.now().isoformat()
        trade['win'] = 1 if trade.get('pnl', 0) > 0 else 0
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade.keys())
            writer.writerow(trade)
        
        # Append to JSON
        trades = self._load_json_trades()
        trades.append(trade)
        with open(self.json_file, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
    
    def _load_json_trades(self) -> List[Dict]:
        """Load all trades from JSON."""
        if not self.json_file.exists():
            return []
        with open(self.json_file) as f:
            return json.load(f)
    
    def get_trades(self, lookback_hours: Optional[int] = None) -> List[Dict]:
        """Get trades from the last N hours."""
        trades = self._load_json_trades()
        
        if lookback_hours:
            cutoff = datetime.now() - timedelta(hours=lookback_hours)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) >= cutoff]
        
        return trades
    
    def save_session_state(self, state: Dict):
        """Save session state for recovery."""
        state['last_save'] = datetime.now().isoformat()
        with open(self.session_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_session_state(self) -> Dict:
        """Load last session state."""
        if not self.session_file.exists():
            return {}
        with open(self.session_file) as f:
            return json.load(f)


class PerformanceMetrics:
    """Calculate real-time performance metrics."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.trades: List[Dict] = []
    
    def add_trade(self, trade: Dict):
        """Add a closed trade to metrics."""
        self.trades.append(trade)
    
    def calculate(self, current_capital: float = None) -> Dict:
        """Calculate all performance metrics."""
        if not self.trades:
            return self._empty_metrics(current_capital)
        
        # Basic counts
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) < 0]
        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        
        # P&L
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) for t in wins)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average metrics
        avg_win = statistics.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = statistics.mean([t['pnl'] for t in losses]) if losses else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        
        # Drawdown (max cumulative loss from peak)
        cumulative = []
        cum_sum = 0
        for trade in self.trades:
            cum_sum += trade.get('pnl', 0)
            cumulative.append(cum_sum)
        
        max_profit = max(cumulative) if cumulative else 0
        max_drawdown = min([cp - max_profit for cp in cumulative]) if cumulative else 0
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Hold times
        hold_times = [t.get('hold_minutes', 0) for t in self.trades]
        avg_hold = statistics.mean(hold_times) if hold_times else 0
        
        # Sharpe ratio (simplified, assumes trades spread over time)
        returns = [t.get('pnl_pct', 0) for t in self.trades]
        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            # Annualize assuming daily trades
            sharpe = (avg_return * 252 / 100) / (std_return * (252 ** 0.5)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate_pct': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy_per_trade': round(expectancy, 2),
            'avg_hold_minutes': round(avg_hold, 1),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe, 2),
            'current_capital': round(current_capital, 2) if current_capital else None
        }
    
    def _empty_metrics(self, current_capital: float = None) -> Dict:
        """Return empty metrics."""
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy_per_trade': 0,
            'avg_hold_minutes': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'current_capital': current_capital
        }


class SessionRecovery:
    """Handles session recovery if system crashes."""
    
    def __init__(self, logger: TradeLogger):
        self.logger = logger
    
    def save_state(self, trading_engine, broker, risk_manager):
        """Save current state before shutdown."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'engine_state': trading_engine.state.value if trading_engine.state else None,
            'open_position': {
                'entry_price': trading_engine.position.entry_price,
                'entry_time': trading_engine.position.entry_time.isoformat(),
                'quantity': trading_engine.position.quantity,
                'side': trading_engine.position.side,
                'stop_loss': trading_engine.position.stop_loss,
                'take_profit_1': trading_engine.position.take_profit_1,
                'take_profit_2': trading_engine.position.take_profit_2,
                'take_profit_3': trading_engine.position.take_profit_3,
            } if trading_engine.position else None,
            'current_capital': trading_engine.current_capital,
            'daily_pnl': trading_engine.daily_pnl,
        }
        self.logger.save_session_state(state)
        return state
    
    def load_state(self) -> Dict:
        """Load last known state."""
        return self.logger.load_session_state()
    
    def restore_from_crash(self, trading_engine, broker, risk_manager) -> bool:
        """Attempt to restore position after crash."""
        state = self.load_state()
        if not state or not state.get('open_position'):
            return False
        
        pos = state['open_position']
        # Restore position object
        from trade_logic import Position
        trading_engine.position = Position(
            entry_price=pos['entry_price'],
            entry_time=datetime.fromisoformat(pos['entry_time']),
            quantity=pos['quantity'],
            side=pos['side'],
            stop_loss=pos['stop_loss'],
            take_profit_1=pos['take_profit_1'],
            take_profit_2=pos['take_profit_2'],
            take_profit_3=pos['take_profit_3'],
            entry_signal_strength=0
        )
        trading_engine.state = 'long'
        trading_engine.current_capital = state['current_capital']
        trading_engine.daily_pnl = state['daily_pnl']
        
        print(f"✓ Recovered position: {pos['quantity']} shares @ ${pos['entry_price']}")
        return True


# Demo function
def demo_monitoring():
    """Demo: Log trades and calculate metrics."""
    logger = TradeLogger()
    metrics = PerformanceMetrics(initial_capital=10000)
    
    print("\n📊 MONITORING & PERSISTENCE DEMO")
    
    # Simulate trades
    trades = [
        {'symbol': 'NVDA', 'entry_price': 132.50, 'exit_price': 133.50, 'quantity': 50, 
         'pnl': 50, 'pnl_pct': 0.75, 'hold_minutes': 25, 'exit_reason': 'TP1', 'signal_strength': 0.9},
        {'symbol': 'NVDA', 'entry_price': 133.50, 'exit_price': 132.75, 'quantity': 50,
         'pnl': -37.50, 'pnl_pct': -0.56, 'hold_minutes': 18, 'exit_reason': 'Stop Loss', 'signal_strength': 0.7},
        {'symbol': 'NVDA', 'entry_price': 133.00, 'exit_price': 134.50, 'quantity': 50,
         'pnl': 75, 'pnl_pct': 1.12, 'hold_minutes': 35, 'exit_reason': 'TP2', 'signal_strength': 0.95},
    ]
    
    for trade in trades:
        logger.log_trade(trade)
        metrics.add_trade(trade)
        print(f"✓ Logged trade: {trade['symbol']} {trade['exit_reason']} PnL: ${trade['pnl']:.2f}")
    
    # Calculate metrics
    perf = metrics.calculate(current_capital=10087.50)
    print(f"\n📈 Performance Metrics:")
    for key, value in perf.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    demo_monitoring()
