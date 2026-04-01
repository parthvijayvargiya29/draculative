"""Layer 4: Order Execution - Broker API, risk management, exit management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
import json


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Single order."""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: int
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_qty': self.filled_qty,
            'filled_price': self.filled_price,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BracketOrder:
    """Bracket order (entry + profit targets + stop loss)."""
    entry_order: Order
    profit_target_1: Order  # Sell 50% at TP1
    profit_target_2: Order  # Sell 30% at TP2
    profit_target_3: Order  # Sell 20% at TP3 with trailing stop
    stop_loss: Order
    
    def all_orders(self) -> List[Order]:
        return [self.entry_order, self.profit_target_1, self.profit_target_2, 
                self.profit_target_3, self.stop_loss]


class BrokerInterface:
    """Abstract broker interface (Alpaca, IB, etc.)."""
    
    def __init__(self, broker_type: str = "alpaca", api_key: str = "", api_secret: str = "", paper_trading: bool = True):
        """
        Args:
            broker_type: 'alpaca' or 'ib'
            api_key: Broker API key
            api_secret: Broker API secret
            paper_trading: If True, use paper trading
        """
        self.broker_type = broker_type
        self.paper_trading = paper_trading
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD-{self.order_counter:08d}"
    
    def submit_order(self, symbol: str, order_type: OrderType, side: str, quantity: int,
                    price: Optional[float] = None) -> Order:
        """Submit a single order (stub - in production, calls real broker API)."""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price
        )
        self.pending_orders[order.order_id] = order
        
        # Simulate immediate market order fill for demo
        if order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_qty = quantity
            order.filled_price = price if price else 0
            self.filled_orders[order.order_id] = order
            del self.pending_orders[order.order_id]
        
        return order
    
    def submit_bracket_order(self, symbol: str, entry_price: float, entry_qty: int,
                            tp1_price: float, tp2_price: float, tp3_price: float,
                            stop_price: float) -> BracketOrder:
        """Submit entry + profit targets + stop loss as a bracket."""
        entry = self.submit_order(symbol, OrderType.MARKET, 'buy', entry_qty, entry_price)
        
        # Profit targets
        tp1 = self.submit_order(symbol, OrderType.LIMIT, 'sell', int(entry_qty * 0.5), tp1_price)
        tp2 = self.submit_order(symbol, OrderType.LIMIT, 'sell', int(entry_qty * 0.3), tp2_price)
        tp3 = self.submit_order(symbol, OrderType.STOP, 'sell', int(entry_qty * 0.2), stop_price)
        
        # Stop loss
        stop = self.submit_order(symbol, OrderType.STOP, 'sell', entry_qty, stop_price)
        
        return BracketOrder(entry, tp1, tp2, tp3, stop)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id in self.pending_orders:
            self.pending_orders[order_id].status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self.pending_orders.get(order_id) or self.filled_orders.get(order_id)
    
    def get_all_orders(self) -> List[Order]:
        """Get all orders (pending + filled)."""
        return list(self.pending_orders.values()) + list(self.filled_orders.values())
    
    def get_account_balance(self) -> float:
        """Get account balance (stub - would fetch from broker)."""
        return 10000.0  # Placeholder


class RiskManager:
    """Position sizing and risk control."""
    
    def __init__(self, max_risk_per_trade: float = 0.01, max_daily_loss_pct: float = 0.03,
                 account_balance: float = 10000):
        """
        Args:
            max_risk_per_trade: Max % of account risked per trade
            max_daily_loss_pct: Max daily loss before stopping trading
            account_balance: Starting account balance
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_loss = account_balance * max_daily_loss_pct
        self.account_balance = account_balance
        self.daily_pnl = 0.0
        self.open_positions = 0
    
    def can_open_position(self) -> bool:
        """Check if we can open another position."""
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False
        # Check max concurrent positions (allow 1-2)
        if self.open_positions >= 2:
            return False
        return True
    
    def calculate_position_size(self, account: float, entry_price: float, stop_price: float) -> int:
        """Calculate position size based on risk."""
        risk_amount = account * self.max_risk_per_trade
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        return int(risk_amount / risk_per_share)
    
    def record_trade_pnl(self, pnl: float):
        """Record P&L from closed trade."""
        self.daily_pnl += pnl
        self.account_balance += pnl
    
    def reset_daily_stats(self):
        """Reset daily stats (call at market open)."""
        self.daily_pnl = 0.0
    
    def get_status(self) -> Dict:
        """Get risk status."""
        return {
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'max_daily_loss': self.max_daily_loss,
            'daily_loss_remaining': self.max_daily_loss + self.daily_pnl,  # How much loss buffer left
            'can_trade': self.can_open_position(),
            'open_positions': self.open_positions
        }


class ExitManager:
    """Manages exit strategies (profit targets, stops, time exits)."""
    
    def __init__(self):
        self.active_exits: List[Dict] = []
    
    def add_exit_condition(self, order_id: str, condition_type: str, trigger_price: Optional[float] = None,
                          hold_minutes: Optional[int] = None, reason: str = ""):
        """
        Add an exit condition.
        condition_type: 'stop_loss', 'take_profit', 'time_exit', 'trailing_stop'
        """
        self.active_exits.append({
            'order_id': order_id,
            'type': condition_type,
            'trigger_price': trigger_price,
            'hold_minutes': hold_minutes,
            'reason': reason,
            'created_at': datetime.now()
        })
    
    def check_exits(self, current_price: float, hold_minutes: float) -> List[Dict]:
        """Check which exit conditions are triggered."""
        triggered = []
        
        for exit_cond in self.active_exits:
            if exit_cond['type'] == 'stop_loss' and current_price <= exit_cond['trigger_price']:
                triggered.append(exit_cond)
            elif exit_cond['type'] == 'take_profit' and current_price >= exit_cond['trigger_price']:
                triggered.append(exit_cond)
            elif exit_cond['type'] == 'time_exit' and hold_minutes >= exit_cond['hold_minutes']:
                triggered.append(exit_cond)
            elif exit_cond['type'] == 'trailing_stop':
                # Trailing stop logic
                pass
        
        return triggered
    
    def remove_exit_condition(self, order_id: str):
        """Remove exit condition (when position closed)."""
        self.active_exits = [e for e in self.active_exits if e['order_id'] != order_id]


class OrderGenerator:
    """Generates orders based on trade signals."""
    
    def __init__(self, broker: BrokerInterface, risk_manager: RiskManager):
        self.broker = broker
        self.risk_manager = risk_manager
    
    def generate_entry_order(self, symbol: str, entry_price: float, signal_strength: float) -> Optional[Order]:
        """Generate entry order based on signal."""
        if not self.risk_manager.can_open_position():
            return None
        
        # Calculate position size
        stop_loss = entry_price * 0.985
        quantity = self.risk_manager.calculate_position_size(
            self.risk_manager.account_balance, entry_price, stop_loss
        )
        
        if quantity <= 0:
            return None
        
        order = self.broker.submit_order(symbol, OrderType.MARKET, 'buy', quantity, entry_price)
        self.risk_manager.open_positions += 1
        
        return order
    
    def generate_bracket_order(self, symbol: str, entry_price: float, quantity: int) -> Optional[BracketOrder]:
        """Generate bracket order with profit targets and stop loss."""
        tp1 = entry_price * 1.010
        tp2 = entry_price * 1.015
        tp3 = entry_price * 1.020
        stop = entry_price * 0.985
        
        bracket = self.broker.submit_bracket_order(symbol, entry_price, quantity, tp1, tp2, tp3, stop)
        return bracket


# Demo function
def demo_order_execution():
    """Demo: Execute orders and manage risk."""
    broker = BrokerInterface(broker_type='alpaca', paper_trading=True)
    risk_mgr = RiskManager(max_risk_per_trade=0.01, account_balance=10000)
    order_gen = OrderGenerator(broker, risk_mgr)
    
    print("\n📋 ORDER EXECUTION DEMO")
    print(f"Account Balance: ${risk_mgr.account_balance:,.2f}")
    print(f"Max Risk Per Trade: {risk_mgr.max_risk_per_trade*100:.1f}%")
    
    # Generate entry order
    entry = order_gen.generate_entry_order('NVDA', 132.50, 0.9)
    print(f"\nEntry Order: {entry.to_dict()}")
    
    # Generate bracket
    bracket = order_gen.generate_bracket_order('NVDA', 132.50, 100)
    print(f"\nBracket Orders:")
    for order in bracket.all_orders():
        print(f"  {order.side.upper()} {order.quantity} @ ${order.price}: {order.status.value}")
    
    # Check risk status
    print(f"\nRisk Status: {risk_mgr.get_status()}")


if __name__ == '__main__':
    demo_order_execution()
