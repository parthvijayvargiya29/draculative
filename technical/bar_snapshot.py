"""
Core data structures for the Draculative Alpha Engine.
Every concept module, simulator, and engine operates on these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ── ENUMS ────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"

class SignalStrength(str, Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"

class MarketRegime(str, Enum):
    TRENDING    = "TRENDING"
    CORRECTIVE  = "CORRECTIVE"
    UNKNOWN     = "UNKNOWN"

class ApprovalStatus(str, Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class ConceptCategory(str, Enum):
    STRUCTURE        = "STRUCTURE"
    LIQUIDITY        = "LIQUIDITY"
    FVG              = "FVG"
    ORDER_BLOCK      = "ORDER_BLOCK"
    PREMIUM_DISCOUNT = "PREMIUM_DISCOUNT"
    TIME             = "TIME"
    FUNDAMENTAL      = "FUNDAMENTAL"
    ENTRY_MECHANIC   = "ENTRY_MECHANIC"
    EXIT_MECHANIC    = "EXIT_MECHANIC"
    RISK_MANAGEMENT  = "RISK_MANAGEMENT"
    BIAS_FRAMEWORK   = "BIAS_FRAMEWORK"
    CONFLUENCE       = "CONFLUENCE"
    CORRELATION      = "CORRELATION"
    ANOMALY          = "ANOMALY"


# ── CORE DATA TYPES ───────────────────────────────────────────────────────────

@dataclass
class BarSnapshot:
    """
    A single bar of market data as seen by a concept module at time N.
    All indicators are pre-computed up to and including this bar.
    No future data is accessible from this object.
    """
    timestamp:   datetime
    symbol:      str
    open:        float
    high:        float
    low:         float
    close:       float
    volume:      float

    # Pre-computed indicators (computed by indicators_v4.py, then injected)
    atr:         Optional[float] = None
    adx:         Optional[float] = None
    sma_20:      Optional[float] = None
    sma_50:      Optional[float] = None
    sma_200:     Optional[float] = None
    ema_9:       Optional[float] = None
    ema_21:      Optional[float] = None
    rsi_14:      Optional[float] = None
    regime:      MarketRegime    = MarketRegime.UNKNOWN

    # Session info
    session:     str             = "UNKNOWN"   # ASIAN | LONDON | NEW_YORK | OFF
    day_of_week: int             = 0           # 0=Mon … 4=Fri

    # Reference to enriched DataFrame up to this bar (injected by live_simulator)
    # Using Any to avoid circular import with pandas type hints
    df_ref:      Optional[Any]   = field(default=None, repr=False, compare=False)

    # Historical context window (most recent N bars, oldest-first)
    history:     List["BarSnapshot"] = field(default_factory=list)

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def equilibrium(self) -> float:
        """50% of the bar's range — premium/discount reference."""
        return self.low + self.range / 2.0


@dataclass
class Signal:
    """
    Output of a concept module's detect() call.
    One signal = one potential trade idea from one concept.
    """
    concept:     str
    symbol:      str
    direction:   Direction
    timestamp:   datetime
    entry_price: float
    stop_loss:   float
    take_profit: float
    confidence:  float              # 0.0 – 1.0

    category:    ConceptCategory    = ConceptCategory.ENTRY_MECHANIC
    regime:      MarketRegime       = MarketRegime.UNKNOWN
    timeframe:   str                = "1D"
    reason:      str                = ""            # WHY it fired
    metadata:    Dict[str, Any]     = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        reward = abs(self.take_profit - self.entry_price)
        risk   = abs(self.entry_price - self.stop_loss)
        return reward / risk if risk > 0 else 0.0

    @property
    def strength(self) -> SignalStrength:
        if self.confidence >= 0.80:
            return SignalStrength.STRONG_BUY if self.direction == Direction.LONG else SignalStrength.STRONG_SELL
        if self.confidence >= 0.60:
            return SignalStrength.BUY if self.direction == Direction.LONG else SignalStrength.SELL
        return SignalStrength.HOLD


@dataclass
class ValidationResult:
    """Result of running a concept against synthetic or historical test data."""
    concept:        str
    total_tests:    int
    passed:         int
    failed:         int
    edge_cases:     List[str]       = field(default_factory=list)
    notes:          str             = ""

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0


@dataclass
class RegimeMetrics:
    trades:   int   = 0
    win_rate: float = 0.0
    pf:       float = 0.0


@dataclass
class WalkForwardResult:
    pf_train:    float = 0.0
    pf_validate: float = 0.0
    pf_test:     float = 0.0
    wfe:         float = 0.0         # walk-forward efficiency = pf_test / pf_train

    # Extended fields populated by WalkForwardValidator
    concept:          str  = ""
    symbol:           str  = ""
    passes_gate:      bool = False
    slice_results:    Dict[str, Any] = field(default_factory=dict)
    simulation_result: Optional[Any] = field(default=None, repr=False)  # SimulationResult

    @property
    def passed(self) -> bool:
        """Gate: pf_test >= 0.90 * pf_train AND wfe >= 0.60"""
        return self.pf_test >= 0.90 * self.pf_train and self.wfe >= 0.60


@dataclass
class SimulationResult:
    """
    Full output of a concept simulation run.
    Required format per Section 6.2.
    """
    concept:          str
    run_date:         str
    data_period:      Dict[str, str]
    universe:         List[str]

    total_trades:     int   = 0
    win_rate:         float = 0.0
    profit_factor:    float = 0.0
    sharpe_annualized: float = 0.0
    sortino_ratio:    float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio:     float = 0.0
    avg_hold_bars:    float = 0.0
    trades_per_month: float = 0.0

    walk_forward:     WalkForwardResult = field(default_factory=WalkForwardResult)
    regime_trending:  RegimeMetrics     = field(default_factory=RegimeMetrics)
    regime_corrective: RegimeMetrics    = field(default_factory=RegimeMetrics)

    wfe:              float          = 0.0   # walk-forward efficiency (set after WF run)
    approval_status:  ApprovalStatus = ApprovalStatus.PENDING
    rejection_reason: str            = ""
    monthly_breakdown: List[Dict]    = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "run_date": self.run_date,
            "data_period": self.data_period,
            "universe": self.universe,
            "metrics": {
                "total_trades":      self.total_trades,
                "win_rate":          self.win_rate,
                "profit_factor":     self.profit_factor,
                "sharpe_annualized": self.sharpe_annualized,
                "sortino_ratio":     self.sortino_ratio,
                "max_drawdown_pct":  self.max_drawdown_pct,
                "calmar_ratio":      self.calmar_ratio,
                "avg_hold_bars":     self.avg_hold_bars,
                "trades_per_month":  self.trades_per_month,
            },
            "walk_forward": {
                "pf_train":    self.walk_forward.pf_train,
                "pf_validate": self.walk_forward.pf_validate,
                "pf_test":     self.walk_forward.pf_test,
                "wfe":         self.walk_forward.wfe,
                "pass":        self.walk_forward.passed,
            },
            "regime_attribution": {
                "trending":   {"trades": self.regime_trending.trades,
                               "win_rate": self.regime_trending.win_rate,
                               "pf": self.regime_trending.pf},
                "corrective": {"trades": self.regime_corrective.trades,
                               "win_rate": self.regime_corrective.win_rate,
                               "pf": self.regime_corrective.pf},
            },
            "approval_status":  self.approval_status.value,
            "rejection_reason": self.rejection_reason,
            "monthly_breakdown": self.monthly_breakdown,
        }


@dataclass
class ExtractedConcept:
    """
    A structured concept extracted from a transcript by the parser.
    Maps to Section 1.3 taxonomy.
    """
    name:             str
    category:         ConceptCategory
    source_transcript: str
    source_quote:     str
    mechanical_rule:  str
    timeframe:        str
    entry_condition:  str
    invalidation:     str
    edge_rationale:   str
    conflicts_with:   List[str]       = field(default_factory=list)
    confidence:       float           = 1.0
    instruments:      List[str]       = field(default_factory=list)
    occurrences:      int             = 1           # incremented on dedup merge
