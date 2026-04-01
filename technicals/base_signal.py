"""
technicals/base_signal.py
===========================
Abstract base class for all Technical Concept (TC) modules in the Draculative system.

PHILOSOPHY
----------
Every technical concept is a self-contained, independently validated module.
No monolithic indicator files. Each TC:
  - Implements the `TechnicalSignal` ABC
  - Exposes a single .compute(df) → Signal method
  - Returns score (float), confidence (float), and metadata (dict)
  - Is validated via Alpaca historical simulation before activation
  - Can be composed with other TCs via the PortfolioSignal aggregator

VALIDATION GATES (mandatory for ACTIVE status)
-----------------------------------------------
A TC is only APPROVED if it passes ALL of:
  - Profit Factor (PF) ≥ 1.20 overall
  - Profit Factor (PF) ≥ 0.90 out-of-sample (OOS)
  - Win Rate (WR) between 35-70%
  - Max Drawdown (DD) < 15%
  - Minimum 8 trades in test period
  - Walk-Forward Efficiency (WFE) ≥ 0.80

LIFECYCLE
---------
PENDING → (validation) → ACTIVE → (rolling OOS degrades) → WATCHLIST → DEACTIVATED

Usage
-----
    from technicals.base_signal import TechnicalSignal
    from technical.bar_snapshot import Signal, Direction
    
    class MyTC(TechnicalSignal):
        def compute(self, df: pd.DataFrame) -> Signal:
            # implementation
            return Signal(...)
        
        def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
            # run Alpaca simulation
            return ValidationResult(...)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Status Enums ──────────────────────────────────────────────────────────────

class TCStatus(str, Enum):
    """Lifecycle status of a Technical Concept."""
    PENDING     = "PENDING"       # Not yet validated
    ACTIVE      = "ACTIVE"        # Passed validation, in production
    WATCHLIST   = "WATCHLIST"     # Performance degrading, halved position size
    DEACTIVATED = "DEACTIVATED"   # Failed validation or sustained poor performance


class TCCategory(str, Enum):
    """Categorization for signal routing and regime weighting."""
    TREND_FOLLOWING    = "TREND_FOLLOWING"      # ADX, Supertrend, MA crossovers
    MEAN_REVERSION     = "MEAN_REVERSION"       # BB, RSI, VWAP deviation
    BREAKOUT           = "BREAKOUT"             # Pivot breaks, liquidity sweeps
    STRUCTURE          = "STRUCTURE"            # FVG, Order Block, ChoCH
    MOMENTUM           = "MOMENTUM"             # PPO, PVO, Stoch RSI
    VOLATILITY         = "VOLATILITY"           # VIX-based, ATR regime
    CONFLUENCE         = "CONFLUENCE"           # Multi-indicator confirmation


# ── Validation Results ────────────────────────────────────────────────────────

@dataclass
class ValidationMetrics:
    """Output from Alpaca live-conditions simulation."""
    total_trades:       int
    win_rate:           float
    profit_factor:      float
    sharpe_ratio:       float
    max_drawdown_pct:   float
    avg_win:            float
    avg_loss:           float
    avg_trade:          float
    trades_per_month:   float
    
    # Walk-forward split results
    oos_profit_factor:  float
    oos_win_rate:       float
    walk_forward_efficiency: float
    
    # Regime attribution (optional)
    trending_pf:        Optional[float] = None
    corrective_pf:      Optional[float] = None
    
    # Execution stats
    avg_slippage_pct:   float = 0.0005
    avg_commission:     float = 1.0
    
    @property
    def passes_gates(self) -> bool:
        """Check if TC passes all validation gates."""
        return (
            self.profit_factor >= 1.20
            and self.oos_profit_factor >= 0.90
            and 0.35 <= self.win_rate <= 0.70
            and self.max_drawdown_pct < 15.0
            and self.total_trades >= 8
            and self.walk_forward_efficiency >= 0.80
        )


@dataclass
class ValidationResult:
    """Complete validation report for a TC."""
    tc_id:              str
    tc_name:            str
    timestamp:          datetime
    data_period:        str              # e.g. "2024-01-01 to 2026-01-01"
    metrics:            ValidationMetrics
    status:             TCStatus
    rejection_reason:   Optional[str] = None
    equity_curve:       Optional[List[float]] = None
    trade_log:          Optional[List[Dict[str, Any]]] = None
    
    @property
    def approved(self) -> bool:
        return self.status == TCStatus.ACTIVE and self.metrics.passes_gates


# ── Abstract Base Class ───────────────────────────────────────────────────────

class TechnicalSignal(ABC):
    """
    Abstract base class for all Technical Concept modules.
    
    Every TC must implement:
      1. compute(df) → Signal
      2. validate(df) → ValidationResult
    
    Optional overrides:
      - MIN_LOOKBACK: minimum bars required before signal generation
      - PARAMS: dict of configurable parameters (loaded from YAML)
    """
    
    # ── Class-level metadata (override in subclass) ───────────────────────────
    TC_ID:       str = "TC-00"                    # e.g. "TC-01"
    TC_NAME:     str = "Generic Signal"
    TC_CATEGORY: TCCategory = TCCategory.TREND_FOLLOWING
    MIN_LOOKBACK: int = 80                        # Default warm-up period
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the TC with configurable parameters.
        
        Parameters
        ----------
        params : dict, optional
            Parameter overrides from config/tc_params.yaml.
            If None, uses class defaults.
        """
        self.params = params or {}
        self.status = TCStatus.PENDING
        self._validation_result: Optional[ValidationResult] = None
    
    # ── Core Interface (must implement) ───────────────────────────────────────
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> "Signal":  # type: ignore
        """
        Compute the signal for the current bar (last row of df).
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data enriched with indicators (via indicators_v4.enrich_dataframe).
            Only data UP TO AND INCLUDING the current bar is visible.
            NO FUTURE DATA. This is the bar-by-bar live view.
        
        Returns
        -------
        Signal
            From technical.bar_snapshot import Signal, Direction
            Signal must contain:
              - direction: Direction.LONG | Direction.SHORT | Direction.FLAT
              - strength: SignalStrength enum
              - score: float (-1.0 to +1.0)
              - confidence: float (0.0 to 1.0)
              - stop_loss: float (price level)
              - take_profit: float (price level)
              - metadata: dict (concept-specific context)
        """
        raise NotImplementedError(f"{self.__class__.__name__}.compute() not implemented")
    
    @abstractmethod
    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        """
        Run full Alpaca live-conditions simulation and return validation metrics.
        
        This method:
          1. Loads 2 years of Alpaca historical data for the symbol(s)
          2. Runs bar-by-bar simulation with realistic execution:
             - Slippage: 0.05% per side
             - Commission: $0.005/share, min $1.00
             - Position sizing: 1% risk per trade (ATR-based)
             - Orders fill at NEXT bar open (no lookahead)
             - EOD exit at 15:45 for intraday signals
          3. Walk-forward split: 60% train / 20% validate / 20% test
          4. Computes all validation metrics
          5. Returns ValidationResult with pass/fail determination
        
        Parameters
        ----------
        historical_df : pd.DataFrame
            Pre-loaded Alpaca historical data (2 years minimum).
            If None, subclass must load data internally.
        
        Returns
        -------
        ValidationResult
            Complete validation report with metrics and pass/fail status.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.validate() not implemented")
    
    # ── Lifecycle Management ──────────────────────────────────────────────────
    
    def mark_active(self) -> None:
        """Mark this TC as ACTIVE (passed validation)."""
        self.status = TCStatus.ACTIVE
    
    def mark_watchlist(self, reason: str = "") -> None:
        """Move to WATCHLIST (performance degrading)."""
        self.status = TCStatus.WATCHLIST
        if self._validation_result:
            self._validation_result.rejection_reason = f"WATCHLIST: {reason}"
    
    def deactivate(self, reason: str = "") -> None:
        """Deactivate this TC (failed validation or sustained poor performance)."""
        self.status = TCStatus.DEACTIVATED
        if self._validation_result:
            self._validation_result.rejection_reason = f"DEACTIVATED: {reason}"
    
    # ── Metadata ──────────────────────────────────────────────────────────────
    
    @property
    def validation_result(self) -> Optional[ValidationResult]:
        """Get cached validation result."""
        return self._validation_result
    
    def set_validation_result(self, result: ValidationResult) -> None:
        """Store validation result and update status."""
        self._validation_result = result
        self.status = result.status
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize TC metadata for registry."""
        return {
            "tc_id": self.TC_ID,
            "tc_name": self.TC_NAME,
            "tc_category": self.TC_CATEGORY.value,
            "status": self.status.value,
            "min_lookback": self.MIN_LOOKBACK,
            "params": self.params,
            "validation": self._validation_result.__dict__ if self._validation_result else None,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.TC_ID}, status={self.status.value})"


# ── Helper: Signal from existing bar_snapshot.Signal ──────────────────────────

# NOTE: The Signal class already exists in technical/bar_snapshot.py
# We import it here for convenience and type hints
try:
    from technical.bar_snapshot import Signal, Direction, SignalStrength
except ImportError:
    # Fallback if running standalone
    class Direction(str, Enum):  # type: ignore
        LONG = "LONG"
        SHORT = "SHORT"
        FLAT = "FLAT"
    
    class SignalStrength(str, Enum):  # type: ignore
        STRONG_BUY = "STRONG_BUY"
        BUY = "BUY"
        HOLD = "HOLD"
        SELL = "SELL"
        STRONG_SELL = "STRONG_SELL"
    
    @dataclass
    class Signal:  # type: ignore
        direction: Direction
        strength: SignalStrength
        score: float
        confidence: float
        stop_loss: float
        take_profit: float
        metadata: Dict[str, Any] = field(default_factory=dict)
