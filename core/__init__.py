"""core package — nucleus engine, convergence predictor, signal router, portfolio manager."""
from core.nucleus_engine import NucleusEngine, NucleusState, NucleusType
from core.convergence_predictor import ConvergencePredictor, ConvergenceResult, ConvergenceTier
from core.signal_router import SignalRouter
from core.portfolio_manager import PortfolioManager

__all__ = [
    "NucleusEngine", "NucleusState", "NucleusType",
    "ConvergencePredictor", "ConvergenceResult", "ConvergenceTier",
    "SignalRouter",
    "PortfolioManager",
]
