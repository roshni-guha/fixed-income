"""
G502 Multi-Horizon Momentum Strategy
=====================================

A production-grade momentum model for government bond ticker G502.
Blends binary momentum signals across multiple lookback periods
using optimized weights to produce a single composite signal.
"""

from .data import load_data, compute_returns
from .signals import compute_momentum_signals
from .optimizer import optimize_weights, WalkForwardOptimizer
from .strategy import compute_composite_signal, generate_positions, compute_strategy_returns
from .metrics import compute_metrics, compare_methods

__version__ = "1.0.0"
__all__ = [
    "load_data",
    "compute_returns",
    "compute_momentum_signals",
    "optimize_weights",
    "WalkForwardOptimizer",
    "compute_composite_signal",
    "generate_positions",
    "compute_strategy_returns",
    "compute_metrics",
    "compare_methods",
]
