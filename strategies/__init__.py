"""
Trading Strategies Module.
Contains implementations of various quantitative trading strategies.
"""

from .base import Strategy, Signal
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .pairs_trading import PairsTradingStrategy

__all__ = [
    "Strategy",
    "Signal",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "PairsTradingStrategy"
]
