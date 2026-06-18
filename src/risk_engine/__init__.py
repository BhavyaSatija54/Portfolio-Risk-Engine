"""
Portfolio Risk Engine

Institutional-grade quantitative risk management:
  * Historical, Parametric, Monte Carlo VaR (Cholesky)
  * GARCH(1,1) volatility forecasting
  * Kupiec / Christoffersen / Basel backtesting
  * Euler risk contribution and portfolio analytics
"""

__version__ = "1.0.0"

from .data_fetcher import DataFetcher, DataStats
from .portfolio import Portfolio
from .var_models import (
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    VaRResult,
)
from .garch_model import GARCHModel, GARCHParams
from .backtest import VaRBacktest, BacktestMetrics
from .visualization import RiskVisualizer

__all__ = [
    "DataFetcher",
    "DataStats",
    "Portfolio",
    "HistoricalVaR",
    "ParametricVaR",
    "MonteCarloVaR",
    "VaRResult",
    "GARCHModel",
    "GARCHParams",
    "VaRBacktest",
    "BacktestMetrics",
    "RiskVisualizer",
]