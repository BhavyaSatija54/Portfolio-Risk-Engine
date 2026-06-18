"""
Portfolio Module — Production Grade

Multi-asset portfolio construction with proper Euler risk allocation,
rolling statistics, and drawdown computation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TradingDays: int = 252


class Portfolio:
    """
    Portfolio analytics with mathematically correct risk attribution.

    Risk contribution follows Euler's homogeneous function theorem:
    for a homogeneous risk measure R of degree 1,
    R(w) = sum_i  w_i * dR/dw_i  = sum_i  CTE_i
    where CTE_i is the contribution to risk (volatility in our case).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        name: str = "Portfolio",
    ) -> None:
        self.returns: pd.DataFrame = returns
        self.tickers: list[str] = list(returns.columns)
        self.name: str = name
        self._w: np.ndarray = self._build_weights(weights)
        self._port_ret: pd.Series = self._compute_portfolio_returns()
        self._cache_stats: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def get_weights(self) -> pd.Series:
        return pd.Series(self._w, index=self.tickers, name="weight")

    def set_weights(
        self, weights: Dict[str, float] | np.ndarray
    ) -> None:
        self._w = self._build_weights(weights)
        self._port_ret = self._compute_portfolio_returns()
        self._cache_stats = None

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def get_portfolio_returns(self) -> pd.Series:
        return self._port_ret

    def get_component_returns(self) -> pd.DataFrame:
        return self.returns

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, float]:
        if self._cache_stats is not None:
            return self._cache_stats

        r = self._port_ret
        ann_ret = float(r.mean() * _TradingDays)
        ann_vol = float(r.std() * np.sqrt(_TradingDays))

        cum = (1.0 + r).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max
        max_dd = float(dd.min())

        var_95 = float(np.percentile(r, 5.0))
        tail = r[r <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else var_95

        self._cache_stats = {
            "n_observations": len(r),
            "total_return": float((1.0 + r).prod() - 1.0),
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": ann_ret / ann_vol if ann_vol > 1e-12 else 0.0,
            "sortino_ratio": self._sortino(r),
            "max_drawdown": max_dd,
            "max_drawdown_date": dd.idxmin(),
            "calmar_ratio": ann_ret / abs(max_dd) if max_dd < 0 else np.inf,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": float(r.skew()),
            "kurtosis": float(r.kurtosis()),
            "tail_ratio_5": self._tail_ratio(r, 0.05),
            "tail_ratio_1": self._tail_ratio(r, 0.01),
        }
        return self._cache_stats

    # ------------------------------------------------------------------
    # Variance & risk contribution (Euler allocation)
    # ------------------------------------------------------------------

    def get_portfolio_variance(self, cov: Optional[np.ndarray] = None) -> float:
        if cov is None:
            cov = self.returns.cov().values
        return float(self._w @ cov @ self._w)

    def get_risk_contribution(self) -> pd.DataFrame:
        """
        Euler allocation of portfolio volatility.

        CTE_i = w_i * (Cov * w)_i / sigma_p
        %CTE_i = CTE_i / sigma_p
        """
        cov = self.returns.cov().values
        sigma_p = np.sqrt(self._w @ cov @ self._w)
        if sigma_p < 1e-15:
            raise ValueError("Portfolio volatility is numerically zero")

        marginal = (cov @ self._w) / sigma_p          # d(sigma)/dw
        cte = self._w * marginal                      # contribution to risk
        pct = cte / sigma_p                           # percentage contribution

        return pd.DataFrame(
            {
                "weight": self._w,
                "marginal_risk": marginal,
                "risk_contrib": cte,
                "pct_contrib": pct,
            },
            index=self.tickers,
        )

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def get_rolling_stats(self, window: int = 252) -> pd.DataFrame:
        r = self._port_ret
        roll_mean = r.rolling(window, min_periods=window // 2).mean() * _TradingDays
        roll_std = r.rolling(window, min_periods=window // 2).std() * np.sqrt(_TradingDays)
        roll_sharpe = roll_mean / roll_std.replace(0, np.nan)

        return pd.DataFrame(
            {
                "ann_return": roll_mean,
                "ann_vol": roll_std,
                "sharpe": roll_sharpe,
            },
            index=r.index,
        )

    def get_drawdown_series(self) -> pd.Series:
        cum = (1.0 + self._port_ret).cumprod()
        running_max = cum.expanding().max()
        return pd.Series(
            (cum - running_max) / running_max,
            index=self._port_ret.index,
            name="drawdown",
        )

    # ------------------------------------------------------------------
    # Summary string
    # ------------------------------------------------------------------

    def summary(self) -> str:
        s = self.get_statistics()
        w = self.get_weights()

        lines = [
            "=" * 56,
            f"  PORTFOLIO SUMMARY: {self.name}",
            "=" * 56,
            "",
            "Weights:",
            "-" * 24,
        ]
        for t, v in w.items():
            lines.append(f"  {t:>6s}  {v:>8.2%}")

        lines += [
            "",
            "Performance (annualized):",
            "-" * 36,
            f"  Total Return      {s['total_return']:>10.2%}",
            f"  Ann. Return       {s['annualized_return']:>10.2%}",
            f"  Ann. Volatility   {s['annualized_volatility']:>10.2%}",
            f"  Sharpe Ratio      {s['sharpe_ratio']:>10.2f}",
            f"  Sortino Ratio     {s['sortino_ratio']:>10.2f}",
            f"  Calmar Ratio      {s['calmar_ratio']:>10.2f}",
            f"  Max Drawdown      {s['max_drawdown']:>10.2%}",
            "",
            "Risk (1-day):",
            "-" * 24,
            f"  VaR  (95%)        {s['var_95']:>10.4f}",
            f"  CVaR (95%)        {s['cvar_95']:>10.4f}",
            "",
            "Distribution:",
            "-" * 24,
            f"  Skewness          {s['skewness']:>10.2f}",
            f"  Kurtosis          {s['kurtosis']:>10.2f}",
            f"  Tail Ratio (5%)   {s['tail_ratio_5']:>10.2f}",
            f"  Tail Ratio (1%)   {s['tail_ratio_1']:>10.2f}",
            "=" * 56,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, n={len(self.tickers)}, "
            f"obs={len(self._port_ret)}, sum_w={self._w.sum():.4f})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_weights(self, weights: Optional[Dict[str, float] | np.ndarray]) -> np.ndarray:
        n = len(self.tickers)
        if weights is None:
            return np.full(n, 1.0 / n)

        if isinstance(weights, np.ndarray):
            arr = weights.astype(float)
        elif isinstance(weights, dict):
            arr = np.array([float(weights.get(t, 0.0)) for t in self.tickers])
        else:
            raise TypeError(f"weights must be dict or ndarray, got {type(weights)}")

        if len(arr) != n:
            raise ValueError(f"weights length {len(arr)} != n_assets {n}")

        w_sum = arr.sum()
        if w_sum < 1e-12:
            raise ValueError("Weights sum to zero")
        return arr / w_sum

    def _compute_portfolio_returns(self) -> pd.Series:
        return pd.Series(
            self.returns.values @ self._w,
            index=self.returns.index,
            name=self.name,
        )

    @staticmethod
    def _sortino(returns: pd.Series) -> float:
        downside = returns[returns < 0].std()
        ann_ret = float(returns.mean() * _TradingDays)
        ann_down = float(downside * np.sqrt(_TradingDays)) if downside > 0 else np.nan
        return ann_ret / ann_down if ann_down and ann_down > 0 else 0.0

    @staticmethod
    def _tail_ratio(returns: pd.Series, alpha: float) -> float:
        """Ratio of upside tail to downside tail at given alpha."""
        return float(
            abs(returns.quantile(1.0 - alpha)) / abs(returns.quantile(alpha))
            if returns.quantile(alpha) != 0
            else np.nan
        )