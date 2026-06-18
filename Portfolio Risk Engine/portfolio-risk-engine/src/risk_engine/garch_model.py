"""
GARCH(1,1) Module — Production Grade

Volatility forecasting via Maximum Likelihood estimation of the
GARCH(1,1) specification.  Returns are scaled to percentages internally
for numerical stability, then converted back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

# arch library works best with returns expressed as percentages
_SCALE: float = 100.0


@dataclass(frozen=True)
class GARCHParams:
    """Immutable parameter struct for GARCH(1,1)."""

    mu: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    half_life: float
    unconditional_variance: float
    unconditional_volatility: float


class GARCHModel:
    """
    GARCH(1,1) with optional Student-t errors.

    Usage
    -----
    model = GARCHModel(returns).fit()
    vol = model.forecast()            # 1-day-ahead volatility
    params = model.params()           # parameter struct
    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        distribution: str = "normal",
        mean: str = "constant",
    ) -> None:
        self.r: pd.Series = returns.dropna()
        if len(self.r) < 60:
            raise ValueError(f"Need >= 60 obs for GARCH, got {len(self.r)}")

        self.dist: str = distribution.lower().strip()
        self.mean_spec: str = mean.lower().strip()
        self._fit_result = None
        self._params: Optional[GARCHParams] = None

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, *, update_freq: int = 0, disp: str = "off") -> "GARCHModel":
        """Fit via Maximum Likelihood."""
        scaled = self.r * _SCALE  # percentage scale

        try:
            mod = arch_model(
                scaled,
                vol="Garch",
                p=1,
                q=1,
                dist=self.dist,
                mean=self.mean_spec,
                rescale=False,
            )
            self._fit_result = mod.fit(
                update_freq=update_freq,
                disp=disp,
                show_warning=False,
            )
        except Exception as exc:
            raise RuntimeError(f"GARCH estimation failed: {exc}") from exc

        self._params = self._extract_params()
        logger.info(
            "GARCH converged=%s  LL=%.1f  AIC=%.1f  persistence=%.4f",
            self._fit_result.convergence_flag == 0,
            self._fit_result.loglikelihood,
            self._fit_result.aic,
            self._params.persistence,
        )
        return self

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        """Forecast conditional variance and volatility."""
        if self._fit_result is None:
            raise RuntimeError("call fit() first")

        fc = self._fit_result.forecast(horizon=horizon, reindex=False)
        var = fc.variance.iloc[-1] / (_SCALE ** 2)
        vol = np.sqrt(var)
        return pd.DataFrame({"variance": var.values, "volatility": vol.values})

    def forecast_vol(self) -> float:
        """Convenience: 1-day-ahead volatility as a decimal."""
        return float(self.forecast(horizon=1)["volatility"].iloc[0])

    def conditional_vol(self) -> pd.Series:
        """Fitted conditional volatility series (daily, decimal)."""
        if self._fit_result is None:
            raise RuntimeError("call fit() first")
        return pd.Series(
            self._fit_result.conditional_volatility / _SCALE,
            index=self.r.index,
        )

    # ------------------------------------------------------------------
    # Parameters & diagnostics
    # ------------------------------------------------------------------

    def params(self) -> GARCHParams:
        if self._params is None:
            raise RuntimeError("call fit() first")
        return self._params

    def loglikelihood(self) -> float:
        self._check_fitted()
        return float(self._fit_result.loglikelihood)

    def aic(self) -> float:
        self._check_fitted()
        return float(self._fit_result.aic)

    def bic(self) -> float:
        self._check_fitted()
        return float(self._fit_result.bic)

    def summary(self) -> str:
        """Human-readable summary."""
        if self._params is None:
            return "GARCH model — not yet fitted."
        p = self._params
        lines = [
            "=" * 50,
            "  GARCH(1,1) ESTIMATION RESULTS",
            "=" * 50,
            "",
            "Parameters:",
            f"  mu         {p.mu:>12.6f}",
            f"  omega      {p.omega:>12.6f}",
            f"  alpha      {p.alpha:>12.6f}",
            f"  beta       {p.beta:>12.6f}",
            "",
            "Properties:",
            f"  Persistence       {p.persistence:>8.4f}",
            f"  Half-life (days)  {p.half_life:>8.1f}",
            f"  Uncond. vol       {p.unconditional_volatility:>8.6f}",
            f"  Log-likelihood    {self.loglikelihood():>8.1f}",
            f"  AIC               {self.aic():>8.1f}",
            f"  BIC               {self.bic():>8.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fit_result is not None else "unfitted"
        return f"GARCHModel(dist={self.dist!r}, status={status})"

    # ------------------------------------------------------------------
    # Rolling forecast
    # ------------------------------------------------------------------

    def rolling_forecast(
        self,
        *,
        window: int = 252,
        step: int = 1,
    ) -> pd.DataFrame:
        """
        Expanding or rolling GARCH forecast.

        Returns DataFrame indexed by forecast date with a
        ``forecast_vol`` column.
        """
        scaled = self.r.values * _SCALE
        n = len(scaled)
        dates = []
        vols = []

        for i in range(window, n, step):
            try:
                mod = arch_model(
                    scaled[i - window : i],
                    vol="Garch",
                    p=1,
                    q=1,
                    dist=self.dist,
                    mean=self.mean_spec,
                    rescale=False,
                )
                fitted = mod.fit(update_freq=0, disp="off", show_warning=False)
                fc = fitted.forecast(horizon=1, reindex=False)
                v = float(np.sqrt(fc.variance.iloc[-1].values[0]) / _SCALE)
            except Exception:
                v = np.nan
            vols.append(v)
            dates.append(self.r.index[i])

        return pd.DataFrame({"forecast_vol": vols}, index=pd.to_datetime(dates))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_params(self) -> GARCHParams:
        assert self._fit_result is not None
        p = self._fit_result.params.to_dict()
        omega = float(p.get("omega", 0.0))
        alpha = float(p.get("alpha[1]", 0.0))
        beta_v = float(p.get("beta[1]", 0.0))
        mu = float(p.get("mu", 0.0))

        persistence = alpha + beta_v
        half_life = (
            float(-np.log(2) / np.log(persistence)) if persistence < 1.0 else np.inf
        )
        if persistence < 1.0 and omega > 0:
            unc_var = omega / (1.0 - persistence)
            unc_vol = float(np.sqrt(unc_var))
        else:
            unc_var = np.nan
            unc_vol = np.nan

        return GARCHParams(
            mu=mu,
            omega=omega,
            alpha=alpha,
            beta=beta_v,
            persistence=persistence,
            half_life=half_life,
            unconditional_variance=float(unc_var),
            unconditional_volatility=unc_vol,
        )

    def _check_fitted(self) -> None:
        if self._fit_result is None:
            raise RuntimeError("call fit() first")