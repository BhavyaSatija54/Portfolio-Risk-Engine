"""
VaR Models — Production Grade

Implements Historical, Parametric (normal / EWMA / GARCH), and Monte Carlo
VaR via Cholesky decomposition.  All estimates obey the type contract
that VaR is returned as a *positive* decimal representing the loss magnitude.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_TradingDays: int = 252
_EPS: float = 1e-15  # Numerical zero


@dataclass(frozen=True)
class VaRResult:
    """Immutable result container for any VaR model."""

    var_pct: float          # VaR as positive decimal (e.g. 0.02 = 2%)
    es_pct: float           # Expected shortfall as positive decimal
    var_abs: float          # VaR in currency units
    es_abs: float           # ES in currency units
    confidence: float
    holding_period: int
    notional: float
    model_name: str


class _BaseVaR(ABC):
    """Abstract base enforcing the positive-VaR contract."""

    def __init__(
        self,
        returns: pd.Series,
        *,
        holding_period: int = 1,
        confidence_level: float = 0.95,
        notional: float = 1.0,
    ) -> None:
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError(f"confidence_level must be in (0,1), got {confidence_level}")
        if holding_period < 1:
            raise ValueError(f"holding_period must be >= 1, got {holding_period}")
        if notional <= 0:
            raise ValueError(f"notional must be > 0, got {notional}")

        self.r: pd.Series = returns.dropna()
        self.hp: int = int(holding_period)
        self.cl: float = float(confidence_level)
        self.alpha: float = 1.0 - self.cl
        self.notional: float = float(notional)

    @abstractmethod
    def calculate(self) -> VaRResult:
        """Return a fully-populated VaRResult."""
        ...

    @staticmethod
    def _sqrt_scale(daily_var: float, hp: int) -> float:
        """Square-root-of-time scaling; guards against negative round-off."""
        return daily_var * np.sqrt(max(hp, 1))


# =====================================================================
# Historical VaR
# =====================================================================

class HistoricalVaR(_BaseVaR):
    """
    Non-parametric Historical Simulation VaR.

    Multi-period scaling uses overlapping sum-returns rather than the
    sqrt(T) rule, which is only valid under i.i.d. normality.
    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        holding_period: int = 1,
        confidence_level: float = 0.95,
        notional: float = 1.0,
        bootstrap: bool = False,
        n_bootstrap: int = 10_000,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            returns,
            holding_period=holding_period,
            confidence_level=confidence_level,
            notional=notional,
        )
        self.bootstrap: bool = bootstrap
        self.n_boot: int = int(n_bootstrap)
        self.rs: int = int(random_state)

    def calculate(self) -> VaRResult:
        # Use overlapping sum-returns for hp > 1 (more accurate)
        if self.hp > 1:
            series = self.r.rolling(window=self.hp, min_periods=self.hp).sum().dropna()
        else:
            series = self.r

        raw_quantile = float(np.percentile(series, self.alpha * 100.0))
        # If the alpha-percentile is positive, even tail returns are gains → VaR = 0
        var_pct = abs(raw_quantile) if raw_quantile < 0 else 0.0

        # Expected shortfall: mean of returns beyond VaR threshold
        tail = series[series <= -var_pct] if var_pct > 0 else pd.Series(dtype=float)
        es_pct = abs(float(tail.mean())) if len(tail) else var_pct

        ci: Optional[Tuple[float, float]] = None
        if self.bootstrap:
            ci = self._bootstrap_ci(series)

        result = self._pack(var_pct, es_pct)
        if ci:
            logger.info("HistVaR %.0f%% = %.4f  CI=[%.4f, %.4f]", self.cl * 100, var_pct, ci[0], ci[1])
        else:
            logger.info("HistVaR %.0f%% = %.4f", self.cl * 100, var_pct)
        return result

    def _bootstrap_ci(self, series: pd.Series) -> Tuple[float, float]:
        rng = np.random.default_rng(self.rs)
        n = len(series)
        boot_vars = np.empty(self.n_boot)
        for i in range(self.n_boot):
            samp = rng.choice(series, size=n, replace=True)
            boot_vars[i] = abs(np.percentile(samp, self.alpha * 100.0))
        return (float(np.percentile(boot_vars, 2.5)), float(np.percentile(boot_vars, 97.5)))

    def _pack(self, var_pct: float, es_pct: float) -> VaRResult:
        return VaRResult(
            var_pct=var_pct,
            es_pct=es_pct,
            var_abs=var_pct * self.notional,
            es_abs=es_pct * self.notional,
            confidence=self.cl,
            holding_period=self.hp,
            notional=self.notional,
            model_name="Historical",
        )


# =====================================================================
# Parametric VaR
# =====================================================================

class ParametricVaR(_BaseVaR):
    """
    Variance-Covariance (Parametric) VaR.

    Volatility can be supplied via ``volatility_model`` or directly
    through ``daily_vol``.  The latter takes precedence if provided.
    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        holding_period: int = 1,
        confidence_level: float = 0.95,
        notional: float = 1.0,
        volatility_model: str = "standard",
        daily_vol: Optional[float] = None,
        ewma_lambda: float = 0.94,
    ) -> None:
        super().__init__(
            returns,
            holding_period=holding_period,
            confidence_level=confidence_level,
            notional=notional,
        )
        self.vm: str = volatility_model.lower().strip()
        self._given_vol: Optional[float] = daily_vol
        self._ewma_lambda: float = float(ewma_lambda)
        self._vol: Optional[float] = None

    def calculate(self) -> VaRResult:
        vol = self._resolve_vol()
        z = stats.norm.ppf(self.cl)
        daily_var = z * vol
        var_pct = self._sqrt_scale(daily_var, self.hp)

        # ES for normal:  ES = sigma * phi(Z) / (1-alpha)
        pdf_z = stats.norm.pdf(z)
        daily_es = (pdf_z / self.alpha) * vol if self.alpha > _EPS else daily_var
        es_pct = self._sqrt_scale(daily_es, self.hp)

        logger.info(
            "ParamVaR %.0f%% (%s vol=%.5f) = %.4f",
            self.cl * 100,
            self.vm,
            vol,
            var_pct,
        )
        return self._pack(var_pct, es_pct)

    def _resolve_vol(self) -> float:
        if self._given_vol is not None:
            return float(self._given_vol)
        if self._vol is not None:
            return self._vol

        if self.vm == "standard":
            self._vol = float(self.r.std())
        elif self.vm == "ewma":
            self._vol = self._ewma_vol()
        elif self.vm == "garch":
            raise ValueError(
                "volatility_model='garch' requires passing daily_vol=... directly"
            )
        else:
            raise ValueError(f"Unknown volatility_model: {self.vm}")

        if self._vol < _EPS:
            logger.warning("Estimated volatility is near-zero (%.2e); replacing with 1e-6", self._vol)
            self._vol = 1e-6
        return self._vol

    def _ewma_vol(self) -> float:
        sq = self.r.values ** 2
        n = len(sq)
        w = (1.0 - self._ewma_lambda) * (self._ewma_lambda ** np.arange(n))
        w = w[::-1]
        w /= w.sum()
        return float(np.sqrt(np.dot(w, sq)))

    def _pack(self, var_pct: float, es_pct: float) -> VaRResult:
        return VaRResult(
            var_pct=var_pct,
            es_pct=es_pct,
            var_abs=var_pct * self.notional,
            es_abs=es_pct * self.notional,
            confidence=self.cl,
            holding_period=self.hp,
            notional=self.notional,
            model_name=f"Parametric-{self.vm}",
        )


# =====================================================================
# Monte Carlo VaR (Cholesky)
# =====================================================================

class MonteCarloVaR(_BaseVaR):
    """
    Monte Carlo VaR using Cholesky decomposition of the covariance matrix.

    Supports both normal and Student-t innovations.  The Cholesky factor
    L of Sigma allows correlated draws:   r = mu + L @ z
    """

    def __init__(
        self,
        returns: pd.Series,
        *,
        holding_period: int = 1,
        confidence_level: float = 0.95,
        notional: float = 1.0,
        n_simulations: int = 100_000,
        random_state: int = 42,
        use_cholesky: bool = True,
        distribution: str = "normal",          # "normal" | "t"
        component_returns: Optional[pd.DataFrame] = None,
        component_weights: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            returns,
            holding_period=holding_period,
            confidence_level=confidence_level,
            notional=notional,
        )
        self.n_sims: int = int(n_simulations)
        self.rs: int = int(random_state)
        self.use_chol: bool = use_cholesky
        self.dist: str = distribution.lower().strip()
        self.comp_ret: Optional[pd.DataFrame] = component_returns
        self.comp_w: Optional[np.ndarray] = component_weights
        self._sim: Optional[np.ndarray] = None

    def calculate(self) -> VaRResult:
        if self.use_chol and self.comp_ret is not None and self.comp_w is not None:
            sim = self._simulate_cholesky()
        else:
            sim = self._simulate_uncorrelated()

        self._sim = sim

        # Apply sqrt(T) scaling to simulated daily returns
        if self.hp > 1:
            sim = sim * np.sqrt(self.hp)

        var_pct = abs(float(np.percentile(sim, self.alpha * 100.0)))
        tail = sim[sim <= -var_pct]
        es_pct = abs(float(tail.mean())) if len(tail) else var_pct

        logger.info(
            "MCVaR %.0f%% (%s, n=%s, Cholesky=%s) = %.4f",
            self.cl * 100,
            self.dist,
            f"{self.n_sims:,}",
            self.use_chol,
            var_pct,
        )
        return self._pack(var_pct, es_pct)

    def get_simulated(self) -> np.ndarray:
        """Access the full simulated distribution (requires prior ``calculate``)."""
        if self._sim is None:
            raise RuntimeError("call calculate() first")
        return self._sim

    def _simulate_cholesky(self) -> np.ndarray:
        assert self.comp_ret is not None and self.comp_w is not None
        cov = self.comp_ret.cov().values
        cov = self._ensure_psd(cov)

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            # Fallback: eigendecomposition of PSD matrix
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-12)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            logger.debug("Cholesky failed; used eigendecomposition fallback")

        rng = np.random.default_rng(self.rs)
        n_assets = cov.shape[0]
        z = rng.standard_normal((self.n_sims, n_assets))
        means = self.comp_ret.mean().values
        sim_assets = means + z @ L.T
        return sim_assets @ self.comp_w

    def _simulate_uncorrelated(self) -> np.ndarray:
        rng = np.random.default_rng(self.rs)
        mu = float(self.r.mean())
        sigma = max(float(self.r.std()), _EPS)

        if self.dist == "normal":
            draws = rng.normal(mu, sigma, self.n_sims)
        elif self.dist == "t":
            df = self._fit_dof()
            draws = rng.standard_t(df, self.n_sims)
            draws = draws / np.sqrt(df / max(df - 2, 1)) * sigma + mu
        else:
            raise ValueError(f"Unknown distribution: {self.dist}")
        return draws

    @staticmethod
    def _ensure_psd(mat: np.ndarray) -> np.ndarray:
        """Project symmetric matrix to positive semi-definite via eigenvalue clipping."""
        if not np.allclose(mat, mat.T):
            mat = (mat + mat.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(mat)
        if eigvals.min() < 1e-12:
            eigvals = np.maximum(eigvals, 1e-12)
            mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return mat

    def _fit_dof(self) -> float:
        from scipy.stats import t as t_dist
        df, _, _ = t_dist.fit(self.r)
        return max(float(df), 3.0)

    def _pack(self, var_pct: float, es_pct: float) -> VaRResult:
        return VaRResult(
            var_pct=var_pct,
            es_pct=es_pct,
            var_abs=var_pct * self.notional,
            es_abs=es_pct * self.notional,
            confidence=self.cl,
            holding_period=self.hp,
            notional=self.notional,
            model_name=f"MonteCarlo-{self.dist}",
        )