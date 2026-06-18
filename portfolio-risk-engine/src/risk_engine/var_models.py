"""
VaR Models — Production Grade

Implements Historical, Parametric (normal / EWMA / GARCH / Student-t),
and Monte Carlo VaR via Cholesky decomposition.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_TradingDays: int = 252
_EPS: float = 1e-15


@dataclass(frozen=True)
class VaRResult:
    """Immutable result container for any VaR model."""

    var_pct: float
    es_pct: float
    var_abs: float
    es_abs: float
    confidence: float
    holding_period: int
    notional: float
    model_name: str
    extra: Dict = None

    def __post_init__(self):
        object.__setattr__(
            self, "extra", self.extra if self.extra is not None else {}
        )


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
        ...

    @staticmethod
    def _sqrt_scale(daily_var: float, hp: int) -> float:
        return daily_var * np.sqrt(max(hp, 1))

    def _pack(self, var_pct: float, es_pct: float, model_name: str, extra: Dict = None) -> VaRResult:
        return VaRResult(
            var_pct=var_pct,
            es_pct=es_pct,
            var_abs=var_pct * self.notional,
            es_abs=es_pct * self.notional,
            confidence=self.cl,
            holding_period=self.hp,
            notional=self.notional,
            model_name=model_name,
            extra=extra,
        )


# =====================================================================
# Historical VaR
# =====================================================================

class HistoricalVaR(_BaseVaR):
    """
    Non-parametric Historical Simulation VaR.

    Supports:
    * Standard historical simulation (empirical quantile)
    * Hull-White age-weighting (exponential decay, recent obs weighted more)
    * Bootstrap confidence intervals
    * Overlapping multi-period sum-returns for hp > 1
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
        age_weighted: bool = False,
        decay: float = 0.99,
    ) -> None:
        super().__init__(
            returns, holding_period=holding_period,
            confidence_level=confidence_level, notional=notional,
        )
        self.bootstrap: bool = bootstrap
        self.n_boot: int = int(n_bootstrap)
        self.rs: int = int(random_state)
        self.age_weighted: bool = age_weighted
        self.decay: float = float(decay)

    def calculate(self) -> VaRResult:
        series = self.r.rolling(window=self.hp, min_periods=self.hp).sum().dropna() if self.hp > 1 else self.r

        if self.age_weighted:
            var_pct, es_pct = self._weighted_var_es(series)
            model_name = "Historical-AgeWeighted"
        else:
            raw_q = float(np.percentile(series, self.alpha * 100.0))
            var_pct = abs(raw_q) if raw_q < 0 else 0.0
            tail = series[series <= -var_pct] if var_pct > 0 else pd.Series(dtype=float)
            es_pct = abs(float(tail.mean())) if len(tail) else var_pct
            model_name = "Historical"

        extra = {"decay": self.decay} if self.age_weighted else {}
        if self.bootstrap:
            ci = self._bootstrap_ci(series)
            extra["ci_lower"] = ci[0]
            extra["ci_upper"] = ci[1]

        return self._pack(var_pct, es_pct, model_name, extra)

    def _weighted_var_es(self, series: pd.Series) -> Tuple[float, float]:
        """Hull-White (1998) exponentially age-weighted historical simulation."""
        r = series.values
        T = len(r)
        lam = self.decay

        # Weights: most recent = highest weight
        raw_w = np.array([lam ** (T - i - 1) for i in range(T)])
        raw_w /= raw_w.sum()

        # Sort by return value
        idx = np.argsort(r)
        sorted_r = r[idx]
        sorted_w = raw_w[idx]

        # Cumulative weight → find alpha quantile
        cum_w = np.cumsum(sorted_w)
        q_idx = min(np.searchsorted(cum_w, self.alpha), len(sorted_r) - 1)
        var_pct = abs(float(sorted_r[q_idx]))

        # ES: weighted mean of tail
        tail_mask = sorted_r <= -var_pct
        if not tail_mask.any():
            es_pct = var_pct
        else:
            es_pct = abs(float(np.average(sorted_r[tail_mask], weights=sorted_w[tail_mask])))

        return var_pct, es_pct

    def _bootstrap_ci(self, series: pd.Series) -> Tuple[float, float]:
        rng = np.random.default_rng(self.rs)
        n = len(series)
        boot_vars = np.empty(self.n_boot)
        for i in range(self.n_boot):
            samp = rng.choice(series, size=n, replace=True)
            raw_q = float(np.percentile(samp, self.alpha * 100.0))
            boot_vars[i] = abs(raw_q) if raw_q < 0 else 0.0
        return float(np.percentile(boot_vars, 2.5)), float(np.percentile(boot_vars, 97.5))


# =====================================================================
# Parametric VaR — Normal, EWMA, GARCH, Student-t
# =====================================================================

class ParametricVaR(_BaseVaR):
    """
    Parametric (Variance-Covariance) VaR.

    Supports volatility models: 'standard', 'ewma', 'garch' (via daily_vol),
    and distribution: 'normal', 't'.
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
        distribution: str = "normal",
    ) -> None:
        super().__init__(
            returns, holding_period=holding_period,
            confidence_level=confidence_level, notional=notional,
        )
        self.vm: str = volatility_model.lower().strip()
        self._given_vol: Optional[float] = daily_vol
        self._ewma_lambda: float = float(ewma_lambda)
        self.dist: str = distribution.lower().strip()
        self._vol: Optional[float] = None

    def calculate(self) -> VaRResult:
        if self.dist == "normal":
            return self._calc_normal()
        elif self.dist == "t":
            return self._calc_t()
        else:
            raise ValueError(f"Unknown distribution: {self.dist}")

    def _calc_normal(self) -> VaRResult:
        vol = self._resolve_vol()
        z = stats.norm.ppf(self.cl)
        daily_var = z * vol
        var_pct = self._sqrt_scale(daily_var, self.hp)

        pdf_z = stats.norm.pdf(z)
        daily_es = (pdf_z / self.alpha) * vol if self.alpha > _EPS else daily_var
        es_pct = self._sqrt_scale(daily_es, self.hp)

        model = f"Parametric-{self.vm}-{self.dist}"
        if self.vm == "garch" or self._given_vol is not None:
            model = "Parametric-GARCH"
        return self._pack(var_pct, es_pct, model,
                         {"volatility_model": self.vm, "daily_vol": vol, "z_score": z})

    def _calc_t(self) -> VaRResult:
        """Student-t VaR with MLE DoF estimation and closed-form ES."""
        r = self.r.values
        nu, loc, scale = stats.t.fit(r, method="MLE")
        nu = max(float(nu), 2.001)

        z = stats.t.ppf(self.alpha, df=nu)
        var_pct = abs(float(loc + scale * z))

        # Closed-form ES for Student-t
        t_pdf_z = stats.t.pdf(z, df=nu)
        es_pct = abs(float(loc - scale * (t_pdf_z / self.alpha) * (nu + z**2) / (nu - 1)))

        # Scale to holding period
        if self.hp > 1:
            var_pct *= np.sqrt(self.hp)
            es_pct *= np.sqrt(self.hp)

        excess_kurt = 6.0 / (nu - 4) if nu > 4 else np.inf
        return self._pack(var_pct, es_pct, "Parametric-Student-t",
                         {"nu": nu, "loc": loc, "scale": scale, "excess_kurtosis": excess_kurt})

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
            raise ValueError("volatility_model='garch' requires passing daily_vol=...")
        else:
            raise ValueError(f"Unknown volatility_model: {self.vm}")

        if self._vol < _EPS:
            self._vol = 1e-6
        return self._vol

    def _ewma_vol(self) -> float:
        sq = self.r.values ** 2
        n = len(sq)
        w = (1.0 - self._ewma_lambda) * (self._ewma_lambda ** np.arange(n))
        w = w[::-1]
        w /= w.sum()
        return float(np.sqrt(np.dot(w, sq)))

    # ------------------------------------------------------------------
    # Rolling VaR (for backtesting)
    # ------------------------------------------------------------------

    def rolling_var_normal(self, window: int = 252) -> pd.Series:
        """Rolling 1-day Normal parametric VaR."""
        z = stats.norm.ppf(self.cl)
        r = self.r.values
        n = len(r)
        var = np.full(n, np.nan)
        for t in range(window, n):
            w = r[t - window:t]
            var[t] = z * w.std(ddof=1)
        return pd.Series(var, index=self.r.index, name=f"ParamNormalVaR_{self.cl:.0%}")

    def rolling_var_t(self, window: int = 252) -> pd.Series:
        """Rolling 1-day Student-t VaR with normal fallback on MLE failure."""
        z_norm = stats.norm.ppf(self.cl)
        r = self.r.values
        n = len(r)
        var = np.full(n, np.nan)
        for t in range(window, n):
            w = r[t - window:t]
            try:
                nu, loc, scale = stats.t.fit(w, method="MLE")
                nu = max(float(nu), 2.001)
                z = stats.t.ppf(self.alpha, df=nu)
                var[t] = float(loc + scale * z)
            except Exception:
                var[t] = z_norm * w.std(ddof=1)
        return pd.Series(var, index=self.r.index, name=f"ParamTVaR_{self.cl:.0%}")

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def test_normality(returns: np.ndarray | pd.Series) -> Dict[str, float]:
        """Jarque-Bera and Shapiro-Wilk tests for normality."""
        r = np.asarray(returns).flatten()
        jb_stat, jb_p = stats.jarque_bera(r)
        # Shapiro-Wilk limited to 5000 samples
        sw_r = r[:min(len(r), 5000)]
        sw_stat, sw_p = stats.shapiro(sw_r)
        return {
            "jb_stat": float(jb_stat), "jb_pval": float(jb_p),
            "sw_stat": float(sw_stat), "sw_pval": float(sw_p),
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
        }


# =====================================================================
# Monte Carlo VaR (Cholesky)
# =====================================================================

class MonteCarloVaR(_BaseVaR):
    """
    Monte Carlo VaR using Cholesky decomposition of the covariance matrix.
    Supports normal and Student-t innovations, GARCH vol scaling,
    multi-day path simulation, and rolling VaR.
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
        distribution: str = "normal",
        component_returns: Optional[pd.DataFrame] = None,
        component_weights: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            returns, holding_period=holding_period,
            confidence_level=confidence_level, notional=notional,
        )
        self.n_sims: int = int(n_simulations)
        self.rs: int = int(random_state)
        self.use_chol: bool = use_cholesky
        self.dist: str = distribution.lower().strip()
        self.comp_ret: Optional[pd.DataFrame] = component_returns
        self.comp_w: Optional[np.ndarray] = component_weights
        self._sim: Optional[np.ndarray] = None

    def calculate(self, garch_vol_vector: Optional[np.ndarray] = None) -> VaRResult:
        if self.use_chol and self.comp_ret is not None and self.comp_w is not None:
            sim = self._simulate_cholesky(garch_vol_vector)
        else:
            sim = self._simulate_uncorrelated()

        self._sim = sim

        if self.hp > 1:
            sim = sim * np.sqrt(self.hp)

        var_pct = abs(float(np.percentile(sim, self.alpha * 100.0)))
        tail = sim[sim <= -var_pct]
        es_pct = abs(float(tail.mean())) if len(tail) else var_pct

        mn = f"MonteCarlo-{self.dist}"
        if garch_vol_vector is not None:
            mn += "-GARCH"
        if self.use_chol:
            mn += "-Cholesky"

        cond_num = float(np.linalg.cond(self.comp_ret.cov())) if self.comp_ret is not None else np.nan
        return self._pack(var_pct, es_pct, mn,
                         {"n_simulations": self.n_sims, "sim_mean": float(sim.mean()),
                          "sim_std": float(sim.std()), "cholesky_cond": cond_num,
                          "tail_obs": len(tail)})

    def get_simulated(self) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("call calculate() first")
        return self._sim

    def _simulate_cholesky(self, garch_vol_vector: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.comp_ret is not None and self.comp_w is not None
        R = np.asarray(self.comp_ret, dtype=float)
        cov = np.cov(R.T, ddof=1)

        # Optional GARCH vol scaling
        if garch_vol_vector is not None:
            hist_sig = R.std(axis=0, ddof=1)
            scale = garch_vol_vector / np.maximum(hist_sig, 1e-12)
            D = np.diag(scale)
            cov = D @ cov @ D

        cov = self._ensure_psd(cov)
        L = self._cholesky(cov)

        rng = np.random.default_rng(self.rs)
        n_assets = cov.shape[0]
        Z = rng.standard_normal((self.n_sims, n_assets))
        mu = R.mean(axis=0)
        X = mu + Z @ L.T
        return X @ self.comp_w

    def _simulate_uncorrelated(self) -> np.ndarray:
        rng = np.random.default_rng(self.rs)
        mu = float(self.r.mean())
        sigma = max(float(self.r.std()), _EPS)

        if self.dist == "normal":
            return rng.normal(mu, sigma, self.n_sims)
        elif self.dist == "t":
            df = self._fit_dof()
            draws = rng.standard_t(df, self.n_sims)
            return draws / np.sqrt(df / max(df - 2, 1)) * sigma + mu
        else:
            raise ValueError(f"Unknown distribution: {self.dist}")

    @staticmethod
    def _ensure_psd(mat: np.ndarray) -> np.ndarray:
        if not np.allclose(mat, mat.T):
            mat = (mat + mat.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(mat)
        if eigvals.min() < 1e-12:
            eigvals = np.maximum(eigvals, 1e-12)
            mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return mat

    def _cholesky(self, Sigma: np.ndarray) -> np.ndarray:
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 1e-10:
            eps = abs(eigvals.min()) + 1e-8
            Sigma = Sigma + eps * np.eye(len(Sigma))
            logger.debug("Covariance regularised with ε=%.2e", eps)
        try:
            return np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, 1e-8)
            Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return np.linalg.cholesky(Sigma)

    def _fit_dof(self) -> float:
        df, _, _ = stats.t.fit(self.r)
        return max(float(df), 3.0)

    # ------------------------------------------------------------------
    # Path simulation for visualisation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        n_days: int = 60,
        n_paths: int = 500,
    ) -> np.ndarray:
        """
        Generate multi-day cumulative portfolio P&L paths.

        Returns (n_paths, n_days) array of cumulative log-returns.
        """
        rng = np.random.default_rng(self.rs)
        R = np.asarray(self.comp_ret if self.comp_ret is not None
                      else self.r.to_frame(), dtype=float)
        w = np.asarray(self.comp_w if self.comp_w is not None
                      else np.array([1.0]))
        w /= w.sum()
        mu = R.mean(axis=0)
        cov = np.cov(R.T, ddof=1)
        L = self._cholesky(cov)
        n = R.shape[1]

        paths = np.zeros((n_paths, n_days))
        for d in range(n_days):
            Z = rng.standard_normal((n_paths, n))
            X = mu + Z @ L.T
            paths[:, d] = X @ w
        return np.cumsum(paths, axis=1)

    # ------------------------------------------------------------------
    # Rolling MC VaR for backtesting
    # ------------------------------------------------------------------

    def rolling_var(
        self,
        window: int = 252,
    ) -> pd.Series:
        """Rolling 1-day MC VaR. Sub-samples to 10k sims for speed."""
        alpha = self.alpha
        w = np.asarray(self.comp_w if self.comp_w is not None else np.array([1.0]))
        w /= w.sum()
        R_full = np.asarray(self.comp_ret if self.comp_ret is not None
                           else self.r.to_frame(), dtype=float)
        n = len(R_full)
        var = np.full(n, np.nan)
        n_sims = min(self.n_sims, 10_000)
        rng = np.random.default_rng(self.rs)

        for t in range(window, n):
            R_t = R_full[t - window:t]
            mu = R_t.mean(axis=0)
            cov = np.cov(R_t.T, ddof=1)
            cov = self._ensure_psd(cov)
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-8)
                L = np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)
            Z = rng.standard_normal((n_sims, R_t.shape[1]))
            rp = (mu + Z @ L.T) @ w
            var[t] = np.percentile(rp, alpha * 100)

        idx = self.comp_ret.index if self.comp_ret is not None else self.r.index
        return pd.Series(var, index=idx, name=f"MCVaR_{self.cl:.0%}")


# =====================================================================
# Convenience: compare all models
# =====================================================================

def compare_var_models(
    returns: pd.Series,
    component_returns: Optional[pd.DataFrame] = None,
    component_weights: Optional[np.ndarray] = None,
    confidence_levels: Sequence[float] = (0.90, 0.95, 0.99),
    holding_periods: Sequence[int] = (1, 5, 10, 21),
    n_simulations: int = 100_000,
    notional: float = 1e6,
) -> pd.DataFrame:
    """Compare all VaR models across multiple confidence levels and holding periods."""
    results = []
    for cl in confidence_levels:
        for hp in holding_periods:
            h = HistoricalVaR(returns, holding_period=hp, confidence_level=cl, notional=notional).calculate()
            p = ParametricVaR(returns, holding_period=hp, confidence_level=cl, notional=notional).calculate()
            g = ParametricVaR(returns, holding_period=hp, confidence_level=cl, notional=notional,
                             volatility_model="ewma").calculate()
            m = MonteCarloVaR(returns, holding_period=hp, confidence_level=cl, notional=notional,
                             n_simulations=n_simulations, random_state=42, use_cholesky=True,
                             component_returns=component_returns, component_weights=component_weights).calculate()
            t = ParametricVaR(returns, holding_period=hp, confidence_level=cl, notional=notional,
                             distribution="t").calculate()
            results.append({
                "CL": cl, "HP": hp,
                "Historical": h.var_abs, "Parametric": p.var_abs,
                "EWMA": g.var_abs, "MonteCarlo": m.var_abs,
                "Student-t": t.var_abs,
            })
    return pd.DataFrame(results)