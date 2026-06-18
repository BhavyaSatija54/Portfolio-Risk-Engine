"""
GARCH(1,1) — From-Scratch Maximum Likelihood Estimation

Educational / interview-grade implementation with:
  * Log-space reparameterisation for positivity
  * Soft penalty for stationarity
  * Dual optimiser fallback (L-BFGS-B → Nelder-Mead)
  * Closed-form multi-step variance forecast
  * GARCH-filtered VaR rescaling
  * Residual diagnostics (Ljung-Box, Jarque-Bera)

For production use, prefer GARCHModel (arch-backed) which is
C-optimised and supports Student-t errors.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, jarque_bera, kurtosis, norm, skew

logger = logging.getLogger(__name__)


@dataclass
class GARCH11Params:
    """GARCH(1,1) parameters."""

    omega: float
    alpha: float
    beta: float
    mu: float = 0.0

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        denom = 1.0 - self.persistence
        return self.omega / denom if denom > 0 else np.nan

    @property
    def unconditional_vol(self) -> float:
        return float(np.sqrt(self.unconditional_variance))

    @property
    def unconditional_vol_ann(self) -> float:
        return self.unconditional_vol * np.sqrt(252)

    @property
    def half_life(self) -> float:
        p = self.persistence
        if 0 < p < 1:
            return float(-np.log(2) / np.log(p))
        return np.nan

    def __repr__(self) -> str:
        return (
            f"GARCH11Params(ω={self.omega:.6f}, α={self.alpha:.4f}, "
            f"β={self.beta:.4f}, pers={self.persistence:.4f})"
        )


@dataclass
class GARCH11Result:
    """Full result container for GARCH(1,1) estimation."""

    params: GARCH11Params
    sigma2: np.ndarray           # conditional variance series
    residuals: np.ndarray        # standardised residuals ε_t / σ_t
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    forecast_sigma: float        # σ_{T+1}
    forecast_var: float          # σ²_{T+1}
    n_obs: int
    extra: Dict = field(default_factory=dict)


class GARCH11:
    """
    GARCH(1,1) estimated by Gaussian MLE from scratch.

    Parameters are optimised in log-space:
        θ = (ln ω, ln α, ln β)

    which enforces positivity without hard constraints.  A soft
    penalty discourages α + β ≥ 1.
    """

    def __init__(
        self,
        max_iter: int = 2_000,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._result: Optional[GARCH11Result] = None

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray | pd.Series) -> GARCH11Result:
        r = np.asarray(returns, dtype=float).flatten()
        mu = float(r.mean())
        eps = r - mu
        T = len(eps)

        # Initial guess: var ≈ σ²(1−α−β) with α=0.05, β=0.90
        var0 = float(np.var(eps, ddof=1))
        omega0 = max(var0 * (1 - 0.05 - 0.90), 1e-8)
        x0 = np.array([np.log(omega0), np.log(0.05), np.log(0.90)])

        # Dual optimiser: L-BFGS-B (fast, gradient) → Nelder-Mead (robust)
        best_res, best_ll = None, np.inf
        for method in ("L-BFGS-B", "Nelder-Mead"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    self._neg_log_lik,
                    x0,
                    args=(eps,),
                    method=method,
                    options={"maxiter": self.max_iter, "ftol": self.tol},
                )
            if res.fun < best_ll:
                best_ll, best_res = res.fun, res

        converged = best_res.success if best_res is not None else False

        # Decode from log-space
        omega = float(np.exp(best_res.x[0]))
        alpha = float(np.exp(best_res.x[1]))
        beta = float(np.exp(best_res.x[2]))

        # Enforce stationarity (post-hoc clip if needed)
        if alpha + beta >= 1.0:
            total = alpha + beta
            alpha = alpha / total * 0.999
            beta = beta / total * 0.999
            logger.warning("GARCH persistence clipped to 0.999 (was %.4f)", total)

        omega = max(omega, 1e-10)
        params = GARCH11Params(omega=omega, alpha=alpha, beta=beta, mu=mu)

        # Reconstruct conditional variance
        sigma2 = self._filter(eps, omega, alpha, beta)
        std_resid = eps / np.sqrt(sigma2)

        k = 3  # ω, α, β
        ll = -float(best_res.fun)
        aic = -2.0 * ll + 2.0 * k
        bic = -2.0 * ll + np.log(T) * k

        # One-step-ahead forecast
        var_T1 = omega + alpha * eps[-1] ** 2 + beta * sigma2[-1]
        sig_T1 = float(np.sqrt(max(var_T1, 1e-10)))

        result = GARCH11Result(
            params=params,
            sigma2=sigma2,
            residuals=std_resid,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            converged=converged,
            forecast_sigma=sig_T1,
            forecast_var=float(var_T1),
            n_obs=T,
            extra={
                "half_life_days": params.half_life,
                "optimiser": best_res.message if best_res is not None else "",
            },
        )
        self._result = result
        if self.verbose:
            self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Multi-step forecast
    # ------------------------------------------------------------------

    def forecast_variance(self, result: GARCH11Result, h: int = 10) -> np.ndarray:
        """
        h-step-ahead conditional variance forecast.

        For stationary GARCH uses the mean-reverting closed form:
            σ²_{T+k} = σ²_∞ + (α+β)^k · (σ²_{T+1} − σ²_∞)
        """
        ω, α, β = result.params.omega, result.params.alpha, result.params.beta
        ab = α + β
        if ab >= 1.0:
            sigma2 = np.empty(h)
            sigma2[0] = result.forecast_var
            for k in range(1, h):
                sigma2[k] = ω + ab * sigma2[k - 1]
        else:
            var_unc = ω / (1 - ab)
            sigma2 = var_unc + (ab ** np.arange(1, h + 1)) * (result.forecast_var - var_unc)
        return sigma2

    def forecast_vol(self, result: GARCH11Result, h: int = 10) -> np.ndarray:
        """Annualised volatility forecast for horizons 1 … h."""
        return np.sqrt(self.forecast_variance(result, h) * 252)

    # ------------------------------------------------------------------
    # GARCH-filtered VaR
    # ------------------------------------------------------------------

    def garch_filtered_var(
        self,
        result: GARCH11Result,
        base_var_pct: float,
        hist_sigma: float,
        confidence: float = 0.95,
        notional: float = 1_000_000,
    ) -> Dict[str, float]:
        """
        Rescale a base VaR by the GARCH vol ratio.

        VaR_GARCH = base_VaR × (σ_{T+1} / σ_hist)
        """
        scale = result.forecast_sigma / max(hist_sigma, 1e-10)
        var_pct = float(base_var_pct * scale)
        alpha = 1.0 - confidence
        z = norm.ppf(alpha)
        es_pct = float(result.forecast_sigma * (-norm.pdf(z) / alpha))
        return {
            "var_pct": var_pct,
            "var_abs": abs(var_pct) * notional,
            "es_pct": es_pct,
            "es_abs": abs(es_pct) * notional,
            "vol_scale": scale,
            "garch_sigma": result.forecast_sigma,
            "hist_sigma": hist_sigma,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def residual_diagnostics(result: GARCH11Result) -> Dict[str, float]:
        """
        Ljung-Box on standardised residuals (z) and squared residuals (z²),
        plus Jarque-Bera test.
        """
        z = result.residuals
        z2 = z ** 2

        def ljung_box(x: np.ndarray, lags: int = 10) -> Tuple[float, float]:
            n = len(x)
            ac = [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, lags + 1)]
            Q = n * (n + 2) * sum(ac[k] ** 2 / (n - k - 1) for k in range(lags))
            p = 1 - chi2.cdf(Q, df=lags)
            return float(Q), float(p)

        lb_z_stat, lb_z_p = ljung_box(z, lags=10)
        lb_z2_stat, lb_z2_p = ljung_box(z2, lags=10)
        jb_stat, jb_p = jarque_bera(z)

        return {
            "lb_z_stat": lb_z_stat, "lb_z_pval": lb_z_p,
            "lb_z2_stat": lb_z2_stat, "lb_z2_pval": lb_z2_p,
            "jb_stat": float(jb_stat), "jb_pval": float(jb_p),
            "skewness": float(skew(z)),
            "kurtosis": float(kurtosis(z)),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        T = len(eps)
        sigma2 = np.empty(T)
        sigma2[0] = np.var(eps, ddof=1)
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)
        return sigma2

    @staticmethod
    def _neg_log_lik(log_params: np.ndarray, eps: np.ndarray) -> float:
        omega = np.exp(log_params[0])
        alpha = np.exp(log_params[1])
        beta = np.exp(log_params[2])

        T = len(eps)
        sigma2 = np.empty(T)
        sigma2[0] = np.var(eps, ddof=1)
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-12)

        if np.any(sigma2 <= 0) or np.any(~np.isfinite(sigma2)):
            return 1e15

        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)

        # Soft penalty for non-stationarity
        ab = alpha + beta
        if ab >= 1.0:
            ll -= 1e6 * (ab - 1.0) ** 2

        return -ll

    @staticmethod
    def _half_life(persistence: float) -> float:
        if persistence <= 0 or persistence >= 1:
            return np.nan
        return float(-np.log(2) / np.log(persistence))

    @staticmethod
    def _print_summary(result: GARCH11Result) -> None:
        p = result.params
        print("\n" + "=" * 50)
        print("  GARCH(1,1) MLE Results (from-scratch)")
        print("=" * 50)
        print(f"  ω: {p.omega:.8f}")
        print(f"  α: {p.alpha:.6f}")
        print(f"  β: {p.beta:.6f}")
        print(f"  α+β: {p.persistence:.6f}")
        print(f"  Long-run σ (ann): {p.unconditional_vol_ann:.2%}")
        print(f"  Half-life: {p.half_life:.1f} days")
        print(f"  Log-Like: {result.log_likelihood:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        print(f"  Forecast σ: {result.forecast_sigma:.6f}")
        print("=" * 50 + "\n")