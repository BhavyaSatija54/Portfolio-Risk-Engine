"""
Backtesting Module — Production Grade

Statistical validation of VaR models via:
  * Kupiec (1995) Proportion-of-Failures (LR_POF)
  * Christoffersen (1998) Independence test (LR_IND)
  * Christoffersen Conditional Coverage joint test (LR_CC)
  * Basel regulatory traffic light (per-250-day and binomial CDF)

Multi-model testing via BacktestSuite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Traffic-light colours for plotting
# ---------------------------------------------------------------------------
_TL_COLOURS = {
    "green":  "#27ae60",
    "yellow": "#f39c12",
    "red":    "#e74c3c",
}


@dataclass(frozen=True)
class BacktestMetrics:
    """Immutable results of a full backtest exercise."""

    model_name: str = ""
    n_obs: int = 0
    n_violations: int = 0
    violation_rate: float = 0.0
    expected_rate: float = 0.0
    expected_violations: float = 0.0
    # Kupiec
    kupiec_lr: float = 0.0
    kupiec_pvalue: float = 0.0
    kupiec_pass: bool = False
    # Christoffersen independence
    ind_lr: float = 0.0
    ind_pvalue: float = 0.0
    ind_transition: Optional[Dict[str, int]] = None
    ind_pass: bool = False
    # Conditional coverage
    cc_lr: float = 0.0
    cc_pvalue: float = 0.0
    cc_pass: bool = False
    # Basel
    basel_zone: str = ""
    tl_colour_hex: str = ""

    def summary_dict(self) -> Dict[str, object]:
        return {
            "Model":               self.model_name or "VaR Model",
            "Obs":                 self.n_obs,
            "Violations":          self.n_violations,
            "Exp. violations":     f"{self.expected_violations:.1f}",
            "Violation rate":      f"{self.violation_rate:.2%}",
            "Kupiec LR":           f"{self.kupiec_lr:.3f}",
            "Kupiec p-val":        f"{self.kupiec_pvalue:.4f}",
            "Kupiec":              "PASS ✓" if self.kupiec_pass else "FAIL ✗",
            "Ind. LR":             f"{self.ind_lr:.3f}",
            "Ind. p-val":          f"{self.ind_pvalue:.4f}",
            "Ind.":                "PASS ✓" if self.ind_pass else "FAIL ✗",
            "CC LR":               f"{self.cc_lr:.3f}",
            "CC p-val":            f"{self.cc_pvalue:.4f}",
            "CC":                  "PASS ✓" if self.cc_pass else "FAIL ✗",
            "Basel":               self.basel_zone,
        }


def _lr_pof_stat(N: int, T: int, p0: float) -> float:
    if T == 0 or N < 0 or N > T:
        return 0.0
    if N == 0:
        return -2.0 * T * np.log(1.0 - p0) if p0 < 1.0 else 0.0
    if N == T:
        return -2.0 * T * np.log(p0) if p0 > 0.0 else 0.0
    p_hat = N / T
    lr = -2.0 * (
        (T - N) * np.log(1.0 - p0) + N * np.log(p0)
        - (T - N) * np.log(1.0 - p_hat) - N * np.log(p_hat)
    )
    return max(lr, 0.0)


class VaRBacktest:
    """
    Backtest a VaR series against realised returns.

    VaR estimates should be *positive* numbers representing the loss
    magnitude.  Both ``returns`` and ``var_estimates`` are cleaned
    (NaN removal, length alignment) automatically.
    """

    def __init__(
        self,
        returns: np.ndarray | pd.Series,
        var_estimates: np.ndarray | pd.Series,
        *,
        confidence_level: float = 0.95,
        model_name: str = "",
    ) -> None:
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError(f"Invalid confidence_level: {confidence_level}")

        r, v = self._clean_inputs(returns, var_estimates)
        if len(r) == 0:
            raise ValueError("No valid overlapping observations after cleaning")

        self.r: np.ndarray = r
        self.v: np.ndarray = v
        self.cl: float = float(confidence_level)
        self.alpha: float = 1.0 - self.cl
        self.model_name: str = model_name

        self.violations: np.ndarray = self.r < -self.v
        self.N: int = int(self.violations.sum())
        self.T: int = len(self.r)

    # ------------------------------------------------------------------
    # Input cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_inputs(
        returns: np.ndarray | pd.Series,
        var_estimates: np.ndarray | pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align inputs: use date-based alignment for Series,
        trailing overlap for ndarrays. Strip NaNs.
        """
        # Date-based alignment for pandas Series
        if isinstance(returns, pd.Series) and isinstance(var_estimates, pd.Series):
            r, v = returns.align(var_estimates, join="inner")
            r = r.values
            v = v.values
        else:
            r = np.asarray(returns, dtype=float).flatten()
            v = np.asarray(var_estimates, dtype=float).flatten()
            n = min(len(r), len(v))
            if n == 0:
                return np.array([]), np.array([])
            r, v = r[-n:], v[-n:]

        # Remove NaNs
        valid = ~(np.isnan(r) | np.isnan(v))
        return r[valid], v[valid]

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def kupiec(self) -> Tuple[float, float, bool]:
        lr = _lr_pof_stat(self.N, self.T, self.alpha)
        pval = 1.0 - stats.chi2.cdf(lr, df=1)
        return lr, pval, pval > 0.05

    def christoffersen(self) -> Tuple[float, float, Optional[Dict[str, int]], bool]:
        V = self.violations.astype(int)
        T = len(V)
        N = int(V.sum())

        if N == 0 or N == T:
            return 0.0, np.nan, None, True

        n00 = int(np.sum((V[:-1] == 0) & (V[1:] == 0)))
        n01 = int(np.sum((V[:-1] == 0) & (V[1:] == 1)))
        n10 = int(np.sum((V[:-1] == 1) & (V[1:] == 0)))
        n11 = int(np.sum((V[:-1] == 1) & (V[1:] == 1)))

        eps = 1e-10
        p01 = np.clip(n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0, eps, 1 - eps)
        p11 = np.clip(n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0, eps, 1 - eps)
        p2  = np.clip((n01 + n11) / (n00 + n01 + n10 + n11), eps, 1 - eps)

        ll_ind = (
            n00 * np.log(1 - p01) + n01 * np.log(p01) +
            n10 * np.log(1 - p11) + n11 * np.log(p11)
        )
        ll_unc = (n00 + n10) * np.log(1 - p2) + (n01 + n11) * np.log(p2)
        lr = max(-2.0 * (ll_unc - ll_ind), 0.0)
        pval = float(1.0 - stats.chi2.cdf(lr, df=1))
        trans = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
        return lr, pval, trans, pval > 0.05

    def conditional_coverage(self) -> Tuple[float, float, bool]:
        lr_pof, _, _ = self.kupiec()
        lr_ind, _, _, _ = self.christoffersen()
        lr_cc = lr_pof + lr_ind
        pval = 1.0 - stats.chi2.cdf(lr_cc, df=2)
        return lr_cc, pval, pval > 0.05

    def basel_zone_cdf(self) -> str:
        """Generalised Basel zone via binomial CDF (valid for any T)."""
        p_binom = stats.binom.cdf(self.N, self.T, self.alpha)
        if p_binom >= 0.95:
            return "green"
        if p_binom >= 0.05:
            return "yellow"
        return "red"

    def basel_zone_250(self) -> str:
        """Strict Basel III zone scaled to per-250-obs basis."""
        viol_250 = round(self.N * (250.0 / self.T)) if self.T > 0 else 0
        if viol_250 <= 4:
            return "green"
        if viol_250 <= 9:
            return "yellow"
        return "red"

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def run(self) -> BacktestMetrics:
        lr_k, pv_k, pass_k = self.kupiec()
        lr_i, pv_i, trans, pass_i = self.christoffersen()
        lr_c, pv_c, pass_c = self.conditional_coverage()
        zone = self.basel_zone_cdf()

        logger.info(
            "Backtest  T=%d  N=%d (%.2f%%)  zone=%s  "
            "Kupiec=%.3f  Indep=%.3f  CC=%.3f",
            self.T, self.N, 100 * self.N / self.T if self.T else 0,
            zone, pv_k, pv_i if not np.isnan(pv_i) else -1.0, pv_c,
        )

        return BacktestMetrics(
            model_name=self.model_name,
            n_obs=self.T,
            n_violations=self.N,
            violation_rate=self.N / self.T if self.T else 0.0,
            expected_rate=self.alpha,
            expected_violations=self.T * self.alpha,
            kupiec_lr=lr_k,
            kupiec_pvalue=pv_k,
            kupiec_pass=pass_k,
            ind_lr=lr_i,
            ind_pvalue=pv_i if not np.isnan(pv_i) else -1.0,
            ind_transition=trans,
            ind_pass=pass_i,
            cc_lr=lr_c,
            cc_pvalue=pv_c,
            cc_pass=pass_c,
            basel_zone=zone,
            tl_colour_hex=_TL_COLOURS.get(zone, "#999999"),
        )

    def summary(self) -> str:
        m = self.run()
        lines = [
            "=" * 60,
            "  VaR BACKTEST RESULTS",
            "=" * 60,
            f"  Observations      {m.n_obs:>8,}",
            f"  Violations        {m.n_violations:>8}  ({m.violation_rate:.2%})",
            f"  Expected          {m.expected_violations:>8.0f}  ({m.expected_rate:.2%})",
            "",
            f"  Kupiec POF        LR={m.kupiec_lr:.3f}  p={m.kupiec_pvalue:.4f}  [{'PASS' if m.kupiec_pass else 'FAIL'}]",
            f"  Christoffersen    LR={m.ind_lr:.3f}  p={m.ind_pvalue:.4f}  [{'PASS' if m.ind_pass else 'FAIL'}]",
            f"  Conditional Cov   LR={m.cc_lr:.3f}  p={m.cc_pvalue:.4f}  [{'PASS' if m.cc_pass else 'FAIL'}]",
            "",
            f"  Basel (CDF)       {m.basel_zone.upper()}",
            f"  Basel (250d)      {self.basel_zone_250().upper()}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def get_violation_series(self) -> pd.Series:
        """Return violations as a boolean Series if original input was a Series."""
        return pd.Series(self.violations, dtype=bool)


# ---------------------------------------------------------------------------
# Multi-model backtesting suite
# ---------------------------------------------------------------------------

class BacktestSuite:
    """
    Run Kupiec + Christoffersen tests across multiple VaR models at once.

    Usage
    -----
    suite = BacktestSuite(confidence_level=0.95)
    results = suite.run(actual_returns, {
        "Historical VaR":  hist_var_series,
        "Parametric VaR":  param_var_series,
        "Monte Carlo VaR": mc_var_series,
    })
    df = suite.summary_table(results)
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level

    def run(
        self,
        actual_returns: np.ndarray | pd.Series,
        var_dict: Dict[str, np.ndarray | pd.Series],
    ) -> Dict[str, BacktestMetrics]:
        results: Dict[str, BacktestMetrics] = {}
        for name, var_series in var_dict.items():
            bt = VaRBacktest(
                actual_returns, var_series,
                confidence_level=self.confidence_level,
                model_name=name,
            )
            results[name] = bt.run()
        return results

    def summary_table(self, results: Dict[str, BacktestMetrics]) -> pd.DataFrame:
        rows = [r.summary_dict() for r in results.values()]
        return pd.DataFrame(rows)

    def print_summary(self, results: Dict[str, BacktestMetrics]) -> None:
        df = self.summary_table(results)
        print(df.to_string(index=False))