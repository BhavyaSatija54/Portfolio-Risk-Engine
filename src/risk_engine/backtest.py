"""
Backtesting Module — Production Grade

Statistical validation of VaR models via:
  * Kupiec Proportion-of-Failures (LR_POF)
  * Christoffersen independence      (LR_IND)
  * Conditional coverage joint test  (LR_CC)
  * Basel regulatory traffic light
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestMetrics:
    """Immutable results of a full backtest exercise."""

    n_obs: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    # Kupiec
    kupiec_lr: float
    kupiec_pvalue: float
    kupiec_pass: bool
    # Christoffersen
    ind_lr: float
    ind_pvalue: float
    ind_transition: Optional[Dict[str, int]]
    ind_pass: bool
    # Conditional coverage
    cc_lr: float
    cc_pvalue: float
    cc_pass: bool
    # Basel
    basel_zone: str  # green | yellow | red


def _lr_pof_stat(N: int, T: int, p0: float) -> float:
    """
    Kupiec LR_POF statistic with guards for boundary cases.
    """
    if T == 0 or N < 0 or N > T:
        return 0.0
    if N == 0:
        return -2.0 * T * np.log(1.0 - p0) if p0 < 1.0 else 0.0
    if N == T:
        return -2.0 * T * np.log(p0) if p0 > 0.0 else 0.0
    p_hat = N / T
    # LR = -2 * [log L(p0) - log L(p_hat)]
    lr = -2.0 * (
        (T - N) * np.log(1.0 - p0)
        + N * np.log(p0)
        - (T - N) * np.log(1.0 - p_hat)
        - N * np.log(p_hat)
    )
    return max(lr, 0.0)


class VaRBacktest:
    """
    Backtest a VaR series against realized returns.

    Both ``returns`` and ``var_estimates`` must be pd.Series with
    overlapping DateTimeIndexes.  VaR estimates should be *positive*
    numbers representing the loss magnitude.
    """

    def __init__(
        self,
        returns: pd.Series,
        var_estimates: pd.Series,
        *,
        confidence_level: float = 0.95,
    ) -> None:
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError(f"Invalid confidence_level: {confidence_level}")

        # Align to intersection of dates
        r, v = returns.align(var_estimates, join="inner")
        if len(r) == 0:
            raise ValueError("No overlapping dates between returns and var_estimates")

        self.r: pd.Series = r
        self.v: pd.Series = v
        self.cl: float = float(confidence_level)
        self.alpha: float = 1.0 - self.cl

        self.violations: pd.Series = self.r < -self.v
        self.N: int = int(self.violations.sum())
        self.T: int = len(self.r)

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def kupiec(self) -> Tuple[float, float, bool]:
        """
        Kupiec POF test.

        Returns
        -------
        (LR_stat, pvalue, pass_null)
        """
        lr = _lr_pof_stat(self.N, self.T, self.alpha)
        pval = 1.0 - stats.chi2.cdf(lr, df=1)
        return lr, pval, pval > 0.05

    def christoffersen(self) -> Tuple[float, float, Optional[Dict[str, int]], bool]:
        """
        Christoffersen interval-forecast independence test.

        Returns
        -------
        (LR_stat, pvalue, transition_counts, pass_null)
        """
        v = self.violations.astype(int).values
        T = len(v)
        N = int(v.sum())

        if N == 0 or N == T:
            return 0.0, np.nan, None, True  # inconclusive

        n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
        n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
        n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
        n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

        p_hat = N / T
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

        # Avoid log(0)
        terms = []
        if p_hat > 0 and p_hat < 1:
            terms.append(((T - N) * np.log(1 - p_hat) + N * np.log(p_hat)))
        if p01 > 0 and p01 < 1 and (n00 + n01) > 0:
            terms.append(-(n00 * np.log(1 - p01) + n01 * np.log(p01)))
        if p11 > 0 and p11 < 1 and (n10 + n11) > 0:
            terms.append(-(n10 * np.log(1 - p11) + n11 * np.log(p11)))

        lr = max(2.0 * sum(terms), 0.0) if terms else 0.0
        pval = 1.0 - stats.chi2.cdf(lr, df=1)
        trans = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
        return lr, pval, trans, pval > 0.05

    def conditional_coverage(self) -> Tuple[float, float, bool]:
        """Joint conditional coverage test (LR_POF + LR_IND)."""
        lr_pof, _, _ = self.kupiec()
        lr_ind, _, _, _ = self.christoffersen()
        lr_cc = lr_pof + lr_ind
        pval = 1.0 - stats.chi2.cdf(lr_cc, df=2)
        return lr_cc, pval, pval > 0.05

    def basel_zone(self) -> str:
        """
        Basel traffic light classification.

        For 250 observations the Basel committee uses exact binomial
        cutoffs; we generalise via the cumulative binomial.
        """
        p_binom = stats.binom.cdf(self.N, self.T, self.alpha)
        if p_binom >= 0.95:
            return "green"
        if p_binom >= 0.05:
            return "yellow"
        return "red"

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def run(self) -> BacktestMetrics:
        """Run the full battery and return an immutable result."""
        lr_k, pv_k, pass_k = self.kupiec()
        lr_i, pv_i, trans, pass_i = self.christoffersen()
        lr_c, pv_c, pass_c = self.conditional_coverage()
        zone = self.basel_zone()

        logger.info(
            "Backtest  T=%d  N=%d (%.2f%%)  zone=%s  "
            "Kupiec=%.3f  Indep=%.3f  CC=%.3f",
            self.T,
            self.N,
            100 * self.N / self.T if self.T else 0,
            zone,
            pv_k,
            pv_i if not np.isnan(pv_i) else -1.0,
            pv_c,
        )

        return BacktestMetrics(
            n_obs=self.T,
            n_violations=self.N,
            violation_rate=self.N / self.T if self.T else 0.0,
            expected_rate=self.alpha,
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
        )

    def summary(self) -> str:
        """Human-readable summary table."""
        m = self.run()
        lines = [
            "=" * 60,
            "  VaR BACKTEST RESULTS",
            "=" * 60,
            "",
            f"  Observations      {m.n_obs:>8,}",
            f"  Violations        {m.n_violations:>8}  ({m.violation_rate:.2%})",
            f"  Expected          {m.expected_rate * m.n_obs:>8.0f}  ({m.expected_rate:.2%})",
            "",
            "  Kupiec POF        "
            f"LR={m.kupiec_lr:.3f}  p={m.kupiec_pvalue:.4f}  [{'PASS' if m.kupiec_pass else 'FAIL'}]",
            "  Christoffersen    "
            f"LR={m.ind_lr:.3f}  p={m.ind_pvalue:.4f}  [{'PASS' if m.ind_pass else 'FAIL'}]",
            "  Conditional Cov   "
            f"LR={m.cc_lr:.3f}  p={m.cc_pvalue:.4f}  [{'PASS' if m.cc_pass else 'FAIL'}]",
            "",
            f"  Basel Zone        {m.basel_zone.upper():>8}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def get_violation_series(self) -> pd.Series:
        return self.violations