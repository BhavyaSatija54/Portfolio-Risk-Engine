"""
Tests for GARCH11 (from-scratch MLE implementation).
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.garch_mle import GARCH11, GARCH11Params, GARCH11Result

rng = np.random.default_rng(42)


@pytest.fixture
def garch_dgp() -> pd.Series:
    """Known GARCH(1,1) DGP: omega=0.01, alpha=0.10, beta=0.85."""
    n = 1500
    omega, alpha_true, beta_true = 0.01, 0.10, 0.85
    r = np.zeros(n)
    vol = np.zeros(n)
    vol[0] = 0.02
    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha_true * r[t - 1] ** 2 + beta_true * vol[t - 1] ** 2)
        r[t] = vol[t] * rng.standard_normal()
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(r, index=dates)


@pytest.fixture
def normal_ret() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.02, len(dates)), index=dates)


class TestGARCH11Fit:
    def test_basic(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        assert isinstance(m, GARCH11Result)
        assert m.converged
        assert isinstance(m.params, GARCH11Params)

    def test_parameter_ranges(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        p = m.params
        assert p.omega > 0
        assert p.alpha >= 0
        assert p.beta >= 0
        assert p.persistence < 1.0

    def test_persistence_estimate(self, garch_dgp):
        """True persistence = 0.10 + 0.85 = 0.95."""
        m = GARCH11().fit(garch_dgp)
        p = m.params
        assert 0.85 < p.persistence < 1.0
        assert 0.90 < p.persistence < 0.99

    def test_forecast_positive(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        assert m.forecast_sigma > 0
        assert m.forecast_var > 0

    def test_multi_step_forecast(self, garch_dgp):
        g = GARCH11()
        m = g.fit(garch_dgp)
        fc = g.forecast_variance(m, h=20)
        assert len(fc) == 20
        assert (fc > 0).all()

    def test_conditional_variance(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        assert (m.sigma2 > 0).all()
        assert len(m.sigma2) == len(garch_dgp)

    def test_standardised_residuals(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        z = m.residuals
        assert np.isclose(z.mean(), 0, atol=0.1)
        assert np.isclose(z.std(), 1, atol=0.2)

    def test_information_criteria(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        assert np.isfinite(m.log_likelihood)
        assert np.isfinite(m.aic)
        assert np.isfinite(m.bic)
        assert m.bic > m.aic

    def test_half_life(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        hl = m.params.half_life
        assert hl > 0 and np.isfinite(hl)

    def test_normal_data_low_persistence(self, normal_ret):
        m = GARCH11().fit(normal_ret)
        assert m.params.alpha < 0.15
        assert m.params.persistence < 0.95

    def test_residual_diagnostics(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        diag = GARCH11.residual_diagnostics(m)
        assert "lb_z_stat" in diag
        assert "lb_z_pval" in diag
        assert "lb_z2_stat" in diag
        assert "jb_stat" in diag
        assert "skewness" in diag
        assert "kurtosis" in diag

    def test_garch_filtered_var(self, garch_dgp):
        g = GARCH11()
        m = g.fit(garch_dgp)
        hist_sig = float(garch_dgp.std())
        base_var = 0.02
        fv = g.garch_filtered_var(m, base_var, hist_sig)
        assert fv["var_pct"] > 0
        assert fv["var_abs"] > 0
        assert fv["vol_scale"] > 0
        assert "garch_sigma" in fv

    def test_too_short_non_convergence(self):
        """Very short series may not converge — check graceful handling."""
        s = pd.Series(rng.normal(0, 1, 30))
        m = GARCH11().fit(s)
        # Should still produce a result even if not converged
        assert isinstance(m, GARCH11Result)
        assert m.forecast_sigma > 0

    def test_repr_params(self, garch_dgp):
        p = GARCH11().fit(garch_dgp).params
        assert "GARCH11Params" in repr(p)
        assert f"pers={p.persistence:.4f}" in repr(p)

    def test_unconditional_vol(self, garch_dgp):
        m = GARCH11().fit(garch_dgp)
        p = m.params
        assert p.unconditional_vol > 0
        assert np.isclose(p.unconditional_vol, np.sqrt(p.unconditional_variance))

    def test_forecast_convergence(self, garch_dgp):
        """Multi-step forecast should converge to unconditional variance."""
        g = GARCH11()
        m = g.fit(garch_dgp)
        fc = g.forecast_variance(m, h=50)
        unc = m.params.unconditional_variance
        assert abs(fc[-1] - unc) < abs(fc[0] - unc) * 2