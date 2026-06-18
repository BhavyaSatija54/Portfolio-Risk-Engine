"""
Tests for GARCH(1,1) — production coverage.
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine import GARCHModel, GARCHParams

rng = np.random.default_rng(42)


@pytest.fixture
def garch_dgp() -> pd.Series:
    """Simulate from a known GARCH(1,1) DGP."""
    n = 1500
    omega, alpha, beta = 0.01, 0.10, 0.85
    r = np.zeros(n)
    vol = np.zeros(n)
    vol[0] = 0.02
    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha * r[t - 1] ** 2 + beta * vol[t - 1] ** 2)
        r[t] = vol[t] * rng.standard_normal()
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(r, index=dates)


@pytest.fixture
def normal_ret() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.02, len(dates)), index=dates)


class TestFit:
    def test_basic(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        assert m._fit_result is not None
        assert m._fit_result.convergence_flag == 0

    def test_params_type(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        p = m.params()
        assert isinstance(p, GARCHParams)

    def test_persistence(self, garch_dgp):
        """True persistence = 0.10 + 0.85 = 0.95."""
        m = GARCHModel(garch_dgp).fit()
        p = m.params()
        assert 0.85 < p.persistence < 1.0
        assert 0.90 < p.persistence < 0.99

    def test_stationarity(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        assert m.params().persistence < 1.0

    def test_omega_positive(self, garch_dgp):
        assert GARCHModel(garch_dgp).fit().params().omega > 0

    def test_half_life_positive(self, garch_dgp):
        hl = GARCHModel(garch_dgp).fit().params().half_life
        assert hl > 0 and np.isfinite(hl)


class TestForecast:
    def test_forecast_shape(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        fc = m.forecast(horizon=5)
        assert len(fc) == 5
        assert list(fc.columns) == ["variance", "volatility"]

    def test_forecast_vol_positive(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        assert m.forecast_vol() > 0

    def test_conditional_vol_shape(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        cv = m.conditional_vol()
        assert len(cv) == len(garch_dgp)
        assert (cv > 0).all()

    def test_forecast_convergence(self, garch_dgp):
        """Multi-step forecast should converge toward unconditional vol."""
        m = GARCHModel(garch_dgp).fit()
        fc = m.forecast(horizon=20)
        unc = m.params().unconditional_volatility
        # Last forecast should be closer to unconditional
        assert abs(fc["volatility"].iloc[-1] - unc) < abs(fc["volatility"].iloc[0] - unc) * 2


class TestDiagnostics:
    def test_loglikelihood(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        assert m.loglikelihood() < 0  # log-lik of normal is negative
        assert np.isfinite(m.loglikelihood())

    def test_aic_bic_finite(self, garch_dgp):
        m = GARCHModel(garch_dgp).fit()
        assert np.isfinite(m.aic())
        assert np.isfinite(m.bic())
        assert m.bic() > m.aic()  # penalty is larger


class TestRolling:
    def test_rolling_shape(self, garch_dgp):
        m = GARCHModel(garch_dgp)
        roll = m.rolling_forecast(window=500, step=50)
        assert len(roll) > 0
        assert list(roll.columns) == ["forecast_vol"]

    def test_rolling_values_positive(self, garch_dgp):
        m = GARCHModel(garch_dgp)
        roll = m.rolling_forecast(window=500, step=50)
        valid = roll["forecast_vol"].dropna()
        assert (valid > 0).all()


class TestNormalData:
    """GARCH on normal i.i.d. data → alpha ≈ 0."""

    def test_low_persistence(self, normal_ret):
        m = GARCHModel(normal_ret).fit()
        assert m.params().alpha < 0.15
        assert m.params().persistence < 0.95


class TestEdgeCases:
    def test_too_short_raises(self):
        s = pd.Series(rng.normal(0, 1, 30))
        with pytest.raises(ValueError):
            GARCHModel(s)

    def test_summary_before_fit(self, normal_ret):
        assert "not yet fitted" in GARCHModel(normal_ret).summary()

    def test_repr(self, normal_ret):
        assert "unfitted" in repr(GARCHModel(normal_ret))
        assert "fitted" in repr(GARCHModel(normal_ret).fit())

    def test_student_t_dist(self, garch_dgp):
        m = GARCHModel(garch_dgp, distribution="t").fit()
        assert m._fit_result.convergence_flag == 0
        assert m.params().persistence < 1.0