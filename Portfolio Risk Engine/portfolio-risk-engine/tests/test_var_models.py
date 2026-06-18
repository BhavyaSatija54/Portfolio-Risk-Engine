"""
Tests for VaR Models — full coverage with numerical checks.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_engine import (
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    VaRResult,
)

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_returns() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    return pd.Series(rng.normal(0.0005, 0.02, len(dates)), index=dates)


@pytest.fixture
def zero_returns() -> pd.Series:
    """Pathological: zero variance."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    return pd.Series(np.zeros(len(dates)), index=dates)


@pytest.fixture
def comp_ret() -> pd.DataFrame:
    """Two correlated assets for Cholesky testing."""
    n = 1000
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, 2))
    r = z @ L.T * 0.02 + 0.0005
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(r, columns=["X1", "X2"], index=dates)


# ---------------------------------------------------------------------------
# HistoricalVaR
# ---------------------------------------------------------------------------

class TestHistoricalVaR:
    def test_basic(self, normal_returns):
        res = HistoricalVaR(normal_returns, confidence_level=0.95).calculate()
        assert isinstance(res, VaRResult)
        assert res.var_pct > 0
        assert res.es_pct > 0
        assert res.es_pct >= res.var_pct * 0.99

    def test_monotonic_with_confidence(self, normal_returns):
        v90 = HistoricalVaR(normal_returns, confidence_level=0.90).calculate()
        v95 = HistoricalVaR(normal_returns, confidence_level=0.95).calculate()
        v99 = HistoricalVaR(normal_returns, confidence_level=0.99).calculate()
        assert v90.var_pct < v95.var_pct < v99.var_pct

    def test_monotonic_with_hp(self, normal_returns):
        v1 = HistoricalVaR(normal_returns, holding_period=1, confidence_level=0.95).calculate()
        v5 = HistoricalVaR(normal_returns, holding_period=5, confidence_level=0.95).calculate()
        v10 = HistoricalVaR(normal_returns, holding_period=10, confidence_level=0.95).calculate()
        assert v1.var_pct < v5.var_pct < v10.var_pct

    def test_notional_scaling(self, normal_returns):
        v1m = HistoricalVaR(normal_returns, confidence_level=0.95, notional=1e6).calculate()
        v2m = HistoricalVaR(normal_returns, confidence_level=0.95, notional=2e6).calculate()
        assert v2m.var_abs == pytest.approx(2 * v1m.var_abs)

    def test_bootstrap_ci(self, normal_returns):
        model = HistoricalVaR(normal_returns, confidence_level=0.95,
                             bootstrap=True, n_bootstrap=5000, random_state=7)
        model.calculate()
        ci = model._bootstrap_ci(model.r)
        assert ci[0] < ci[1]

    def test_model_name(self, normal_returns):
        res = HistoricalVaR(normal_returns).calculate()
        assert res.model_name == "Historical"


# ---------------------------------------------------------------------------
# ParametricVaR
# ---------------------------------------------------------------------------

class TestParametricVaR:
    def test_theoretical_var(self, normal_returns):
        """Parametric VaR should match closed-form for normal data."""
        z95 = stats.norm.ppf(0.95)
        expected = z95 * normal_returns.std()
        res = ParametricVaR(normal_returns, confidence_level=0.95).calculate()
        assert res.var_pct == pytest.approx(expected, rel=1e-6)

    def test_es_exceeds_var(self, normal_returns):
        res = ParametricVaR(normal_returns, confidence_level=0.95).calculate()
        assert res.es_pct > res.var_pct

    def test_sqrt_time_scaling(self, normal_returns):
        v1 = ParametricVaR(normal_returns, holding_period=1).calculate()
        v10 = ParametricVaR(normal_returns, holding_period=10).calculate()
        expected_ratio = np.sqrt(10)
        assert v10.var_pct == pytest.approx(v1.var_pct * expected_ratio, rel=0.01)

    def test_ewma(self, normal_returns):
        res = ParametricVaR(normal_returns, volatility_model="ewma").calculate()
        assert res.var_pct > 0

    def test_explicit_vol(self, normal_returns):
        res = ParametricVaR(normal_returns, daily_vol=0.015).calculate()
        z = stats.norm.ppf(0.95)
        assert res.var_pct == pytest.approx(z * 0.015, rel=1e-6)

    def test_zero_vol_raises_or_warns(self, zero_returns):
        """Zero variance should produce a tiny positive vol, not crash."""
        res = ParametricVaR(zero_returns, confidence_level=0.95).calculate()
        assert res.var_pct < 1e-4  # Essentially zero VaR

    def test_invalid_vm(self, normal_returns):
        with pytest.raises(ValueError):
            ParametricVaR(normal_returns, volatility_model="garch").calculate()

    def test_invalid_vm_name(self, normal_returns):
        with pytest.raises(ValueError):
            ParametricVaR(normal_returns, volatility_model="invalid").calculate()

    def test_99_vs_95(self, normal_returns):
        v95 = ParametricVaR(normal_returns, confidence_level=0.95).calculate()
        v99 = ParametricVaR(normal_returns, confidence_level=0.99).calculate()
        assert v99.var_pct > v95.var_pct
        ratio = v99.var_pct / v95.var_pct
        expected = stats.norm.ppf(0.99) / stats.norm.ppf(0.95)
        assert ratio == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# MonteCarloVaR
# ---------------------------------------------------------------------------

class TestMonteCarloVaR:
    def test_basic(self, normal_returns):
        res = MonteCarloVaR(normal_returns, n_simulations=20000, random_state=5).calculate()
        assert res.var_pct > 0
        assert res.es_pct >= res.var_pct * 0.95

    def test_reproducibility(self, normal_returns):
        v1 = MonteCarloVaR(normal_returns, n_simulations=20000, random_state=99).calculate()
        v2 = MonteCarloVaR(normal_returns, n_simulations=20000, random_state=99).calculate()
        assert v1.var_pct == pytest.approx(v2.var_pct, rel=1e-10)

    def test_larger_sim_more_stable(self, normal_returns):
        vals_10k = [MonteCarloVaR(normal_returns, n_simulations=10000, random_state=i).calculate().var_pct
                    for i in range(5)]
        vals_100k = [MonteCarloVaR(normal_returns, n_simulations=100000, random_state=i).calculate().var_pct
                     for i in range(5)]
        assert np.std(vals_100k) <= np.std(vals_10k) * 0.5

    def test_cholesky(self, comp_ret):
        w = np.array([0.5, 0.5])
        port_r = comp_ret @ w
        res = MonteCarloVaR(
            port_r, n_simulations=50000, random_state=3,
            use_cholesky=True, component_returns=comp_ret, component_weights=w,
        ).calculate()
        assert res.var_pct > 0

    def test_student_t(self, normal_returns):
        res = MonteCarloVaR(normal_returns, n_simulations=20000,
                           random_state=3, distribution="t").calculate()
        assert res.var_pct > 0

    def test_simulated_mean(self, normal_returns):
        model = MonteCarloVaR(normal_returns, n_simulations=50000, random_state=7)
        model.calculate()
        sim = model.get_simulated()
        assert np.isclose(sim.mean(), normal_returns.mean(), atol=0.002)

    def test_simulated_std(self, normal_returns):
        model = MonteCarloVaR(normal_returns, n_simulations=50000, random_state=7)
        model.calculate()
        sim = model.get_simulated()
        assert np.isclose(sim.std(), normal_returns.std(), rtol=0.05)

    def test_get_simulated_before_calculate(self, normal_returns):
        model = MonteCarloVaR(normal_returns, n_simulations=1000, random_state=1)
        with pytest.raises(RuntimeError):
            model.get_simulated()

    def test_psd_fallback(self):
        """Cholesky on near-singular matrix should still work."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Two nearly identical columns → near-singular
        a = rng.normal(0, 0.02, n)
        df = pd.DataFrame({"X1": a, "X2": a + rng.normal(0, 1e-6, n)}, index=dates)
        w = np.array([0.5, 0.5])
        port = df @ w
        res = MonteCarloVaR(port, n_simulations=10000, random_state=1,
                           use_cholesky=True, component_returns=df, component_weights=w).calculate()
        assert res.var_pct > 0


# ---------------------------------------------------------------------------
# Edge cases & invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_var_result_immutable(self, normal_returns):
        res = HistoricalVaR(normal_returns).calculate()
        with pytest.raises(AttributeError):
            res.var_pct = 0.5  # frozen dataclass

    def test_confidence_bounds(self, normal_returns):
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, confidence_level=0.0)
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, confidence_level=1.0)
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, confidence_level=-0.1)
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, confidence_level=1.5)

    def test_holding_period_positive(self, normal_returns):
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, holding_period=0)
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, holding_period=-5)

    def test_notional_positive(self, normal_returns):
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, notional=0)
        with pytest.raises(ValueError):
            HistoricalVaR(normal_returns, notional=-1)

    def test_all_models_agree_roughly(self, normal_returns):
        """For normal i.i.d. data all models should be in the same ballpark."""
        h = HistoricalVaR(normal_returns).calculate()
        p = ParametricVaR(normal_returns).calculate()
        m = MonteCarloVaR(normal_returns, n_simulations=100000, random_state=42).calculate()

        # Parametric is the theoretical baseline for normal data
        assert h.var_pct == pytest.approx(p.var_pct, rel=0.15)
        assert m.var_pct == pytest.approx(p.var_pct, rel=0.10)