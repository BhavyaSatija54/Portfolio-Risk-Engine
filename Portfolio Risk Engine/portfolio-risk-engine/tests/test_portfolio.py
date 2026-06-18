"""
Tests for Portfolio — institutional-grade coverage.
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine import Portfolio

rng = np.random.default_rng(42)


@pytest.fixture
def returns_3() -> pd.DataFrame:
    """Three correlated assets, 500 observations."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    cov = np.array([[0.0004, 0.0002, 0.0001],
                    [0.0002, 0.0003, 0.0001],
                    [0.0001, 0.0001, 0.0005]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((500, 3))
    r = z @ L.T + 0.0005
    return pd.DataFrame(r, columns=["A", "B", "C"], index=dates)


@pytest.fixture
def portfolio_eq(returns_3):
    return Portfolio(returns=returns_3, name="Test")


class TestInitialization:
    def test_equal_weights_default(self, returns_3):
        p = Portfolio(returns_3)
        np.testing.assert_allclose(p.get_weights().values, [1 / 3, 1 / 3, 1 / 3])

    def test_custom_weights(self, returns_3):
        p = Portfolio(returns_3, weights={"A": 0.5, "B": 0.3, "C": 0.2})
        np.testing.assert_allclose(p.get_weights().values, [0.5, 0.3, 0.2])

    def test_weights_normalization(self, returns_3):
        p = Portfolio(returns_3, weights=np.array([50, 30, 20]))
        np.testing.assert_allclose(p.get_weights().values, [0.5, 0.3, 0.2])

    def test_empty_raises(self, returns_3):
        with pytest.raises(ValueError):
            Portfolio(returns_3, weights={})

    def test_zero_sum_raises(self, returns_3):
        with pytest.raises(ValueError):
            Portfolio(returns_3, weights=np.array([0, 0, 0]))

    def test_mismatched_length_raises(self, returns_3):
        with pytest.raises(ValueError):
            Portfolio(returns_3, weights=np.array([0.5, 0.5]))

    def test_portfolio_returns_shape(self, portfolio_eq):
        assert len(portfolio_eq.get_portfolio_returns()) == 500

    def test_portfolio_returns_manual(self, returns_3):
        p = Portfolio(returns_3, weights={"A": 0.5, "B": 0.3, "C": 0.2})
        expected = pd.Series(
            returns_3.values @ np.array([0.5, 0.3, 0.2]),
            index=returns_3.index,
            name="Portfolio",
        )
        pd.testing.assert_series_equal(p.get_portfolio_returns(), expected)


class TestStatistics:
    def test_statistics_keys(self, portfolio_eq):
        s = portfolio_eq.get_statistics()
        expected_keys = {
            "n_observations", "total_return", "annualized_return",
            "annualized_volatility", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "max_drawdown_date", "calmar_ratio",
            "var_95", "cvar_95", "skewness", "kurtosis",
            "tail_ratio_5", "tail_ratio_1",
        }
        assert set(s.keys()) == expected_keys

    def test_vol_positive(self, portfolio_eq):
        assert portfolio_eq.get_statistics()["annualized_volatility"] > 0

    def test_sharpe_finite(self, portfolio_eq):
        assert np.isfinite(portfolio_eq.get_statistics()["sharpe_ratio"])

    def test_max_drawdown_negative(self, portfolio_eq):
        assert portfolio_eq.get_statistics()["max_drawdown"] <= 0

    def test_var_less_than_cvar(self, portfolio_eq):
        s = portfolio_eq.get_statistics()
        assert s["cvar_95"] <= s["var_95"]  # both negative; cvar more extreme

    def test_caching(self, portfolio_eq):
        s1 = portfolio_eq.get_statistics()
        s2 = portfolio_eq.get_statistics()
        assert s1["sharpe_ratio"] == s2["sharpe_ratio"]


class TestRiskContribution:
    def test_risk_contrib_columns(self, portfolio_eq):
        rc = portfolio_eq.get_risk_contribution()
        assert set(rc.columns) == {"weight", "marginal_risk", "risk_contrib", "pct_contrib"}

    def test_pct_contrib_sums_to_one(self, portfolio_eq):
        rc = portfolio_eq.get_risk_contribution()
        np.testing.assert_allclose(rc["pct_contrib"].sum(), 1.0, atol=1e-6)

    def test_risk_contrib_sums_to_vol(self, portfolio_eq):
        rc = portfolio_eq.get_risk_contribution()
        vol = np.sqrt(portfolio_eq.get_portfolio_variance())
        np.testing.assert_allclose(rc["risk_contrib"].sum(), vol, atol=1e-8)

    def test_single_asset(self, returns_3):
        """100% in A → risk_contrib ≈ portfolio vol."""
        p = Portfolio(returns_3, weights={"A": 1.0, "B": 0.0, "C": 0.0})
        rc = p.get_risk_contribution()
        np.testing.assert_allclose(rc.loc["A", "pct_contrib"], 1.0, atol=1e-6)


class TestRollingAndDrawdown:
    def test_rolling_shape(self, portfolio_eq):
        roll = portfolio_eq.get_rolling_stats(window=60)
        assert set(roll.columns) == {"ann_return", "ann_vol", "sharpe"}

    def test_drawdown_range(self, portfolio_eq):
        dd = portfolio_eq.get_drawdown_series()
        assert dd.max() <= 1e-10
        assert dd.min() <= 0
        assert len(dd) == 500

    def test_drawdown_monotonic_recovery(self, portfolio_eq):
        """Drawdown should be <= 0 everywhere."""
        dd = portfolio_eq.get_drawdown_series()
        assert (dd <= 0).all()


class TestRepr:
    def test_repr(self, portfolio_eq):
        assert "Portfolio" in repr(portfolio_eq)
        assert "Test" in repr(portfolio_eq)