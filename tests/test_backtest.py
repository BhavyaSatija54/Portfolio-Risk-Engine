"""
Tests for Backtesting — Kupiec, Christoffersen, Basel.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from risk_engine import VaRBacktest, BacktestMetrics

rng = np.random.default_rng(42)


@pytest.fixture
def perfect_var() -> Tuple[pd.Series, pd.Series]:
    """VaR calibrated so ~5% violations occur."""
    n = 1000
    returns = pd.Series(rng.normal(0, 0.02, n),
                       index=pd.date_range("2020-01-01", periods=n, freq="B"))
    var95 = abs(np.percentile(returns, 5))
    var_series = pd.Series([var95] * n, index=returns.index)
    return returns, var_series


@pytest.fixture
def conservative_var() -> Tuple[pd.Series, pd.Series]:
    n = 1000
    returns = pd.Series(rng.normal(0, 0.02, n),
                       index=pd.date_range("2020-01-01", periods=n, freq="B"))
    var_series = pd.Series([abs(np.percentile(returns, 5)) * 2.5] * n, index=returns.index)
    return returns, var_series


@pytest.fixture
def aggressive_var() -> Tuple[pd.Series, pd.Series]:
    n = 1000
    returns = pd.Series(rng.normal(0, 0.02, n),
                       index=pd.date_range("2020-01-01", periods=n, freq="B"))
    var_series = pd.Series([abs(np.percentile(returns, 5)) * 0.3] * n, index=returns.index)
    return returns, var_series


class TestInitialization:
    def test_basic(self, perfect_var):
        r, v = perfect_var
        bt = VaRBacktest(r, v, confidence_level=0.95)
        assert bt.T == 1000
        assert 30 <= bt.N <= 70  # ~50 expected

    def test_misaligned(self):
        r = pd.Series([0.01, -0.02, 0.005],
                       index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
        v = pd.Series([0.03, 0.03],
                       index=pd.to_datetime(["2020-01-02", "2020-01-04"]))
        bt = VaRBacktest(r, v, confidence_level=0.95)
        assert bt.T == 1  # only 2020-01-02 overlaps

    def test_no_overlap_raises(self):
        r = pd.Series([0.01], index=pd.to_datetime(["2020-01-01"]))
        v = pd.Series([0.03], index=pd.to_datetime(["2020-01-02"]))
        with pytest.raises(ValueError):
            VaRBacktest(r, v)

    def test_invalid_confidence(self):
        s = pd.Series([0.0])
        with pytest.raises(ValueError):
            VaRBacktest(s, s, confidence_level=0.0)
        with pytest.raises(ValueError):
            VaRBacktest(s, s, confidence_level=1.0)


class TestKupiec:
    def test_perfect_calibration(self, perfect_var):
        r, v = perfect_var
        bt = VaRBacktest(r, v, confidence_level=0.95)
        lr, pval, passed = bt.kupiec()
        assert lr >= 0
        assert 0 <= pval <= 1
        # Should usually pass for well-calibrated VaR
        assert passed or pval > 0.01  # allow occasional false rejection

    def test_conservative_few_violations(self, conservative_var):
        r, v = conservative_var
        bt = VaRBacktest(r, v, confidence_level=0.95)
        assert bt.N < 15  # very few violations
        _, pval, _ = bt.kupiec()
        assert pval >= 0  # still a valid probability

    def test_aggressive_many_violations(self, aggressive_var):
        r, v = aggressive_var
        bt = VaRBacktest(r, v, confidence_level=0.95)
        assert bt.N > 100  # many violations
        _, _, passed = bt.kupiec()
        assert not passed  # should reject


class TestChristoffersen:
    def test_returns_tuple(self, perfect_var):
        r, v = perfect_var
        bt = VaRBacktest(r, v)
        lr, pval, trans, passed = bt.christoffersen()
        assert lr >= 0
        if trans is not None:
            assert set(trans.keys()) == {"n00", "n01", "n10", "n11"}

    def test_zero_violations(self):
        r = pd.Series(np.full(100, 0.01))
        v = pd.Series(np.full(100, 0.10))
        bt = VaRBacktest(r, v)
        lr, pval, trans, passed = bt.christoffersen()
        assert passed  # inconclusive → True
        assert trans is None

    def test_all_violations(self):
        r = pd.Series(np.full(100, -0.05))
        v = pd.Series(np.full(100, 0.001))
        bt = VaRBacktest(r, v)
        lr, pval, trans, passed = bt.christoffersen()
        assert passed  # inconclusive
        assert trans is None


class TestConditionalCoverage:
    def test_joint(self, perfect_var):
        r, v = perfect_var
        bt = VaRBacktest(r, v)
        lr, pval, passed = bt.conditional_coverage()
        assert lr >= 0
        assert 0 <= pval <= 1


class TestBasel:
    def test_returns_valid_zone(self, perfect_var):
        zone = VaRBacktest(*perfect_var).basel_zone()
        assert zone in ("green", "yellow", "red")

    def test_conservative_few_violations(self, conservative_var):
        r, v = conservative_var
        bt = VaRBacktest(r, v)
        assert bt.N < 30  # very few breaches

    def test_aggressive_many_violations(self, aggressive_var):
        r, v = aggressive_var
        bt = VaRBacktest(r, v)
        assert bt.N > 80  # many breaches


class TestRun:
    def test_returns_metrics(self, perfect_var):
        r, v = perfect_var
        m = VaRBacktest(r, v).run()
        assert isinstance(m, BacktestMetrics)
        assert m.n_obs == 1000
        assert m.basel_zone in ("green", "yellow", "red")
        assert m.kupiec_pass in (True, False)
        assert m.ind_pass in (True, False)
        assert m.cc_pass in (True, False)

    def test_violation_rate_close_to_expected(self, perfect_var):
        r, v = perfect_var
        m = VaRBacktest(r, v, confidence_level=0.95).run()
        assert abs(m.violation_rate - m.expected_rate) < 0.03  # within 3%

    def test_summary(self, perfect_var):
        s = VaRBacktest(*perfect_var).summary()
        assert "BACKTEST" in s
        assert "Kupiec" in s
        assert "Christoffersen" in s
        assert "Basel" in s


class TestViolationSeries:
    def test_boolean_dtype(self, perfect_var):
        bt = VaRBacktest(*perfect_var)
        v = bt.get_violation_series()
        assert v.dtype == bool
        assert v.sum() == bt.N