# Portfolio Risk Engine

> Institutional-grade quantitative risk management — inspired by the precision and rigour expected at top-tier trading firms.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-106%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-55%25-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

### VaR Models (3 Families)

| Model | Key Features |
|-------|-------------|
| **Historical VaR** | Empirical percentile with optional **Hull-White age-weighting** (`decay=0.99`) and bootstrap confidence intervals |
| **Parametric VaR** | Normal or **Student-t** distribution, 4 volatility models (standard, EWMA, **GARCH(1,1)**, explicit vol), closed-form ES for all variants |
| **Monte Carlo VaR** | Cholesky-decomposed correlated simulation, normal/Student-t innovations, path visualization, condition-number tracking |

### Dual GARCH(1,1) Implementations

| Implementation | Best For | Key Details |
|---------------|----------|-------------|
| **`GARCHModel`** (arch-backed) | Production speed | C-optimised, supports Student-t errors, rolling forecasts, full diagnostic suite |
| **`GARCH11`** (from-scratch MLE) | Educational/interview | Log-space reparameterisation, soft stationarity penalty, dual optimiser (L-BFGS-B → Nelder-Mead), Ljung-Box diagnostics |

### Backtesting Framework

- **Kupiec POF test** (unconditional coverage)
- **Christoffersen** independence test + **conditional coverage**
- **Basel traffic light** approach (binomial CDF + 250-day scaling)
- **`BacktestSuite`** for multi-model comparison

### Portfolio Analytics

- **Euler risk allocation**: mathematically correct contribution-to-risk via homogeneous function theorem
- Rolling Sharpe/volatility, max drawdown, Sortino, Calmar ratios
- Tail ratios (5% and 1%)
- Component correlation heatmap

### Visualization

- **GitHub-dark theme** for publication-quality charts
- 8 chart types: VaR comparison, GARCH 3-panel dashboard, MC paths, drawdown, correlation heatmap, returns histogram with QQ-plot, backtest violations, 7-panel full dashboard

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourname/portfolio-risk-engine.git
cd portfolio-risk-engine
pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
from risk_engine import (
    Portfolio, HistoricalVaR, ParametricVaR, MonteCarloVaR,
    GARCH11, VaRBacktest, BacktestSuite
)

# 1. Load returns (or use DataFetcher)
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
pf = Portfolio(returns)

# 2. Calculate VaR across models
hist  = HistoricalVaR(pf.get_portfolio_returns(), confidence_level=0.99, notional=10e6)
par   = ParametricVaR(pf.get_portfolio_returns(), confidence_level=0.99,
                      notional=10e6, distribution='t')
mc    = MonteCarloVaR(pf.get_portfolio_returns(), confidence_level=0.99,
                      notional=10e6, n_simulations=100000)

r_h = hist.calculate()   # VaRResult(var_pct, var_abs, es_pct, es_abs, ...)
r_p = par.calculate()
r_m = mc.calculate()

# 3. GARCH(1,1) from scratch
g11 = GARCH11()
res = g11.fit(pf.get_portfolio_returns())
print(f"GARCH: alpha={res.params.alpha:.4f}, beta={res.params.beta:.4f}")

# 4. Backtest
bt = VaRBacktest(pf.get_portfolio_returns().values,
                 var_series=r_h.var_pct,
                 confidence_level=0.99,
                 model_name="Historical VaR")
metrics = bt.run()
print(f"Kupiec p-value: {metrics.kupiec_pvalue:.3f}")
```

### Hull-White Age-Weighted Historical VaR

```python
hist_aw = HistoricalVaR(
    returns, confidence_level=0.99, notional=10e6,
    age_weighted=True, decay=0.99   # recent observations weighted exponentially
)
result = hist_aw.calculate()
```

### Student-t Parametric VaR with Rolling Window

```python
par = ParametricVaR(returns, confidence_level=0.99,
                    notional=10e6, distribution='t')
result = par.calculate()           # uses MLE-fitted Student-t
```

### Multi-Model Backtest Comparison

```python
suite = BacktestSuite(confidence_level=0.99)
results = suite.run(actual_returns, {
    "Historical":  hist_var_series,
    "Student-t":   t_var_series,
    "Monte Carlo": mc_var_series,
})
df = suite.summary_table(results)
```

### Visualization

```python
from risk_engine.visualization import RiskVisualizer

viz = RiskVisualizer(dark=True)   # or dark=False for light theme

# Individual charts
fig = viz.plot_returns_hist(returns, var_lines={"VaR": var_pct})
fig = viz.plot_var_comparison({"Hist": h_var, "Par": p_var, "MC": m_var})
fig = viz.plot_garch_vol(returns, cond_vol, garch_params=params)
fig = viz.plot_mc_paths(paths_2d, var_pct=var_pct, confidence=0.99)
fig = viz.plot_backtest(returns, var_series, violations, metrics=metrics)

# Full 7-panel dashboard
fig = viz.dashboard(
    portfolio,
    var_results={"Hist": h_var, "Par": p_var, "MC": m_var},
    cond_vol=cond_vol,
    drawdown=drawdown_series,
    garch_params=params,
    garch_residuals=residuals,
)
```

---

## Project Structure

```
portfolio-risk-engine/
  src/
    risk_engine/
      __init__.py          # Package exports
      data_fetcher.py      # Yahoo Finance integration (~30yr lookback)
      portfolio.py         # Multi-asset portfolio with Euler allocation
      var_models.py        # Historical, Parametric, Monte Carlo VaR
      garch_model.py       # arch-backed GARCH (production)
      garch_mle.py         # From-scratch GARCH MLE (educational)
      backtest.py          # Kupiec, Christoffersen, Basel traffic light
      visualization.py     # 8 chart types, dark/light themes
  tests/
    test_var_models.py     # 28 tests — all models, edge cases, invariants
    test_portfolio.py      # 20 tests — weights, stats, risk contribution
    test_garch.py          # 18 tests — arch-backed GARCH
    test_garch11.py        # 16 tests — from-scratch GARCH MLE
    test_backtest.py       # 24 tests — Kupiec, Christoffersen, Basel
  notebooks/
    demo.ipynb             # Interactive walkthrough
  dashboard/
    app.py                 # Streamlit interactive dashboard
  docs/images/             # 8 dark-theme visualization examples
```

---

## Running Tests

```bash
# Full suite (106 tests)
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/ -v --tb=short

# With coverage (minimum 50%)
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/ --cov=risk_engine --cov-report=term-missing

# Single module
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_garch11.py -v
```

---

## Mathematical Foundations

**VaR & ES**: Square-root-of-time scaling, closed-form Student-t ES

$$\text{ES}_\alpha = \mu - \sigma \cdot \frac{f_T(z_\alpha)}{\alpha} \cdot \frac{\nu + z_\alpha^2}{\nu - 1}$$

**Euler Risk Allocation**: For homogeneous risk measure $R$ of degree 1:

$$R(w) = \sum_i w_i \frac{\partial R}{\partial w_i} = \sum_i \text{CTE}_i$$

**GARCH(1,1)**: Log-space MLE with soft stationarity penalty

$$\mathcal{L}(\theta) = -\frac{1}{2}\sum_{t=1}^T \left[\ln(2\pi\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2}\right] - \lambda \cdot \max(\alpha+\beta-1, 0)^2$$

**Cholesky Monte Carlo**: $\Sigma = LL^T$, $R = \mu + L \cdot Z$ where $Z \sim N(0,I)$. PSD fallback via eigendecomposition.

**Hull-White Age-Weighting**: $w_i = \lambda^{T-i-1} / \sum_j \lambda^{T-j-1}$ for decay $\lambda \in (0,1)$.

**Kupiec POF**: $LR_{POF} = -2\ln\left[\frac{(1-p)^{T-x}p^x}{(1-x/T)^{T-x}(x/T)^x}\right] \sim \chi^2_1$

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built with the precision expected at top quantitative trading firms — immutable data structures, defensive validation, mathematically correct formulas, and comprehensive test coverage.*
