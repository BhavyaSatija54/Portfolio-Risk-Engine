# Portfolio Risk Engine

[![CI](https://github.com/yourusername/portfolio-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/portfolio-risk-engine/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/portfolio-risk-engine/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/portfolio-risk-engine)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive quantitative risk management framework implementing **Value at Risk (VaR)** estimation across multiple methodologies, **GARCH(1,1) volatility modeling**, and **statistical backtesting** (Kupiec's POF test, Christoffersen independence test, Basel regulatory framework) for model validation and performance assessment.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [VaR Methodologies](#var-methodologies)
  - [Historical VaR](#historical-var)
  - [Parametric VaR](#parametric-var)
  - [Monte Carlo VaR (Cholesky)](#monte-carlo-var)
- [GARCH(1,1) Volatility Modeling](#garch11-volatility-modeling)
- [Backtesting Framework](#backtesting-framework)
- [Project Structure](#project-structure)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Jupyter Notebook](#jupyter-notebook)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Theory and Formulas](#theory-and-formulas)
- [License](#license)

---

## Overview

This risk engine provides a production-ready toolkit for portfolio risk assessment, supporting:

- **Multi-asset portfolio analysis** with live data from Yahoo Finance
- **Three VaR methodologies**: Historical (non-parametric), Parametric (normal/EWMA/GARCH), and Monte Carlo (Cholesky decomposition)
- **GARCH(1,1) volatility forecasting** for time-varying risk
- **Comprehensive backtesting**: Kupiec's POF test, Christoffersen independence test, conditional coverage, and Basel traffic light framework
- **Interactive visualizations** via Streamlit dashboard and Jupyter notebooks

---

## Features

| Component | Description |
|-----------|-------------|
| **DataFetcher** | Live market data retrieval from Yahoo Finance with data validation |
| **Portfolio** | Multi-asset portfolio construction with risk contribution analysis |
| **HistoricalVaR** | Non-parametric VaR from empirical distribution with bootstrap CIs |
| **ParametricVaR** | Normal/EWMA/GARCH-enhanced VaR with square-root-of-time scaling |
| **MonteCarloVaR** | Correlated return simulation via Cholesky decomposition |
| **GARCHModel** | GARCH(1,1) estimation with rolling forecasts and diagnostics |
| **KupiecBacktest** | Kupiec POF, Christoffersen independence, Basel framework |
| **RiskVisualizer** | Publication-quality risk dashboards and analysis charts |

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-risk-engine.git
cd portfolio-risk-engine

# Create virtual environment (recommended)
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Optional Dependencies

```bash
# For Jupyter notebook
pip install jupyter ipywidgets

# For Streamlit dashboard
pip install streamlit plotly

# For development (testing, linting)
pip install pytest pytest-cov black flake8
```

---

## Quick Start

```python
from risk_engine import (
    DataFetcher, Portfolio,
    HistoricalVaR, ParametricVaR, MonteCarloVaR,
    GARCHModel, KupiecBacktest, RiskVisualizer
)

# 1. Fetch data
fetcher = DataFetcher(tickers=["AAPL", "MSFT", "GOOGL"], lookback_days=7560)
fetcher.fetch_data()

# 2. Build portfolio
portfolio = Portfolio(returns=fetcher.get_returns(), name="Tech Portfolio")

# 3. Calculate VaR (all methods)
returns = portfolio.get_portfolio_returns()

historical_var = HistoricalVaR(returns, confidence_level=0.95).calculate()
parametric_var = ParametricVaR(returns, confidence_level=0.95).calculate()

# Monte Carlo with Cholesky
weights = portfolio.get_weights().values
mc_var = MonteCarloVaR(
    returns, confidence_level=0.95, n_simulations=100000,
    use_cholesky=True, component_returns=fetcher.get_returns(),
    component_weights=weights
).calculate()

# 4. GARCH volatility
from risk_engine.garch_model import GARCHModel
garch = GARCHModel(returns).fit()
garch_vol = garch.get_forecasted_volatility()

# GARCH-enhanced VaR
garch_var = ParametricVaR(
    returns, confidence_level=0.95, volatility_model='garch',
    garch_forecast=garch_vol
).calculate()

# 5. Backtest
from risk_engine.backtest import KupiecBacktest
backtest = KupiecBacktest(returns, var_series, confidence_level=0.95)
print(backtest.summary())
```

---

## VaR Methodologies

### Historical VaR

Non-parametric approach using the empirical distribution of historical returns. No distributional assumptions required - naturally captures fat tails, skewness, and other non-normal features.

```python
hist_var = HistoricalVaR(
    returns=portfolio_returns,
    confidence_level=0.95,
    portfolio_value=1_000_000,
    bootstrap=True,  # Enable confidence intervals
    n_bootstrap=10000
)
var = hist_var.calculate()
ci = hist_var.get_confidence_interval()
```

**Formula:**
```
VaR_hist = |percentile(returns, alpha)|
```
where `alpha = 1 - confidence_level`

**Features:**
- Bootstrap confidence intervals
- Overlapping multi-period scaling (more accurate than sqrt-time)
- Expected Shortfall (CVaR) calculation

---

### Parametric VaR

Assumes returns follow a normal distribution. Supports multiple volatility estimation methods:

```python
# Standard historical volatility
param_var = ParametricVaR(returns, confidence_level=0.95, volatility_model='standard')

# EWMA (RiskMetrics)
ewma_var = ParametricVaR(returns, confidence_level=0.95, volatility_model='ewma', ewma_lambda=0.94)

# GARCH-enhanced
garch = GARCHModel(returns).fit()
garch_var = ParametricVaR(
    returns, confidence_level=0.95, volatility_model='garch',
    garch_forecast=garch.get_forecasted_volatility()
)
```

**Formula:**
```
VaR_param = Z_alpha * sigma * sqrt(holding_period)
```
where `Z_alpha` is the standard normal quantile at confidence level `alpha`

---

### Monte Carlo VaR

Generates correlated random returns using Cholesky decomposition of the covariance matrix for realistic multi-asset simulation.

```python
mc_var = MonteCarloVaR(
    returns=portfolio_returns,
    confidence_level=0.95,
    portfolio_value=1_000_000,
    n_simulations=100_000,
    random_seed=42,
    use_cholesky=True,                          # Enable correlation modeling
    simulation_model='normal',                  # or 't' for Student-t
    component_returns=individual_asset_returns,  # For correlation structure
    component_weights=portfolio_weights         # Asset allocation
)
var = mc_var.calculate()
simulated = mc_var.get_simulated_distribution()
```

**Cholesky Decomposition:**
```
Sigma = L * L^T   (covariance matrix factorization)
r_sim = mu + L * z   where z ~ N(0, I)
```

**Features:**
- Normal and Student-t distributions
- Full correlation structure via Cholesky
- Simulated distribution output for further analysis

---

## GARCH(1,1) Volatility Modeling

Captures volatility clustering and mean-reversion in financial returns.

**Model Specification:**
```
r_t = mu + epsilon_t,  epsilon_t ~ N(0, sigma_t^2)
sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
```

**Parameters:**
- `omega > 0`: Long-run average variance
- `alpha >= 0`: Reaction to recent shocks
- `beta >= 0`: Persistence of volatility
- Stationarity requires: `alpha + beta < 1`

```python
# Fit model
garch = GARCHModel(returns, distribution='normal').fit()

# Get parameters
params = garch.get_parameters()
print(f"Persistence: {params['persistence']:.4f}")
print(f"Half-life: {params['half_life']:.1f} days")

# Forecast volatility
forecast = garch.forecast_volatility(horizon=5)

# Rolling forecasts for backtesting
rolling_vol = garch.rolling_forecast(window_size=252, step_size=1)
```

---

## Backtesting Framework

Comprehensive statistical validation of VaR models.

### Kupiec's Proportion of Failures (POF) Test

Tests if the observed violation frequency matches the expected rate.

**Hypothesis:** `H0: p = p0` where `p0 = 1 - confidence_level`

**Test Statistic:**
```
LR_POF = -2 * ln[(1-p0)^(T-N) * p0^N / ((1-N/T)^(T-N) * (N/T)^N)] ~ chi^2(1)
```

### Christoffersen's Independence Test

Tests whether violations are independently distributed (no clustering).

**Test Statistic:**
```
LR_IND = -2 * ln[L(p) / L(p01, p11)] ~ chi^2(1)
```

where `p01 = P(V_t=1 | V_{t-1}=0)` and `p11 = P(V_t=1 | V_{t-1}=1)`

### Basel Traffic Light Framework

| Zone | Condition | Action |
|------|-----------|--------|
| **Green** | Model acceptable | Continue using |
| **Yellow** | Model needs review | Investigate and enhance |
| **Red** | Model unacceptable | Immediate remediation required |

```python
backtest = KupiecBacktest(returns, var_series, confidence_level=0.95)
print(backtest.summary())

# Individual tests
kupiec_result = backtest.kupiec_test()
independence_result = backtest.christoffersen_test()
conditional_result = backtest.conditional_coverage_test()
basel_zone = backtest.basel_framework()
```

---

## Project Structure

```
portfolio-risk-engine/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── data/                       # Data directory (gitignored)
├── notebooks/
│   └── demo.ipynb              # Interactive Jupyter demonstration
├── src/
│   └── risk_engine/
│       ├── __init__.py         # Package exports
│       ├── data_fetcher.py     # Yahoo Finance data retrieval
│       ├── portfolio.py        # Portfolio construction & analytics
│       ├── var_models.py       # Historical, Parametric, Monte Carlo VaR
│       ├── garch_model.py      # GARCH(1,1) volatility modeling
│       ├── backtest.py         # Kupiec POF & backtesting framework
│       └── visualization.py    # Risk visualization toolkit
├── tests/
│   ├── test_portfolio.py       # Portfolio tests
│   ├── test_var_models.py      # VaR model tests
│   ├── test_garch.py           # GARCH model tests
│   └── test_backtest.py        # Backtesting tests
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── pytest.ini                 # Test configuration
├── .gitignore
└── README.md                   # This file
```

---

## Streamlit Dashboard

Launch the interactive web dashboard:

```bash
streamlit run dashboard/app.py
```

**Features:**
- Configure tickers, confidence levels, and parameters via sidebar
- Real-time data fetching from Yahoo Finance
- Interactive Plotly visualizations
- Tabbed interface: Overview, VaR Analysis, GARCH Volatility, Backtesting
- Portfolio composition pie charts and correlation heatmaps
- Statistical test results with Basel framework indicators

---

## Jupyter Notebook

Open the interactive demonstration notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook includes:
1. Data fetching and validation
2. Portfolio construction with risk metrics
3. All three VaR methodologies with comparison
4. GARCH volatility modeling and forecasting
5. Kupiec backtesting with violation analysis
6. Comprehensive risk dashboard
7. Multi-period VaR analysis
8. Tail risk and rolling statistics

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=risk_engine --cov-report=html

# Run specific test file
pytest tests/test_var_models.py -v

# Run with markers
pytest tests/ -v -m "not slow"
```

Test coverage includes:
- **Portfolio**: Weight allocation, returns calculation, statistics, risk contribution
- **VaR Models**: Historical (with bootstrap), Parametric (standard/EWMA/GARCH), Monte Carlo (Cholesky)
- **GARCH**: Parameter estimation, forecasts, diagnostics, rolling windows
- **Backtest**: Kupiec POF, Christoffersen independence, Basel zones, edge cases

---

## CI/CD

GitHub Actions workflow (`/.github/workflows/ci.yml`) includes:

| Stage | Description |
|-------|-------------|
| **Lint** | flake8 style checking |
| **Format** | black code formatting verification |
| **Test** | pytest with coverage (Python 3.9-3.12) |
| **Build** | Package build and installation verification |
| **Notebook** | Jupyter notebook structure validation |

---

## Theory and Formulas

### Value at Risk

**Historical VaR:**
```
VaR_α = |F⁻¹(α)|
```
where `F⁻¹` is the empirical quantile function.

**Parametric VaR:**
```
VaR_α = Φ⁻¹(α) × σ × √h
```
where `Φ⁻¹` is the inverse standard normal CDF, `σ` is daily volatility, and `h` is the holding period.

**Monte Carlo VaR:**
```
r_sim = μ + L × z,  z ~ N(0, I)
VaR_α = |percentile(r_sim, α)|
```

### Expected Shortfall (CVaR)

**Historical:**
```
ES_α = mean({r_i : r_i ≤ -VaR_α})
```

**Parametric (Normal):**
```
ES_α = σ × φ(Φ⁻¹(α)) / (1-α)
```
where `φ` is the standard normal PDF.

### GARCH(1,1)

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Unconditional Variance: σ² = ω / (1 - α - β)
Half-life: τ_{1/2} = -ln(2) / ln(α + β)
```

### Kupiec POF Test

```
LR_POF = -2 × ln[(1-p₀)^{T-N} × p₀^N / ((1-N/T)^{T-N} × (N/T)^N)]
```

Under H₀, LR_POF ~ χ²(1)

---

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2024 Portfolio Risk Engine Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. All tests pass (`pytest tests/`)
3. New features include corresponding tests
4. Docstrings are provided for all public methods
5. Type hints are used where appropriate

---

## Acknowledgments

- **ARCH Library**: Kevin Sheppard's ARCH/GARCH implementation
- **Yahoo Finance**: Market data via yfinance
- **Basel Committee**: Regulatory framework for model validation