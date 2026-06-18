# GARCH Implementation Comparison

## Executive Summary

Their implementation is a **from-scratch MLE** with proper mathematical rigor. Yours uses the battle-tested `arch` library. Each has distinct advantages. A production-grade repo should ideally offer **both** — the from-scratch version demonstrates depth, the library version provides reliability.

---

## Their Implementation (from-scratch)

### Strengths

| Feature | Assessment |
|---------|-----------|
| **Log-space reparameterization** | `theta = (ln w, ln a, ln b)` — textbook approach for enforcing positivity without hard constraints. This is exactly how you'd implement it in C++ for a prop trading system. |
| **Soft penalty for stationarity** | `ll -= 1e6 * (ab - 1)^2` when `a+b >= 1` — elegant, differentiable, better than hard clipping during optimization. |
| **Dual optimizer fallback** | L-BFGS-B (gradient-based, fast) → Nelder-Mead (gradient-free, robust) — proper production pattern for numerical stability. |
| **Closed-form multi-step forecast** | `var_unc + (ab^k) * (s2_T1 - var_unc)` — mathematically correct mean-reversion formula, not naive recursion. |
| **GARCH-filtered VaR** | `VaR_garch = VaR_hist * (sigma_T+1 / sigma_hist)` — this is an **industry technique** used by RiskMetrics and some bank systems. Different from standard parametric-GARCH but valid. |
| **Built-in diagnostics** | Ljung-Box on z and z^2, Jarque-Bera — essential for model validation. |
| **No external GARCH dependency** | Self-contained, auditable, no black-box library calls. |

### Weaknesses

| Issue | Severity |
|-------|----------|
| **Python loop in likelihood** | `for t in range(1, T):` — O(n) Python loop per likelihood eval. For 10k+ obs with L-BFGS-B (hundreds of evals), this is **slow**. `arch` uses Cython. |
| **Normal errors only** | No Student-t, skew-t, or GED. Real financial returns have fat tails — t-distribution GARCH is standard in industry. |
| **Mu pre-estimated** | Mean is subtracted before MLE, not jointly estimated. Biased inference if mean is uncertain. `arch` estimates jointly. |
| **Persistence clipping** | `alpha / total * 0.999` — post-hoc fix that changes the MLE optimum. Better to let the penalty handle it during optimization. |
| **No rolling forecast API** | Must re-fit for each window manually. `arch` + your wrapper provide `rolling_forecast()`. |
| **Mutable dataclasses** | `@dataclass` (not frozen) — results can be mutated accidentally. |

---

## Your Implementation (arch-backed)

### Strengths

| Feature | Assessment |
|---------|-----------|
| **Uses `arch` library** | Kevin Sheppard's package — the standard in academic finance. C-optimized likelihood, battle-tested optimizers, handles edge cases. |
| **Multiple distributions** | Normal, Student-t, skewed-t via `dist='normal'` / `dist='t'` — essential for realistic tail modeling. |
| **Percentage-scaled returns** | `returns * 100` before fitting — avoids numerical underflow with very small daily returns. Proper practice. |
| **Immutable results** | `frozen=True` dataclasses — mutation bugs impossible. Production-grade. |
| **Rolling forecasts** | `rolling_forecast(window=252, step=1)` — essential for backtesting VaR through time. |
| **Cleaner API** | `fit() -> forecast() -> conditional_vol()` — method chaining, less boilerplate. |

### Weaknesses

| Issue | Severity |
|-------|----------|
| **Black-box dependency** | If `arch` breaks or changes API, you're dependent. Their code is self-contained. |
| **No from-scratch MLE** | Doesn't demonstrate understanding of the optimization problem — important for quant interviews. |
| **No residual diagnostics** | Missing Ljung-Box, Jarque-Bera, ARCH-LM tests in the public API. Users must implement separately. |
| **GARCH-parametric VaR only** | No GARCH-filtered VaR rescaling approach (which their code has). Both are valid industry techniques. |

---

## Side-by-Side: Key Code Patterns

### Likelihood Evaluation

```python
# Their approach: Python loop (slow but transparent)
for t in range(1, T):
    sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    sigma2[t] = max(sigma2[t], 1e-12)
ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)

# Your approach: C-optimized via arch library (fast, opaque)
# Hidden inside arch_model(...).fit(...)
```

### Parameter Constraints

```python
# Their approach: Log-space reparametrization + soft penalty
# Enforces w>0, a>0, b>0 naturally; stationarity via penalty

# Your approach: arch handles internally (trust_region algorithm with bounds)
```

### Multi-Step Forecast

```python
# Their approach: Closed-form mean-reversion (analytically correct)
var_unc = omega / (1 - ab)
sigma2_h = var_unc + (ab ** np.arange(1, h+1)) * (var_T1 - var_unc)

# Your approach: arch.forecast(horizon=h) — equivalent internally
```

---

## Recommendation

### For Your GitHub Repo

**Merge both approaches into a unified module:**

```
src/risk_engine/
    garch_model.py      ← your arch-based version (primary, production)
    garch_mle.py        ← their from-scratch MLE (educational, interview-ready)
```

This gives you:
1. **Production path**: `GARCHModel` (arch-backed) — fast, reliable, multiple distributions
2. **Interview path**: `GARCH11MLE` (from-scratch) — demonstrates you understand every line

### What to Port From Their Code

These specific pieces are **better** than yours and should be adopted:

1. **Residual diagnostics** — port their `residual_diagnostics()` method directly
2. **GARCH-filtered VaR** — add their `garch_var()` static method as an alternative VaR calculation
3. **Log-space reparametrization** — document it in comments for educational value
4. **Closed-form multi-step forecast** — verify your arch-based forecasts match this analytically

### What They Should Port From Yours

1. **Frozen dataclasses** — `@dataclass(frozen=True)` is strictly better
2. **Percentage scaling** — `returns * 100` before fitting improves numerical stability
3. **Distribution flexibility** — Student-t GARCH is standard; normal-only is a limitation
4. **Rolling forecast API** — essential for time-series backtesting

---

## Verdict

| Criterion | Winner | Notes |
|-----------|--------|-------|
| Mathematical rigor | **Theirs** | From-scratch MLE, log-reparam, soft penalty |
| Production reliability | **Yours** | arch library, C-optimized, multiple distributions |
| Code quality | **Yours** | Frozen dataclasses, cleaner API, type hints |
| Performance | **Yours** | Cython backend vs Python loops |
| Educational value | **Theirs** | Every step is visible and auditable |
| Industry realism | **Tie** | Both valid; arch-backed is more common in practice |

**Bottom line**: Their implementation would score higher in a **quant interview** (shows you can derive and implement MLE from scratch). Yours would score higher in a **production code review** (reliable, maintainable, tested). A senior Jane Street engineer would want to see **both** in a portfolio.