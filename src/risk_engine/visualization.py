"""
Visualization Module — Production Grade

Static matplotlib / seaborn plots with defensive style handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_FIGSIZE = (12, 7)
_VALID_STYLES = ("seaborn-v0_8-whitegrid", "seaborn-v0_8-darkgrid",
                  "seaborn-whitegrid", "seaborn-darkgrid",
                  "ggplot", "bmh")


def _safe_style(style: str) -> str:
    """Fall back gracefully if the requested style is unavailable."""
    if style in plt.style.available:
        return style
    for s in _VALID_STYLES:
        if s in plt.style.available:
            return s
    logger.debug("No preferred style found; using default")
    return "default"


class RiskVisualizer:
    """Publication-quality risk plots."""

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
    ) -> None:
        self.style = _safe_style(style)
        self.figsize = figsize
        plt.style.use(self.style)

    # ------------------------------------------------------------------
    # Core plots
    # ------------------------------------------------------------------

    def plot_returns_hist(
        self,
        returns: pd.Series,
        var_lines: Optional[Dict[str, float]] = None,
        bins: int = 120,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        ax.hist(returns, bins=bins, density=True, color="#2E86AB",
                alpha=0.65, edgecolor="black", linewidth=0.3, label="Returns")

        # Tail shading
        var_5 = np.percentile(returns, 5)
        for patch, left in zip(ax.patches, ax.patches):
            if left.get_x() < var_5:
                patch.set_facecolor("#C73E1D")
                patch.set_alpha(0.75)

        # Normal overlay
        mu, sig = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 500)
        ax.plot(x, self._norm_pdf(x, mu, sig), "k-", lw=1.5, label="Normal fit")

        # VaR reference lines
        if var_lines:
            cmap = {"Historical": "#C73E1D", "Parametric": "#F18F01",
                    "MonteCarlo": "#2E86AB", "GARCH": "#A23B72"}
            for name, v in var_lines.items():
                ax.axvline(-v, color=cmap.get(name, "#555"), ls="--", lw=1.5,
                          label=f"{name} VaR")

        ax.set_xlabel("Daily return", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Return Distribution", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        return ax

    def plot_var_comparison(
        self,
        results: Dict[str, float],
        notional: float = 1.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        names = list(results.keys())
        vals = [results[n] * notional for n in names]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

        bars = ax.barh(names, vals, color=colors[: len(names)],
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                   f"${v:,.0f}" if notional != 1.0 else f"{v:.4f}",
                   va="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("VaR", fontsize=11)
        ax.set_title("VaR Comparison", fontsize=13, fontweight="bold")
        return ax

    def plot_backtest(
        self,
        returns: pd.Series,
        var_series: pd.Series,
        violations: pd.Series,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(returns.index, returns.values, 0, alpha=0.3, color="gray")
        ax.plot(var_series.index, -var_series.values, color="#C73E1D",
               lw=1.2, label="VaR")

        vret = returns[violations]
        if len(vret):
            ax.scatter(vret.index, vret.values, color="red", s=35,
                      marker="v", zorder=5, edgecolors="black", linewidth=0.4,
                      label=f"Violations ({violations.sum()})")

        ax.axhline(0, color="black", lw=0.5)
        rate = violations.sum() / len(violations) if len(violations) else 0
        ax.text(0.02, 0.97, f"Violation rate: {rate:.2%}",
               transform=ax.transAxes, fontsize=10, va="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Return", fontsize=11)
        ax.set_title("VaR Backtest — Violations", fontsize=13, fontweight="bold")
        ax.legend(loc="lower left")
        return ax

    def plot_garch_vol(
        self,
        returns: pd.Series,
        cond_vol: pd.Series,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 6))

        ax.plot(returns.index, returns.values, color="gray", alpha=0.35, lw=0.4)
        ax.plot(cond_vol.index, cond_vol.values, color="#C73E1D", lw=1,
               label=r"Conditional Vol ($+1\sigma$)")
        ax.plot(cond_vol.index, -cond_vol.values, color="#C73E1D", lw=1, ls="--",
               label=r"$-1\sigma$")
        ax.fill_between(cond_vol.index, 2 * cond_vol, -2 * cond_vol,
                       alpha=0.06, color="blue")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Return / Vol", fontsize=11)
        ax.set_title("GARCH(1,1) Conditional Volatility", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left")
        return ax

    def plot_drawdown(
        self,
        drawdown: pd.Series,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(drawdown.index, drawdown.values, 0,
                       alpha=0.5, color="#C73E1D")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Drawdown", fontsize=11)
        ax.set_title("Drawdown History", fontsize=13, fontweight="bold")
        ax.axhline(0, color="black", lw=0.5)
        return ax

    def plot_rolling_sharpe(
        self,
        rolling_sharpe: pd.Series,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#2E86AB", lw=1.2)
        ax.axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(rolling_sharpe.mean(), color="green", ls=":", lw=1, alpha=0.6,
                  label=f"Mean = {rolling_sharpe.mean():.2f}")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Sharpe Ratio", fontsize=11)
        ax.set_title("Rolling 1-Year Sharpe Ratio", fontsize=13, fontweight="bold")
        ax.legend()
        return ax

    def plot_mc_vs_hist(
        self,
        simulated: np.ndarray,
        historical: pd.Series,
        var_mc: float,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        ax.hist(simulated, bins=250, density=True, alpha=0.5, color="#2E86AB",
               edgecolor="none", label="Monte Carlo")
        ax.hist(historical, bins=100, density=True, alpha=0.5, color="#F18F01",
               edgecolor="none", label="Historical")
        ax.axvline(-var_mc, color="#C73E1D", lw=2, ls="--",
                  label=f"MC VaR = {var_mc:.4f}")
        lo, hi = np.percentile(simulated, [1, 99])
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Return", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Monte Carlo vs Historical Distribution", fontsize=13,
                    fontweight="bold")
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Composite dashboard
    # ------------------------------------------------------------------

    def dashboard(
        self,
        portfolio,
        var_results: Dict[str, float],
        cond_vol: Optional[pd.Series] = None,
        drawdown: Optional[pd.Series] = None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        fig.suptitle(f"Risk Dashboard — {portfolio.name}", fontsize=16,
                    fontweight="bold")

        r = portfolio.get_portfolio_returns()
        w = portfolio.get_weights()

        # (0,0) cumulative return
        cum = (1.0 + r).cumprod()
        axes[0, 0].plot(cum.index, cum.values, lw=1.5, color="#2E86AB")
        axes[0, 0].set_title("Cumulative Return")
        axes[0, 0].axhline(1, color="black", lw=0.5)

        # (0,1) return distribution + VaR
        self.plot_returns_hist(r, var_lines=var_results, ax=axes[0, 1])

        # (0,2) VaR comparison
        self.plot_var_comparison(var_results, ax=axes[0, 2])

        # (1,0) weights pie
        colors_pie = matplotlib.cm.get_cmap("Set3")(np.linspace(0, 1, len(w)))
        axes[1, 0].pie(w.values, labels=w.index.tolist(), autopct="%1.1f%%",
                      colors=colors_pie, startangle=90)
        axes[1, 0].set_title("Allocation")

        # (1,1) drawdown or rolling sharpe
        if drawdown is not None:
            self.plot_drawdown(drawdown, ax=axes[1, 1])
        else:
            roll = portfolio.get_rolling_stats(window=252)
            self.plot_rolling_sharpe(roll["sharpe"].dropna(), ax=axes[1, 1])

        # (1,2) GARCH or correlation
        if cond_vol is not None:
            self.plot_garch_vol(r.iloc[-500:], cond_vol.iloc[-500:], ax=axes[1, 2])
        else:
            corr = portfolio.get_component_returns().corr()
            im = axes[1, 2].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1, 2].set_xticks(range(len(corr)))
            axes[1, 2].set_yticks(range(len(corr)))
            axes[1, 2].set_xticklabels(corr.columns, rotation=45, ha="right")
            axes[1, 2].set_yticklabels(corr.columns)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    axes[1, 2].text(j, i, f"{corr.iloc[i, j]:.2f}",
                                   ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=axes[1, 2], shrink=0.7)
            axes[1, 2].set_title("Correlation Matrix")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    @staticmethod
    def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))