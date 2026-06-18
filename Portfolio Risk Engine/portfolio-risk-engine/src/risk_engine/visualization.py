"""
Visualization Module — Production Grade

Publication-quality static plots with optional dark theme,
QQ-plots, multi-panel GARCH, MC paths, and comprehensive dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_DEFAULT_FIGSIZE = (12, 7)

# Dark theme (GitHub-dark inspired)
DARK_STYLE = {
    "fig_facecolor": "#0d1117",
    "ax_facecolor":  "#161b22",
    "text_color":    "#e6edf3",
    "grid_color":    "#21262d",
    "accent":        "#58a6ff",
    "positive":      "#3fb950",
    "negative":      "#f85149",
    "warning":       "#d29922",
    "colors": [
        "#58a6ff", "#3fb950", "#f85149", "#d29922",
        "#bc8cff", "#79c0ff", "#56d364", "#ffa657",
    ],
}

# Light theme (classic)
LIGHT_STYLE = {
    "fig_facecolor": "white",
    "ax_facecolor":  "white",
    "text_color":    "black",
    "grid_color":    "#e0e0e0",
    "accent":        "#1f77b4",
    "positive":      "#2ca02c",
    "negative":      "#d62728",
    "warning":       "#ff7f0e",
    "colors": plt.rcParams["axes.prop_cycle"].by_key()["color"],
}


def _safe_style(style_name: str) -> str:
    candidates = [style_name, "seaborn-v0_8-whitegrid", "seaborn-whitegrid", "bmh", "ggplot"]
    for s in candidates:
        if s in plt.style.available:
            return s
    return "default"


class RiskVisualizer:
    """Publication-quality risk plots with light or dark theme."""

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
        dark: bool = False,
    ) -> None:
        self.style_name = _safe_style(style)
        self.figsize = figsize
        self.dark = dark
        self.S = DARK_STYLE if dark else LIGHT_STYLE
        if not dark:
            plt.style.use(self.style_name)

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _apply_theme(self, fig: plt.Figure, axes) -> None:
        if not self.dark:
            return
        fc, ac, tc, gc = self.S["fig_facecolor"], self.S["ax_facecolor"], self.S["text_color"], self.S["grid_color"]
        fig.patch.set_facecolor(fc)
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        for ax in np.array(axes).flatten():
            ax.set_facecolor(ac)
            ax.tick_params(colors=tc, labelsize=8)
            ax.xaxis.label.set_color(tc)
            ax.yaxis.label.set_color(tc)
            ax.title.set_color(tc)
            for spine in ax.spines.values():
                spine.set_edgecolor(gc)
            ax.grid(True, color=gc, linewidth=0.5, alpha=0.7)

    # ------------------------------------------------------------------
    # 1. Return distribution + VaR + QQ-plot
    # ------------------------------------------------------------------

    def plot_returns_hist(
        self,
        returns: pd.Series,
        var_lines: Optional[Dict[str, float]] = None,
        bins: int = 120,
        ax: Optional[plt.Axes] = None,
        show_qq: bool = True,
    ) -> plt.Figure:
        r = np.asarray(returns).flatten()

        if show_qq:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, axes = plt.subplots(figsize=self.figsize)
            axes = [axes]
        self._apply_theme(fig, axes)

        # --- Left: histogram ---
        ax0 = axes[0]
        ax0.hist(r, bins=bins, density=True, color=self.S["accent"],
                alpha=0.55, edgecolor="none")

        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(r)
        x_range = np.linspace(r.min(), r.max(), 600)
        ax0.plot(x_range, kde(x_range), color=self.S["accent"], lw=1.5, label="KDE")

        # Normal fit
        mu, sig = r.mean(), r.std(ddof=1)
        ax0.plot(x_range, stats.norm.pdf(x_range, mu, sig),
                color=self.S["warning"], lw=1.2, ls="--", alpha=0.8, label="Normal fit")

        # Tail shading
        var_5 = np.percentile(r, 5)
        for patch, left in zip(ax0.patches, ax0.patches):
            if left.get_x() < var_5:
                patch.set_facecolor(self.S["negative"])
                patch.set_alpha(0.7)

        # VaR lines
        if var_lines:
            cmap = self.S["colors"]
            for i, (name, v) in enumerate(var_lines.items()):
                ax0.axvline(-v, color=cmap[i % len(cmap)], ls="--", lw=1.5,
                           label=f"{name} VaR")

        ax0.set_xlabel("Daily return")
        ax0.set_ylabel("Density")
        ax0.set_title("Return Distribution")
        ax0.legend(loc="upper left", fontsize=8)

        # --- Right: QQ-plot ---
        if show_qq:
            ax1 = axes[1]
            (osm, osr), (slope, intercept, r_val) = stats.probplot(r, dist="norm", fit=True)
            ax1.scatter(osm, osr, s=4, alpha=0.5, color=self.S["accent"])
            x_line = np.array([osm.min(), osm.max()])
            ax1.plot(x_line, slope * x_line + intercept,
                    color=self.S["warning"], lw=1.5, ls="--", label="Normal reference")
            ax1.set_xlabel("Theoretical Quantiles")
            ax1.set_ylabel("Sample Quantiles")
            ax1.set_title(f"Normal Q-Q Plot (R² = {r_val**2:.4f})")
            ax1.legend(fontsize=8)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. VaR comparison bar chart
    # ------------------------------------------------------------------

    def plot_var_comparison(
        self,
        results: Dict[str, float],
        notional: float = 1.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            self._apply_theme(fig, ax)
        else:
            fig = ax.figure

        names = list(results.keys())
        vals = [results[n] * notional for n in names]
        colors = self.S["colors"]

        bars = ax.barh(names, vals, color=colors[: len(names)],
                      alpha=0.85, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                   f"${v:,.0f}" if notional != 1.0 else f"{v:.4f}",
                   va="center", fontsize=10, fontweight="bold",
                   color=self.S["text_color"])

        ax.set_xlabel("VaR" if notional == 1.0 else "VaR ($)")
        ax.set_title("VaR Comparison")
        return fig

    # ------------------------------------------------------------------
    # 3. Three-panel GARCH visualization
    # ------------------------------------------------------------------

    def plot_garch_vol(
        self,
        returns: pd.Series,
        cond_vol: pd.Series,
        garch_params=None,
        forecast_days: int = 60,
    ) -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        self._apply_theme(fig, axes)

        idx = returns.index[-len(cond_vol):]

        # Panel 1: Returns
        r = returns.values[-len(cond_vol):]
        pos = r > 0
        axes[0].bar(idx[pos], r[pos] * 100, color=self.S["positive"], alpha=0.7, width=1)
        axes[0].bar(idx[~pos], r[~pos] * 100, color=self.S["negative"], alpha=0.7, width=1)
        axes[0].set_ylabel("Return (%)")
        axes[0].set_title("Portfolio Daily Returns")

        # Panel 2: GARCH conditional vol (annualised)
        ann_vol = np.sqrt(cond_vol.values ** 2 * 252) * 100
        axes[1].plot(idx, ann_vol, color=self.S["accent"], lw=1.2, label="GARCH σ (ann.)")
        if garch_params is not None:
            lr_vol = getattr(garch_params, 'unconditional_vol_ann', None)
            if lr_vol is not None:
                lr_vol_pct = lr_vol * 100 if lr_vol < 1 else lr_vol
                axes[1].axhline(lr_vol_pct, color=self.S["warning"], ls="--", lw=1.0,
                               label=f"Long-run σ = {lr_vol_pct:.1f}%")
        axes[1].set_ylabel("Annualised Vol (%)")
        axes[1].set_title("GARCH(1,1) Conditional Volatility")
        axes[1].legend(fontsize=8)

        # Panel 3: Multi-step vol forecast
        if garch_params is not None:
            ab = getattr(garch_params, 'persistence', 0.95)
            var_uc = getattr(garch_params, 'unconditional_variance', cond_vol.values[-1] ** 2)
            fwd = var_uc + (ab ** np.arange(1, forecast_days + 1)) * (cond_vol.values[-1] ** 2 - var_uc)
            fwd_vol = np.sqrt(np.maximum(fwd, 0) * 252) * 100
            fwd_idx = pd.bdate_range(idx[-1], periods=forecast_days + 1)[1:]
            axes[2].plot(fwd_idx, fwd_vol, color=self.S["accent"], lw=1.5, label="Vol forecast")
            if lr_vol is not None:
                axes[2].axhline(lr_vol_pct, color=self.S["warning"], ls="--", lw=1.0,
                               label=f"Long-run mean = {lr_vol_pct:.1f}%")
                axes[2].fill_between(fwd_idx, lr_vol_pct * 0.85, lr_vol_pct * 1.15,
                                    color=self.S["warning"], alpha=0.1)
            axes[2].set_ylabel("Forecast Vol (%)")
            axes[2].set_title(f"{forecast_days}-Day Forward Volatility Forecast (α+β = {ab:.4f})")
            axes[2].legend(fontsize=8)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. Backtest violations
    # ------------------------------------------------------------------

    def plot_backtest(
        self,
        returns: pd.Series,
        var_series: pd.Series,
        violations: pd.Series,
        metrics=None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 6))
            self._apply_theme(fig, ax)
        else:
            fig = ax.figure

        ax.fill_between(returns.index, returns.values, 0, alpha=0.3, color="gray")
        ax.plot(var_series.index, -var_series.values, color=self.S["negative"],
               lw=1.2, label="VaR")

        vret = returns[violations]
        if len(vret):
            ax.scatter(vret.index, vret.values, color="#ff0000", s=35,
                      marker="v", zorder=5, edgecolors="black", linewidth=0.4,
                      label=f"Violations ({violations.sum()})")

        rate = violations.sum() / len(violations) if len(violations) else 0
        info = f"Violation rate: {rate:.2%}"
        if metrics is not None:
            info += (f" | Kupiec p={metrics.kupiec_pvalue:.3f} {'✓' if metrics.kupiec_pass else '✗'}"
                    f" | CC p={metrics.cc_pvalue:.3f} {'✓' if metrics.cc_pass else '✗'}"
                    f" | {metrics.basel_zone}")
            tl_col = metrics.tl_colour_hex if hasattr(metrics, 'tl_colour_hex') else ""
            if tl_col and self.dark:
                ax.set_title(info, color=tl_col, fontsize=10, fontweight="bold")
            else:
                ax.set_title(info, fontsize=10)
        else:
            ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=10, va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend(loc="lower left")
        return fig

    # ------------------------------------------------------------------
    # 5. Monte Carlo paths + terminal distribution
    # ------------------------------------------------------------------

    def plot_mc_paths(
        self,
        paths: np.ndarray,
        var_pct: float,
        confidence: float = 0.95,
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        self._apply_theme(fig, axes)

        # Left: simulated paths
        n_show = min(300, paths.shape[0])
        ax = axes[0]
        for i in range(n_show):
            final = paths[i, -1]
            c = self.S["positive"] if final > 0 else self.S["negative"]
            ax.plot(paths[i] * 100, color=c, alpha=0.04, lw=0.5)

        pctiles = {5: self.S["negative"], 25: self.S["warning"],
                   50: self.S["positive"], 75: self.S["warning"], 95: self.S["negative"]}
        for p, c in pctiles.items():
            line = np.percentile(paths, p, axis=0) * 100
            ax.plot(line, color=c, lw=1.4, label=f"P{p}")

        ax.axhline(0, color=self.S["text_color"], lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_title(f"MC Simulation Paths (n={paths.shape[0]:,})")
        ax.legend(fontsize=8)

        # Right: terminal distribution
        terminal = paths[:, -1] * 100
        ax2 = axes[1]
        ax2.hist(terminal, bins=100, density=True, color=self.S["accent"],
                alpha=0.5, edgecolor="none")
        var_line = var_pct * 100
        ax2.axvline(var_line, color=self.S["negative"], lw=2,
                   label=f"VaR ({confidence:.0%}) = {var_line:.2f}%")
        ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1],
                         terminal.min(), var_line,
                         color=self.S["negative"], alpha=0.15, label="Loss tail")
        ax2.set_xlabel("Terminal Return (%)")
        ax2.set_ylabel("Density")
        ax2.set_title("Terminal Return Distribution")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 6. Drawdown
    # ------------------------------------------------------------------

    def plot_drawdown(self, drawdown: pd.Series, ax: Optional[plt.Axes] = None) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 5))
            self._apply_theme(fig, ax)
        else:
            fig = ax.figure

        ax.fill_between(drawdown.index, drawdown.values, 0,
                       alpha=0.5, color=self.S["negative"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.set_title("Drawdown History")
        ax.axhline(0, color="black", lw=0.5)
        return fig

    # ------------------------------------------------------------------
    # 7. Correlation heatmap
    # ------------------------------------------------------------------

    def plot_corr_heatmap(self, corr: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Figure:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
            self._apply_theme(fig, ax)
        else:
            fig = ax.figure

        n = len(corr)
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                col = "white" if abs(v) > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=9, color=col, fontweight="bold")
        ax.set_title("Asset Return Correlation Matrix")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 8. Full dashboard (7-panel)
    # ------------------------------------------------------------------

    def dashboard(
        self,
        portfolio,
        var_results: Dict[str, float],
        cond_vol: Optional[pd.Series] = None,
        drawdown: Optional[pd.Series] = None,
        garch_params=None,
        garch_residuals: Optional[np.ndarray] = None,
        backtest_results: Optional[Dict] = None,
    ) -> plt.Figure:
        fig = plt.figure(figsize=(18, 12))
        if self.dark:
            fig.patch.set_facecolor(self.S["fig_facecolor"])
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
        colors = self.S["colors"]

        r = portfolio.get_portfolio_returns()

        # (0,0-1) Return distribution
        ax1 = fig.add_subplot(gs[0, :2])
        if self.dark:
            ax1.set_facecolor(self.S["ax_facecolor"])
        cum = (np.exp(np.cumsum(r.values)) - 1) * 100
        ax1.plot(r.index, cum, color=self.S["accent"], lw=1.5)
        ax1.fill_between(r.index, 0, cum, where=np.array(cum) >= 0,
                        color=self.S["positive"], alpha=0.2)
        ax1.fill_between(r.index, 0, cum, where=np.array(cum) < 0,
                        color=self.S["negative"], alpha=0.2)
        ax1.set_title("Cumulative Portfolio Return", fontweight="bold", fontsize=9)
        ax1.set_ylabel("%", fontsize=8)

        # (0,2) VaR bar chart
        ax2 = fig.add_subplot(gs[0, 2])
        if self.dark:
            ax2.set_facecolor(self.S["ax_facecolor"])
        names = list(var_results.keys())
        vals = [abs(v) * 100 for v in var_results.values()]
        ax2.barh(names, vals, color=colors[:len(names)], alpha=0.85, edgecolor="none")
        ax2.set_title("VaR Estimates (95%)", fontweight="bold", fontsize=9)

        # (1,0-1) GARCH vol or rolling Sharpe
        ax3 = fig.add_subplot(gs[1, :2])
        if self.dark:
            ax3.set_facecolor(self.S["ax_facecolor"])
        if cond_vol is not None and garch_params is not None:
            ann_vol = np.sqrt(cond_vol.values ** 2 * 252) * 100
            idx_g = r.index[-len(cond_vol):]
            ax3.plot(idx_g, ann_vol, color=self.S["accent"], lw=1.0)
            lr = getattr(garch_params, 'unconditional_vol_ann', None)
            if lr is not None:
                lr_pct = lr * 100 if lr < 1 else lr
                ax3.axhline(lr_pct, color=self.S["warning"], ls="--", lw=1.0)
            ax3.set_title("GARCH Conditional Volatility", fontweight="bold", fontsize=9)
        else:
            roll = portfolio.get_rolling_stats(window=252)
            ax3.plot(roll.index, roll["sharpe"].dropna(), color=self.S["accent"], lw=1)
            ax3.axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
            ax3.set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold", fontsize=9)
        ax3.set_ylabel("", fontsize=8)

        # (1,2) Backtest table or weights
        ax4 = fig.add_subplot(gs[1, 2])
        if self.dark:
            ax4.set_facecolor(self.S["ax_facecolor"])
        ax4.axis("off")
        if backtest_results:
            rows = []
            for m, br in backtest_results.items():
                rows.append([
                    m[:18], str(br.n_violations), f"{br.kupiec_pvalue:.3f}",
                    "✓" if br.kupiec_pass else "✗", br.basel_zone,
                ])
            table = ax4.table(cellText=rows, colLabels=["Model", "Viol.", "Kup.p", "Pass", "Basel"],
                            loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            for key, cell in table.get_celld().items():
                cell.set_facecolor(self.S["ax_facecolor"])
                cell.set_edgecolor(self.S["grid_color"])
                cell.set_text_props(color=self.S["text_color"])
            ax4.set_title("Backtesting Summary", fontweight="bold", fontsize=9, pad=10)
        else:
            w = portfolio.get_weights()
            pie_colors = plt.cm.Set3(np.linspace(0, 1, len(w)))
            txt_col = self.S["text_color"] if self.dark else "black"
            _, texts, autotexts = ax4.pie(
                w.values, labels=w.index, autopct="%1.1f%%",
                colors=pie_colors, startangle=90,
                textprops={"color": txt_col, "fontsize": 8})
            for t in texts:
                t.set_color(txt_col)
            for t in autotexts:
                t.set_color("black")
                t.set_fontsize(7)
            ax4.set_title("Portfolio Allocation", fontweight="bold", fontsize=9)

        # (2,0) Residual distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if self.dark:
            ax5.set_facecolor(self.S["ax_facecolor"])
        if garch_residuals is not None:
            z = garch_residuals
            ax5.hist(z, bins=60, density=True, color=self.S["accent"], alpha=0.5, edgecolor="none")
            xr = np.linspace(z.min(), z.max(), 400)
            ax5.plot(xr, stats.norm.pdf(xr), color=self.S["warning"], lw=1.2, ls="--", label="N(0,1)")
            ax5.set_title("GARCH Std. Residuals", fontweight="bold", fontsize=9)
            ax5.legend(fontsize=7)
        else:
            corr = portfolio.get_component_returns().corr()
            im = ax5.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax5, shrink=0.7)
            ax5.set_title("Correlations", fontweight="bold", fontsize=9)

        # (2,1) Drawdown or ACF
        ax6 = fig.add_subplot(gs[2, 1])
        if self.dark:
            ax6.set_facecolor(self.S["ax_facecolor"])
        if drawdown is not None:
            ax6.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color=self.S["negative"])
            ax6.set_title("Drawdown History", fontweight="bold", fontsize=9)
        elif garch_residuals is not None:
            z2 = garch_residuals ** 2
            lags = 20
            acf = [np.corrcoef(z2[:-k], z2[k:])[0, 1] for k in range(1, lags + 1)]
            ci = 1.96 / np.sqrt(len(z2))
            ax6.bar(range(1, lags + 1), acf, color=self.S["accent"], alpha=0.7, width=0.6)
            ax6.axhline(ci, color=self.S["warning"], ls="--", lw=0.8)
            ax6.axhline(-ci, color=self.S["warning"], ls="--", lw=0.8)
            ax6.set_title("ACF of Squared Residuals", fontweight="bold", fontsize=9)
        ax6.set_xlabel("" if drawdown is not None else "Lag", fontsize=8)

        # (2,2) GARCH parameters or risk contribution
        ax7 = fig.add_subplot(gs[2, 2])
        if self.dark:
            ax7.set_facecolor(self.S["ax_facecolor"])
        ax7.axis("off")
        if garch_params is not None:
            p = garch_params
            lines = [
                ("ω (omega)",     f"{p.omega:.2e}"),
                ("α (alpha)",     f"{p.alpha:.4f}"),
                ("β (beta)",      f"{p.beta:.4f}"),
                ("α + β",         f"{p.persistence:.4f}"),
                ("Long-run σ",    f"{np.sqrt(p.unconditional_variance):.4f}"),
                ("Half-life",     f"{getattr(p, 'half_life', 'N/A'):.1f}d"),
            ]
            for row_i, (k, v) in enumerate(lines):
                ax7.text(0.05, 0.92 - row_i * 0.15, k, transform=ax7.transAxes,
                        color=self.S["warning"] if self.dark else "darkorange", fontsize=8)
                ax7.text(0.65, 0.92 - row_i * 0.15, v, transform=ax7.transAxes,
                        color=self.S["text_color"] if self.dark else "black", fontsize=8)
            ax7.set_title("GARCH Parameters", fontweight="bold", fontsize=9)
        else:
            rc = portfolio.get_risk_contribution()
            ax7.barh(rc.index, rc["pct_contrib"] * 100, color=self.S["colors"][:len(rc)], alpha=0.8)
            ax7.set_title("Risk Contribution (%)", fontweight="bold", fontsize=9)

        fig.suptitle(f"Risk Dashboard — {portfolio.name}", fontsize=16,
                    fontweight="bold", y=1.01,
                    color=self.S["text_color"] if self.dark else "black")

        # Apply dark theme to all dashboard axes
        if self.dark:
            all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
            self._apply_theme(fig, all_axes)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        return fig