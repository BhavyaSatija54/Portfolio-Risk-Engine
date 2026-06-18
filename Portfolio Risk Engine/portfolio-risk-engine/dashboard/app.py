"""
Streamlit Dashboard — Production Risk Analytics

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from risk_engine import (
    DataFetcher,
    Portfolio,
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    GARCHModel,
    VaRBacktest,
)

st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-radius: 0.5rem; }
    .stTabs [data-baseweb="tab"] { color: #e6edf3; }
    .stTabs [aria-selected="true"] { background-color: #21262d; }
    .stSidebar { background-color: #161b22; }
    .stSidebar [data-testid="stSidebarContent"] { color: #e6edf3; }
    h1, h2, h3, h4, h5, h6 { color: #e6edf3 !important; }
    p, li, td, th { color: #c9d1d9; }
    .stDataFrame { background-color: #161b22; }
    .stButton>button { background-color: #238636; color: white; }
    div[data-testid="stMetricValue"] { color: #58a6ff; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
    .metric-card { background-color: #161b22; padding: 1rem; border-radius: 0.5rem; border: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> dict:
    st.sidebar.title("⚙️ Configuration")

    tickers = st.sidebar.text_input(
        "Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, JPM, NVDA"
    )
    lookback = st.sidebar.slider(
        "Lookback (trading days)", 252, 7560, 7560, step=252,
        help="~252 days/year"
    )
    notional = st.sidebar.number_input(
        "Portfolio Notional ($)", 10_000, 1_000_000_000, 1_000_000, step=100_000
    )
    cl = st.sidebar.select_slider(
        "Confidence", options=[0.90, 0.95, 0.99], value=0.95,
        format_func=lambda x: f"{x:.0%}"
    )
    n_sims = st.sidebar.select_slider(
        "MC Sims", options=[10_000, 50_000, 100_000, 250_000], value=100_000,
        format_func=lambda x: f"{x:,}"
    )
    custom = None
    if st.sidebar.checkbox("Custom Weights"):
        ts = [t.strip().upper() for t in tickers.split(",")]
        w = {}
        for t in ts:
            w[t] = st.sidebar.number_input(f"{t} (%)", 0, 100, int(100 / len(ts))) / 100
        custom = w

    st.sidebar.markdown("---")
    run = st.sidebar.button("🚀 Run Analysis", type="primary")
    return {
        "tickers": tickers, "lookback": lookback, "notional": notional,
        "cl": cl, "n_sims": n_sims, "custom": custom, "run": run,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_prices(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0]
    fig = go.Figure()
    for c in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c))
    fig.update_layout(title="Normalized Price Performance", xaxis_title="Date",
                     yaxis_title="Normalized", height=500, hovermode="x unified")
    return fig


def plot_hist_with_var(returns: pd.Series, var_map: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=150, histnorm="probability density",
                              name="Returns", marker_color="#2E86AB", opacity=0.7))
    cmap = {"Historical": "#C73E1D", "Parametric": "#F18F01",
            "MonteCarlo": "#2E86AB", "GARCH": "#A23B72"}
    for name, v in var_map.items():
        fig.add_vline(x=-v, line_dash="dash", line_color=cmap.get(name, "#555"),
                     annotation_text=name)
    fig.update_layout(title="Return Distribution with VaR", xaxis_title="Daily Return",
                     yaxis_title="Density", height=500, bargap=0.1)
    return fig


def plot_backtest(returns: pd.Series, var_s: pd.Series, viol: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns.values, mode="lines",
                              line=dict(color="gray", width=0.5), name="Returns"))
    fig.add_trace(go.Scatter(x=var_s.index, y=-var_s.values, mode="lines",
                              line=dict(color="#C73E1D", width=1.5), name="VaR"))
    vret = returns[viol]
    if len(vret):
        fig.add_trace(go.Scatter(x=vret.index, y=vret.values, mode="markers",
                                  marker=dict(color="red", size=8, symbol="triangle-down"),
                                  name=f"Violations ({viol.sum()})"))
    fig.update_layout(title="VaR Backtest — Violations", xaxis_title="Date",
                     yaxis_title="Return", height=500, hovermode="x unified")
    return fig


def plot_garch_vol(returns: pd.Series, cond_vol: pd.Series) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=returns.index, y=returns.values, mode="lines",
                              line=dict(color="gray", width=0.5), name="Returns"))
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol.values, mode="lines",
                              line=dict(color="#C73E1D", width=1), name="+1σ"))
    fig.add_trace(go.Scatter(x=cond_vol.index, y=-cond_vol.values, mode="lines",
                              line=dict(color="#C73E1D", width=1, dash="dash"), name="-1σ"))
    fig.update_layout(title="GARCH(1,1) Conditional Volatility", height=500,
                     xaxis_title="Date", yaxis_title="Return / Vol",
                     hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_var_analysis(portfolio, returns_data, cfg):
    pr = portfolio.get_portfolio_returns()
    w = portfolio.get_weights().values
    pv = cfg["notional"]
    cl = cfg["cl"]

    progress = st.progress(0)
    status = st.empty()

    status.text("Historical VaR...")
    h = HistoricalVaR(pr, confidence_level=cl, notional=pv, bootstrap=True, random_state=42).calculate()
    progress.progress(25)

    status.text("Parametric VaR...")
    p = ParametricVaR(pr, confidence_level=cl, notional=pv).calculate()
    progress.progress(50)

    status.text("GARCH estimation...")
    garch = GARCHModel(pr, distribution="normal").fit()
    g_vol = garch.forecast_vol()
    g = ParametricVaR(pr, confidence_level=cl, notional=pv, daily_vol=g_vol).calculate()
    progress.progress(75)

    status.text(f"Monte Carlo ({cfg['n_sims']:,} paths)...")
    m = MonteCarloVaR(pr, confidence_level=cl, notional=pv,
                     n_simulations=cfg["n_sims"], random_state=42,
                     use_cholesky=True, component_returns=returns_data,
                     component_weights=w).calculate()
    progress.progress(100)

    status.empty()
    progress.empty()
    return {"Historical": h, "Parametric": p, "GARCH": g, "MonteCarlo": m}, garch


def main():
    st.markdown('<h1 style="color:#1f77b4;">📊 Portfolio Risk Engine</h1>',
               unsafe_allow_html=True)
    st.caption("Institutional-grade VaR, GARCH, and Backtesting")
    st.divider()

    cfg = sidebar()

    if cfg["run"]:
        with st.spinner("Fetching market data..."):
            try:
                ts = [t.strip().upper() for t in cfg["tickers"].split(",")]
                fetcher = DataFetcher(tickers=ts, lookback_days=cfg["lookback"])
                prices = fetcher.fetch_data()
                ret = fetcher.get_returns()
                port = Portfolio(returns=ret, weights=cfg["custom"], name="Portfolio")
                st.success(f"✅ {len(ts)} assets, {len(prices):,} observations")
            except Exception as exc:
                st.error(f"❌ {exc}")
                return

        t_overview, t_var, t_garch, t_back = st.tabs([
            "📈 Overview", "⚠️ VaR", "📉 GARCH", "🧪 Backtest"
        ])

        with t_overview:
            s = port.get_statistics()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ann. Return", f"{s['annualized_return']:.2%}")
            c2.metric("Ann. Vol", f"{s['annualized_volatility']:.2%}")
            c3.metric("Sharpe", f"{s['sharpe_ratio']:.2f}")
            c4.metric("Max DD", f"{s['max_drawdown']:.2%}")

            c5, c6 = st.columns(2)
            c5.plotly_chart(plot_prices(prices), use_container_width=True)
            w = port.get_weights()
            fig_pie = go.Figure(data=[go.Pie(labels=w.index, values=w.values,
                                            hole=0.4, textinfo="label+percent")])
            fig_pie.update_layout(title="Allocation", height=400)
            c6.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### Risk Contribution (Euler)")
            st.dataframe(port.get_risk_contribution().style.format({
                "weight": "{:.2%}", "marginal_risk": "{:.6f}",
                "risk_contrib": "{:.6f}", "pct_contrib": "{:.2%}"
            }), use_container_width=True)

        with t_var:
            with st.spinner("Running models..."):
                results, garch_model = run_var_analysis(port, ret, cfg)

            cols = st.columns(len(results))
            for i, (name, res) in enumerate(results.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center;padding:1rem;background:#f0f2f6;border-radius:0.5rem;">
                        <h4>{name}</h4>
                        <p style="font-size:1.4rem;font-weight:bold;color:#d62728;">
                            ${res.var_abs:,.0f}
                        </p>
                        <p>ES: ${res.es_abs:,.0f} | {res.var_pct:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.plotly_chart(
                plot_hist_with_var(port.get_portfolio_returns(),
                                  {n: r.var_pct for n, r in results.items()}),
                use_container_width=True,
            )

            df_comp = pd.DataFrame({
                "Model": list(results.keys()),
                "VaR ($)": [f"${r.var_abs:,.0f}" for r in results.values()],
                "ES ($)": [f"${r.es_abs:,.0f}" for r in results.values()],
                "VaR (bps)": [f"{r.var_pct*10000:.1f}" for r in results.values()],
            })
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

        with t_garch:
            gp = garch_model.params()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ω (omega)", f"{gp.omega:.6f}")
            c2.metric("α (alpha)", f"{gp.alpha:.4f}")
            c3.metric("β (beta)", f"{gp.beta:.4f}")
            c4.metric("Persistence", f"{gp.persistence:.4f}")
            c5, c6, c7 = st.columns(3)
            c5.metric("Half-life", f"{gp.half_life:.1f}d")
            c6.metric("AIC", f"{garch_model.aic():.1f}")
            c7.metric("BIC", f"{garch_model.bic():.1f}")

            st.plotly_chart(
                plot_garch_vol(port.get_portfolio_returns().iloc[-500:],
                              garch_model.conditional_vol().iloc[-500:]),
                use_container_width=True,
            )
            st.markdown("#### 5-Day Volatility Forecast")
            st.dataframe(garch_model.forecast(horizon=5).style.format({
                "variance": "{:.8f}", "volatility": "{:.6f}"
            }), use_container_width=True)

        with t_back:
            with st.spinner("Running backtest..."):
                wnd = 252
                pr = port.get_portfolio_returns()
                var_s = pd.Series(index=pr.index[wnd:], dtype=float)
                for i in range(wnd, len(pr)):
                    var_s.iloc[i - wnd] = HistoricalVaR(
                        pr.iloc[i - wnd:i], confidence_level=cfg["cl"]
                    ).calculate().var_pct
                tr = pr[var_s.index]
                bt = VaRBacktest(tr, var_s, confidence_level=cfg["cl"])
                m = bt.run()
                viol = bt.get_violation_series()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Observations", f"{m.n_obs:,}")
            c2.metric("Violations", f"{m.n_violations}")
            c3.metric("Violation Rate", f"{m.violation_rate:.2%}")
            c4.metric("Basel Zone", m.basel_zone.upper())

            tc = st.columns(3)
            ok_k = "🟢" if m.kupiec_pass else "🔴"
            ok_i = "🟢" if m.ind_pass else "🔴"
            with tc[0]:
                st.markdown(f"**Kupiec POF** {ok_k}\n\nLR={m.kupiec_lr:.3f}  p={m.kupiec_pvalue:.4f}")
            with tc[1]:
                st.markdown(f"**Independence** {ok_i}\n\nLR={m.ind_lr:.3f}  p={m.ind_pvalue:.4f}")
            with tc[2]:
                st.markdown(f"**Conditional Cov**\n\nLR={m.cc_lr:.3f}  p={m.cc_pvalue:.4f}")

            st.plotly_chart(plot_backtest(tr, var_s, viol), use_container_width=True)

    else:
        st.info("👈 Configure and click **Run Analysis**.")
        st.markdown("""
        ### Capabilities

        | Model | Methodology |
        |-------|-------------|
        | **Historical VaR** | Empirical quantile, overlapping periods, bootstrap CI |
        | **Parametric VaR** | Normal with standard / EWMA / GARCH volatility |
        | **Monte Carlo VaR** | Cholesky decomposition of covariance matrix |
        | **GARCH(1,1)** | ML estimation, rolling forecasts, diagnostics |
        | **Backtesting** | Kupiec POF, Christoffersen independence, Basel traffic light |
        """)


if __name__ == "__main__":
    main()