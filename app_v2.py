"""
Polished Dashboard v2 - Dark theme with all features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from backtester import Backtester
from strategies.momentum import MomentumStrategy, DualMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pairs_trading import PairsTradingStrategy
from risk_management import RiskManager
from monte_carlo import MonteCarloSimulator
from report_generator import ReportGenerator
from config import STOCK_TICKERS, ETF_TICKERS, CRYPTO_TICKERS

st.set_page_config(
    page_title="Quant Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #e0e0e0;
    }

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-container:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value.green {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value.red {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a5a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #666;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #667eea;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1.5rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: #888;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 2rem 0;
    }

    .stCheckbox label {
        color: #ccc;
    }

    .section-title {
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    fetcher = DataFetcher()
    return fetcher.fetch_multiple_tickers(tickers, start, end, show_progress=False)


def create_equity_chart(equity, benchmark=None, drawdown=None):
    """Create polished equity chart with dark theme."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#667eea', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ), row=1, col=1)

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='SPY Benchmark',
            line=dict(color='#888', width=1.5, dash='dot')
        ), row=1, col=1)

    if drawdown is not None:
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#ff6b6b', width=1),
            fillcolor='rgba(255, 107, 107, 0.2)'
        ), row=2, col=1)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#888', size=11),
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(showgrid=False, showline=False, color='#666'),
        xaxis2=dict(showgrid=False, showline=False, color='#666'),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            showline=False,
            tickprefix='$',
            tickformat=',.0f',
            color='#888'
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            showline=False,
            ticksuffix='%',
            color='#888'
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1a1a2e', bordercolor='#667eea', font=dict(color='white'))
    )

    return fig


def create_returns_chart(returns):
    """Create returns distribution chart."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        marker=dict(
            color='rgba(102, 126, 234, 0.6)',
            line=dict(color='#667eea', width=1)
        )
    ))

    fig.add_vline(x=0, line=dict(color='#888', width=1, dash='dash'))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=250,
        xaxis=dict(title='Return %', showgrid=False, color='#888'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#888'),
        showlegend=False
    )

    return fig


def create_monte_carlo_chart(mc_result):
    """Create Monte Carlo distribution charts."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Return Distribution", "Sharpe Distribution"))

    fig.add_trace(go.Histogram(
        x=mc_result.return_distribution * 100,
        nbinsx=30,
        marker=dict(color='rgba(102, 126, 234, 0.6)', line=dict(color='#667eea', width=1)),
        name='Returns'
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=mc_result.sharpe_distribution,
        nbinsx=30,
        marker=dict(color='rgba(118, 75, 162, 0.6)', line=dict(color='#764ba2', width=1)),
        name='Sharpe'
    ), row=1, col=2)

    fig.add_vline(x=0, line=dict(color='#888', width=1, dash='dash'), row=1, col=1)
    fig.add_vline(x=0, line=dict(color='#888', width=1, dash='dash'), row=1, col=2)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
        showlegend=False,
        font=dict(color='#888')
    )

    fig.update_xaxes(showgrid=False, color='#888')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#888')

    return fig


def metric_card(value, label, is_pct=False, is_money=False):
    """Render styled metric card."""
    if is_pct:
        display = f"{value*100:+.1f}%" if value >= 0 else f"{value*100:.1f}%"
        color_class = "green" if value > 0 else "red" if value < 0 else ""
    elif is_money:
        display = f"${value:,.0f}"
        color_class = "green" if value > 0 else ""
    else:
        display = f"{value:.2f}" if isinstance(value, float) else str(value)
        color_class = ""

    st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value {color_class}">{display}</p>
            <p class="metric-label">{label}</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### Strategy")
        strategy = st.selectbox(
            "Type",
            ["Momentum", "Mean Reversion", "Pairs Trading", "Dual Momentum"],
            label_visibility="collapsed"
        )

        st.markdown("### Assets")
        asset_class = st.multiselect(
            "Asset Classes",
            ["Stocks", "ETFs", "Crypto"],
            default=["Stocks", "ETFs"],
            label_visibility="collapsed"
        )

        available_tickers = []
        if "Stocks" in asset_class:
            available_tickers.extend(STOCK_TICKERS)
        if "ETFs" in asset_class:
            available_tickers.extend(ETF_TICKERS)
        if "Crypto" in asset_class:
            available_tickers.extend(CRYPTO_TICKERS)

        tickers = st.multiselect(
            "Tickers",
            available_tickers,
            default=available_tickers[:8] if available_tickers else [],
            label_visibility="collapsed"
        )

        st.markdown("### Strategy Parameters")

        if strategy == "Momentum":
            lookback = st.slider("Lookback (days)", 5, 60, 20)
            holding = st.slider("Holding period (days)", 1, 20, 5)
            top_n = st.slider("Top N assets", 1, 10, 3)
            long_short = st.checkbox("Long-Short", value=False)

        elif strategy == "Mean Reversion":
            lookback = st.slider("Lookback (days)", 10, 60, 20)
            entry_z = st.slider("Entry Z-score", -4.0, -1.0, -2.0)
            exit_z = st.slider("Exit Z-score", -1.0, 1.0, 0.0)
            use_rsi = st.checkbox("Use RSI filter", value=True)

        elif strategy == "Pairs Trading":
            lookback = st.slider("Lookback (days)", 30, 120, 60)
            entry_z = st.slider("Entry Z-score", 1.0, 3.0, 2.0)
            exit_z = st.slider("Exit Z-score", 0.0, 1.5, 0.5)
            corr_thresh = st.slider("Correlation threshold", 0.5, 0.95, 0.7)

        else:  # Dual Momentum
            lookback = st.slider("Lookback (days)", 60, 300, 252)
            top_n = st.slider("Top N assets", 1, 5, 3)

        st.markdown("### Risk Management")
        stop_loss = st.slider("Stop Loss (%)", 1.0, 20.0, 5.0) / 100
        take_profit = st.slider("Take Profit (%)", 5.0, 50.0, 15.0) / 100
        max_position = st.slider("Max Position (%)", 5.0, 50.0, 20.0) / 100

        st.markdown("### Backtest Settings")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", datetime.now() - timedelta(days=365*3))
        with col2:
            end_date = st.date_input("End", datetime.now())

        capital = st.number_input("Initial Capital ($)", 10000, 10000000, 100000, step=10000)
        commission = st.slider("Commission (bps)", 0, 50, 10) / 10000
        slippage = st.slider("Slippage (bps)", 0, 20, 5) / 10000

        st.markdown("### Monte Carlo")
        run_mc = st.checkbox("Run Monte Carlo Simulation", value=False)
        if run_mc:
            mc_sims = st.slider("Simulations", 100, 2000, 500)

        st.markdown("")
        run = st.button("🚀 Run Backtest", type="primary")

    # Main content
    st.markdown('<p class="main-header">Quant Backtester</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Backtest trading strategies with historical market data</p>', unsafe_allow_html=True)

    if run and tickers:
        with st.spinner("Fetching market data..."):
            prices = load_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        if prices.empty:
            st.error("No data available for selected tickers")
            return

        st.success(f"Loaded {len(prices)} trading days for {len(prices.columns)} assets")

        with st.spinner(f"Running {strategy} backtest..."):
            # Create strategy
            if strategy == "Momentum":
                strat = MomentumStrategy(lookback_period=lookback, holding_period=holding, top_n=top_n, long_short=long_short)
            elif strategy == "Mean Reversion":
                strat = MeanReversionStrategy(lookback_period=lookback, entry_zscore=entry_z, exit_zscore=exit_z, use_rsi_filter=use_rsi)
            elif strategy == "Pairs Trading":
                strat = PairsTradingStrategy(lookback_period=lookback, entry_zscore=entry_z, exit_zscore=exit_z, correlation_threshold=corr_thresh)
            else:
                strat = DualMomentumStrategy(lookback_period=lookback, top_n=top_n)

            risk_mgr = RiskManager(stop_loss_pct=stop_loss, take_profit_pct=take_profit, max_position_size=max_position)
            bt = Backtester(initial_capital=capital, commission_pct=commission, slippage_pct=slippage, risk_manager=risk_mgr)
            result = bt.run(strat, prices, show_progress=False)

        # Metrics - Row 1
        cols = st.columns(5)
        with cols[0]:
            metric_card(result.metrics.total_return, "Total Return", is_pct=True)
        with cols[1]:
            metric_card(result.metrics.cagr, "CAGR", is_pct=True)
        with cols[2]:
            metric_card(result.metrics.sharpe_ratio, "Sharpe Ratio")
        with cols[3]:
            metric_card(result.metrics.sortino_ratio, "Sortino Ratio")
        with cols[4]:
            metric_card(result.metrics.max_drawdown, "Max Drawdown", is_pct=True)

        # Metrics - Row 2
        cols = st.columns(5)
        with cols[0]:
            metric_card(result.metrics.win_rate, "Win Rate", is_pct=True)
        with cols[1]:
            metric_card(result.metrics.profit_factor, "Profit Factor")
        with cols[2]:
            metric_card(result.metrics.total_trades, "Total Trades")
        with cols[3]:
            metric_card(result.metrics.volatility, "Volatility", is_pct=True)
        with cols[4]:
            metric_card(result.final_capital, "Final Capital", is_money=True)

        st.markdown("---")

        # Charts
        tab1, tab2, tab3 = st.tabs(["📈 Performance", "📊 Analysis", "📋 Trades"])

        with tab1:
            chart = create_equity_chart(result.equity_curve, result.benchmark_equity, result.drawdown_series)
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Returns Distribution**")
                ret_chart = create_returns_chart(result.returns.dropna())
                st.plotly_chart(ret_chart, use_container_width=True, config={'displayModeBar': False})

            with col2:
                st.markdown("**Risk Metrics**")
                stats_data = {
                    "Metric": ["Calmar Ratio", "VaR (95%)", "CVaR (95%)", "Skewness", "Kurtosis", "Max DD Duration"],
                    "Value": [
                        f"{result.metrics.calmar_ratio:.2f}",
                        f"{result.metrics.var_95*100:.2f}%",
                        f"{result.metrics.cvar_95*100:.2f}%",
                        f"{result.metrics.skewness:.2f}",
                        f"{result.metrics.kurtosis:.2f}",
                        f"{result.metrics.max_drawdown_duration} days"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

            if result.metrics.beta is not None:
                st.markdown("**Benchmark Comparison**")
                bench_data = {
                    "Metric": ["Beta", "Alpha (Ann.)", "Information Ratio"],
                    "Value": [
                        f"{result.metrics.beta:.2f}",
                        f"{result.metrics.alpha*100:.2f}%" if result.metrics.alpha else "N/A",
                        f"{result.metrics.information_ratio:.2f}" if result.metrics.information_ratio else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

        with tab3:
            if not result.trades_df.empty:
                st.markdown("**Trade History**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    winning = len(result.trades_df[result.trades_df["pnl"] > 0])
                    metric_card(winning, "Winning Trades")
                with col2:
                    losing = len(result.trades_df[result.trades_df["pnl"] < 0])
                    metric_card(losing, "Losing Trades")
                with col3:
                    avg_pnl = result.trades_df["pnl"].mean()
                    metric_card(avg_pnl, "Avg Trade P&L", is_money=True)

                trades = result.trades_df[["ticker", "entry_date", "exit_date", "direction", "pnl", "pnl_pct", "exit_reason"]].copy()
                trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d")
                trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.strftime("%Y-%m-%d")
                trades["pnl"] = trades["pnl"].apply(lambda x: f"${x:,.0f}")
                trades["pnl_pct"] = trades["pnl_pct"].apply(lambda x: f"{x*100:+.1f}%")
                trades.columns = ["Asset", "Entry", "Exit", "Side", "P&L", "Return", "Exit Reason"]
                st.dataframe(trades, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No trades executed")

        # Monte Carlo
        if run_mc:
            st.markdown("---")
            st.markdown("### Monte Carlo Simulation")

            with st.spinner(f"Running {mc_sims} simulations..."):
                mc = MonteCarloSimulator(n_simulations=mc_sims, random_seed=42)
                mc_result = mc.run_simulation(strat, prices, show_progress=False)

            col1, col2, col3, col4 = st.columns(4)
            ci = mc_result.confidence_intervals

            with col1:
                median_ret = ci['return']['50%']
                metric_card(median_ret, "Median Return", is_pct=True)
            with col2:
                median_sharpe = ci['sharpe']['50%']
                metric_card(median_sharpe, "Median Sharpe")
            with col3:
                prob_pos = (mc_result.return_distribution > 0).mean()
                metric_card(prob_pos, "P(Return > 0)", is_pct=True)
            with col4:
                prob_sharpe = (mc_result.sharpe_distribution > 0).mean()
                metric_card(prob_sharpe, "P(Sharpe > 0)", is_pct=True)

            mc_chart = create_monte_carlo_chart(mc_result)
            st.plotly_chart(mc_chart, use_container_width=True, config={'displayModeBar': False})

        # Export
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Generating report..."):
                    generator = ReportGenerator()
                    mc_res = mc_result if run_mc else None
                    report_path = generator.generate_report(result, mc_res)
                    st.success(f"Report saved: {report_path}")

        with col2:
            if not result.trades_df.empty:
                csv = result.trades_df.to_csv(index=False)
                st.download_button("📥 Download Trades CSV", csv, file_name=f"{strategy}_trades.csv", mime="text/csv")

    elif not tickers:
        st.warning("👈 Select assets in the sidebar to begin")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem; color: #666;">
            <p style="font-size: 4rem; margin-bottom: 1rem;">📈</p>
            <p style="font-size: 1.2rem;">Configure your strategy in the sidebar and click <strong>Run Backtest</strong></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
