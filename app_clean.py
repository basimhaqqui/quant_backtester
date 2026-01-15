"""
Clean, Minimal Streamlit Dashboard for the Quant Backtester.
A modern, distraction-free interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from backtester import Backtester
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from risk_management import RiskManager

# Page config - clean minimal look
st.set_page_config(
    page_title="Backtester",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean design
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Clean fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #666;
    }

    /* Button styling */
    .stButton > button {
        background: #000;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        width: 100%;
    }

    .stButton > button:hover {
        background: #333;
        color: #fff;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 8px;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: #000;
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #eee;
    }

    /* Cards */
    .metric-card {
        background: #fafafa;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000;
        margin: 0;
    }

    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        margin-top: 0.5rem;
    }

    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    fetcher = DataFetcher()
    return fetcher.fetch_multiple_tickers(tickers, start, end, show_progress=False)


def create_chart(equity, benchmark=None):
    """Create minimal equity chart."""
    fig = go.Figure()

    # Main equity line
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#000', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,0,0,0.03)'
    ))

    # Benchmark if available
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='SPY',
            line=dict(color='#999', width=1, dash='dot')
        ))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=20, b=0),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            showline=False,
            zeroline=False,
            tickprefix='$',
            tickformat=',.0f'
        ),
        hovermode='x unified'
    )

    return fig


def metric_card(value, label, is_pct=False, is_money=False, good_positive=True):
    """Render a custom metric card."""
    if is_pct:
        display = f"{value*100:.1f}%"
        css_class = "positive" if (value > 0) == good_positive else "negative"
    elif is_money:
        display = f"${value:,.0f}"
        css_class = ""
    else:
        display = f"{value:.2f}"
        css_class = "positive" if value > 1 else ""

    st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {css_class}">{display}</p>
            <p class="metric-label">{label}</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("# ◐ Backtester")
    st.markdown("Test trading strategies on historical data")

    st.markdown("---")

    # Controls row
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])

    with col1:
        strategy = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion"],
            label_visibility="collapsed"
        )

    with col2:
        tickers = st.multiselect(
            "Assets",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "SPY", "QQQ"],
            default=["AAPL", "MSFT", "GOOGL", "META", "NVDA", "SPY"],
            label_visibility="collapsed"
        )

    with col3:
        years = st.selectbox("Period", ["1Y", "2Y", "3Y", "5Y"], index=2, label_visibility="collapsed")
        year_map = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5}

    with col4:
        capital = st.selectbox("Capital", ["$50K", "$100K", "$500K"], index=1, label_visibility="collapsed")
        capital_map = {"$50K": 50000, "$100K": 100000, "$500K": 500000}

    with col5:
        run = st.button("Run", type="primary")

    st.markdown("---")

    # Run backtest
    if run and tickers:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * year_map[years])

        with st.spinner(""):
            # Load data
            prices = load_data(
                tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            if prices.empty:
                st.error("No data available")
                return

            # Create strategy
            if strategy == "Momentum":
                strat = MomentumStrategy(lookback_period=20, top_n=3, holding_period=5)
            else:
                strat = MeanReversionStrategy(lookback_period=20, entry_zscore=-2.0)

            # Run backtest
            bt = Backtester(
                initial_capital=capital_map[capital],
                risk_manager=RiskManager(stop_loss_pct=0.05, max_position_size=0.25)
            )
            result = bt.run(strat, prices, show_progress=False)

        # Metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            metric_card(result.metrics.total_return, "Total Return", is_pct=True)
        with col2:
            metric_card(result.metrics.cagr, "CAGR", is_pct=True)
        with col3:
            metric_card(result.metrics.sharpe_ratio, "Sharpe")
        with col4:
            metric_card(result.metrics.max_drawdown, "Max Drawdown", is_pct=True, good_positive=False)
        with col5:
            metric_card(result.metrics.win_rate, "Win Rate", is_pct=True)
        with col6:
            metric_card(result.final_capital, "Final Value", is_money=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart
        chart = create_chart(result.equity_curve, result.benchmark_equity)
        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})

        # Trade summary
        st.markdown("---")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Summary")
            st.markdown(f"""
            - **{result.metrics.total_trades}** trades executed
            - **{result.metrics.win_rate*100:.0f}%** win rate
            - **{result.metrics.profit_factor:.1f}x** profit factor
            - **{abs(result.metrics.max_drawdown)*100:.1f}%** max drawdown
            - **{result.metrics.volatility*100:.1f}%** annual volatility
            """)

        with col2:
            if not result.trades_df.empty:
                st.markdown("### Recent Trades")
                trades = result.trades_df[["ticker", "direction", "pnl", "pnl_pct"]].tail(8).copy()
                trades["pnl"] = trades["pnl"].apply(lambda x: f"${x:,.0f}")
                trades["pnl_pct"] = trades["pnl_pct"].apply(lambda x: f"{x*100:.1f}%")
                trades.columns = ["Asset", "Side", "P&L", "Return"]
                st.dataframe(trades, use_container_width=True, hide_index=True)

    elif not tickers:
        st.info("Select assets to begin")

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem; color: #888;">
            <p style="font-size: 3rem; margin-bottom: 1rem;">◐</p>
            <p>Select your parameters above and click <strong>Run</strong></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
