"""
Polished Dashboard v2 - Dark theme with gradient accents
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data_fetcher import DataFetcher
from backtester import Backtester
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from risk_management import RiskManager

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

    /* Header */
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

    /* Metric cards */
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
        font-size: 2rem;
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
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Buttons */
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

    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        color: white;
    }

    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tabs */
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

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 2rem 0;
    }

    /* Info boxes */
    .stAlert {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
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

    # Equity curve with gradient fill
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#667eea', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ), row=1, col=1)

    # Benchmark
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='SPY Benchmark',
            line=dict(color='#888', width=1.5, dash='dot')
        ), row=1, col=1)

    # Drawdown
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
        xaxis=dict(
            showgrid=False,
            showline=False,
            color='#666'
        ),
        xaxis2=dict(
            showgrid=False,
            showline=False,
            color='#666'
        ),
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
            color='#888',
            title=''
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='#667eea',
            font=dict(color='white')
        )
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
        ),
        name='Daily Returns'
    ))

    fig.add_vline(x=0, line=dict(color='#888', width=1, dash='dash'))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        xaxis=dict(
            title='Return %',
            showgrid=False,
            color='#888'
        ),
        yaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            color='#888'
        ),
        showlegend=False
    )

    return fig


def metric_card(value, label, is_pct=False, color="purple"):
    """Render styled metric card."""
    if is_pct:
        display = f"{value*100:+.1f}%" if value >= 0 else f"{value*100:.1f}%"
        if value > 0:
            color_class = "green"
        elif value < 0:
            color_class = "red"
        else:
            color_class = ""
    else:
        display = f"{value:.2f}" if isinstance(value, float) else str(value)
        color_class = color

    st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value {color_class}">{display}</p>
            <p class="metric-label">{label}</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## Strategy")
        strategy = st.selectbox(
            "Type",
            ["Momentum", "Mean Reversion"],
            label_visibility="collapsed"
        )

        st.markdown("## Assets")
        tickers = st.multiselect(
            "Select",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "SPY", "QQQ", "IWM", "BTC-USD", "ETH-USD"],
            default=["AAPL", "MSFT", "GOOGL", "META", "NVDA", "SPY"],
            label_visibility="collapsed"
        )

        st.markdown("## Parameters")

        if strategy == "Momentum":
            lookback = st.slider("Lookback (days)", 10, 60, 20)
            top_n = st.slider("Top N assets", 2, 6, 3)
            holding = st.slider("Holding (days)", 3, 20, 5)
        else:
            lookback = st.slider("Lookback (days)", 10, 60, 20)
            entry_z = st.slider("Entry Z-score", -3.0, -1.0, -2.0)

        st.markdown("## Backtest")
        years = st.select_slider("History", options=["1Y", "2Y", "3Y", "5Y"], value="3Y")
        capital = st.select_slider("Capital", options=["$50K", "$100K", "$250K", "$500K"], value="$100K")

        st.markdown("")
        run = st.button("🚀 Run Backtest", type="primary")

    # Main content
    st.markdown('<p class="main-header">Quant Backtester</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Backtest trading strategies with historical market data</p>', unsafe_allow_html=True)

    if run and tickers:
        year_map = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5}
        capital_map = {"$50K": 50000, "$100K": 100000, "$250K": 250000, "$500K": 500000}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * year_map[years])

        with st.spinner("Running backtest..."):
            prices = load_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if prices.empty:
                st.error("No data available")
                return

            if strategy == "Momentum":
                strat = MomentumStrategy(lookback_period=lookback, top_n=top_n, holding_period=holding)
            else:
                strat = MeanReversionStrategy(lookback_period=lookback, entry_zscore=entry_z)

            bt = Backtester(
                initial_capital=capital_map[capital],
                risk_manager=RiskManager(stop_loss_pct=0.05, max_position_size=0.25)
            )
            result = bt.run(strat, prices, show_progress=False)

        # Metrics row
        cols = st.columns(6)
        metrics = [
            (result.metrics.total_return, "Total Return", True),
            (result.metrics.cagr, "CAGR", True),
            (result.metrics.sharpe_ratio, "Sharpe Ratio", False),
            (result.metrics.max_drawdown, "Max Drawdown", True),
            (result.metrics.win_rate, "Win Rate", True),
            (result.metrics.total_trades, "Trades", False),
        ]

        for col, (val, label, is_pct) in zip(cols, metrics):
            with col:
                metric_card(val, label, is_pct)

        st.markdown("---")

        # Charts
        tab1, tab2 = st.tabs(["📈 Performance", "📊 Analysis"])

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
                st.markdown("**Key Statistics**")
                stats_data = {
                    "Metric": ["Volatility", "Sortino Ratio", "Calmar Ratio", "Profit Factor", "Avg Win", "Avg Loss"],
                    "Value": [
                        f"{result.metrics.volatility*100:.1f}%",
                        f"{result.metrics.sortino_ratio:.2f}",
                        f"{result.metrics.calmar_ratio:.2f}",
                        f"{result.metrics.profit_factor:.2f}",
                        f"{result.metrics.avg_win*100:.2f}%",
                        f"{result.metrics.avg_loss*100:.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # Trades table
        if not result.trades_df.empty:
            st.markdown("---")
            st.markdown("**Recent Trades**")
            trades = result.trades_df[["ticker", "entry_date", "exit_date", "direction", "pnl", "pnl_pct"]].tail(10).copy()
            trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d")
            trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.strftime("%Y-%m-%d")
            trades["pnl"] = trades["pnl"].apply(lambda x: f"${x:,.0f}")
            trades["pnl_pct"] = trades["pnl_pct"].apply(lambda x: f"{x*100:+.1f}%")
            trades.columns = ["Asset", "Entry", "Exit", "Side", "P&L", "Return"]
            st.dataframe(trades, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Configure your strategy in the sidebar and click **Run Backtest**")


if __name__ == "__main__":
    main()
