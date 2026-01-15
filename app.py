"""
Streamlit Dashboard for the Quantitative Trading Strategy Backtester.
Interactive web interface for strategy selection, backtesting, and analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Import local modules
from data_fetcher import DataFetcher
from backtester import Backtester
from strategies.momentum import MomentumStrategy, DualMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pairs_trading import PairsTradingStrategy
from risk_management import RiskManager
from monte_carlo import MonteCarloSimulator
from report_generator import ReportGenerator
from config import (
    STOCK_TICKERS, ETF_TICKERS, CRYPTO_TICKERS, ALL_TICKERS,
    MOMENTUM_PARAMS, MEAN_REVERSION_PARAMS, PAIRS_TRADING_PARAMS,
    RISK_PARAMS, BACKTEST_PARAMS
)

# Page configuration
st.set_page_config(
    page_title="Quant Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    """Fetch and cache price data."""
    fetcher = DataFetcher()
    return fetcher.fetch_multiple_tickers(
        tickers,
        start_date=start_date,
        end_date=end_date,
        show_progress=False
    )


def create_equity_chart(result):
    """Create interactive equity curve chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Value", "Drawdown")
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode="lines",
            name=result.strategy_name,
            line=dict(color="#2E86AB", width=2)
        ),
        row=1, col=1
    )

    # Benchmark
    if result.benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=result.benchmark_equity.index,
                y=result.benchmark_equity.values,
                mode="lines",
                name="Benchmark (SPY)",
                line=dict(color="#A23B72", width=1.5, dash="dash")
            ),
            row=1, col=1
        )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=result.drawdown_series.index,
            y=result.drawdown_series.values * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#F18F01", width=1),
            fillcolor="rgba(241, 143, 1, 0.3)"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)

    return fig


def create_returns_distribution(result):
    """Create returns distribution histogram."""
    returns = result.returns.dropna() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Daily Returns",
        marker_color="#2E86AB",
        opacity=0.7
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=returns.mean(), line_color="green", annotation_text=f"Mean: {returns.mean():.2f}%")

    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        height=350
    )

    return fig


def create_trade_chart(result):
    """Create trade P&L chart."""
    if result.trades_df.empty:
        return None

    pnl = result.trades_df["pnl"]
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in pnl]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(pnl))),
        y=pnl,
        marker_color=colors,
        name="Trade P&L"
    ))

    fig.update_layout(
        title="Individual Trade P&L",
        xaxis_title="Trade #",
        yaxis_title="P&L ($)",
        height=300
    )

    return fig


def main():
    """Main Streamlit application."""
    st.title("📈 Quantitative Trading Strategy Backtester")

    # Sidebar - Configuration
    st.sidebar.header("Configuration")

    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Select Strategy",
        ["Momentum", "Mean Reversion", "Pairs Trading", "Dual Momentum"]
    )

    # Asset selection
    st.sidebar.subheader("Assets")
    asset_class = st.sidebar.multiselect(
        "Asset Classes",
        ["Stocks", "ETFs", "Crypto"],
        default=["Stocks", "ETFs"]
    )

    # Build ticker list based on selection
    available_tickers = []
    if "Stocks" in asset_class:
        available_tickers.extend(STOCK_TICKERS)
    if "ETFs" in asset_class:
        available_tickers.extend(ETF_TICKERS)
    if "Crypto" in asset_class:
        available_tickers.extend(CRYPTO_TICKERS)

    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        available_tickers,
        default=available_tickers[:10] if available_tickers else []
    )

    # Date range
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)

    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*5)

    start_date = col1.date_input("Start", default_start)
    end_date = col2.date_input("End", default_end)

    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")

    if strategy_type == "Momentum":
        lookback = st.sidebar.slider("Lookback Period (days)", 5, 60, MOMENTUM_PARAMS["lookback_period"])
        holding = st.sidebar.slider("Holding Period (days)", 1, 20, MOMENTUM_PARAMS["holding_period"])
        top_n = st.sidebar.slider("Top N Assets", 1, 10, MOMENTUM_PARAMS["top_n"])
        long_short = st.sidebar.checkbox("Long-Short", value=False)

        strategy_params = {
            "lookback_period": lookback,
            "holding_period": holding,
            "top_n": top_n,
            "long_short": long_short
        }

    elif strategy_type == "Mean Reversion":
        lookback = st.sidebar.slider("Lookback Period (days)", 10, 60, MEAN_REVERSION_PARAMS["lookback_period"])
        entry_z = st.sidebar.slider("Entry Z-Score", -4.0, -1.0, MEAN_REVERSION_PARAMS["entry_zscore"])
        exit_z = st.sidebar.slider("Exit Z-Score", -1.0, 1.0, MEAN_REVERSION_PARAMS["exit_zscore"])
        use_rsi = st.sidebar.checkbox("Use RSI Filter", value=True)

        strategy_params = {
            "lookback_period": lookback,
            "entry_zscore": entry_z,
            "exit_zscore": exit_z,
            "use_rsi_filter": use_rsi
        }

    elif strategy_type == "Pairs Trading":
        lookback = st.sidebar.slider("Lookback Period (days)", 30, 120, PAIRS_TRADING_PARAMS["lookback_period"])
        entry_z = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, PAIRS_TRADING_PARAMS["entry_zscore"])
        exit_z = st.sidebar.slider("Exit Z-Score", 0.0, 1.5, PAIRS_TRADING_PARAMS["exit_zscore"])
        corr_thresh = st.sidebar.slider("Correlation Threshold", 0.5, 0.95, PAIRS_TRADING_PARAMS["correlation_threshold"])

        strategy_params = {
            "lookback_period": lookback,
            "entry_zscore": entry_z,
            "exit_zscore": exit_z,
            "correlation_threshold": corr_thresh
        }

    else:  # Dual Momentum
        lookback = st.sidebar.slider("Lookback Period (days)", 60, 300, 252)
        top_n = st.sidebar.slider("Top N Assets", 1, 5, 3)

        strategy_params = {
            "lookback_period": lookback,
            "top_n": top_n
        }

    # Risk management
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1.0, 20.0, RISK_PARAMS["stop_loss_pct"] * 100) / 100
    take_profit = st.sidebar.slider("Take Profit (%)", 5.0, 50.0, RISK_PARAMS["take_profit_pct"] * 100) / 100
    max_position = st.sidebar.slider("Max Position Size (%)", 5.0, 50.0, RISK_PARAMS["max_position_size"] * 100) / 100

    # Backtest settings
    st.sidebar.subheader("Backtest Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=BACKTEST_PARAMS["initial_capital"],
        step=10000
    )
    commission = st.sidebar.slider("Commission (bps)", 0, 50, 10) / 10000
    slippage = st.sidebar.slider("Slippage (bps)", 0, 20, 5) / 10000

    # Run backtest button
    run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")

    # Monte Carlo option
    run_monte_carlo = st.sidebar.checkbox("Run Monte Carlo Simulation")
    if run_monte_carlo:
        mc_simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 500)

    # Main content area
    if not selected_tickers:
        st.warning("Please select at least one ticker to backtest.")
        return

    if run_backtest:
        with st.spinner("Fetching market data..."):
            prices = fetch_data(
                selected_tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

        if prices.empty:
            st.error("No data available for selected tickers and date range.")
            return

        st.success(f"Loaded data for {len(prices.columns)} assets, {len(prices)} trading days")

        # Create strategy
        with st.spinner(f"Running {strategy_type} backtest..."):
            if strategy_type == "Momentum":
                strategy = MomentumStrategy(**strategy_params)
            elif strategy_type == "Mean Reversion":
                strategy = MeanReversionStrategy(**strategy_params)
            elif strategy_type == "Pairs Trading":
                strategy = PairsTradingStrategy(**strategy_params)
            else:
                strategy = DualMomentumStrategy(**strategy_params)

            # Create risk manager and backtester
            risk_manager = RiskManager(
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                max_position_size=max_position
            )

            backtester = Backtester(
                initial_capital=initial_capital,
                commission_pct=commission,
                slippage_pct=slippage,
                risk_manager=risk_manager
            )

            # Run backtest
            result = backtester.run(strategy, prices, show_progress=False)

        # Display results
        st.header("📊 Backtest Results")

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_return = result.metrics.total_return * 100
            st.metric(
                "Total Return",
                f"{total_return:.1f}%",
                delta=f"{total_return:.1f}%" if total_return > 0 else None
            )

        with col2:
            st.metric("CAGR", f"{result.metrics.cagr*100:.1f}%")

        with col3:
            st.metric("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}")

        with col4:
            st.metric("Max Drawdown", f"{result.metrics.max_drawdown*100:.1f}%")

        with col5:
            st.metric("Win Rate", f"{result.metrics.win_rate*100:.1f}%")

        # Second row of metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Sortino Ratio", f"{result.metrics.sortino_ratio:.2f}")

        with col2:
            st.metric("Calmar Ratio", f"{result.metrics.calmar_ratio:.2f}")

        with col3:
            st.metric("Profit Factor", f"{result.metrics.profit_factor:.2f}")

        with col4:
            st.metric("Total Trades", f"{result.metrics.total_trades}")

        with col5:
            st.metric("Volatility", f"{result.metrics.volatility*100:.1f}%")

        # Charts
        st.subheader("Performance Charts")

        # Equity curve
        equity_chart = create_equity_chart(result)
        st.plotly_chart(equity_chart, use_container_width=True)

        # Two column layout for additional charts
        col1, col2 = st.columns(2)

        with col1:
            returns_chart = create_returns_distribution(result)
            st.plotly_chart(returns_chart, use_container_width=True)

        with col2:
            trade_chart = create_trade_chart(result)
            if trade_chart:
                st.plotly_chart(trade_chart, use_container_width=True)
            else:
                st.info("No trades to display")

        # Trade details
        if not result.trades_df.empty:
            st.subheader("Trade History")

            # Summary stats
            trades_df = result.trades_df.copy()

            col1, col2, col3 = st.columns(3)
            with col1:
                winning_trades = len(trades_df[trades_df["pnl"] > 0])
                st.metric("Winning Trades", winning_trades)
            with col2:
                losing_trades = len(trades_df[trades_df["pnl"] < 0])
                st.metric("Losing Trades", losing_trades)
            with col3:
                avg_pnl = trades_df["pnl"].mean()
                st.metric("Avg Trade P&L", f"${avg_pnl:.2f}")

            # Trade table
            display_df = trades_df[["ticker", "entry_date", "exit_date", "direction", "pnl", "pnl_pct", "exit_reason"]].copy()
            display_df["pnl"] = display_df["pnl"].round(2)
            display_df["pnl_pct"] = (display_df["pnl_pct"] * 100).round(2)
            display_df.columns = ["Ticker", "Entry", "Exit", "Direction", "P&L ($)", "P&L (%)", "Exit Reason"]

            st.dataframe(display_df, use_container_width=True, height=300)

        # Monte Carlo analysis
        if run_monte_carlo:
            st.subheader("📈 Monte Carlo Simulation")

            with st.spinner(f"Running {mc_simulations} Monte Carlo simulations..."):
                mc = MonteCarloSimulator(n_simulations=mc_simulations, random_seed=42)
                mc_result = mc.run_simulation(strategy, prices, show_progress=False)

            # Display MC results
            col1, col2, col3, col4 = st.columns(4)

            ci = mc_result.confidence_intervals

            with col1:
                st.metric(
                    "Return (50th %)",
                    f"{ci['return']['50%']*100:.1f}%",
                    delta=f"5th: {ci['return']['5%']*100:.1f}%"
                )

            with col2:
                st.metric(
                    "Sharpe (50th %)",
                    f"{ci['sharpe']['50%']:.2f}",
                    delta=f"5th: {ci['sharpe']['5%']:.2f}"
                )

            with col3:
                prob_positive = (mc_result.return_distribution > 0).mean() * 100
                st.metric("P(Return > 0)", f"{prob_positive:.1f}%")

            with col4:
                prob_sharpe = (mc_result.sharpe_distribution > 0).mean() * 100
                st.metric("P(Sharpe > 0)", f"{prob_sharpe:.1f}%")

            # Distribution charts
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=mc_result.return_distribution * 100,
                    nbinsx=30,
                    marker_color="#2E86AB",
                    opacity=0.7
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="Return Distribution", xaxis_title="Return (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=mc_result.sharpe_distribution,
                    nbinsx=30,
                    marker_color="#2E86AB",
                    opacity=0.7
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.add_vline(x=1, line_dash="dot", line_color="green")
                fig.update_layout(title="Sharpe Distribution", xaxis_title="Sharpe Ratio", height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Export options
        st.subheader("📥 Export")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating report..."):
                    generator = ReportGenerator()
                    if run_monte_carlo:
                        report_path = generator.generate_report(result, mc_result)
                    else:
                        report_path = generator.generate_report(result)
                    st.success(f"Report saved to: {report_path}")

        with col2:
            # Export trades to CSV
            if not result.trades_df.empty:
                csv = result.trades_df.to_csv(index=False)
                st.download_button(
                    "Download Trades CSV",
                    csv,
                    file_name=f"{strategy_type}_trades.csv",
                    mime="text/csv"
                )

        # Store result in session state
        st.session_state["last_result"] = result

    else:
        # Show instructions if no backtest run yet
        st.info("""
        **Welcome to the Quantitative Trading Strategy Backtester!**

        To get started:
        1. Select a trading strategy from the sidebar
        2. Choose the assets you want to trade
        3. Set the date range for your backtest
        4. Adjust strategy and risk parameters
        5. Click "Run Backtest" to see results

        **Available Strategies:**
        - **Momentum**: Buy assets with highest recent returns
        - **Mean Reversion**: Buy oversold assets expecting reversion to mean
        - **Pairs Trading**: Trade the spread between correlated assets
        - **Dual Momentum**: Combines absolute and relative momentum
        """)


if __name__ == "__main__":
    main()
