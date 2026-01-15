"""
Visualization Module for the Quantitative Trading Strategy Backtester.
Creates charts and plots for backtest analysis using Matplotlib and Plotly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from backtester import BacktestResult
from monte_carlo import MonteCarloResult
from config import PLOT_SETTINGS

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Creates visualizations for backtest results.

    Supports both Matplotlib (static) and Plotly (interactive) outputs.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (PLOT_SETTINGS["figure_width"], PLOT_SETTINGS["figure_height"]),
        colors: Optional[Dict[str, str]] = None,
        use_plotly: bool = False
    ):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size (width, height)
            colors: Color scheme dictionary
            use_plotly: Use Plotly instead of Matplotlib
        """
        self.figsize = figsize
        self.colors = colors or PLOT_SETTINGS["color_scheme"]
        self.use_plotly = use_plotly

        # Try to set matplotlib style
        try:
            plt.style.use(PLOT_SETTINGS["style"])
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn" in plt.style.available else "ggplot")

    def plot_equity_curve(
        self,
        result: BacktestResult,
        include_benchmark: bool = True,
        log_scale: bool = False,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot equity curve with optional benchmark comparison.

        Args:
            result: BacktestResult object
            include_benchmark: Include benchmark if available
            log_scale: Use logarithmic y-axis
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot strategy equity
        ax.plot(
            result.equity_curve.index,
            result.equity_curve.values,
            label=result.strategy_name,
            color=self.colors["equity"],
            linewidth=2
        )

        # Plot benchmark if available
        if include_benchmark and result.benchmark_equity is not None:
            ax.plot(
                result.benchmark_equity.index,
                result.benchmark_equity.values,
                label="Benchmark (SPY)",
                color=self.colors["benchmark"],
                linewidth=1.5,
                linestyle="--"
            )

        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(title or f"{result.strategy_name} Equity Curve")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # Add final values annotation
        final_value = result.equity_curve.iloc[-1]
        ax.annotate(
            f"${final_value:,.0f}",
            xy=(result.equity_curve.index[-1], final_value),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
            color=self.colors["equity"]
        )

        plt.tight_layout()
        return fig

    def plot_drawdown(
        self,
        result: BacktestResult,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot drawdown chart.

        Args:
            result: BacktestResult object
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot drawdown as filled area
        ax.fill_between(
            result.drawdown_series.index,
            result.drawdown_series.values * 100,
            0,
            color=self.colors["drawdown"],
            alpha=0.7,
            label="Drawdown"
        )

        # Highlight maximum drawdown
        max_dd_idx = result.drawdown_series.idxmin()
        max_dd_val = result.drawdown_series.min()

        ax.scatter(
            [max_dd_idx],
            [max_dd_val * 100],
            color=self.colors["negative"],
            s=100,
            zorder=5,
            label=f"Max DD: {max_dd_val*100:.1f}%"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(title or f"{result.strategy_name} Drawdown")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def plot_returns_distribution(
        self,
        result: BacktestResult,
        bins: int = 50,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot histogram of daily returns.

        Args:
            result: BacktestResult object
            bins: Number of histogram bins
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        returns = result.returns.dropna() * 100  # Convert to percentage

        # Plot histogram
        n, bins_edges, patches = ax.hist(
            returns,
            bins=bins,
            density=True,
            alpha=0.7,
            color=self.colors["equity"],
            edgecolor="white"
        )

        # Color negative returns differently
        for patch, edge in zip(patches, bins_edges[:-1]):
            if edge < 0:
                patch.set_facecolor(self.colors["negative"])

        # Add normal distribution overlay
        from scipy import stats
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k--', linewidth=2, label="Normal")

        # Add statistics
        stats_text = f"Mean: {mu:.2f}%\nStd: {sigma:.2f}%\nSkew: {returns.skew():.2f}\nKurt: {returns.kurtosis():.2f}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Density")
        ax.set_title(title or f"{result.strategy_name} Returns Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_returns_heatmap(
        self,
        result: BacktestResult,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot monthly returns heatmap.

        Args:
            result: BacktestResult object
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        # Calculate monthly returns
        equity = result.equity_curve
        monthly_equity = equity.resample("ME").last()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Create year-month pivot table
        monthly_df = pd.DataFrame({
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values * 100
        })

        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.5)))

        # Create heatmap
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

        # Set ticks
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(np.arange(len(pivot)))
        ax.set_yticklabels(pivot.index)

        # Add values to cells
        for i in range(len(pivot)):
            for j in range(12):
                if j < len(pivot.columns) and not pd.isna(pivot.iloc[i, j]):
                    value = pivot.iloc[i, j]
                    color = "white" if abs(value) > 5 else "black"
                    ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=8)

        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("Monthly Return (%)", rotation=-90, va="bottom")

        ax.set_title(title or f"{result.strategy_name} Monthly Returns (%)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

        plt.tight_layout()
        return fig

    def plot_rolling_metrics(
        self,
        result: BacktestResult,
        window: int = 252,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot rolling performance metrics.

        Args:
            result: BacktestResult object
            window: Rolling window size in days
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), sharex=True)

        returns = result.returns

        # Rolling return
        rolling_return = (1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False) * 100
        axes[0].plot(rolling_return.index, rolling_return.values, color=self.colors["equity"])
        axes[0].axhline(y=0, color="gray", linestyle="--")
        axes[0].set_ylabel(f"{window}d Return (%)")
        axes[0].set_title(title or f"{result.strategy_name} Rolling Metrics ({window} days)")
        axes[0].grid(True, alpha=0.3)

        # Rolling Sharpe
        rolling_sharpe = (
            returns.rolling(window).mean() * 252 /
            (returns.rolling(window).std() * np.sqrt(252))
        )
        axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, color=self.colors["benchmark"])
        axes[1].axhline(y=0, color="gray", linestyle="--")
        axes[1].axhline(y=1, color="green", linestyle=":", alpha=0.5)
        axes[1].set_ylabel("Rolling Sharpe")
        axes[1].grid(True, alpha=0.3)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        axes[2].plot(rolling_vol.index, rolling_vol.values, color=self.colors["drawdown"])
        axes[2].set_ylabel("Rolling Vol (%)")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trade_analysis(
        self,
        result: BacktestResult,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot trade analysis charts.

        Args:
            result: BacktestResult object
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        if result.trades_df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No trades to analyze", ha="center", va="center", fontsize=14)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        trades_df = result.trades_df

        # 1. P&L Distribution
        ax1 = axes[0, 0]
        pnl = trades_df["pnl"]
        colors = [self.colors["positive"] if x > 0 else self.colors["negative"] for x in pnl]
        ax1.bar(range(len(pnl)), pnl, color=colors, alpha=0.7)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.set_xlabel("Trade #")
        ax1.set_ylabel("P&L ($)")
        ax1.set_title("Trade P&L")
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative P&L
        ax2 = axes[0, 1]
        cumulative_pnl = pnl.cumsum()
        ax2.plot(cumulative_pnl.values, color=self.colors["equity"], linewidth=2)
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, alpha=0.3, color=self.colors["equity"])
        ax2.set_xlabel("Trade #")
        ax2.set_ylabel("Cumulative P&L ($)")
        ax2.set_title("Cumulative P&L")
        ax2.grid(True, alpha=0.3)

        # 3. Win/Loss by ticker
        ax3 = axes[1, 0]
        ticker_pnl = trades_df.groupby("ticker")["pnl"].sum().sort_values()
        colors = [self.colors["positive"] if x > 0 else self.colors["negative"] for x in ticker_pnl]
        ax3.barh(ticker_pnl.index, ticker_pnl.values, color=colors, alpha=0.7)
        ax3.axvline(x=0, color="black", linewidth=0.5)
        ax3.set_xlabel("Total P&L ($)")
        ax3.set_ylabel("Ticker")
        ax3.set_title("P&L by Ticker")
        ax3.grid(True, alpha=0.3)

        # 4. Trade duration histogram
        ax4 = axes[1, 1]
        if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
            trades_df["duration"] = (
                pd.to_datetime(trades_df["exit_date"]) -
                pd.to_datetime(trades_df["entry_date"])
            ).dt.days
            ax4.hist(trades_df["duration"], bins=20, color=self.colors["equity"], alpha=0.7, edgecolor="white")
            ax4.set_xlabel("Trade Duration (days)")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Trade Duration Distribution")
            ax4.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()
        return fig

    def plot_monte_carlo_results(
        self,
        mc_result: MonteCarloResult,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot Monte Carlo simulation results.

        Args:
            mc_result: MonteCarloResult object
            title: Custom title

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Equity curve fan chart
        ax1 = axes[0, 0]
        percentiles = sorted(mc_result.percentile_curves.keys())

        # Plot confidence bands
        if len(percentiles) >= 2:
            for i, p in enumerate(percentiles[:-1]):
                ax1.fill_between(
                    mc_result.percentile_curves[p].index,
                    mc_result.percentile_curves[p].values,
                    mc_result.percentile_curves[percentiles[i+1]].values,
                    alpha=0.3,
                    color=self.colors["equity"]
                )

        # Plot median
        if 0.5 in mc_result.percentile_curves:
            ax1.plot(
                mc_result.percentile_curves[0.5].index,
                mc_result.percentile_curves[0.5].values,
                color=self.colors["equity"],
                linewidth=2,
                label="Median"
            )

        # Plot original result
        if mc_result.original_result:
            ax1.plot(
                mc_result.original_result.equity_curve.index,
                mc_result.original_result.equity_curve.values,
                color=self.colors["benchmark"],
                linewidth=2,
                linestyle="--",
                label="Original"
            )

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Monte Carlo Equity Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Return distribution
        ax2 = axes[0, 1]
        ax2.hist(mc_result.return_distribution * 100, bins=30, color=self.colors["equity"],
                alpha=0.7, edgecolor="white", density=True)
        ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax2.axvline(x=np.median(mc_result.return_distribution) * 100,
                   color=self.colors["benchmark"], linestyle="-", linewidth=2, label="Median")
        ax2.set_xlabel("Total Return (%)")
        ax2.set_ylabel("Density")
        ax2.set_title("Return Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Sharpe distribution
        ax3 = axes[1, 0]
        ax3.hist(mc_result.sharpe_distribution, bins=30, color=self.colors["equity"],
                alpha=0.7, edgecolor="white", density=True)
        ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax3.axvline(x=1, color="green", linestyle=":", linewidth=1, label="Sharpe = 1")
        ax3.set_xlabel("Sharpe Ratio")
        ax3.set_ylabel("Density")
        ax3.set_title("Sharpe Ratio Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Max drawdown distribution
        ax4 = axes[1, 1]
        ax4.hist(mc_result.max_drawdown_distribution * 100, bins=30, color=self.colors["drawdown"],
                alpha=0.7, edgecolor="white", density=True)
        ax4.axvline(x=-20, color="red", linestyle="--", linewidth=1, label="-20% threshold")
        ax4.set_xlabel("Max Drawdown (%)")
        ax4.set_ylabel("Density")
        ax4.set_title("Max Drawdown Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()
        return fig

    def create_dashboard_figure(
        self,
        result: BacktestResult
    ) -> Figure:
        """
        Create a comprehensive dashboard figure.

        Args:
            result: BacktestResult object

        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Equity curve (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(result.equity_curve.index, result.equity_curve.values,
                color=self.colors["equity"], linewidth=2, label=result.strategy_name)
        if result.benchmark_equity is not None:
            ax1.plot(result.benchmark_equity.index, result.benchmark_equity.values,
                    color=self.colors["benchmark"], linewidth=1.5, linestyle="--", label="Benchmark")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(f"{result.strategy_name} Performance Dashboard")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Metrics summary (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        metrics_text = [
            f"Total Return: {result.metrics.total_return*100:.1f}%",
            f"CAGR: {result.metrics.cagr*100:.1f}%",
            f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}",
            f"Max Drawdown: {result.metrics.max_drawdown*100:.1f}%",
            f"Win Rate: {result.metrics.win_rate*100:.1f}%",
            f"Profit Factor: {result.metrics.profit_factor:.2f}",
            f"Total Trades: {result.metrics.total_trades}",
        ]
        ax2.text(0.1, 0.9, "\n".join(metrics_text), transform=ax2.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # 3. Drawdown (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.fill_between(result.drawdown_series.index, result.drawdown_series.values * 100, 0,
                        color=self.colors["drawdown"], alpha=0.7)
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_title("Drawdown")
        ax3.grid(True, alpha=0.3)

        # 4. Returns distribution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        returns_pct = result.returns.dropna() * 100
        ax4.hist(returns_pct, bins=30, color=self.colors["equity"], alpha=0.7, edgecolor="white")
        ax4.axvline(x=0, color="black", linestyle="--")
        ax4.set_xlabel("Daily Return (%)")
        ax4.set_title("Returns Distribution")
        ax4.grid(True, alpha=0.3)

        # 5. Cumulative trade P&L (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if not result.trades_df.empty:
            cum_pnl = result.trades_df["pnl"].cumsum()
            ax5.plot(cum_pnl.values, color=self.colors["equity"], linewidth=2)
            ax5.fill_between(range(len(cum_pnl)), cum_pnl.values, alpha=0.3, color=self.colors["equity"])
        ax5.set_xlabel("Trade #")
        ax5.set_ylabel("Cumulative P&L ($)")
        ax5.set_title("Trade Performance")
        ax5.grid(True, alpha=0.3)

        # 6. Rolling Sharpe (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        window = 63  # ~3 months
        rolling_sharpe = (
            result.returns.rolling(window).mean() * 252 /
            (result.returns.rolling(window).std() * np.sqrt(252))
        )
        ax6.plot(rolling_sharpe.index, rolling_sharpe.values, color=self.colors["benchmark"])
        ax6.axhline(y=0, color="gray", linestyle="--")
        ax6.axhline(y=1, color="green", linestyle=":", alpha=0.5, label="Sharpe = 1")
        ax6.set_xlabel("Date")
        ax6.set_ylabel("Rolling Sharpe (63d)")
        ax6.set_title("Rolling Sharpe Ratio")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        return fig


def create_interactive_equity_chart(result: BacktestResult) -> go.Figure:
    """
    Create interactive Plotly equity curve chart.

    Args:
        result: BacktestResult object

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{result.strategy_name} Equity Curve", "Drawdown")
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode="lines",
            name=result.strategy_name,
            line=dict(color=PLOT_SETTINGS["color_scheme"]["equity"], width=2)
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
                name="Benchmark",
                line=dict(color=PLOT_SETTINGS["color_scheme"]["benchmark"], width=1.5, dash="dash")
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
            line=dict(color=PLOT_SETTINGS["color_scheme"]["drawdown"], width=1),
            fillcolor=f"rgba(241, 143, 1, 0.3)"
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{result.strategy_name} Backtest Results",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600
    )

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


if __name__ == "__main__":
    # Test visualizations
    logging.basicConfig(level=logging.INFO)

    from data_fetcher import DataFetcher
    from strategies.momentum import MomentumStrategy
    from backtester import Backtester

    print("Testing Visualizations...")

    # Run a backtest
    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
    prices = fetcher.fetch_multiple_tickers(tickers, start_date="2020-01-01")

    strategy = MomentumStrategy(lookback_period=20, top_n=3)
    backtester = Backtester()
    result = backtester.run(strategy, prices)

    # Create visualizations
    viz = BacktestVisualizer()

    # Equity curve
    fig1 = viz.plot_equity_curve(result)
    fig1.savefig("test_equity_curve.png", dpi=150, bbox_inches="tight")
    print("Saved test_equity_curve.png")

    # Drawdown
    fig2 = viz.plot_drawdown(result)
    fig2.savefig("test_drawdown.png", dpi=150, bbox_inches="tight")
    print("Saved test_drawdown.png")

    # Dashboard
    fig3 = viz.create_dashboard_figure(result)
    fig3.savefig("test_dashboard.png", dpi=150, bbox_inches="tight")
    print("Saved test_dashboard.png")

    plt.close("all")
    print("\nVisualization tests complete!")
