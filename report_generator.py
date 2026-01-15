"""
PDF Report Generator for the Quantitative Trading Strategy Backtester.
Creates professional PDF reports with charts, metrics, and analysis.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import tempfile
import logging

from fpdf import FPDF
import matplotlib.pyplot as plt

from backtester import BacktestResult
from monte_carlo import MonteCarloResult
from visualizations import BacktestVisualizer

logger = logging.getLogger(__name__)


class BacktestPDF(FPDF):
    """Custom PDF class with header and footer."""

    def __init__(self, strategy_name: str = "Strategy"):
        super().__init__()
        self.strategy_name = strategy_name

    def header(self):
        """Page header."""
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, f"{self.strategy_name} Backtest Report", align="C")
        self.ln(10)

    def footer(self):
        """Page footer."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


class ReportGenerator:
    """
    Generates comprehensive PDF reports from backtest results.

    Features:
    - Executive summary with key metrics
    - Equity curve and drawdown charts
    - Trade analysis
    - Returns distribution
    - Monte Carlo results (if available)
    - Strategy parameters
    """

    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize ReportGenerator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = BacktestVisualizer()
        self._temp_files: List[str] = []

    def generate_report(
        self,
        result: BacktestResult,
        mc_result: Optional[MonteCarloResult] = None,
        filename: Optional[str] = None,
        include_trades: bool = True
    ) -> str:
        """
        Generate a comprehensive PDF report.

        Args:
            result: BacktestResult object
            mc_result: Optional Monte Carlo results
            filename: Output filename (auto-generated if None)
            include_trades: Include detailed trade list

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.strategy_name}_report_{timestamp}.pdf"

        output_path = self.output_dir / filename

        # Create PDF
        pdf = BacktestPDF(strategy_name=result.strategy_name)
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add sections
        self._add_title_page(pdf, result)
        self._add_executive_summary(pdf, result)
        self._add_equity_curve_page(pdf, result)
        self._add_drawdown_page(pdf, result)
        self._add_returns_analysis(pdf, result)

        if not result.trades_df.empty:
            self._add_trade_analysis(pdf, result)

        if include_trades and not result.trades_df.empty:
            self._add_trade_list(pdf, result)

        if mc_result:
            self._add_monte_carlo_page(pdf, mc_result)

        self._add_parameters_page(pdf, result)

        # Save PDF
        pdf.output(str(output_path))

        # Clean up temporary files
        self._cleanup_temp_files()

        logger.info(f"Report saved to: {output_path}")
        return str(output_path)

    def _add_title_page(self, pdf: BacktestPDF, result: BacktestResult):
        """Add title page."""
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 24)
        pdf.ln(40)
        pdf.cell(0, 20, "Backtest Report", align="C")

        # Strategy name
        pdf.ln(20)
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 15, result.strategy_name, align="C")

        # Date range
        pdf.ln(30)
        pdf.set_font("Helvetica", "", 12)
        date_range = f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}"
        pdf.cell(0, 10, f"Period: {date_range}", align="C")

        # Generated date
        pdf.ln(10)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

        # Summary metrics
        pdf.ln(30)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Key Results", align="C")

        pdf.ln(15)
        pdf.set_font("Helvetica", "", 12)

        metrics_summary = [
            f"Total Return: {result.metrics.total_return*100:.1f}%",
            f"CAGR: {result.metrics.cagr*100:.1f}%",
            f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}",
            f"Max Drawdown: {result.metrics.max_drawdown*100:.1f}%",
        ]

        for metric in metrics_summary:
            pdf.cell(0, 8, metric, align="C")
            pdf.ln(8)

    def _add_executive_summary(self, pdf: BacktestPDF, result: BacktestResult):
        """Add executive summary page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Executive Summary")
        pdf.ln(15)

        # Create metrics table
        pdf.set_font("Helvetica", "", 10)

        metrics_data = [
            ("Performance Metrics", ""),
            ("Total Return", f"{result.metrics.total_return*100:.2f}%"),
            ("CAGR", f"{result.metrics.cagr*100:.2f}%"),
            ("Volatility (Ann.)", f"{result.metrics.volatility*100:.2f}%"),
            ("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.2f}"),
            ("Sortino Ratio", f"{result.metrics.sortino_ratio:.2f}"),
            ("Calmar Ratio", f"{result.metrics.calmar_ratio:.2f}"),
            ("", ""),
            ("Risk Metrics", ""),
            ("Max Drawdown", f"{result.metrics.max_drawdown*100:.2f}%"),
            ("Max DD Duration", f"{result.metrics.max_drawdown_duration} days"),
            ("VaR (95%)", f"{result.metrics.var_95*100:.2f}%"),
            ("CVaR (95%)", f"{result.metrics.cvar_95*100:.2f}%"),
            ("", ""),
            ("Trade Statistics", ""),
            ("Total Trades", f"{result.metrics.total_trades}"),
            ("Win Rate", f"{result.metrics.win_rate*100:.1f}%"),
            ("Profit Factor", f"{result.metrics.profit_factor:.2f}"),
            ("Avg Win", f"{result.metrics.avg_win*100:.2f}%"),
            ("Avg Loss", f"{result.metrics.avg_loss*100:.2f}%"),
        ]

        # Add benchmark metrics if available
        if result.metrics.beta is not None:
            metrics_data.extend([
                ("", ""),
                ("Benchmark Comparison", ""),
                ("Beta", f"{result.metrics.beta:.2f}"),
                ("Alpha (Ann.)", f"{result.metrics.alpha*100:.2f}%" if result.metrics.alpha else "N/A"),
                ("Information Ratio", f"{result.metrics.information_ratio:.2f}" if result.metrics.information_ratio else "N/A"),
            ])

        col_width = 60
        row_height = 7

        for label, value in metrics_data:
            if label and not value:
                # Section header
                pdf.set_font("Helvetica", "B", 11)
                pdf.ln(5)
                pdf.cell(col_width, row_height, label)
                pdf.ln(row_height)
                pdf.set_font("Helvetica", "", 10)
            elif label:
                pdf.cell(col_width, row_height, label)
                pdf.cell(col_width, row_height, value)
                pdf.ln(row_height)

        # Capital summary
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 10, "Capital Summary")
        pdf.ln(10)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(60, 7, "Initial Capital:")
        pdf.cell(60, 7, f"${result.initial_capital:,.2f}")
        pdf.ln(7)
        pdf.cell(60, 7, "Final Capital:")
        pdf.cell(60, 7, f"${result.final_capital:,.2f}")
        pdf.ln(7)
        pdf.cell(60, 7, "Net Profit/Loss:")
        net_pnl = result.final_capital - result.initial_capital
        pdf.cell(60, 7, f"${net_pnl:,.2f}")

    def _add_equity_curve_page(self, pdf: BacktestPDF, result: BacktestResult):
        """Add equity curve chart page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Equity Curve")
        pdf.ln(15)

        # Generate and save chart
        fig = self.visualizer.plot_equity_curve(result)
        img_path = self._save_temp_figure(fig, "equity_curve")

        # Add image to PDF
        pdf.image(img_path, x=10, w=190)

    def _add_drawdown_page(self, pdf: BacktestPDF, result: BacktestResult):
        """Add drawdown analysis page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Drawdown Analysis")
        pdf.ln(15)

        # Generate and save chart
        fig = self.visualizer.plot_drawdown(result)
        img_path = self._save_temp_figure(fig, "drawdown")

        pdf.image(img_path, x=10, w=190)

        # Add drawdown statistics
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Drawdown Statistics")
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 10)

        # Calculate drawdown periods
        dd = result.drawdown_series
        in_drawdown = dd < -0.05  # Significant drawdowns

        stats = [
            f"Maximum Drawdown: {result.metrics.max_drawdown*100:.2f}%",
            f"Average Drawdown: {dd[dd < 0].mean()*100:.2f}%" if (dd < 0).any() else "N/A",
            f"Time in Drawdown > 5%: {in_drawdown.sum() / len(dd) * 100:.1f}% of time",
        ]

        for stat in stats:
            pdf.cell(0, 7, stat)
            pdf.ln(7)

    def _add_returns_analysis(self, pdf: BacktestPDF, result: BacktestResult):
        """Add returns analysis page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Returns Analysis")
        pdf.ln(15)

        # Returns distribution chart
        fig = self.visualizer.plot_returns_distribution(result)
        img_path = self._save_temp_figure(fig, "returns_dist")
        pdf.image(img_path, x=10, w=190)

        # Add monthly returns heatmap on new page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Monthly Returns")
        pdf.ln(15)

        fig = self.visualizer.plot_monthly_returns_heatmap(result)
        img_path = self._save_temp_figure(fig, "monthly_returns")
        pdf.image(img_path, x=10, w=190)

    def _add_trade_analysis(self, pdf: BacktestPDF, result: BacktestResult):
        """Add trade analysis page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Trade Analysis")
        pdf.ln(15)

        fig = self.visualizer.plot_trade_analysis(result)
        img_path = self._save_temp_figure(fig, "trade_analysis")
        pdf.image(img_path, x=10, w=190)

    def _add_trade_list(self, pdf: BacktestPDF, result: BacktestResult):
        """Add detailed trade list."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Trade History")
        pdf.ln(15)

        if result.trades_df.empty:
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 10, "No trades executed.")
            return

        # Table headers
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [25, 20, 20, 20, 20, 20, 15, 25, 25]
        headers = ["Ticker", "Entry Date", "Exit Date", "Entry $", "Exit $", "Qty", "Dir", "P&L", "P&L %"]

        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, border=1, align="C")
        pdf.ln(7)

        # Table rows
        pdf.set_font("Helvetica", "", 7)
        trades = result.trades_df.head(50)  # Limit to 50 trades for space

        for _, trade in trades.iterrows():
            entry_date = str(trade.get("entry_date", ""))[:10]
            exit_date = str(trade.get("exit_date", ""))[:10]

            row_data = [
                str(trade.get("ticker", ""))[:8],
                entry_date,
                exit_date,
                f"{trade.get('entry_price', 0):.2f}",
                f"{trade.get('exit_price', 0):.2f}",
                f"{trade.get('quantity', 0):.1f}",
                str(trade.get("direction", ""))[:5],
                f"${trade.get('pnl', 0):.0f}",
                f"{trade.get('pnl_pct', 0)*100:.1f}%",
            ]

            for i, data in enumerate(row_data):
                pdf.cell(col_widths[i], 6, data, border=1, align="C")
            pdf.ln(6)

        if len(result.trades_df) > 50:
            pdf.ln(5)
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, f"(Showing first 50 of {len(result.trades_df)} trades)")

    def _add_monte_carlo_page(self, pdf: BacktestPDF, mc_result: MonteCarloResult):
        """Add Monte Carlo analysis page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Monte Carlo Analysis")
        pdf.ln(15)

        # Summary statistics
        pdf.set_font("Helvetica", "", 10)

        ci = mc_result.confidence_intervals

        pdf.cell(0, 7, f"Number of Simulations: {mc_result.n_simulations}")
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Confidence Intervals (5th - 50th - 95th percentile):")
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 10)

        metrics_ci = [
            ("Total Return", "return", "%"),
            ("CAGR", "cagr", "%"),
            ("Sharpe Ratio", "sharpe", ""),
            ("Max Drawdown", "max_dd", "%"),
        ]

        for label, key, suffix in metrics_ci:
            if key in ci:
                vals = ci[key]
                mult = 100 if suffix == "%" else 1
                pdf.cell(50, 7, f"{label}:")
                pdf.cell(
                    0, 7,
                    f"{vals['5%']*mult:.1f}{suffix} | {vals['50%']*mult:.1f}{suffix} | {vals['95%']*mult:.1f}{suffix}"
                )
                pdf.ln(7)

        # Probability metrics
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Risk Assessment:")
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 10)

        prob_positive = (mc_result.return_distribution > 0).mean() * 100
        prob_sharpe = (mc_result.sharpe_distribution > 0).mean() * 100
        prob_dd = (mc_result.max_drawdown_distribution > -0.2).mean() * 100

        pdf.cell(0, 7, f"Probability of Positive Return: {prob_positive:.1f}%")
        pdf.ln(7)
        pdf.cell(0, 7, f"Probability of Positive Sharpe: {prob_sharpe:.1f}%")
        pdf.ln(7)
        pdf.cell(0, 7, f"Probability of Max DD < 20%: {prob_dd:.1f}%")

        # Add Monte Carlo charts
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Monte Carlo Distributions")
        pdf.ln(15)

        fig = self.visualizer.plot_monte_carlo_results(mc_result)
        img_path = self._save_temp_figure(fig, "monte_carlo")
        pdf.image(img_path, x=5, w=200)

    def _add_parameters_page(self, pdf: BacktestPDF, result: BacktestResult):
        """Add strategy parameters page."""
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Strategy Parameters")
        pdf.ln(15)

        pdf.set_font("Helvetica", "", 10)

        if result.params:
            for key, value in result.params.items():
                pdf.cell(60, 7, str(key))
                pdf.cell(60, 7, str(value))
                pdf.ln(7)
        else:
            pdf.cell(0, 10, "No parameters recorded.")

        # Backtest settings
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Backtest Configuration")
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 10)

        settings = [
            f"Initial Capital: ${result.initial_capital:,.2f}",
            f"Start Date: {result.start_date.strftime('%Y-%m-%d')}",
            f"End Date: {result.end_date.strftime('%Y-%m-%d')}",
            f"Trading Days: {len(result.equity_curve)}",
        ]

        for setting in settings:
            pdf.cell(0, 7, setting)
            pdf.ln(7)

    def _save_temp_figure(self, fig: plt.Figure, name: str) -> str:
        """Save figure to temporary file."""
        temp_path = tempfile.mktemp(suffix=f"_{name}.png")
        fig.savefig(temp_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self._temp_files.append(temp_path)
        return temp_path

    def _cleanup_temp_files(self):
        """Remove temporary files."""
        for path in self._temp_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._temp_files.clear()


def generate_comparison_report(
    results: Dict[str, BacktestResult],
    output_path: str = "./reports/comparison_report.pdf"
) -> str:
    """
    Generate a report comparing multiple strategy results.

    Args:
        results: Dictionary mapping strategy names to BacktestResult
        output_path: Output file path

    Returns:
        Path to generated PDF
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(40)
    pdf.cell(0, 15, "Strategy Comparison Report", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Comparing {len(results)} strategies", align="C")
    pdf.ln(10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

    # Comparison table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Performance Comparison")
    pdf.ln(15)

    # Table headers
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [35, 25, 25, 25, 25, 25, 20]
    headers = ["Strategy", "Return", "CAGR", "Sharpe", "Max DD", "Win Rate", "Trades"]

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align="C")
    pdf.ln(8)

    # Table data
    pdf.set_font("Helvetica", "", 9)
    for name, result in results.items():
        m = result.metrics
        row = [
            name[:12],
            f"{m.total_return*100:.1f}%",
            f"{m.cagr*100:.1f}%",
            f"{m.sharpe_ratio:.2f}",
            f"{m.max_drawdown*100:.1f}%",
            f"{m.win_rate*100:.1f}%",
            str(m.total_trades)
        ]
        for i, data in enumerate(row):
            pdf.cell(col_widths[i], 7, data, border=1, align="C")
        pdf.ln(7)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)

    return output_path


if __name__ == "__main__":
    # Test report generation
    logging.basicConfig(level=logging.INFO)

    from data_fetcher import DataFetcher
    from strategies.momentum import MomentumStrategy
    from backtester import Backtester

    print("Testing Report Generator...")

    # Run a backtest
    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
    prices = fetcher.fetch_multiple_tickers(tickers, start_date="2021-01-01")

    strategy = MomentumStrategy(lookback_period=20, top_n=3)
    backtester = Backtester()
    result = backtester.run(strategy, prices)

    # Generate report
    generator = ReportGenerator()
    report_path = generator.generate_report(result)

    print(f"\nReport generated: {report_path}")
