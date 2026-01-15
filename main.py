#!/usr/bin/env python3
"""
Main Entry Point for the Quantitative Trading Strategy Backtester.

This script provides a command-line interface for running backtests,
generating reports, and launching the Streamlit dashboard.

Usage:
    python main.py backtest --strategy momentum --tickers AAPL MSFT GOOGL
    python main.py dashboard
    python main.py demo
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_backtest(
    strategy: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float,
    generate_report: bool,
    run_monte_carlo: bool,
    mc_simulations: int,
    **kwargs
):
    """
    Run a backtest with specified parameters.

    Args:
        strategy: Strategy type ('momentum', 'mean_reversion', 'pairs', 'dual_momentum')
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        generate_report: Generate PDF report
        run_monte_carlo: Run Monte Carlo simulation
        mc_simulations: Number of MC simulations
    """
    from data_fetcher import DataFetcher
    from backtester import Backtester
    from risk_management import RiskManager
    from strategies.momentum import MomentumStrategy, DualMomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.pairs_trading import PairsTradingStrategy
    from monte_carlo import MonteCarloSimulator
    from report_generator import ReportGenerator

    print("\n" + "="*60)
    print("QUANTITATIVE TRADING STRATEGY BACKTESTER")
    print("="*60)

    # Fetch data
    print(f"\nFetching data for {len(tickers)} tickers...")
    fetcher = DataFetcher()
    prices = fetcher.fetch_multiple_tickers(tickers, start_date, end_date)

    if prices.empty:
        print("Error: No data available for specified tickers and date range.")
        return

    print(f"Loaded {len(prices)} trading days for {len(prices.columns)} assets")

    # Create strategy
    print(f"\nInitializing {strategy} strategy...")

    strategy_map = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "pairs": PairsTradingStrategy,
        "dual_momentum": DualMomentumStrategy
    }

    if strategy not in strategy_map:
        print(f"Error: Unknown strategy '{strategy}'")
        print(f"Available strategies: {list(strategy_map.keys())}")
        return

    # Extract strategy-specific kwargs
    strategy_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    trading_strategy = strategy_map[strategy](**strategy_kwargs)

    # Create backtester
    risk_manager = RiskManager()
    backtester = Backtester(
        initial_capital=initial_capital,
        risk_manager=risk_manager
    )

    # Run backtest
    print(f"\nRunning backtest...")
    result = backtester.run(trading_strategy, prices)

    # Print results
    print("\n" + result.metrics.summary())

    print(f"\nCapital: ${result.initial_capital:,.2f} -> ${result.final_capital:,.2f}")
    print(f"Total Trades: {len(result.trades)}")

    # Monte Carlo simulation
    mc_result = None
    if run_monte_carlo:
        print(f"\nRunning Monte Carlo simulation ({mc_simulations} iterations)...")
        mc = MonteCarloSimulator(n_simulations=mc_simulations)
        mc_result = mc.run_simulation(trading_strategy, prices)
        print(mc_result.summary())

    # Generate report
    if generate_report:
        print("\nGenerating PDF report...")
        generator = ReportGenerator()
        report_path = generator.generate_report(result, mc_result)
        print(f"Report saved to: {report_path}")

    print("\n" + "="*60)
    print("Backtest complete!")
    print("="*60 + "\n")

    return result


def run_demo():
    """Run a demonstration backtest with sample data."""
    from config import STOCK_TICKERS, ETF_TICKERS

    print("\n" + "="*60)
    print("RUNNING DEMO BACKTEST")
    print("="*60)

    # Demo parameters
    tickers = STOCK_TICKERS[:5] + ETF_TICKERS[:3]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years

    print(f"\nDemo Configuration:")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Strategy: Momentum (20-day lookback, top 3 assets)")
    print(f"  Initial Capital: $100,000")

    result = run_backtest(
        strategy="momentum",
        tickers=tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=100000,
        generate_report=True,
        run_monte_carlo=True,
        mc_simulations=500,
        lookback_period=20,
        top_n=3,
        holding_period=5
    )

    return result


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess

    print("\nLaunching Streamlit dashboard...")
    print("Open http://localhost:8501 in your browser\n")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")

    subprocess.run(["streamlit", "run", app_path])


def run_comparison():
    """Run comparison of multiple strategies."""
    from data_fetcher import DataFetcher
    from backtester import run_multiple_strategies
    from strategies.momentum import MomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from report_generator import generate_comparison_report
    from config import STOCK_TICKERS, ETF_TICKERS

    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)

    # Fetch data
    tickers = STOCK_TICKERS[:8] + ETF_TICKERS[:4]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)

    print(f"\nFetching data for {len(tickers)} tickers...")
    fetcher = DataFetcher()
    prices = fetcher.fetch_multiple_tickers(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    # Create strategies
    strategies = [
        MomentumStrategy(lookback_period=20, top_n=3, holding_period=5),
        MomentumStrategy(lookback_period=60, top_n=5, holding_period=20),
        MeanReversionStrategy(lookback_period=20, entry_zscore=-2.0),
    ]

    # Rename for clarity
    strategies[0].name = "Momentum_20d"
    strategies[1].name = "Momentum_60d"
    strategies[2].name = "MeanRev_20d"

    print(f"\nRunning {len(strategies)} strategy backtests...")
    results = run_multiple_strategies(strategies, prices)

    # Print comparison
    print("\n" + "-"*60)
    print(f"{'Strategy':<20} {'Return':>10} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-"*60)

    for name, result in results.items():
        m = result.metrics
        print(f"{name:<20} {m.total_return*100:>9.1f}% {m.cagr*100:>9.1f}% "
              f"{m.sharpe_ratio:>10.2f} {m.max_drawdown*100:>9.1f}%")

    print("-"*60)

    # Generate comparison report
    report_path = generate_comparison_report(results)
    print(f"\nComparison report saved to: {report_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantitative Trading Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo
  python main.py dashboard
  python main.py backtest -s momentum -t AAPL MSFT GOOGL --report
  python main.py compare
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest")
    bt_parser.add_argument(
        "-s", "--strategy",
        type=str,
        choices=["momentum", "mean_reversion", "pairs", "dual_momentum"],
        default="momentum",
        help="Trading strategy to use"
    )
    bt_parser.add_argument(
        "-t", "--tickers",
        type=str,
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY"],
        help="Ticker symbols to trade"
    )
    bt_parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    bt_parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    bt_parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital"
    )
    bt_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate PDF report"
    )
    bt_parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo simulation"
    )
    bt_parser.add_argument(
        "--mc-sims",
        type=int,
        default=500,
        help="Number of Monte Carlo simulations"
    )
    # Strategy-specific parameters
    bt_parser.add_argument("--lookback", type=int, help="Lookback period")
    bt_parser.add_argument("--holding", type=int, help="Holding period")
    bt_parser.add_argument("--top-n", type=int, help="Top N assets")
    bt_parser.add_argument("--entry-zscore", type=float, help="Entry z-score")
    bt_parser.add_argument("--exit-zscore", type=float, help="Exit z-score")

    # Demo command
    subparsers.add_parser("demo", help="Run a demonstration backtest")

    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    # Compare command
    subparsers.add_parser("compare", help="Compare multiple strategies")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "demo":
        run_demo()

    elif args.command == "dashboard":
        run_dashboard()

    elif args.command == "compare":
        run_comparison()

    elif args.command == "backtest":
        # Build kwargs for strategy
        strategy_kwargs = {}
        if args.lookback:
            strategy_kwargs["lookback_period"] = args.lookback
        if args.holding:
            strategy_kwargs["holding_period"] = args.holding
        if args.top_n:
            strategy_kwargs["top_n"] = args.top_n
        if args.entry_zscore:
            strategy_kwargs["entry_zscore"] = args.entry_zscore
        if args.exit_zscore:
            strategy_kwargs["exit_zscore"] = args.exit_zscore

        run_backtest(
            strategy=args.strategy,
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            generate_report=args.report,
            run_monte_carlo=args.monte_carlo,
            mc_simulations=args.mc_sims,
            **strategy_kwargs
        )


if __name__ == "__main__":
    main()
