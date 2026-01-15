"""
Monte Carlo Simulation Module for the Quantitative Trading Strategy Backtester.
Runs multiple simulations with randomized data to test strategy robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from strategies.base import Strategy
from backtester import Backtester, BacktestResult
from metrics import MetricsCalculator, PerformanceMetrics
from config import MONTE_CARLO_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    simulation_results: List[Dict[str, float]]

    # Distributions of key metrics
    return_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    cagr_distribution: np.ndarray

    # Confidence intervals
    confidence_intervals: Dict[str, Dict[str, float]]

    # Percentile equity curves
    percentile_curves: Dict[float, pd.Series]

    # Original backtest for comparison
    original_result: Optional[BacktestResult]

    def summary(self) -> str:
        """Generate text summary of Monte Carlo results."""
        ci = self.confidence_intervals
        lines = [
            "=" * 60,
            f"MONTE CARLO SIMULATION RESULTS ({self.n_simulations} simulations)",
            "=" * 60,
            "",
            "Metric Distributions (5th - 50th - 95th percentile):",
            "-" * 60,
            f"Total Return:   {ci['return']['5%']:>8.1%} | {ci['return']['50%']:>8.1%} | {ci['return']['95%']:>8.1%}",
            f"CAGR:           {ci['cagr']['5%']:>8.1%} | {ci['cagr']['50%']:>8.1%} | {ci['cagr']['95%']:>8.1%}",
            f"Sharpe Ratio:   {ci['sharpe']['5%']:>8.2f} | {ci['sharpe']['50%']:>8.2f} | {ci['sharpe']['95%']:>8.2f}",
            f"Max Drawdown:   {ci['max_dd']['5%']:>8.1%} | {ci['max_dd']['50%']:>8.1%} | {ci['max_dd']['95%']:>8.1%}",
            "",
            "-" * 60,
            f"Probability of Positive Return: {(self.return_distribution > 0).mean()*100:.1f}%",
            f"Probability of Positive Sharpe: {(self.sharpe_distribution > 0).mean()*100:.1f}%",
            f"Probability of < 20% Max DD:    {(self.max_drawdown_distribution > -0.2).mean()*100:.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for testing strategy robustness.

    Supports multiple simulation methods:
    1. Return perturbation: Add random noise to returns
    2. Bootstrap sampling: Resample historical returns with replacement
    3. Block bootstrap: Preserve autocorrelation structure
    4. Parameter variation: Vary strategy parameters
    """

    def __init__(
        self,
        n_simulations: int = MONTE_CARLO_PARAMS["n_simulations"],
        return_perturbation: float = MONTE_CARLO_PARAMS["return_perturbation"],
        bootstrap_block_size: int = MONTE_CARLO_PARAMS["bootstrap_block_size"],
        confidence_levels: List[float] = MONTE_CARLO_PARAMS["confidence_levels"],
        random_seed: Optional[int] = None,
        n_jobs: int = 1  # Number of parallel jobs
    ):
        """
        Initialize Monte Carlo Simulator.

        Args:
            n_simulations: Number of simulation runs
            return_perturbation: Standard deviation of random noise to add
            bootstrap_block_size: Block size for block bootstrap
            confidence_levels: Percentiles for confidence intervals
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel processes (1 = no parallelization)
        """
        self.n_simulations = n_simulations
        self.return_perturbation = return_perturbation
        self.bootstrap_block_size = bootstrap_block_size
        self.confidence_levels = confidence_levels
        self.n_jobs = min(n_jobs, multiprocessing.cpu_count())

        if random_seed:
            np.random.seed(random_seed)

    def run_simulation(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        backtester: Optional[Backtester] = None,
        method: str = "perturbation",
        show_progress: bool = True
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            strategy: Trading strategy to test
            prices: Historical price data
            backtester: Backtester instance (creates default if None)
            method: Simulation method ('perturbation', 'bootstrap', 'block_bootstrap')
            show_progress: Show progress bar

        Returns:
            MonteCarloResult with simulation statistics
        """
        if backtester is None:
            backtester = Backtester()

        # Run original backtest first
        logger.info("Running original backtest...")
        original_result = backtester.run(strategy, prices, show_progress=False)

        # Generate perturbed datasets
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations ({method})...")

        simulation_results = []
        equity_curves = []

        iterator = range(self.n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo")

        for i in iterator:
            # Generate perturbed prices
            perturbed_prices = self._generate_perturbed_data(prices, method)

            # Run backtest on perturbed data
            try:
                # Create fresh strategy instance to avoid state issues
                strategy_copy = self._copy_strategy(strategy)
                bt = Backtester(
                    initial_capital=backtester.initial_capital,
                    commission_pct=backtester.commission_pct,
                    commission_fixed=backtester.commission_fixed,
                    slippage_pct=backtester.slippage_pct
                )
                result = bt.run(strategy_copy, perturbed_prices, show_progress=False)

                # Extract metrics
                metrics_dict = {
                    "return": result.metrics.total_return,
                    "cagr": result.metrics.cagr,
                    "sharpe": result.metrics.sharpe_ratio,
                    "sortino": result.metrics.sortino_ratio,
                    "max_dd": result.metrics.max_drawdown,
                    "volatility": result.metrics.volatility,
                    "win_rate": result.metrics.win_rate,
                    "profit_factor": result.metrics.profit_factor
                }
                simulation_results.append(metrics_dict)
                equity_curves.append(result.equity_curve)

            except Exception as e:
                logger.warning(f"Simulation {i} failed: {e}")
                continue

        if len(simulation_results) == 0:
            raise ValueError("All simulations failed")

        # Build result
        return self._build_result(
            simulation_results,
            equity_curves,
            original_result
        )

    def _generate_perturbed_data(
        self,
        prices: pd.DataFrame,
        method: str
    ) -> pd.DataFrame:
        """Generate perturbed price data based on method."""
        if method == "perturbation":
            return self._perturb_returns(prices)
        elif method == "bootstrap":
            return self._bootstrap_returns(prices)
        elif method == "block_bootstrap":
            return self._block_bootstrap_returns(prices)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _perturb_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Add random noise to returns and reconstruct prices."""
        returns = prices.pct_change()

        # Add random perturbation
        noise = np.random.normal(0, self.return_perturbation, returns.shape)
        perturbed_returns = returns + noise

        # Reconstruct prices
        perturbed_prices = prices.iloc[0:1].copy()
        for i in range(1, len(prices)):
            new_row = perturbed_prices.iloc[-1] * (1 + perturbed_returns.iloc[i])
            new_row.name = prices.index[i]
            perturbed_prices = pd.concat([perturbed_prices, new_row.to_frame().T])

        return perturbed_prices

    def _bootstrap_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Resample returns with replacement."""
        returns = prices.pct_change().dropna()

        # Bootstrap sample
        n_samples = len(returns)
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrapped_returns = returns.iloc[sample_indices].reset_index(drop=True)

        # Reconstruct prices
        initial_prices = prices.iloc[0]
        perturbed_prices = [initial_prices]

        for i in range(len(bootstrapped_returns)):
            new_row = perturbed_prices[-1] * (1 + bootstrapped_returns.iloc[i])
            perturbed_prices.append(new_row)

        result = pd.DataFrame(perturbed_prices, index=prices.index[:len(perturbed_prices)])
        return result

    def _block_bootstrap_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Block bootstrap to preserve autocorrelation structure.

        Samples contiguous blocks of returns instead of individual observations.
        """
        returns = prices.pct_change().dropna()
        n_samples = len(returns)
        block_size = self.bootstrap_block_size

        # Calculate number of blocks needed
        n_blocks = int(np.ceil(n_samples / block_size))

        # Sample block starting points
        max_start = max(1, n_samples - block_size)
        block_starts = np.random.randint(0, max_start, size=n_blocks)

        # Build bootstrapped returns
        bootstrapped_returns = []
        for start in block_starts:
            end = min(start + block_size, n_samples)
            block = returns.iloc[start:end]
            bootstrapped_returns.append(block)

        bootstrapped_returns = pd.concat(bootstrapped_returns, ignore_index=True)
        bootstrapped_returns = bootstrapped_returns.iloc[:n_samples]  # Trim to original length

        # Reconstruct prices
        initial_prices = prices.iloc[0]
        perturbed_prices = [initial_prices]

        for i in range(len(bootstrapped_returns)):
            new_row = perturbed_prices[-1] * (1 + bootstrapped_returns.iloc[i])
            perturbed_prices.append(new_row)

        result = pd.DataFrame(perturbed_prices, index=prices.index[:len(perturbed_prices)])
        return result

    def _copy_strategy(self, strategy: Strategy) -> Strategy:
        """Create a fresh copy of the strategy."""
        # Import strategy classes
        from strategies.momentum import MomentumStrategy, DualMomentumStrategy
        from strategies.mean_reversion import MeanReversionStrategy, StatisticalArbitrageStrategy
        from strategies.pairs_trading import PairsTradingStrategy

        strategy_map = {
            "Momentum": MomentumStrategy,
            "DualMomentum": DualMomentumStrategy,
            "MeanReversion": MeanReversionStrategy,
            "StatArb": StatisticalArbitrageStrategy,
            "PairsTrading": PairsTradingStrategy
        }

        strategy_class = strategy_map.get(strategy.name)
        if strategy_class:
            return strategy_class(**strategy.params)

        # Fallback: return same strategy (not ideal but works)
        strategy.clear_history()
        return strategy

    def _build_result(
        self,
        simulation_results: List[Dict[str, float]],
        equity_curves: List[pd.Series],
        original_result: BacktestResult
    ) -> MonteCarloResult:
        """Build MonteCarloResult from simulation data."""
        # Convert to arrays
        results_df = pd.DataFrame(simulation_results)

        return_dist = results_df["return"].values
        sharpe_dist = results_df["sharpe"].values
        max_dd_dist = results_df["max_dd"].values
        cagr_dist = results_df["cagr"].values

        # Calculate confidence intervals
        confidence_intervals = {}
        metrics_map = {
            "return": return_dist,
            "cagr": cagr_dist,
            "sharpe": sharpe_dist,
            "max_dd": max_dd_dist,
            "volatility": results_df["volatility"].values,
            "win_rate": results_df["win_rate"].values
        }

        for metric_name, values in metrics_map.items():
            ci = {}
            for level in self.confidence_levels:
                percentile = level * 100
                ci[f"{percentile:.0f}%"] = np.percentile(values, percentile)
            confidence_intervals[metric_name] = ci

        # Calculate percentile equity curves
        percentile_curves = self._calculate_percentile_curves(
            equity_curves,
            self.confidence_levels
        )

        return MonteCarloResult(
            n_simulations=len(simulation_results),
            simulation_results=simulation_results,
            return_distribution=return_dist,
            sharpe_distribution=sharpe_dist,
            max_drawdown_distribution=max_dd_dist,
            cagr_distribution=cagr_dist,
            confidence_intervals=confidence_intervals,
            percentile_curves=percentile_curves,
            original_result=original_result
        )

    def _calculate_percentile_curves(
        self,
        equity_curves: List[pd.Series],
        percentiles: List[float]
    ) -> Dict[float, pd.Series]:
        """Calculate percentile equity curves from simulations."""
        if not equity_curves:
            return {}

        # Align all curves to same index
        reference_index = equity_curves[0].index
        aligned_curves = []

        for curve in equity_curves:
            # Reindex to reference, forward fill gaps
            reindexed = curve.reindex(reference_index, method='ffill')
            aligned_curves.append(reindexed.values)

        # Stack into matrix
        curves_matrix = np.array(aligned_curves)

        # Calculate percentiles at each time point
        result = {}
        for p in percentiles:
            percentile_values = np.percentile(curves_matrix, p * 100, axis=0)
            result[p] = pd.Series(percentile_values, index=reference_index)

        return result


def run_sensitivity_analysis(
    strategy_class,
    base_params: Dict[str, Any],
    param_ranges: Dict[str, List[Any]],
    prices: pd.DataFrame,
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Run sensitivity analysis by varying strategy parameters.

    Args:
        strategy_class: Strategy class to instantiate
        base_params: Base parameter dictionary
        param_ranges: Dictionary mapping parameter names to lists of values to test
        prices: Price data for backtesting
        n_samples: Number of random parameter combinations to test

    Returns:
        DataFrame with parameter combinations and resulting metrics
    """
    results = []
    backtester = Backtester()

    # Generate random parameter combinations
    param_names = list(param_ranges.keys())

    for _ in tqdm(range(n_samples), desc="Sensitivity Analysis"):
        # Sample parameters
        params = base_params.copy()
        sampled_values = {}

        for param_name in param_names:
            values = param_ranges[param_name]
            sampled_value = np.random.choice(values)
            params[param_name] = sampled_value
            sampled_values[param_name] = sampled_value

        # Run backtest
        try:
            strategy = strategy_class(**params)
            result = backtester.run(strategy, prices, show_progress=False)

            # Record results
            record = sampled_values.copy()
            record.update({
                "total_return": result.metrics.total_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate
            })
            results.append(record)

        except Exception as e:
            logger.warning(f"Sensitivity analysis iteration failed: {e}")
            continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test Monte Carlo simulation
    logging.basicConfig(level=logging.INFO)

    from data_fetcher import DataFetcher
    from strategies.momentum import MomentumStrategy

    print("Testing Monte Carlo Simulation...")

    # Fetch sample data
    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
    prices = fetcher.fetch_multiple_tickers(tickers, start_date="2020-01-01")

    # Create strategy
    strategy = MomentumStrategy(lookback_period=20, top_n=3)

    # Run Monte Carlo
    mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
    result = mc.run_simulation(strategy, prices, method="perturbation")

    print(result.summary())

    # Print confidence intervals
    print("\nDetailed Confidence Intervals:")
    for metric, ci in result.confidence_intervals.items():
        print(f"\n{metric}:")
        for level, value in ci.items():
            print(f"  {level}: {value:.4f}")
