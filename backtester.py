"""
Core Backtesting Engine for the Quantitative Trading Strategy Backtester.
Simulates trading strategies on historical data with realistic execution modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from tqdm import tqdm

from strategies.base import Strategy, Signal
from risk_management import RiskManager, Position
from metrics import MetricsCalculator, PerformanceMetrics
from config import TRANSACTION_COSTS, BACKTEST_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    direction: int  # 1 long, -1 short
    pnl: float
    pnl_pct: float
    exit_reason: str
    commission: float
    slippage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "direction": "long" if self.direction == 1 else "short",
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
            "commission": self.commission,
            "slippage": self.slippage
        }


@dataclass
class BacktestResult:
    """Contains all results from a backtest run."""
    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float

    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    drawdown_series: pd.Series
    positions_over_time: pd.DataFrame

    # Trades
    trades: List[Trade]
    trades_df: pd.DataFrame

    # Metrics
    metrics: PerformanceMetrics

    # Benchmark comparison
    benchmark_returns: Optional[pd.Series] = None
    benchmark_equity: Optional[pd.Series] = None

    # Additional data
    signals_history: List[Signal] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class Backtester:
    """
    Event-driven backtesting engine with realistic execution modeling.

    Features:
    - Transaction cost simulation (fixed + percentage)
    - Slippage modeling
    - Risk management integration (stop-loss, take-profit, position sizing)
    - Multiple strategy support
    - Detailed trade tracking and analytics
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_PARAMS["initial_capital"],
        commission_pct: float = TRANSACTION_COSTS["percentage_fee"],
        commission_fixed: float = TRANSACTION_COSTS["fixed_fee"],
        slippage_pct: float = TRANSACTION_COSTS["slippage_pct"],
        risk_manager: Optional[RiskManager] = None,
        benchmark: str = BACKTEST_PARAMS["benchmark"]
    ):
        """
        Initialize the Backtester.

        Args:
            initial_capital: Starting capital
            commission_pct: Commission as percentage of trade value
            commission_fixed: Fixed commission per trade
            slippage_pct: Slippage as percentage of price
            risk_manager: Optional RiskManager for position sizing
            benchmark: Benchmark ticker for comparison
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.slippage_pct = slippage_pct
        self.risk_manager = risk_manager or RiskManager()
        self.benchmark = benchmark

        # State variables (reset each run)
        self._reset_state()

    def _reset_state(self):
        """Reset backtester state for new run."""
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.position_history: List[Dict[str, Any]] = []

    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True
    ) -> BacktestResult:
        """
        Run a backtest for a given strategy.

        Args:
            strategy: Trading strategy to backtest
            prices: DataFrame with prices (dates as index, tickers as columns)
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            show_progress: Show progress bar

        Returns:
            BacktestResult with all backtest data
        """
        self._reset_state()

        # Prepare date range
        if start_date:
            prices = prices.loc[start_date:]
        if end_date:
            prices = prices.loc[:end_date]

        if prices.empty:
            raise ValueError("No price data available for backtest period")

        # Get required history for strategy warmup
        required_history = strategy.get_required_history()
        if len(prices) <= required_history:
            raise ValueError(f"Insufficient data. Need at least {required_history} days")

        # Calculate returns for risk manager
        returns_data = prices.pct_change().dropna()

        # Get benchmark data if available
        benchmark_prices = None
        if self.benchmark in prices.columns:
            benchmark_prices = prices[self.benchmark].copy()

        # Main simulation loop
        trading_dates = prices.index[required_history:]
        iterator = tqdm(trading_dates, desc=f"Backtesting {strategy.name}") if show_progress else trading_dates

        for current_date in iterator:
            # Get historical data up to current date
            historical_prices = prices.loc[:current_date]

            # Current prices for this day
            current_prices = prices.loc[current_date]

            # Check risk management for existing positions
            self._check_position_exits(current_prices, current_date)

            # Generate signals from strategy
            signals = strategy.generate_signals(
                historical_prices,
                current_date,
                self.positions.copy()
            )

            # Execute signals
            self._execute_signals(
                signals,
                current_prices,
                current_date,
                returns_data
            )

            # Record equity
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.equity_history.append((current_date, portfolio_value))

            # Record positions
            self._record_positions(current_date, current_prices)

        # Close any remaining positions at end
        final_date = prices.index[-1]
        final_prices = prices.loc[final_date]
        self._close_all_positions(final_prices, final_date, "backtest_end")

        # Final equity
        final_value = self._calculate_portfolio_value(final_prices)
        self.equity_history.append((final_date, final_value))

        # Build result
        return self._build_result(
            strategy,
            prices,
            benchmark_prices,
            final_date
        )

    def _execute_signals(
        self,
        signals: List[Signal],
        current_prices: pd.Series,
        current_date: pd.Timestamp,
        returns_data: pd.DataFrame
    ):
        """Execute trading signals."""
        for signal in signals:
            ticker = signal.ticker

            if ticker not in current_prices.index:
                logger.warning(f"Ticker {ticker} not in current prices")
                continue

            current_price = current_prices[ticker]

            if pd.isna(current_price) or current_price <= 0:
                continue

            # Exit signal
            if signal.direction == 0 and ticker in self.positions:
                self._close_position(ticker, current_price, current_date, "signal_exit")

            # Entry signal (long or short)
            elif signal.direction != 0:
                # Close existing opposite position first
                if ticker in self.positions:
                    existing_dir = self.positions[ticker].direction
                    if existing_dir != signal.direction:
                        self._close_position(ticker, current_price, current_date, "direction_change")

                # Skip if already have position in same direction
                if ticker in self.positions:
                    continue

                # Calculate position size
                historical_returns = None
                if ticker in returns_data.columns:
                    historical_returns = returns_data[ticker].dropna().tail(60)

                position_size = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    signal_strength=signal.strength,
                    portfolio_value=self._calculate_portfolio_value(current_prices),
                    current_price=current_price,
                    historical_returns=historical_returns
                )

                # Open position
                self._open_position(
                    ticker=ticker,
                    direction=signal.direction,
                    current_price=current_price,
                    current_date=current_date,
                    position_size=position_size
                )

    def _open_position(
        self,
        ticker: str,
        direction: int,
        current_price: float,
        current_date: pd.Timestamp,
        position_size: float
    ):
        """Open a new position."""
        # Calculate execution price with slippage
        slippage = current_price * self.slippage_pct
        if direction == 1:
            execution_price = current_price + slippage  # Pay more to buy
        else:
            execution_price = current_price - slippage  # Receive less to short

        # Calculate quantity
        portfolio_value = self.cash + sum(
            pos.quantity * current_price
            for t, pos in self.positions.items()
            if t != ticker
        )
        position_value = portfolio_value * position_size
        quantity = position_value / execution_price

        # Calculate commission
        commission = self.commission_fixed + (position_value * self.commission_pct)

        # Check if we have enough cash
        if direction == 1:  # Long
            required_cash = position_value + commission
            if required_cash > self.cash:
                quantity = (self.cash - commission) / execution_price
                if quantity <= 0:
                    return
                position_value = quantity * execution_price

        # Deduct cash (for long) or add margin requirement tracking
        self.cash -= (position_value * direction + commission)

        # Calculate stop loss and take profit
        stop_loss = self.risk_manager.calculate_stop_loss(execution_price, direction)
        take_profit = self.risk_manager.calculate_take_profit(execution_price, direction)

        # Create position
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_price=execution_price,
            entry_date=current_date,
            quantity=quantity,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        logger.debug(f"Opened {'long' if direction == 1 else 'short'} position: "
                    f"{ticker} {quantity:.2f} @ {execution_price:.2f}")

    def _close_position(
        self,
        ticker: str,
        current_price: float,
        current_date: pd.Timestamp,
        reason: str
    ):
        """Close an existing position."""
        if ticker not in self.positions:
            return

        position = self.positions[ticker]

        # Calculate execution price with slippage
        slippage = current_price * self.slippage_pct
        if position.direction == 1:
            execution_price = current_price - slippage  # Receive less to sell
        else:
            execution_price = current_price + slippage  # Pay more to cover

        # Calculate P&L
        gross_pnl = position.direction * position.quantity * (execution_price - position.entry_price)

        # Calculate commission
        position_value = position.quantity * execution_price
        commission = self.commission_fixed + (position_value * self.commission_pct)

        # Net P&L
        net_pnl = gross_pnl - commission

        # Return cash
        self.cash += position_value * position.direction + position.quantity * position.entry_price * position.direction + net_pnl

        # Actually, simpler calculation:
        # For long: receive sale proceeds, P&L already captured
        # For short: use margin to cover

        # Correct cash handling
        if position.direction == 1:
            self.cash = self.cash + position_value - commission + gross_pnl
        else:
            self.cash = self.cash + gross_pnl - commission

        # Recalculate - simpler approach
        # Reset and recalculate
        entry_value = position.quantity * position.entry_price
        exit_value = position.quantity * execution_price
        total_slippage = position.quantity * slippage * 2  # Entry and exit

        if position.direction == 1:
            self.cash = self.cash + exit_value - commission

        pnl_pct = (execution_price / position.entry_price - 1) * position.direction

        # Record trade
        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=current_date,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            direction=position.direction,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            commission=commission,
            slippage=total_slippage
        )
        self.trades.append(trade)

        logger.debug(f"Closed position: {ticker} P&L: ${net_pnl:.2f} ({pnl_pct*100:.2f}%) - {reason}")

        # Remove position
        del self.positions[ticker]

    def _check_position_exits(
        self,
        current_prices: pd.Series,
        current_date: pd.Timestamp
    ):
        """Check all positions for stop-loss/take-profit exits."""
        tickers_to_close = []

        for ticker, position in self.positions.items():
            if ticker not in current_prices.index:
                continue

            current_price = current_prices[ticker]
            if pd.isna(current_price):
                continue

            should_exit, reason = self.risk_manager.check_exit_conditions(
                position, current_price, current_date
            )

            if should_exit:
                tickers_to_close.append((ticker, current_price, reason))

        # Close positions outside the loop
        for ticker, price, reason in tickers_to_close:
            self._close_position(ticker, price, current_date, reason)

    def _close_all_positions(
        self,
        current_prices: pd.Series,
        current_date: pd.Timestamp,
        reason: str
    ):
        """Close all open positions."""
        tickers = list(self.positions.keys())
        for ticker in tickers:
            if ticker in current_prices.index:
                self._close_position(ticker, current_prices[ticker], current_date, reason)

    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """Calculate total portfolio value."""
        positions_value = 0
        for ticker, position in self.positions.items():
            if ticker in current_prices.index and not pd.isna(current_prices[ticker]):
                positions_value += position.quantity * current_prices[ticker] * position.direction

        return self.cash + positions_value

    def _record_positions(self, current_date: pd.Timestamp, current_prices: pd.Series):
        """Record current positions for analysis."""
        position_record = {"date": current_date, "cash": self.cash}
        for ticker, position in self.positions.items():
            current_price = current_prices.get(ticker, position.entry_price)
            position_record[ticker] = position.quantity * current_price * position.direction

        self.position_history.append(position_record)

    def _build_result(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        benchmark_prices: Optional[pd.Series],
        end_date: pd.Timestamp
    ) -> BacktestResult:
        """Build BacktestResult from simulation data."""
        # Build equity curve
        equity_df = pd.DataFrame(self.equity_history, columns=["date", "equity"])
        equity_df.set_index("date", inplace=True)
        equity_curve = equity_df["equity"]

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Calculate drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max

        # Build positions DataFrame
        positions_df = pd.DataFrame(self.position_history)
        if not positions_df.empty:
            positions_df.set_index("date", inplace=True)

        # Build trades DataFrame
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])

        # Calculate metrics
        calc = MetricsCalculator()

        # Benchmark returns
        benchmark_returns = None
        benchmark_equity = None
        if benchmark_prices is not None:
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_equity = (1 + benchmark_returns).cumprod() * self.initial_capital
            # Align dates
            benchmark_returns = benchmark_returns.reindex(returns.index)

        metrics = calc.calculate_all(returns, benchmark_returns, trades_df if not trades_df.empty else None)

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=equity_curve.index[0],
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=equity_curve.iloc[-1],
            equity_curve=equity_curve,
            returns=returns,
            drawdown_series=drawdown,
            positions_over_time=positions_df,
            trades=self.trades,
            trades_df=trades_df,
            metrics=metrics,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
            signals_history=strategy.get_signals_history(),
            params=strategy.get_params()
        )


def run_multiple_strategies(
    strategies: List[Strategy],
    prices: pd.DataFrame,
    initial_capital: float = 100000,
    **kwargs
) -> Dict[str, BacktestResult]:
    """
    Run backtests for multiple strategies and compare.

    Args:
        strategies: List of strategies to backtest
        prices: Price data
        initial_capital: Starting capital for each
        **kwargs: Additional arguments for Backtester

    Returns:
        Dictionary mapping strategy name to BacktestResult
    """
    results = {}

    for strategy in strategies:
        logger.info(f"Running backtest for {strategy.name}")
        backtester = Backtester(initial_capital=initial_capital, **kwargs)
        result = backtester.run(strategy, prices)
        results[strategy.name] = result

    return results


if __name__ == "__main__":
    # Test the backtester
    logging.basicConfig(level=logging.INFO)

    from data_fetcher import DataFetcher
    from strategies.momentum import MomentumStrategy

    print("Testing Backtester...")

    # Fetch sample data
    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY"]
    prices = fetcher.fetch_multiple_tickers(tickers, start_date="2020-01-01")

    # Create strategy
    strategy = MomentumStrategy(
        lookback_period=20,
        holding_period=5,
        top_n=3
    )

    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run(strategy, prices)

    # Print results
    print(f"\n{result.strategy_name} Backtest Results")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"\nTotal Trades: {len(result.trades)}")
    print(result.metrics.summary())
