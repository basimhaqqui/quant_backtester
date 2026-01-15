"""
Performance Metrics Module for the Quantitative Trading Strategy Backtester.
Calculates key performance indicators for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # Days
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    volatility: float
    beta: Optional[float]
    alpha: Optional[float]
    information_ratio: Optional[float]
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "Total Return (%)": self.total_return * 100,
            "CAGR (%)": self.cagr * 100,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown (%)": self.max_drawdown * 100,
            "Max DD Duration (days)": self.max_drawdown_duration,
            "Calmar Ratio": self.calmar_ratio,
            "Win Rate (%)": self.win_rate * 100,
            "Profit Factor": self.profit_factor,
            "Avg Win (%)": self.avg_win * 100,
            "Avg Loss (%)": self.avg_loss * 100,
            "Total Trades": self.total_trades,
            "Volatility (%)": self.volatility * 100,
            "Beta": self.beta if self.beta else np.nan,
            "Alpha (%)": self.alpha * 100 if self.alpha else np.nan,
            "Information Ratio": self.information_ratio if self.information_ratio else np.nan,
            "VaR 95% (%)": self.var_95 * 100,
            "CVaR 95% (%)": self.cvar_95 * 100,
            "Skewness": self.skewness,
            "Kurtosis": self.kurtosis,
        }

    def summary(self) -> str:
        """Generate a text summary of metrics."""
        lines = [
            "=" * 50,
            "PERFORMANCE SUMMARY",
            "=" * 50,
            f"Total Return:      {self.total_return*100:>10.2f}%",
            f"CAGR:              {self.cagr*100:>10.2f}%",
            f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"Sortino Ratio:     {self.sortino_ratio:>10.2f}",
            f"Max Drawdown:      {self.max_drawdown*100:>10.2f}%",
            f"Volatility:        {self.volatility*100:>10.2f}%",
            "-" * 50,
            f"Win Rate:          {self.win_rate*100:>10.2f}%",
            f"Profit Factor:     {self.profit_factor:>10.2f}",
            f"Total Trades:      {self.total_trades:>10d}",
            "-" * 50,
            f"VaR (95%):         {self.var_95*100:>10.2f}%",
            f"CVaR (95%):        {self.cvar_95*100:>10.2f}%",
            "=" * 50,
        ]
        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics for backtesting results.
    """

    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize the MetricsCalculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1

    def calculate_all(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            returns: Daily returns series (not equity curve)
            benchmark_returns: Optional benchmark returns for comparison
            trades: Optional DataFrame with trade history

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        returns = returns.dropna()

        if len(returns) < 2:
            logger.warning("Insufficient data for metrics calculation")
            return self._empty_metrics()

        # Basic returns metrics
        total_return = self.total_return(returns)
        cagr = self.cagr(returns)
        volatility = self.volatility(returns)

        # Risk-adjusted returns
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)

        # Drawdown analysis
        dd_series = self.drawdown_series(returns)
        max_dd = self.max_drawdown(returns)
        max_dd_duration = self.max_drawdown_duration(dd_series)

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        if trades is not None and len(trades) > 0:
            trade_stats = self._calculate_trade_stats(trades)
        else:
            trade_stats = self._trade_stats_from_returns(returns)

        # Risk metrics
        var_95 = self.value_at_risk(returns, 0.05)
        cvar_95 = self.conditional_var(returns, 0.05)

        # Distribution metrics
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned) > 10:
                beta = self.beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
                alpha = self.alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], beta)
                ir = self.information_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1])
            else:
                beta, alpha, ir = None, None, None
        else:
            beta, alpha, ir = None, None, None

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            total_trades=trade_stats["total_trades"],
            volatility=volatility,
            beta=beta,
            alpha=alpha,
            information_ratio=ir,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skew,
            kurtosis=kurt
        )

    def total_return(self, returns: pd.Series) -> float:
        """Calculate total cumulative return."""
        return (1 + returns).prod() - 1

    def cagr(self, returns: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        total_ret = self.total_return(returns)
        n_years = len(returns) / self.trading_days
        if n_years <= 0 or total_ret <= -1:
            return 0
        return (1 + total_ret) ** (1 / n_years) - 1

    def volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(self.trading_days)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - self.daily_rf
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        excess_returns = returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        downside_std = downside_returns.std() * np.sqrt(self.trading_days)
        annual_excess = excess_returns.mean() * self.trading_days

        return annual_excess / downside_std

    def drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series from returns."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown

    def max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        dd_series = self.drawdown_series(returns)
        return dd_series.min()

    def max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in trading days."""
        # Find periods where we're in drawdown
        in_drawdown = drawdown_series < 0

        if not in_drawdown.any():
            return 0

        # Calculate duration of each drawdown period
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        durations = in_drawdown.groupby(groups).sum()

        return int(durations.max()) if len(durations) > 0 else 0

    def value_at_risk(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate historical Value at Risk."""
        return returns.quantile(confidence)

    def conditional_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark."""
        covariance = returns.cov(benchmark_returns)
        benchmark_var = benchmark_returns.var()
        if benchmark_var == 0:
            return 0
        return covariance / benchmark_var

    def alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        beta: float
    ) -> float:
        """Calculate Jensen's alpha (annualized)."""
        portfolio_return = returns.mean() * self.trading_days
        benchmark_return = benchmark_returns.mean() * self.trading_days

        return portfolio_return - self.risk_free_rate - beta * (benchmark_return - self.risk_free_rate)

    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Information Ratio."""
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)

        if tracking_error == 0:
            return 0

        active_return_annual = active_returns.mean() * self.trading_days
        return active_return_annual / tracking_error

    def _calculate_trade_stats(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistics from trade DataFrame."""
        if "pnl" not in trades.columns or len(trades) == 0:
            return self._empty_trade_stats()

        pnl = trades["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return {
            "win_rate": win_rate,
            "profit_factor": min(profit_factor, 100),  # Cap for display
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": len(pnl)
        }

    def _trade_stats_from_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Estimate trade statistics from daily returns."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return {
            "win_rate": win_rate,
            "profit_factor": min(profit_factor, 100),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": len(returns)
        }

    def _empty_trade_stats(self) -> Dict[str, float]:
        """Return empty trade statistics."""
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "total_trades": 0
        }

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics object."""
        return PerformanceMetrics(
            total_return=0, cagr=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, max_drawdown_duration=0, calmar_ratio=0,
            win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
            total_trades=0, volatility=0, beta=None, alpha=None,
            information_ratio=None, var_95=0, cvar_95=0, skewness=0, kurtosis=0
        )


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Daily returns series
        window: Rolling window size in days

    Returns:
        DataFrame with rolling metrics
    """
    rolling_return = returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )

    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    rolling_sharpe = (
        returns.rolling(window).mean() * 252 /
        (returns.rolling(window).std() * np.sqrt(252))
    )

    # Rolling max drawdown
    def rolling_max_dd(x):
        cum = (1 + x).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        return dd.min()

    rolling_dd = returns.rolling(window).apply(rolling_max_dd, raw=False)

    return pd.DataFrame({
        "Rolling Return": rolling_return,
        "Rolling Volatility": rolling_vol,
        "Rolling Sharpe": rolling_sharpe,
        "Rolling Max DD": rolling_dd
    })


if __name__ == "__main__":
    # Test the metrics calculator
    np.random.seed(42)

    # Generate sample returns
    n_days = 252 * 3  # 3 years
    daily_returns = pd.Series(
        np.random.normal(0.0003, 0.015, n_days),
        index=pd.date_range("2020-01-01", periods=n_days, freq="B")
    )

    # Calculate metrics
    calc = MetricsCalculator()
    metrics = calc.calculate_all(daily_returns)

    print(metrics.summary())
    print("\nAll metrics:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
