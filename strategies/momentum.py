"""
Momentum Strategy Implementation.

This strategy buys assets with the highest recent returns, based on the
momentum anomaly where past winners tend to continue outperforming.

Key Parameters:
- lookback_period: Period to measure momentum (default: 20 days)
- holding_period: How long to hold positions (default: 5 days)
- top_n: Number of top performers to buy (default: 5)
- threshold: Minimum return threshold to consider (default: 0%)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base import Strategy, Signal
from config import MOMENTUM_PARAMS

logger = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    """
    Cross-sectional momentum strategy.

    Ranks assets by recent performance and goes long the top performers.
    Can optionally short the worst performers for a long-short version.
    """

    def __init__(
        self,
        lookback_period: int = MOMENTUM_PARAMS["lookback_period"],
        holding_period: int = MOMENTUM_PARAMS["holding_period"],
        top_n: int = MOMENTUM_PARAMS["top_n"],
        threshold: float = MOMENTUM_PARAMS["threshold"],
        long_short: bool = False,
        use_volatility_adjustment: bool = True,
        skip_recent_days: int = 1  # Skip most recent day to avoid reversal
    ):
        """
        Initialize the Momentum Strategy.

        Args:
            lookback_period: Days to calculate momentum
            holding_period: Days to hold positions before rebalancing
            top_n: Number of top assets to go long
            threshold: Minimum momentum to consider (filter weak signals)
            long_short: If True, also short bottom performers
            use_volatility_adjustment: Adjust momentum by volatility
            skip_recent_days: Skip recent days (helps avoid short-term reversal)
        """
        params = {
            "lookback_period": lookback_period,
            "holding_period": holding_period,
            "top_n": top_n,
            "threshold": threshold,
            "long_short": long_short,
            "use_volatility_adjustment": use_volatility_adjustment,
            "skip_recent_days": skip_recent_days
        }
        super().__init__(name="Momentum", params=params)

        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.threshold = threshold
        self.long_short = long_short
        self.use_volatility_adjustment = use_volatility_adjustment
        self.skip_recent_days = skip_recent_days

        self._last_rebalance_date: Optional[pd.Timestamp] = None
        self._days_since_rebalance: int = 0

    def get_required_history(self) -> int:
        """Return required historical days."""
        return self.lookback_period + self.skip_recent_days + 5

    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        positions: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate momentum signals.

        Args:
            prices: Historical price data (dates as index, tickers as columns)
            current_date: Current simulation date
            positions: Current portfolio positions

        Returns:
            List of trading signals
        """
        required_history = self.get_required_history()
        if not self.validate_data(prices, required_history):
            return []

        # Get data up to current date
        historical = prices.loc[:current_date].tail(required_history)

        # Check if it's time to rebalance
        if not self._should_rebalance(current_date):
            self._days_since_rebalance += 1
            return []

        signals = []

        # Calculate momentum scores
        momentum_scores = self._calculate_momentum(historical)

        if momentum_scores.empty:
            return []

        # Filter by threshold
        valid_scores = momentum_scores[momentum_scores > self.threshold]

        if len(valid_scores) == 0:
            logger.debug(f"No assets above threshold {self.threshold}")
            return []

        # Select top performers
        top_performers = valid_scores.nlargest(min(self.top_n, len(valid_scores)))

        # Calculate signal strength based on rank and score
        max_score = top_performers.max()
        min_score = top_performers.min()
        score_range = max_score - min_score if max_score != min_score else 1

        for ticker, score in top_performers.items():
            # Normalize strength between 0.5 and 1.0
            normalized_strength = 0.5 + 0.5 * (score - min_score) / score_range

            signal = Signal(
                ticker=ticker,
                direction=1,  # Long
                strength=min(normalized_strength, 1.0),
                timestamp=current_date,
                metadata={
                    "momentum_score": score,
                    "rank": list(top_performers.index).index(ticker) + 1,
                    "strategy": "momentum"
                }
            )
            signals.append(signal)
            self.record_signal(signal)

        # Long-short: also short bottom performers
        if self.long_short and len(momentum_scores) >= 2 * self.top_n:
            bottom_performers = momentum_scores.nsmallest(self.top_n)

            for ticker, score in bottom_performers.items():
                # Avoid shorting assets we're going long
                if ticker in top_performers.index:
                    continue

                signal = Signal(
                    ticker=ticker,
                    direction=-1,  # Short
                    strength=0.5,  # Equal weight for shorts
                    timestamp=current_date,
                    metadata={
                        "momentum_score": score,
                        "strategy": "momentum_short"
                    }
                )
                signals.append(signal)
                self.record_signal(signal)

        # Update rebalance tracking
        self._last_rebalance_date = current_date
        self._days_since_rebalance = 0

        logger.info(f"Momentum: Generated {len(signals)} signals on {current_date.date()}")

        return signals

    def _should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Check if portfolio should be rebalanced."""
        if self._last_rebalance_date is None:
            return True

        return self._days_since_rebalance >= self.holding_period

    def _calculate_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for all assets.

        Args:
            prices: Historical prices

        Returns:
            Series with momentum scores indexed by ticker
        """
        # Skip most recent day(s) to avoid short-term reversal
        if self.skip_recent_days > 0:
            prices_for_calc = prices.iloc[:-self.skip_recent_days]
        else:
            prices_for_calc = prices

        if len(prices_for_calc) < self.lookback_period:
            return pd.Series(dtype=float)

        # Calculate returns over lookback period
        start_prices = prices_for_calc.iloc[-self.lookback_period]
        end_prices = prices_for_calc.iloc[-1]

        # Simple momentum: total return over period
        momentum = (end_prices / start_prices) - 1

        # Remove NaN and infinite values
        momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()

        # Volatility adjustment (Sharpe-like momentum)
        if self.use_volatility_adjustment:
            returns = prices_for_calc.pct_change()
            volatility = returns.iloc[-self.lookback_period:].std()
            volatility = volatility.replace(0, np.nan)

            # Risk-adjusted momentum
            momentum = momentum / volatility
            momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()

        return momentum

    def get_momentum_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum scores over time for analysis.

        Args:
            prices: Full price history

        Returns:
            DataFrame with momentum scores over time
        """
        scores_over_time = {}

        for i in range(self.lookback_period, len(prices)):
            window = prices.iloc[i-self.lookback_period:i]
            momentum = self._calculate_momentum(window)
            scores_over_time[prices.index[i]] = momentum

        return pd.DataFrame(scores_over_time).T


class DualMomentumStrategy(MomentumStrategy):
    """
    Dual Momentum Strategy combining absolute and relative momentum.

    Only invests in top relative performers if they also have positive
    absolute momentum (i.e., outperforming a risk-free benchmark).
    """

    def __init__(
        self,
        lookback_period: int = 252,  # 12-month momentum typical
        top_n: int = 3,
        risk_free_return: float = 0.02,  # Annual risk-free rate
        **kwargs
    ):
        """
        Initialize Dual Momentum Strategy.

        Args:
            lookback_period: Period for momentum calculation
            top_n: Number of assets to hold
            risk_free_return: Annual risk-free rate for absolute momentum filter
        """
        super().__init__(
            lookback_period=lookback_period,
            top_n=top_n,
            **kwargs
        )
        self.name = "DualMomentum"
        self.risk_free_return = risk_free_return
        self.params["risk_free_return"] = risk_free_return

    def _calculate_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate momentum with absolute momentum filter."""
        # Get relative momentum scores
        momentum = super()._calculate_momentum(prices)

        if momentum.empty:
            return momentum

        # Calculate absolute return threshold
        daily_rf = (1 + self.risk_free_return) ** (1/252) - 1
        period_rf = (1 + daily_rf) ** self.lookback_period - 1

        # Calculate raw returns for absolute momentum check
        if self.skip_recent_days > 0:
            prices_for_calc = prices.iloc[:-self.skip_recent_days]
        else:
            prices_for_calc = prices

        start_prices = prices_for_calc.iloc[-self.lookback_period]
        end_prices = prices_for_calc.iloc[-1]
        raw_returns = (end_prices / start_prices) - 1

        # Filter: only keep assets with positive absolute momentum
        positive_momentum = raw_returns > period_rf

        return momentum[positive_momentum]
