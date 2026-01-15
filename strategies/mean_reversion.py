"""
Mean Reversion Strategy Implementation.

This strategy buys assets that have deviated significantly below their
moving average, expecting them to revert to the mean.

Key Parameters:
- lookback_period: Period for moving average calculation (default: 20 days)
- entry_zscore: Z-score threshold for entry (default: -2.0)
- exit_zscore: Z-score threshold for exit (default: 0.0)
- bollinger_std: Standard deviations for Bollinger Bands (default: 2.0)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base import Strategy, Signal
from config import MEAN_REVERSION_PARAMS

logger = logging.getLogger(__name__)


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy using Bollinger Bands and Z-Scores.

    Enters long positions when price falls below lower band (oversold),
    and exits when price returns to the mean.
    """

    def __init__(
        self,
        lookback_period: int = MEAN_REVERSION_PARAMS["lookback_period"],
        entry_zscore: float = MEAN_REVERSION_PARAMS["entry_zscore"],
        exit_zscore: float = MEAN_REVERSION_PARAMS["exit_zscore"],
        bollinger_std: float = MEAN_REVERSION_PARAMS["bollinger_std"],
        use_rsi_filter: bool = True,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        max_positions: int = 5
    ):
        """
        Initialize the Mean Reversion Strategy.

        Args:
            lookback_period: Period for MA and std calculation
            entry_zscore: Z-score threshold to enter (negative = oversold)
            exit_zscore: Z-score threshold to exit position
            bollinger_std: Number of std devs for Bollinger Bands
            use_rsi_filter: Also check RSI for confirmation
            rsi_oversold: RSI level considered oversold
            rsi_overbought: RSI level considered overbought
            max_positions: Maximum concurrent positions
        """
        params = {
            "lookback_period": lookback_period,
            "entry_zscore": entry_zscore,
            "exit_zscore": exit_zscore,
            "bollinger_std": bollinger_std,
            "use_rsi_filter": use_rsi_filter,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "max_positions": max_positions
        }
        super().__init__(name="MeanReversion", params=params)

        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.bollinger_std = bollinger_std
        self.use_rsi_filter = use_rsi_filter
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_positions = max_positions

        self._active_positions: Dict[str, float] = {}  # ticker -> entry zscore

    def get_required_history(self) -> int:
        """Return required historical days."""
        # Need extra for RSI calculation (typically 14 days)
        return self.lookback_period + 20

    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        positions: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate mean reversion signals.

        Args:
            prices: Historical price data
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

        signals = []

        # Track current positions from input
        current_positions = set()
        if positions:
            current_positions = set(positions.keys())

        # Calculate indicators for all assets
        for ticker in prices.columns:
            if ticker not in historical.columns:
                continue

            ticker_prices = historical[ticker].dropna()
            if len(ticker_prices) < self.lookback_period:
                continue

            signal = self._analyze_ticker(
                ticker,
                ticker_prices,
                current_date,
                ticker in current_positions
            )

            if signal:
                signals.append(signal)
                self.record_signal(signal)

        # Limit new entries if at max positions
        entry_signals = [s for s in signals if s.direction == 1]
        exit_signals = [s for s in signals if s.direction == 0]

        new_positions_allowed = self.max_positions - len(current_positions) + len(exit_signals)

        if len(entry_signals) > new_positions_allowed:
            # Sort by signal strength and take strongest
            entry_signals.sort(key=lambda s: s.strength, reverse=True)
            entry_signals = entry_signals[:max(0, new_positions_allowed)]

        final_signals = entry_signals + exit_signals

        if final_signals:
            logger.info(f"MeanReversion: Generated {len(final_signals)} signals on {current_date.date()}")

        return final_signals

    def _analyze_ticker(
        self,
        ticker: str,
        prices: pd.Series,
        current_date: pd.Timestamp,
        has_position: bool
    ) -> Optional[Signal]:
        """
        Analyze a single ticker for entry/exit signals.

        Args:
            ticker: Asset ticker
            prices: Price series
            current_date: Current date
            has_position: Whether we have an existing position

        Returns:
            Signal if action needed, None otherwise
        """
        # Calculate z-score
        ma = prices.rolling(window=self.lookback_period).mean()
        std = prices.rolling(window=self.lookback_period).std()

        current_price = prices.iloc[-1]
        current_ma = ma.iloc[-1]
        current_std = std.iloc[-1]

        if current_std == 0 or pd.isna(current_std):
            return None

        zscore = (current_price - current_ma) / current_std

        # Calculate RSI if using filter
        rsi = None
        if self.use_rsi_filter:
            rsi = self._calculate_rsi(prices)

        # Check for exit signal if we have a position
        if has_position:
            if zscore >= self.exit_zscore:
                # Mean reversion complete, exit
                return Signal(
                    ticker=ticker,
                    direction=0,  # Exit
                    strength=1.0,
                    timestamp=current_date,
                    metadata={
                        "zscore": zscore,
                        "rsi": rsi,
                        "reason": "mean_reversion_exit",
                        "strategy": "mean_reversion"
                    }
                )
            # Also exit if severely overbought
            if rsi and rsi > self.rsi_overbought:
                return Signal(
                    ticker=ticker,
                    direction=0,
                    strength=1.0,
                    timestamp=current_date,
                    metadata={
                        "zscore": zscore,
                        "rsi": rsi,
                        "reason": "rsi_overbought_exit",
                        "strategy": "mean_reversion"
                    }
                )

        # Check for entry signal if no position
        if not has_position:
            if zscore <= self.entry_zscore:
                # RSI confirmation if enabled
                if self.use_rsi_filter and rsi is not None:
                    if rsi > self.rsi_oversold:
                        # RSI not confirming oversold condition
                        return None

                # Calculate signal strength based on how oversold
                # More oversold = stronger signal
                strength = min(1.0, abs(zscore - self.entry_zscore) / 2 + 0.5)

                return Signal(
                    ticker=ticker,
                    direction=1,  # Long entry
                    strength=strength,
                    timestamp=current_date,
                    metadata={
                        "zscore": zscore,
                        "rsi": rsi,
                        "ma": current_ma,
                        "std": current_std,
                        "reason": "oversold_entry",
                        "strategy": "mean_reversion"
                    }
                )

        return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            prices: Price series
            period: RSI period

        Returns:
            Current RSI value
        """
        if len(prices) < period + 1:
            return None

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate Bollinger Bands for all assets.

        Args:
            prices: Price DataFrame

        Returns:
            Dict with 'upper', 'middle', 'lower' band DataFrames
        """
        middle = prices.rolling(window=self.lookback_period).mean()
        std = prices.rolling(window=self.lookback_period).std()

        upper = middle + (self.bollinger_std * std)
        lower = middle - (self.bollinger_std * std)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "percent_b": (prices - lower) / (upper - lower)
        }


class StatisticalArbitrageStrategy(MeanReversionStrategy):
    """
    Statistical Arbitrage variant using Ornstein-Uhlenbeck process.

    Uses half-life of mean reversion to time entries more precisely.
    """

    def __init__(
        self,
        lookback_period: int = 60,
        entry_zscore: float = -2.0,
        exit_zscore: float = 0.0,
        min_half_life: int = 5,
        max_half_life: int = 60,
        **kwargs
    ):
        """
        Initialize Statistical Arbitrage Strategy.

        Args:
            lookback_period: Period for calculations
            entry_zscore: Entry threshold
            exit_zscore: Exit threshold
            min_half_life: Minimum acceptable half-life (days)
            max_half_life: Maximum acceptable half-life (days)
        """
        super().__init__(
            lookback_period=lookback_period,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            **kwargs
        )
        self.name = "StatArb"
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.params.update({
            "min_half_life": min_half_life,
            "max_half_life": max_half_life
        })

    def _calculate_half_life(self, prices: pd.Series) -> Optional[float]:
        """
        Calculate half-life of mean reversion using OLS regression.

        Models price as AR(1) process: P(t) = alpha + beta * P(t-1) + epsilon
        Half-life = -log(2) / log(beta)
        """
        if len(prices) < 20:
            return None

        lagged = prices.shift(1)
        delta = prices - lagged

        # Remove NaN
        valid_idx = ~(lagged.isna() | delta.isna())
        lagged = lagged[valid_idx]
        delta = delta[valid_idx]

        if len(lagged) < 10:
            return None

        # OLS regression: delta = alpha + beta * lagged
        X = np.column_stack([np.ones(len(lagged)), lagged])
        y = delta.values

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0][1]

            if beta >= 0:
                return None  # Not mean-reverting

            half_life = -np.log(2) / beta
            return half_life

        except np.linalg.LinAlgError:
            return None

    def _analyze_ticker(
        self,
        ticker: str,
        prices: pd.Series,
        current_date: pd.Timestamp,
        has_position: bool
    ) -> Optional[Signal]:
        """Analyze ticker with half-life filter."""
        # Check half-life constraint
        half_life = self._calculate_half_life(prices)

        if half_life is None:
            return None

        if not (self.min_half_life <= half_life <= self.max_half_life):
            return None

        # Call parent analysis
        signal = super()._analyze_ticker(ticker, prices, current_date, has_position)

        if signal and signal.metadata:
            signal.metadata["half_life"] = half_life

        return signal
