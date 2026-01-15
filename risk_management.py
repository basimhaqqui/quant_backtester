"""
Risk Management Module for the Quantitative Trading Strategy Backtester.
Implements stop-loss, take-profit, position sizing, and risk controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

from config import RISK_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    ticker: str
    entry_price: float
    entry_date: pd.Timestamp
    quantity: float
    direction: int  # 1 for long, -1 for short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def notional_value(self) -> float:
        """Calculate position notional value."""
        return abs(self.quantity * self.entry_price)

    def pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return self.direction * self.quantity * (current_price - self.entry_price)

    def pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        return self.direction * (current_price / self.entry_price - 1)


class RiskManager:
    """
    Manages risk controls for the backtesting system.
    Handles position sizing, stop-loss, take-profit, and portfolio constraints.
    """

    def __init__(
        self,
        stop_loss_pct: float = RISK_PARAMS["stop_loss_pct"],
        take_profit_pct: float = RISK_PARAMS["take_profit_pct"],
        max_position_size: float = RISK_PARAMS["max_position_size"],
        min_position_size: float = RISK_PARAMS["min_position_size"],
        use_volatility_sizing: bool = RISK_PARAMS["use_volatility_sizing"],
        target_volatility: float = RISK_PARAMS["target_volatility"],
        use_kelly_criterion: bool = RISK_PARAMS["use_kelly_criterion"],
        kelly_fraction: float = RISK_PARAMS["kelly_fraction"]
    ):
        """
        Initialize the RiskManager.

        Args:
            stop_loss_pct: Stop loss threshold (e.g., 0.05 for 5%)
            take_profit_pct: Take profit threshold (e.g., 0.15 for 15%)
            max_position_size: Maximum position as fraction of portfolio
            min_position_size: Minimum position as fraction of portfolio
            use_volatility_sizing: Scale positions by volatility
            target_volatility: Target annual volatility for vol sizing
            use_kelly_criterion: Use Kelly criterion for sizing
            kelly_fraction: Fractional Kelly multiplier (0-1)
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.use_volatility_sizing = use_volatility_sizing
        self.target_volatility = target_volatility
        self.use_kelly_criterion = use_kelly_criterion
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        ticker: str,
        signal_strength: float,
        portfolio_value: float,
        current_price: float,
        historical_returns: Optional[pd.Series] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            ticker: Asset ticker
            signal_strength: Strategy signal strength (0 to 1)
            portfolio_value: Current portfolio value
            current_price: Current asset price
            historical_returns: Historical returns for volatility calculation
            win_rate: Historical win rate for Kelly criterion
            avg_win_loss_ratio: Ratio of average win to average loss

        Returns:
            Position size as fraction of portfolio (0 to max_position_size)
        """
        # Base position size from signal
        base_size = signal_strength * self.max_position_size

        # Volatility-based sizing
        if self.use_volatility_sizing and historical_returns is not None:
            vol_size = self._volatility_position_size(historical_returns)
            base_size = min(base_size, vol_size)

        # Kelly criterion sizing
        if self.use_kelly_criterion and win_rate and avg_win_loss_ratio:
            kelly_size = self._kelly_position_size(win_rate, avg_win_loss_ratio)
            base_size = min(base_size, kelly_size)

        # Apply constraints
        position_size = np.clip(base_size, self.min_position_size, self.max_position_size)

        logger.debug(f"{ticker}: calculated position size = {position_size:.4f}")
        return position_size

    def _volatility_position_size(self, returns: pd.Series) -> float:
        """
        Calculate position size to target specific volatility.

        Uses inverse volatility scaling: allocate more to less volatile assets.
        """
        if len(returns) < 20:
            return self.max_position_size

        # Calculate annualized volatility
        vol = returns.std() * np.sqrt(252)

        if vol == 0:
            return self.max_position_size

        # Scale position inversely with volatility
        # If vol = target_vol, size = max_position_size
        # If vol > target_vol, size < max_position_size
        size = (self.target_volatility / vol) * self.max_position_size

        return min(size, self.max_position_size)

    def _kelly_position_size(
        self,
        win_rate: float,
        avg_win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly criterion position size.

        Kelly formula: f* = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio

        Args:
            win_rate: Probability of winning trade
            avg_win_loss_ratio: Average win / Average loss

        Returns:
            Kelly fraction position size
        """
        if avg_win_loss_ratio <= 0:
            return 0

        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio

        kelly = (p * b - q) / b

        # Apply fractional Kelly (more conservative)
        kelly = kelly * self.kelly_fraction

        # Constrain to valid range
        return np.clip(kelly, 0, self.max_position_size)

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop-loss price.

        Args:
            entry_price: Position entry price
            direction: 1 for long, -1 for short
            atr: Average True Range for volatility-based stop
            atr_multiplier: ATR multiplier for stop distance

        Returns:
            Stop-loss price
        """
        if atr is not None:
            # Volatility-based stop loss
            stop_distance = atr * atr_multiplier
        else:
            # Percentage-based stop loss
            stop_distance = entry_price * self.stop_loss_pct

        if direction == 1:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take-profit price.

        Args:
            entry_price: Position entry price
            direction: 1 for long, -1 for short
            risk_reward_ratio: Ratio of profit target to stop loss

        Returns:
            Take-profit price
        """
        profit_distance = entry_price * self.take_profit_pct

        if direction == 1:  # Long position
            return entry_price + profit_distance
        else:  # Short position
            return entry_price - profit_distance

    def check_exit_conditions(
        self,
        position: Position,
        current_price: float,
        current_date: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited based on risk rules.

        Args:
            position: Current position
            current_price: Current market price
            current_date: Current date

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        pnl_pct = position.pnl_pct(current_price)

        # Check stop loss
        if position.stop_loss:
            if position.direction == 1 and current_price <= position.stop_loss:
                return True, "stop_loss"
            elif position.direction == -1 and current_price >= position.stop_loss:
                return True, "stop_loss"

        # Check take profit
        if position.take_profit:
            if position.direction == 1 and current_price >= position.take_profit:
                return True, "take_profit"
            elif position.direction == -1 and current_price <= position.take_profit:
                return True, "take_profit"

        # Check percentage-based stop loss (fallback)
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss_pct"

        # Check percentage-based take profit (fallback)
        if pnl_pct >= self.take_profit_pct:
            return True, "take_profit_pct"

        return False, ""

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.

        Args:
            positions: Dictionary of current positions
            current_prices: Current prices for each asset
            returns_data: Historical returns DataFrame

        Returns:
            Dictionary with portfolio risk metrics
        """
        if not positions:
            return {"portfolio_var": 0, "portfolio_vol": 0, "concentration": 0}

        # Calculate position weights
        total_value = sum(
            pos.quantity * current_prices.get(ticker, pos.entry_price)
            for ticker, pos in positions.items()
        )

        if total_value == 0:
            return {"portfolio_var": 0, "portfolio_vol": 0, "concentration": 0}

        weights = {
            ticker: (pos.quantity * current_prices.get(ticker, pos.entry_price)) / total_value
            for ticker, pos in positions.items()
        }

        # Calculate concentration (Herfindahl index)
        concentration = sum(w ** 2 for w in weights.values())

        # Calculate portfolio volatility using covariance matrix
        tickers = list(positions.keys())
        available_tickers = [t for t in tickers if t in returns_data.columns]

        if not available_tickers:
            return {"portfolio_var": 0, "portfolio_vol": 0, "concentration": concentration}

        returns_subset = returns_data[available_tickers].dropna()
        if len(returns_subset) < 20:
            return {"portfolio_var": 0, "portfolio_vol": 0, "concentration": concentration}

        cov_matrix = returns_subset.cov() * 252  # Annualized
        weight_vector = np.array([weights.get(t, 0) for t in available_tickers])

        portfolio_var = weight_vector @ cov_matrix.values @ weight_vector
        portfolio_vol = np.sqrt(portfolio_var)

        return {
            "portfolio_var": portfolio_var,
            "portfolio_vol": portfolio_vol,
            "concentration": concentration
        }


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range for volatility-based position sizing.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_volatility_target_weights(
    returns: pd.DataFrame,
    target_vol: float = 0.15,
    lookback: int = 60
) -> pd.DataFrame:
    """
    Calculate volatility-targeted weights for multiple assets.

    Args:
        returns: Returns DataFrame with assets as columns
        target_vol: Target annualized volatility
        lookback: Lookback period for volatility estimation

    Returns:
        DataFrame with target weights for each asset over time
    """
    # Calculate rolling volatility
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)

    # Calculate inverse volatility weights
    inv_vol = 1 / rolling_vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Scale to target volatility
    portfolio_vol = (weights ** 2 * rolling_vol ** 2).sum(axis=1).apply(np.sqrt)
    scaling = target_vol / portfolio_vol.replace(0, np.nan)

    scaled_weights = weights.mul(scaling, axis=0)

    # Cap individual weights
    scaled_weights = scaled_weights.clip(upper=0.25)  # Max 25% per asset

    return scaled_weights


if __name__ == "__main__":
    # Test risk management
    logging.basicConfig(level=logging.DEBUG)

    rm = RiskManager()

    # Test position sizing
    size = rm.calculate_position_size(
        ticker="AAPL",
        signal_strength=0.8,
        portfolio_value=100000,
        current_price=150,
        historical_returns=pd.Series(np.random.normal(0.001, 0.02, 100)),
        win_rate=0.55,
        avg_win_loss_ratio=1.5
    )
    print(f"Calculated position size: {size:.4f}")

    # Test stop loss calculation
    stop = rm.calculate_stop_loss(entry_price=100, direction=1)
    print(f"Stop loss price: {stop:.2f}")

    # Test Kelly criterion
    kelly = rm._kelly_position_size(win_rate=0.55, avg_win_loss_ratio=1.5)
    print(f"Kelly position size: {kelly:.4f}")

    # Test position exit conditions
    pos = Position(
        ticker="AAPL",
        entry_price=100,
        entry_date=pd.Timestamp("2023-01-01"),
        quantity=10,
        direction=1,
        stop_loss=95,
        take_profit=115
    )

    should_exit, reason = rm.check_exit_conditions(pos, 94, pd.Timestamp("2023-01-15"))
    print(f"Should exit: {should_exit}, Reason: {reason}")
