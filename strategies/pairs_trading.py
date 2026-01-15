"""
Pairs Trading Strategy Implementation.

This strategy identifies pairs of correlated assets and trades the spread
between them when it deviates from its historical mean.

Key Parameters:
- lookback_period: Period for cointegration analysis (default: 60 days)
- entry_zscore: Z-score threshold for entry (default: 2.0)
- exit_zscore: Z-score threshold for exit (default: 0.5)
- correlation_threshold: Minimum correlation required (default: 0.7)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from itertools import combinations
from dataclasses import dataclass
import logging
from scipy import stats

from .base import Strategy, Signal
from config import PAIRS_TRADING_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class TradingPair:
    """Represents a cointegrated trading pair."""
    ticker_a: str
    ticker_b: str
    correlation: float
    hedge_ratio: float
    half_life: Optional[float]
    adf_pvalue: float
    spread_mean: float
    spread_std: float


class PairsTradingStrategy(Strategy):
    """
    Pairs Trading / Statistical Arbitrage Strategy.

    Identifies cointegrated pairs and trades mean reversion of the spread.
    """

    def __init__(
        self,
        lookback_period: int = PAIRS_TRADING_PARAMS["lookback_period"],
        entry_zscore: float = PAIRS_TRADING_PARAMS["entry_zscore"],
        exit_zscore: float = PAIRS_TRADING_PARAMS["exit_zscore"],
        correlation_threshold: float = PAIRS_TRADING_PARAMS["correlation_threshold"],
        adf_significance: float = 0.05,
        max_pairs: int = 5,
        predefined_pairs: Optional[List[Tuple[str, str]]] = None,
        recalculate_period: int = 20  # Recalculate pairs every N days
    ):
        """
        Initialize the Pairs Trading Strategy.

        Args:
            lookback_period: Period for cointegration analysis
            entry_zscore: Z-score threshold for entry (absolute value)
            exit_zscore: Z-score threshold for exit (absolute value)
            correlation_threshold: Minimum correlation for pair consideration
            adf_significance: ADF test p-value threshold
            max_pairs: Maximum number of pairs to trade
            predefined_pairs: List of predefined pairs to trade (skip discovery)
            recalculate_period: Days between pair recalculation
        """
        params = {
            "lookback_period": lookback_period,
            "entry_zscore": entry_zscore,
            "exit_zscore": exit_zscore,
            "correlation_threshold": correlation_threshold,
            "adf_significance": adf_significance,
            "max_pairs": max_pairs,
            "recalculate_period": recalculate_period
        }
        super().__init__(name="PairsTrading", params=params)

        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.correlation_threshold = correlation_threshold
        self.adf_significance = adf_significance
        self.max_pairs = max_pairs
        self.predefined_pairs = predefined_pairs
        self.recalculate_period = recalculate_period

        self._trading_pairs: List[TradingPair] = []
        self._active_trades: Dict[str, Dict[str, Any]] = {}  # pair_id -> trade info
        self._last_recalculation: Optional[pd.Timestamp] = None
        self._days_since_recalc: int = 0

    def get_required_history(self) -> int:
        """Return required historical days."""
        return self.lookback_period + 10

    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        positions: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate pairs trading signals.

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

        historical = prices.loc[:current_date].tail(required_history)

        # Recalculate pairs if needed
        if self._should_recalculate_pairs(current_date):
            self._find_trading_pairs(historical)
            self._last_recalculation = current_date
            self._days_since_recalc = 0
        else:
            self._days_since_recalc += 1

        if not self._trading_pairs:
            logger.debug("No valid trading pairs found")
            return []

        signals = []

        # Generate signals for each pair
        for pair in self._trading_pairs:
            pair_signals = self._analyze_pair(pair, historical, current_date, positions)
            signals.extend(pair_signals)

        if signals:
            logger.info(f"PairsTrading: Generated {len(signals)} signals on {current_date.date()}")

        return signals

    def _should_recalculate_pairs(self, current_date: pd.Timestamp) -> bool:
        """Check if pairs should be recalculated."""
        if self.predefined_pairs:
            # If using predefined pairs, only calculate once
            return len(self._trading_pairs) == 0

        if self._last_recalculation is None:
            return True

        return self._days_since_recalc >= self.recalculate_period

    def _find_trading_pairs(self, prices: pd.DataFrame):
        """
        Find cointegrated pairs from price data.

        Args:
            prices: Historical prices
        """
        if self.predefined_pairs:
            # Use predefined pairs
            for ticker_a, ticker_b in self.predefined_pairs:
                if ticker_a in prices.columns and ticker_b in prices.columns:
                    pair = self._analyze_pair_relationship(
                        prices[ticker_a], prices[ticker_b], ticker_a, ticker_b
                    )
                    if pair:
                        self._trading_pairs.append(pair)
            return

        # Discover pairs through correlation and cointegration
        tickers = prices.columns.tolist()
        candidates = []

        # First pass: correlation filter
        for ticker_a, ticker_b in combinations(tickers, 2):
            series_a = prices[ticker_a].dropna()
            series_b = prices[ticker_b].dropna()

            # Align series
            aligned = pd.concat([series_a, series_b], axis=1).dropna()
            if len(aligned) < self.lookback_period:
                continue

            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

            if abs(corr) >= self.correlation_threshold:
                candidates.append((ticker_a, ticker_b, corr))

        # Second pass: cointegration test
        valid_pairs = []
        for ticker_a, ticker_b, corr in candidates:
            pair = self._analyze_pair_relationship(
                prices[ticker_a], prices[ticker_b], ticker_a, ticker_b
            )
            if pair:
                valid_pairs.append(pair)

        # Sort by ADF p-value (lower is better) and take top pairs
        valid_pairs.sort(key=lambda p: p.adf_pvalue)
        self._trading_pairs = valid_pairs[:self.max_pairs]

        logger.info(f"Found {len(self._trading_pairs)} valid trading pairs")

    def _analyze_pair_relationship(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        ticker_a: str,
        ticker_b: str
    ) -> Optional[TradingPair]:
        """
        Analyze relationship between two assets.

        Tests for cointegration and calculates hedge ratio.
        """
        # Align series
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if len(aligned) < self.lookback_period:
            return None

        a = aligned.iloc[:, 0].values
        b = aligned.iloc[:, 1].values

        # Calculate hedge ratio using OLS
        hedge_ratio = self._calculate_hedge_ratio(a, b)

        # Calculate spread
        spread = a - hedge_ratio * b

        # Test for stationarity (cointegration)
        adf_result = self._adf_test(spread)

        if adf_result["pvalue"] > self.adf_significance:
            return None  # Not cointegrated

        # Calculate spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        # Calculate correlation
        correlation = np.corrcoef(a, b)[0, 1]

        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)

        return TradingPair(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            adf_pvalue=adf_result["pvalue"],
            spread_mean=spread_mean,
            spread_std=spread_std
        )

    def _calculate_hedge_ratio(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate optimal hedge ratio using OLS regression."""
        X = np.column_stack([np.ones(len(b)), b])
        result = np.linalg.lstsq(X, a, rcond=None)
        return result[0][1]

    def _adf_test(self, series: np.ndarray) -> Dict[str, float]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Simplified implementation without statsmodels dependency.
        """
        # Simple ADF test approximation using AR(1) regression
        n = len(series)
        if n < 20:
            return {"pvalue": 1.0, "statistic": 0}

        # First difference
        diff = np.diff(series)
        lagged = series[:-1]

        # Regression: diff = alpha + beta * lagged + error
        X = np.column_stack([np.ones(len(lagged)), lagged])
        result = np.linalg.lstsq(X, diff, rcond=None)
        beta = result[0][1]

        # Calculate t-statistic
        residuals = diff - X @ result[0]
        se = np.sqrt(np.sum(residuals**2) / (n - 3) / np.sum((lagged - lagged.mean())**2))

        if se == 0:
            return {"pvalue": 1.0, "statistic": 0}

        t_stat = beta / se

        # Approximate p-value using critical values
        # ADF critical values at 5%: approximately -2.86 for n=100
        if t_stat < -3.43:  # 1% significance
            pvalue = 0.01
        elif t_stat < -2.86:  # 5% significance
            pvalue = 0.05
        elif t_stat < -2.57:  # 10% significance
            pvalue = 0.10
        else:
            pvalue = 0.5  # Not significant

        return {"pvalue": pvalue, "statistic": t_stat}

    def _calculate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """Calculate half-life of mean reversion."""
        lagged = spread[:-1]
        diff = np.diff(spread)

        X = np.column_stack([np.ones(len(lagged)), lagged])
        result = np.linalg.lstsq(X, diff, rcond=None)
        beta = result[0][1]

        if beta >= 0:
            return None

        return -np.log(2) / beta

    def _analyze_pair(
        self,
        pair: TradingPair,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        positions: Optional[Dict[str, Any]]
    ) -> List[Signal]:
        """
        Generate signals for a specific pair.

        Args:
            pair: Trading pair info
            prices: Historical prices
            current_date: Current date
            positions: Current positions

        Returns:
            List of signals for this pair
        """
        signals = []

        # Get current prices
        if pair.ticker_a not in prices.columns or pair.ticker_b not in prices.columns:
            return signals

        series_a = prices[pair.ticker_a].dropna()
        series_b = prices[pair.ticker_b].dropna()

        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if len(aligned) < self.lookback_period:
            return signals

        # Calculate current spread
        current_a = aligned.iloc[-1, 0]
        current_b = aligned.iloc[-1, 1]
        current_spread = current_a - pair.hedge_ratio * current_b

        # Calculate z-score of spread
        recent_spreads = aligned.iloc[:, 0] - pair.hedge_ratio * aligned.iloc[:, 1]
        spread_mean = recent_spreads.rolling(self.lookback_period).mean().iloc[-1]
        spread_std = recent_spreads.rolling(self.lookback_period).std().iloc[-1]

        if spread_std == 0 or pd.isna(spread_std):
            return signals

        zscore = (current_spread - spread_mean) / spread_std

        pair_id = f"{pair.ticker_a}_{pair.ticker_b}"
        has_position = pair_id in self._active_trades

        # Check for exit
        if has_position:
            trade_info = self._active_trades[pair_id]
            if abs(zscore) <= self.exit_zscore:
                # Exit both legs
                signals.extend([
                    Signal(
                        ticker=pair.ticker_a,
                        direction=0,  # Exit
                        strength=1.0,
                        timestamp=current_date,
                        metadata={
                            "pair_id": pair_id,
                            "zscore": zscore,
                            "reason": "spread_converged",
                            "strategy": "pairs_trading"
                        }
                    ),
                    Signal(
                        ticker=pair.ticker_b,
                        direction=0,
                        strength=1.0,
                        timestamp=current_date,
                        metadata={
                            "pair_id": pair_id,
                            "zscore": zscore,
                            "reason": "spread_converged",
                            "strategy": "pairs_trading"
                        }
                    )
                ])
                del self._active_trades[pair_id]

        # Check for entry
        elif abs(zscore) >= self.entry_zscore:
            strength = min(1.0, (abs(zscore) - self.entry_zscore) / 2 + 0.5)

            if zscore > 0:
                # Spread is high: short A, long B
                signals.extend([
                    Signal(
                        ticker=pair.ticker_a,
                        direction=-1,
                        strength=strength,
                        timestamp=current_date,
                        metadata={
                            "pair_id": pair_id,
                            "zscore": zscore,
                            "hedge_ratio": pair.hedge_ratio,
                            "leg": "short",
                            "strategy": "pairs_trading"
                        }
                    ),
                    Signal(
                        ticker=pair.ticker_b,
                        direction=1,
                        strength=min(1.0, abs(strength * pair.hedge_ratio)),
                        timestamp=current_date,
                        metadata={
                            "pair_id": pair_id,
                            "zscore": zscore,
                            "hedge_ratio": pair.hedge_ratio,
                            "leg": "long",
                            "strategy": "pairs_trading"
                        }
                    )
                ])
            else:
                # Spread is low: long A, short B
                signals.extend([
                    Signal(
                        ticker=pair.ticker_a,
                        direction=1,
                        strength=strength,
                        timestamp=current_date,
                        metadata={
                            "pair_id": pair_id,
                            "zscore": zscore,
                            "hedge_ratio": pair.hedge_ratio,
                            "leg": "long",
                            "strategy": "pairs_trading"
                        }
                    ),
                    Signal(
                        ticker=pair.ticker_b,
                        direction=-1,
                        strength=min(1.0, abs(strength * pair.hedge_ratio)),
                        timestamp=current_date,
                        metadata={
                            "pair_id": pair_id,
                            "zscore": zscore,
                            "hedge_ratio": pair.hedge_ratio,
                            "leg": "short",
                            "strategy": "pairs_trading"
                        }
                    )
                ])

            # Track active trade
            self._active_trades[pair_id] = {
                "entry_zscore": zscore,
                "entry_date": current_date
            }

        for signal in signals:
            self.record_signal(signal)

        return signals

    def get_pair_spreads(self, prices: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate spread series for all trading pairs.

        Useful for visualization and analysis.
        """
        spreads = {}

        for pair in self._trading_pairs:
            if pair.ticker_a in prices.columns and pair.ticker_b in prices.columns:
                spread = prices[pair.ticker_a] - pair.hedge_ratio * prices[pair.ticker_b]
                pair_id = f"{pair.ticker_a}_{pair.ticker_b}"
                spreads[pair_id] = spread

        return spreads
