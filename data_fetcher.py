"""
Data Fetching Module for the Quantitative Trading Strategy Backtester.
Handles downloading and caching historical market data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DATA_FREQUENCY,
    STOCK_TICKERS, ETF_TICKERS, CRYPTO_TICKERS, ALL_TICKERS
)

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches and manages historical market data from Yahoo Finance.
    Supports stocks, ETFs, and cryptocurrencies with caching capability.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the DataFetcher.

        Args:
            cache_dir: Directory for caching downloaded data. If None, no caching.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_cache: Dict[str, pd.DataFrame] = {}

    def fetch_single_ticker(
        self,
        ticker: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker.

        Args:
            ticker: Stock/ETF/Crypto ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"

        # Check memory cache
        if use_cache and cache_key in self._data_cache:
            logger.debug(f"Using memory cache for {ticker}")
            return self._data_cache[cache_key].copy()

        # Check file cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                logger.debug(f"Loading {ticker} from file cache")
                df = pd.read_parquet(cache_file)
                self._data_cache[cache_key] = df
                return df.copy()

        # Fetch from Yahoo Finance
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, interval=DATA_FREQUENCY)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Clean and standardize columns
            df = self._clean_data(df, ticker)

            # Cache the data
            self._data_cache[cache_key] = df
            if self.cache_dir:
                df.to_parquet(self.cache_dir / f"{cache_key}.parquet")

            return df.copy()

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        price_type: str = "Close",
        use_cache: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers and return a combined DataFrame.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            price_type: Which price to use (Open, High, Low, Close, Adj Close)
            use_cache: Whether to use cached data
            show_progress: Whether to show download progress

        Returns:
            DataFrame with tickers as columns and dates as index
        """
        from tqdm import tqdm

        price_data = {}
        iterator = tqdm(tickers, desc="Fetching data") if show_progress else tickers

        for ticker in iterator:
            df = self.fetch_single_ticker(ticker, start_date, end_date, use_cache)
            if not df.empty and price_type in df.columns:
                price_data[ticker] = df[price_type]

        if not price_data:
            logger.warning("No data fetched for any ticker")
            return pd.DataFrame()

        # Combine into single DataFrame
        combined = pd.DataFrame(price_data)
        combined = combined.sort_index()

        # Forward fill small gaps (up to 5 days for holidays)
        combined = combined.ffill(limit=5)

        logger.info(f"Fetched data for {len(combined.columns)} tickers, "
                   f"{len(combined)} trading days")

        return combined

    def fetch_ohlcv(
        self,
        tickers: List[str],
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch full OHLCV data for multiple tickers.

        Returns:
            Dictionary mapping ticker to full OHLCV DataFrame
        """
        from tqdm import tqdm

        data = {}
        for ticker in tqdm(tickers, desc="Fetching OHLCV"):
            df = self.fetch_single_ticker(ticker, start_date, end_date, use_cache)
            if not df.empty:
                data[ticker] = df

        return data

    def _clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Clean and standardize fetched data.

        Args:
            df: Raw DataFrame from yfinance
            ticker: Ticker symbol for logging

        Returns:
            Cleaned DataFrame
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Rename columns to standard format
        column_map = {
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume',
            'Dividends': 'Dividends',
            'Stock Splits': 'Stock_Splits'
        }
        df = df.rename(columns=column_map)

        # Drop rows with all NaN prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        existing_price_cols = [c for c in price_cols if c in df.columns]
        if existing_price_cols:
            df = df.dropna(subset=existing_price_cols, how='all')

        # Handle zero or negative prices
        for col in existing_price_cols:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan

        # Forward fill remaining NaN values
        df = df.ffill()

        # Remove duplicate index values
        df = df[~df.index.duplicated(keep='first')]

        # Sort by date
        df = df.sort_index()

        # Add returns
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        logger.debug(f"Cleaned {ticker}: {len(df)} rows, columns: {list(df.columns)}")

        return df

    def get_returns(
        self,
        tickers: List[str],
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        log_returns: bool = False
    ) -> pd.DataFrame:
        """
        Get returns matrix for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            log_returns: If True, return log returns; otherwise simple returns

        Returns:
            DataFrame with daily returns
        """
        prices = self.fetch_multiple_tickers(tickers, start_date, end_date)

        if prices.empty:
            return pd.DataFrame()

        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()

    def get_available_tickers(self) -> Dict[str, List[str]]:
        """
        Get available ticker lists by asset class.

        Returns:
            Dictionary with asset classes as keys and ticker lists as values
        """
        return {
            "stocks": STOCK_TICKERS,
            "etfs": ETF_TICKERS,
            "crypto": CRYPTO_TICKERS,
            "all": ALL_TICKERS
        }

    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """
        Validate tickers by attempting to fetch minimal data.

        Returns:
            List of valid tickers that returned data
        """
        valid = []
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
                if not df.empty:
                    valid.append(ticker)
            except Exception:
                pass

        return valid

    def clear_cache(self):
        """Clear all cached data."""
        self._data_cache.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
        logger.info("Cache cleared")


def get_sample_data(n_assets: int = 10, years: int = 5) -> pd.DataFrame:
    """
    Convenience function to get sample data for testing.

    Args:
        n_assets: Number of assets to fetch
        years: Years of history

    Returns:
        DataFrame with closing prices
    """
    fetcher = DataFetcher()
    tickers = ALL_TICKERS[:n_assets]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*years)).strftime("%Y-%m-%d")

    return fetcher.fetch_multiple_tickers(tickers, start_date, end_date)


if __name__ == "__main__":
    # Test the data fetcher
    logging.basicConfig(level=logging.INFO)

    print("Testing DataFetcher...")
    fetcher = DataFetcher(cache_dir="./data_cache")

    # Test single ticker
    aapl = fetcher.fetch_single_ticker("AAPL")
    print(f"\nAAPL data shape: {aapl.shape}")
    print(aapl.tail())

    # Test multiple tickers
    tickers = ["AAPL", "MSFT", "SPY", "BTC-USD"]
    prices = fetcher.fetch_multiple_tickers(tickers)
    print(f"\nMultiple ticker data shape: {prices.shape}")
    print(prices.tail())

    # Test returns
    returns = fetcher.get_returns(tickers)
    print(f"\nReturns data shape: {returns.shape}")
    print(returns.tail())
