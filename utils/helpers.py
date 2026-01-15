"""
Helper utility functions for the Quantitative Trading Strategy Backtester.
"""

from datetime import datetime
from typing import Tuple, Optional
import pandas as pd


def validate_date_range(
    start_date: str,
    end_date: str,
    date_format: str = "%Y-%m-%d"
) -> Tuple[datetime, datetime]:
    """
    Validate and parse date range strings.

    Args:
        start_date: Start date string
        end_date: End date string
        date_format: Expected date format

    Returns:
        Tuple of (start_datetime, end_datetime)

    Raises:
        ValueError: If dates are invalid or start > end
    """
    try:
        start = datetime.strptime(start_date, date_format)
        end = datetime.strptime(end_date, date_format)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected {date_format}: {e}")

    if start >= end:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

    return start, end


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a number as currency.

    Args:
        value: Numeric value
        decimals: Decimal places

    Returns:
        Formatted string (e.g., "$1,234.56")
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2, multiply: bool = True) -> str:
    """
    Format a number as percentage.

    Args:
        value: Numeric value
        decimals: Decimal places
        multiply: If True, multiply by 100 (for values like 0.15)

    Returns:
        Formatted string (e.g., "15.00%")
    """
    if multiply:
        value = value * 100
    return f"{value:.{decimals}f}%"


def calculate_trading_days(start: datetime, end: datetime) -> int:
    """
    Estimate number of trading days between dates.

    Args:
        start: Start date
        end: End date

    Returns:
        Estimated trading days
    """
    calendar_days = (end - start).days
    # Approximate: 252 trading days per 365 calendar days
    return int(calendar_days * 252 / 365)


def resample_to_frequency(
    data: pd.DataFrame,
    frequency: str = "W"
) -> pd.DataFrame:
    """
    Resample price data to different frequency.

    Args:
        data: Price DataFrame with datetime index
        frequency: Pandas frequency string ('D', 'W', 'M', 'Q', 'Y')

    Returns:
        Resampled DataFrame with OHLC aggregation
    """
    if frequency == "D":
        return data

    return data.resample(frequency).agg({
        col: "last" for col in data.columns
    })


def align_dataframes(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Align multiple DataFrames to common index (intersection of dates).

    Args:
        *dfs: Variable number of DataFrames

    Returns:
        Tuple of aligned DataFrames
    """
    if not dfs:
        return ()

    # Find common index
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    return tuple(df.loc[common_index] for df in dfs)
