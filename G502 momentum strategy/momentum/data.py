"""
Data loading, cleaning, and return computation for G502 momentum strategy.

This module handles:
- Loading price data from CSV or yfinance
- Data cleaning and validation
- Log return computation
- Stale price detection and handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings


def load_data(
    csv_path: Optional[str] = None,
    ticker: Optional[str] = None,
    date_column: str = "Date",
    price_column: str = "G5O2",
    date_format: str = "%Y%m%d",
    max_ffill_days: int = 1,
) -> pd.DataFrame:
    """
    Load and clean G502 price data from CSV file or yfinance.

    Parameters
    ----------
    csv_path : str, optional
        Path to CSV file containing price data.
    ticker : str, optional
        Yahoo Finance ticker symbol (used if csv_path not provided).
    date_column : str
        Name of the date column in CSV.
    price_column : str
        Name of the price column in CSV.
    date_format : str
        Format string for parsing dates.
    max_ffill_days : int
        Maximum consecutive days to forward-fill stale prices.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and 'price' column.

    Raises
    ------
    ValueError
        If neither csv_path nor ticker is provided, or if data validation fails.
    """
    if csv_path is not None:
        df = _load_from_csv(csv_path, date_column, price_column, date_format)
    elif ticker is not None:
        df = _load_from_yfinance(ticker)
    else:
        raise ValueError("Either csv_path or ticker must be provided")

    # Validate and clean data
    df = _validate_and_clean(df, max_ffill_days)

    return df


def _load_from_csv(
    csv_path: str,
    date_column: str,
    price_column: str,
    date_format: str,
) -> pd.DataFrame:
    """Load price data from CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path)

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in CSV")
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in CSV")

    # Parse dates
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    df = df.set_index(date_column)
    df.index.name = "date"

    # Keep only price column and rename
    df = df[[price_column]].rename(columns={price_column: "price"})

    # Ensure price is numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def _load_from_yfinance(ticker: str) -> pd.DataFrame:
    """Load price data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for downloading data. Install with: pip install yfinance")

    data = yf.download(ticker, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Use adjusted close if available, otherwise close
    if "Adj Close" in data.columns:
        price_col = "Adj Close"
    elif "Close" in data.columns:
        price_col = "Close"
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in yfinance data")

    df = data[[price_col]].rename(columns={price_col: "price"})
    df.index.name = "date"

    return df


def _validate_and_clean(df: pd.DataFrame, max_ffill_days: int) -> pd.DataFrame:
    """
    Validate data integrity and handle missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw price DataFrame with DatetimeIndex.
    max_ffill_days : int
        Maximum days to forward-fill.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with stale prices flagged and dropped.
    """
    # Sort by date
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Check for NaN prices
    nan_count = df["price"].isna().sum()
    if nan_count > 0:
        warnings.warn(f"Found {nan_count} NaN prices, will be handled by forward fill")

    # Check for non-positive prices
    invalid_prices = (df["price"] <= 0).sum()
    if invalid_prices > 0:
        warnings.warn(f"Found {invalid_prices} non-positive prices, setting to NaN")
        df.loc[df["price"] <= 0, "price"] = np.nan

    # Detect stale prices (consecutive duplicate values)
    df["is_stale"] = _detect_stale_prices(df["price"], max_ffill_days)

    # Forward fill up to max_ffill_days
    if max_ffill_days > 0:
        df["price"] = df["price"].ffill(limit=max_ffill_days)

    # Drop rows with stale prices (beyond ffill limit) or remaining NaN
    rows_before = len(df)
    df = df.dropna(subset=["price"])

    # Also drop rows flagged as stale beyond threshold
    df = df[~df["is_stale"]]
    df = df.drop(columns=["is_stale"])

    rows_after = len(df)
    if rows_before != rows_after:
        warnings.warn(f"Dropped {rows_before - rows_after} rows due to missing/stale data")

    return df


def _detect_stale_prices(prices: pd.Series, max_days: int) -> pd.Series:
    """
    Detect prices that have been unchanged for more than max_days.

    Returns a boolean Series where True indicates a stale price.
    """
    if max_days <= 0:
        return pd.Series(False, index=prices.index)

    # Find consecutive equal values
    is_duplicate = prices == prices.shift(1)

    # Count consecutive duplicates
    # Shape: (n_days,) boolean series
    stale = pd.Series(False, index=prices.index)
    consecutive_count = 0

    for i, (idx, is_dup) in enumerate(is_duplicate.items()):
        if is_dup:
            consecutive_count += 1
            if consecutive_count > max_days:
                stale.loc[idx] = True
        else:
            consecutive_count = 0

    return stale


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price data.

    Log returns: r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with 'price' column and DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'price' and 'returns' columns.
        Shape: (n_days, 2)

    Notes
    -----
    - First row will have NaN return (no prior price).
    - Uses natural log for return calculation.
    """
    df = prices.copy()

    # Log returns: r_t = ln(P_t / P_{t-1})
    # Shape: (n_days,)
    df["returns"] = np.log(df["price"] / df["price"].shift(1))

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for loaded data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and returns data.

    Returns
    -------
    dict
        Dictionary containing data summary statistics.
    """
    summary = {
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "n_observations": len(df),
        "n_years": (df.index.max() - df.index.min()).days / 365.25,
        "price_min": df["price"].min(),
        "price_max": df["price"].max(),
        "price_mean": df["price"].mean(),
    }

    if "returns" in df.columns:
        returns = df["returns"].dropna()
        summary.update({
            "return_mean_daily": returns.mean(),
            "return_std_daily": returns.std(),
            "return_mean_annual": returns.mean() * 252,
            "return_std_annual": returns.std() * np.sqrt(252),
        })

    return summary
