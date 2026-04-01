"""
Momentum signal construction for G502 multi-horizon strategy.

This module handles:
- Raw momentum calculation with skip-1 convention
- Binary signal generation (long=1, flat=0)
- No look-ahead bias enforcement via assertions
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_momentum_signals(
    prices: pd.DataFrame,
    lookback_periods: List[int] = [10, 20, 60, 90, 120, 240],
    skip_days: int = 1,
) -> pd.DataFrame:
    """
    Compute binary momentum signals across multiple lookback horizons.

    Raw momentum at time t (with skip-1, no look-ahead):
        mom_L(t) = P_{t-1} / P_{t-1-L} - 1

    Binary signal rule:
        signal_L(t) = 1 if mom_L(t) > 0 else 0

    signal = 1 means long; signal = 0 means flat (no short).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with 'price' column and DatetimeIndex.
    lookback_periods : list of int
        Lookback periods L in trading days.
    skip_days : int
        Number of days to skip (typically 1) to avoid look-ahead.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary signal columns named 'signal_{L}' for each L.
        Shape: (n_days, n_horizons)
        All values are integers in {0, 1}.

    Raises
    ------
    AssertionError
        If look-ahead bias is detected.
    """
    df = prices.copy()

    # Verify price column exists
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain 'price' column")

    # Sort by date
    df = df.sort_index()

    for L in lookback_periods:
        # Compute raw momentum with skip-1 convention
        # mom_L(t) = P_{t-skip} / P_{t-skip-L} - 1
        # This uses prices from t-skip and t-skip-L, both strictly before t
        # Shape: (n_days,)
        momentum = _compute_raw_momentum(df["price"], L, skip_days)

        # Convert to binary signal
        # signal_L(t) = 1 if mom_L(t) > 0 else 0
        # Shape: (n_days,)
        signal = _momentum_to_binary_signal(momentum)

        # Store as nullable integer column (supports NaN)
        df[f"signal_{L}"] = signal.astype("Int64")

    # Verify no look-ahead bias
    _assert_no_lookahead(df, lookback_periods, skip_days)

    # Keep only signal columns
    signal_columns = [f"signal_{L}" for L in lookback_periods]
    signals_df = df[signal_columns].copy()

    # Verify binary output (checking non-null values)
    _assert_binary_signals(signals_df)

    return signals_df


def _compute_raw_momentum(
    prices: pd.Series,
    lookback: int,
    skip_days: int,
) -> pd.Series:
    """
    Compute raw momentum with skip convention.

    mom_L(t) = P_{t-skip} / P_{t-skip-L} - 1

    Parameters
    ----------
    prices : pd.Series
        Price series with DatetimeIndex.
    lookback : int
        Lookback period L in days.
    skip_days : int
        Number of days to skip (typically 1).

    Returns
    -------
    pd.Series
        Raw momentum values. NaN for periods with insufficient history.
    """
    # P_{t-skip}: price shifted by skip_days
    # Shape: (n_days,)
    p_recent = prices.shift(skip_days)

    # P_{t-skip-L}: price shifted by skip_days + lookback
    # Shape: (n_days,)
    p_past = prices.shift(skip_days + lookback)

    # Momentum: (P_{t-skip} / P_{t-skip-L}) - 1
    # Shape: (n_days,)
    momentum = (p_recent / p_past) - 1

    return momentum


def _momentum_to_binary_signal(momentum: pd.Series) -> pd.Series:
    """
    Convert raw momentum to binary signal.

    signal = 1 if momentum > 0 else 0
    NaN momentum results in NaN signal.

    Parameters
    ----------
    momentum : pd.Series
        Raw momentum values.

    Returns
    -------
    pd.Series
        Binary signal values {0, 1, NaN}.
    """
    signal = (momentum > 0).astype(float)

    # Preserve NaN from momentum
    signal[momentum.isna()] = np.nan

    return signal


def _assert_no_lookahead(
    df: pd.DataFrame,
    lookback_periods: List[int],
    skip_days: int,
) -> None:
    """
    Verify that no look-ahead bias exists in signal computation.

    At each time t, signals must only use price information from
    strictly before time t (i.e., from t-skip_days and earlier).

    Raises
    ------
    AssertionError
        If look-ahead bias is detected.
    """
    for L in lookback_periods:
        signal_col = f"signal_{L}"

        # Find first non-NaN signal
        first_valid_idx = df[signal_col].first_valid_index()
        if first_valid_idx is None:
            continue

        # Get position of first valid signal
        first_valid_pos = df.index.get_loc(first_valid_idx)

        # Minimum required history: skip_days + lookback
        min_history = skip_days + L

        # Assert: first valid signal should be at position >= min_history
        # (0-indexed, so position min_history means we have min_history+1 prices)
        assert first_valid_pos >= min_history, (
            f"Look-ahead bias detected for L={L}: "
            f"First signal at position {first_valid_pos}, "
            f"but need at least {min_history} days of history"
        )


def _assert_binary_signals(signals_df: pd.DataFrame) -> None:
    """
    Verify that all signals are strictly binary {0, 1}.

    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame containing signal columns.

    Raises
    ------
    AssertionError
        If any signal value is not in {0, 1, NaN}.
    """
    for col in signals_df.columns:
        valid_values = signals_df[col].dropna()
        unique_values = set(valid_values.unique())

        # Valid values should only be 0 and 1
        assert unique_values.issubset({0, 1}), (
            f"Signal column '{col}' contains non-binary values: "
            f"{unique_values - {0, 1}}"
        )


def compute_signal_returns(
    signals: pd.DataFrame,
    returns: pd.Series,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Compute signal returns (alpha) for each horizon.

    alpha_L(t) = signal_L(t-lag) * r(t)

    This represents the return earned by following the signal
    with a 1-day execution lag.

    Parameters
    ----------
    signals : pd.DataFrame
        Binary signals DataFrame with columns 'signal_{L}'.
    returns : pd.Series
        Daily log returns series.
    lag : int
        Execution lag in days (default 1 to avoid look-ahead).

    Returns
    -------
    pd.DataFrame
        Signal returns DataFrame with columns 'alpha_{L}'.
        Shape: (n_days, n_horizons)
    """
    alpha_df = pd.DataFrame(index=returns.index)

    for col in signals.columns:
        # Extract lookback period from column name
        L = int(col.replace("signal_", ""))

        # Lag signal by 1 day to avoid look-ahead
        # alpha_L(t) = signal_L(t-1) * r(t)
        lagged_signal = signals[col].shift(lag)

        # Signal return: position * return
        # Since signal is binary, alpha is either r(t) or 0
        alpha_df[f"alpha_{L}"] = lagged_signal * returns

    return alpha_df


def get_signal_statistics(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each signal horizon.

    Parameters
    ----------
    signals : pd.DataFrame
        Binary signals DataFrame.

    Returns
    -------
    pd.DataFrame
        Statistics including signal frequency, transitions, etc.
    """
    stats = []

    for col in signals.columns:
        L = int(col.replace("signal_", ""))
        signal = signals[col].dropna()

        # Signal frequency (% of time signal = 1)
        pct_long = (signal == 1).mean() * 100

        # Number of transitions
        transitions = (signal != signal.shift(1)).sum()

        # Average holding period (rough estimate)
        if transitions > 0:
            avg_holding = len(signal) / transitions
        else:
            avg_holding = len(signal)

        stats.append({
            "lookback": L,
            "pct_long": pct_long,
            "pct_flat": 100 - pct_long,
            "n_transitions": transitions,
            "avg_holding_days": avg_holding,
        })

    return pd.DataFrame(stats).set_index("lookback")
