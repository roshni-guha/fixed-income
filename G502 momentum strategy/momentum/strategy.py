"""
Strategy implementation for G502 multi-horizon momentum model.

This module handles:
- Composite signal computation from weighted binary signals
- Position generation with execution lag
- Strategy return computation with transaction costs
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_composite_signal(
    signals: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute weighted composite signal from binary momentum signals.

    S(t) = sum_L w_L * signal_L(t)

    The composite signal is a continuous score in [0, 1] representing
    the fraction of horizons (by weight) agreeing the trend is positive.

    Parameters
    ----------
    signals : pd.DataFrame
        Binary signals DataFrame with columns 'signal_{L}'.
        Shape: (n_days, n_horizons)
    weights : pd.DataFrame
        Time-varying weights DataFrame with columns 'alpha_{L}'.
        Shape: (n_days, n_horizons)

    Returns
    -------
    pd.Series
        Composite signal in [0, 1]. Shape: (n_days,)

    Notes
    -----
    The weight columns are named 'alpha_{L}' to match optimizer output.
    Signal columns are named 'signal_{L}'.
    """
    # Align signals and weights to common index
    common_idx = signals.index.intersection(weights.index)

    signals_aligned = signals.loc[common_idx]
    weights_aligned = weights.loc[common_idx]

    # Extract lookback periods from column names
    lookbacks = []
    for col in signals.columns:
        L = int(col.replace("signal_", ""))
        lookbacks.append(L)

    # Compute weighted sum
    # S(t) = sum_L w_L * signal_L(t)
    composite = pd.Series(0.0, index=common_idx)

    for L in lookbacks:
        signal_col = f"signal_{L}"
        weight_col = f"alpha_{L}"

        if signal_col in signals_aligned.columns and weight_col in weights_aligned.columns:
            composite += weights_aligned[weight_col] * signals_aligned[signal_col]

    return composite


def generate_positions(
    composite_signal: pd.Series,
    threshold: float = 0.5,
    execution_lag: int = 1,
) -> pd.Series:
    """
    Generate binary positions from composite signal.

    Position rule with execution lag (no look-ahead):
        pos(t) = 1 if S(t-lag) > threshold else 0

    Parameters
    ----------
    composite_signal : pd.Series
        Composite signal in [0, 1].
    threshold : float
        Threshold for taking long position (default 0.5).
        Position = 1 if composite > threshold.
    execution_lag : int
        Execution lag in days (default 1 for no look-ahead).

    Returns
    -------
    pd.Series
        Binary positions {0, 1}. Shape: (n_days,)

    Raises
    ------
    AssertionError
        If look-ahead bias is detected.
    """
    # Apply execution lag
    # pos(t) uses signal from t-lag
    lagged_signal = composite_signal.shift(execution_lag)

    # Binary position: 1 if lagged_signal > threshold, else 0
    # Use nullable Int64 to support NaN values
    positions = (lagged_signal > threshold).astype("Int64")

    # Verify no look-ahead
    _assert_position_lag(composite_signal, positions, execution_lag)

    return positions


def _assert_position_lag(
    signal: pd.Series,
    positions: pd.Series,
    lag: int,
) -> None:
    """
    Verify that positions are properly lagged.

    The position at time t should only depend on signals up to t-lag.

    Raises
    ------
    AssertionError
        If position depends on same-day or future signals.
    """
    # Find first valid position (non-NaN)
    first_valid_pos_idx = positions.first_valid_index()

    if first_valid_pos_idx is None:
        return

    # Get position in integer index
    pos_loc = positions.index.get_loc(first_valid_pos_idx)

    # First valid signal
    first_valid_sig_idx = signal.first_valid_index()

    if first_valid_sig_idx is None:
        return

    sig_loc = signal.index.get_loc(first_valid_sig_idx)

    # Position should start at least 'lag' days after first signal
    assert pos_loc >= sig_loc + lag, (
        f"Position lag violation: first position at {pos_loc}, "
        f"first signal at {sig_loc}, required lag={lag}"
    )


def compute_strategy_returns(
    positions: pd.Series,
    returns: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compute strategy returns accounting for transaction costs.

    ret(t) = pos(t) * r(t) - cost * |pos(t) - pos(t-1)|

    Parameters
    ----------
    positions : pd.Series
        Binary positions {0, 1}.
    returns : pd.Series
        Daily log returns.
    transaction_cost_bps : float
        Transaction cost in basis points (applied on position changes).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'position': binary positions
        - 'returns': underlying asset returns
        - 'gross_returns': strategy returns before costs
        - 'trade': trade indicator (|position change|)
        - 'cost': transaction costs
        - 'net_returns': strategy returns after costs
    """
    # Align positions and returns
    common_idx = positions.index.intersection(returns.index)
    pos = positions.loc[common_idx]
    ret = returns.loc[common_idx]

    # Convert cost to decimal
    cost_decimal = transaction_cost_bps / 10000

    # Gross strategy returns: pos(t) * r(t)
    gross_returns = pos * ret

    # Trade indicator: |pos(t) - pos(t-1)|
    # For binary positions, this is 1 on entry/exit, 0 otherwise
    trades = (pos - pos.shift(1)).abs()
    trades = trades.fillna(0)

    # Transaction costs
    transaction_costs = cost_decimal * trades

    # Net returns
    net_returns = gross_returns - transaction_costs

    result = pd.DataFrame({
        "position": pos,
        "returns": ret,
        "gross_returns": gross_returns,
        "trade": trades,
        "cost": transaction_costs,
        "net_returns": net_returns,
    })

    return result


def compute_buy_and_hold_returns(returns: pd.Series) -> pd.Series:
    """
    Compute buy-and-hold returns (fully invested).

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    pd.Series
        Buy-and-hold returns (equals underlying returns).
    """
    return returns.copy()


def compute_equity_curve(returns: pd.Series, initial_value: float = 100.0) -> pd.Series:
    """
    Compute cumulative equity curve from returns.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns (or simple returns work too for short horizons).
    initial_value : float
        Starting value of the equity curve.

    Returns
    -------
    pd.Series
        Cumulative equity curve.
    """
    # For log returns, cumulative sum gives log(final/initial)
    # exp(cumsum) gives the growth factor
    cumulative = initial_value * np.exp(returns.cumsum())

    return cumulative


def get_position_summary(positions: pd.Series) -> dict:
    """
    Compute summary statistics for positions.

    Parameters
    ----------
    positions : pd.Series
        Binary positions.

    Returns
    -------
    dict
        Position summary statistics.
    """
    valid_pos = positions.dropna()

    # Time in market
    time_in_market = (valid_pos == 1).mean() * 100

    # Number of trades (position changes)
    trades = (valid_pos != valid_pos.shift(1)).sum()

    # Average holding period
    if trades > 0:
        avg_holding = len(valid_pos) / trades
    else:
        avg_holding = len(valid_pos)

    # Turnover: average daily position change
    daily_turnover = (valid_pos - valid_pos.shift(1)).abs().mean()

    return {
        "time_in_market_pct": time_in_market,
        "n_trades": trades,
        "avg_holding_days": avg_holding,
        "daily_turnover": daily_turnover,
    }
