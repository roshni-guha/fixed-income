"""
Performance metrics and analytics for G502 momentum strategy.

This module computes:
- Standard performance metrics (Sharpe, max drawdown, etc.)
- Per-horizon attribution
- Method comparison tables
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_metrics(
    strategy_df: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    strategy_df : pd.DataFrame
        DataFrame from compute_strategy_returns() with columns:
        position, returns, gross_returns, trade, cost, net_returns
    trading_days_per_year : int
        Annualization factor.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    net_returns = strategy_df["net_returns"].dropna()
    gross_returns = strategy_df["gross_returns"].dropna()
    positions = strategy_df["position"].dropna()
    trades = strategy_df["trade"].dropna()
    costs = strategy_df["cost"].dropna()
    underlying_returns = strategy_df["returns"].dropna()

    metrics = {}

    # Basic return metrics
    metrics["total_return"] = np.exp(net_returns.sum()) - 1
    metrics["annualized_return"] = net_returns.mean() * trading_days_per_year
    metrics["annualized_volatility"] = net_returns.std() * np.sqrt(trading_days_per_year)

    # Sharpe ratio
    if metrics["annualized_volatility"] > 0:
        metrics["sharpe_ratio"] = (
            metrics["annualized_return"] / metrics["annualized_volatility"]
        )
    else:
        metrics["sharpe_ratio"] = 0.0

    # Drawdown metrics
    equity_curve = np.exp(net_returns.cumsum())
    drawdown, max_dd, max_dd_duration = _compute_drawdown_stats(equity_curve)
    metrics["max_drawdown"] = max_dd
    metrics["max_drawdown_duration_days"] = max_dd_duration

    # Calmar ratio
    if max_dd > 0:
        metrics["calmar_ratio"] = metrics["annualized_return"] / max_dd
    else:
        metrics["calmar_ratio"] = np.inf if metrics["annualized_return"] > 0 else 0.0

    # Hit rate (% of long days with positive return)
    long_days = positions == 1
    if long_days.sum() > 0:
        aligned_returns = underlying_returns.loc[long_days.index[long_days]]
        metrics["hit_rate"] = (aligned_returns > 0).mean() * 100
    else:
        metrics["hit_rate"] = 0.0

    # Time in market
    metrics["time_in_market_pct"] = (positions == 1).mean() * 100

    # Turnover
    metrics["daily_turnover"] = trades.mean()
    metrics["annual_turnover"] = trades.sum() / (len(trades) / trading_days_per_year)

    # Transaction cost drag
    metrics["total_cost"] = costs.sum()
    metrics["annualized_cost_drag"] = costs.mean() * trading_days_per_year

    # Gross vs net comparison
    metrics["gross_annualized_return"] = gross_returns.mean() * trading_days_per_year

    return metrics


def _compute_drawdown_stats(
    equity_curve: pd.Series,
) -> Tuple[pd.Series, float, int]:
    """
    Compute drawdown series and statistics.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve.

    Returns
    -------
    Tuple[pd.Series, float, int]
        - Drawdown series (as positive percentage)
        - Maximum drawdown
        - Maximum drawdown duration in days
    """
    # Running maximum
    running_max = equity_curve.cummax()

    # Drawdown: (peak - current) / peak
    drawdown = (running_max - equity_curve) / running_max

    # Maximum drawdown
    max_dd = drawdown.max()

    # Drawdown duration
    in_drawdown = drawdown > 0
    dd_starts = (~in_drawdown).cumsum()

    max_duration = 0
    for _, group in drawdown.groupby(dd_starts):
        if len(group) > 0 and group.max() > 0:
            duration = len(group)
            max_duration = max(max_duration, duration)

    return drawdown, max_dd, max_duration


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    window : int
        Rolling window in days.
    trading_days_per_year : int
        Annualization factor.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    annualization = np.sqrt(trading_days_per_year)

    # Avoid division by zero
    rolling_sharpe = np.where(
        rolling_std > 1e-10,
        (rolling_mean / rolling_std) * annualization,
        0.0,
    )

    return pd.Series(rolling_sharpe, index=returns.index)


def compute_horizon_attribution(
    alpha_df: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compute per-horizon performance attribution.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Signal returns for each horizon.
    trading_days_per_year : int
        Annualization factor.

    Returns
    -------
    pd.DataFrame
        Attribution metrics for each horizon.
    """
    results = []

    for col in alpha_df.columns:
        alpha = alpha_df[col].dropna()

        if len(alpha) == 0:
            continue

        L = int(col.replace("alpha_", ""))

        ann_return = alpha.mean() * trading_days_per_year
        ann_vol = alpha.std() * np.sqrt(trading_days_per_year)

        if ann_vol > 0:
            sharpe = ann_return / ann_vol
        else:
            sharpe = 0.0

        results.append({
            "lookback": L,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "n_observations": len(alpha),
        })

    return pd.DataFrame(results).set_index("lookback")


def compute_weight_stability(
    weight_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weight stability metrics across refit dates.

    Parameters
    ----------
    weight_history : pd.DataFrame
        Weight history from optimizer (one row per refit date).

    Returns
    -------
    pd.DataFrame
        Mean and std of weights for each horizon.
    """
    stability = pd.DataFrame({
        "mean_weight": weight_history.mean(),
        "std_weight": weight_history.std(),
        "min_weight": weight_history.min(),
        "max_weight": weight_history.max(),
    })

    return stability


def compare_methods(
    results: Dict[str, dict],
) -> pd.DataFrame:
    """
    Create side-by-side comparison of different optimization methods.

    Parameters
    ----------
    results : dict
        Dictionary mapping method names to their metrics dict.

    Returns
    -------
    pd.DataFrame
        Comparison table with methods as columns.
    """
    # Define metrics to compare
    compare_metrics = [
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "max_drawdown_duration_days",
        "calmar_ratio",
        "hit_rate",
        "time_in_market_pct",
        "daily_turnover",
        "annual_turnover",
        "annualized_cost_drag",
    ]

    comparison = {}

    for method, metrics in results.items():
        comparison[method] = {
            metric: metrics.get(metric, np.nan)
            for metric in compare_metrics
        }

    df = pd.DataFrame(comparison)

    # Format for display
    df.index.name = "Metric"

    return df


def format_metrics_table(metrics: dict) -> str:
    """
    Format metrics dictionary as a readable string table.

    Parameters
    ----------
    metrics : dict
        Performance metrics dictionary.

    Returns
    -------
    str
        Formatted string table.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Performance Metrics")
    lines.append("=" * 50)

    format_specs = {
        "total_return": ("{:.2%}", "Total Return"),
        "annualized_return": ("{:.2%}", "Annualized Return"),
        "annualized_volatility": ("{:.2%}", "Annualized Volatility"),
        "sharpe_ratio": ("{:.3f}", "Sharpe Ratio"),
        "max_drawdown": ("{:.2%}", "Max Drawdown"),
        "max_drawdown_duration_days": ("{:.0f}", "Max DD Duration (days)"),
        "calmar_ratio": ("{:.3f}", "Calmar Ratio"),
        "hit_rate": ("{:.1f}%", "Hit Rate"),
        "time_in_market_pct": ("{:.1f}%", "Time in Market"),
        "daily_turnover": ("{:.4f}", "Daily Turnover"),
        "annual_turnover": ("{:.1f}", "Annual Turnover"),
        "annualized_cost_drag": ("{:.4%}", "Ann. Cost Drag"),
    }

    for key, (fmt, label) in format_specs.items():
        if key in metrics:
            value = metrics[key]
            if np.isfinite(value):
                formatted = fmt.format(value)
            else:
                formatted = "N/A"
            lines.append(f"{label:.<30} {formatted:>15}")

    lines.append("=" * 50)

    return "\n".join(lines)


def compute_signal_agreement(signals: pd.DataFrame) -> pd.Series:
    """
    Compute daily count of signals agreeing to go long.

    Parameters
    ----------
    signals : pd.DataFrame
        Binary signals DataFrame.

    Returns
    -------
    pd.Series
        Count of signals equal to 1 each day.
    """
    return signals.sum(axis=1)
