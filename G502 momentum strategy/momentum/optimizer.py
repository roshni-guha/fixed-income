"""
Weight optimization for G502 multi-horizon momentum strategy.

This module implements three optimization methods:
1. Sharpe maximization
2. Mean-variance optimization (Markowitz tangency)
3. Equal weight baseline

All methods use walk-forward optimization with rolling in-sample windows.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import warnings


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: np.ndarray  # Shape: (n_horizons,)
    horizon_names: List[str]
    method: str
    sharpe_in_sample: float
    success: bool
    message: str


class WalkForwardOptimizer:
    """
    Walk-forward optimizer for momentum signal weights.

    Fits weights on rolling in-sample windows and applies them
    out-of-sample until the next refit date.

    Parameters
    ----------
    in_sample_window : int
        Number of days for in-sample estimation (default 504 ~ 2 years).
    refit_frequency : int
        Days between refits (default 63 ~ quarterly).
    risk_aversion : float
        Lambda parameter for MVO (default 1.0).
    use_ledoit_wolf : bool
        Whether to use Ledoit-Wolf shrinkage for covariance.
    min_weight : float
        Minimum weight constraint (default 0.0).
    """

    def __init__(
        self,
        in_sample_window: int = 504,
        refit_frequency: int = 63,
        risk_aversion: float = 1.0,
        use_ledoit_wolf: bool = True,
        min_weight: float = 0.0,
    ):
        self.in_sample_window = in_sample_window
        self.refit_frequency = refit_frequency
        self.risk_aversion = risk_aversion
        self.use_ledoit_wolf = use_ledoit_wolf
        self.min_weight = min_weight

        # Weight history: {date: weights array}
        self.weight_history: Dict[pd.Timestamp, np.ndarray] = {}
        self.horizon_names: List[str] = []

    def fit(
        self,
        alpha_df: pd.DataFrame,
        method: Literal["sharpe", "mvo", "equal"] = "sharpe",
    ) -> pd.DataFrame:
        """
        Run walk-forward optimization.

        Parameters
        ----------
        alpha_df : pd.DataFrame
            Signal returns DataFrame with columns 'alpha_{L}'.
            Shape: (n_days, n_horizons)
        method : str
            Optimization method: 'sharpe', 'mvo', or 'equal'.

        Returns
        -------
        pd.DataFrame
            Time series of weights for each horizon.
            Shape: (n_days, n_horizons)
        """
        # Store horizon names
        self.horizon_names = list(alpha_df.columns)
        n_horizons = len(self.horizon_names)

        # Drop NaN rows for optimization
        alpha_clean = alpha_df.dropna()

        if len(alpha_clean) < self.in_sample_window:
            raise ValueError(
                f"Insufficient data: {len(alpha_clean)} rows, "
                f"need at least {self.in_sample_window} for in-sample window"
            )

        # Initialize output weights DataFrame
        weights_df = pd.DataFrame(
            index=alpha_df.index,
            columns=self.horizon_names,
            dtype=float,
        )

        # Determine refit dates
        valid_dates = alpha_clean.index
        first_fit_idx = self.in_sample_window - 1  # 0-indexed
        refit_positions = list(range(
            first_fit_idx,
            len(valid_dates),
            self.refit_frequency
        ))

        # Walk-forward loop
        current_weights = None
        self.weight_history = {}

        for i, pos in enumerate(refit_positions):
            # In-sample data: [pos - window + 1 : pos + 1]
            start_pos = max(0, pos - self.in_sample_window + 1)
            in_sample = alpha_clean.iloc[start_pos:pos + 1]

            # Fit weights - ensure float64 type for scipy
            in_sample_values = in_sample.values.astype(np.float64)

            if method == "equal":
                weights = _equal_weights(n_horizons)
            elif method == "sharpe":
                weights = _optimize_sharpe(
                    in_sample_values,
                    min_weight=self.min_weight,
                )
            elif method == "mvo":
                weights = _optimize_mvo(
                    in_sample_values,
                    risk_aversion=self.risk_aversion,
                    min_weight=self.min_weight,
                    use_ledoit_wolf=self.use_ledoit_wolf,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Record fit date and weights
            fit_date = valid_dates[pos]
            self.weight_history[fit_date] = weights.copy()
            current_weights = weights

            # Apply weights out-of-sample until next refit
            if i + 1 < len(refit_positions):
                next_pos = refit_positions[i + 1]
                # Out-of-sample period: (pos, next_pos]
                oos_dates = valid_dates[pos + 1:next_pos + 1]
            else:
                # Last period: apply to all remaining dates
                oos_dates = valid_dates[pos + 1:]

            # Assign weights to out-of-sample dates
            for date in oos_dates:
                if date in weights_df.index:
                    weights_df.loc[date] = current_weights

            # Also assign to the fit date itself (first valid date)
            if i == 0:
                weights_df.loc[fit_date] = current_weights

        # Forward fill weights to handle any gaps
        weights_df = weights_df.ffill()

        return weights_df

    def get_weight_history(self) -> pd.DataFrame:
        """
        Get weight history as DataFrame.

        Returns
        -------
        pd.DataFrame
            Weights at each refit date.
            Shape: (n_refits, n_horizons)
        """
        if not self.weight_history:
            raise ValueError("No weight history available. Run fit() first.")

        return pd.DataFrame.from_dict(
            self.weight_history,
            orient="index",
            columns=self.horizon_names,
        )


def optimize_weights(
    alpha_df: pd.DataFrame,
    method: Literal["sharpe", "mvo", "equal"] = "sharpe",
    in_sample_window: int = 504,
    refit_frequency: int = 63,
    risk_aversion: float = 1.0,
    use_ledoit_wolf: bool = True,
    min_weight: float = 0.0,
) -> Tuple[pd.DataFrame, WalkForwardOptimizer]:
    """
    Convenience function to run walk-forward optimization.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Signal returns DataFrame.
    method : str
        Optimization method.
    in_sample_window : int
        In-sample window size.
    refit_frequency : int
        Refit frequency in days.
    risk_aversion : float
        MVO risk aversion parameter.
    use_ledoit_wolf : bool
        Whether to use Ledoit-Wolf shrinkage.
    min_weight : float
        Minimum weight constraint.

    Returns
    -------
    Tuple[pd.DataFrame, WalkForwardOptimizer]
        Weights time series and fitted optimizer object.
    """
    optimizer = WalkForwardOptimizer(
        in_sample_window=in_sample_window,
        refit_frequency=refit_frequency,
        risk_aversion=risk_aversion,
        use_ledoit_wolf=use_ledoit_wolf,
        min_weight=min_weight,
    )

    weights_df = optimizer.fit(alpha_df, method=method)

    return weights_df, optimizer


def _equal_weights(n_horizons: int) -> np.ndarray:
    """
    Generate equal weights.

    Parameters
    ----------
    n_horizons : int
        Number of horizons.

    Returns
    -------
    np.ndarray
        Equal weights summing to 1. Shape: (n_horizons,)
    """
    return np.ones(n_horizons) / n_horizons


def _optimize_sharpe(
    alpha: np.ndarray,
    min_weight: float = 0.0,
) -> np.ndarray:
    """
    Maximize Sharpe ratio of weighted signal returns.

    max_w  mean(r_p) / std(r_p)
    where r_p = sum_L w_L * alpha_L

    Constraints: sum(w) = 1, w_L >= min_weight

    Parameters
    ----------
    alpha : np.ndarray
        Signal returns matrix. Shape: (n_days, n_horizons)
    min_weight : float
        Minimum weight for each horizon.

    Returns
    -------
    np.ndarray
        Optimal weights. Shape: (n_horizons,)
    """
    n_days, n_horizons = alpha.shape

    # Initial guess: equal weights
    w0 = np.ones(n_horizons) / n_horizons

    def neg_sharpe(w):
        """Negative Sharpe ratio (to minimize)."""
        # Portfolio return: r_p = alpha @ w
        r_p = alpha @ w  # Shape: (n_days,)
        mean_r = np.mean(r_p)
        std_r = np.std(r_p, ddof=1)

        if std_r < 1e-10:
            return 0.0  # Avoid division by zero

        return -mean_r / std_r

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum = 1
    ]

    # Bounds: w_L >= min_weight
    bounds = [(min_weight, 1.0) for _ in range(n_horizons)]

    # Optimize
    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        warnings.warn(f"Sharpe optimization did not converge: {result.message}")

    # Ensure weights sum to 1 and are non-negative
    weights = np.maximum(result.x, min_weight)
    weights = weights / np.sum(weights)

    return weights


def _optimize_mvo(
    alpha: np.ndarray,
    risk_aversion: float = 1.0,
    min_weight: float = 0.0,
    use_ledoit_wolf: bool = True,
) -> np.ndarray:
    """
    Mean-variance optimization (Markowitz tangency portfolio).

    max_w  w' * mu - (lambda/2) * w' * Sigma * w

    Constraints: sum(w) = 1, w_L >= min_weight

    Parameters
    ----------
    alpha : np.ndarray
        Signal returns matrix. Shape: (n_days, n_horizons)
    risk_aversion : float
        Lambda parameter (risk aversion).
    min_weight : float
        Minimum weight for each horizon.
    use_ledoit_wolf : bool
        Whether to use Ledoit-Wolf shrinkage for covariance.

    Returns
    -------
    np.ndarray
        Optimal weights. Shape: (n_horizons,)
    """
    n_days, n_horizons = alpha.shape

    # Compute mean returns: mu
    # Shape: (n_horizons,)
    mu = np.mean(alpha, axis=0)

    # Compute covariance matrix: Sigma
    # Shape: (n_horizons, n_horizons)
    if use_ledoit_wolf:
        lw = LedoitWolf()
        lw.fit(alpha)
        Sigma = lw.covariance_

        # Verify positive definiteness
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(eigvals) < -1e-10:
            warnings.warn("Ledoit-Wolf covariance is not positive definite")
            # Regularize
            Sigma += np.eye(n_horizons) * (-np.min(eigvals) + 1e-6)
    else:
        Sigma = np.cov(alpha, rowvar=False, ddof=1)

    # Initial guess: equal weights
    w0 = np.ones(n_horizons) / n_horizons

    def neg_utility(w):
        """Negative utility (to minimize)."""
        expected_return = w @ mu
        variance = w @ Sigma @ w
        utility = expected_return - (risk_aversion / 2) * variance
        return -utility

    # Gradient for faster convergence
    def neg_utility_grad(w):
        return -(mu - risk_aversion * Sigma @ w)

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]

    # Bounds
    bounds = [(min_weight, 1.0) for _ in range(n_horizons)]

    # Optimize
    result = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        jac=neg_utility_grad,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        warnings.warn(f"MVO optimization did not converge: {result.message}")

    # Ensure weights sum to 1 and are non-negative
    weights = np.maximum(result.x, min_weight)
    weights = weights / np.sum(weights)

    return weights


def verify_weights(weights: np.ndarray, min_weight: float = 0.0) -> bool:
    """
    Verify that weights satisfy constraints.

    Parameters
    ----------
    weights : np.ndarray
        Weight vector.
    min_weight : float
        Minimum weight constraint.

    Returns
    -------
    bool
        True if all constraints are satisfied.
    """
    # Check sum to 1
    if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
        return False

    # Check non-negativity
    if np.any(weights < min_weight - 1e-10):
        return False

    return True
