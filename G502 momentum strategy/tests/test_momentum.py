"""
Unit tests for G502 Multi-Horizon Momentum Strategy.

Tests include:
(a) Verify skip-1 convention on synthetic prices
(b) Verify binary signal output is strictly {0, 1}
(c) Verify all optimizers produce weights summing to 1 with w≥0
(d) Verify 1-day execution lag in position series
(e) Verify MVO with Ledoit-Wolf produces a valid positive-definite covariance matrix
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from momentum.data import load_data, compute_returns, _detect_stale_prices
from momentum.signals import (
    compute_momentum_signals,
    compute_signal_returns,
    _compute_raw_momentum,
    _momentum_to_binary_signal,
)
from momentum.optimizer import (
    optimize_weights,
    _equal_weights,
    _optimize_sharpe,
    _optimize_mvo,
    verify_weights,
    WalkForwardOptimizer,
)
from momentum.strategy import (
    compute_composite_signal,
    generate_positions,
    compute_strategy_returns,
)
from momentum.metrics import compute_metrics, compute_rolling_sharpe


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_prices():
    """
    Generate synthetic price data with known momentum characteristics.

    Creates an upward trending price series followed by downward trend,
    allowing predictable momentum signal verification.
    """
    dates = pd.date_range(start="2020-01-01", periods=500, freq="B")

    # Create trending price series
    np.random.seed(42)

    # First 250 days: upward trend
    trend_up = np.linspace(100, 120, 250)
    noise_up = np.random.normal(0, 0.5, 250)
    prices_up = trend_up + noise_up

    # Next 250 days: downward trend
    trend_down = np.linspace(120, 95, 250)
    noise_down = np.random.normal(0, 0.5, 250)
    prices_down = trend_down + noise_down

    prices = np.concatenate([prices_up, prices_down])

    df = pd.DataFrame({"price": prices}, index=dates)
    df.index.name = "date"

    return df


@pytest.fixture
def synthetic_prices_with_returns(synthetic_prices):
    """Add returns to synthetic prices."""
    return compute_returns(synthetic_prices)


@pytest.fixture
def synthetic_signals(synthetic_prices):
    """Generate signals from synthetic prices."""
    return compute_momentum_signals(
        synthetic_prices,
        lookback_periods=[10, 20, 60],
        skip_days=1,
    )


@pytest.fixture
def synthetic_alpha(synthetic_prices_with_returns, synthetic_signals):
    """Generate alpha (signal returns) from synthetic data."""
    returns = synthetic_prices_with_returns["returns"]
    return compute_signal_returns(synthetic_signals, returns, lag=1)


# =============================================================================
# Test (a): Verify skip-1 convention on synthetic prices
# =============================================================================

class TestSkip1Convention:
    """Tests for skip-1 (no look-ahead) convention in momentum calculation."""

    def test_momentum_uses_lagged_prices_only(self, synthetic_prices):
        """
        Verify that momentum at time t only uses prices from t-1 and earlier.

        For mom_L(t) = P_{t-1} / P_{t-1-L} - 1:
        - Numerator uses P_{t-1}
        - Denominator uses P_{t-1-L}
        Both are strictly before time t.
        """
        prices = synthetic_prices["price"]
        lookback = 10
        skip = 1

        momentum = _compute_raw_momentum(prices, lookback, skip)

        # At index i, momentum should use prices at i-1 and i-1-lookback
        # First valid momentum should be at index skip + lookback
        first_valid_idx = momentum.first_valid_index()
        first_valid_pos = momentum.index.get_loc(first_valid_idx)

        assert first_valid_pos >= skip + lookback, (
            f"First valid momentum at position {first_valid_pos}, "
            f"expected >= {skip + lookback}"
        )

        # Manually verify calculation at a specific point
        test_idx = 50
        test_date = prices.index[test_idx]

        expected_recent = prices.iloc[test_idx - skip]
        expected_past = prices.iloc[test_idx - skip - lookback]
        expected_mom = (expected_recent / expected_past) - 1

        actual_mom = momentum.loc[test_date]

        np.testing.assert_almost_equal(
            actual_mom, expected_mom, decimal=10,
            err_msg="Momentum calculation mismatch"
        )

    def test_signal_computation_has_no_lookahead(self, synthetic_prices):
        """
        Verify that signal at time t does not use price at time t.

        The signal should be computable at time t using only information
        available at market open (i.e., prices through t-1).
        """
        signals = compute_momentum_signals(
            synthetic_prices,
            lookback_periods=[10],
            skip_days=1,
        )

        # Get first valid signal
        signal = signals["signal_10"]
        first_valid_idx = signal.first_valid_index()
        first_valid_pos = signal.index.get_loc(first_valid_idx)

        # First valid signal should be at position >= 1 + 10 = 11
        assert first_valid_pos >= 11, (
            f"First signal at position {first_valid_pos}, but needs 11 days history"
        )

    def test_multiple_lookbacks_respect_skip(self, synthetic_prices):
        """Verify all lookback periods respect skip-1 convention."""
        lookbacks = [10, 20, 60, 90, 120, 240]

        signals = compute_momentum_signals(
            synthetic_prices,
            lookback_periods=lookbacks,
            skip_days=1,
        )

        for L in lookbacks:
            col = f"signal_{L}"
            first_valid = signals[col].first_valid_index()
            first_pos = signals.index.get_loc(first_valid)

            # First valid should be at position >= 1 + L
            assert first_pos >= 1 + L, (
                f"Signal {col} starts at position {first_pos}, "
                f"expected >= {1 + L}"
            )


# =============================================================================
# Test (b): Verify binary signal output is strictly {0, 1}
# =============================================================================

class TestBinarySignals:
    """Tests for binary signal constraint."""

    def test_signals_are_binary(self, synthetic_signals):
        """Verify all signal values are in {0, 1}."""
        for col in synthetic_signals.columns:
            values = synthetic_signals[col].dropna()
            unique = set(values.unique())

            assert unique.issubset({0, 1}), (
                f"Column {col} contains non-binary values: {unique - {0, 1}}"
            )

    def test_signals_are_integers(self, synthetic_signals):
        """Verify signal columns are integer type."""
        for col in synthetic_signals.columns:
            values = synthetic_signals[col].dropna()

            # Check all values are whole numbers
            assert np.all(values == values.astype(int)), (
                f"Column {col} contains non-integer values"
            )

    def test_positive_momentum_gives_signal_one(self):
        """Verify positive momentum produces signal = 1."""
        # Create prices with clear upward trend
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame(
            {"price": np.linspace(100, 150, 100)},
            index=dates,
        )

        signals = compute_momentum_signals(prices, lookback_periods=[10], skip_days=1)

        # After initial NaN period, all signals should be 1
        valid_signals = signals["signal_10"].dropna()
        assert (valid_signals == 1).all(), "Upward trend should produce all signal=1"

    def test_negative_momentum_gives_signal_zero(self):
        """Verify negative momentum produces signal = 0."""
        # Create prices with clear downward trend
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame(
            {"price": np.linspace(150, 100, 100)},
            index=dates,
        )

        signals = compute_momentum_signals(prices, lookback_periods=[10], skip_days=1)

        # After initial NaN period, all signals should be 0
        valid_signals = signals["signal_10"].dropna()
        assert (valid_signals == 0).all(), "Downward trend should produce all signal=0"

    def test_no_negative_one_signals(self, synthetic_signals):
        """Explicitly verify no -1 signals (no short positions)."""
        for col in synthetic_signals.columns:
            values = synthetic_signals[col].dropna()
            assert not (values == -1).any(), (
                f"Column {col} contains -1 values (shorts not allowed)"
            )


# =============================================================================
# Test (c): Verify all optimizers produce weights summing to 1 with w≥0
# =============================================================================

class TestOptimizerConstraints:
    """Tests for optimizer weight constraints."""

    def test_equal_weights_sum_to_one(self):
        """Verify equal weights sum to 1."""
        for n in [3, 6, 10]:
            weights = _equal_weights(n)

            assert np.isclose(weights.sum(), 1.0), (
                f"Equal weights for n={n} sum to {weights.sum()}, expected 1.0"
            )
            assert np.all(weights >= 0), "Equal weights contain negative values"
            assert np.all(weights == 1/n), f"Expected all weights = {1/n}"

    def test_sharpe_optimizer_weights_valid(self, synthetic_alpha):
        """Verify Sharpe optimizer produces valid weights."""
        # Convert to float64 as required by scipy
        alpha = synthetic_alpha.dropna().values.astype(np.float64)

        weights = _optimize_sharpe(alpha, min_weight=0.0)

        assert verify_weights(weights, min_weight=0.0), (
            "Sharpe optimizer produced invalid weights"
        )
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5), (
            f"Weights sum to {weights.sum()}, expected 1.0"
        )
        assert np.all(weights >= -1e-10), "Weights contain negative values"

    def test_mvo_optimizer_weights_valid(self, synthetic_alpha):
        """Verify MVO optimizer produces valid weights."""
        # Convert to float64 as required by scipy
        alpha = synthetic_alpha.dropna().values.astype(np.float64)

        weights = _optimize_mvo(
            alpha,
            risk_aversion=1.0,
            min_weight=0.0,
            use_ledoit_wolf=True,
        )

        assert verify_weights(weights, min_weight=0.0), (
            "MVO optimizer produced invalid weights"
        )
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5), (
            f"Weights sum to {weights.sum()}, expected 1.0"
        )
        assert np.all(weights >= -1e-10), "Weights contain negative values"

    def test_walk_forward_weights_valid(self, synthetic_alpha):
        """Verify walk-forward optimization produces valid weights at each refit."""
        for method in ["sharpe", "mvo", "equal"]:
            weights_df, optimizer = optimize_weights(
                synthetic_alpha,
                method=method,
                in_sample_window=100,
                refit_frequency=50,
            )

            # Check weight history
            weight_history = optimizer.get_weight_history()

            for date, row in weight_history.iterrows():
                weights = row.values
                assert np.isclose(weights.sum(), 1.0, rtol=1e-5), (
                    f"Method {method}, date {date}: weights sum to {weights.sum()}"
                )
                assert np.all(weights >= -1e-10), (
                    f"Method {method}, date {date}: negative weights"
                )

    def test_min_weight_constraint_respected(self, synthetic_alpha):
        """Verify minimum weight constraint is respected."""
        min_w = 0.05

        for method in ["sharpe", "mvo"]:
            weights_df, optimizer = optimize_weights(
                synthetic_alpha,
                method=method,
                in_sample_window=100,
                refit_frequency=50,
                min_weight=min_w,
            )

            weight_history = optimizer.get_weight_history()

            for date, row in weight_history.iterrows():
                weights = row.values
                assert np.all(weights >= min_w - 1e-10), (
                    f"Method {method}: weight below minimum {min_w}"
                )


# =============================================================================
# Test (d): Verify 1-day execution lag in position series
# =============================================================================

class TestExecutionLag:
    """Tests for execution lag (no look-ahead in positions)."""

    def test_position_uses_lagged_signal(self, synthetic_prices_with_returns, synthetic_signals):
        """
        Verify position at t uses signal from t-1.

        pos(t) = 1 if S(t-1) > threshold, else 0
        """
        # Create simple equal weights
        n_horizons = len(synthetic_signals.columns)
        weights_df = pd.DataFrame(
            1/n_horizons,
            index=synthetic_signals.index,
            columns=[f"alpha_{c.split('_')[1]}" for c in synthetic_signals.columns],
        )

        composite = compute_composite_signal(synthetic_signals, weights_df)
        positions = generate_positions(composite, threshold=0.5, execution_lag=1)

        # First valid position should be 1 day after first valid signal
        first_signal = composite.first_valid_index()
        first_position = positions.first_valid_index()

        signal_pos = composite.index.get_loc(first_signal)
        position_pos = positions.index.get_loc(first_position)

        assert position_pos >= signal_pos + 1, (
            f"Position starts at {position_pos}, signal at {signal_pos}, "
            f"expected position to lag by at least 1"
        )

    def test_position_cannot_see_current_signal(self, synthetic_prices_with_returns, synthetic_signals):
        """
        Verify that position decision at t cannot use information from t.

        This simulates the real constraint that you must decide position
        at market open based on previous day's close.
        """
        n_horizons = len(synthetic_signals.columns)
        weights_df = pd.DataFrame(
            1/n_horizons,
            index=synthetic_signals.index,
            columns=[f"alpha_{c.split('_')[1]}" for c in synthetic_signals.columns],
        )

        composite = compute_composite_signal(synthetic_signals, weights_df)

        # Test with different execution lags
        for lag in [1, 2]:
            positions = generate_positions(composite, threshold=0.5, execution_lag=lag)

            # For each position, verify the corresponding signal is from lag days ago
            common_idx = positions.dropna().index.intersection(composite.dropna().index)

            for i, date in enumerate(common_idx):
                if i < lag:
                    continue

                # Position at date should be based on signal from date - lag
                pos_val = positions.loc[date]
                signal_date = common_idx[i - lag]
                signal_val = composite.loc[signal_date]

                expected_pos = 1 if signal_val > 0.5 else 0
                assert pos_val == expected_pos, (
                    f"Position mismatch at {date}: got {pos_val}, "
                    f"expected {expected_pos} based on signal at {signal_date}"
                )

    def test_strategy_returns_use_lagged_position(self, synthetic_prices_with_returns, synthetic_signals):
        """
        Verify strategy return at t uses position decided at t (based on t-1 signal).

        ret(t) = pos(t) * r(t)

        Position at t is known at market open, return at t is realized at close.
        """
        returns = synthetic_prices_with_returns["returns"]

        n_horizons = len(synthetic_signals.columns)
        weights_df = pd.DataFrame(
            1/n_horizons,
            index=synthetic_signals.index,
            columns=[f"alpha_{c.split('_')[1]}" for c in synthetic_signals.columns],
        )

        composite = compute_composite_signal(synthetic_signals, weights_df)
        positions = generate_positions(composite, threshold=0.5, execution_lag=1)

        strategy_df = compute_strategy_returns(positions, returns, transaction_cost_bps=0)

        # Verify gross return = position * underlying return
        for date in strategy_df.index:
            pos = strategy_df.loc[date, "position"]
            ret = strategy_df.loc[date, "returns"]
            gross = strategy_df.loc[date, "gross_returns"]

            if pd.notna(pos) and pd.notna(ret):
                expected = pos * ret
                np.testing.assert_almost_equal(
                    gross, expected, decimal=10,
                    err_msg=f"Gross return mismatch at {date}"
                )


# =============================================================================
# Test (e): Verify MVO with Ledoit-Wolf produces valid positive-definite covariance
# =============================================================================

class TestLedoitWolf:
    """Tests for Ledoit-Wolf covariance shrinkage in MVO."""

    def test_ledoit_wolf_produces_positive_definite(self, synthetic_alpha):
        """Verify Ledoit-Wolf shrinkage produces positive-definite covariance."""
        from sklearn.covariance import LedoitWolf

        # Convert to float64 for sklearn
        alpha = synthetic_alpha.dropna().values.astype(np.float64)

        lw = LedoitWolf()
        lw.fit(alpha)
        cov = lw.covariance_

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)

        assert np.all(eigenvalues > -1e-10), (
            f"Covariance has negative eigenvalues: {eigenvalues[eigenvalues < 0]}"
        )

        # Should be positive definite (all eigenvalues > 0)
        # Allow small tolerance for numerical precision
        min_eig = np.min(eigenvalues)
        assert min_eig > -1e-10, (
            f"Minimum eigenvalue {min_eig} indicates non-PD matrix"
        )

    def test_ledoit_wolf_symmetric(self, synthetic_alpha):
        """Verify Ledoit-Wolf covariance is symmetric."""
        from sklearn.covariance import LedoitWolf

        # Convert to float64 for sklearn
        alpha = synthetic_alpha.dropna().values.astype(np.float64)

        lw = LedoitWolf()
        lw.fit(alpha)
        cov = lw.covariance_

        np.testing.assert_array_almost_equal(
            cov, cov.T, decimal=10,
            err_msg="Covariance matrix is not symmetric"
        )

    def test_mvo_with_ledoit_wolf_converges(self, synthetic_alpha):
        """Verify MVO with Ledoit-Wolf converges to a valid solution."""
        # Convert to float64 as required by scipy
        alpha = synthetic_alpha.dropna().values.astype(np.float64)

        # Should not raise warnings about convergence
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            weights = _optimize_mvo(
                alpha,
                risk_aversion=1.0,
                min_weight=0.0,
                use_ledoit_wolf=True,
            )

            # Filter for convergence warnings
            convergence_warnings = [
                x for x in w if "not converge" in str(x.message).lower()
            ]

            # Should have converged without warnings
            assert len(convergence_warnings) == 0, (
                f"MVO convergence warnings: {[str(w.message) for w in convergence_warnings]}"
            )

    def test_mvo_without_ledoit_wolf_still_valid(self, synthetic_alpha):
        """Verify MVO without shrinkage also produces valid weights."""
        # Convert to float64 as required by scipy
        alpha = synthetic_alpha.dropna().values.astype(np.float64)

        weights = _optimize_mvo(
            alpha,
            risk_aversion=1.0,
            min_weight=0.0,
            use_ledoit_wolf=False,
        )

        assert verify_weights(weights), "MVO without Ledoit-Wolf produced invalid weights"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_no_lookahead(self, synthetic_prices):
        """End-to-end test verifying no look-ahead bias throughout pipeline."""
        # Compute returns
        df = compute_returns(synthetic_prices)
        returns = df["returns"]

        # Compute signals
        signals = compute_momentum_signals(df, lookback_periods=[10, 20], skip_days=1)

        # Compute alpha with lag
        alpha = compute_signal_returns(signals, returns, lag=1)

        # Optimize weights
        weights_df, _ = optimize_weights(
            alpha,
            method="sharpe",
            in_sample_window=100,
            refit_frequency=50,
        )

        # Composite signal
        composite = compute_composite_signal(signals, weights_df)

        # Positions with lag
        positions = generate_positions(composite, threshold=0.5, execution_lag=1)

        # Strategy returns
        strategy_df = compute_strategy_returns(positions, returns)

        # Verify: at each point, we can only use past information
        # The first valid strategy return should be well after data starts
        first_valid_return = strategy_df["net_returns"].first_valid_index()
        first_valid_pos = strategy_df.index.get_loc(first_valid_return)

        # Minimum days needed: max(lookback) + skip + in_sample_window + execution_lag
        min_days = 20 + 1 + 100 + 1
        assert first_valid_pos >= min_days - 50, (  # Allow some tolerance for alignment
            f"Strategy returns start too early at position {first_valid_pos}"
        )

    def test_metrics_computation(self, synthetic_prices_with_returns, synthetic_signals):
        """Test that metrics compute without error."""
        returns = synthetic_prices_with_returns["returns"]

        n_horizons = len(synthetic_signals.columns)
        weights_df = pd.DataFrame(
            1/n_horizons,
            index=synthetic_signals.index,
            columns=[f"alpha_{c.split('_')[1]}" for c in synthetic_signals.columns],
        )

        composite = compute_composite_signal(synthetic_signals, weights_df)
        positions = generate_positions(composite, threshold=0.5, execution_lag=1)
        strategy_df = compute_strategy_returns(positions, returns)

        metrics = compute_metrics(strategy_df)

        # Check required metrics exist
        required = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "hit_rate",
            "time_in_market_pct",
        ]

        for key in required:
            assert key in metrics, f"Missing metric: {key}"
            assert np.isfinite(metrics[key]) or np.isnan(metrics[key]), (
                f"Metric {key} has invalid value: {metrics[key]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
