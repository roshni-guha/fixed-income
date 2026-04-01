"""
Main pipeline entry point for G502 multi-horizon momentum strategy.

This script:
1. Loads configuration
2. Loads and prepares data
3. Computes momentum signals
4. Runs walk-forward optimization for all methods
5. Generates strategy returns
6. Computes metrics and comparisons
7. Generates report outputs
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .data import load_data, compute_returns, get_data_summary
from .signals import compute_momentum_signals, compute_signal_returns, get_signal_statistics
from .optimizer import optimize_weights, WalkForwardOptimizer
from .strategy import (
    compute_composite_signal,
    generate_positions,
    compute_strategy_returns,
    compute_equity_curve,
    get_position_summary,
)
from .metrics import (
    compute_metrics,
    compute_rolling_sharpe,
    compute_horizon_attribution,
    compute_weight_stability,
    compare_methods,
    format_metrics_table,
    compute_signal_agreement,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def run_pipeline(config_path: str = "config.yaml") -> Dict:
    """
    Run the complete momentum strategy pipeline.

    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.

    Returns
    -------
    dict
        Dictionary containing all results and objects.
    """
    # Load configuration
    config = load_config(config_path)
    print("=" * 60)
    print("G502 Multi-Horizon Momentum Strategy")
    print("=" * 60)

    # =========================================================================
    # 1. Load and prepare data
    # =========================================================================
    print("\n[1/6] Loading data...")

    data_config = config["data"]
    df = load_data(
        csv_path=data_config["csv_path"],
        date_column=data_config["date_column"],
        price_column=data_config["price_column"],
        date_format=data_config["date_format"],
        max_ffill_days=data_config["max_ffill_days"],
    )

    # Compute returns
    df = compute_returns(df)

    # Print data summary
    summary = get_data_summary(df)
    print(f"  Data range: {summary['start_date']} to {summary['end_date']}")
    print(f"  Observations: {summary['n_observations']:,} ({summary['n_years']:.1f} years)")

    # =========================================================================
    # 2. Compute momentum signals
    # =========================================================================
    print("\n[2/6] Computing momentum signals...")

    signal_config = config["signals"]
    signals = compute_momentum_signals(
        df,
        lookback_periods=signal_config["lookback_periods"],
        skip_days=signal_config["skip_days"],
    )

    # Print signal statistics
    signal_stats = get_signal_statistics(signals)
    print("  Signal statistics:")
    print(signal_stats.to_string(index=True))

    # Compute signal returns (alpha)
    returns = df["returns"]
    alpha_df = compute_signal_returns(signals, returns, lag=1)

    # =========================================================================
    # 3. Run optimization for all methods
    # =========================================================================
    print("\n[3/6] Running walk-forward optimization...")

    opt_config = config["optimizer"]
    methods = ["sharpe", "mvo", "equal"]

    results = {}
    weight_histories = {}
    optimizers = {}

    for method in methods:
        print(f"  Optimizing with method: {method}...")

        weights_df, optimizer = optimize_weights(
            alpha_df,
            method=method,
            in_sample_window=opt_config["in_sample_window"],
            refit_frequency=opt_config["refit_frequency"],
            risk_aversion=opt_config["risk_aversion"],
            use_ledoit_wolf=opt_config["use_ledoit_wolf"],
            min_weight=opt_config["min_weight"],
        )

        optimizers[method] = optimizer
        weight_histories[method] = optimizer.get_weight_history()

        # Store for later
        results[method] = {
            "weights_df": weights_df,
            "optimizer": optimizer,
        }

    # =========================================================================
    # 4. Generate positions and strategy returns
    # =========================================================================
    print("\n[4/6] Generating positions and computing returns...")

    strategy_config = config["strategy"]

    for method in methods:
        weights_df = results[method]["weights_df"]

        # Compute composite signal
        composite = compute_composite_signal(signals, weights_df)

        # Generate positions with execution lag
        positions = generate_positions(
            composite,
            threshold=strategy_config["threshold"],
            execution_lag=strategy_config["execution_lag"],
        )

        # Compute strategy returns
        strategy_df = compute_strategy_returns(
            positions,
            returns,
            transaction_cost_bps=strategy_config["transaction_cost_bps"],
        )

        results[method]["composite_signal"] = composite
        results[method]["positions"] = positions
        results[method]["strategy_df"] = strategy_df

    # =========================================================================
    # 5. Compute metrics
    # =========================================================================
    print("\n[5/6] Computing performance metrics...")

    metrics_config = config["metrics"]

    for method in methods:
        strategy_df = results[method]["strategy_df"]

        # Compute metrics
        metrics = compute_metrics(
            strategy_df,
            trading_days_per_year=metrics_config["trading_days_per_year"],
        )

        results[method]["metrics"] = metrics

        # Compute rolling Sharpe
        rolling_sharpe = compute_rolling_sharpe(
            strategy_df["net_returns"],
            window=metrics_config["rolling_sharpe_window"],
            trading_days_per_year=metrics_config["trading_days_per_year"],
        )
        results[method]["rolling_sharpe"] = rolling_sharpe

    # Per-horizon attribution
    horizon_attribution = compute_horizon_attribution(
        alpha_df,
        trading_days_per_year=metrics_config["trading_days_per_year"],
    )

    # Weight stability for Sharpe and MVO
    for method in ["sharpe", "mvo"]:
        stability = compute_weight_stability(weight_histories[method])
        results[method]["weight_stability"] = stability

    # Buy-and-hold benchmark
    bh_returns = returns.dropna()
    bh_equity = compute_equity_curve(bh_returns, initial_value=100)

    # Create comparison table
    metrics_dict = {method: results[method]["metrics"] for method in methods}
    comparison_df = compare_methods(metrics_dict)

    # Print results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for method in methods:
        print(f"\n--- {method.upper()} Method ---")
        print(format_metrics_table(results[method]["metrics"]))

    print("\n--- Method Comparison ---")
    print(comparison_df.to_string())

    print("\n--- Per-Horizon Attribution ---")
    print(horizon_attribution.to_string())

    for method in ["sharpe", "mvo"]:
        print(f"\n--- Weight Stability ({method.upper()}) ---")
        print(results[method]["weight_stability"].to_string())

    # =========================================================================
    # 6. Generate report
    # =========================================================================
    print("\n[6/6] Generating report...")

    output_config = config["outputs"]
    output_dir = Path(output_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / output_config["report_filename"]

    generate_report(
        results=results,
        signals=signals,
        returns=returns,
        bh_equity=bh_equity,
        weight_histories=weight_histories,
        comparison_df=comparison_df,
        horizon_attribution=horizon_attribution,
        output_path=report_path,
        rolling_window=metrics_config["rolling_sharpe_window"],
    )

    print(f"  Report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    # Return all results
    return {
        "config": config,
        "data": df,
        "signals": signals,
        "alpha": alpha_df,
        "results": results,
        "weight_histories": weight_histories,
        "comparison": comparison_df,
        "horizon_attribution": horizon_attribution,
        "bh_equity": bh_equity,
    }


def generate_report(
    results: Dict,
    signals: pd.DataFrame,
    returns: pd.Series,
    bh_equity: pd.Series,
    weight_histories: Dict[str, pd.DataFrame],
    comparison_df: pd.DataFrame,
    horizon_attribution: pd.DataFrame,
    output_path: Path,
    rolling_window: int = 63,
) -> None:
    """
    Generate PDF tearsheet with all visualizations.

    Parameters
    ----------
    results : dict
        Results dictionary with strategy data for each method.
    signals : pd.DataFrame
        Binary signals DataFrame.
    returns : pd.Series
        Daily returns.
    bh_equity : pd.Series
        Buy-and-hold equity curve.
    weight_histories : dict
        Weight history for each method.
    comparison_df : pd.DataFrame
        Method comparison table.
    horizon_attribution : pd.DataFrame
        Per-horizon attribution.
    output_path : Path
        Output PDF path.
    rolling_window : int
        Rolling window for Sharpe calculation.
    """
    methods = ["sharpe", "mvo", "equal"]
    colors = {"sharpe": "#1f77b4", "mvo": "#ff7f0e", "equal": "#2ca02c", "bh": "#d62728"}

    with PdfPages(output_path) as pdf:
        # =====================================================================
        # Figure 1: Equity Curves
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for method in methods:
            strategy_df = results[method]["strategy_df"]
            equity = compute_equity_curve(strategy_df["net_returns"].dropna(), initial_value=100)
            ax.plot(equity.index, equity.values, label=f"{method.upper()}", color=colors[method], linewidth=1.5)

        # Buy-and-hold
        ax.plot(bh_equity.index, bh_equity.values, label="Buy & Hold", color=colors["bh"], linewidth=1.5, linestyle="--")

        ax.set_title("Equity Curves: Strategy Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (starting = 100)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # Figure 2: Weight Evolution (Sharpe vs MVO)
        # =====================================================================
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for idx, method in enumerate(["sharpe", "mvo"]):
            ax = axes[idx]
            wh = weight_histories[method]

            # Create time series of weights (expand to daily)
            weights_df = results[method]["weights_df"].dropna()

            # Stacked area plot
            weights_df.plot.area(ax=ax, stacked=True, alpha=0.7, linewidth=0)

            ax.set_title(f"Weight Evolution: {method.upper()}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Weight")
            ax.set_ylim(0, 1)
            ax.legend(loc="upper left", ncol=3, fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # Figure 3: Signal Agreement Over Time
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 5))

        agreement = compute_signal_agreement(signals)

        # Rolling mean for smoothing
        agreement_smooth = agreement.rolling(window=21, center=True).mean()

        ax.fill_between(agreement.index, 0, agreement.values, alpha=0.3, color="#1f77b4", label="Daily Count")
        ax.plot(agreement_smooth.index, agreement_smooth.values, color="#1f77b4", linewidth=1.5, label="21-day MA")

        ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5, label="Neutral (3/6)")

        ax.set_title("Signal Agreement: Number of Horizons Signaling Long", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count of Long Signals (max 6)")
        ax.set_ylim(0, 6)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # Figure 4: Rolling Sharpe Ratio
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 5))

        for method in methods:
            rolling_sharpe = results[method]["rolling_sharpe"]
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=method.upper(), color=colors[method], linewidth=1)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=-1, color="gray", linestyle="--", alpha=0.5)

        ax.set_title(f"Rolling {rolling_window}-Day Sharpe Ratio", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio (annualized)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-4, 4)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # Figure 5: Summary Statistics Table
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

        # Format comparison table for display
        table_data = comparison_df.astype(object).copy()  # Convert to object dtype for string assignment

        # Format numeric values
        format_map = {
            "annualized_return": lambda x: f"{x:.2%}",
            "annualized_volatility": lambda x: f"{x:.2%}",
            "sharpe_ratio": lambda x: f"{x:.3f}",
            "max_drawdown": lambda x: f"{x:.2%}",
            "max_drawdown_duration_days": lambda x: f"{x:.0f}",
            "calmar_ratio": lambda x: f"{x:.3f}",
            "hit_rate": lambda x: f"{x:.1f}%",
            "time_in_market_pct": lambda x: f"{x:.1f}%",
            "daily_turnover": lambda x: f"{x:.4f}",
            "annual_turnover": lambda x: f"{x:.1f}",
            "annualized_cost_drag": lambda x: f"{x:.4%}",
        }

        for idx in table_data.index:
            if idx in format_map:
                for col in table_data.columns:
                    val = comparison_df.loc[idx, col]  # Get original numeric value
                    if pd.notna(val):
                        table_data.loc[idx, col] = format_map[idx](val)

        # Create table
        table = ax.table(
            cellText=table_data.values,
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        ax.set_title("Performance Comparison", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # Figure 6: Per-Horizon Attribution
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(horizon_attribution))
        width = 0.35

        sharpes = horizon_attribution["sharpe_ratio"].values
        returns_ann = horizon_attribution["annualized_return"].values * 100  # Convert to %

        bars1 = ax.bar(x - width/2, sharpes, width, label="Sharpe Ratio", color="#1f77b4")
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, returns_ann, width, label="Ann. Return (%)", color="#ff7f0e", alpha=0.7)

        ax.set_xlabel("Lookback Period (days)")
        ax.set_ylabel("Sharpe Ratio", color="#1f77b4")
        ax2.set_ylabel("Annualized Return (%)", color="#ff7f0e")

        ax.set_xticks(x)
        ax.set_xticklabels(horizon_attribution.index)

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

        ax.set_title("Per-Horizon Attribution", fontsize=14, fontweight="bold")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="G502 Multi-Horizon Momentum Strategy")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        results = run_pipeline(args.config)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
