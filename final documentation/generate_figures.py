"""
Figure Generation Script for Fixed Income Strategy Whitepaper
Generates all required figures from the strategies notebook data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# Set the working directory to the fixed-income folder
os.chdir(r"C:\Users\nimas\Regentfund Repository\fixed-income")

# Create figures directory
FIGURES_DIR = r"C:\Users\nimas\Regentfund Repository\fixed-income\final documentation\figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# =============================================================================
# DATA LOADING FUNCTIONS (from notebook)
# =============================================================================

def getPricesFromCSV(file_path, reference_dates):
    df = pd.read_csv(reference_dates, skiprows=15)
    df["Date"] = pd.to_datetime(df["Date"])
    date_series = df["Date"]

    df = pd.read_csv(file_path)
    if type(df["Date"].iloc[0]) == np.int64:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    else:
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

    df_filtered = df[df["Date"].isin(date_series)]
    return df_filtered

def create_bmom(dataframe, weights, price_col="Price"):
    data = dataframe.dropna().sort_values("Date").set_index("Date")
    px = data[price_col]
    momentums = {}
    weighted_momentums = {}

    for lookback, weight in weights.items():
        momentums[lookback] = px.pct_change(lookback)
        weighted_momentums[lookback] = momentums[lookback] * weight

    wm_df = pd.concat(weighted_momentums, axis=1)
    first_valid = wm_df.dropna().index[0]
    bmom = wm_df.loc[first_valid:].sum(axis=1)
    daily_returns = px.pct_change(1)

    return bmom, momentums, bmom.index, daily_returns

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")

dataFrame_H5A4 = getPricesFromCSV("Data Files/H5A4.csv", "Data Files/MLH5A4_Price_History(Fixed Income).xls")
dataFrame_SPY = getPricesFromCSV("Data Files/SPY 1.csv", "Data Files/MLH5A4_Price_History(Fixed Income).xls")
dataFrame_G5O2 = getPricesFromCSV("Data Files/G5O2.csv", "Data Files/MLH5A4_Price_History(Fixed Income).xls")
dataFrame_C5A0 = getPricesFromCSV("Data Files/C5A0.csv", "Data Files/MLH5A4_Price_History(Fixed Income).xls")

bmom_H5A4, momentums_H5A4, dates_H5A4, daily_returns_H5A4 = create_bmom(
    dataFrame_H5A4, {20: 0.15, 60: 0.35, 120: 0.35, 240: 0.15}, "H5A4")
bmom_SPY, momentums_SPY, dates_SPY, daily_returns_SPY = create_bmom(
    dataFrame_SPY, {20: 0.15, 60: 0.35, 120: 0.35, 240: 0.15}, "SPY_adj_close")
bmom_G5O2, momentums_G5O2, dates_G5O2, daily_returns_G5O2 = create_bmom(
    dataFrame_G5O2, {20: 0.15, 60: 0.35, 120: 0.35, 240: 0.15}, "G5O2")
bmom_C5A0, momentums_C5A0, dates_C5A0, daily_returns_C5A0 = create_bmom(
    dataFrame_C5A0, {20: 0.15, 60: 0.35, 120: 0.35, 240: 0.15}, "C5A0")

# =============================================================================
# STRATEGY FUNCTIONS (from notebook)
# =============================================================================

def strat1_signals(bmom_asset, bmom_benchmark, mom240_asset, thresh=-0.03, init_wgt=0):
    idx = bmom_asset.index
    a = bmom_asset.astype(float)
    s = bmom_benchmark.reindex(idx).astype(float)
    m240 = mom240_asset.reindex(idx).astype(float)

    cond1 = (s.shift(1) > thresh) & (s < thresh) & (a < thresh)
    cond2 = (a.shift(1) > thresh) & (a < thresh) & (s < thresh)
    risk_off = (cond1 | cond2) & (m240 < 0)

    sig = pd.Series(np.nan, index=idx, dtype="float64")
    sig = sig.mask(risk_off, 0)
    sig = sig.mask(~risk_off & (a > 0), 1)
    sig.iloc[0] = init_wgt
    sig = sig.ffill().astype(int)

    min_idx = idx.intersection(bmom_benchmark.index)
    sig = sig.reindex(min_idx)
    return sig

def strat2_from_strat1(strat1, hyg_ret, drawdiffthresh=0.10):
    idx = strat1.index
    r_hyg = hyg_ret.reindex(idx).fillna(0.0)
    r_strat1 = r_hyg * strat1.shift(1).fillna(0)

    def drawdown(r):
        eq = (1.0 + r).cumprod()
        return eq / eq.cummax() - 1

    dd_strat1 = drawdown(r_strat1)
    dd_hyg = drawdown(r_hyg)
    dd_diff = dd_strat1 - dd_hyg
    override_trigger = dd_diff > drawdiffthresh
    override_start = override_trigger & (~override_trigger.shift(1, fill_value=False))
    regime_id = override_start.cumsum()
    end_override = (regime_id > 0) & (strat1 == 1)
    first_exit = end_override.groupby(regime_id).transform(lambda x: x & (~x.shift(1, fill_value=False)))
    active_override = (regime_id > 0) & ~(first_exit.groupby(regime_id).cumsum().astype(bool))

    strat2 = strat1.copy()
    strat2[active_override] = 1
    strat2[override_trigger] = 1
    strat2.iloc[0] = 0
    return strat2

def strat_3(bmom, signals, daily_returns, rebalance_thresh=0.05):
    assets = ["H5A4", "C5A0", "G5O2"]
    idx = bmom[assets[0]].index

    scores = pd.DataFrame(index=idx, columns=assets, dtype=float)
    for asset in assets:
        excess_bmom = np.maximum(0.0, bmom[asset].reindex(idx))
        scores[asset] = excess_bmom * signals[asset].reindex(idx)

    weights = pd.DataFrame(index=idx, columns=["Cash"] + assets, dtype=float)

    for i in range(len(scores)):
        row_scores = scores.iloc[i]
        total = row_scores.sum()

        if total == 0:
            weights.iloc[i] = [0.10, 0.0, 0.0, 0.90]
            continue

        cash = 0.05
        raw = (row_scores / total) * 0.95
        active_assets = raw[raw > 0].index.tolist()

        for a in active_assets:
            if raw[a] < 0.05:
                deficit = 0.05 - raw[a]
                raw[a] = 0.05
                others = [x for x in active_assets if x != a]
                if len(others) > 0:
                    other_total = raw[others].sum()
                    for o in others:
                        raw[o] -= deficit * (raw[o] / other_total)

        if raw["H5A4"] > 0.12:
            excess = raw["H5A4"] - 0.12
            raw["H5A4"] = 0.12
            if raw["C5A0"] > 0:
                raw["C5A0"] += excess
            elif raw["G5O2"] > 0:
                raw["G5O2"] += excess
            else:
                cash += excess

        weights.iloc[i] = [cash, raw["H5A4"], raw["C5A0"], raw["G5O2"]]

    if rebalance_thresh > 0:
        w = weights.values.copy()
        for i in range(1, len(w)):
            max_change = np.abs(w[i] - w[i-1]).max()
            if max_change <= rebalance_thresh:
                w[i] = w[i-1]
        weights = pd.DataFrame(w, index=weights.index, columns=weights.columns)

    portfolio_return = (
        weights["H5A4"].shift(1) * daily_returns["H5A4"].reindex(idx) +
        weights["C5A0"].shift(1) * daily_returns["C5A0"].reindex(idx) +
        weights["G5O2"].shift(1) * daily_returns["G5O2"].reindex(idx)
    ).fillna(0.0)

    equity_curve = (1 + portfolio_return).cumprod()
    weights = weights.fillna(0.0)

    return weights, portfolio_return, equity_curve, scores

def monthly_rebalance(bmom, signals, daily_returns):
    assets = ["H5A4", "C5A0", "G5O2"]
    idx = bmom[assets[0]].index

    scores = pd.DataFrame(index=idx, columns=assets, dtype=float)
    for asset in assets:
        excess_bmom = np.maximum(0.0, bmom[asset].reindex(idx))
        scores[asset] = excess_bmom * signals[asset].reindex(idx)

    rebalance_dates = scores.index.to_series().groupby(scores.index.to_period("M")).tail(1).index
    signals_df = pd.DataFrame(signals).reindex(idx)
    risk_trigger = (signals_df == 0).any(axis=1)
    risk_dates = signals_df.index[risk_trigger]
    trade_dates = sorted(set(rebalance_dates).union(set(risk_dates)))

    weights = pd.DataFrame(index=idx, columns=["Cash"] + assets, dtype=float)

    for i in range(len(scores)):
        row_scores = scores.iloc[i]
        total = row_scores.sum()

        if total == 0:
            weights.iloc[i] = [0.10, 0.0, 0.0, 0.90]
            continue

        cash = 0.05
        raw = (row_scores / total) * 0.95
        active_assets = raw[raw > 0].index.tolist()

        for a in active_assets:
            if raw[a] < 0.05:
                deficit = 0.05 - raw[a]
                raw[a] = 0.05
                others = [x for x in active_assets if x != a]
                if len(others) > 0:
                    other_total = raw[others].sum()
                    for o in others:
                        raw[o] -= deficit * (raw[o] / other_total)

        if raw["H5A4"] > 0.12:
            excess = raw["H5A4"] - 0.12
            raw["H5A4"] = 0.12
            if raw["C5A0"] > 0:
                raw["C5A0"] += excess
            elif raw["G5O2"] > 0:
                raw["G5O2"] += excess
            else:
                cash += excess

        weights.iloc[i] = [cash, raw["H5A4"], raw["C5A0"], raw["G5O2"]]

    target_weights = weights.copy()
    actual_weights = pd.DataFrame(index=idx, columns=target_weights.columns, dtype=float)
    current_weights = None

    for date in idx:
        if date in trade_dates:
            current_weights = target_weights.loc[date]
        actual_weights.loc[date] = current_weights

    actual_weights = actual_weights.ffill().fillna(0.0)

    portfolio_return_monthly = (
        actual_weights["H5A4"].shift(1) * daily_returns["H5A4"].reindex(idx) +
        actual_weights["C5A0"].shift(1) * daily_returns["C5A0"].reindex(idx) +
        actual_weights["G5O2"].shift(1) * daily_returns["G5O2"].reindex(idx)
    ).fillna(0.0)

    equity_curve_monthly = (1 + portfolio_return_monthly).cumprod()
    weights = weights.fillna(0.0)

    return actual_weights, portfolio_return_monthly, equity_curve_monthly, scores

def G502_momentum_strat(bmom, daily_returns, threshold=0.0, execution_lag=1):
    idx = bmom.index
    returns = daily_returns.reindex(idx).fillna(0.0)
    lagged_bmom = bmom.shift(1)
    signal = (lagged_bmom > threshold).astype(int)
    position = signal.shift(execution_lag).fillna(0).astype(int)
    strat_returns = position * returns
    equity = (1 + strat_returns).cumprod()

    ann_return = equity.iloc[-1] ** (252 / len(strat_returns)) - 1
    ann_vol = strat_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    time_in_market = position.mean()
    trades = (position.diff().abs() > 0).sum()

    return {
        "signal": signal,
        "position": position,
        "returns": strat_returns,
        "equity": equity,
        "drawdown": drawdown,
        "metrics": {
            "threshold": threshold,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "time_in_market": time_in_market,
            "trades": trades
        }
    }

# =============================================================================
# COMPUTE ALL STRATEGIES
# =============================================================================
print("Computing strategies...")

# Strategy 1 signals
signals_strat1_H5A4 = strat1_signals(bmom_H5A4, bmom_SPY, momentums_H5A4[240], thresh=-0.03)
signals_strat1_G5O2 = strat1_signals(bmom_G5O2, bmom_SPY, momentums_G5O2[240], thresh=-0.03)
signals_strat1_C5A0 = strat1_signals(bmom_C5A0, bmom_SPY, momentums_C5A0[240], thresh=-0.03)

# Strategy 2 signals
strat2_signals_H5A4 = strat2_from_strat1(signals_strat1_H5A4, daily_returns_H5A4, drawdiffthresh=0.10)
strat2_signals_G5O2 = strat2_from_strat1(signals_strat1_G5O2, daily_returns_G5O2, drawdiffthresh=0.10)
strat2_signals_C5A0 = strat2_from_strat1(signals_strat1_C5A0, daily_returns_C5A0, drawdiffthresh=0.10)

# G5O2 Momentum signals (threshold = 0.005 = 0.5%)
G502_result = G502_momentum_strat(bmom_G5O2, daily_returns_G5O2, threshold=0.005)
G502_momentum_signals = G502_result["signal"]

# Strat 3 (standard)
weights3, ret3, eq3, scores3 = strat_3(
    bmom={"H5A4": bmom_H5A4, "C5A0": bmom_C5A0, "G5O2": bmom_G5O2},
    signals={"H5A4": strat2_signals_H5A4, "C5A0": strat2_signals_C5A0, "G5O2": strat2_signals_G5O2},
    daily_returns={"H5A4": daily_returns_H5A4, "C5A0": daily_returns_C5A0, "G5O2": daily_returns_G5O2}
)

# Monthly Strategy (standard)
monthly_weights, monthly_ret, monthly_eq, monthly_scores = monthly_rebalance(
    bmom={"H5A4": bmom_H5A4, "C5A0": bmom_C5A0, "G5O2": bmom_G5O2},
    signals={"H5A4": strat2_signals_H5A4, "C5A0": strat2_signals_C5A0, "G5O2": strat2_signals_G5O2},
    daily_returns={"H5A4": daily_returns_H5A4, "C5A0": daily_returns_C5A0, "G5O2": daily_returns_G5O2}
)

# Strat 3 (Momentum)
weights3_mom, ret3_mom, eq3_mom, scores3_mom = strat_3(
    bmom={"H5A4": bmom_H5A4, "C5A0": bmom_C5A0, "G5O2": bmom_G5O2},
    signals={"H5A4": strat2_signals_H5A4, "C5A0": strat2_signals_C5A0, "G5O2": G502_momentum_signals},
    daily_returns={"H5A4": daily_returns_H5A4, "C5A0": daily_returns_C5A0, "G5O2": daily_returns_G5O2}
)

# Monthly Strategy (Momentum)
monthly_weights_mom, monthly_ret_mom, monthly_eq_mom, monthly_scores_mom = monthly_rebalance(
    bmom={"H5A4": bmom_H5A4, "C5A0": bmom_C5A0, "G5O2": bmom_G5O2},
    signals={"H5A4": strat2_signals_H5A4, "C5A0": strat2_signals_C5A0, "G5O2": G502_momentum_signals},
    daily_returns={"H5A4": daily_returns_H5A4, "C5A0": daily_returns_C5A0, "G5O2": daily_returns_G5O2}
)

# Benchmark
idx = bmom_G5O2.index
benchmark_ret = (
    0.095 * daily_returns_H5A4.reindex(idx).fillna(0.0) +
    0.665 * daily_returns_C5A0.reindex(idx).fillna(0.0) +
    0.190 * daily_returns_G5O2.reindex(idx).fillna(0.0)
)
benchmark_eq = (1 + benchmark_ret).cumprod()

# =============================================================================
# FIGURE 1: BMOM Signals (3-panel)
# =============================================================================
print("Generating Figure 1: BMOM Signals...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Common date range
common_idx = bmom_H5A4.index.intersection(bmom_SPY.index)

# H5A4 Panel
ax1 = axes[0]
ax1.plot(common_idx, bmom_H5A4.reindex(common_idx), 'b-', linewidth=1.2, label='H5A4 BMOM', alpha=0.8)
ax1.plot(common_idx, bmom_SPY.reindex(common_idx), 'orange', linewidth=1, label='SPY BMOM', alpha=0.6)
ax1.axhline(y=0, color='green', linestyle='-', linewidth=1.5, label='Entry Threshold (0)')
ax1.axhline(y=-0.03, color='red', linestyle='--', linewidth=1.5, label='Exit Threshold (-3%)')
ax1.set_ylabel('BMOM', fontsize=12)
ax1.set_title('H5A4 (High Yield) Blended Momentum', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.25, 0.35)

# C5A0 Panel
ax2 = axes[1]
ax2.plot(common_idx, bmom_C5A0.reindex(common_idx), 'b-', linewidth=1.2, label='C5A0 BMOM', alpha=0.8)
ax2.plot(common_idx, bmom_SPY.reindex(common_idx), 'orange', linewidth=1, label='SPY BMOM', alpha=0.6)
ax2.axhline(y=0, color='green', linestyle='-', linewidth=1.5)
ax2.axhline(y=-0.03, color='red', linestyle='--', linewidth=1.5)
ax2.set_ylabel('BMOM', fontsize=12)
ax2.set_title('C5A0 (Investment Grade) Blended Momentum', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.25, 0.35)

# G5O2 Panel
ax3 = axes[2]
ax3.plot(common_idx, bmom_G5O2.reindex(common_idx), 'b-', linewidth=1.2, label='G5O2 BMOM', alpha=0.8)
ax3.plot(common_idx, bmom_SPY.reindex(common_idx), 'orange', linewidth=1, label='SPY BMOM', alpha=0.6)
ax3.axhline(y=0, color='green', linestyle='-', linewidth=1.5)
ax3.axhline(y=-0.03, color='red', linestyle='--', linewidth=1.5)
ax3.set_ylabel('BMOM', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_title('G5O2 (Government) Blended Momentum', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.25, 0.35)

ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_bmom_signals.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: fig_bmom_signals.png")

# =============================================================================
# FIGURE 3: Strategy Comparison (Main 5-strategy chart)
# =============================================================================
print("Generating Figure 3: Strategy Comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(eq3.index, eq3, 'navy', linewidth=2, label='Strat 3', alpha=0.9)
ax.plot(monthly_eq.index, monthly_eq, 'red', linewidth=2, label='Monthly Strat', alpha=0.9)
ax.plot(eq3_mom.index, eq3_mom, 'dodgerblue', linewidth=2, linestyle='--', label='Strat 3 (Momentum)', alpha=0.9)
ax.plot(monthly_eq_mom.index, monthly_eq_mom, 'salmon', linewidth=2, linestyle='--', label='Monthly (Momentum)', alpha=0.9)
ax.plot(benchmark_eq.index, benchmark_eq, 'grey', linewidth=2, linestyle=':', label='Benchmark', alpha=0.8)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right')

ax.set_title('Strategy Comparison: All Strategies vs Benchmark', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Equity (Growth of $1)', fontsize=13)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_strategy_comparison.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: fig_strategy_comparison.png")

# =============================================================================
# FIGURE 4: G5O2 Threshold Optimization
# =============================================================================
print("Generating Figure 4: G5O2 Threshold Optimization...")

# Run threshold optimization
thresholds = np.arange(-0.06, 0.065, 0.005)
split_idx = int(len(bmom_G5O2) * 0.6)
train_bmom = bmom_G5O2.iloc[:split_idx]
train_returns = daily_returns_G5O2.reindex(train_bmom.index)

results = []
for thresh in thresholds:
    result = G502_momentum_strat(train_bmom, train_returns, threshold=thresh)
    results.append({
        "threshold": thresh,
        "sharpe_train": result["metrics"]["sharpe"],
        "return_train": result["metrics"]["ann_return"],
    })

results_df = pd.DataFrame(results)
best_idx = results_df["sharpe_train"].idxmax()
optimal_threshold = results_df.loc[best_idx, "threshold"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sharpe vs Threshold
ax1 = axes[0]
ax1.plot(results_df["threshold"] * 100, results_df["sharpe_train"], 'b-o', linewidth=2, markersize=6)
ax1.axvline(x=optimal_threshold * 100, color='r', linestyle='--', linewidth=2,
            label=f'Optimal: {optimal_threshold:.1%}')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.set_xlabel("Threshold (%)", fontsize=12)
ax1.set_ylabel("Sharpe Ratio (Training)", fontsize=12)
ax1.set_title("G5O2 Momentum: Sharpe vs Threshold", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Equity Curves
ax2 = axes[1]
full_result = G502_momentum_strat(bmom_G5O2, daily_returns_G5O2, threshold=optimal_threshold)
bh_equity = (1 + daily_returns_G5O2.reindex(bmom_G5O2.index).fillna(0)).cumprod()

ax2.plot(full_result["equity"].index, full_result["equity"], 'b-', linewidth=2,
         label=f'G5O2 Momentum (thresh={optimal_threshold:.1%})')
ax2.plot(bh_equity.index, bh_equity, 'gray', linewidth=1.5, linestyle='--',
         label='Buy & Hold G5O2', alpha=0.7)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Equity", fontsize=12)
ax2.set_title("G5O2 Momentum Strategy vs Buy & Hold", fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_g502_threshold.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: fig_g502_threshold.png")

# =============================================================================
# FIGURE 5: Strat 3 Weights (Stacked Area)
# =============================================================================
print("Generating Figure 5: Strat 3 Weights...")

fig, ax = plt.subplots(figsize=(14, 6))

# Resample to weekly for cleaner visualization
weights_weekly = weights3.resample('W').last()

ax.stackplot(weights_weekly.index,
             weights_weekly['H5A4'], weights_weekly['C5A0'],
             weights_weekly['G5O2'], weights_weekly['Cash'],
             labels=['H5A4 (High Yield)', 'C5A0 (Inv. Grade)', 'G5O2 (Government)', 'Cash'],
             colors=['#E74C3C', '#3498DB', '#2ECC71', '#95A5A6'],
             alpha=0.8)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right')

ax.set_title('Strat 3: Portfolio Weight Evolution', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Weight', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_strat3_weights.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: fig_strat3_weights.png")

# =============================================================================
# FIGURE 6: Monthly Weights
# =============================================================================
print("Generating Figure 6: Monthly Weights...")

fig, ax = plt.subplots(figsize=(14, 6))

weights_weekly_m = monthly_weights.resample('W').last()

ax.stackplot(weights_weekly_m.index,
             weights_weekly_m['H5A4'], weights_weekly_m['C5A0'],
             weights_weekly_m['G5O2'], weights_weekly_m['Cash'],
             labels=['H5A4 (High Yield)', 'C5A0 (Inv. Grade)', 'G5O2 (Government)', 'Cash'],
             colors=['#E74C3C', '#3498DB', '#2ECC71', '#95A5A6'],
             alpha=0.8)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right')

ax.set_title('Monthly Strategy: Portfolio Weight Evolution', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Weight', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_monthly_weights.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: fig_monthly_weights.png")

# =============================================================================
# FIGURE 7: Drawdown Comparison
# =============================================================================
print("Generating Figure 7: Drawdown Comparison...")

def compute_drawdown(equity):
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max

dd_strat3 = compute_drawdown(eq3)
dd_monthly = compute_drawdown(monthly_eq)
dd_strat3_mom = compute_drawdown(eq3_mom)
dd_monthly_mom = compute_drawdown(monthly_eq_mom)
dd_benchmark = compute_drawdown(benchmark_eq)

fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(dd_benchmark.index, dd_benchmark, 0, alpha=0.3, color='grey', label='Benchmark')
ax.plot(dd_strat3.index, dd_strat3, 'navy', linewidth=1.5, label='Strat 3', alpha=0.9)
ax.plot(dd_monthly.index, dd_monthly, 'red', linewidth=1.5, label='Monthly Strat', alpha=0.9)
ax.plot(dd_strat3_mom.index, dd_strat3_mom, 'dodgerblue', linewidth=1.5, linestyle='--',
        label='Strat 3 (Momentum)', alpha=0.8)
ax.plot(dd_monthly_mom.index, dd_monthly_mom, 'salmon', linewidth=1.5, linestyle='--',
        label='Monthly (Momentum)', alpha=0.8)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right')

ax.set_title('Drawdown Comparison: All Strategies vs Benchmark', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown', fontsize=12)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_drawdown_comparison.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Saved: fig_drawdown_comparison.png")

# =============================================================================
# PRINT SUMMARY METRICS
# =============================================================================
print("\n" + "="*70)
print("METRICS SUMMARY")
print("="*70)

def calc_metrics(returns, name):
    equity = (1 + returns).cumprod()
    ann_return = equity.iloc[-1] ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    win_rate = (returns > 0).sum() / len(returns)

    print(f"{name}: Return={ann_return:.2%}, Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}, WinRate={win_rate:.2%}")
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate}

calc_metrics(ret3, "Strat 3")
calc_metrics(monthly_ret, "Monthly Strat")
calc_metrics(ret3_mom, "Strat 3 (Momentum)")
calc_metrics(monthly_ret_mom, "Monthly (Momentum)")
calc_metrics(benchmark_ret, "Benchmark")

print("\n" + "="*70)
print("All figures saved to:", FIGURES_DIR)
print("="*70)
