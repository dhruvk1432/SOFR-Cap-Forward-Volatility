from __future__ import annotations

import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import write_manifest, write_table_artifacts
from src.curve import build_cap_curve, build_forward_vol_panel, bootstrap_discount_curve
from src.data import load_data
from src.inference import (
    forward_realized_sofr_vol,
    policy_regime,
    predictive_regressions,
    premium_summary,
    term_premium_panel,
)
from src.plotting import draw_regime_spans, plot_forward_vol_surface, save_figure
from src.pricing import black_caplet_price, normal_caplet_price
from src.strategy import (
    VolCarryConfig,
    block_bootstrap_path_summary,
    block_bootstrap_performance_ci,
    build_vol_carry_frame,
    expanding_ridge_forecast,
    forecast_validation_table,
    regime_performance,
    strategy_block_permutation_null,
    strategy_cpcv_table,
    strategy_performance,
    summarize_forecast_validation,
    summarize_strategy_cpcv,
    volatility_carry_backtest,
)
from src.validation import PurgedSplitConfig

TENORS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
TAU_LIST = [1.0, 1.5, 2.0, 3.0]
DELTA = 0.5
FEATURE_COLS = [
    "implied_vol",
    "lag_realized_vol",
    "vol_slope",
    "vol_momentum_13w",
    "vol_of_vol_13w",
]


@dataclass
class ResearchState:
    data: object
    overview: pd.DataFrame
    validation_curve: pd.DataFrame
    validation_summary: pd.DataFrame
    black_panel: pd.DataFrame
    normal_panel: pd.DataFrame
    panel_summary: pd.DataFrame
    pca_explained: pd.DataFrame
    pca_loadings: pd.DataFrame
    regression_table: pd.DataFrame
    premium: pd.DataFrame
    premium_stats: pd.DataFrame
    vrp: pd.DataFrame
    vrp_stats: pd.DataFrame
    regime_table: pd.DataFrame
    strategy_config: VolCarryConfig
    strategy_frame: pd.DataFrame
    strategy_predictions: pd.Series
    strategy_bt: pd.DataFrame
    strategy_stats: pd.DataFrame
    strategy_regime_stats: pd.DataFrame
    split_config: PurgedSplitConfig
    forecast_cpcv: pd.DataFrame
    forecast_cpcv_summary: pd.DataFrame
    strategy_cpcv: pd.DataFrame
    strategy_cpcv_summary: pd.DataFrame
    null_summary: pd.DataFrame
    null_draws: pd.DataFrame
    bootstrap_path: pd.DataFrame
    sensitivity: pd.DataFrame
    interpolation_robustness: pd.DataFrame


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _clean_for_display(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "ridge":
            out[col] = out[col].map(lambda x: f"{x:.0e}" if pd.notna(x) else x)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _data_overview(data) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": [
                len(data.cap_normal_vol_bp),
                len(data.sofr_swaps),
                len(data.sofr_daily),
            ],
            "start": [
                data.cap_normal_vol_bp.index.min().date().isoformat(),
                data.sofr_swaps.index.min().date().isoformat(),
                data.sofr_daily.index.min().date().isoformat(),
            ],
            "end": [
                data.cap_normal_vol_bp.index.max().date().isoformat(),
                data.sofr_swaps.index.max().date().isoformat(),
                data.sofr_daily.index.max().date().isoformat(),
            ],
            "columns_or_series": [
                len(data.cap_normal_vol_bp.columns),
                len(data.sofr_swaps.columns),
                1,
            ],
        },
        index=["cap normal vol quotes", "SOFR swap quotes", "daily SOFR"],
    )


def _validation_summary(curve: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.DataFrame(
        {
            "discount_error": curve["discount"] - validation["discounts"],
            "forward_rate_error": curve["forward_rate"] - validation["forwards"],
            "flat_black_vol_error": curve["flat_black_vol"] - validation["flat vols"],
            "forward_black_vol_error": curve["forward_black_vol"] - validation["fwd vols"],
        }
    )
    out = comparison.abs().agg(["max", "mean"]).T
    out["max_bp_or_abs"] = [
        out.loc["discount_error", "max"],
        out.loc["forward_rate_error", "max"] * 10000,
        out.loc["flat_black_vol_error", "max"] * 10000,
        out.loc["forward_black_vol_error", "max"] * 10000,
    ]
    return out


def _pca_tables(normal_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    changes = normal_panel.dropna().diff().dropna() * 10000
    scaled = StandardScaler().fit_transform(changes)
    pca = PCA(n_components=3).fit(scaled)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=changes.columns,
        columns=["PC1", "PC2", "PC3"],
    )
    explained = pd.DataFrame(
        {"explained_variance": pca.explained_variance_ratio_},
        index=["PC1", "PC2", "PC3"],
    )
    loadings.index.name = "tenor"
    return explained, loadings


def _regime_table(premium: pd.DataFrame, vrp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in {
        "term_premium_1Y": premium[[1.0]].rename(columns={1.0: "value"}),
        "vrp_3M": vrp[["VRP_3M"]].rename(columns={"VRP_3M": "value"}),
    }.items():
        tmp = frame.dropna().copy()
        tmp["regime"] = tmp.index.map(policy_regime)
        tmp = tmp[tmp["regime"] != "Other"]
        for regime_name, sub in tmp.groupby("regime"):
            s = sub["value"]
            rows.append(
                {
                    "measure": label,
                    "regime": regime_name,
                    "mean_bp": s.mean() * 10000,
                    "std_bp": s.std() * 10000,
                    "frac_positive": (s > 0).mean(),
                    "n": len(s),
                }
            )
    return pd.DataFrame(rows).set_index(["measure", "regime"]).sort_index()


def _strategy_configs() -> list[VolCarryConfig]:
    return [
        VolCarryConfig(
            threshold_bp=threshold,
            risk_target_bp=risk_target,
            transaction_cost_bp=cost,
            min_train_weeks=52,
        )
        for threshold in (5.0, 8.0, 12.0)
        for risk_target in (25.0, 35.0)
        for cost in (2.0, 4.0)
    ]


def _null_summary(null_draws: pd.DataFrame, bootstrap_perf: dict[str, float], bootstrap_path: pd.DataFrame) -> pd.DataFrame:
    summary = pd.Series(
        {
            "observed_total_pnl_bp": null_draws["observed_total_pnl_bp"].iloc[0],
            "observed_ann_sharpe": null_draws["observed_ann_sharpe"].iloc[0],
            "block_permutation_pvalue_total": null_draws["pvalue_total_pnl"].iloc[0],
            "block_permutation_pvalue_sharpe": null_draws["pvalue_ann_sharpe"].iloc[0],
            "null_total_pnl_mean": null_draws["total_pnl_bp"].mean(),
            "null_total_pnl_median": null_draws["total_pnl_bp"].median(),
            "null_total_pnl_5pct": null_draws["total_pnl_bp"].quantile(0.05),
            "null_total_pnl_95pct": null_draws["total_pnl_bp"].quantile(0.95),
            "null_sharpe_mean": null_draws["ann_sharpe"].mean(),
            "null_sharpe_median": null_draws["ann_sharpe"].median(),
            "bootstrap_path_final_mean": bootstrap_path["mean_path"].iloc[-1],
            "bootstrap_path_final_median": bootstrap_path["median_path"].iloc[-1],
            **bootstrap_perf,
        },
        name="sofr_robustness_nulls",
    )
    return summary.to_frame()


def _sensitivity(normal_panel: pd.DataFrame, sofr_daily: pd.Series, base_config: VolCarryConfig) -> pd.DataFrame:
    rows = []
    for realized_window in (42, 63, 84):
        sens_frame = build_vol_carry_frame(
            normal_panel,
            sofr_daily,
            tenor=base_config.tenor,
            realized_window_days=realized_window,
        )
        sens_preds = expanding_ridge_forecast(
            sens_frame,
            FEATURE_COLS,
            min_train=base_config.min_train_weeks,
        )
        for threshold in (5.0, 8.0, 12.0):
            for cost in (1.0, 2.0, 4.0):
                sens_cfg = VolCarryConfig(
                    tenor=base_config.tenor,
                    realized_window_days=realized_window,
                    min_train_weeks=52,
                    threshold_bp=threshold,
                    transaction_cost_bp=cost,
                    risk_target_bp=base_config.risk_target_bp,
                    max_abs_position=base_config.max_abs_position,
                )
                sens_bt = volatility_carry_backtest(sens_frame, sens_preds, sens_cfg)
                rows.append(
                    {
                        "realized_window_days": realized_window,
                        "threshold_bp": threshold,
                        "transaction_cost_bp": cost,
                        **strategy_performance(sens_bt),
                    }
                )
    return pd.DataFrame(rows)


def _interpolation_robustness(data) -> pd.DataFrame:
    _, cubic = build_forward_vol_panel(data, tenors=TENORS, max_tenor=max(TENORS), interpolation="cubic")
    _, pchip = build_forward_vol_panel(data, tenors=TENORS, max_tenor=max(TENORS), interpolation="pchip")
    aligned = cubic.align(pchip, join="inner", axis=0)
    diff = (aligned[0] - aligned[1]).abs() * 10000
    rows = []
    for tenor in TENORS:
        rows.append(
            {
                "tenor": tenor,
                "mean_abs_diff_bp": diff[tenor].mean(),
                "p95_abs_diff_bp": diff[tenor].quantile(0.95),
                "max_abs_diff_bp": diff[tenor].max(),
            }
        )
    return pd.DataFrame(rows).set_index("tenor")


def load_research_state(root: Path = ROOT) -> ResearchState:
    data = load_data(root)
    overview = _data_overview(data)
    validation_curve = build_cap_curve("2025-06-30", data, interpolation="cubic")
    validation_summary = _validation_summary(validation_curve, data.validation_curve)
    black_panel, normal_panel = build_forward_vol_panel(
        data,
        tenors=TENORS,
        max_tenor=max(TENORS),
        interpolation="cubic",
    )
    panel_summary = (normal_panel * 10000).agg(["mean", "std", "min", "max"]).T
    panel_summary.columns = ["mean_bp", "std_bp", "min_bp", "max_bp"]
    pca_explained, pca_loadings = _pca_tables(normal_panel)
    regression_table = predictive_regressions(normal_panel, delta=DELTA, tau_list=TAU_LIST)
    premium = term_premium_panel(normal_panel, delta=DELTA, tau_list=TAU_LIST)
    premium_stats = premium_summary(premium, delta=DELTA)

    realized_3m = forward_realized_sofr_vol(data.sofr_daily, 63).resample("W-FRI").last()
    realized_6m = forward_realized_sofr_vol(data.sofr_daily, 126).resample("W-FRI").last()
    implied_1y = normal_panel[1.0]
    vrp = pd.DataFrame(
        {
            "VRP_3M": implied_1y - realized_3m.reindex(implied_1y.index),
            "VRP_6M": implied_1y - realized_6m.reindex(implied_1y.index),
        }
    ).dropna()
    vrp_stats = (vrp * 10000).agg(["mean", "std", "min", "max"]).T
    vrp_stats["frac_positive"] = (vrp > 0).mean()
    regime_table = _regime_table(premium, vrp)

    strategy_config = VolCarryConfig(
        tenor=0.50,
        realized_window_days=63,
        min_train_weeks=52,
        threshold_bp=8.0,
        transaction_cost_bp=2.0,
        risk_target_bp=35.0,
        max_abs_position=1.0,
    )
    strategy_frame = build_vol_carry_frame(
        normal_panel,
        data.sofr_daily,
        tenor=strategy_config.tenor,
        realized_window_days=strategy_config.realized_window_days,
    )
    strategy_predictions = expanding_ridge_forecast(
        strategy_frame,
        FEATURE_COLS,
        min_train=strategy_config.min_train_weeks,
    )
    strategy_bt = volatility_carry_backtest(strategy_frame, strategy_predictions, strategy_config)
    strategy_stats = pd.Series(strategy_performance(strategy_bt), name="vol_carry_strategy").to_frame()
    strategy_regime_stats = regime_performance(strategy_bt)

    split_config = PurgedSplitConfig(n_groups=6, n_test_groups=2, label_horizon=13, embargo=2)
    feature_sets = {
        "implied_only": ["implied_vol"],
        "implied_plus_realized": ["implied_vol", "lag_realized_vol"],
        "full_macro_curve_proxy": FEATURE_COLS,
    }
    forecast_cpcv = forecast_validation_table(
        strategy_frame,
        feature_sets,
        ridges=(1e-6, 1e-5, 1e-4),
        split_config=split_config,
        min_train=strategy_config.min_train_weeks,
    )
    forecast_cpcv_summary = summarize_forecast_validation(forecast_cpcv)
    strategy_cpcv = strategy_cpcv_table(
        strategy_frame,
        FEATURE_COLS,
        _strategy_configs(),
        ridges=(1e-5, 1e-4),
        split_config=split_config,
    )
    strategy_cpcv_summary = summarize_strategy_cpcv(strategy_cpcv)
    null_draws = strategy_block_permutation_null(
        strategy_frame,
        strategy_predictions,
        strategy_config,
        block_size=8,
        n_sims=1000,
        seed=17,
    )
    bootstrap_perf = block_bootstrap_performance_ci(
        strategy_bt["net_pnl_bp"],
        block_size=8,
        n_boot=1000,
        seed=17,
    )
    bootstrap_path = block_bootstrap_path_summary(
        strategy_bt["net_pnl_bp"],
        block_size=8,
        n_boot=1000,
        seed=17,
    )
    null_summary = _null_summary(null_draws, bootstrap_perf, bootstrap_path)
    sensitivity = _sensitivity(normal_panel, data.sofr_daily, strategy_config)
    interpolation_robustness = _interpolation_robustness(data)
    return ResearchState(
        data=data,
        overview=overview,
        validation_curve=validation_curve,
        validation_summary=validation_summary,
        black_panel=black_panel,
        normal_panel=normal_panel,
        panel_summary=panel_summary,
        pca_explained=pca_explained,
        pca_loadings=pca_loadings,
        regression_table=regression_table,
        premium=premium,
        premium_stats=premium_stats,
        vrp=vrp,
        vrp_stats=vrp_stats,
        regime_table=regime_table,
        strategy_config=strategy_config,
        strategy_frame=strategy_frame,
        strategy_predictions=strategy_predictions,
        strategy_bt=strategy_bt,
        strategy_stats=strategy_stats,
        strategy_regime_stats=strategy_regime_stats,
        split_config=split_config,
        forecast_cpcv=forecast_cpcv,
        forecast_cpcv_summary=forecast_cpcv_summary,
        strategy_cpcv=strategy_cpcv,
        strategy_cpcv_summary=strategy_cpcv_summary,
        null_summary=null_summary,
        null_draws=null_draws,
        bootstrap_path=bootstrap_path,
        sensitivity=sensitivity,
        interpolation_robustness=interpolation_robustness,
    )


def write_processed_data(state: ResearchState, root: Path = ROOT) -> list[Path]:
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in {
        "forward_black_vol_panel": state.black_panel,
        "forward_normal_vol_panel": state.normal_panel,
        "validation_curve_2025_06_30": state.validation_curve,
        "vol_carry_frame": state.strategy_frame,
        "vol_carry_backtest": state.strategy_bt,
    }.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path)
        paths.append(path)
    state.strategy_predictions.to_frame().to_csv(out_dir / "vol_carry_predictions.csv")
    paths.append(out_dir / "vol_carry_predictions.csv")
    return paths


def write_all_tables(state: ResearchState, root: Path = ROOT) -> list[Path]:
    tables = root / "tables"
    generated = root / "paper" / "generated"
    outputs: list[Path] = []
    table_map = {
        "data_overview": state.overview,
        "validation_errors": state.validation_summary,
        "forward_vol_summary": state.panel_summary,
        "pca_explained": state.pca_explained,
        "pca_loadings": state.pca_loadings,
        "expectations_regressions": state.regression_table[
            ["horizon_years", "alpha_bp", "beta", "beta_hac_se", "t_beta_eq_1", "r2", "n_weekly", "effective_nonoverlap_n"]
        ],
        "term_premium_summary": state.premium_stats,
        "realized_vol_premium": state.vrp_stats,
        "regime_premium_summary": state.regime_table,
        "strategy_stats": state.strategy_stats,
        "strategy_regime_stats": state.strategy_regime_stats,
        "forecast_cpcv_summary": state.forecast_cpcv_summary.reset_index(),
        "strategy_cpcv_summary": state.strategy_cpcv_summary.head(12).reset_index(),
        "null_bootstrap_summary": state.null_summary,
        "sensitivity_ann_sharpe": state.sensitivity.pivot_table(
            index=["realized_window_days", "threshold_bp"],
            columns="transaction_cost_bp",
            values="ann_sharpe",
        ),
        "interpolation_robustness": state.interpolation_robustness,
    }
    for stem, frame in table_map.items():
        written = write_table_artifacts(_clean_for_display(frame), stem, tables, generated, index=True)
        outputs.extend(written.values())
    return outputs


def write_all_figures(state: ResearchState, root: Path = ROOT) -> list[Path]:
    figures = root / "figures"
    paper_figures = root / "paper" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    outputs.extend(save_figure(plot_forward_vol_surface(state.normal_panel), "sofr_forward_vol_surface", figures, paper_figures).values())

    fig, ax = plt.subplots(figsize=(10, 5))
    (state.normal_panel[[0.5, 1.0, 2.0, 3.0, 5.0]] * 10000).plot(ax=ax, linewidth=1.2)
    ax.set_title("Stripped SOFR caplet forward normal vols")
    ax.set_ylabel("Normal vol, bp")
    outputs.extend(save_figure(fig, "sofr_forward_vol_panel", figures, paper_figures).values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    state.validation_summary[["max_bp_or_abs"]].plot(kind="bar", ax=ax, legend=False, color="tab:blue")
    ax.set_title("Validation errors against 2025-06-30 benchmark workbook")
    ax.set_ylabel("Max error, bp for rates/vols; abs for discount")
    ax.tick_params(axis="x", rotation=25)
    outputs.extend(save_figure(fig, "sofr_validation_errors", figures, paper_figures).values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    state.pca_loadings.plot(ax=ax, marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("PCA loadings of weekly forward-vol changes")
    ax.set_xlabel("Forward-vol tenor")
    ax.set_ylabel("Loading")
    outputs.extend(save_figure(fig, "sofr_pca_loadings", figures, paper_figures).values())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    spot = state.normal_panel[DELTA]
    for ax, tau in zip(axes.ravel(), TAU_LIST):
        horizon_weeks = int(round((tau - DELTA) * 52))
        frame = pd.DataFrame({"x": state.normal_panel[tau], "y": spot.shift(-horizon_weeks)}).dropna() * 10000
        beta = state.regression_table.loc[tau, "beta"]
        alpha = state.regression_table.loc[tau, "alpha_bp"]
        ax.scatter(frame["x"], frame["y"], s=16, alpha=0.45)
        lo = min(frame["x"].min(), frame["y"].min())
        hi = max(frame["x"].max(), frame["y"].max())
        xgrid = np.linspace(lo, hi, 100)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="unbiased line")
        ax.plot(xgrid, alpha + beta * xgrid, "k-", linewidth=1.2, label="HAC OLS fit")
        ax.set_title(f"tau={tau:g}Y, horizon={tau - DELTA:g}Y")
        ax.set_xlabel("Current forward normal vol, bp")
        ax.set_ylabel("Future 0.5Y normal vol, bp")
        ax.legend(fontsize=7)
    fig.tight_layout()
    outputs.extend(save_figure(fig, "sofr_expectations_scatter", figures, paper_figures).values())

    fig, ax = plt.subplots(figsize=(10, 5))
    (state.premium[[1.0, 1.5, 2.0, 3.0]] * 10000).plot(ax=ax, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Forward-vol term premium vs future short forward vol")
    ax.set_ylabel("Premium, normal-vol bp")
    outputs.extend(save_figure(fig, "sofr_term_premium", figures, paper_figures).values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    (state.vrp * 10000).plot(ax=axes[0], linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Forward implied normal vol minus forward realized SOFR vol")
    axes[0].set_ylabel("Premium, normal-vol bp")
    regime_plot = state.regime_table.reset_index()
    regime_plot[regime_plot["measure"].eq("vrp_3M")].plot.bar(x="regime", y="mean_bp", ax=axes[1], legend=False, color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("3M realized-vol premium by policy regime")
    axes[1].set_ylabel("Mean bp")
    fig.tight_layout()
    outputs.extend(save_figure(fig, "sofr_realized_vol_premium", figures, paper_figures).values())

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    (state.strategy_bt[["implied_vol", "predicted_realized_vol", "future_realized_vol"]] * 10000).plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("Forward vol richness: implied vs model forecast vs future realized")
    axes[0].set_ylabel("Normal-vol bp")
    state.strategy_bt["cum_net_pnl_bp"].plot(ax=axes[1], color="tab:green", linewidth=1.5)
    axes[1].set_title("Stylized forward-vol carry P&L after costs")
    axes[1].set_ylabel("Cumulative bp")
    fig.tight_layout()
    outputs.extend(save_figure(fig, "sofr_strategy_diagnostics", figures, paper_figures).values())

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    cumulative = state.strategy_bt["net_pnl_bp"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    cumulative.plot(ax=axes[0], color="tab:green", linewidth=1.5)
    axes[0].set_title("Cost-aware carry proxy cumulative P&L")
    axes[0].set_ylabel("Cumulative bp")
    drawdown.plot(ax=axes[1], color="tab:red", linewidth=1.2)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("bp")
    fig.tight_layout()
    outputs.extend(save_figure(fig, "sofr_carry_pnl_drawdown", figures, paper_figures).values())

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    plot_cv = state.forecast_cpcv_summary.reset_index()
    plot_cv = plot_cv[plot_cv["ridge"].eq(1e-5)]
    axes[0].bar(plot_cv["feature_set"], plot_cv["mean_rmse_bp"], color="tab:blue")
    axes[0].set_title("CPCV forecast RMSE")
    axes[0].set_ylabel("RMSE, bp")
    axes[0].tick_params(axis="x", rotation=25)

    plot_cpcv = state.strategy_cpcv_summary.reset_index().head(8)
    labels = [f"T{row.threshold_bp:.0f}/R{row.risk_target_bp:.0f}" for _, row in plot_cpcv.iterrows()]
    axes[1].bar(labels, plot_cpcv["mean_ann_sharpe"], color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("CPCV strategy Sharpe")
    axes[1].set_ylabel("Mean fold Sharpe")
    axes[1].tick_params(axis="x", rotation=35)

    observed_total = state.null_summary.loc["observed_total_pnl_bp", "sofr_robustness_nulls"]
    null_median = state.null_summary.loc["null_total_pnl_median", "sofr_robustness_nulls"]
    axes[2].hist(state.null_draws["total_pnl_bp"], bins=35, color="0.7", edgecolor="white")
    axes[2].axvline(observed_total, color="tab:red", linewidth=2, label="Observed")
    axes[2].axvline(null_median, color="tab:purple", linestyle="--", linewidth=1.5, label="Null median")
    axes[2].set_title("Block-permuted null")
    axes[2].set_xlabel("Total P&L, bp")
    axes[2].legend()

    axes[3].plot(state.bootstrap_path.index, state.bootstrap_path["observed_path"], color="tab:red", linewidth=1.8, label="Observed")
    axes[3].plot(state.bootstrap_path.index, state.bootstrap_path["median_path"], color="tab:blue", linewidth=1.6, label="Bootstrap median")
    axes[3].plot(state.bootstrap_path.index, state.bootstrap_path["mean_path"], color="tab:green", linewidth=1.1, linestyle="--", label="Bootstrap mean")
    axes[3].fill_between(
        state.bootstrap_path.index,
        state.bootstrap_path["lower_path"],
        state.bootstrap_path["upper_path"],
        color="tab:blue",
        alpha=0.15,
        label="90% block band",
    )
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_title("Block-bootstrap cumulative P&L paths")
    axes[3].set_xlabel("Backtest observation")
    axes[3].set_ylabel("Cumulative bp")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    outputs.extend(save_figure(fig, "sofr_robustness_summary", figures, paper_figures).values())

    return outputs


def _paper_number(state: ResearchState, key: str) -> float:
    if key == "total_pnl":
        return state.strategy_stats.loc["total_pnl_bp", "vol_carry_strategy"]
    if key == "sharpe":
        return state.strategy_stats.loc["ann_sharpe", "vol_carry_strategy"]
    if key == "max_drawdown":
        return state.strategy_stats.loc["max_drawdown_bp", "vol_carry_strategy"]
    if key == "hit_rate":
        return state.strategy_stats.loc["hit_rate", "vol_carry_strategy"] * 100
    if key == "vrp_mean":
        return state.vrp_stats.loc["VRP_3M", "mean"]
    if key == "term_premium_1y":
        return state.premium_stats.loc[1.0, "mean_bp"]
    if key == "term_premium_2y":
        return state.premium_stats.loc[2.0, "mean_bp"]
    if key == "pca_pc1":
        return 100 * state.pca_explained.loc["PC1", "explained_variance"]
    if key == "pca_pc2":
        return 100 * state.pca_explained.loc["PC2", "explained_variance"]
    if key == "pca_pc3":
        return 100 * state.pca_explained.loc["PC3", "explained_variance"]
    if key == "beta_min":
        return state.regression_table["beta"].min()
    if key == "beta_max":
        return state.regression_table["beta"].max()
    if key == "cpcv_full_rmse":
        return state.forecast_cpcv_summary.xs("full_macro_curve_proxy", level="feature_set")["mean_rmse_bp"].min()
    if key == "cpcv_implied_rmse":
        return state.forecast_cpcv_summary.xs("implied_only", level="feature_set")["mean_rmse_bp"].min()
    if key == "strategy_cpcv_positive_rate":
        return state.strategy_cpcv_summary["positive_fold_rate"].max() * 100
    if key == "strategy_cpcv_mean_sharpe":
        return state.strategy_cpcv_summary["mean_ann_sharpe"].max()
    if key == "null_total_lo":
        return state.null_summary.loc["total_pnl_lo", "sofr_robustness_nulls"]
    if key == "null_total_hi":
        return state.null_summary.loc["total_pnl_hi", "sofr_robustness_nulls"]
    if key == "pvalue_total":
        return state.null_summary.loc["block_permutation_pvalue_total", "sofr_robustness_nulls"]
    if key == "pvalue_sharpe":
        return state.null_summary.loc["block_permutation_pvalue_sharpe", "sofr_robustness_nulls"]
    raise KeyError(key)


def write_paper(state: ResearchState, root: Path = ROOT) -> Path:
    paper_dir = root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    strategy_col = "vol_carry_strategy"
    active_periods = int(round(state.strategy_stats.loc["active_periods", strategy_col]))
    turnover = state.strategy_stats.loc["turnover", strategy_col]

    replacements = {
        "@@N_DATES@@": str(len(state.data.cap_normal_vol_bp)),
        "@@START_DATE@@": str(state.data.cap_normal_vol_bp.index.min().date()),
        "@@END_DATE@@": str(state.data.cap_normal_vol_bp.index.max().date()),
        "@@VOL_05_MEAN@@": f"{state.panel_summary.loc[0.5, 'mean_bp']:.2f}",
        "@@VOL_10_MEAN@@": f"{state.panel_summary.loc[1.0, 'mean_bp']:.2f}",
        "@@PCA_PC1@@": f"{_paper_number(state, 'pca_pc1'):.1f}",
        "@@PCA_PC2@@": f"{_paper_number(state, 'pca_pc2'):.1f}",
        "@@PCA_PC3@@": f"{_paper_number(state, 'pca_pc3'):.1f}",
        "@@BETA_MIN@@": f"{_paper_number(state, 'beta_min'):.2f}",
        "@@BETA_MAX@@": f"{_paper_number(state, 'beta_max'):.2f}",
        "@@TERM_PREMIUM_1Y@@": f"{_paper_number(state, 'term_premium_1y'):.2f}",
        "@@TERM_PREMIUM_2Y@@": f"{_paper_number(state, 'term_premium_2y'):.2f}",
        "@@VRP_MEAN@@": f"{_paper_number(state, 'vrp_mean'):.2f}",
        "@@TOTAL_PNL@@": f"{_paper_number(state, 'total_pnl'):.2f}",
        "@@SHARPE@@": f"{_paper_number(state, 'sharpe'):.2f}",
        "@@MAX_DRAWDOWN@@": f"{_paper_number(state, 'max_drawdown'):.2f}",
        "@@HIT_RATE@@": f"{_paper_number(state, 'hit_rate'):.2f}",
        "@@ACTIVE_PERIODS@@": str(active_periods),
        "@@TURNOVER@@": f"{turnover:.2f}",
        "@@CPCV_FULL_RMSE@@": f"{_paper_number(state, 'cpcv_full_rmse'):.2f}",
        "@@CPCV_IMPLIED_RMSE@@": f"{_paper_number(state, 'cpcv_implied_rmse'):.2f}",
        "@@STRATEGY_CPCV_SHARPE@@": f"{_paper_number(state, 'strategy_cpcv_mean_sharpe'):.2f}",
        "@@STRATEGY_CPCV_POSITIVE@@": f"{_paper_number(state, 'strategy_cpcv_positive_rate'):.1f}",
        "@@PVALUE_TOTAL@@": f"{_paper_number(state, 'pvalue_total'):.3f}",
        "@@PVALUE_SHARPE@@": f"{_paper_number(state, 'pvalue_sharpe'):.3f}",
        "@@NULL_TOTAL_LO@@": f"{_paper_number(state, 'null_total_lo'):.2f}",
        "@@NULL_TOTAL_HI@@": f"{_paper_number(state, 'null_total_hi'):.2f}",
        "@@BOOTSTRAP_MEDIAN@@": f"{state.null_summary.loc['total_pnl_median', 'sofr_robustness_nulls']:.2f}",
    }

    text = r"""
\documentclass[12pt]{article}
\usepackage{iftex}
\ifPDFTeX
  \PackageError{SOFR_Cap_Forward_Volatility}{Compile with LuaLaTeX or XeLaTeX}{Run `lualatex SOFR_Cap_Forward_Volatility.tex`.}
\fi
\usepackage{fontspec}
\defaultfontfeatures{Ligatures=TeX}
\setmainfont{Times New Roman}
\usepackage[a4paper,top=0.85in,bottom=0.85in,left=0.95in,right=0.95in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,float,array,tabularx,hyperref,setspace,titlesec}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}
\setstretch{1.05}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 1pt}{6pt plus 1pt}
\titlespacing*{\subsection}{0pt}{8pt plus 2pt minus 1pt}{4pt plus 1pt}
\setlength{\floatsep}{8pt}
\setlength{\textfloatsep}{10pt}
\setlength{\intextsep}{8pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{2pt}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\sloppy

\title{\vspace{-0.5cm}SOFR Cap Forward Volatility, Risk Premia, and Rates-Volatility Carry}
\author{Dhruv Kohli}
\date{June 2026}

\begin{document}
\maketitle
\vspace{-0.5cm}
\noindent\rule{\textwidth}{0.4pt}

\begin{abstract}
This paper asks whether the forward-volatility curve implied by SOFR caps behaves like an expectations curve or like a compensated rates-volatility risk-premium curve. I strip caplet-level forward normal volatility from quoted SOFR cap flat vols, validate the construction against an independent benchmark workbook, and test whether longer forward-vol points forecast future short forward vol. The evidence favors the premium interpretation. Forward vols sit persistently above later 0.5Y forward vols and above forward realized SOFR volatility, while expectations-hypothesis slopes are far from one. A walk-forward ridge forecast converts the premium into a cost-aware carry screen with total stylized P\&L of @@TOTAL_PNL@@ bp and annualized Sharpe @@SHARPE@@. CPCV forecast validation, CPCV strategy folds, block-bootstrap paths, and block-permutation nulls reduce path-fit risk but do not make this a deployable caplet trading system. The trading conclusion is narrower: the SOFR cap curve contains economically meaningful rates-vol compensation, and promotion to live alpha requires caplet marks, bid/ask, skew, margin, hedge P\&L, and execution rules.
\end{abstract}

\noindent\rule{\textwidth}{0.4pt}

\section{Introduction}

SOFR caps quote a flat volatility for a package of caplets. That single number is useful on a broker screen, but it is not the marginal risk a trader actually owns. A cap is a strip of caplets; the long-maturity quote mixes near-dated policy uncertainty with later rates-volatility compensation. Stripping the cap into forward caplet volatilities is therefore not a cosmetic transformation. It changes the economic question from ``what is the average vol of this cap?'' to ``where along the forward curve is the market charging for uncertainty?''

This paper follows that marginal object. The central question is whether stripped SOFR cap forward vols behave like an expectations curve or a risk-premium curve. If the curve is an unbiased forecast, then a high 2Y or 3Y forward-vol point should mostly predict high future short-tenor vol. If the curve contains compensation, then the same rich point is better read as the price investors demand for warehousing rates convexity, policy uncertainty, and supply-demand imbalance in options markets.

That distinction is the trading story. A pure expectations curve is mostly a forecasting object. A premium curve is a potential carry and relative-value object, but only after the researcher separates forecastability from compensation and then separates both from executable P\&L. This paper therefore moves in stages: construct the forward-vol surface, test the expectations interpretation, measure the implied and realized premium, and only then ask whether a leakage-aware carry screen preserves the signal.

The ``so what'' is direct. SOFR caps are bundled flat-vol instruments; stripping them exposes where marginal rates-vol compensation sits. In this sample, the curve behaves more like a premium than an unbiased forecast. The carry screen is economically meaningful, but promotion from research signal to trade requires instrument-level marks, skew, bid/ask, margin, hedge P\&L, and execution constraints.

The contribution is deliberately disciplined. I build the caplet curve from quoted cap vols, validate the stripping engine, report HAC inference and effective non-overlapping sample counts, compare implied forward vol with future implied and realized vol, and then run a leakage-aware carry screen. The paper tells a finance story, but the claim hierarchy remains conservative: this is evidence of rates-volatility compensation and a researchable carry channel, not proof of a deployable caplet strategy.

\section{Related Literature and Research Design}

The curve-construction layer follows the standard logic of caplet stripping in interest-rate option markets. A cap can be valued as a portfolio of caplets under the Black framework \cite{black1976}, and the forward-volatility curve is obtained by solving the marginal caplet vols that reproduce quoted flat cap prices. The conceptual analogy is the same as stripping zero rates from par swap rates: the market quote is a bundled price, while the research object is the sequence of marginal forward prices.

The empirical layer is closer to tests of expectations hypotheses and risk premia than to option pricing alone. A forward rate, forward volatility, or forward variance can be read as an expectation plus compensation for bearing risk. The paper is designed to separate those two readings as far as the sample permits. Newey--West inference is used because overlapping forecast horizons create serially correlated forecast errors \cite{neweywest1987}. CPCV is used because finance data are ordered, dependent, and regime-heavy; shuffled cross-validation would give a false sense of precision.

The strategy layer is intentionally framed as a research screen. It asks whether the measured premium can be organized into a rule that uses only information available at the signal date, pays turnover costs, and survives purged validation. That is a useful step toward trading research, but it is still not the same as running an executable caplet book.

\section{Data and Public Research Contract}

The local data set contains @@N_DATES@@ SOFR cap-vol quote dates from @@START_DATE@@ through @@END_DATE@@, SOFR swap quotes, and daily SOFR reference rates. These inputs give the three ingredients needed for the research question: option prices, the rates curve used to discount and forward-price cash flows, and realized SOFR changes used to compare option-implied compensation with subsequent rate volatility.

The public data contract is intentionally narrow. Raw workbooks are not tracked in Git; the repository exposes code, tests, generated tables, generated figures, and this paper. That means a reader can audit the construction logic and the aggregate evidence without relying on private file paths or committing proprietary market workbooks. Table~\ref{tab:data_overview} is included for that reason: before interpreting any figure, the reader should know exactly what coverage the analysis is built on.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/data_overview.tex}}
\caption{Local input coverage. Raw workbooks remain outside version control; public artifacts report only aggregate diagnostics.}
\label{tab:data_overview}
\end{table}

\section{Caplet Stripping Methodology}

Let $T$ index cap maturity and let $i=1,\ldots,n(T)$ index quarterly caplets. A quoted flat volatility $\sigma_T^{\mathrm{flat}}$ prices the full cap:
\begin{equation}
C(T,K,\sigma_T^{\mathrm{flat}})=\sum_i Z_i B(F_i,K,\sigma_T^{\mathrm{flat}},t_i),
\end{equation}
where $Z_i$ is the discount factor, $F_i$ is the forward SOFR rate, $K$ is the cap strike, and $B(\cdot)$ is the Black caplet price. The stripped marginal caplet volatility solves the residual equation:
\begin{align}
R_i &= C(T_i,K,\sigma_{T_i}^{\mathrm{flat}})
      -\sum_{j<i} Z_jB(F_j,K,\sigma_j^{\mathrm{fwd}},t_j), \\
R_i &= Z_iB(F_i,K,\sigma_i^{\mathrm{fwd}},t_i).
\end{align}
Economically, this residual equation asks how much volatility must be assigned to the next marginal caplet after the earlier caplets have already been paid for. That is why stripping is more informative than charting flat cap vols. A flat 3Y cap vol can be high because the first year is expensive, because the third year is expensive, or because every point on the strip is expensive. The forward-vol curve separates those cases.

Quoted normal vols are converted into Black-equivalent flat vols using the ATM bridge
\begin{equation}
\sigma_T^B \approx \frac{\sigma_T^N}{F_T},
\end{equation}
stripped through the Black caplet engine, and converted back to normal-vol units for inference. This bridge is an implementation device, not a claim that the normal and lognormal models are identical. The final object is reported in normal-vol basis points because that is the cleanest unit for comparing implied rates vol, realized SOFR volatility, and carry-screen P\&L.

Validation is the first empirical result, not a back-office detail. If the stripping engine cannot reproduce an independent workbook, then any premium estimate could simply be a curve-construction artifact. Table~\ref{tab:validation} and Figure~\ref{fig:validation_errors} show the benchmark check across discount factors, forwards, flat Black vols, and stripped forward Black vols before the paper uses the surface for inference.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/validation_errors.tex}}
\caption{Curve-validation errors on the 2025-06-30 benchmark workbook. Rate and vol errors are reported both in native units and basis-point-scaled diagnostics.}
\label{tab:validation}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/sofr_validation_errors.pdf}
\caption{Independent workbook validation of discount factors, forwards, flat Black vols, and stripped forward Black vols.}
\label{fig:validation_errors}
\end{figure}

\section{Empirical Objects}

The first empirical object is an expectations-hypothesis analog for forward volatility. For short tenor $\delta=0.5$ and longer tenor $\tau$, let $h=\tau-\delta$. The regression is
\begin{equation}
\sigma^N_{t+h,\delta}
= \alpha_\tau + \beta_\tau \sigma^N_{t,\tau} + \varepsilon_{t+h,\tau}.
\end{equation}
If the longer forward-vol point were an unbiased forecast of future short forward vol, the slope would be near one after accounting for measurement error and risk-premium terms. In this sample, forecast horizons overlap heavily. The paper therefore reports HAC standard errors with lags matched to the horizon and an effective non-overlapping sample count.

The second object is the ex post forward-volatility term premium,
\begin{equation}
\mathrm{VTP}_{t,\tau} =
\sigma^N_{t,\tau} - \sigma^N_{t+h,0.5}.
\end{equation}
This is a realized premium measure: it compares the long forward-vol point observed at time $t$ with the short forward-vol point eventually observed at $t+h$. I also compare implied forward vol with forward realized SOFR volatility. That realized-vol benchmark is economically important because option sellers ultimately care about realized rate volatility, not only about later implied-vol marks.

The third object is a walk-forward investability screen. Let $\widehat{\sigma}^{RV}_{t+1}$ denote an expanding-window forecast of forward realized SOFR volatility using information available at $t$. The signal is
\begin{equation}
q_t = \sigma^N_{t,0.5} - \widehat{\sigma}^{RV}_{t+1}.
\end{equation}
The screen sells forward vol when $q_t$ is above a threshold, buys forward vol when it is below the negative threshold, scales exposure to a volatility-risk target, and charges normal-vol transaction costs on turnover. This design is deliberately less ambitious than a production trade. It is a test of whether a lagged volatility-risk-premium signal has enough structure to justify deeper modeling.

\section{Empirical Results}

The surface in Figure~\ref{fig:surface} is the main object of the paper. Each row is a quote date and each column is a stripped forward tenor, so the figure is not a single curve but a weekly time series of curves. The front of the surface moves sharply when near-term policy uncertainty is repriced. The longer forward points are smoother and more persistent, which is exactly where a risk-premium interpretation becomes economically plausible.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{figures/sofr_forward_vol_surface.pdf}
\caption{Stripped SOFR forward normal-volatility surface. Each date carries a full forward-tenor curve, so the figure shows both cross-sectional term structure and time-series evolution.}
\label{fig:surface}
\end{figure}

The weekly forward-vol panel is sampled at Friday frequency so the inference works with a stable curve object rather than noisy daily quote changes. Table~\ref{tab:forward_vol_summary} shows a clear hump: the 0.5Y point averages @@VOL_05_MEAN@@ bp, the 1Y point averages @@VOL_10_MEAN@@ bp, and the middle of the curve remains richer than the far end. That shape already hints that the market is not simply projecting one constant future volatility level.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/forward_vol_summary.tex}}
\caption{Distribution of stripped forward normal vols, in basis points.}
\label{tab:forward_vol_summary}
\end{table}

Figure~\ref{fig:forward_panel} turns the surface into time series by tenor. The reason to show it separately is that the economics are easier to see in motion: short forward vol jumps when the market reprices immediate policy risk, while longer tenors move less violently and retain a persistent cushion. A trader would read that cushion as possible compensation for selling convexity into uncertain future rate regimes, not as a clean point forecast.

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/sofr_forward_vol_panel.pdf}
\caption{Stripped forward normal volatility by tenor.}
\label{fig:forward_panel}
\end{figure}

PCA formalizes the same visual point. The first three components explain @@PCA_PC1@@\%, @@PCA_PC2@@\%, and @@PCA_PC3@@\% of weekly forward-vol changes. PC1 is broad level repricing, PC2 rotates the front against the back, and PC3 loads most strongly on the far end. The surface therefore has economically distinct level, slope, and curvature movements; reducing it to one average cap vol would throw away the part of the curve where relative value lives.

\begin{figure}[H]
\centering
\includegraphics[width=0.86\textwidth]{figures/sofr_pca_loadings.pdf}
\caption{PCA loadings of weekly forward-vol changes.}
\label{fig:pca}
\end{figure}

The first formal test asks whether the forward-vol curve is just an expectations curve. The expectations-hypothesis analog compares today's $\tau$ forward vol with the future 0.5Y forward vol after $\tau-0.5$ years. If the market were only forecasting future short vol, the slope should be near one and the intercept should be economically small.

That is not what the data show. Across the tested tenors, HAC slopes range from @@BETA_MIN@@ to @@BETA_MAX@@, far below the unbiased benchmark of one. The effective non-overlapping sample count also falls quickly with horizon, so the paper does not pretend this is a large-sample structural estimate. The narrower but important conclusion is that the expectations-only interpretation fails in the sample.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/expectations_regressions.tex}}
\caption{Expectations-hypothesis regressions with HAC standard errors.}
\label{tab:expectations}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{figures/sofr_expectations_scatter.pdf}
\caption{Forward-vol expectations tests. The dashed line is the unbiased forecast line; the fitted HAC OLS slopes are far from one.}
\label{fig:expectations}
\end{figure}

Figure~\ref{fig:expectations} is included to make the regression result visually auditable. If longer forward vol were an unbiased forecast, the cloud would organize around the dashed line. Instead the fitted lines are much flatter. In trading language, high forward vol is not reliably followed by equally high realized short forward vol; part of the quote appears to be compensation paid to the option seller.

The second test measures that compensation directly. Forward-vol term premia are positive across the main tenors: the 1Y premium averages @@TERM_PREMIUM_1Y@@ bp and the 2Y premium averages @@TERM_PREMIUM_2Y@@ bp relative to later 0.5Y forward vol. Comparing 1Y implied forward vol with forward realized SOFR volatility gives a mean 3M realized-vol premium of @@VRP_MEAN@@ bp. These numbers are the economic bridge between the stripping exercise and the carry screen: the curve is not only statistically inconsistent with expectations, it is also rich relative to subsequent realized rate movement.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/term_premium_summary.tex}}
\caption{Forward-vol term premia relative to future 0.5Y forward vol.}
\label{tab:term_premium}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/sofr_term_premium.pdf}
\caption{Forward-vol term premium relative to future short forward vol.}
\label{fig:term_premium}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/sofr_realized_vol_premium.pdf}
\caption{Forward implied normal vol minus forward realized SOFR volatility, including policy-regime attribution.}
\label{fig:realized_vrp}
\end{figure}

Figure~\ref{fig:realized_vrp} adds the regime layer. The premium is not constant through time; it is largest when the policy path is uncertain and when realized rate movement later disappoints the implied price. That is financially intuitive. Rates-vol sellers are paid most when the market wants protection against policy uncertainty, but the realized payoff depends on whether that uncertainty actually converts into future SOFR moves.

\section{Investability Screen}

The strategy layer asks a different question from the regressions. The regressions ask whether forward vol looks like an unbiased forecast. The investability screen asks whether that premium can be organized into a rule that would have been knowable at the time. The screen forecasts forward realized SOFR volatility with expanding-window ridge regression, uses only lagged features, sells forward vol when implied vol is rich versus the forecast, buys when it is cheap, applies turnover costs, and scales risk only with forecast errors whose full label horizon has elapsed.

The economic interpretation is deliberately modest. A positive signal means the curve looks expensive relative to a lagged forecast of realized rates volatility. It does not mean the desk can immediately sell a caplet and earn the plotted P\&L. The screen ignores skew, discrete strikes, caplet liquidity, margin, delta-hedging slippage, funding, liquidation constraints, and dealer bid/ask. Its purpose is to test whether the premium survives a first pass through time ordering, costs, turnover, and drawdown.

The base screen produces total stylized P\&L of @@TOTAL_PNL@@ bp, annualized Sharpe @@SHARPE@@, max drawdown @@MAX_DRAWDOWN@@ bp, active hit rate @@HIT_RATE@@\%, @@ACTIVE_PERIODS@@ active weekly observations, and turnover of @@TURNOVER@@. These are research-proxy statistics in normal-vol basis points, not caplet mark-to-market P\&L.

\begin{table}[H]
\centering
\scriptsize
\resizebox{0.72\textwidth}{!}{\input{generated/strategy_stats.tex}}
\caption{Base forward-vol carry screen diagnostics. P\&L is a research proxy in normal-vol basis points.}
\label{tab:strategy_stats}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/sofr_strategy_diagnostics.pdf}
\caption{Walk-forward implied vol, predicted realized vol, future realized vol, and cumulative carry proxy P\&L.}
\label{fig:strategy_diagnostics}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{figures/sofr_carry_pnl_drawdown.pdf}
\caption{Cumulative P\&L and drawdown for the stylized carry proxy.}
\label{fig:pnl_drawdown}
\end{figure}

Figures~\ref{fig:strategy_diagnostics} and~\ref{fig:pnl_drawdown} should be read together. The first shows when the model thinks implied forward vol is rich or cheap relative to future realized volatility; the second shows the path cost of acting on that view. The drawdown is large enough to matter, which is why the paper treats the result as a research signal rather than a finished trading product. A good signal that cannot be held through realistic drawdowns is not yet a strategy.

\section{Robustness and Claim Audit}

The short post-2022 sample and overlapping realized-vol labels create overfit risk. This is the main statistical danger in the project: a rates-vol premium can look attractive simply because one regime was favorable and the label windows overlap. The repo therefore uses CPCV forecast validation, CPCV strategy folds, feature ablations, block-bootstrap path intervals, block-permutation nulls, and parameter sensitivity checks.

CPCV is used because finance data are ordered, dependent, and regime-heavy. Random folds would leak information across overlapping horizons; a single chronological split would make the conclusion too dependent on one train-test cut. The CPCV design purges overlapping labels and recombines blocked folds, giving a more honest view of whether the signal survives multiple out-of-sample paths.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/forecast_cpcv_summary.tex}}
\caption{CPCV forecast RMSE by feature set and ridge penalty.}
\label{tab:forecast_cpcv}
\end{table}

The forecast table gives a useful sanity check on what the model is doing. The full feature set reaches a CPCV RMSE of @@CPCV_FULL_RMSE@@ bp, compared with @@CPCV_IMPLIED_RMSE@@ bp for the implied-only baseline. The gain is not magical, but it says that curve shape, realized-vol history, and macro/rates proxies add information beyond the current implied level.

The strategy is tested across pre-declared configurations rather than optimized on the full path. Thresholds, risk targets, transaction costs, and ridge penalties are evaluated with CPCV-style folds. The best average CPCV Sharpe is @@STRATEGY_CPCV_SHARPE@@, with positive P\&L in @@STRATEGY_CPCV_POSITIVE@@\% of folds. This matters because a single attractive backtest can be a parameter accident; a signal that stays near the top across purged folds is harder to dismiss as pure tuning noise.

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/strategy_cpcv_summary.tex}}
\caption{CPCV strategy diagnostics across pre-declared thresholds, risk targets, costs, and ridge penalties.}
\label{tab:strategy_cpcv}
\end{table}

\begin{table}[H]
\centering
\scriptsize
\resizebox{\textwidth}{!}{\input{generated/null_bootstrap_summary.tex}}
\caption{Block-permutation null and block-bootstrap uncertainty for the base carry screen.}
\label{tab:null_bootstrap}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{figures/sofr_robustness_summary.pdf}
\caption{Robustness dashboard: CPCV forecast validation, CPCV strategy Sharpe, block-permuted null distribution, and block-bootstrap cumulative P\&L paths.}
\label{fig:robustness}
\end{figure}

The robustness dashboard in Figure~\ref{fig:robustness} is the claim audit in one picture. The forecast panel asks whether the prediction problem is stable. The strategy-CPCV panel asks whether the carry rule survives multiple purged train-test paths. The permutation panel asks whether the realized P\&L is unusual relative to a block-shuffled timing null. The bootstrap panel asks how fragile the path is to sampling uncertainty.

The observed total-P\&L block-permutation p-value is @@PVALUE_TOTAL@@; the Sharpe p-value is @@PVALUE_SHARPE@@. The bootstrap 90\% final-P\&L interval runs from @@NULL_TOTAL_LO@@ bp to @@NULL_TOTAL_HI@@ bp, with a median final path of @@BOOTSTRAP_MEDIAN@@ bp. This is economically supportive but below a strict 5\% threshold, and the interval still includes adverse outcomes. The right interpretation is that the premium remains researchable after anti-overfit controls, while execution data are still required before making a live-alpha claim.

\section{Trading Interpretation and Production Gap}

The clean research object is forward normal volatility. The tradable object would be a caplet, cap, floor, swaption overlay, or structured package with actual bid/ask, skew, hedge P\&L, funding, margin, and liquidation rules. That difference is not a technical footnote; it is the line between an empirical premium and live alpha.

The finance implication is still valuable. A rates-vol desk could use this work to decide where the cap curve looks rich, which tenors deserve closer caplet-level marking, and when implied vol is high relative to a lagged realized-vol forecast. A researcher could extend it into skew-aware caplet valuation, executable quote collection, hedged P\&L, and portfolio construction. But the current repo stops at the research artifact. It demonstrates instrument construction, validation, inference, CPCV evaluation, cost-aware screening, and claim discipline; it does not pretend that a stylized realized-vol payoff is an executable options book.

That boundary is especially important for recruiters and traders reading the repo. The project shows the right instincts: define the instrument, build the curve, validate the engine, test the economics, run leakage-aware evaluation, and state where the claim stops. The remaining work is not better language. It is product-level data: executable caplet or cap quotes, strike conventions, skew, bid/ask, margin, funding, hedging, and portfolio risk controls.

\section{Conclusion}

SOFR cap forward vols in this sample look more like compensated rates-vol risk premia than unbiased forecasts of future short forward vol. The paper reaches that conclusion through a sequence rather than a single chart: the stripped surface reveals a persistent forward-vol hump, expectations regressions reject the one-for-one forecast interpretation, term-premium and realized-vol comparisons show positive compensation, and a leakage-aware carry screen preserves the signal after costs and CPCV-style validation.

The final conclusion is intentionally bounded. The result justifies deeper caplet-level execution research and is strong enough to be useful for rates-vol relative-value discussion. It does not yet prove deployable rates-options alpha. That distinction is the point of the paper: good quant research should find the premium, test whether it survives realistic statistical pressure, and then state clearly what still has to be true before capital can be put behind it.

\appendix
\section{Claim-by-Claim Audit}

\begin{table}[H]
\centering
\scriptsize
\begin{tabular}{L{0.23\textwidth}L{0.33\textwidth}L{0.34\textwidth}}
\toprule
Claim & Implemented evidence & Allowed interpretation \\
\midrule
Forward vols are stripped from caps & Cap flat vols are repriced as caplet portfolios and validated against a benchmark workbook. & The repo implements the stripping workflow rather than relying on precomputed forward vols. \\
\addlinespace
The curve is not an unbiased expectations curve & HAC regressions compare longer forward vols with future 0.5Y forward vols. & Strong descriptive evidence of a premium; not a large-sample structural estimate. \\
\addlinespace
The premium is economically meaningful & Term-premium and realized-vol comparisons are persistently positive. & SOFR cap sellers appear compensated for rates-vol uncertainty in this sample. \\
\addlinespace
The carry signal is researchable & Walk-forward forecasts, costs, lagged risk scaling, regime attribution, and robustness checks are implemented. & The signal deserves product-level execution work; it is not yet a caplet strategy. \\
\addlinespace
The result is not just a path-fit artifact & CPCV forecast validation, CPCV strategy folds, feature ablations, sensitivity tables, block-bootstrap paths, and block-permutation null tests are implemented. & The evidence is robust enough for a research artifact, but the bootstrap interval still crosses zero. \\
\addlinespace
The public repo is safe to inspect & Raw workbooks and private tooling are outside version control; generated outputs carry aggregate provenance. & Recruiters can read code, tables, figures, and paper without private data exposure. \\
\bottomrule
\end{tabular}
\caption{Claim hierarchy for the public repository.}
\label{tab:claim_audit}
\end{table}

\section{How to Read the Repository}

Start with this paper for the research object and claim hierarchy. Then read \texttt{README.md} for the reproduction contract and \texttt{run\_all.py} for the end-to-end pipeline. The numbered scripts in \texttt{scripts/} regenerate processed curves, tables, figures, paper inputs, TeX integrity checks, and final artifact hashes. The notebook in \texttt{notebooks/} is a walkthrough only; paper numbers are generated by scripts.

The reusable implementation is split by responsibility: \texttt{src/data.py} loads local inputs, \texttt{src/curve.py} builds the rates and cap-vol panels, \texttt{src/pricing.py} contains option pricing and implied-vol routines, \texttt{src/validation.py} checks the benchmark workbook, \texttt{src/inference.py} runs premium and forecast tests, \texttt{src/strategy.py} implements the carry screen and CPCV validation, \texttt{src/plotting.py} writes figures, and \texttt{src/artifacts.py} writes provenance. The compatibility shim \texttt{src/sofr\_forward\_vol.py} exists so older imports do not break, but the modular files are the public code path.

\begin{thebibliography}{9}
\bibitem{black1976} Black, F. (1976). The pricing of commodity contracts. \textit{Journal of Financial Economics}, 3(1--2):167--179.
\bibitem{andersenpiterbarg2010} Andersen, L. and Piterbarg, V. (2010). \textit{Interest Rate Modeling}. Atlantic Financial Press.
\bibitem{neweywest1987} Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. \textit{Econometrica}, 55(3):703--708.
\end{thebibliography}

\end{document}
    """
    for token, value in replacements.items():
        text = text.replace(token, value)

    path = paper_dir / "SOFR_Cap_Forward_Volatility.tex"
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def manifest_paths(root: Path = ROOT) -> list[Path]:
    patterns = [
        "tables/*.csv",
        "tables/*.tex",
        "figures/*.png",
        "paper/generated/*.tex",
        "paper/figures/*.pdf",
        "paper/SOFR_Cap_Forward_Volatility.tex",
        "paper/SOFR_Cap_Forward_Volatility.pdf",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted({p for p in paths if p.exists() and p.is_file()})
