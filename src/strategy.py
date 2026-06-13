from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .inference import (
                    _align_daily_to_weekly,
                    forward_realized_sofr_vol,
                    policy_regime,
                    trailing_realized_sofr_vol,
                )
from .validation import (
                    PurgedSplitConfig,
                    combinatorial_purged_splits,
                    purged_blocked_splits,
                )


@dataclass
class VolCarryConfig:
    """Configuration for the stylized forward-volatility carry backtest.

    The strategy is intentionally expressed in volatility points rather than as
    a live caplet execution engine.  It tests whether implied forward vol is
    rich or cheap versus a strictly out-of-sample realized-vol forecast before
    any claim about a tradable cap structure is made.
    """

    tenor: float = 0.50
    realized_window_days: int = 63
    min_train_weeks: int = 52
    threshold_bp: float = 8.0
    transaction_cost_bp: float = 2.0
    risk_target_bp: float = 35.0
    max_abs_position: float = 1.0

    @property
    def label_horizon_rows(self) -> int:
        """Rows before forward-realized-vol outcomes can be observed."""

        return max(1, int(np.ceil(self.realized_window_days / 5.0)))


def build_vol_carry_frame(
    normal_panel: pd.DataFrame,
    sofr_daily: pd.Series,
    tenor: float = 0.50,
    realized_window_days: int = 63,
) -> pd.DataFrame:
    """Build an out-of-sample forecasting frame for forward-vol carry.

    Columns are in decimal volatility units.  ``future_realized_vol`` is only
    used as the label and payoff-settlement proxy; every feature is observable
    at the weekly signal date.
    """

    if tenor not in normal_panel.columns:
        raise KeyError(f"Tenor {tenor} is not present in the forward-vol panel.")

    panel = normal_panel.sort_index().astype(float)
    weekly_index = panel.index
    trailing = _align_daily_to_weekly(
        trailing_realized_sofr_vol(sofr_daily, realized_window_days),
        weekly_index,
    )
    future = _align_daily_to_weekly(
        forward_realized_sofr_vol(sofr_daily, realized_window_days),
        weekly_index,
    )

    curve_max = max(panel.columns.astype(float))
    frame = pd.DataFrame(index=weekly_index)
    frame["implied_vol"] = panel[tenor]
    frame["lag_realized_vol"] = trailing
    frame["vol_slope"] = panel[curve_max] - panel[tenor]
    frame["vol_momentum_13w"] = panel[tenor] - panel[tenor].shift(13)
    frame["vol_of_vol_13w"] = panel[tenor].diff().rolling(13, min_periods=6).std()
    frame["future_realized_vol"] = future
    frame["policy_regime"] = [policy_regime(date) for date in frame.index]
    return frame.dropna(subset=["implied_vol", "future_realized_vol"])


def expanding_ridge_forecast(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "future_realized_vol",
    min_train: int = 52,
    ridge: float = 1e-5,
) -> pd.Series:
    """Expanding-window ridge forecasts with train-sample standardization.

    The function refits at every row using only prior observations.  Ridge is
    used for numerical stability because the sample is short and features are
    naturally collinear.
    """

    features = list(feature_cols)
    clean = frame[features + [target_col]].astype(float)
    predictions = pd.Series(index=frame.index, dtype=float, name="predicted_realized_vol")

    for i, date in enumerate(clean.index):
        train = clean.iloc[:i].dropna()
        current = clean.loc[[date], features].dropna()
        if len(train) < min_train or current.empty:
            continue

        x_train = train[features]
        y_train = train[target_col]
        mu = x_train.mean()
        sigma = x_train.std(ddof=0).replace(0.0, 1.0)
        x = (x_train - mu) / sigma
        x = np.column_stack([np.ones(len(x)), x.values])
        y = y_train.values

        penalty = np.eye(x.shape[1]) * ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)

        x_now = (current.iloc[0] - mu) / sigma
        pred = float(np.r_[1.0, x_now.values] @ beta)
        predictions.loc[date] = max(pred, 1e-8)

    return predictions


def _ridge_predict_train_test(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_col: str = "future_realized_vol",
    ridge: float = 1e-5,
    min_train: int = 52,
) -> pd.Series:
    features = list(feature_cols)
    clean = frame[features + [target_col]].astype(float)
    train = clean.iloc[train_idx].dropna()
    test = clean.iloc[test_idx].dropna(subset=features)
    predictions = pd.Series(index=frame.index[test_idx], dtype=float, name="prediction")
    if len(train) < min_train or test.empty:
        return predictions

    x_train = train[features]
    y_train = train[target_col]
    mu = x_train.mean()
    sigma = x_train.std(ddof=0).replace(0.0, 1.0)
    x = (x_train - mu) / sigma
    x = np.column_stack([np.ones(len(x)), x.values])
    y = y_train.values

    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)

    x_test = (test[features] - mu) / sigma
    pred = np.column_stack([np.ones(len(x_test)), x_test.values]) @ beta
    predictions.loc[test.index] = np.maximum(pred, 1e-8)
    return predictions


def forecast_validation_table(
    frame: pd.DataFrame,
    feature_sets: dict[str, Iterable[str]],
    ridges: Iterable[float] = (1e-6, 1e-5, 1e-4),
    split_config: PurgedSplitConfig | None = None,
    min_train: int = 52,
) -> pd.DataFrame:
    """Evaluate forecast specifications with combinatorial purged CV.

    Forecast-model selection uses CPCV because financial labels overlap,
    adjacent market states are dependent, and a single chronological fold
    can understate path-selection risk in short rates-vol samples.
    """

    cfg = split_config or PurgedSplitConfig()
    splits = combinatorial_purged_splits(frame.index, cfg)
    rows = []
    target = frame["future_realized_vol"].astype(float)
    for feature_name, cols in feature_sets.items():
        cols = list(cols)
        for ridge in ridges:
            for fold, (train_idx, test_idx) in enumerate(splits):
                preds = _ridge_predict_train_test(
                    frame,
                    cols,
                    train_idx,
                    test_idx,
                    ridge=float(ridge),
                    min_train=min_train,
                )
                aligned = pd.DataFrame({"prediction": preds, "actual": target}).dropna()
                if aligned.empty:
                    continue
                error = aligned["prediction"] - aligned["actual"]
                corr = aligned["prediction"].corr(aligned["actual"])
                rows.append(
                    {
                        "feature_set": feature_name,
                        "ridge": float(ridge),
                        "validation_scheme": "combinatorial_purged_cv",
                        "fold": fold,
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(aligned)),
                        "rmse_bp": float(np.sqrt(np.mean(np.square(error))) * 10000.0),
                        "mae_bp": float(np.mean(np.abs(error)) * 10000.0),
                        "bias_bp": float(error.mean() * 10000.0),
                        "corr": float(corr) if pd.notna(corr) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_forecast_validation(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level forecast validation into public-facing diagnostics."""

    if table.empty:
        return pd.DataFrame()
    grouped = table.groupby(["feature_set", "ridge"], dropna=False)
    out = grouped.agg(
        folds=("fold", "nunique"),
        mean_test_n=("test_n", "mean"),
        mean_rmse_bp=("rmse_bp", "mean"),
        mean_mae_bp=("mae_bp", "mean"),
        mean_bias_bp=("bias_bp", "mean"),
        mean_corr=("corr", "mean"),
    )
    return out.sort_values(["mean_rmse_bp", "mean_mae_bp"])


def volatility_carry_backtest(
    frame: pd.DataFrame,
    predictions: pd.Series,
    config: VolCarryConfig | None = None,
) -> pd.DataFrame:
    """Backtest a stylized forward-volatility carry rule.

    Position is positive when selling forward volatility and negative when
    buying it.  P&L is the horizon-settled difference between implied and future
    realized volatility, scaled to basis points of normal volatility.  This is
    a research proxy for option carry, not an executable caplet valuation.
    """

    cfg = config or VolCarryConfig()
    data = frame.join(predictions).dropna(
        subset=["implied_vol", "future_realized_vol", "predicted_realized_vol"]
    ).copy()
    data["richness"] = data["implied_vol"] - data["predicted_realized_vol"]

    threshold = cfg.threshold_bp / 10000.0
    raw_position = np.select(
        [data["richness"] > threshold, data["richness"] < -threshold],
        [1.0, -1.0],
        default=0.0,
    )

    forecast_error_vol = (
        data["future_realized_vol"] - data["predicted_realized_vol"]
    ).shift(cfg.label_horizon_rows).rolling(26, min_periods=8).std()
    risk_scale = (cfg.risk_target_bp / 10000.0) / forecast_error_vol
    risk_scale = risk_scale.replace([np.inf, -np.inf], np.nan).clip(
        lower=0.0,
        upper=cfg.max_abs_position,
    )
    data["position"] = pd.Series(raw_position, index=data.index) * risk_scale.fillna(0.0)

    data["gross_pnl_bp"] = (
        data["position"]
        * (data["implied_vol"] - data["future_realized_vol"])
        * 10000.0
    )
    data["turnover"] = data["position"].diff().abs().fillna(data["position"].abs())
    data["cost_bp"] = data["turnover"] * cfg.transaction_cost_bp
    data["net_pnl_bp"] = data["gross_pnl_bp"] - data["cost_bp"]
    data["cum_net_pnl_bp"] = data["net_pnl_bp"].cumsum()
    return data


def strategy_cpcv_table(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    configs: Iterable[VolCarryConfig],
    ridges: Iterable[float] = (1e-5,),
    split_config: PurgedSplitConfig | None = None,
) -> pd.DataFrame:
    """Evaluate pre-declared SOFR carry configurations on CPCV folds."""

    cfg = split_config or PurgedSplitConfig()
    splits = combinatorial_purged_splits(frame.index, cfg)
    rows = []
    for config_id, strategy_cfg in enumerate(configs):
        for ridge in ridges:
            for fold, (train_idx, test_idx) in enumerate(splits):
                preds = _ridge_predict_train_test(
                    frame,
                    feature_cols,
                    train_idx,
                    test_idx,
                    ridge=float(ridge),
                    min_train=strategy_cfg.min_train_weeks,
                ).rename("predicted_realized_vol")
                test_frame = frame.iloc[test_idx].copy()
                bt = volatility_carry_backtest(test_frame, preds, strategy_cfg)
                stats = strategy_performance(bt)
                rows.append(
                    {
                        "config_id": config_id,
                        "ridge": float(ridge),
                        "validation_scheme": "combinatorial_purged_cv",
                        "threshold_bp": strategy_cfg.threshold_bp,
                        "min_train_weeks": strategy_cfg.min_train_weeks,
                        "risk_target_bp": strategy_cfg.risk_target_bp,
                        "transaction_cost_bp": strategy_cfg.transaction_cost_bp,
                        "fold": fold,
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(bt)),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def summarize_strategy_cpcv(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    grouped = table.groupby(
        ["config_id", "ridge", "threshold_bp", "min_train_weeks", "risk_target_bp"],
        dropna=False,
    )
    out = grouped.agg(
        folds=("fold", "nunique"),
        mean_test_n=("test_n", "mean"),
        mean_total_pnl_bp=("total_pnl_bp", "mean"),
        median_total_pnl_bp=("total_pnl_bp", "median"),
        positive_fold_rate=("total_pnl_bp", lambda x: float((x > 0).mean())),
        mean_ann_sharpe=("ann_sharpe", "mean"),
        worst_drawdown_bp=("max_drawdown_bp", "min"),
        mean_turnover=("turnover", "mean"),
    )
    return out.sort_values(["mean_ann_sharpe", "mean_total_pnl_bp"], ascending=False)


def _block_permute_values(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])[: len(values)]


def strategy_block_permutation_null(
    frame: pd.DataFrame,
    predictions: pd.Series,
    config: VolCarryConfig | None = None,
    block_size: int = 8,
    n_sims: int = 500,
    seed: int = 7,
) -> pd.DataFrame:
    """Block-permute realized-vol labels to test signal/label path fitting."""

    cfg = config or VolCarryConfig()
    observed_bt = volatility_carry_backtest(frame, predictions, cfg)
    observed = strategy_performance(observed_bt)
    base = frame.join(predictions.rename("predicted_realized_vol")).dropna(
        subset=["implied_vol", "future_realized_vol", "predicted_realized_vol"]
    )
    rng = np.random.default_rng(seed)
    values = base["future_realized_vol"].astype(float).values
    rows = []
    for sim in range(n_sims):
        sim_frame = base.drop(columns=["predicted_realized_vol"]).copy()
        sim_frame["future_realized_vol"] = _block_permute_values(values, block_size, rng)
        sim_preds = base["predicted_realized_vol"]
        sim_bt = volatility_carry_backtest(sim_frame, sim_preds, cfg)
        stats = strategy_performance(sim_bt)
        stats["sim"] = sim
        rows.append(stats)
    null = pd.DataFrame(rows)
    null["observed_total_pnl_bp"] = observed["total_pnl_bp"]
    null["observed_ann_sharpe"] = observed["ann_sharpe"]
    null["pvalue_total_pnl"] = (null["total_pnl_bp"] >= observed["total_pnl_bp"]).mean()
    null["pvalue_ann_sharpe"] = (null["ann_sharpe"] >= observed["ann_sharpe"]).mean()
    null["mean_sim_total_pnl_bp"] = null["total_pnl_bp"].mean()
    null["median_sim_total_pnl_bp"] = null["total_pnl_bp"].median()
    null["mean_sim_ann_sharpe"] = null["ann_sharpe"].mean()
    null["median_sim_ann_sharpe"] = null["ann_sharpe"].median()
    return null


def block_bootstrap_performance_ci(
    pnl: pd.Series,
    block_size: int = 8,
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Block-bootstrap total P&L and Sharpe intervals for dependent P&L."""

    clean = pnl.dropna().astype(float).values
    if len(clean) == 0:
        return {
            "total_pnl_lo": np.nan,
            "total_pnl_median": np.nan,
            "total_pnl_hi": np.nan,
            "ann_sharpe_lo": np.nan,
            "ann_sharpe_median": np.nan,
            "ann_sharpe_hi": np.nan,
        }
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(clean) - block_size + 1))
    totals = []
    sharpes = []
    for _ in range(n_boot):
        draws = []
        while len(draws) < len(clean):
            start = int(rng.choice(starts))
            draws.extend(clean[start : start + block_size])
        sample = np.asarray(draws[: len(clean)], dtype=float)
        totals.append(float(sample.sum()))
        sharpes.append(float(sample.mean() / sample.std(ddof=1) * np.sqrt(52.0)) if sample.std(ddof=1) > 0 else np.nan)
    lo, median, hi = np.quantile(totals, [alpha / 2.0, 0.50, 1.0 - alpha / 2.0])
    slo, smedian, shi = np.nanquantile(sharpes, [alpha / 2.0, 0.50, 1.0 - alpha / 2.0])
    return {
        "total_pnl_lo": float(lo),
        "total_pnl_median": float(median),
        "total_pnl_hi": float(hi),
        "ann_sharpe_lo": float(slo),
        "ann_sharpe_median": float(smedian),
        "ann_sharpe_hi": float(shi),
    }


def block_bootstrap_path_summary(
    pnl: pd.Series,
    block_size: int = 8,
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Return observed, mean, median, and tail bootstrap cumulative P&L paths.

    The path summary is deliberately nonparametric.  It resamples contiguous
    P&L blocks rather than assuming a Gaussian return distribution, which is
    more appropriate for overlapping forward-volatility labels and crisis-like
    clustered losses.
    """

    clean = pnl.dropna().astype(float).values
    if len(clean) == 0:
        return pd.DataFrame(columns=["observed_path", "mean_path", "median_path", "lower_path", "upper_path"])

    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(clean) - block_size + 1))
    paths = np.empty((n_boot, len(clean)), dtype=float)
    for sim in range(n_boot):
        draws = []
        while len(draws) < len(clean):
            start = int(rng.choice(starts))
            draws.extend(clean[start : start + block_size])
        paths[sim] = np.cumsum(np.asarray(draws[: len(clean)], dtype=float))

    observed = np.cumsum(clean)
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    return pd.DataFrame(
        {
            "observed_path": observed,
            "mean_path": paths.mean(axis=0),
            "median_path": np.median(paths, axis=0),
            "lower_path": np.quantile(paths, lo_q, axis=0),
            "upper_path": np.quantile(paths, hi_q, axis=0),
        },
        index=np.arange(1, len(clean) + 1),
    )


def strategy_performance(
    backtest: pd.DataFrame,
    pnl_col: str = "net_pnl_bp",
    periods_per_year: int = 52,
) -> dict[str, float]:
    """Return high-level performance diagnostics for strategy P&L."""

    pnl = backtest[pnl_col].dropna()
    if pnl.empty:
        return {
            "total_pnl_bp": np.nan,
            "ann_sharpe": np.nan,
            "max_drawdown_bp": np.nan,
            "hit_rate": np.nan,
            "active_periods": 0,
            "turnover": np.nan,
        }

    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    active = backtest["position"].abs().reindex(pnl.index).fillna(0.0) > 1e-12
    active_pnl = pnl[active]
    std = pnl.std()
    return {
        "total_pnl_bp": float(pnl.sum()),
        "ann_sharpe": float(pnl.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "max_drawdown_bp": float(drawdown.min()),
        "hit_rate": float((active_pnl > 0).mean()) if len(active_pnl) else np.nan,
        "active_periods": int(active.sum()),
        "turnover": float(backtest["turnover"].sum()) if "turnover" in backtest else np.nan,
    }


def regime_performance(backtest: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy performance by the policy regimes used in the notebook."""

    if "policy_regime" not in backtest:
        raise KeyError("backtest must contain a policy_regime column.")
    rows = []
    for regime_name, group in backtest.groupby("policy_regime"):
        stats = strategy_performance(group)
        stats["regime"] = regime_name
        rows.append(stats)
    return pd.DataFrame(rows).set_index("regime").sort_index()


def block_bootstrap_mean_ci(
    pnl: pd.Series,
    block_size: int = 8,
    n_boot: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Block-bootstrap confidence interval for mean P&L.

    The bootstrap respects the overlapping-horizon structure better than an
    iid resample.  It is deliberately simple and deterministic for notebooks.
    """

    clean = pnl.dropna().astype(float).values
    if len(clean) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(clean) - block_size + 1))
    means = []
    for _ in range(n_boot):
        draws = []
        while len(draws) < len(clean):
            start = int(rng.choice(starts))
            draws.extend(clean[start : start + block_size])
        means.append(np.mean(draws[: len(clean)]))
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
