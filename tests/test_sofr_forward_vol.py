from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sofr_forward_vol import (
    PurgedSplitConfig,
    SofrCapData,
    VolCarryConfig,
    black_caplet_price,
    block_bootstrap_path_summary,
    bootstrap_discount_curve,
    build_cap_curve,
    build_vol_carry_frame,
    combinatorial_purged_splits,
    expanding_ridge_forecast,
    forecast_validation_table,
    load_data,
    normal_caplet_price,
    purged_blocked_splits,
    strategy_block_permutation_null,
    strategy_cpcv_table,
    strategy_performance,
    volatility_carry_backtest,
)


ROOT = Path(__file__).resolve().parents[1]


def _has_data() -> bool:
    return (ROOT / "project_cap_vol_ts.xlsx").exists() and (ROOT / "ref_rates.xlsx").exists()


def _assert_no_label_overlap(splits, n_obs: int, horizon: int, embargo: int) -> None:
    for train_idx, test_idx in splits:
        assert not set(train_idx).intersection(set(test_idx))
        split_points = np.where(np.diff(test_idx) > 1)[0] + 1
        test_blocks = np.split(test_idx, split_points)
        for train in train_idx:
            train_window = set(range(train, min(n_obs, train + horizon + 1)))
            for block in test_blocks:
                padded_test = set(
                    range(
                        max(0, int(block.min()) - embargo),
                        min(n_obs, int(block.max()) + embargo + 1),
                    )
                )
                assert train_window.isdisjoint(padded_test)


def test_caplet_prices_are_monotone_and_converge_to_intrinsic():
    low_black = black_caplet_price(0.10, 1.0, 0.04, 0.041, 0.96)
    high_black = black_caplet_price(0.30, 1.0, 0.04, 0.041, 0.96)
    low_normal = normal_caplet_price(0.005, 1.0, 0.04, 0.041, 0.96)
    high_normal = normal_caplet_price(0.015, 1.0, 0.04, 0.041, 0.96)
    intrinsic = 0.96 * max(0.041 - 0.04, 0.0)
    assert high_black > low_black
    assert high_normal > low_normal
    assert black_caplet_price(0.0, 1.0, 0.04, 0.041, 0.96) == pytest.approx(intrinsic)
    assert normal_caplet_price(0.0, 1.0, 0.04, 0.041, 0.96) == pytest.approx(intrinsic)


def test_bootstrap_discount_curve_is_positive_and_decreasing():
    quotes = pd.Series(
        [0.045, 0.046, 0.047, 0.048, 0.049],
        index=[0.25, 1.0, 2.0, 5.0, 10.0],
    )
    curve = bootstrap_discount_curve(quotes)
    assert (curve["discount"] > 0).all()
    assert curve["discount"].is_monotonic_decreasing
    assert curve["forward_rate"].iloc[1:].notna().all()


def _synthetic_data() -> SofrCapData:
    date = pd.Timestamp("2025-06-30")
    maturities = [1.0, 1.5, 2.0, 3.0, 5.0]
    cap = pd.DataFrame([[75.0, 85.0, 95.0, 105.0, 115.0]], index=[date], columns=maturities)
    sofr = pd.DataFrame([[0.040, 0.041, 0.042, 0.043, 0.044]], index=[date], columns=[0.25, 1.0, 2.0, 5.0, 10.0])
    daily = pd.Series(0.04, index=pd.date_range("2024-01-01", "2025-12-31", freq="B"), name="SOFR")
    return SofrCapData(cap, sofr, daily)


def test_cap_repricing_identity_holds_after_stripping():
    data = _synthetic_data()
    curve = build_cap_curve("2025-06-30", data, max_tenor=5.0, interpolation="pchip")
    for tenor in [1.0, 1.5, 2.0, 3.0, 5.0]:
        strike = curve.loc[tenor, "swap_rate_quarterly"]
        flat = curve.loc[tenor, "flat_black_vol"]
        flat_price = 0.0
        stripped_price = 0.0
        for pay_tenor in np.round(np.arange(0.50, tenor + 0.001, 0.25), 2):
            flat_price += black_caplet_price(
                flat,
                pay_tenor - 0.25,
                strike,
                curve.loc[pay_tenor, "forward_rate"],
                curve.loc[pay_tenor, "discount"],
            )
            stripped_price += black_caplet_price(
                curve.loc[pay_tenor, "forward_black_vol"],
                pay_tenor - 0.25,
                strike,
                curve.loc[pay_tenor, "forward_rate"],
                curve.loc[pay_tenor, "discount"],
            )
        assert stripped_price == pytest.approx(flat_price, rel=1e-8, abs=1e-10)


def test_vol_carry_backtest_uses_only_lagged_forecasts():
    weekly = pd.date_range("2023-01-06", periods=90, freq="W-FRI")
    daily = pd.date_range("2022-01-03", periods=620, freq="B")
    rng = np.random.default_rng(42)
    sofr = pd.Series(0.04 + np.cumsum(rng.normal(0, 0.00008, len(daily))), index=daily)
    panel = pd.DataFrame(
        {
            0.50: 0.010 + 0.001 * np.sin(np.linspace(0, 5, len(weekly))),
            1.00: 0.011 + 0.001 * np.sin(np.linspace(0, 5, len(weekly)) + 0.2),
            2.00: 0.012 + 0.001 * np.sin(np.linspace(0, 5, len(weekly)) + 0.4),
        },
        index=weekly,
    )

    frame = build_vol_carry_frame(panel, sofr, tenor=0.50, realized_window_days=21)
    features = ["implied_vol", "lag_realized_vol", "vol_slope", "vol_momentum_13w", "vol_of_vol_13w"]
    preds = expanding_ridge_forecast(frame, features, min_train=20)
    target_date = preds.dropna().index[5]
    truncated = frame.loc[:target_date]
    truncated_preds = expanding_ridge_forecast(truncated, features, min_train=20)
    assert preds.loc[target_date] == pytest.approx(truncated_preds.loc[target_date])

    bt = volatility_carry_backtest(
        frame,
        preds,
        VolCarryConfig(realized_window_days=21, min_train_weeks=20, threshold_bp=0.1),
    )
    stats = strategy_performance(bt)
    assert {"position", "gross_pnl_bp", "cost_bp", "net_pnl_bp", "cum_net_pnl_bp"}.issubset(bt.columns)
    assert len(bt) > 20
    assert stats["active_periods"] >= 0


def test_purged_splits_remove_overlapping_label_windows():
    index = pd.date_range("2024-01-05", periods=36, freq="W-FRI")
    horizon = 5
    embargo = 2
    splits = purged_blocked_splits(index, n_splits=4, label_horizon=horizon, embargo=embargo)
    assert len(splits) == 4
    _assert_no_label_overlap(splits, len(index), horizon, embargo)


def test_cpcv_splits_purge_each_non_adjacent_test_block():
    index = pd.date_range("2024-01-05", periods=72, freq="W-FRI")
    cfg = PurgedSplitConfig(n_groups=6, n_test_groups=2, label_horizon=5, embargo=2)
    splits = combinatorial_purged_splits(index, cfg)
    assert len(splits) == 15
    assert any(np.any(np.diff(test_idx) > 1) for _, test_idx in splits)
    _assert_no_label_overlap(splits, len(index), cfg.label_horizon, cfg.embargo)


def test_cpcv_and_monte_carlo_are_deterministic():
    weekly = pd.date_range("2023-01-06", periods=120, freq="W-FRI")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "implied_vol": 0.012 + rng.normal(0, 0.0002, len(weekly)),
            "lag_realized_vol": 0.009 + rng.normal(0, 0.0002, len(weekly)),
            "vol_slope": 0.002 + rng.normal(0, 0.0001, len(weekly)),
            "vol_momentum_13w": rng.normal(0, 0.0001, len(weekly)),
            "vol_of_vol_13w": 0.00015 + rng.normal(0, 0.00002, len(weekly)),
            "future_realized_vol": 0.010 + rng.normal(0, 0.00025, len(weekly)),
        },
        index=weekly,
    )
    feature_sets = {
        "implied_only": ["implied_vol"],
        "full": ["implied_vol", "lag_realized_vol", "vol_slope", "vol_momentum_13w", "vol_of_vol_13w"],
    }
    cfg = PurgedSplitConfig(n_groups=5, n_test_groups=2, label_horizon=4, embargo=1)
    splits = combinatorial_purged_splits(frame.index, cfg)
    table = forecast_validation_table(frame, feature_sets, ridges=(1e-5,), split_config=cfg, min_train=20)
    strategy_table = strategy_cpcv_table(
        frame,
        feature_sets["full"],
        [VolCarryConfig(min_train_weeks=20, threshold_bp=0.1, realized_window_days=21)],
        ridges=(1e-5,),
        split_config=cfg,
    )
    preds = expanding_ridge_forecast(frame, feature_sets["full"], min_train=20)
    null_a = strategy_block_permutation_null(
        frame,
        preds,
        VolCarryConfig(min_train_weeks=20, threshold_bp=0.1, realized_window_days=21),
        n_sims=25,
        seed=99,
    )
    null_b = strategy_block_permutation_null(
        frame,
        preds,
        VolCarryConfig(min_train_weeks=20, threshold_bp=0.1, realized_window_days=21),
        n_sims=25,
        seed=99,
    )
    path_a = block_bootstrap_path_summary(null_a["total_pnl_bp"], block_size=5, n_boot=25, seed=99)
    path_b = block_bootstrap_path_summary(null_a["total_pnl_bp"], block_size=5, n_boot=25, seed=99)
    assert len(splits) == 10
    assert not table.empty
    assert table["validation_scheme"].eq("combinatorial_purged_cv").all()
    assert table["fold"].nunique() == len(splits)
    assert strategy_table["validation_scheme"].eq("combinatorial_purged_cv").all()
    assert strategy_table["fold"].nunique() == len(splits)
    pd.testing.assert_frame_equal(null_a, null_b)
    pd.testing.assert_frame_equal(path_a, path_b)
    assert {"observed_path", "mean_path", "median_path", "lower_path", "upper_path"}.issubset(path_a.columns)


@pytest.mark.skipif(not _has_data(), reason="local raw data not present")
def test_validation_curve_replicates_course_workbook():
    data = load_data(ROOT)
    assert data.validation_curve is not None
    curve = build_cap_curve("2025-06-30", data, interpolation="cubic")
    validation = data.validation_curve
    assert (curve["discount"].sub(validation["discounts"]).abs().max()) < 1e-4
    assert (curve["forward_rate"].sub(validation["forwards"]).abs().max()) < 2e-4
    assert (curve["forward_black_vol"].sub(validation["fwd vols"]).abs().dropna().max()) < 3e-3
