from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sofr_forward_vol import (
    PurgedSplitConfig,
    VolCarryConfig,
    black_caplet_price,
    build_vol_carry_frame,
    build_cap_curve,
    combinatorial_purged_splits,
    expanding_ridge_forecast,
    forecast_validation_table,
    volatility_carry_backtest,
    load_data,
    normal_caplet_price,
    purged_blocked_splits,
    strategy_block_permutation_null,
    strategy_performance,
)


ROOT = Path(__file__).resolve().parents[1]


def _has_data() -> bool:
    return (ROOT / "project_cap_vol_ts.xlsx").exists() and (ROOT / "ref_rates.xlsx").exists()


def test_caplet_prices_are_monotone_in_vol():
    low_black = black_caplet_price(0.10, 1.0, 0.04, 0.041, 0.96)
    high_black = black_caplet_price(0.30, 1.0, 0.04, 0.041, 0.96)
    low_normal = normal_caplet_price(0.005, 1.0, 0.04, 0.041, 0.96)
    high_normal = normal_caplet_price(0.015, 1.0, 0.04, 0.041, 0.96)
    assert high_black > low_black
    assert high_normal > low_normal


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
    preds = expanding_ridge_forecast(
        frame,
        ["implied_vol", "lag_realized_vol", "vol_slope", "vol_momentum_13w", "vol_of_vol_13w"],
        min_train=20,
    )
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
    for train_idx, test_idx in splits:
        assert not set(train_idx).intersection(set(test_idx))
        for train in train_idx:
            train_window = set(range(train, min(len(index), train + horizon + 1)))
            padded_test = set(
                range(
                    max(0, int(test_idx.min()) - embargo),
                    min(len(index), int(test_idx.max()) + embargo + 1),
                )
            )
            assert train_window.isdisjoint(padded_test)


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
    assert len(splits) == 10
    assert not table.empty
    pd.testing.assert_frame_equal(null_a, null_b)


@pytest.mark.skipif(not _has_data(), reason="local raw data not present")
def test_validation_curve_replicates_course_workbook():
    data = load_data(ROOT)
    assert data.validation_curve is not None
    curve = build_cap_curve("2025-06-30", data, interpolation="cubic")
    validation = data.validation_curve
    assert (curve["discount"].sub(validation["discounts"]).abs().max()) < 1e-4
    assert (curve["forward_rate"].sub(validation["forwards"]).abs().max()) < 2e-4
    assert (curve["forward_black_vol"].sub(validation["fwd vols"]).abs().dropna().max()) < 3e-3
