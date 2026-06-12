from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sofr_forward_vol import (
    VolCarryConfig,
    black_caplet_price,
    build_vol_carry_frame,
    build_cap_curve,
    expanding_ridge_forecast,
    volatility_carry_backtest,
    load_data,
    normal_caplet_price,
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


@pytest.mark.skipif(not _has_data(), reason="local raw data not present")
def test_validation_curve_replicates_course_workbook():
    data = load_data(ROOT)
    assert data.validation_curve is not None
    curve = build_cap_curve("2025-06-30", data, interpolation="cubic")
    validation = data.validation_curve
    assert (curve["discount"].sub(validation["discounts"]).abs().max()) < 1e-4
    assert (curve["forward_rate"].sub(validation["forwards"]).abs().max()) < 2e-4
    assert (curve["forward_black_vol"].sub(validation["fwd vols"]).abs().dropna().max()) < 3e-3
