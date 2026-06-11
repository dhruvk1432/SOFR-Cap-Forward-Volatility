from pathlib import Path

import pytest

from src.sofr_forward_vol import (
    black_caplet_price,
    build_cap_curve,
    load_data,
    normal_caplet_price,
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


@pytest.mark.skipif(not _has_data(), reason="local raw data not present")
def test_validation_curve_replicates_course_workbook():
    data = load_data(ROOT)
    assert data.validation_curve is not None
    curve = build_cap_curve("2025-06-30", data, interpolation="cubic")
    validation = data.validation_curve
    assert (curve["discount"].sub(validation["discounts"]).abs().max()) < 1e-4
    assert (curve["forward_rate"].sub(validation["forwards"]).abs().max()) < 2e-4
    assert (curve["forward_black_vol"].sub(validation["fwd vols"]).abs().dropna().max()) < 3e-3
