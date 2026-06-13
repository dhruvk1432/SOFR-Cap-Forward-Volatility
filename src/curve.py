from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import brentq

from .constants import GRID, QUARTER
from .data import SofrCapData
from .pricing import black_caplet_price


def bootstrap_discount_curve(swap_quotes: pd.Series, grid: np.ndarray = GRID) -> pd.DataFrame:
    """Interpolate SOFR swap quotes and bootstrap quarterly discount factors."""

    swap_obs = swap_quotes.dropna().sort_index()
    swap_ann = pd.Series(
        np.interp(grid, swap_obs.index.values.astype(float), swap_obs.values.astype(float)),
        index=grid,
        name="swap_rate_annual",
    )
    swap_rates = pd.Series(
        4.0 * ((1.0 + swap_ann) ** QUARTER - 1.0),
        index=grid,
        name="swap_rate_quarterly",
    )

    discounts = pd.Series(index=grid, dtype=float, name="discount")
    for i, tenor in enumerate(grid):
        if i == 0:
            discounts.loc[tenor] = 1.0 / (1.0 + QUARTER * swap_rates.loc[tenor])
        else:
            previous = discounts.iloc[:i].sum()
            discounts.loc[tenor] = (
                1.0 - QUARTER * swap_rates.loc[tenor] * previous
            ) / (1.0 + QUARTER * swap_rates.loc[tenor])

    forwards = pd.Series(index=grid, dtype=float, name="forward_rate")
    forwards.iloc[0] = np.nan
    for i in range(1, len(grid)):
        forwards.iloc[i] = (discounts.iloc[i - 1] / discounts.iloc[i] - 1.0) / QUARTER

    spot_rates = pd.Series(-np.log(discounts.values) / grid, index=grid, name="spot_rate")
    return pd.concat([swap_rates, spot_rates, discounts, forwards], axis=1)


def _interpolate_flat_black_vols(
    quoted_maturities: pd.Index,
    quoted_black_vols: pd.Series,
    grid: np.ndarray,
    method: str,
) -> pd.Series:
    x = quoted_maturities.astype(float).values
    y = quoted_black_vols.astype(float).values
    if method == "cubic":
        interpolator = interp1d(x, y, kind="cubic", fill_value="extrapolate")
        values = interpolator(grid)
    elif method == "pchip":
        interpolator = PchipInterpolator(x, y, extrapolate=True)
        values = interpolator(grid)
    else:
        raise ValueError("method must be 'cubic' or 'pchip'")

    # Short end extrapolation is linear because the first quoted cap is 1Y.
    slope_short = (y[1] - y[0]) / (x[1] - x[0])
    values = pd.Series(values, index=grid, name="flat_black_vol")
    short = values.index < x[0]
    values.loc[short] = y[0] + slope_short * (values.index[short] - x[0])
    values.loc[QUARTER] = np.nan
    return values.clip(lower=1e-6)


def build_cap_curve(
    date: str | pd.Timestamp,
    data: SofrCapData,
    max_tenor: float = 10.0,
    interpolation: str = "cubic",
) -> pd.DataFrame:
    """Build discount, forward-rate, flat-vol, and stripped forward-vol curves.

    The quoted cap vols are normal vols in bp.  To replicate the course
    validation workbook, they are converted into Black-equivalent flat vols
    with the ATM approximation normal_vol / forward_rate before bootstrapping.
    The notebook also converts the stripped Black vols back into normal vols
    for time-series regressions.
    """

    date = pd.Timestamp(date)
    if date not in data.cap_normal_vol_bp.index:
        raise KeyError(f"{date.date()} not found in cap vol data.")
    if date not in data.sofr_swaps.index:
        raise KeyError(f"{date.date()} not found in SOFR swap data.")

    rates = bootstrap_discount_curve(data.sofr_swaps.loc[date])
    forwards = rates["forward_rate"]

    normal_quotes = (data.cap_normal_vol_bp.loc[date].dropna().sort_index() / 10000.0)
    quoted_forwards = forwards.reindex(normal_quotes.index.astype(float))
    if quoted_forwards.isna().any():
        raise ValueError("Missing forwards at one or more quoted cap maturities.")
    flat_black_quotes = normal_quotes / quoted_forwards.clip(lower=1e-5)
    flat_black = _interpolate_flat_black_vols(
        normal_quotes.index,
        flat_black_quotes,
        rates.index.values,
        interpolation,
    )

    forward_black = pd.Series(index=rates.index, dtype=float, name="forward_black_vol")
    forward_black.loc[QUARTER] = np.nan
    forward_black.loc[0.50] = flat_black.loc[0.50]

    boot_grid = rates.index[rates.index <= max_tenor]
    for tenor in boot_grid[2:]:
        strike = rates.loc[tenor, "swap_rate_quarterly"]
        sigma_flat = flat_black.loc[tenor]

        total_cap_price = 0.0
        for pay_tenor in np.round(np.arange(0.50, tenor + 0.001, QUARTER), 2):
            total_cap_price += black_caplet_price(
                sigma_flat,
                pay_tenor - QUARTER,
                strike,
                rates.loc[pay_tenor, "forward_rate"],
                rates.loc[pay_tenor, "discount"],
            )

        previous_caplets = 0.0
        for pay_tenor in np.round(np.arange(0.50, tenor, QUARTER), 2):
            previous_caplets += black_caplet_price(
                forward_black.loc[pay_tenor],
                pay_tenor - QUARTER,
                strike,
                rates.loc[pay_tenor, "forward_rate"],
                rates.loc[pay_tenor, "discount"],
            )

        target = total_cap_price - previous_caplets
        forward = rates.loc[tenor, "forward_rate"]
        discount = rates.loc[tenor, "discount"]
        low = black_caplet_price(1e-12, tenor - QUARTER, strike, forward, discount)
        high = black_caplet_price(5.0, tenor - QUARTER, strike, forward, discount)
        if target < low - 1e-10 or target > high + 1e-10:
            raise ValueError(
                f"Caplet inversion failed on {date.date()} at {tenor}Y: "
                f"target={target:.8g}, bracket=({low:.8g}, {high:.8g})."
            )
        forward_black.loc[tenor] = brentq(
            lambda vol: black_caplet_price(vol, tenor - QUARTER, strike, forward, discount) - target,
            1e-12,
            5.0,
            maxiter=100,
        )

    flat_normal = (flat_black * forwards).rename("flat_normal_vol")
    forward_normal = (forward_black * forwards).rename("forward_normal_vol")
    out = pd.concat([rates, flat_black, forward_black, flat_normal, forward_normal], axis=1)
    out.index.name = "tenor"
    return out


def weekly_dates(data: SofrCapData) -> pd.DatetimeIndex:
    common = data.cap_normal_vol_bp.index.intersection(data.sofr_swaps.index).sort_values()
    return pd.DatetimeIndex(pd.Series(common, index=common).resample("W-FRI").last().dropna().values)


def build_forward_vol_panel(
    data: SofrCapData,
    tenors: Iterable[float],
    max_tenor: float,
    interpolation: str = "cubic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tenors = [float(t) for t in tenors]
    black = pd.DataFrame(index=weekly_dates(data), columns=tenors, dtype=float)
    normal = pd.DataFrame(index=black.index, columns=tenors, dtype=float)

    for date in black.index:
        try:
            curve = build_cap_curve(date, data, max_tenor=max_tenor, interpolation=interpolation)
            black.loc[date] = curve.loc[tenors, "forward_black_vol"].values
            normal.loc[date] = curve.loc[tenors, "forward_normal_vol"].values
        except Exception:
            continue

    black.index.name = normal.index.name = "date"
    black.columns.name = normal.columns.name = "tenor"
    return black, normal
