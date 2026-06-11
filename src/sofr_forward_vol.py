"""Utilities for stripping forward volatility from SOFR cap quotes.

The functions keep the notebook focused on analysis while making the curve
construction testable.  The data files are intentionally loaded from local
workbooks rather than checked into git.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import brentq
from scipy.stats import norm
import statsmodels.api as sm


QUARTER = 0.25
GRID = np.round(np.arange(QUARTER, 10.0 + QUARTER, QUARTER), 2)


@dataclass(frozen=True)
class SofrCapData:
    cap_normal_vol_bp: pd.DataFrame
    sofr_swaps: pd.DataFrame
    sofr_daily: pd.Series
    validation_curve: pd.DataFrame | None = None


def _find_file(root: Path, filename: str) -> Path:
    candidates = [
        root / filename,
        root / "data" / filename,
        root / "data" / "raw" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in the repo root or data/raw/."
    )


def _normalize_maturity_index(values: Iterable[float], freq: int) -> np.ndarray:
    return np.round(freq * np.asarray(values, dtype=float), 0) / freq


def load_data(root: str | Path = ".") -> SofrCapData:
    root = Path(root)
    cap_book = _find_file(root, "project_cap_vol_ts.xlsx")
    ref_book = _find_file(root, "ref_rates.xlsx")

    cap_raw = pd.read_excel(cap_book, sheet_name="cap", index_col=0)
    cap_maturities = cap_raw.loc["maturity"].astype(float)
    cap = cap_raw.drop(index="maturity").astype(float)
    cap.index = pd.to_datetime(cap.index)
    cap.columns = _normalize_maturity_index(cap_maturities.values, 4)
    cap.columns.name = "maturity"
    cap = cap.T.drop_duplicates().T.sort_index()

    sofr_raw = pd.read_excel(cap_book, sheet_name="sofr", index_col=0)
    sofr_maturities = sofr_raw.loc["maturity"].astype(float)
    sofr = sofr_raw.drop(index="maturity").astype(float)
    sofr.index = pd.to_datetime(sofr.index)
    sofr.columns = _normalize_maturity_index(sofr_maturities.values, 12)
    sofr.columns.name = "maturity"
    sofr = sofr.T.drop_duplicates().T.sort_index() / 100.0

    ref_rates = pd.read_excel(ref_book, sheet_name="data")
    ref_rates["date"] = pd.to_datetime(ref_rates["date"])
    ref_rates = ref_rates.set_index("date").sort_index()
    sofr_daily = (ref_rates["SOFR"].dropna() / 100.0).rename("SOFR")

    validation = None
    try:
        validation_book = _find_file(root, "cap_curves_2025-06-30.xlsx")
        validation = pd.read_excel(
            validation_book,
            sheet_name="rate curves 2025-06-30",
            index_col=0,
        )
        validation.index = validation.index.astype(float)
        validation.index.name = "tenor"
    except FileNotFoundError:
        validation = None

    return SofrCapData(cap, sofr, sofr_daily, validation)


def black_caplet_price(vol: float, expiry: float, strike: float, forward: float, discount: float) -> float:
    """Black caplet price per unit notional and unit accrual.

    The quarterly accrual factor is omitted because it multiplies every caplet
    in the bootstrap and therefore cancels in the implied-vol inversion.
    """

    expiry = max(float(expiry), 1e-12)
    if vol <= 1e-12:
        return discount * max(forward - strike, 0.0)
    if forward <= 0 or strike <= 0:
        raise ValueError("Black pricing requires positive forward and strike rates.")
    sigma_root_t = vol * np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * sigma_root_t**2) / sigma_root_t
    d2 = d1 - sigma_root_t
    return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def normal_caplet_price(vol: float, expiry: float, strike: float, forward: float, discount: float) -> float:
    """Bachelier caplet price per unit notional and unit accrual."""

    expiry = max(float(expiry), 1e-12)
    if vol <= 1e-12:
        return discount * max(forward - strike, 0.0)
    sigma_root_t = vol * np.sqrt(expiry)
    d = (forward - strike) / sigma_root_t
    return discount * ((forward - strike) * norm.cdf(d) + sigma_root_t * norm.pdf(d))


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


def predictive_regressions(
    normal_panel: pd.DataFrame,
    delta: float,
    tau_list: Iterable[float],
) -> pd.DataFrame:
    rows = []
    spot = normal_panel[delta]
    for tau in tau_list:
        horizon_weeks = max(1, int(round((tau - delta) * 52)))
        frame = pd.DataFrame(
            {
                "forward_vol": normal_panel[tau],
                "future_spot": spot.shift(-horizon_weeks),
            }
        ).dropna()
        x = sm.add_constant(frame["forward_vol"].astype(float))
        y = frame["future_spot"].astype(float)
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": horizon_weeks})
        beta = model.params["forward_vol"]
        beta_se = model.bse["forward_vol"]
        rows.append(
            {
                "tau": tau,
                "horizon_years": tau - delta,
                "horizon_weeks": horizon_weeks,
                "alpha_bp": model.params["const"] * 10000.0,
                "beta": beta,
                "beta_hac_se": beta_se,
                "t_beta_eq_1": (beta - 1.0) / beta_se if beta_se > 0 else np.nan,
                "r2": model.rsquared,
                "n_weekly": len(frame),
                "effective_nonoverlap_n": max(1, int(len(frame) / horizon_weeks)),
            }
        )
    return pd.DataFrame(rows).set_index("tau")


def term_premium_panel(normal_panel: pd.DataFrame, delta: float, tau_list: Iterable[float]) -> pd.DataFrame:
    out = pd.DataFrame(index=normal_panel.index)
    for tau in tau_list:
        horizon_weeks = max(1, int(round((tau - delta) * 52)))
        out[tau] = normal_panel[tau] - normal_panel[delta].shift(-horizon_weeks)
    out.index.name = "date"
    out.columns.name = "tau"
    return out


def premium_summary(premium: pd.DataFrame, delta: float) -> pd.DataFrame:
    rows = []
    total_weeks = (premium.index.max() - premium.index.min()).days / 7.0
    for tau in premium.columns:
        s = premium[tau].dropna()
        horizon_weeks = max(1, int(round((tau - delta) * 52)))
        effective_n = int(total_weeks / horizon_weeks)
        weekly_ir = s.mean() / s.std() if s.std() > 0 else np.nan
        rows.append(
            {
                "tau": tau,
                "mean_bp": s.mean() * 10000.0,
                "std_bp": s.std() * 10000.0,
                "frac_positive": (s > 0).mean(),
                "weekly_ir": weekly_ir,
                "effective_nonoverlap_n": max(1, effective_n),
                "horizon_weeks": horizon_weeks,
            }
        )
    return pd.DataFrame(rows).set_index("tau")


def forward_realized_sofr_vol(sofr_daily: pd.Series, window_days: int) -> pd.Series:
    changes = sofr_daily.diff()
    realized = changes[::-1].rolling(
        window=window_days,
        min_periods=max(5, int(window_days * 0.7)),
    ).std()[::-1]
    return (realized * np.sqrt(252.0)).rename(f"realized_{window_days}d")


def policy_regime(date: pd.Timestamp) -> str:
    date = pd.Timestamp(date)
    if pd.Timestamp("2022-03-01") <= date <= pd.Timestamp("2023-07-31"):
        return "Hiking"
    if pd.Timestamp("2023-08-01") <= date <= pd.Timestamp("2024-08-31"):
        return "Pause"
    if pd.Timestamp("2024-09-01") <= date <= pd.Timestamp("2025-12-31"):
        return "Easing"
    return "Other"
