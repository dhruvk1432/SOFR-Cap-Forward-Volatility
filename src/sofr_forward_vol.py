"""Utilities for stripping forward volatility from SOFR cap quotes.

The functions keep the notebook focused on analysis while making the curve
construction testable.  The data files are intentionally loaded from local
workbooks rather than checked into git.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PurgedSplitConfig:
    """Configuration for leakage-aware time-series validation.

    ``label_horizon`` and ``embargo`` are measured in rows of the validation
    frame.  For weekly SOFR labels based on a 63 business-day forward realized
    window, a 13-row horizon is the natural default.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    label_horizon: int = 13
    embargo: int = 2


def _contiguous_blocks(n_obs: int, n_groups: int) -> list[np.ndarray]:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    if n_groups > n_obs:
        raise ValueError("n_groups cannot exceed the number of observations.")
    return [block.astype(int) for block in np.array_split(np.arange(n_obs), n_groups) if len(block)]


def _purged_train_indices(
    n_obs: int,
    test_indices: np.ndarray,
    label_horizon: int,
    embargo: int,
) -> np.ndarray:
    """Return train rows whose label windows do not overlap the test region."""

    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative.")
    test_indices = np.asarray(test_indices, dtype=int)
    if len(test_indices) == 0:
        raise ValueError("test_indices cannot be empty.")

    starts = np.arange(n_obs)
    ends = starts + int(label_horizon)
    keep = np.ones(n_obs, dtype=bool)
    keep[test_indices] = False

    # Purge against every contiguous test block.  This supports CPCV splits
    # where test groups are not adjacent.
    split_points = np.where(np.diff(test_indices) > 1)[0] + 1
    for block in np.split(test_indices, split_points):
        block_start = max(0, int(block.min()) - int(embargo))
        block_end = min(n_obs - 1, int(block.max()) + int(embargo))
        overlaps = (starts <= block_end) & (ends >= block_start)
        keep &= ~overlaps

    return np.flatnonzero(keep)


def purged_blocked_splits(
    index: Iterable[object],
    n_splits: int = 5,
    label_horizon: int = 13,
    embargo: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create chronological blocked folds with purge and embargo controls."""

    n_obs = len(pd.Index(index))
    blocks = _contiguous_blocks(n_obs, n_splits)
    splits = []
    for test in blocks:
        train = _purged_train_indices(n_obs, test, label_horizon, embargo)
        if len(train) and len(test):
            splits.append((train, test))
    return splits


def combinatorial_purged_splits(
    index: Iterable[object],
    config: PurgedSplitConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create CPCV-style train/test splits from combinations of time blocks."""

    cfg = config or PurgedSplitConfig()
    n_obs = len(pd.Index(index))
    blocks = _contiguous_blocks(n_obs, cfg.n_groups)
    if cfg.n_test_groups < 1 or cfg.n_test_groups >= len(blocks):
        raise ValueError("n_test_groups must be between 1 and n_groups - 1.")

    splits = []
    for group_ids in combinations(range(len(blocks)), cfg.n_test_groups):
        test = np.sort(np.concatenate([blocks[i] for i in group_ids])).astype(int)
        train = _purged_train_indices(n_obs, test, cfg.label_horizon, cfg.embargo)
        if len(train) and len(test):
            splits.append((train, test))
    return splits


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


def trailing_realized_sofr_vol(sofr_daily: pd.Series, window_days: int) -> pd.Series:
    """Annualized trailing realized volatility of daily SOFR changes."""

    changes = sofr_daily.sort_index().diff()
    realized = changes.rolling(
        window=window_days,
        min_periods=max(5, int(window_days * 0.70)),
    ).std()
    return (realized * np.sqrt(252.0)).rename(f"trailing_realized_{window_days}d")


def _align_daily_to_weekly(series: pd.Series, weekly_index: pd.DatetimeIndex) -> pd.Series:
    return series.sort_index().reindex(weekly_index, method="ffill")


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
    """Evaluate forecast specifications on purged, embargoed blocked folds."""

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
    ).shift(1).rolling(26, min_periods=8).std()
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
            "total_pnl_hi": np.nan,
            "ann_sharpe_lo": np.nan,
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
    lo, hi = np.quantile(totals, [alpha / 2.0, 1.0 - alpha / 2.0])
    slo, shi = np.nanquantile(sharpes, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "total_pnl_lo": float(lo),
        "total_pnl_hi": float(hi),
        "ann_sharpe_lo": float(slo),
        "ann_sharpe_hi": float(shi),
    }


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
