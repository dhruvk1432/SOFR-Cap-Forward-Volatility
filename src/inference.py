from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
