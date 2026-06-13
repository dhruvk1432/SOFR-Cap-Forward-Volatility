from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
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
