"""Download optional public FRED context series into data/raw.

The core project runs from the local course workbooks.  This helper is for
research extensions that condition rates-vol signals on macro/rates regimes.
It requires no credentials because it uses FRED's public graph CSV endpoint.
Downloaded CSV files are intentionally ignored by git.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


DEFAULT_SERIES = [
    "SOFR",
    "DFF",
    "DGS2",
    "DGS5",
    "DGS10",
    "T10Y2Y",
    "BAMLC0A0CM",
    "VIXCLS",
]


def fetch_series(series_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    target = output_dir / f"{series_id}.csv"
    with urlopen(url, timeout=30) as response:
        target.write_bytes(response.read())
    return target


def main() -> None:
    output_dir = Path("data/raw/fred")
    for series_id in DEFAULT_SERIES:
        path = fetch_series(series_id, output_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
