# Data Contract

Required local workbooks:

- `project_cap_vol_ts.xlsx`: SOFR cap normal-vol and SOFR swap quote history.
- `cap_curves_2025-06-30.xlsx`: validation curve for the stripping pipeline.
- `ref_rates.xlsx`: daily SOFR reference-rate history.

These files are intentionally ignored by Git. Put them in the repo root
or `data/raw/` before running `python run_all.py --mode full`.

Generated aggregate research outputs may be written under `data/processed/`
during local runs. That directory is ignored because final public
provenance is carried by `tables/`, `figures/`, `paper/figures/`, and
`artifact_manifest.json`.
