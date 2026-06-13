# SOFR Cap Forward Volatility

This repository studies whether the forward-volatility curve implied by SOFR
caps behaves like an expectations curve or like a compensated volatility-risk
premium. The project strips caplet-level forward normal volatility from quoted
SOFR cap flat vols, validates the curve construction, tests the
expectations-hypothesis analog, and then turns the premium into a walk-forward
rates-volatility carry screen.

The central result is intentionally disciplined. The evidence supports a
persistent forward-volatility premium, but it does not claim that the current
notebook is a production caplet trading system. The strategy layer is an
investability test: lagged features, expanding-window ridge forecasts,
transaction costs, leakage-safe volatility targeting, regime attribution,
purged/CPCV validation, feature ablations, block-bootstrap uncertainty, and
block-permutation null tests are used to decide whether the premium deserves
deeper product-level execution research.
The robustness summaries report mean and median simulated outcomes, including
median cumulative P&L paths, so the public claim is not based on a single
average endpoint.

## What is in the repo

- `SOFR_Cap_Forward_Volatility.ipynb`: executed research notebook.
- `paper/SOFR_Cap_Forward_Volatility_Mini_Paper.pdf`: single-column empirical
  paper explaining the research object, methodology, results, limitations, and
  claim audit.
- `paper/figures/`: vector figures used in the paper.
- `src/sofr_forward_vol.py`: reusable curve construction, caplet pricing,
  regression, and strategy utilities.
- `scripts/fetch_fred_series.py`: optional public-data downloader for macro and
  rate-regime context. Downloaded CSV files stay under ignored `data/raw/`.
- `docs/knowledge_base_protocol.md`: local NotebookLM/knowledge-base protocol
  used to ground the research extension without shipping private research
  material.
- `tests/test_sofr_forward_vol.py`: smoke tests for the reusable implementation.
- `data/README.md`: data contract and public-repo policy.

## Claim hierarchy

The repo is written to make the strength of each claim clear:

1. **Implemented and validated:** SOFR cap flat vols are converted into a
   caplet-level forward normal-volatility curve.
2. **Strong descriptive result:** longer forward-vol points are far above later
   short forward vols and forward realized SOFR volatility in the post-2022
   sample.
3. **Disciplined inference:** HAC standard errors and effective
   non-overlapping sample counts are reported because the horizon tests overlap.
4. **Researchable strategy result:** a walk-forward realized-vol forecast and
   cost-aware carry rule produce positive stylized proxy P&L.
5. **Anti-overfit evidence:** purged forecast folds, CPCV-style strategy folds,
   sensitivity checks, block-permutation nulls, and median bootstrap paths
   reduce the chance that the result is only path fitting.
6. **Not yet claimed:** executable caplet alpha. A production version needs
   caplet marks, bid/ask, skew, margin, and concrete cap/floor structures.

## Research design

1. Bootstrap a quarterly SOFR discount and forward-rate curve.
2. Convert quoted normal cap vols into Black-equivalent flat vols to replicate
   the validation workbook.
3. Strip caplet forward vols by repricing each cap and solving the marginal
   caplet volatility.
4. Convert stripped Black vols back into normal-vol units for time-series work.
5. Test an expectations-hypothesis analog with HAC standard errors and
   effective non-overlapping sample counts.
6. Compare cap-implied forward vol against forward realized SOFR volatility and
   policy-cycle regimes.
7. Build a walk-forward realized-vol forecast from only lagged information.
8. Trade a stylized forward-vol carry signal when implied volatility is rich or
   cheap versus that forecast, with transaction costs and risk scaling.
9. Stress the forecast and strategy with purged/embargoed folds, CPCV-style
   strategy configurations, block bootstrap intervals, block-permutation nulls,
   median cumulative path summaries, and parameter sensitivity tables.

## Data

Raw Excel workbooks are not tracked. Place these files in the repo root or
`data/raw/` before running the notebook:

- `project_cap_vol_ts.xlsx`
- `cap_curves_2025-06-30.xlsx`
- `ref_rates.xlsx`

The notebook reports only summary statistics, plots, and validation diagnostics.
Optional public FRED series can be downloaded with:

```bash
python scripts/fetch_fred_series.py
```

Those files are written to `data/raw/fred/` and are not tracked.

## How to review

Start with the PDF paper for the claim hierarchy. Then open the notebook to see
the executed calculations and figures. The reusable implementation is kept in
`src/`, and the test suite provides quick checks that the public code still
loads and preserves key numerical behavior.
