# SOFR Cap Forward Volatility

This project strips forward volatility from SOFR cap quotes and asks whether the
cap-implied forward volatility curve behaves like an expectations curve or like
a risk-premium curve.

## What is in the repo

- `SOFR_Cap_Forward_Volatility.ipynb`: presentation-ready research notebook.
- `paper/SOFR_Cap_Forward_Volatility_Mini_Paper.pdf`: short mini-paper for first-pass reading.
- `src/sofr_forward_vol.py`: reusable curve construction, caplet pricing, and
  regression utilities.
- `tests/test_sofr_forward_vol.py`: smoke tests that run when the local data
  workbooks are present.
- `data/README.md`: data contract and public-repo policy.

## Research design

1. Bootstrap a quarterly SOFR discount and forward-rate curve.
2. Convert quoted normal cap vols into Black-equivalent flat vols to replicate
   the course validation workbook.
3. Strip caplet forward vols by repricing each cap and solving the marginal
   caplet volatility.
4. Convert stripped Black vols back into normal-vol units for time-series work.
5. Test an expectations-hypothesis analog with HAC standard errors and effective
   non-overlapping sample counts.
6. Compare cap-implied forward vol against forward realized SOFR volatility and
   policy-cycle regimes.

## Data

Raw Excel workbooks are not tracked. Place these files in the repo root or
`data/raw/` before running the notebook:

- `project_cap_vol_ts.xlsx`
- `cap_curves_2025-06-30.xlsx`
- `ref_rates.xlsx`

The notebooks show only summary statistics, plots, and validation diagnostics.
