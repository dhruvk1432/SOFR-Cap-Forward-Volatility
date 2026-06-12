# SOFR Cap Forward Volatility

This project strips forward volatility from SOFR cap quotes and asks whether the
cap-implied forward volatility curve behaves like an expectations curve or like
a risk-premium curve.  The upgraded strategy layer then asks whether that
premium can be converted into an empirical rates-vol carry rule after
out-of-sample forecasting, transaction costs, volatility targeting, and regime
attribution.

## What is in the repo

- `SOFR_Cap_Forward_Volatility.ipynb`: presentation-ready research notebook.
- `paper/SOFR_Cap_Forward_Volatility_Mini_Paper.pdf`: short mini-paper for first-pass reading.
- `src/sofr_forward_vol.py`: reusable curve construction, caplet pricing, and
  regression/strategy utilities.
- `scripts/fetch_fred_series.py`: optional public-data downloader for macro and
  rate-regime context.  Downloaded CSV files stay under ignored `data/raw/`.
- `docs/knowledge_base_protocol.md`: local NotebookLM/knowledge-base protocol
  used to ground the research extension without shipping private notes.
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
7. Build a walk-forward realized-vol forecast from only lagged information.
8. Trade a stylized forward-vol carry signal when implied volatility is rich or
   cheap versus that forecast, with transaction costs and risk scaling.
9. Treat the result as an investability screen: the next research step is a
   caplet-level mark-to-market and execution model, not a production claim.

## Data

Raw Excel workbooks are not tracked. Place these files in the repo root or
`data/raw/` before running the notebook:

- `project_cap_vol_ts.xlsx`
- `cap_curves_2025-06-30.xlsx`
- `ref_rates.xlsx`

The notebooks show only summary statistics, plots, and validation diagnostics.
Optional public FRED series can be downloaded with:

```bash
python scripts/fetch_fred_series.py
```

Those files are written to `data/raw/fred/` and are not tracked.
