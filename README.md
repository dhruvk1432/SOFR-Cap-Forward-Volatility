# SOFR Cap Forward Volatility

**Rates-volatility research repository | Public recruiter/trader artifact**

This project studies whether the forward-volatility curve implied by SOFR
caps behaves like an expectations curve or like a compensated
rates-volatility risk-premium curve. The repo strips caplet-level forward
normal volatility from SOFR cap flat vols, validates the curve construction,
tests the expectations-hypothesis analog, and turns the premium into a
leakage-aware rates-vol carry research screen.

The public claim is intentionally bounded. The evidence supports a
persistent forward-volatility premium and a researchable carry screen. It
does **not** claim executable caplet alpha. A production trading system
would require caplet marks, bid/ask, skew, margin, funding, liquidation
rules, and concrete cap/floor structures.

**Final deliverables**

- [Final paper PDF](paper/SOFR_Cap_Forward_Volatility.pdf)
- [Final LaTeX source](paper/SOFR_Cap_Forward_Volatility.tex)
- [Reproducible analysis pipeline](run_all.py)
- [Generated tables and numeric provenance](tables/)
- [README-ready figures](figures/)
- [Paper vector figures](paper/figures/)
- [Artifact manifest](artifact_manifest.json)

## Executive Summary

SOFR caps quote a flat volatility for a package of caplets. This project
strips the package into marginal forward-volatility points and asks a
trading question: are longer SOFR cap forward vols mostly forecasts of
future short-tenor volatility, or do they embed a risk premium that can be
monitored and potentially harvested?

The answer is a premium, not a clean forecast. Longer forward-vol points
sit persistently above later short forward vols and forward realized SOFR
volatility. A walk-forward ridge forecast converts the premium into a
stylized carry screen with costs, lagged risk scaling, CPCV forecast validation,
CPCV strategy checks, block-bootstrap paths, and block-permutation
null tests.

## Core Contributions

1. **Caplet stripping engine:** bootstraps SOFR discount and forward-rate
   curves, converts quoted normal caps to Black-equivalent flat vols,
   reprices caplets, and solves residual marginal forward vols.
2. **Independent validation:** reproduces a benchmark workbook on the
   2025-06-30 validation date with explicit error tolerances.
3. **Rates-vol premium evidence:** compares stripped forward vols with
   future short forward vols and forward realized SOFR volatility.
4. **Leakage-aware carry screen:** uses expanding-window forecasts,
   transaction costs, risk targeting, and only information available at
   the signal date.
5. **Anti-overfit controls:** CPCV forecast validation, CPCV
   strategy folds, feature ablations, interpolation robustness, block
   bootstrap intervals, and block-permutation nulls.
6. **Public-repo discipline:** raw workbooks and private tooling stay out
   of Git; tables, figures, and paper artifacts are generated and hashed.

## Paper in Brief

### Research Question

SOFR cap flat vols are bundled option-package quotes. The project asks:

**Do stripped SOFR cap forward vols behave like unbiased expectations of
future short-tenor volatility, or like compensated rates-volatility risk
premia?**

### Data

Local inputs:

- `project_cap_vol_ts.xlsx`: SOFR cap normal-vol and SOFR swap quote history.
- `cap_curves_2025-06-30.xlsx`: benchmark curve-validation workbook.
- `ref_rates.xlsx`: daily SOFR reference-rate history.

These workbooks are intentionally ignored by Git. Place them in the repo
root or `data/raw/` before running the full pipeline.

### Main Objects

A cap flat vol prices the bundled object:

```text
C(T, K, sigma_flat) = sum_i Z_i B(F_i, K, sigma_flat, t_i)
```

The stripped forward vol solves the marginal residual:

```text
C(T_i, K, sigma_flat_i) - previous caplets = Z_i B(F_i, K, sigma_fwd_i, t_i)
```

The empirical premium is:

```text
VTP(t, tau) = forward_vol(t, tau) - forward_vol(t + tau - 0.5, 0.5)
```

The carry screen sells forward vol when implied vol is rich versus a
walk-forward forecast of future realized SOFR volatility and buys it when
it is cheap.

## Repository Structure

```text
.
├── README.md
├── run_all.py
├── pyproject.toml
├── src/
│   ├── curve.py
│   ├── data.py
│   ├── inference.py
│   ├── pricing.py
│   ├── strategy.py
│   ├── validation.py
│   ├── plotting.py
│   ├── artifacts.py
│   └── sofr_forward_vol.py
├── scripts/
│   ├── 01_build_curves.py
│   ├── 02_analysis_tables.py
│   ├── 03_generate_figures.py
│   ├── 04_compile_paper_inputs.py
│   ├── 05_tex_integrity_check.py
│   └── 06_final_artifact_check.py
├── notebooks/
├── tables/
├── figures/
├── paper/
├── tests/
└── docs/
```

## Reproduce the Results

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### 2. Run the full pipeline

```bash
python run_all.py --mode full
```

Pipeline steps:

| Step | Script | Purpose |
|---|---|---|
| 1 | `scripts/01_build_curves.py` | Write processed forward-vol panels and validation curves. |
| 2 | `scripts/02_analysis_tables.py` | Regenerate CSV and LaTeX table provenance. |
| 3 | `scripts/03_generate_figures.py` | Regenerate README PNGs and paper PDF figures. |
| 4 | `scripts/04_compile_paper_inputs.py` | Regenerate paper source from scripted diagnostics. |
| 5 | `scripts/05_tex_integrity_check.py` | Check labels, refs, generated inputs, and figure usage. |
| 6 | `scripts/06_final_artifact_check.py` | Run public hygiene checks and write artifact hashes. |

### 3. Build the paper

```bash
cd paper
lualatex SOFR_Cap_Forward_Volatility.tex
```

### 4. Validate artifacts

```bash
python scripts/05_tex_integrity_check.py
python scripts/06_final_artifact_check.py
pytest -q
```

## Methods Glossary

- **SOFR cap:** interest-rate option package that pays when SOFR exceeds a strike.
- **Caplet:** one optionlet inside the cap package.
- **Flat vol:** single volatility quoted for the whole cap package.
- **Forward vol:** marginal volatility assigned to a future caplet after earlier
  caplets have been priced.
- **Normal vol:** Bachelier-style volatility in rate units.
- **Black vol:** lognormal caplet volatility used for the stripping bridge.
- **HAC/Newey-West:** standard errors robust to serial correlation from overlapping horizons.
- **CPCV:** combinatorial purged cross-validation over time blocks; all forecast and strategy validation in the paper uses purged/embargoed CPCV folds.
- **Block bootstrap:** resampling contiguous P&L blocks to preserve local dependence.

## Recruiter and Trader Notes

This repo demonstrates:

- fixed-income derivatives curve construction;
- caplet pricing and implied-vol inversion;
- leakage-aware financial ML validation;
- rates-volatility risk-premium measurement;
- cost-aware strategy research without overclaiming live tradability;
- public research hygiene with generated provenance and artifact checks.

## Disclaimer

This repository is an academic research project. It is not investment
advice, a production trading system, or a recommendation to trade SOFR
options.
