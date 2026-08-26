# Freight Rate Cold Start

Lane-level refrigerated truck rate prediction with cold-start focus, using USDA `acar-e3r8` + EIA diesel (`x88w-atzp`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup
make download
```

## Training pipeline

Scripts (and `make`) own all computation; notebooks are visualization only.

```bash
make help                  # list all targets
make pipeline              # download → HPO → walk-forward → manifest
make pipeline-no-download  # skip download when data/raw/ exists
make pipeline-smoke        # fast check (2 HPO configs, 3 folds)
```

| Target | Script | What it does | Output |
|---|---|---|---|
| `download-rates` | `download_usda_data.py` | Fetches weekly refrigerated truck rates from USDA API | `data/raw/usda_refrigerated_truck_rates.parquet` |
| `download-diesel` | `download_diesel_data.py` | Fetches weekly US on-highway diesel (EIA via USDA) | `data/raw/usda_diesel_weekly.parquet` |
| `hpo` | `run_nested_hpo_gbm.py` | Grid search on val window; picks θ* (cold-start lift ≥ 0, then lowest MAE) | `models/nested_hpo/best_params.json` |
| `walkforward` | `train_walkforward_gbm.py` | Full expanding-window eval with θ*: re-fits GBM each week, scores val + test, writes predictions and MAE metrics | `models/walkforward_gbm/*` |
| `manifest` | — | Writes run metadata (θ*, windows, diesel mode) | `models/run_manifest.json` |

Equivalent one-liner: `python scripts/run_pipeline.py` (same steps as `make pipeline`).

**Protocol (Case B):** expanding walk-forward — train `date < t`, score `date == t` from `2025-01-07`. Residual target `rpm − rpm_lag_1`; baseline = lag-1. Diesel always included (`diesel_us`, `diesel_distance`) with as-of lag.

**HPO:** tune θ* on val (`2025-01-07` → `2025-06-24`). Test (`≥ 2025-07-01`) scored only in `walkforward`.

**Headline metric:** dual-regime — overall MAE lift + cold-start 0–4 lift.

## Recommended operational cadence

1. **Tune θ* (infrequent):** run `make hpo` on the validation window when you want to refresh hyperparameters (not every week).
2. **Score weekly (each USDA release):** re-fit the GBM on all history `date < t` with fixed θ* from `models/nested_hpo/best_params.json`, then predict lanes for `date == t`.
3. **Re-tune θ* only** when validation performance drifts — the weekly step keeps θ* fixed.

```bash
make score                 # download → score latest week
make score-no-download     # score with existing data/raw/
python scripts/score_week.py --date 2025-08-26   # specific week
```

## Scoring pipeline

| Target | Script | What it does | Output |
|---|---|---|---|
| `score` | `score_week.py` | Download fresh snapshots, re-fit GBM on `date < t` with θ*, predict `date == t` (latest week by default) | `models/score/score_*_predictions.parquet`, `latest_predictions.parquet`, `latest.json` |
| `score-no-download` | `score_week.py` | Same without download | same |

Requires `models/nested_hpo/best_params.json` from `make hpo` (or `make pipeline`).

## Artifacts (`models/`, gitignored)

```
models/
├── run_manifest.json
├── nested_hpo/best_params.json, hpo_val_summary.csv
├── walkforward_gbm/*_predictions.parquet, *_metrics_*.csv
└── score/score_*_predictions.parquet, latest_predictions.parquet, latest.json
```

## Notebooks

| # | Notebook | Role |
|---|---|---|
| 01–03 | exploration | EDA, signal, splits |
| 04 | `04_hpo_results.ipynb` | HPO val charts |
| 05 | `05_walkforward_results.ipynb` | Walk-forward charts + val/test slices |

Run after `make pipeline`: `jupyter notebook notebooks/04_hpo_results.ipynb`

## Layout

```
freight_rate/
├── Makefile
├── data/raw/
├── models/
├── notebooks/
├── scripts/
├── src/freight_rates/
└── tests/
```

## Development

```bash
make test
make lint
```

## Data notes

- **Scope:** ~800 lanes → 10 USDA wholesale markets.
- **Target:** `rpm` ($/mile). No leakage from rate bands; `availability_lag_1` only.
- **Sources:** [acar-e3r8](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8), [x88w-atzp](https://agtransport.usda.gov/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/x88w-atzp). See `data/README.md`.
