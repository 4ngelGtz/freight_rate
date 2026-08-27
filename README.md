# Freight Rate Cold Start

Lane-level refrigerated truck rate prediction with cold-start focus, using USDA `acar-e3r8` + EIA diesel (`x88w-atzp`).

## Executive summary

**Problem:** predict weekly refrigerated truck rates (`rpm`, $/mile) for origin→destination **lanes**. Many lanes have little history (“cold start”); a strong lag-1 baseline is hard to beat everywhere, but the GBM should help on sparse lanes.

**Model:** global `HistGradientBoostingRegressor`, re-fit each week on all history `date < t` with fixed hyperparameters θ* from Hyper Parameter Optimization. Trains on the residual `rpm − rpm_lag_1` and adds the prediction back to the lag-1 baseline. Features: lane OD, distance, calendar, lagged RPM/availability, diesel (Case-B as-of).

**Evaluation (Case B):** expanding walk-forward — train past only, score week `t`. HPO picks θ* on val (`2025-01-07` → `2025-06-24`); test (`≥ 2025-07-01`) is scored once in the full walk-forward run.

**Success metric (“dual-regime”):** we report MAE in two regimes, not a single number:

| Regime | Who | Question |
|---|---|---|
| **Overall** | all lane-weeks | Does the GBM beat lag-1 on average? (`mae_lift = MAE_baseline − MAE_model`) |
| **Cold-start** | lanes with ≤4 prior weeks (`history_bucket` 0–4) | Does the GBM beat lag-1 where history is thin? |

θ* is chosen on val with **cold-start lift ≥ 0**, then lowest overall MAE. The model is not expected to beat lag-1 on every lane-week; cold-start lift is the primary design goal.

**Operations:** `make pipeline` = offline backtest; `make score` = weekly forward forecast for the **next** Tuesday after the latest USDA data (see [Scoring pipeline](#scoring-pipeline)).

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

**HPO:** tune θ* on val (`2025-01-07` → `2025-06-24`). Test (`≥ 2025-07-01`) scored only in `walkforward`.

See [Executive summary](#executive-summary) for protocol and dual-regime metrics in full.

## Recommended operational cadence

1. **Tune θ* (infrequent):** run `make hpo` on the validation window when you want to refresh hyperparameters (not every week).
2. **Score weekly (each USDA release):** after downloading the latest published week, re-fit the GBM on all history `date < t` with fixed θ*, then predict the **next** week-ending Tuesday (`t = last_observed + 7 days`).
3. **Re-tune θ* only** when validation performance drifts — the weekly step keeps θ* fixed.

```bash
make score                 # download → score next Tuesday (forward forecast)
make score-no-download     # score with existing data/raw/
python scripts/score_week.py --date 2026-08-25   # specific week
```

## Scoring pipeline

| Target | Script | What it does | Output |
|---|---|---|---|
| `score` | `score_week.py` | Download fresh snapshots, re-fit GBM on `date < t` with θ*, predict the **next** week-ending Tuesday after the latest USDA data (scaffold lanes from last observed week) | `models/score/score_*_predictions.parquet`, `latest_predictions.parquet`, `latest.json` |
| `score-no-download` | `score_week.py` | Same without download | same |

Requires `models/nested_hpo/best_params.json` from `make hpo` (or `make pipeline`).

**Example:** data through `2026-08-18` → `make score` forecasts `2026-08-25` (73 lanes, no `y_true` yet).

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
