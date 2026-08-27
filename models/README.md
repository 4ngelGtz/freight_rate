# Model artifacts

Everything under `models/` is **gitignored**. Regenerate with `make pipeline` (eval) and/or `make score` (forecast).

---

## Directory layout

```
models/
├── run_manifest.json              # last pipeline run metadata
├── nested_hpo/
│   ├── best_params.json           # θ* — fixed GBM hyperparameters
│   ├── hpo_val_summary.csv        # all HPO configs on val window
│   └── hpo_val_folds_best.csv     # per-week preds for best config
├── walkforward_gbm/               # full backtest (make walkforward)
│   ├── walkforward_gbm_predictions.parquet
│   ├── walkforward_gbm_metrics_overall.csv
│   ├── walkforward_gbm_metrics_by_week.csv
│   ├── walkforward_gbm_metrics_by_history.csv
│   └── walkforward_gbm_metrics_dual_regime.csv
└── score/                         # operational forward forecast (make score)
    ├── latest_predictions.parquet
    ├── latest.json
    └── score_YYYY-MM-DD_predictions.parquet
```

**Not part of default pipeline:** `models/diesel_ablation/` — one-off diesel feature comparison.

---

## `nested_hpo/best_params.json`

GBM hyperparameters (θ*) selected on the validation window.

```json
{
  "config": "cfg_05",
  "max_depth": 3,
  "min_samples_leaf": 60,
  "l2_regularization": 1.0,
  "learning_rate": 0.08,
  "max_iter": 200
}
```

Used by `make walkforward` and `make score`. Re-tune with `make hpo` — not every week.

---

## `run_manifest.json`

Pipeline run metadata: git commit, diesel mode, val/test windows, paths to θ*.

---

## Predictions parquet (`*_predictions.parquet`)

One row per **lane-week** forecast.

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Forecast week-ending Tuesday |
| `lane_id` | str | `"ORIGIN -> DESTINATION"` |
| `origin` | str | USDA origin market |
| `destination` | str | USDA destination market |
| `y_pred` | float | GBM RPM prediction ($/mile) |
| `y_pred_baseline` | float | Lag-1 baseline ($/mile) |
| `delta_hat` | float | GBM residual; `y_pred = y_pred_baseline + delta_hat` |
| `y_true` | float | Actual RPM when published; **NaN** for forward score |
| `lane_history_n` | float | Prior weekly obs for this lane before `date` |
| `history_bucket` | str | `0-4`, `5-19`, `20-99`, `100+` |
| `residual` | float | `y_true - y_pred` (NaN when `y_true` missing) |

**Load example:**

```python
import pandas as pd
preds = pd.read_parquet("models/score/latest_predictions.parquet")
preds[["origin", "destination", "y_pred", "y_pred_baseline"]].head()
```

---

## `score/latest.json`

Metadata for the most recent operational score.

| Field | Description |
|---|---|
| `forecast_date` | Tuesday being predicted (e.g. `2026-08-25`) |
| `last_observed_date` | Latest Tuesday in USDA data (e.g. `2026-08-18`) |
| `n_predictions` | Number of lane forecasts |
| `predictions_path` | Filename of dated parquet |
| `hpo_params_path` | Path to θ* used |
| `created_at` | UTC timestamp |

---

## Metrics CSVs

Produced by walk-forward backtest and score runs (when `y_true` exists).

| File | Content |
|---|---|
| `*_metrics_overall.csv` | Single-row MAE / lift vs lag-1 for all rows |
| `*_metrics_by_week.csv` | MAE per forecast Tuesday |
| `*_metrics_by_history.csv` | MAE by history bucket (`0-4`, `5-19`, …) |
| `*_metrics_dual_regime.csv` | Headline: overall + cold-start 0–4 in one row |

**Key metric columns:**

| Column | Meaning |
|---|---|
| `mae` | Mean absolute error of GBM ($/mile) |
| `mae_baseline` | MAE of lag-1 baseline |
| `mae_lift` | `mae_baseline - mae` — **positive = GBM wins** |
| `cold_mae`, `cold_mae_lift` | Same, for cold-start bucket 0–4 only (dual-regime file) |

Forward score runs have `y_true = NaN`, so MAE columns in score metrics will be NaN — that is expected.
