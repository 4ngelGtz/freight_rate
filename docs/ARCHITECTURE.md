# Architecture

Data flow from USDA download to weekly forecast.

```
USDA API                    EIA diesel (USDA mirror)
    │                              │
    ▼                              ▼
data/raw/*.parquet          data/raw/usda_diesel_weekly.parquet
    │                              │
    └──────────┬───────────────────┘
               ▼
    ingestion.load_raw_snapshot()
               ▼
    preprocessing.build_modeling_panel()
      • collapse to lane-week (origin, destination, date)
      • attach diesel Case-B as-of
               ▼
    splits.filter_model_window()     ← date ≥ 2024-07-01
               │
       ┌───────┴───────┐
       ▼               ▼
  make pipeline    make score
  (backtest)       (forward forecast)
       │               │
       │         preprocessing.extend_panel_for_forecast()
       │           • scaffold lanes for next Tuesday
       ▼               ▼
    hpo.run_nested_hpo()     score_week.py
      • pick θ* on val              │
       ▼                            │
    walkforward.run_walkforward_gbm()  ← same core loop
      • each week t:
          train date < t
          features.LaneWeekFeatureBuilder
          HistGradientBoostingRegressor (θ*)
          predict date == t
       ▼               ▼
 models/walkforward_gbm/   models/score/
       │               │
       └───────┬───────┘
               ▼
    notebooks 04–05 (visualization only)
```

---

## Module map (`src/freight_rates/`)

| Module | Responsibility |
|---|---|
| `ingestion.py` | Download / load USDA rate parquet; schema validation |
| `diesel.py` | Diesel download, US weekly series, Case-B as-of merge |
| `preprocessing.py` | Lane-week panel, diesel attach, **forecast scaffold** |
| `features.py` | Lags, calendar, OHE; `LaneWeekFeatureBuilder` |
| `splits.py` | Temporal windows, walk-forward folds, `resolve_forecast_date` |
| `walkforward.py` | Expanding-window GBM train/score loop |
| `hpo.py` | Grid search θ* on validation window |
| `evaluation.py` | MAE, history buckets, **dual-regime headline** |
| `artifacts.py` | Loaders for notebooks (`require_*_artifacts`) |

---

## Scripts (`scripts/`)

| Script | Entry for |
|---|---|
| `download_usda_data.py` | Rate snapshot |
| `download_diesel_data.py` | Diesel snapshot |
| `run_nested_hpo_gbm.py` | HPO |
| `train_walkforward_gbm.py` | Walk-forward backtest |
| `score_week.py` | Operational forward score |
| `run_pipeline.py` | Full eval pipeline |

All training/scoring logic lives in `src/`; scripts are thin CLI wrappers.

---

## Design choices (for new contributors)

1. **Case B only** — no same-week features in train for week `t`; lags from prior weeks.
2. **Global GBM** — one model across all lanes; cold-start handled via history features + residual over lag-1.
3. **θ* fixed between HPO runs** — weekly ops re-fit the model on expanding history, not hyperparameters.
4. **Notebooks read artifacts** — they do not retrain; keeps research reproducible from `make` outputs.
