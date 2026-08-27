# Freight Rate Cold Start

Predict weekly refrigerated truck rates ($/mile) for **lanes** (`origin → destination`) using USDA public data. The project focuses on **cold-start lanes** — routes with little history — where a simple “repeat last week’s rate” baseline is weak.

**Sources:** USDA rates [`acar-e3r8`](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8) + EIA diesel via USDA [`x88w-atzp`](https://agtransport.usda.gov/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/x88w-atzp).

This repo is an **offline batch pipeline** (scripts + `make`) plus **visualization notebooks**. It is not a live API or dashboard.

---

## Quick start (~30 minutes)

Requires **Python ≥ 3.11** and network access for USDA downloads.

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup
make download              # ~800 lanes, data/raw/
make pipeline-smoke        # fast sanity check (2 HPO configs, 3 folds)
make pipeline-no-download  # full backtest eval (slower)
make score-no-download     # forward forecast for next Tuesday
jupyter notebook notebooks/05_walkforward_results.ipynb
```

**Suggested reading order:** this README → [`data/README.md`](data/README.md) → notebook **03** (data contract) → **04** (after `make hpo`) → **05** (after `make pipeline` + `make score`).

---

## Two pipelines (don’t mix them up)

| | **Training / eval** (`make pipeline`) | **Operational score** (`make score`) |
|---|---|---|
| **Purpose** | Measure model quality on history | Produce next week’s forecast |
| **When** | After data refresh, occasionally | Every USDA weekly release |
| **Trains on** | All rows with `date < t` | All rows with `date < t` (through last observed week) |
| **Predicts** | Each historical week `t` (backtest) | **Next** week-ending Tuesday after latest data |
| **Has actual `y_true`?** | Yes (USDA published that week) | No — forward forecast (`y_true` is NaN) |
| **Outputs** | `models/walkforward_gbm/` | `models/score/` |
| **θ* (hyperparameters)** | Tuned in HPO step | Fixed from `models/nested_hpo/best_params.json` |

**Example:** if USDA data runs through **2026-08-18** (Tuesday), then `make score` forecasts **2026-08-25** using history through the 18th. See [`models/README.md`](models/README.md) for artifact schemas.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup
make download
make help    # list all targets
```

---

## Training pipeline (backtest / eval)

Scripts (and `make`) own all computation; **notebooks are visualization only**.

```bash
make pipeline              # download → HPO → walk-forward → manifest
make pipeline-no-download  # skip download when data/raw/ exists
make pipeline-smoke        # fast check (2 HPO configs, 3 folds)
```

| Target | Script | What it does | Output |
|---|---|---|---|
| `download-rates` | `download_usda_data.py` | Fetches weekly refrigerated truck rates | `data/raw/usda_refrigerated_truck_rates.parquet` |
| `download-diesel` | `download_diesel_data.py` | Fetches weekly US on-highway diesel | `data/raw/usda_diesel_weekly.parquet` |
| `hpo` | `run_nested_hpo_gbm.py` | Grid search on val window; picks θ* | `models/nested_hpo/best_params.json` |
| `walkforward` | `train_walkforward_gbm.py` | Full expanding-window eval with θ* | `models/walkforward_gbm/*` |
| `manifest` | — | Run metadata (θ*, windows, diesel mode) | `models/run_manifest.json` |

Equivalent: `python scripts/run_pipeline.py`

### Temporal protocol (Case B)

- **Unit of time:** week-ending **Tuesday** (`date` in USDA data).
- **Expanding walk-forward:** for each forecast week `t`, train on `date < t`, predict `date == t`.
- **Eval start:** first forecast Tuesday **2025-01-07** (after burn-in from **2024-07-01**).
- **Target:** residual `rpm − rpm_lag_1`; final prediction = `rpm_lag_1 + delta_hat`.
- **Baseline:** lag-1 (last week’s RPM for that lane).
- **Diesel:** `diesel_us`, `diesel_distance` with Case-B as-of lag (never same-week diesel for Tuesday `t`).

| Window | Dates | Role |
|---|---|---|
| API / burn-in | `date ≥ 2024-07-01` | Minimum history before 2025 forecasts |
| HPO validation | `2025-01-07` → `2025-06-24` | Pick θ* (hyperparameters) |
| Test | `≥ 2025-07-01` | Held-out eval in walk-forward only |
| Operational score | `last_observed + 7 days` | Forward forecast (`make score`) |

### How we judge the model (headline metrics)

We compare the GBM to a **lag-1 baseline** (“predict this week = last week’s RPM”). Error is **MAE** (mean absolute error) in **$/mile**. Lower MAE is better.

**Lift** = `MAE_baseline − MAE_model`. Positive lift means the GBM beats lag-1.

We report **two slices** (this is what “dual-regime” means):

| Slice | Who is included | Why it matters |
|---|---|---|
| **Overall** | All lane-weeks in the window | General accuracy |
| **Cold-start (0–4)** | Lanes with **0–4 prior weekly observations** in training history (`lane_history_n ≤ 4`) | Core product goal — new or sparse lanes where lag-1 is weakest |

**Example CLI output after walk-forward:**

```
=== Dual-regime headline [full] ===
overall: n=5,689  MAE=0.1402  lag1=0.1393  lift=-0.0009  beats_lag1=False
cold 0-4: n=246  MAE=0.3896  lag1=0.4068  lift=+0.0171  beats_lag1=True
```

Read this as:

- **Overall:** GBM is roughly tied with lag-1 (slightly worse; lift negative).
- **Cold-start 0–4:** GBM wins — MAE 0.39 vs 0.41 for lag-1; lift **+0.017** $/mile on sparse lanes.

**HPO selection rule for θ*:** among configs with **cold-start lift ≥ 0**, pick the **lowest overall MAE**. If none qualify, pick best cold-start lift.

See also [Glossary](#glossary) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Operational scoring (weekly forecast)

1. **Tune θ* (infrequent):** `make hpo` when validation performance drifts — not every week.
2. **Score weekly:** `make score` after each USDA release — download latest data, re-fit GBM with fixed θ*, forecast **next Tuesday**.

```bash
make score                 # download → score next Tuesday
make score-no-download     # score with existing data/raw/
python scripts/score_week.py --date 2026-08-25   # specific week
```

| Target | Script | Output |
|---|---|---|
| `score` | `score_week.py` | `models/score/score_*_predictions.parquet`, `latest_predictions.parquet`, `latest.json` |
| `score-no-download` | same, no download | same |

Requires `models/nested_hpo/best_params.json` from `make hpo` or `make pipeline`.

---

## Notebooks

| # | Notebook | When to run | Purpose |
|---|---|---|---|
| 01–02 | EDA / signal | Optional | Historical exploration; not required for pipeline |
| **03** | `03_base_dataset_and_splits.ipynb` | Recommended | Week-ending Tuesday contract, panel grain, splits |
| **04** | `04_hpo_results.ipynb` | After `make hpo` or `make pipeline` | θ* selection charts |
| **05** | `05_walkforward_results.ipynb` | After `make pipeline` **and** `make score` | Backtest charts + red forward forecast point |

```bash
jupyter notebook notebooks/04_hpo_results.ipynb
jupyter notebook notebooks/05_walkforward_results.ipynb
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Lane** | One `origin → destination` pair at weekly grain |
| **RPM** | Rate per mile ($/mile); model target |
| **Week-ending Tuesday** | USDA `date`; e.g. `2026-08-18` ≈ week Aug 12–18 |
| **Case B** | Train on `date < t`, score `date == t` (no same-week leakage) |
| **θ*** | Fixed GBM hyperparameters from HPO (`best_params.json`) |
| **Lag-1 baseline** | Predict `rpm` = previous week’s `rpm` for that lane |
| **Residual target** | GBM predicts `rpm − rpm_lag_1`; add lag-1 back for final RPM |
| **Lane history (`lane_history_n`)** | Count of prior weeks seen for that lane before week `t` |
| **Cold-start bucket 0–4** | Lanes with 0–4 prior observations — main evaluation focus |
| **Lift** | `MAE_baseline − MAE_model`; positive = model beats lag-1 |
| **Dual-regime** | Report metrics on **overall** lanes and **cold-start 0–4** separately |
| **Scaffold** | Synthetic lane rows for a future Tuesday (no USDA `rpm` yet) |
| **`last_observed_date`** | Latest Tuesday present in downloaded USDA data |
| **`forecast_date`** | Tuesday being predicted (`make score` → usually `last_observed + 7d`) |

---

## Layout

```
freight_rate/
├── Makefile
├── README.md                 ← you are here
├── data/README.md            ← sources, columns, download
├── models/README.md          ← artifact schemas
├── docs/ARCHITECTURE.md      ← code module map
├── data/raw/                 ← gitignored snapshots
├── models/                   ← gitignored outputs
├── notebooks/
├── scripts/
├── src/freight_rates/
└── tests/
```

---

## Development

```bash
make test
make lint
```

---

## Data & leakage rules

- **Scope:** ~800 lanes across 10 USDA wholesale markets (varies by week).
- **Never use as features:** `weeklow`, `weekhigh`, `midpoint` (target leakage from rate bands).
- **Availability:** use `availability_lag_1` only — same-week availability is Case B leakage.
- **Diesel:** weekly Monday series; joined as-of `<= t − 7` days.

Details: [`data/README.md`](data/README.md)

---

## FAQ

**`make score` fails: HPO params not found**  
Run `make hpo` or `make pipeline` first to create `models/nested_hpo/best_params.json`.

**NB05 has no red dot**  
Run `make score-no-download` and re-execute cells 1 and 3.

**Why only ~73 lanes in score vs ~800 in docs?**  
Score uses lanes **active in the last observed week** (scaffold copies those OD pairs forward).

**`y_true` is NaN in `models/score/`**  
Expected — operational score is a **forward** forecast; actuals arrive next USDA release.

**`make pipeline` is slow**  
Use `make pipeline-smoke` locally; full run is for proper eval.

**What is `models/diesel_ablation/`?**  
Offline experiment comparing diesel feature modes — not part of the default pipeline.
