# Freight Rate Cold Start

Study lane-level freight-rate prediction and cold-start behavior using the USDA Refrigerated Truck Rates dataset.

## Objective

Build a reproducible pipeline to predict refrigerated truck rates at the origin–destination lane level, with a focus on sparse and newly observed lanes (cold start).

## Planned model comparison

1. **Global Gradient Boosting** — expanding-window walk-forward (`scripts/train_walkforward_gbm.py`, `freight_rates.walkforward`)
2. **Hierarchical Bayesian model** — partial pooling for sparse / new lanes (not implemented yet)

## Data pipeline

```
USDA AgTransport API
  → raw Parquet snapshot (data/raw/)
      - refrigerated truck rates (acar-e3r8)
      - weekly on-highway diesel (x88w-atzp, EIA via USDA)
  → lane-week panel + Case-B diesel as-of merge
  → feature engineering
  → modeling
```

**Selected sources:**

- [Refrigerated Truck Rates and Availability](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8) (`acar-e3r8`)
- [Weekly On-Highway Diesel Fuel Prices](https://agtransport.usda.gov/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/x88w-atzp) (`x88w-atzp`) — EIA retail on-highway diesel republished on AgTransport

This is the official **weekly** historical origin–destination series (rates, distance, availability, commodity) on USDA’s [Agricultural Transportation Open Data](https://agtransport.usda.gov/) platform, sourced from the AMS Specialty Crops [Fruit and Vegetable Truck Rate Report](https://www.ams.usda.gov/mnreports/fvwtrk.pdf) (FVWTRK).

### Scope: 10 USDA destination markets

This project uses **only** the weekly `acar-e3r8` series: refrigerated truck spot rates from U.S. agricultural shipping areas to **ten fixed wholesale destination markets** — not general freight to arbitrary U.S. cities.

Per the weekly FVWTRK methodology, rates are quoted to: Atlanta, Baltimore, Boston, Chicago, Dallas, Los Angeles, Miami, New York, Philadelphia, and Seattle. **Origin districts (~107)** and **lanes (~800)** provide most of the cross-sectional variation; cold-start experiments target sparse or newly observed **lanes**, not new destination cities.

The curated modeling population is the **lane-week panel** (`date + origin + destination`), built from the raw snapshot via `build_lane_week_panel()` — one row per lane-week, with `rpm` aggregated by mean when reporting duplicates exist. `build_modeling_panel()` then attaches **US diesel** features with a Case-B as-of merge (see below).

### Diesel (exogenous cost driver)

EIA weekly on-highway diesel (`x88w-atzp`) is week-ending **Monday**; rate `date` is week-ending **Tuesday**. For forecasts of Tuesday `t`, only information through the prior week-ending Tuesday `t-7` is allowed. Diesel is therefore joined with:

```text
latest diesel Monday where diesel_date <= t - 7 days
```

Same-week Monday diesel (`t-1`) is **not** used. Panel columns: `diesel_us`, `diesel_us_chg_1w` (WoW change), plus audit `diesel_date`. Production default uses **`diesel_us` + `diesel_distance`** (`diesel_us × distance/1000`) via `--diesel level`.

### Field glossary

Definitions follow the weekly AMS Specialty Crops Market News [Fruit and Vegetable Truck Rate Report](https://www.ams.usda.gov/mnreports/fvwtrk.pdf) and the [`acar-e3r8`](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8) catalog.

| Field | Meaning |
|---|---|
| `date` | Week-ending Tuesday for the weekly spot quotes — not an individual shipment timestamp |
| `origin` | Agricultural shipping district / producing area |
| `destination` | One of the 10 wholesale receiving cities listed above |
| `distance` | Typical haul miles from origin area to destination city |
| `commodity` | Commodity (or commodity group) for which the rate was reported |
| `weeklow` / `weekhigh` | Spot truckload rate range for the week ($/load) |
| `midpoint` | Midpoint of the weekly rate band (used with distance to derive RPM) |
| `rpm` | Rate per mile ($/mile) — modeling target; reconstructs as `midpoint / distance` |
| `availability` | Spot refrigerated-truck availability at origin (ordinal 1–5; see below) |
| `region` | AMS Transportation Services regional assignment |

**Week window for `date`:** FVWTRK labels each report as *week ending Tuesday*. A row with `date = 2026-08-18` covers roughly **Aug 12–18** (prior Wednesday through that Tuesday), not Aug 18–24. One-step-ahead prediction of `date = 2026-08-25` therefore targets the next week-ending Tuesday (rates for ~Aug 19–25), using only information available through the prior week-ending Tuesday.

**`availability` scale** (labels as published in FVWTRK; numeric codes as stored in `acar-e3r8`):

| Code | Label |
|---:|---|
| 1 | Surplus |
| 2 | Slight surplus |
| 3 | Adequate |
| 4 | Slight shortage |
| 5 | Shortage |

Rates are open (**spot**) market truckload quotes including broker fees for single-destination loads in 48–53 ft refrigerated trailers. For modeling, treat `weeklow`, `weekhigh`, and `midpoint` as **target leakage** if `rpm` is the outcome; `distance` is a legitimate ex-ante feature. Same-week `availability` is published with the same weekly report as `rpm`, so under Case B the feature builder uses **`availability_lag_1`** (prior week only; train median fill for cold start) — never contemporaneous `availability`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Download data

```bash
python scripts/download_usda_data.py
python scripts/download_diesel_data.py
```

By default the rates script fetches rows with **`date >= 2024-07-01`** (burn-in before 2025 walk-forward forecasts) via the Socrata SODA API, validates the schema, and writes:

- `data/raw/usda_refrigerated_truck_rates.parquet`
- `data/raw/usda_refrigerated_truck_rates.metadata.json` (includes `query_start_date` / `query_end_date`)

The diesel script defaults to **`date >= 2024-06-01`** (extra burn-in for the as-of lag) and writes:

- `data/raw/usda_diesel_weekly.parquet`
- `data/raw/usda_diesel_weekly.metadata.json`

Options:

```bash
python scripts/download_usda_data.py --start-date 2024-07-01
python scripts/download_usda_data.py --start-date 2000-01-01          # full history
python scripts/download_usda_data.py --start-date 2024-07-01 --end-date 2025-12-31
python scripts/download_diesel_data.py --region US
python scripts/download_diesel_data.py --start-date 2024-06-01
```
Raw and processed files are gitignored; regenerate them locally with the command above.

**Modeling window:** API/raw snapshot starts at mid-2024; walk-forward forecasts begin at the first week-ending Tuesday of **2025** (`2025-01-07`), expanding within the downloaded history only. Official eval is expanding-window walk-forward via `freight_rates.splits.iter_walk_forward` (train `date < t`, score `date == t`); fixed train/val/test cutoffs in that module are diagnostic only.

```bash
python scripts/train_walkforward_gbm.py
python scripts/train_walkforward_gbm.py --diesel none   # ablation variant
python scripts/ablate_diesel_walkforward.py             # none vs level vs full
```

**Notebooks (pipeline order):**

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | Raw USDA snapshot |
| 02 | `02_signal_exploration.ipynb` | Feature / diesel signal exploration |
| 03 | `03_base_dataset_and_splits.ipynb` | Lane-week panel + walk-forward splits |
| 04 | `04_nested_hpo_gbm.ipynb` | Nested HPO — tune θ* on val (`2025-01-07`→`2025-06-24`), test holdout `≥ 2025-07-01` |
| 05 | `05_walkforward_gbm.ipynb` | Final walk-forward results + visualizations (uses θ* from notebook 04) |

Walk-forward writes predictions and metric CSVs under `models/walkforward_gbm/` (gitignored). Baseline is prior-week `rpm_lag_1` (train global mean when a lane has no history).
**Dual-regime headline:** report (1) overall MAE vs lag-1 and (2) MAE lift on cold-start history bucket `0-4`. The residual GBM is not required to beat lag-1 on mature lanes; cold-start lift is the primary success criterion for the global model.

## Project layout

```
freight-rate/
├── data/                  # raw / processed snapshots (not committed)
├── models/                # walk-forward predictions / metrics (not committed)
├── notebooks/             # exploration notebooks
├── scripts/               # CLI entry points
├── src/freight_rates/     # reusable package
└── tests/                 # unit tests (API mocked)
```

## Development

```bash
pytest
ruff check src tests scripts
```
