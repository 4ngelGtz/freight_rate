# Data

## Official sources

**Rates:** [Refrigerated Truck Rates and Availability](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8) (`acar-e3r8`)  
**Diesel:** [Weekly On-Highway Diesel](https://agtransport.usda.gov/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/x88w-atzp) (`x88w-atzp`, EIA via AgTransport)

Weekly refrigerated truck rates by origin, destination, and commodity (AMS FVWTRK). Diesel is merged into the modeling panel with a Case-B as-of rule (see main [README](../README.md)).

---

## Week-ending Tuesday

USDA labels each rate report with a **week-ending Tuesday** in the `date` column.

Example: `date = 2026-08-18` covers roughly **Aug 12–18, 2026**.

All splits, walk-forward folds, and operational scores use this Tuesday index — not calendar months or ISO weeks in isolation.

---

## Download

```bash
make download              # rates + diesel
make download-rates        # USDA rates only
make download-diesel       # EIA diesel only
```

Or directly:

```bash
python scripts/download_usda_data.py    # default date >= 2024-07-01
python scripts/download_diesel_data.py  # default date >= 2024-06-01
```

Writes under `data/raw/` (gitignored):

| File | Content |
|---|---|
| `usda_refrigerated_truck_rates.parquet` | Raw USDA rate rows |
| `usda_refrigerated_truck_rates.metadata.json` | Query bounds, row count |
| `usda_diesel_weekly.parquet` | US weekly on-highway diesel |
| `usda_diesel_weekly.metadata.json` | Query bounds |

---

## Raw rate columns (`usda_refrigerated_truck_rates.parquet`)

| Column | Description |
|---|---|
| `date` | Week-ending Tuesday |
| `origin`, `destination` | Lane endpoints (USDA wholesale markets) |
| `region` | Origin region |
| `distance` | Miles |
| `commodity` | Produce / product category |
| `rpm` | Rate per mile ($/mile) — **model target** |
| `availability` | Truck availability indicator (same-week — do not use as feature) |
| `weeklow`, `weekhigh`, `midpoint` | Rate band stats — **leakage if used as features** |
| `year`, `month`, `week`, `quarter` | Calendar fields from USDA |

Multiple commodities per lane-week are collapsed to **one row per `(date, origin, destination)`** with `rpm = mean` in preprocessing.

---

## Raw diesel columns (`usda_diesel_weekly.parquet`)

| Column | Description |
|---|---|
| `date` | Week-ending **Monday** (diesel calendar) |
| `region` | `US` (national) or other |
| `diesel_price` | On-highway retail diesel ($/gallon) |

Merged into the rate panel as `diesel_us`, `diesel_us_chg_1w` using as-of join: for rate Tuesday `t`, use latest diesel Monday with `date <= t − 7`.

---

## Modeling panel (after `build_modeling_panel()`)

Built in code — not stored as a separate file by default.

| Column | Description |
|---|---|
| `date`, `origin`, `destination`, `lane_id` | Lane-week key |
| `rpm` | Target ($/mile) |
| `distance`, `region` | Lane attributes |
| `availability` | Raw same-week (features use `availability_lag_1` instead) |
| `diesel_us`, `diesel_us_chg_1w`, `diesel_date` | Case-B diesel features |
| `lane_history_n` | Derived at feature time — prior weeks for lane |

---

## Operational scoring

`make score` treats the **latest Tuesday in `data/raw/`** as `last_observed_date`, then forecasts **`last_observed + 7 days`**.

Lane rows for the forecast week are **scaffolded** from the last observed week (same OD pairs and distance; `rpm` missing until USDA publishes).

Example: data through **2026-08-18** → predict **2026-08-25**.

See [`models/README.md`](../models/README.md) for prediction output schema.

---

## Git policy

`data/raw/` and `data/processed/` are not committed. Regenerate with `make download`.

---

## Leakage checklist

| Rule | Reason |
|---|---|
| Do not feature `weeklow` / `weekhigh` / `midpoint` | Published with same-week rate bands |
| Use `availability_lag_1`, not same-week `availability` | Case B — availability co-published with `rpm` |
| Diesel as-of `<= t − 7` | Diesel is Monday; rate is Tuesday; no same-week peek |

More context: notebook `03_base_dataset_and_splits.ipynb`, notebook `02_signal_exploration.ipynb` (diesel section).
