# Freight Rate Cold Start

Study lane-level freight-rate prediction and cold-start behavior using the USDA Refrigerated Truck Rates dataset.

## Objective

Build a reproducible pipeline to predict refrigerated truck rates at the origin–destination lane level, with a focus on sparse and newly observed lanes (cold start).

## Planned model comparison

1. **Global Gradient Boosting** — a single model across all lanes
2. **Hierarchical Bayesian model** — partial pooling for sparse / new lanes

Modeling libraries are intentionally not included yet; this repository currently covers project layout and data ingestion only.

## Data pipeline

```
USDA AgTransport API
  → raw Parquet snapshot (data/raw/)
  → processed dataset (data/processed/)
  → feature engineering
  → modeling
```

**Selected source:** [Refrigerated Truck Rates and Availability](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8) (`acar-e3r8`).

This is the official **weekly** historical origin–destination series (rates, distance, availability, commodity) on USDA’s [Agricultural Transportation Open Data](https://agtransport.usda.gov/) platform, sourced from the AMS Specialty Crops [Fruit and Vegetable Truck Rate Report](https://www.ams.usda.gov/mnreports/fvwtrk.pdf) (FVWTRK).

### Scope: 10 USDA destination markets

This project uses **only** the weekly `acar-e3r8` series: refrigerated truck spot rates from U.S. agricultural shipping areas to **ten fixed wholesale destination markets** — not general freight to arbitrary U.S. cities.

Per the weekly FVWTRK methodology, rates are quoted to: Atlanta, Baltimore, Boston, Chicago, Dallas, Los Angeles, Miami, New York, Philadelphia, and Seattle. **Origin districts (~107)** and **lanes (~800)** provide most of the cross-sectional variation; cold-start experiments target sparse or newly observed **lanes**, not new destination cities.

The curated modeling population is the **lane-week panel** (`date + origin + destination`), built from the raw snapshot via `build_lane_week_panel()` — one row per lane-week, with `rpm` aggregated by mean when reporting duplicates exist.

### Field glossary

Definitions follow the weekly AMS Specialty Crops Market News [Fruit and Vegetable Truck Rate Report](https://www.ams.usda.gov/mnreports/fvwtrk.pdf) and the [`acar-e3r8`](https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8) catalog.

| Field | Meaning |
|---|---|
| `date` | Report-week reference date (usually Tuesday) for the weekly spot quotes — not an individual shipment timestamp |
| `origin` | Agricultural shipping district / producing area |
| `destination` | One of the 10 wholesale receiving cities listed above |
| `distance` | Typical haul miles from origin area to destination city |
| `commodity` | Commodity (or commodity group) for which the rate was reported |
| `weeklow` / `weekhigh` | Spot truckload rate range for the week ($/load) |
| `midpoint` | Midpoint of the weekly rate band (used with distance to derive RPM) |
| `rpm` | Rate per mile ($/mile) — modeling target; reconstructs as `midpoint / distance` |
| `availability` | Spot refrigerated-truck availability at origin (ordinal 1–5; see below) |
| `region` | AMS Transportation Services regional assignment |

**`availability` scale** (labels as published in FVWTRK; numeric codes as stored in `acar-e3r8`):

| Code | Label |
|---:|---|
| 1 | Surplus |
| 2 | Slight surplus |
| 3 | Adequate |
| 4 | Slight shortage |
| 5 | Shortage |

Rates are open (**spot**) market truckload quotes including broker fees for single-destination loads in 48–53 ft refrigerated trailers. For modeling, treat `weeklow`, `weekhigh`, and `midpoint` as **target leakage** if `rpm` is the outcome; `distance` and `availability` are legitimate pre-shipment features when known at prediction time. Same-week `availability` is published with the same weekly report as `rpm`, so realistic (Case B) prediction should use prior-week information only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Download data

```bash
python scripts/download_usda_data.py
```

This fetches the full dataset via the Socrata SODA API, validates the schema, and writes:

- `data/raw/usda_refrigerated_truck_rates.parquet`
- `data/raw/usda_refrigerated_truck_rates.metadata.json`

Raw and processed files are gitignored; regenerate them locally with the command above.

## Project layout

```
freight-rate/
├── data/                  # raw / processed snapshots (not committed)
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
