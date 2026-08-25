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

This is the official weekly historical origin–destination series (rates, distance, availability, commodity) on USDA’s [Agricultural Transportation Open Data](https://agtransport.usda.gov/) platform. We prefer it over the quarterly aggregate O-D dataset ([`qm5q-5r5f`](https://agtransport.usda.gov/Truck/Quarterly-Refrigerated-Truck-Rates-by-Origin-Desti/qm5q-5r5f)) because it is more granular in time.

### Scope: 10 USDA destination markets

This project focuses on **weekly refrigerated truck rates from U.S. agricultural shipping areas to ten fixed wholesale destination markets** defined by USDA AMS methodology — not general freight to arbitrary U.S. cities.

USDA publishes rate data for **10 destination markets**, which are used to compute regional and national specialty-crop truck rates. See the [Agricultural Refrigerated Truck Quarterly Datasets](https://www.ams.usda.gov/services/transportation-analysis/agricultural-refrigerated-truck-quarterly-datasets) documentation:

> *"Rate data for **10 destination markets** are used to calculate average origin regional rates."*

Those destinations in our snapshot are: Atlanta, Baltimore, Boston, Chicago, Dallas, Los Angeles, Miami, New York, Philadelphia, and Seattle. **Origin districts (~107)** and **lanes (~800)** provide most of the cross-sectional variation; cold-start experiments target sparse or newly observed **lanes**, not new destination cities.

The curated modeling population is the **lane-week panel** (`date + origin + destination`), built from the raw snapshot via `build_lane_week_panel()` — one row per lane-week, with `rpm` aggregated by mean when reporting duplicates exist.

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
