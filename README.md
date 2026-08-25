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

This is the official weekly historical origin–destination series (rates, distance, availability, commodity) on USDA’s Agricultural Transportation Open Data platform. We prefer it over the quarterly aggregate O-D dataset (`qm5q-5r5f`) because it is more granular in time.

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
