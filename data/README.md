# Data

## Official source

**Dataset:** Refrigerated Truck Rates and Availability  
**Platform:** [USDA Agricultural Transportation Open Data](https://agtransport.usda.gov/)  
**Dataset ID:** `acar-e3r8`  
**Catalog:** https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8  
**API:** `https://agtransport.usda.gov/resource/acar-e3r8.json`

Weekly refrigerated truck rates by origin, destination, and commodity (AMS FVWTRK).

**Diesel:** [Weekly On-Highway Diesel](https://agtransport.usda.gov/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/x88w-atzp) (`x88w-atzp`), merged Case-B as-of in the modeling panel.

## Download

From repo root (package installed):

```bash
make download              # rates + diesel (pipeline step 1)
make download-rates        # USDA rates only
make download-diesel       # EIA diesel only
```

Or directly:

```bash
python scripts/download_usda_data.py    # default date >= 2024-07-01
python scripts/download_diesel_data.py  # default date >= 2024-06-01
```

Writes under `data/raw/`:

| File | Content |
|---|---|
| `usda_refrigerated_truck_rates.parquet` | Lane-level weekly rates |
| `usda_refrigerated_truck_rates.metadata.json` | Query bounds, row count |
| `usda_diesel_weekly.parquet` | US weekly on-highway diesel |
| `usda_diesel_weekly.metadata.json` | Query bounds |

## Git policy

`data/raw/` and `data/processed/` are not committed. Regenerate with `make download`.

## Operational scoring

`make score` uses the latest week-ending Tuesday in `data/raw/` as **last observed**, then forecasts the **next** Tuesday (e.g. data through `2026-08-18` → predict `2026-08-25`). Lane scaffolds for the forecast week copy origin/destination/distance from the last observed week.
