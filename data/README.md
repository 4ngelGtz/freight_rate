# Data

## Official source

**Dataset:** Refrigerated Truck Rates and Availability  
**Platform:** [USDA Agricultural Transportation Open Data](https://agtransport.usda.gov/)  
**Dataset ID:** `acar-e3r8`  
**Catalog page:** https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8  
**API endpoint:** `https://agtransport.usda.gov/resource/acar-e3r8.json`

Weekly historical refrigerated truck rates and availability by origin, destination, and commodity (AMS Specialty Crops Market News), with region assigned by AMS Transportation Services Division.

### Why this dataset

| Dataset | ID | Granularity | Notes |
|---------|----|-------------|-------|
| **Refrigerated Truck Rates and Availability** | `acar-e3r8` | **Weekly**, origin–destination | Selected: most granular historical O-D series |
| Latest week only | `25pi-t6xr` | Single week | Not useful for history |
| Average weekly regional rates by distance | `c69n-pfv3` | Weekly, regional | No lane-level O-D |
| Quarterly rates by O-D pair | `qm5q-5r5f` | Quarterly | Aggregate; avoided unless weekly is unavailable |

## Regenerating data

From the repository root (with the package installed):

```bash
python scripts/download_usda_data.py
```

This writes:

- `raw/usda_refrigerated_truck_rates.parquet`
- `raw/usda_refrigerated_truck_rates.metadata.json`

## Git policy

`data/raw/` and `data/processed/` snapshots are **intentionally not committed**. They can be large and are reproducible from the public USDA API. Only this README (and empty `.gitkeep` placeholders) is tracked under `data/`.
