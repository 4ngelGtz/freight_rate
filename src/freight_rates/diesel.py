"""Weekly on-highway diesel prices (USDA AgTransport mirror of EIA).

Selected dataset
----------------
Weekly On-Highway Diesel Fuel Prices (``x88w-atzp``)

Catalog: https://agtransport.usda.gov/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/x88w-atzp
API:     https://agtransport.usda.gov/resource/x88w-atzp.json

EIA retail on-highway diesel ($/gallon), week-ending **Monday**, by PADD region
and US. Used as an exogenous cost driver for refrigerated truck rates.

Leakage policy (Case B)
-----------------------
Rate panel ``date`` is week-ending **Tuesday**. Forecasts for Tuesday ``t`` may
only use information available through the prior week-ending Tuesday ``t-7``.

Diesel Mondays are therefore attached with an as-of merge on
``rate_date - 7 days`` (latest diesel Monday ``<= t-7``). Same-week Monday
diesel (``t-1``) is **not** used — it is published after the information cutoff.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from freight_rates.ingestion import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_RAW_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    SchemaValidationError,
    build_metadata,
    fetch_soda_resource,
    save_raw_snapshot,
    validate_schema,
)

# --- Dataset constants -------------------------------------------------------

SOURCE_NAME = "Weekly On-Highway Diesel Fuel Prices"
DATASET_ID = "x88w-atzp"
BASE_URL = "https://agtransport.usda.gov"
API_ENDPOINT = f"{BASE_URL}/resource/{DATASET_ID}.json"
CATALOG_URL = f"{BASE_URL}/Fuel/Weekly-On-Highway-Diesel-Fuel-Prices/{DATASET_ID}"

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "week",
    "month",
    "year",
    "region",
    "diesel_price",
)

DEFAULT_PARQUET_NAME = "usda_diesel_weekly.parquet"
DEFAULT_METADATA_NAME = "usda_diesel_weekly.metadata.json"

# One month of burn-in before the rate snapshot default (as-of lag needs prior Mondays).
DEFAULT_START_DATE = "2024-06-01"

# National series used for the first global diesel features.
DEFAULT_REGION = "US"

# Case B: only diesel known by the prior week-ending Tuesday.
DIESEL_ASOF_LAG_DAYS: Final[int] = 7

DIESEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "diesel_us",
    "diesel_us_chg_1w",
)


def fetch_diesel_data(
    *,
    endpoint: str = API_ENDPOINT,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    max_rows: int | None = None,
    start_date: str | pd.Timestamp | datetime | None = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp | datetime | None = None,
    region: str | None = None,
) -> pd.DataFrame:
    """Download weekly diesel rows from the USDA SODA API.

    Parameters
    ----------
    region:
        Optional exact ``region`` filter (e.g. ``\"US\"``). ``None`` fetches all
        PADD / US rows.
    """
    extra_where = f"region = '{region}'" if region is not None else None
    return fetch_soda_resource(
        endpoint=endpoint,
        expected_columns=EXPECTED_COLUMNS,
        page_size=page_size,
        timeout=timeout,
        session=session,
        max_rows=max_rows,
        start_date=start_date,
        end_date=end_date,
        extra_where=extra_where,
    )


def download_and_snapshot(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    max_rows: int | None = None,
    start_date: str | pd.Timestamp | datetime | None = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp | datetime | None = None,
    region: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    """Fetch, validate, and persist a raw diesel snapshot.

    Returns ``(dataframe, metadata, parquet_path, metadata_path)``.
    """
    df = fetch_diesel_data(
        page_size=page_size,
        timeout=timeout,
        session=session,
        max_rows=max_rows,
        start_date=start_date,
        end_date=end_date,
        region=region,
    )
    validate_schema(df, required_columns=EXPECTED_COLUMNS)
    metadata = build_metadata(
        df,
        source_name=SOURCE_NAME,
        source_url=CATALOG_URL,
        api_endpoint=API_ENDPOINT,
        dataset_id=DATASET_ID,
        query_start_date=start_date,
        query_end_date=end_date,
    )
    parquet_path, metadata_path = save_raw_snapshot(
        df,
        raw_dir=raw_dir,
        parquet_name=DEFAULT_PARQUET_NAME,
        metadata_name=DEFAULT_METADATA_NAME,
        metadata=metadata,
    )
    return df, metadata, parquet_path, metadata_path


def load_diesel_snapshot(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    parquet_name: str = DEFAULT_PARQUET_NAME,
) -> pd.DataFrame:
    """Load a previously saved diesel Parquet snapshot."""
    path = Path(raw_dir) / parquet_name
    if not path.exists():
        raise FileNotFoundError(
            f"Diesel snapshot not found: {path}. Run scripts/download_diesel_data.py first."
        )
    return pd.read_parquet(path)


def prepare_us_diesel_series(
    diesel_raw: pd.DataFrame,
    *,
    region: str = DEFAULT_REGION,
) -> pd.DataFrame:
    """Collapse raw diesel rows to a sorted US (or one-region) weekly series.

    Returns columns: ``diesel_date``, ``diesel_us``, ``diesel_us_chg_1w``.
    ``diesel_us_chg_1w`` is the week-over-week change on the diesel calendar
    (prior Monday → this Monday), then carried through the as-of merge.
    """
    if "region" not in diesel_raw.columns or "diesel_price" not in diesel_raw.columns:
        raise SchemaValidationError("Diesel frame must include 'region' and 'diesel_price' columns")

    work = diesel_raw.loc[diesel_raw["region"].astype(str) == region].copy()
    if work.empty:
        raise ValueError(f"No diesel rows for region={region!r}")

    work["diesel_date"] = pd.to_datetime(work["date"], errors="coerce")
    work["diesel_us"] = pd.to_numeric(work["diesel_price"], errors="coerce")
    work = work.dropna(subset=["diesel_date", "diesel_us"])
    work = (
        work.groupby("diesel_date", as_index=False)["diesel_us"]
        .mean()
        .sort_values("diesel_date", kind="mergesort")
        .reset_index(drop=True)
    )
    work["diesel_us_chg_1w"] = work["diesel_us"].diff()
    return work


def attach_diesel_asof(
    panel: pd.DataFrame,
    diesel_raw: pd.DataFrame,
    *,
    region: str = DEFAULT_REGION,
    asof_lag_days: int = DIESEL_ASOF_LAG_DAYS,
) -> pd.DataFrame:
    """Attach Case-B-safe US diesel features to a lane-week rate panel.

    For each rate week-ending Tuesday ``t``, joins the latest diesel Monday with
    ``diesel_date <= t - asof_lag_days`` (default 7 → prior week-ending Tuesday).

    Adds ``diesel_us``, ``diesel_us_chg_1w``, and audit column ``diesel_date``.
    """
    if "date" not in panel.columns:
        raise SchemaValidationError("Lane-week panel must include a 'date' column")
    if asof_lag_days < 0:
        raise ValueError("asof_lag_days must be >= 0")

    diesel = prepare_us_diesel_series(diesel_raw, region=region)
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["_diesel_asof"] = out["date"] - pd.Timedelta(days=asof_lag_days)

    # Drop any pre-existing diesel columns to keep the merge idempotent.
    out = out.drop(columns=["diesel_us", "diesel_us_chg_1w", "diesel_date"], errors="ignore")

    left = out.sort_values("_diesel_asof", kind="mergesort")
    right = diesel.sort_values("diesel_date", kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on="_diesel_asof",
        right_on="diesel_date",
        direction="backward",
    )
    # Restore original row order.
    merged = merged.sort_index(kind="mergesort")
    merged = merged.drop(columns=["_diesel_asof"])
    return merged
