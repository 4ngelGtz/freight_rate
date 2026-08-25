"""Ingestion of USDA Refrigerated Truck Rates via the AgTransport SODA API.

Selected dataset
----------------
Refrigerated Truck Rates and Availability (``acar-e3r8``)

Catalog: https://agtransport.usda.gov/Truck/Refrigerated-Truck-Rates-and-Availability/acar-e3r8
API:     https://agtransport.usda.gov/resource/acar-e3r8.json

Weekly historical origin–destination rates with distance, low/high/midpoint
truckload rates, rate per mile, availability, and commodity. Preferred over
the quarterly O-D aggregate (``qm5q-5r5f``) for temporal granularity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# --- Dataset constants -------------------------------------------------------

SOURCE_NAME = "Refrigerated Truck Rates and Availability"
DATASET_ID = "acar-e3r8"
BASE_URL = "https://agtransport.usda.gov"
API_ENDPOINT = f"{BASE_URL}/resource/{DATASET_ID}.json"
CATALOG_URL = f"{BASE_URL}/Truck/Refrigerated-Truck-Rates-and-Availability/{DATASET_ID}"

# API returns these field names (Socrata fieldName). Values are preserved as-is.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "date",
    "week",
    "month",
    "quarter",
    "year",
    "region",
    "origin",
    "destination",
    "distance",
    "commodity",
    "weeklow",
    "weekhigh",
    "midpoint",
    "rpm",
    "availability",
)

DEFAULT_PAGE_SIZE = 50_000
DEFAULT_TIMEOUT_SECONDS = 60.0

DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_PARQUET_NAME = "usda_refrigerated_truck_rates.parquet"
DEFAULT_METADATA_NAME = "usda_refrigerated_truck_rates.metadata.json"


class UsdaApiError(RuntimeError):
    """Raised when the USDA / Socrata API request fails."""


class SchemaValidationError(ValueError):
    """Raised when a DataFrame is missing required columns."""


# --- Public helpers ----------------------------------------------------------


def make_lane_id(origin: str, destination: str) -> str:
    """Derive a human-readable lane key for reporting / inspection only.

    Not intended as a modeling feature at this stage.
    """
    return f"{origin} -> {destination}"


def add_lane_id(df: pd.DataFrame, origin_col: str = "origin", dest_col: str = "destination") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``lane_id`` column for inspection."""
    if origin_col not in df.columns or dest_col not in df.columns:
        missing = [c for c in (origin_col, dest_col) if c not in df.columns]
        raise SchemaValidationError(f"Cannot create lane_id; missing columns: {missing}")
    out = df.copy()
    out["lane_id"] = [
        make_lane_id(str(o), str(d)) for o, d in zip(out[origin_col], out[dest_col], strict=True)
    ]
    return out


def validate_schema(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str] = EXPECTED_COLUMNS,
) -> None:
    """Ensure all expected columns are present. Does not coerce types."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")


# --- Fetch -------------------------------------------------------------------


def fetch_usda_data(
    *,
    endpoint: str = API_ENDPOINT,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Download records from the USDA SODA API with pagination.

    Parameters
    ----------
    endpoint:
        SODA resource URL.
    page_size:
        Page size for ``$limit`` / ``$offset`` pagination.
    timeout:
        Per-request timeout in seconds.
    session:
        Optional ``requests.Session`` (useful for tests / connection reuse).
    max_rows:
        Optional cap on total rows (for smoke tests). ``None`` fetches all.

    Returns
    -------
    pandas.DataFrame
        Raw API fields with values preserved (mostly object/string from JSON).
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    own_session = session is None
    http = session or requests.Session()
    records: list[dict[str, Any]] = []
    offset = 0

    try:
        while True:
            limit = page_size
            if max_rows is not None:
                remaining = max_rows - len(records)
                if remaining <= 0:
                    break
                limit = min(page_size, remaining)

            params = {"$limit": limit, "$offset": offset, "$order": ":id"}
            try:
                response = http.get(endpoint, params=params, timeout=timeout)
            except requests.Timeout as exc:
                raise UsdaApiError(
                    f"Request timed out after {timeout}s (offset={offset}): {endpoint}"
                ) from exc
            except requests.RequestException as exc:
                raise UsdaApiError(f"Request failed (offset={offset}): {exc}") from exc

            if not response.ok:
                raise UsdaApiError(
                    f"USDA API returned HTTP {response.status_code} at offset={offset}: "
                    f"{response.text[:300]}"
                )

            try:
                page = response.json()
            except ValueError as exc:
                raise UsdaApiError(f"Invalid JSON in API response at offset={offset}") from exc

            if not isinstance(page, list):
                raise UsdaApiError(
                    f"Expected a JSON list from SODA API, got {type(page).__name__}"
                )

            if not page:
                break

            records.extend(page)
            offset += len(page)

            if len(page) < limit:
                break
            if max_rows is not None and len(records) >= max_rows:
                break
    finally:
        if own_session:
            http.close()

    if not records:
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    # Preserve raw API values; do not coerce numerics/dates at ingestion time.
    return pd.DataFrame.from_records(records)


# --- Snapshot I/O ------------------------------------------------------------


def _infer_date_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if "date" not in df.columns or df.empty:
        return None, None
    dates = pd.to_datetime(df["date"], errors="coerce", utc=False)
    if dates.isna().all():
        return None, None
    min_ts = dates.min()
    max_ts = dates.max()
    return (
        min_ts.date().isoformat() if pd.notna(min_ts) else None,
        max_ts.date().isoformat() if pd.notna(max_ts) else None,
    )


def build_metadata(
    df: pd.DataFrame,
    *,
    source_name: str = SOURCE_NAME,
    source_url: str = CATALOG_URL,
    api_endpoint: str = API_ENDPOINT,
    dataset_id: str = DATASET_ID,
    download_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a metadata dict for the raw snapshot."""
    min_date, max_date = _infer_date_bounds(df)
    return {
        "source_name": source_name,
        "source_url": source_url,
        "api_endpoint": api_endpoint,
        "dataset_id": dataset_id,
        "download_timestamp_utc": download_timestamp_utc
        or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": int(len(df)),
        "column_names": list(df.columns),
        "min_date": min_date,
        "max_date": max_date,
    }


def save_raw_snapshot(
    df: pd.DataFrame,
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    parquet_name: str = DEFAULT_PARQUET_NAME,
    metadata_name: str = DEFAULT_METADATA_NAME,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write Parquet snapshot and JSON metadata under ``raw_dir``."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    parquet_path = raw_path / parquet_name
    metadata_path = raw_path / metadata_name

    df.to_parquet(parquet_path, index=False)

    meta = metadata if metadata is not None else build_metadata(df)
    metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return parquet_path, metadata_path


def load_raw_snapshot(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    parquet_name: str = DEFAULT_PARQUET_NAME,
) -> pd.DataFrame:
    """Load a previously saved Parquet snapshot."""
    path = Path(raw_dir) / parquet_name
    if not path.exists():
        raise FileNotFoundError(
            f"Raw snapshot not found: {path}. Run scripts/download_usda_data.py first."
        )
    return pd.read_parquet(path)


def load_raw_metadata(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    metadata_name: str = DEFAULT_METADATA_NAME,
) -> dict[str, Any]:
    """Load snapshot metadata JSON."""
    path = Path(raw_dir) / metadata_name
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def download_and_snapshot(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    """Fetch, validate, and persist a full (or capped) raw snapshot.

    Returns ``(dataframe, metadata, parquet_path, metadata_path)``.
    """
    df = fetch_usda_data(
        page_size=page_size,
        timeout=timeout,
        session=session,
        max_rows=max_rows,
    )
    validate_schema(df)
    metadata = build_metadata(df)
    parquet_path, metadata_path = save_raw_snapshot(df, raw_dir=raw_dir, metadata=metadata)
    return df, metadata, parquet_path, metadata_path
