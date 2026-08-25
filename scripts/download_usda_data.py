#!/usr/bin/env python3
"""Download USDA refrigerated truck rates into a local Parquet snapshot.

Usage (from repository root)::

    python scripts/download_usda_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install when developing from a checkout.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freight_rates.ingestion import (  # noqa: E402
    DEFAULT_RAW_DIR,
    add_lane_id,
    download_and_snapshot,
)


def _summarize(df) -> None:
    n_rows, n_cols = df.shape
    print(f"rows:          {n_rows:,}")
    print(f"columns:       {n_cols}")

    if "date" in df.columns and not df.empty:
        dates = df["date"].astype(str)
        print(f"date range:    {dates.min()} → {dates.max()}")
    else:
        print("date range:    n/a")

    if "origin" in df.columns:
        print(f"origins:       {df['origin'].nunique():,}")
    if "destination" in df.columns:
        print(f"destinations:  {df['destination'].nunique():,}")

    if "origin" in df.columns and "destination" in df.columns:
        lanes = add_lane_id(df)
        print(f"unique lanes:  {lanes['lane_id'].nunique():,}")


def main() -> int:
    raw_dir = _ROOT / DEFAULT_RAW_DIR
    print("Fetching USDA Refrigerated Truck Rates (acar-e3r8)...")
    df, metadata, parquet_path, metadata_path = download_and_snapshot(raw_dir=raw_dir)

    print()
    print("Download complete.")
    print(f"parquet:       {parquet_path}")
    print(f"metadata:      {metadata_path}")
    print(f"dataset_id:    {metadata['dataset_id']}")
    print(f"downloaded_at: {metadata['download_timestamp_utc']}")
    print()
    _summarize(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
