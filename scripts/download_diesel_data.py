#!/usr/bin/env python3
"""Download USDA/EIA weekly on-highway diesel prices into a local Parquet snapshot.

Usage (from repository root)::

    python scripts/download_diesel_data.py
    python scripts/download_diesel_data.py --start-date 2024-06-01
    python scripts/download_diesel_data.py --region US
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freight_rates.diesel import (  # noqa: E402
    DATASET_ID,
    DEFAULT_START_DATE,
    download_and_snapshot,
)
from freight_rates.ingestion import DEFAULT_RAW_DIR  # noqa: E402


def _summarize(df: pd.DataFrame) -> None:
    print(f"rows:          {len(df):,}")
    print(f"columns:       {list(df.columns)}")
    if "date" in df.columns and not df.empty:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print(f"date range:    {dates.min().date()} → {dates.max().date()}")
    if "region" in df.columns:
        print(f"regions:       {sorted(df['region'].astype(str).unique())}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download weekly on-highway diesel prices (x88w-atzp) to data/raw/."
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=(
            "Inclusive lower bound on diesel week-ending Monday (YYYY-MM-DD). "
            f"Default: {DEFAULT_START_DATE} (burn-in before rate as-of merge)."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive upper bound on diesel date (YYYY-MM-DD). Default: no upper bound.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Optional exact region filter (e.g. US). Default: download all regions.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap (smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    raw_dir = _ROOT / DEFAULT_RAW_DIR

    print("Fetching Weekly On-Highway Diesel Fuel Prices (x88w-atzp)...")
    print(f"start_date:    {args.start_date}")
    print(f"end_date:      {args.end_date or '(none)'}")
    print(f"region:        {args.region or '(all)'}")

    df, metadata, parquet_path, metadata_path = download_and_snapshot(
        raw_dir=raw_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        region=args.region,
        max_rows=args.max_rows,
    )

    print()
    print("Download complete.")
    print(f"parquet:       {parquet_path}")
    print(f"metadata:      {metadata_path}")
    print(f"dataset_id:    {metadata['dataset_id']}")
    print(f"downloaded_at: {metadata['download_timestamp_utc']}")
    print(f"query_start:   {metadata.get('query_start_date')}")
    print(f"query_end:     {metadata.get('query_end_date')}")
    print()
    _summarize(df)
    assert metadata["dataset_id"] == DATASET_ID
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
