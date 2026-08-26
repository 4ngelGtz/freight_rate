"""Preprocessing utilities for USDA refrigerated truck rate snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from freight_rates.diesel import attach_diesel_asof, load_diesel_snapshot
from freight_rates.ingestion import DEFAULT_RAW_DIR, add_lane_id, validate_schema

LANE_WEEK_KEY: Final[tuple[str, ...]] = ("date", "origin", "destination")

NUMERIC_RAW_COLUMNS: Final[tuple[str, ...]] = (
    "distance",
    "weeklow",
    "weekhigh",
    "midpoint",
    "rpm",
    "availability",
)

# Published rate bands — leakage if RPM is the target.
LEAKAGE_COLUMNS: Final[tuple[str, ...]] = ("weeklow", "weekhigh", "midpoint")


def coerce_raw_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric USDA columns coerced via ``errors='coerce'``."""
    out = df.copy()
    for col in NUMERIC_RAW_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_lane_week_panel(
    df: pd.DataFrame,
    *,
    validate: bool = True,
    drop_invalid_rpm: bool = True,
    drop_invalid_distance: bool = False,
) -> pd.DataFrame:
    """Collapse raw USDA rows to lane-week grain.

    One output row represents ``date + origin + destination``. When multiple raw
    rows share the same lane-week, ``rpm`` is aggregated with the **mean**. Groups
    with more than one distinct RPM value are flagged via ``rpm_conflict``.

    Parameters
    ----------
    df:
        Raw snapshot as returned by :func:`freight_rates.ingestion.load_raw_snapshot`.
    validate:
        When ``True``, require all :data:`EXPECTED_COLUMNS`.
    drop_invalid_rpm:
        Drop lane-week groups whose aggregated ``rpm`` is missing.
    drop_invalid_distance:
        Drop lane-week groups whose aggregated ``distance`` is missing or zero.

    Returns
    -------
    pandas.DataFrame
        Lane-week panel without leakage columns. Includes ``lane_id``,
        ``rpm_conflict``, and ``raw_rows_in_group`` audit fields.
    """
    if validate:
        validate_schema(df)

    work = coerce_raw_numeric(df)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    grouped = (
        work.groupby(list(LANE_WEEK_KEY), dropna=False)
        .agg(
            raw_rows_in_group=("commodity", "size"),
            commodity_count=("commodity", "nunique"),
            region_count=("region", "nunique"),
            region=("region", "first"),
            year=("year", "first"),
            week=("week", "first"),
            month=("month", "first"),
            quarter=("quarter", "first"),
            distance=("distance", "mean"),
            availability=("availability", "mean"),
            rpm_nunique=("rpm", "nunique"),
            rpm=("rpm", "mean"),
        )
        .reset_index()
    )

    grouped["rpm_conflict"] = grouped["rpm_nunique"] > 1
    grouped = add_lane_id(grouped)

    if drop_invalid_rpm:
        grouped = grouped[grouped["rpm"].notna()].copy()
    if drop_invalid_distance:
        grouped = grouped[grouped["distance"].notna() & (grouped["distance"] != 0)].copy()

    grouped = grouped.drop(columns=list(LEAKAGE_COLUMNS), errors="ignore")
    grouped = grouped.drop(columns=["rpm_nunique"], errors="ignore")

    sort_cols = ["date", "origin", "destination"]
    return grouped.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def build_modeling_panel(
    rates_raw: pd.DataFrame,
    diesel_raw: pd.DataFrame | None = None,
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    validate: bool = True,
    drop_invalid_rpm: bool = True,
    drop_invalid_distance: bool = False,
) -> pd.DataFrame:
    """Build the lane-week panel and attach Case-B-safe US diesel features.

    If ``diesel_raw`` is omitted, loads ``usda_diesel_weekly.parquet`` from
    ``raw_dir`` (see :func:`freight_rates.diesel.load_diesel_snapshot`).
    """
    panel = build_lane_week_panel(
        rates_raw,
        validate=validate,
        drop_invalid_rpm=drop_invalid_rpm,
        drop_invalid_distance=drop_invalid_distance,
    )
    diesel = diesel_raw if diesel_raw is not None else load_diesel_snapshot(raw_dir=raw_dir)
    return attach_diesel_asof(panel, diesel)
