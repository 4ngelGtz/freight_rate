"""Unit tests for diesel ingestion and Case-B as-of merge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from freight_rates.diesel import (
    DIESEL_ASOF_LAG_DAYS,
    EXPECTED_COLUMNS,
    attach_diesel_asof,
    fetch_diesel_data,
    prepare_us_diesel_series,
)
from freight_rates.ingestion import SchemaValidationError


def _diesel_row(
    *,
    date: str,
    price: str,
    region: str = "US",
    week: str = "1",
) -> dict:
    ts = pd.Timestamp(date)
    return {
        "date": f"{date}T00:00:00.000",
        "week": week,
        "month": str(ts.month),
        "year": str(ts.year),
        "region": region,
        "diesel_price": price,
    }


def test_prepare_us_diesel_series_filters_region_and_diff() -> None:
    raw = pd.DataFrame(
        [
            _diesel_row(date="2024-12-23", price="3.50", region="US"),
            _diesel_row(date="2024-12-30", price="3.60", region="US"),
            _diesel_row(date="2024-12-30", price="4.00", region="Midwest"),
            _diesel_row(date="2025-01-06", price="3.70", region="US"),
        ]
    )
    series = prepare_us_diesel_series(raw)
    assert list(series["diesel_date"]) == [
        pd.Timestamp("2024-12-23"),
        pd.Timestamp("2024-12-30"),
        pd.Timestamp("2025-01-06"),
    ]
    assert series.loc[1, "diesel_us"] == pytest.approx(3.60)
    assert pd.isna(series.loc[0, "diesel_us_chg_1w"])
    assert series.loc[1, "diesel_us_chg_1w"] == pytest.approx(0.10)
    assert series.loc[2, "diesel_us_chg_1w"] == pytest.approx(0.10)


def test_attach_diesel_asof_uses_prior_week_tuesday_cutoff() -> None:
    """Rate Tuesday t may only see diesel Mondays <= t-7 (Case B)."""
    diesel = pd.DataFrame(
        [
            _diesel_row(date="2024-12-23", price="3.50"),  # Monday
            _diesel_row(date="2024-12-30", price="3.60"),  # Monday
            _diesel_row(date="2025-01-06", price="3.70"),  # Monday before rate Tue
        ]
    )
    # Rate week-ending Tuesday 2025-01-07 → asof key 2024-12-31 → diesel 2024-12-30
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-07"), pd.Timestamp("2025-01-14")],
            "origin": ["A", "A"],
            "destination": ["ATLANTA", "ATLANTA"],
            "rpm": [2.0, 2.1],
        }
    )
    out = attach_diesel_asof(panel, diesel)
    assert DIESEL_ASOF_LAG_DAYS == 7

    row0 = out.loc[out["date"] == pd.Timestamp("2025-01-07")].iloc[0]
    assert row0["diesel_date"] == pd.Timestamp("2024-12-30")
    assert row0["diesel_us"] == pytest.approx(3.60)
    assert row0["diesel_us_chg_1w"] == pytest.approx(0.10)

    # 2025-01-14 → asof 2025-01-07 → latest Monday <= that is 2025-01-06
    row1 = out.loc[out["date"] == pd.Timestamp("2025-01-14")].iloc[0]
    assert row1["diesel_date"] == pd.Timestamp("2025-01-06")
    assert row1["diesel_us"] == pytest.approx(3.70)


def test_attach_diesel_asof_excludes_same_week_monday() -> None:
    """Monday t-1 must not attach to rate Tuesday t under the 7-day lag."""
    diesel = pd.DataFrame(
        [
            _diesel_row(date="2024-12-30", price="3.60"),
            _diesel_row(date="2025-01-06", price="9.99"),  # would leak if asof < t
        ]
    )
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-07")],
            "origin": ["A"],
            "destination": ["ATLANTA"],
            "rpm": [2.0],
        }
    )
    out = attach_diesel_asof(panel, diesel)
    assert out.loc[0, "diesel_date"] == pd.Timestamp("2024-12-30")
    assert out.loc[0, "diesel_us"] == pytest.approx(3.60)
    assert out.loc[0, "diesel_us"] != pytest.approx(9.99)


def test_fetch_diesel_data_applies_region_and_start_date() -> None:
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = []
    session = MagicMock()
    session.get.return_value = resp

    fetch_diesel_data(session=session, page_size=10, start_date="2024-06-01", region="US")
    params = session.get.call_args.kwargs["params"]
    assert "date >= '2024-06-01T00:00:00.000'" in params["$where"]
    assert "region = 'US'" in params["$where"]


def test_fetch_diesel_data_empty_schema() -> None:
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = []
    session = MagicMock()
    session.get.return_value = resp

    df = fetch_diesel_data(session=session, page_size=10, start_date=None)
    assert list(df.columns) == list(EXPECTED_COLUMNS)
    assert df.empty


def test_prepare_us_diesel_requires_columns() -> None:
    with pytest.raises(SchemaValidationError):
        prepare_us_diesel_series(pd.DataFrame({"date": ["2024-01-01"]}))
