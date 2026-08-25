"""Unit tests for USDA ingestion helpers (no live API downloads)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freight_rates.ingestion import (
    EXPECTED_COLUMNS,
    SchemaValidationError,
    UsdaApiError,
    add_lane_id,
    build_metadata,
    fetch_usda_data,
    make_lane_id,
    save_raw_snapshot,
    validate_schema,
)


def _sample_records(n: int = 2) -> list[dict]:
    base = {
        "date": "2000-01-04T00:00:00.000",
        "week": "1",
        "month": "1",
        "quarter": "1",
        "year": "2000",
        "region": "ARIZONA",
        "origin": "ARIZONA ORIGIN",
        "destination": "ATLANTA",
        "distance": "2100",
        "commodity": "LETTUCE",
        "weeklow": "2100",
        "weekhigh": "2400",
        "midpoint": "2250",
        "rpm": "1.07",
        "availability": "3",
    }
    records = []
    for i in range(n):
        row = dict(base)
        row["destination"] = ["ATLANTA", "CHICAGO"][i % 2]
        row["date"] = f"2000-01-0{4 + i}T00:00:00.000"
        records.append(row)
    return records


def test_validate_schema_ok() -> None:
    df = pd.DataFrame(_sample_records())
    validate_schema(df)  # does not raise


def test_validate_schema_missing_columns() -> None:
    df = pd.DataFrame({"date": ["2000-01-01"], "origin": ["A"]})
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_schema(df)
    assert "Missing required columns" in str(exc_info.value)
    assert "destination" in str(exc_info.value)


def test_make_lane_id() -> None:
    assert make_lane_id("Arizona", "Atlanta") == "Arizona -> Atlanta"


def test_add_lane_id() -> None:
    df = pd.DataFrame({"origin": ["A", "B"], "destination": ["X", "Y"]})
    out = add_lane_id(df)
    assert list(out["lane_id"]) == ["A -> X", "B -> Y"]
    # original unchanged
    assert "lane_id" not in df.columns


def test_add_lane_id_missing_columns() -> None:
    df = pd.DataFrame({"origin": ["A"]})
    with pytest.raises(SchemaValidationError):
        add_lane_id(df)


def test_build_metadata() -> None:
    df = pd.DataFrame(_sample_records())
    meta = build_metadata(
        df,
        download_timestamp_utc="2026-01-01T00:00:00Z",
    )
    assert meta["source_name"] == "Refrigerated Truck Rates and Availability"
    assert meta["dataset_id"] == "acar-e3r8"
    assert meta["row_count"] == 2
    assert meta["column_names"] == list(df.columns)
    assert meta["download_timestamp_utc"] == "2026-01-01T00:00:00Z"
    assert meta["min_date"] == "2000-01-04"
    assert meta["max_date"] == "2000-01-05"
    assert "api_endpoint" in meta
    assert "source_url" in meta


def test_save_raw_snapshot(tmp_path: Path) -> None:
    df = pd.DataFrame(_sample_records())
    meta = build_metadata(df, download_timestamp_utc="2026-01-01T00:00:00Z")
    parquet_path, metadata_path = save_raw_snapshot(df, raw_dir=tmp_path, metadata=meta)

    assert parquet_path.exists()
    assert metadata_path.exists()
    loaded = pd.read_parquet(parquet_path)
    assert len(loaded) == 2
    assert set(EXPECTED_COLUMNS).issubset(loaded.columns)

    saved_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_meta["row_count"] == 2
    assert saved_meta["dataset_id"] == "acar-e3r8"


def test_fetch_usda_data_paginates() -> None:
    # Two full pages then an empty page (SODA end-of-data signal).
    page1 = _sample_records(2)
    page2 = [
        {
            **_sample_records(1)[0],
            "destination": "BOSTON",
            "date": "2000-01-06T00:00:00.000",
        },
        {
            **_sample_records(1)[0],
            "destination": "MIAMI",
            "date": "2000-01-07T00:00:00.000",
        },
    ]

    responses = []
    for payload in (page1, page2, []):
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = payload
        responses.append(resp)

    session = MagicMock()
    session.get.side_effect = responses

    df = fetch_usda_data(session=session, page_size=2, timeout=5.0)
    assert len(df) == 4
    assert session.get.call_count == 3
    assert session.get.call_args_list[0].kwargs["params"]["$offset"] == 0
    assert session.get.call_args_list[1].kwargs["params"]["$offset"] == 2
    assert session.get.call_args_list[2].kwargs["params"]["$offset"] == 4


def test_fetch_usda_data_stops_on_short_page() -> None:
    page1 = _sample_records(2)
    page2 = _sample_records(1)  # fewer than page_size → end

    responses = []
    for payload in (page1, page2):
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = payload
        responses.append(resp)

    session = MagicMock()
    session.get.side_effect = responses

    df = fetch_usda_data(session=session, page_size=2, timeout=5.0)
    assert len(df) == 3
    assert session.get.call_count == 2


def test_fetch_usda_data_http_error() -> None:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = 500
    resp.text = "Internal Server Error"

    session = MagicMock()
    session.get.return_value = resp

    with pytest.raises(UsdaApiError) as exc_info:
        fetch_usda_data(session=session, page_size=10)
    assert "HTTP 500" in str(exc_info.value)
