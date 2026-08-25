"""Unit tests for lane-week preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from freight_rates.ingestion import EXPECTED_COLUMNS, SchemaValidationError
from freight_rates.preprocessing import (
    LANE_WEEK_KEY,
    LEAKAGE_COLUMNS,
    build_lane_week_panel,
    coerce_raw_numeric,
)


def _raw_row(**overrides) -> dict:
    base = {
        "date": "2000-01-04T00:00:00.000",
        "week": "1",
        "month": "1",
        "quarter": "1",
        "year": "2000",
        "region": "ARIZONA",
        "origin": "ORIGIN A",
        "destination": "ATLANTA",
        "distance": "1000",
        "commodity": "LETTUCE",
        "weeklow": "2100",
        "weekhigh": "2400",
        "midpoint": "2250",
        "rpm": "2.0",
        "availability": "3",
    }
    base.update(overrides)
    return base


def test_coerce_raw_numeric() -> None:
    df = pd.DataFrame([{"rpm": "1.5", "distance": "bad"}])
    out = coerce_raw_numeric(df)
    assert out.loc[0, "rpm"] == pytest.approx(1.5)
    assert pd.isna(out.loc[0, "distance"])


def test_build_lane_week_panel_collapses_with_mean_rpm() -> None:
    df = pd.DataFrame(
        [
            _raw_row(rpm="2.0", commodity="LETTUCE", region="ARIZONA"),
            _raw_row(rpm="4.0", commodity="BERRIES", region="CALIFORNIA"),
        ]
    )
    panel = build_lane_week_panel(df, validate=False)

    assert len(panel) == 1
    assert panel.loc[0, "rpm"] == pytest.approx(3.0)
    assert panel.loc[0, "raw_rows_in_group"] == 2
    assert panel.loc[0, "commodity_count"] == 2
    assert panel.loc[0, "region_count"] == 2
    assert bool(panel.loc[0, "rpm_conflict"])
    assert panel.loc[0, "lane_id"] == "ORIGIN A -> ATLANTA"


def test_build_lane_week_panel_no_conflict_when_rpm_identical() -> None:
    df = pd.DataFrame(
        [
            _raw_row(rpm="2.0", commodity="LETTUCE"),
            _raw_row(rpm="2.0", commodity="BERRIES"),
        ]
    )
    panel = build_lane_week_panel(df, validate=False)
    assert not bool(panel.loc[0, "rpm_conflict"])
    assert panel.loc[0, "rpm"] == pytest.approx(2.0)


def test_build_lane_week_panel_drops_leakage_columns() -> None:
    df = pd.DataFrame([_raw_row()])
    panel = build_lane_week_panel(df, validate=False)
    for col in LEAKAGE_COLUMNS:
        assert col not in panel.columns


def test_build_lane_week_panel_respects_lane_week_key() -> None:
    df = pd.DataFrame(
        [
            _raw_row(destination="ATLANTA"),
            _raw_row(destination="CHICAGO"),
        ]
    )
    panel = build_lane_week_panel(df, validate=False)
    assert len(panel) == 2
    assert set(panel["destination"]) == {"ATLANTA", "CHICAGO"}


def test_build_lane_week_panel_drop_invalid_rpm() -> None:
    df = pd.DataFrame([_raw_row(rpm="not-a-number")])
    panel = build_lane_week_panel(df, validate=False, drop_invalid_rpm=True)
    assert panel.empty


def test_build_lane_week_panel_drop_invalid_distance() -> None:
    df = pd.DataFrame([_raw_row(distance="0")])
    panel = build_lane_week_panel(
        df,
        validate=False,
        drop_invalid_rpm=False,
        drop_invalid_distance=True,
    )
    assert panel.empty


def test_build_lane_week_panel_validate_schema() -> None:
    df = pd.DataFrame({"date": ["2000-01-01"], "origin": ["A"]})
    with pytest.raises(SchemaValidationError):
        build_lane_week_panel(df, validate=True)


def test_lane_week_key_constant() -> None:
    assert LANE_WEEK_KEY == ("date", "origin", "destination")


def test_build_lane_week_panel_expected_columns_subset() -> None:
    df = pd.DataFrame([_raw_row()])
    panel = build_lane_week_panel(df, validate=False)
    assert set(LANE_WEEK_KEY).issubset(panel.columns)
    assert "lane_id" in panel.columns
    assert set(EXPECTED_COLUMNS).issuperset({"date", "origin", "destination", "rpm"})
