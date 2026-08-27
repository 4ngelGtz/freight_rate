"""Unit tests for lane-week preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from freight_rates.ingestion import EXPECTED_COLUMNS, SchemaValidationError
from freight_rates.preprocessing import (
    LANE_WEEK_KEY,
    LEAKAGE_COLUMNS,
    build_forecast_scaffold,
    build_lane_week_panel,
    build_modeling_panel,
    coerce_raw_numeric,
    extend_panel_for_forecast,
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


def test_build_modeling_panel_attaches_diesel() -> None:
    from freight_rates.preprocessing import build_modeling_panel

    rates = pd.DataFrame(
        [
            _raw_row(date="2025-01-07T00:00:00.000", week="2"),
        ]
    )
    diesel = pd.DataFrame(
        [
            {
                "date": "2024-12-30T00:00:00.000",
                "week": "1",
                "month": "12",
                "year": "2024",
                "region": "US",
                "diesel_price": "3.55",
            },
            {
                "date": "2025-01-06T00:00:00.000",
                "week": "2",
                "month": "1",
                "year": "2025",
                "region": "US",
                "diesel_price": "3.75",
            },
        ]
    )
    panel = build_modeling_panel(rates, diesel, validate=False)
    assert "diesel_us" in panel.columns
    assert panel.loc[0, "diesel_date"] == pd.Timestamp("2024-12-30")
    assert panel.loc[0, "diesel_us"] == pytest.approx(3.55)


def _rates_and_diesel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rates = pd.DataFrame(
        [
            _raw_row(date="2025-01-07T00:00:00.000", rpm="2.0"),
            _raw_row(date="2025-01-14T00:00:00.000", rpm="2.1"),
        ]
    )
    diesel = pd.DataFrame(
        [
            {
                "date": "2024-12-30T00:00:00.000",
                "week": "1",
                "month": "12",
                "quarter": "4",
                "year": "2024",
                "region": "US",
                "diesel_price": "3.55",
            },
            {
                "date": "2025-01-06T00:00:00.000",
                "week": "2",
                "month": "1",
                "quarter": "1",
                "year": "2025",
                "region": "US",
                "diesel_price": "3.75",
            },
        ]
    )
    return rates, diesel


def test_build_forecast_scaffold_forward_week() -> None:
    rates, diesel = _rates_and_diesel()
    panel = build_modeling_panel(rates, diesel, validate=False)
    scaffold = build_forecast_scaffold(panel, "2025-01-21", diesel_raw=diesel)

    assert len(scaffold) == len(panel.loc[panel["date"] == pd.Timestamp("2025-01-14")])
    assert scaffold["date"].nunique() == 1
    assert pd.Timestamp(scaffold["date"].iloc[0]) == pd.Timestamp("2025-01-21")
    assert scaffold["rpm"].isna().all()
    assert scaffold["diesel_us"].notna().all()


def test_extend_panel_for_forecast_appends_scaffold() -> None:
    rates, diesel = _rates_and_diesel()
    panel = build_modeling_panel(rates, diesel, validate=False)
    extended = extend_panel_for_forecast(panel, "2025-01-21", diesel_raw=diesel)

    assert len(extended) == len(panel) + len(
        panel.loc[panel["date"] == pd.Timestamp("2025-01-14")]
    )
    assert (extended["date"] == pd.Timestamp("2025-01-21")).any()
