"""Unit tests for lane-week feature engineering."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from freight_rates.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    LaneWeekFeatureBuilder,
)
from freight_rates.preprocessing import LEAKAGE_COLUMNS, build_lane_week_panel


def _raw_row(
    *,
    date: str,
    week: str,
    origin: str = "ORIGIN A",
    destination: str = "ATLANTA",
    region: str = "R1",
    rpm: str = "2.0",
    availability: str = "3",
    distance: str = "1000",
    commodity: str = "LETTUCE",
) -> dict:
    ts = pd.Timestamp(date)
    return {
        "date": date,
        "week": week,
        "month": str(ts.month),
        "quarter": str((ts.month - 1) // 3 + 1),
        "year": str(ts.year),
        "region": region,
        "origin": origin,
        "destination": destination,
        "distance": distance,
        "commodity": commodity,
        "weeklow": "2000",
        "weekhigh": "2200",
        "midpoint": "2100",
        "rpm": rpm,
        "availability": availability,
    }


def _panel_rows() -> pd.DataFrame:
    """Lane A: three consecutive weeks + a later stale reappearance; lane B: one week."""
    raw = pd.DataFrame(
        [
            _raw_row(date="2000-01-04T00:00:00.000", week="1", rpm="2.0", availability="3"),
            _raw_row(date="2000-01-11T00:00:00.000", week="2", rpm="2.2", availability="5"),
            _raw_row(date="2000-01-18T00:00:00.000", week="3", rpm="2.4", availability="4"),
            # Gap of two weeks (skip 2000-01-25) → stale lag on 2000-02-01.
            _raw_row(date="2000-02-01T00:00:00.000", week="5", rpm="2.6", availability="2"),
            _raw_row(
                date="2000-01-04T00:00:00.000",
                week="1",
                origin="ORIGIN B",
                destination="CHICAGO",
                region="R2",
                rpm="2.5",
                availability="2",
                distance="800",
                commodity="BERRIES",
            ),
        ]
    )
    return build_lane_week_panel(raw, validate=False)


def _lane_a_idx(panel: pd.DataFrame, date: str) -> int:
    return panel.index[
        (panel["origin"] == "ORIGIN A")
        & (panel["destination"] == "ATLANTA")
        & (panel["date"] == pd.Timestamp(date))
    ][0]


def test_feature_builder_fit_transform_shape() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    assert len(features) == len(panel)
    assert len(features.columns) == len(builder.get_feature_names_out())
    assert set(NUMERIC_FEATURES).issubset(features.columns)


def test_feature_builder_no_leakage_columns() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)
    for col in LEAKAGE_COLUMNS:
        assert col not in features.columns
    assert "availability" not in features.columns
    assert "availability_lag_1" in features.columns


def test_feature_builder_lane_history_starts_at_zero() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    first_idx = _lane_a_idx(panel, "2000-01-04")
    second_idx = _lane_a_idx(panel, "2000-01-11")
    assert features.loc[first_idx, "lane_history_n"] == 0.0
    assert features.loc[second_idx, "lane_history_n"] == 1.0


def test_feature_builder_rpm_lag_uses_prior_observation() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    second_week_idx = _lane_a_idx(panel, "2000-01-11")
    third_week_idx = _lane_a_idx(panel, "2000-01-18")
    assert features.loc[second_week_idx, "rpm_lag_1"] == pytest.approx(2.0)
    assert features.loc[third_week_idx, "rpm_lag_1"] == pytest.approx(2.2)
    assert features.loc[third_week_idx, "rpm_lag_2"] == pytest.approx(2.0)


def test_feature_builder_rpm_delta_and_vs_rolling() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    third_idx = _lane_a_idx(panel, "2000-01-18")
    # lag_1=2.2, lag_2=2.0 → delta 0.2
    assert features.loc[third_idx, "rpm_delta_1_2"] == pytest.approx(0.2)
    # rolling_4 of prior obs: mean(2.0, 2.2) = 2.1; vs rolling = 2.2 - 2.1
    assert features.loc[third_idx, "rpm_vs_rolling_4"] == pytest.approx(0.1)


def test_feature_builder_availability_lag_uses_prior_observation() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    second_week_idx = _lane_a_idx(panel, "2000-01-11")
    assert panel.loc[second_week_idx, "availability"] == pytest.approx(5.0)
    assert features.loc[second_week_idx, "availability_lag_1"] == pytest.approx(3.0)


def test_feature_builder_recency_and_stale_flags() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    first_idx = _lane_a_idx(panel, "2000-01-04")
    second_idx = _lane_a_idx(panel, "2000-01-11")
    stale_idx = _lane_a_idx(panel, "2000-02-01")

    assert features.loc[first_idx, "has_prior_observation"] == 0.0
    assert features.loc[first_idx, "weeks_since_last_observation"] == 0.0
    assert features.loc[first_idx, "is_stale_lag"] == 0.0

    assert features.loc[second_idx, "has_prior_observation"] == 1.0
    assert features.loc[second_idx, "weeks_since_last_observation"] == pytest.approx(1.0)
    assert features.loc[second_idx, "is_stale_lag"] == 0.0

    # 2000-01-18 → 2000-02-01 is 14 days = 2 weeks.
    assert features.loc[stale_idx, "has_prior_observation"] == 1.0
    assert features.loc[stale_idx, "weeks_since_last_observation"] == pytest.approx(2.0)
    assert features.loc[stale_idx, "is_stale_lag"] == 1.0


def test_feature_builder_cold_start_uses_global_mean_for_first_obs() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    expected_global = panel[TARGET_COLUMN].mean()
    assert builder.global_rpm_mean_ == pytest.approx(expected_global)

    for origin in ("ORIGIN A", "ORIGIN B"):
        first_idx = panel.index[
            (panel["origin"] == origin) & (panel["date"] == pd.Timestamp("2000-01-04"))
        ][0]
        assert features.loc[first_idx, "rpm_lag_1"] == pytest.approx(expected_global)
        assert features.loc[first_idx, "rpm_lag_2"] == pytest.approx(expected_global)


def test_feature_builder_availability_cold_start_uses_train_median() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    expected_fill = float(panel["availability"].median())
    assert builder.global_availability_fill_ == pytest.approx(expected_fill)

    for origin in ("ORIGIN A", "ORIGIN B"):
        first_idx = panel.index[
            (panel["origin"] == origin) & (panel["date"] == pd.Timestamp("2000-01-04"))
        ][0]
        assert features.loc[first_idx, "availability_lag_1"] == pytest.approx(expected_fill)


def test_feature_builder_encodes_region() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)
    region_cols = [c for c in features.columns if c.startswith("cat__region_")]
    assert region_cols
    assert set(NUMERIC_FEATURES).isdisjoint(region_cols)


def test_feature_builder_serializable_with_pickle(tmp_path: Path) -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    builder.fit(panel)

    path = tmp_path / "feature_builder.pkl"
    path.write_bytes(pickle.dumps(builder))
    loaded = pickle.loads(path.read_bytes())

    pd.testing.assert_frame_equal(
        loaded.transform(panel),
        builder.transform(panel),
    )


def test_feature_builder_unknown_category_does_not_fail() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    builder.fit(panel)

    scored = panel.iloc[[0]].copy()
    scored.loc[scored.index[0], "origin"] = "BRAND NEW ORIGIN"
    scored.loc[scored.index[0], "region"] = "BRAND NEW REGION"
    features = builder.transform(scored)
    assert len(features) == 1


def test_feature_builder_requires_fit() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    with pytest.raises(RuntimeError, match="not been fitted"):
        builder.transform(panel)


def test_feature_builder_missing_columns() -> None:
    panel = _panel_rows().drop(columns=["distance"])
    builder = LaneWeekFeatureBuilder()
    with pytest.raises(ValueError, match="missing required columns"):
        builder.fit(panel)


def test_feature_builder_missing_region() -> None:
    panel = _panel_rows().drop(columns=["region"])
    builder = LaneWeekFeatureBuilder()
    with pytest.raises(ValueError, match="missing required columns"):
        builder.fit(panel)


def test_categorical_features_constant() -> None:
    assert CATEGORICAL_FEATURES == ("origin", "destination", "region")
    assert "availability" not in NUMERIC_FEATURES
    assert "availability_lag_1" in NUMERIC_FEATURES
    for name in (
        "rpm_lag_2",
        "weeks_since_last_observation",
        "has_prior_observation",
        "is_stale_lag",
        "rpm_delta_1_2",
        "rpm_vs_rolling_4",
    ):
        assert name in NUMERIC_FEATURES
