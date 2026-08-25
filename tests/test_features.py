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


def _panel_rows() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "date": "2000-01-04T00:00:00.000",
                "week": "1",
                "month": "1",
                "quarter": "1",
                "year": "2000",
                "region": "R1",
                "origin": "ORIGIN A",
                "destination": "ATLANTA",
                "distance": "1000",
                "commodity": "LETTUCE",
                "weeklow": "2000",
                "weekhigh": "2200",
                "midpoint": "2100",
                "rpm": "2.0",
                "availability": "3",
            },
            {
                "date": "2000-01-11T00:00:00.000",
                "week": "2",
                "month": "1",
                "quarter": "1",
                "year": "2000",
                "region": "R1",
                "origin": "ORIGIN A",
                "destination": "ATLANTA",
                "distance": "1000",
                "commodity": "LETTUCE",
                "weeklow": "2100",
                "weekhigh": "2300",
                "midpoint": "2200",
                "rpm": "2.2",
                "availability": "3",
            },
            {
                "date": "2000-01-04T00:00:00.000",
                "week": "1",
                "month": "1",
                "quarter": "1",
                "year": "2000",
                "region": "R2",
                "origin": "ORIGIN B",
                "destination": "CHICAGO",
                "distance": "800",
                "commodity": "BERRIES",
                "weeklow": "1500",
                "weekhigh": "1700",
                "midpoint": "1600",
                "rpm": "2.5",
                "availability": "2",
            },
        ]
    )
    return build_lane_week_panel(raw, validate=False)


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


def test_feature_builder_lane_history_starts_at_zero() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    first_idx = panel.index[
        (panel["origin"] == "ORIGIN A")
        & (panel["destination"] == "ATLANTA")
        & (panel["date"] == pd.Timestamp("2000-01-04"))
    ][0]
    second_idx = panel.index[
        (panel["origin"] == "ORIGIN A")
        & (panel["destination"] == "ATLANTA")
        & (panel["date"] == pd.Timestamp("2000-01-11"))
    ][0]
    assert features.loc[first_idx, "lane_history_n"] == 0.0
    assert features.loc[second_idx, "lane_history_n"] == 1.0


def test_feature_builder_rpm_lag_uses_prior_observation() -> None:
    panel = _panel_rows()
    builder = LaneWeekFeatureBuilder()
    features = builder.fit_transform(panel)

    second_week_idx = panel.index[
        (panel["origin"] == "ORIGIN A")
        & (panel["destination"] == "ATLANTA")
        & (panel["date"] == pd.Timestamp("2000-01-11"))
    ][0]
    assert features.loc[second_week_idx, "rpm_lag_1"] == pytest.approx(2.0)


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


def test_categorical_features_constant() -> None:
    assert CATEGORICAL_FEATURES == ("origin", "destination")
