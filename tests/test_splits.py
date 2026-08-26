"""Unit tests for temporal splits and walk-forward folds."""

from __future__ import annotations

import pandas as pd
import pytest

from freight_rates.splits import (
    API_START_DATE,
    DIAGNOSTIC_TEST_START,
    DIAGNOSTIC_TRAIN_END,
    DIAGNOSTIC_VAL_END,
    DIAGNOSTIC_VAL_START,
    FIRST_FORECAST_DATE,
    assign_temporal_split,
    filter_model_window,
    iter_walk_forward,
)


def _panel(dates: list[str], *, origin: str = "ORIGIN A") -> pd.DataFrame:
    """Minimal lane-week panel for split tests."""
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "date": pd.Timestamp(d),
                "origin": origin,
                "destination": "ATLANTA",
                "lane_id": f"{origin} -> ATLANTA",
                "rpm": 2.0 + 0.01 * i,
                "distance": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_assign_temporal_split_diagnostic_labels() -> None:
    dates = pd.Series(
        [
            "2024-07-02",
            "2024-12-31",
            "2025-01-07",
            "2025-06-24",
            "2025-07-01",
            "2026-01-06",
        ]
    )
    labels = assign_temporal_split(dates)
    assert labels.tolist() == ["train", "train", "val", "val", "test", "test"]
    assert DIAGNOSTIC_TRAIN_END == pd.Timestamp("2024-12-31")
    assert DIAGNOSTIC_VAL_START == FIRST_FORECAST_DATE
    assert DIAGNOSTIC_VAL_END == pd.Timestamp("2025-06-24")
    assert DIAGNOSTIC_TEST_START == pd.Timestamp("2025-07-01")


def test_filter_model_window() -> None:
    panel = _panel(["2024-06-25", "2024-07-02", "2025-01-07"])
    filtered = filter_model_window(panel)
    assert filtered["date"].min() >= API_START_DATE
    assert len(filtered) == 2
    assert list(filtered["date"].dt.strftime("%Y-%m-%d")) == ["2024-07-02", "2025-01-07"]


def test_iter_walk_forward_no_future_leakage() -> None:
    panel = _panel(
        [
            "2024-12-17",
            "2024-12-24",
            "2024-12-31",
            "2025-01-07",
            "2025-01-14",
            "2025-01-21",
        ]
    )
    folds = list(iter_walk_forward(panel))
    assert folds
    for fold in folds:
        assert fold.train["date"].max() < fold.t
        assert (fold.score["date"] == fold.t).all()
        assert fold.t >= FIRST_FORECAST_DATE


def test_iter_walk_forward_first_t_at_or_after_first_forecast() -> None:
    panel = _panel(
        [
            "2024-12-31",
            "2025-01-07",
            "2025-01-14",
        ]
    )
    first = next(iter_walk_forward(panel))
    assert first.t == FIRST_FORECAST_DATE
    assert first.t >= FIRST_FORECAST_DATE
    assert len(first.train) == 1
    assert len(first.score) == 1


def test_iter_walk_forward_skips_empty_score_weeks() -> None:
    # Gap week 2025-01-14 has no rows → must not be yielded.
    panel = _panel(
        [
            "2024-12-31",
            "2025-01-07",
            "2025-01-21",
        ]
    )
    yielded = [fold.t for fold in iter_walk_forward(panel)]
    assert yielded == [pd.Timestamp("2025-01-07"), pd.Timestamp("2025-01-21")]
    assert pd.Timestamp("2025-01-14") not in yielded


def test_iter_walk_forward_respects_custom_first_forecast_date() -> None:
    panel = _panel(
        [
            "2024-12-31",
            "2025-01-07",
            "2025-01-14",
            "2025-01-21",
        ]
    )
    folds = list(iter_walk_forward(panel, first_forecast_date="2025-01-14"))
    assert folds[0].t == pd.Timestamp("2025-01-14")
    assert all(f.t >= pd.Timestamp("2025-01-14") for f in folds)


def test_iter_walk_forward_requires_date_column() -> None:
    with pytest.raises(KeyError, match="date"):
        list(iter_walk_forward(pd.DataFrame({"rpm": [1.0]})))
