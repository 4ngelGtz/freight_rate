"""Unit tests for evaluation metrics and walk-forward GBM smoke."""

from __future__ import annotations

import pandas as pd
import pytest

from freight_rates.evaluation import (
    HISTORY_BUCKETS,
    assign_history_bucket,
    dual_regime_headline,
    format_dual_regime_report,
    mae,
    mape,
    medae,
    summarize_predictions,
)
from freight_rates.features import TARGET_COLUMN
from freight_rates.walkforward import default_gbm, run_walkforward_gbm


def test_mae_medae_mape() -> None:
    y_true = pd.Series([2.0, 4.0, 6.0])
    y_pred = pd.Series([3.0, 4.0, 5.0])
    assert mae(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert medae(y_true, y_pred) == pytest.approx(1.0)
    # |1|/2 + 0/4 + |1|/6 → mean * 100
    expected_mape = ((0.5 + 0.0 + 1.0 / 6.0) / 3.0) * 100.0
    assert mape(y_true, y_pred) == pytest.approx(expected_mape)


def test_mape_skips_near_zero_truth() -> None:
    y_true = pd.Series([0.0, 2.0])
    y_pred = pd.Series([1.0, 3.0])
    assert mape(y_true, y_pred) == pytest.approx(50.0)


def test_assign_history_bucket() -> None:
    n = pd.Series([0, 4, 5, 19, 20, 99, 100, 250])
    buckets = assign_history_bucket(n)
    assert buckets.tolist() == [
        "0-4",
        "0-4",
        "5-19",
        "5-19",
        "20-99",
        "20-99",
        "100+",
        "100+",
    ]
    assert tuple(HISTORY_BUCKETS) == ("0-4", "5-19", "20-99", "100+")


def test_summarize_predictions_overall_and_buckets() -> None:
    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-07", "2025-01-07", "2025-01-14"]),
            "lane_history_n": [0, 10, 25],
            "y_true": [2.0, 3.0, 4.0],
            "y_pred": [2.5, 2.5, 3.5],
            "y_pred_baseline": [3.0, 3.0, 4.0],
        }
    )
    out = summarize_predictions(preds)
    assert "overall" in out and "by_week" in out and "by_history" in out
    assert out["overall"].loc[0, "n"] == 3
    assert set(out["by_history"]["history_bucket"]) == {"0-4", "5-19", "20-99"}
    assert len(out["by_week"]) == 2


def _synthetic_panel() -> pd.DataFrame:
    """Tiny lane-week panel spanning burn-in + two forecast Tuesdays."""
    rows = []
    dates = [
        "2024-12-17",
        "2024-12-24",
        "2024-12-31",
        "2025-01-07",
        "2025-01-14",
    ]
    for origin, dest, base in (
        ("ORIGIN A", "ATLANTA", 2.0),
        ("ORIGIN B", "CHICAGO", 3.0),
    ):
        for i, d in enumerate(dates):
            ts = pd.Timestamp(d)
            rows.append(
                {
                    "date": ts,
                    "origin": origin,
                    "destination": dest,
                    "lane_id": f"{origin} -> {dest}",
                    "region": "R1",
                    "year": ts.year,
                    "month": ts.month,
                    "week": int(ts.isocalendar().week),
                    "quarter": (ts.month - 1) // 3 + 1,
                    "distance": 1000.0,
                    "availability": 3.0,
                    "diesel_us": 3.5 + 0.01 * i,
                    "diesel_us_chg_1w": 0.01,
                    TARGET_COLUMN: base + 0.1 * i,
                }
            )
    return pd.DataFrame(rows)


def test_run_walkforward_gbm_smoke_no_leakage() -> None:
    panel = _synthetic_panel()
    result = run_walkforward_gbm(
        panel,
        first_forecast_date="2025-01-07",
        model=default_gbm(max_iter=20, max_depth=2, min_samples_leaf=1),
        verbose=False,
    )

    assert not result.predictions.empty
    assert len(result.predictions) == 4  # 2 lanes × 2 forecast weeks
    assert set(result.predictions["date"]) == {
        pd.Timestamp("2025-01-07"),
        pd.Timestamp("2025-01-14"),
    }
    # Predictions align 1:1 with score rows.
    for t, g in result.predictions.groupby("date"):
        score_n = int((panel["date"] == t).sum())
        assert len(g) == score_n

    # No future in any implied train: first forecast uses only pre-2025 dates.
    assert result.predictions["date"].min() >= pd.Timestamp("2025-01-07")
    assert "y_pred_baseline" in result.predictions.columns
    assert "delta_hat" in result.predictions.columns
    # Residual target: y_pred = lag-1 + delta_hat
    reconstructed = result.predictions["y_pred_baseline"] + result.predictions["delta_hat"]
    pd.testing.assert_series_equal(
        result.predictions["y_pred"],
        reconstructed,
        check_names=False,
        atol=1e-9,
    )
    assert result.overall.loc[0, "n"] == 4


def test_run_walkforward_respects_max_folds() -> None:
    panel = _synthetic_panel()
    result = run_walkforward_gbm(
        panel,
        first_forecast_date="2025-01-07",
        max_folds=1,
        model=default_gbm(max_iter=10, max_depth=2, min_samples_leaf=1),
    )
    assert set(result.predictions["date"]) == {pd.Timestamp("2025-01-07")}
    assert len(result.predictions) == 2


def test_run_walkforward_direct_target_still_works() -> None:
    panel = _synthetic_panel()
    result = run_walkforward_gbm(
        panel,
        first_forecast_date="2025-01-07",
        max_folds=1,
        residual_target=False,
        model=default_gbm(max_iter=10, max_depth=2, min_samples_leaf=1),
    )
    assert len(result.predictions) == 2
    assert "delta_hat" in result.predictions.columns


def test_dual_regime_headline_cold_start() -> None:
    overall = pd.DataFrame(
        [
            {
                "n": 100,
                "mae": 0.15,
                "medae": 0.05,
                "mape": 4.0,
                "mae_baseline": 0.14,
                "medae_baseline": 0.02,
                "mae_lift": -0.01,
            }
        ]
    )
    by_history = pd.DataFrame(
        [
            {
                "history_bucket": "0-4",
                "n": 20,
                "mae": 0.35,
                "medae": 0.2,
                "mape": 10.0,
                "mae_baseline": 0.40,
                "medae_baseline": 0.22,
                "mae_lift": 0.05,
            },
            {
                "history_bucket": "20-99",
                "n": 80,
                "mae": 0.10,
                "medae": 0.02,
                "mape": 3.0,
                "mae_baseline": 0.09,
                "medae_baseline": 0.01,
                "mae_lift": -0.01,
            },
        ]
    )
    headline = dual_regime_headline(overall, by_history, label="full")
    assert headline.loc[0, "label"] == "full"
    assert bool(headline.loc[0, "beats_lag1_overall"]) is False
    assert bool(headline.loc[0, "beats_lag1_cold"]) is True
    assert headline.loc[0, "cold_mae_lift"] == pytest.approx(0.05)
    text = format_dual_regime_report(headline)
    assert "Dual-regime" in text
    assert "cold 0-4" in text


def test_run_walkforward_diesel_none_mode() -> None:
    panel = _synthetic_panel()
    result = run_walkforward_gbm(
        panel,
        first_forecast_date="2025-01-07",
        max_folds=1,
        diesel_features="none",
        model=default_gbm(max_iter=10, max_depth=2, min_samples_leaf=1),
    )
    assert result.diesel_features == "none"
    assert len(result.predictions) == 2
