"""Unit tests for walk-forward GBM helpers (leakage checks)."""

from __future__ import annotations

import pandas as pd

from freight_rates.features import TARGET_COLUMN
from freight_rates.preprocessing import extend_panel_for_forecast
from freight_rates.splits import iter_walk_forward
from freight_rates.walkforward import default_gbm, load_hpo_best_params, run_walkforward_gbm


def test_default_gbm_optimizes_absolute_error() -> None:
    model = default_gbm()
    assert model.loss == "absolute_error"


def _panel() -> pd.DataFrame:
    rows = []
    for i, d in enumerate(["2024-12-24", "2024-12-31", "2025-01-07", "2025-01-14", "2025-01-21"]):
        ts = pd.Timestamp(d)
        rows.append(
            {
                "date": ts,
                "origin": "ORIGIN A",
                "destination": "ATLANTA",
                "lane_id": "ORIGIN A -> ATLANTA",
                "region": "R1",
                "year": ts.year,
                "month": ts.month,
                "week": int(ts.isocalendar().week),
                "quarter": (ts.month - 1) // 3 + 1,
                "distance": 900.0,
                "availability": float(2 + (i % 3)),
                "diesel_us": 3.5 + 0.01 * i,
                "diesel_us_chg_1w": 0.01,
                TARGET_COLUMN: 2.0 + 0.05 * i,
            }
        )
    return pd.DataFrame(rows)


def test_iter_walk_forward_train_before_t() -> None:
    for fold in iter_walk_forward(_panel(), first_forecast_date="2025-01-07"):
        assert fold.train["date"].max() < fold.t
        assert (fold.score["date"] == fold.t).all()


def test_load_hpo_best_params(tmp_path) -> None:
    path = tmp_path / "best_params.json"
    pd.Series(
        {
            "config": "d3_l60_l2_1.0",
            "max_depth": 3,
            "min_samples_leaf": 60,
            "l2_regularization": 1.0,
            "learning_rate": 0.08,
            "max_iter": 200,
        }
    ).to_json(path)

    params = load_hpo_best_params(path)

    assert params == {
        "max_depth": 3,
        "min_samples_leaf": 60,
        "l2_regularization": 1.0,
        "learning_rate": 0.08,
        "max_iter": 200,
    }
    model = default_gbm(**params)
    assert model.max_depth == 3
    assert model.min_samples_leaf == 60


def test_walkforward_prediction_count_matches_score() -> None:
    panel = _panel()
    result = run_walkforward_gbm(
        panel,
        first_forecast_date="2025-01-07",
        model=default_gbm(max_iter=15, max_depth=2, min_samples_leaf=1),
    )
    expected = sum(len(f.score) for f in iter_walk_forward(panel, first_forecast_date="2025-01-07"))
    assert len(result.predictions) == expected


def test_walkforward_single_week_score() -> None:
    panel = _panel()
    t = pd.Timestamp("2025-01-21")
    result = run_walkforward_gbm(
        panel,
        first_forecast_date=t,
        last_forecast_date=t,
        model=default_gbm(max_iter=15, max_depth=2, min_samples_leaf=1),
    )
    assert len(result.predictions) == 1
    assert set(result.predictions["date"]) == {t}
    assert result.predictions["y_pred"].notna().all()


def test_walkforward_forward_week_with_scaffold() -> None:
    panel = _panel()
    t = pd.Timestamp("2025-01-28")
    extended = extend_panel_for_forecast(panel, t)
    result = run_walkforward_gbm(
        extended,
        first_forecast_date=t,
        last_forecast_date=t,
        model=default_gbm(max_iter=15, max_depth=2, min_samples_leaf=1),
    )
    assert len(result.predictions) == 1
    assert set(result.predictions["date"]) == {t}
    assert pd.isna(result.predictions["y_true"].iloc[0])
    assert result.predictions["y_pred"].notna().all()
