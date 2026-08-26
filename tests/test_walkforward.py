"""Unit tests for walk-forward GBM helpers (leakage checks)."""

from __future__ import annotations

import pandas as pd

from freight_rates.features import TARGET_COLUMN
from freight_rates.splits import iter_walk_forward
from freight_rates.walkforward import default_gbm, run_walkforward_gbm


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
                TARGET_COLUMN: 2.0 + 0.05 * i,
            }
        )
    return pd.DataFrame(rows)


def test_iter_walk_forward_train_before_t() -> None:
    for fold in iter_walk_forward(_panel(), first_forecast_date="2025-01-07"):
        assert fold.train["date"].max() < fold.t
        assert (fold.score["date"] == fold.t).all()


def test_walkforward_prediction_count_matches_score() -> None:
    panel = _panel()
    result = run_walkforward_gbm(
        panel,
        first_forecast_date="2025-01-07",
        model=default_gbm(max_iter=15, max_depth=2, min_samples_leaf=1),
    )
    expected = sum(len(f.score) for f in iter_walk_forward(panel, first_forecast_date="2025-01-07"))
    assert len(result.predictions) == expected
