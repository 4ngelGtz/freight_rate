"""Unit tests for nested HPO selection logic."""

from __future__ import annotations

import pandas as pd

from freight_rates.hpo import select_best_config


def _row(
    label: str,
    mae: float,
    cold_lift: float,
    *,
    depth: int = 3,
    leaf: int = 20,
    l2: float = 0.1,
) -> dict:
    return {
        "label": label,
        "mae": mae,
        "cold_mae_lift": cold_lift,
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "l2_regularization": l2,
        "learning_rate": 0.08,
        "max_iter": 200,
    }


def test_select_best_prefers_cold_lift_then_lowest_mae() -> None:
    summary = pd.DataFrame(
        [
            _row("a", mae=0.10, cold_lift=-0.01),
            _row("b", mae=0.12, cold_lift=0.02),
            _row("c", mae=0.11, cold_lift=0.01),
        ]
    )
    _, params, label = select_best_config(summary)
    assert label == "c"
    assert params["max_depth"] == 3


def test_select_best_fallback_when_no_cold_lift() -> None:
    summary = pd.DataFrame(
        [
            _row("a", mae=0.10, cold_lift=-0.01),
            _row("b", mae=0.12, cold_lift=-0.05),
        ]
    )
    _, _, label = select_best_config(summary)
    assert label == "a"
