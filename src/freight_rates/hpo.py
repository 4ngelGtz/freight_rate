"""Nested walk-forward hyperparameter tuning for the global GBM (validation only)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from freight_rates.evaluation import dual_regime_headline, format_dual_regime_report
from freight_rates.splits import DIAGNOSTIC_VAL_END, DIAGNOSTIC_VAL_START
from freight_rates.walkforward import WalkForwardResult, default_gbm, run_walkforward_gbm

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 4, 6],
    "min_samples_leaf": [20, 40, 60],
    "l2_regularization": [0.1, 1.0],
    "learning_rate": [0.08],
    "max_iter": [200],
}

_PARAM_COLS = (
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "learning_rate",
    "max_iter",
)


@dataclass(frozen=True)
class HpoResult:
    """Validation HPO outputs."""

    val_summary: pd.DataFrame
    best_params: dict[str, Any]
    best_label: str
    best_row: pd.Series
    val_folds_best: pd.DataFrame


def iter_param_configs(
    grid: dict[str, list[Any]] | None = None,
    *,
    max_configs: int | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(label, params)`` for each grid combination."""
    g = grid if grid is not None else DEFAULT_PARAM_GRID
    keys = list(g.keys())
    configs = [dict(zip(keys, vals, strict=True)) for vals in product(*(g[k] for k in keys))]
    if max_configs is not None:
        configs = configs[:max_configs]
    for i, params in enumerate(configs):
        yield f"cfg_{i:02d}", params


def evaluate_config(
    panel: pd.DataFrame,
    params: dict[str, Any],
    *,
    first_forecast_date: pd.Timestamp | str = DIAGNOSTIC_VAL_START,
    last_forecast_date: pd.Timestamp | str = DIAGNOSTIC_VAL_END,
    label: str,
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, WalkForwardResult]:
    """Run val-window walk-forward for one config; return headline + full result."""
    result = run_walkforward_gbm(
        panel,
        first_forecast_date=first_forecast_date,
        last_forecast_date=last_forecast_date,
        max_folds=max_folds,
        model=default_gbm(**params),
        residual_target=True,
        diesel_features="level",
        verbose=False,
    )
    headline = dual_regime_headline(result.overall, result.by_history, label=label)
    for key, value in params.items():
        headline[key] = value
    return headline, result


def select_best_config(val_summary: pd.DataFrame) -> tuple[pd.Series, dict[str, Any], str]:
    """Pick θ* on val: cold lift ≥ 0 first, then lowest overall MAE."""
    eligible = val_summary.loc[val_summary["cold_mae_lift"] >= 0].copy()
    if eligible.empty:
        best_row = val_summary.sort_values(
            ["cold_mae_lift", "mae"], ascending=[False, True]
        ).iloc[0]
    else:
        best_row = eligible.sort_values("mae", ascending=True).iloc[0]

    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "l2_regularization": float(best_row["l2_regularization"]),
        "learning_rate": float(best_row["learning_rate"]),
        "max_iter": int(best_row["max_iter"]),
    }
    return best_row, best_params, str(best_row["label"])


def run_nested_hpo(
    panel: pd.DataFrame,
    *,
    output_dir: Path | str,
    grid: dict[str, list[Any]] | None = None,
    val_first: pd.Timestamp | str = DIAGNOSTIC_VAL_START,
    val_last: pd.Timestamp | str = DIAGNOSTIC_VAL_END,
    max_configs: int | None = None,
    max_folds: int | None = None,
    verbose: bool = True,
) -> HpoResult:
    """Tune on validation expanding-window only; write artifacts under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = list(iter_param_configs(grid, max_configs=max_configs))
    val_rows: list[pd.DataFrame] = []
    val_by_week: dict[str, pd.DataFrame] = {}

    for i, (label, params) in enumerate(configs):
        if verbose:
            print(f"[{i + 1}/{len(configs)}] val {label} {params}")
        headline, result = evaluate_config(
            panel,
            params,
            first_forecast_date=val_first,
            last_forecast_date=val_last,
            label=label,
            max_folds=max_folds,
        )
        val_rows.append(headline)
        val_by_week[label] = result.by_week.assign(config=label, **params)

    val_summary = pd.concat(val_rows, ignore_index=True)
    best_row, best_params, best_label = select_best_config(val_summary)

    val_folds = val_by_week[best_label].copy()
    val_folds["date"] = pd.to_datetime(val_folds["date"])
    val_folds = val_folds.sort_values("date").reset_index(drop=True)
    val_folds.insert(0, "fold", range(1, len(val_folds) + 1))

    save_hpo_outputs(
        out,
        val_summary=val_summary,
        best_params=best_params,
        best_label=best_label,
        val_folds_best=val_folds,
    )

    if verbose:
        print("θ* =", best_params)
        print(format_dual_regime_report(best_row.to_frame().T))

    return HpoResult(
        val_summary=val_summary,
        best_params=best_params,
        best_label=best_label,
        best_row=best_row,
        val_folds_best=val_folds,
    )


def save_hpo_outputs(
    output_dir: Path | str,
    *,
    val_summary: pd.DataFrame,
    best_params: dict[str, Any],
    best_label: str,
    val_folds_best: pd.DataFrame,
) -> dict[str, Path]:
    """Write HPO CSV/JSON artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "val_summary": out / "hpo_val_summary.csv",
        "best_params": out / "best_params.json",
        "val_folds_best": out / "hpo_val_folds_best.csv",
    }
    val_summary.to_csv(paths["val_summary"], index=False)
    pd.Series({"config": best_label, **best_params}).to_json(paths["best_params"])
    fold_cols = [
        "fold",
        "date",
        "n",
        "mae",
        "mae_baseline",
        "mae_lift",
        "medae",
        "mape",
    ]
    val_folds_best[fold_cols].to_csv(paths["val_folds_best"], index=False)
    return paths
