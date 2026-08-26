"""Expanding-window walk-forward training for a global GBM (Case B)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from freight_rates.evaluation import assign_history_bucket, summarize_predictions
from freight_rates.features import TARGET_COLUMN, LaneWeekFeatureBuilder
from freight_rates.splits import FIRST_FORECAST_DATE, WalkForwardFold, iter_walk_forward

_GBM_PARAM_KEYS = frozenset(
    {
        "max_depth",
        "min_samples_leaf",
        "l2_regularization",
        "learning_rate",
        "max_iter",
    }
)


@dataclass(frozen=True)
class WalkForwardResult:
    """Predictions and metric tables from a walk-forward run."""

    predictions: pd.DataFrame
    overall: pd.DataFrame
    by_week: pd.DataFrame
    by_history: pd.DataFrame
    diesel_features: str = "level"


def default_gbm(**overrides: Any) -> HistGradientBoostingRegressor:
    """HistGradientBoostingRegressor with modest defaults for weekly re-fits.

    Default ``loss="absolute_error"`` aligns training with the headline MAE
    metric (more robust to large week-to-week jumps than squared error).
    """
    params: dict[str, Any] = {
        "loss": "absolute_error",
        "learning_rate": 0.08,
        "max_iter": 200,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 42,
    }
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def load_hpo_best_params(path: Path | str) -> dict[str, Any]:
    """Load GBM hyperparameters from nested HPO ``best_params.json``."""
    series = pd.read_json(path, typ="series")
    return {key: series[key] for key in _GBM_PARAM_KEYS if key in series.index}


def run_walkforward_gbm(
    panel: pd.DataFrame,
    *,
    first_forecast_date: pd.Timestamp | str = FIRST_FORECAST_DATE,
    last_forecast_date: pd.Timestamp | str | None = None,
    max_folds: int | None = None,
    model: HistGradientBoostingRegressor | None = None,
    residual_target: bool = True,
    diesel_features: str = "level",
    verbose: bool = False,
) -> WalkForwardResult:
    """Train/score an expanding-window global GBM under Case B.

    For each forecast Tuesday ``t``:

    - fit :class:`LaneWeekFeatureBuilder` on ``date < t`` only
    - transform train+score together so lags see prior history
    - by default fit GBM on residual ``delta = rpm - rpm_lag_1``, then
      ``y_pred = rpm_lag_1 + delta_hat`` (anchors to the lag-1 baseline)
    - baseline = ``rpm_lag_1`` (train global mean fill when no history)

    Parameters
    ----------
    panel:
        Lane-week panel already filtered to the modeling window.
    first_forecast_date:
        First week-ending Tuesday to score (default ``2025-01-07``).
    last_forecast_date:
        Optional inclusive upper bound on score weeks (nested val / test
        windows). ``None`` scores through the end of the panel.
    max_folds:
        Optional cap on number of forecast weeks (smoke / debug).
    model:
        Optional pre-configured regressor; a fresh clone is not required —
        the same estimator instance is re-fit each fold.
    residual_target:
        When ``True`` (default), train on ``rpm - rpm_lag_1`` and reconstruct
        RPM predictions by adding the lag-1 baseline back. When ``False``,
        train directly on ``rpm``.
    diesel_features:
        Diesel ablation mode passed to :class:`LaneWeekFeatureBuilder`:
        ``\"none\"``, ``\"level\"`` (``diesel_us`` + ``diesel_distance``), or
        ``\"full\"`` (level + WoW change).
    verbose:
        Print progress every fold when True.
    """
    estimator = model if model is not None else default_gbm()
    rows: list[dict[str, Any]] = []
    folds: Iterator[WalkForwardFold] = iter_walk_forward(
        panel,
        first_forecast_date=first_forecast_date,
        last_forecast_date=last_forecast_date,
    )

    for i, fold in enumerate(folds):
        if max_folds is not None and i >= max_folds:
            break
        if fold.train.empty:
            if verbose:
                print(f"skip {fold.t.date()}: empty train")
            continue

        # Sanity: Case B expanding window — no future leakage into train.
        if fold.train["date"].max() >= fold.t:
            raise RuntimeError(f"train leaks into/after score week t={fold.t}")

        builder = LaneWeekFeatureBuilder(diesel_features=diesel_features)
        builder.fit(fold.train)

        combined = pd.concat([fold.train, fold.score], axis=0)
        features = builder.transform(combined)
        x_train = features.loc[fold.train.index]
        x_score = features.loc[fold.score.index]
        y_train_rpm = fold.train[TARGET_COLUMN].astype(float)
        y_base_train = x_train["rpm_lag_1"].astype(float)
        y_base = x_score["rpm_lag_1"].to_numpy(dtype=float)

        if residual_target:
            y_train = (y_train_rpm - y_base_train).astype(float)
            estimator.fit(x_train, y_train)
            delta_hat = np.asarray(estimator.predict(x_score), dtype=float)
            y_pred = y_base + delta_hat
        else:
            estimator.fit(x_train, y_train_rpm)
            y_pred = np.asarray(estimator.predict(x_score), dtype=float)
            delta_hat = y_pred - y_base

        y_true = fold.score[TARGET_COLUMN].astype(float).to_numpy()
        lane_hist = x_score["lane_history_n"].to_numpy(dtype=float)

        score = fold.score.reset_index(drop=True)
        if "lane_id" in fold.score.columns:
            lane_ids = fold.score["lane_id"].astype(str).to_numpy()
        else:
            lane_ids = (
                fold.score["origin"].astype(str) + " -> " + fold.score["destination"].astype(str)
            ).to_numpy()

        fold_frame = pd.DataFrame(
            {
                "date": fold.t,
                "lane_id": lane_ids,
                "origin": fold.score["origin"].to_numpy(),
                "destination": fold.score["destination"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_baseline": y_base,
                "delta_hat": delta_hat,
                "lane_history_n": lane_hist,
                "residual": y_true - y_pred,
            }
        )
        rows.extend(fold_frame.to_dict(orient="records"))

        if verbose:
            fold_mae = float(np.mean(np.abs(y_true - y_pred)))
            print(
                f"fold {i + 1}: t={fold.t.date()}  "
                f"train={len(fold.train)}  score={len(score)}  mae={fold_mae:.4f}"
            )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "date",
                "lane_id",
                "origin",
                "destination",
                "y_true",
                "y_pred",
                "y_pred_baseline",
                "delta_hat",
                "lane_history_n",
                "residual",
                "history_bucket",
            ]
        )
        summaries = summarize_predictions(empty)
        return WalkForwardResult(
            predictions=empty,
            overall=summaries["overall"],
            by_week=summaries["by_week"],
            by_history=summaries["by_history"],
            diesel_features=diesel_features,
        )

    predictions = pd.DataFrame(rows)
    predictions["history_bucket"] = assign_history_bucket(predictions["lane_history_n"])
    summaries = summarize_predictions(predictions)
    return WalkForwardResult(
        predictions=predictions,
        overall=summaries["overall"],
        by_week=summaries["by_week"],
        by_history=summaries["by_history"],
        diesel_features=diesel_features,
    )


def save_walkforward_outputs(
    result: WalkForwardResult,
    output_dir: Path | str,
    *,
    prefix: str = "walkforward_gbm",
) -> dict[str, Path]:
    """Write predictions parquet and metric CSVs under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out / f"{prefix}_predictions.parquet",
        "overall": out / f"{prefix}_metrics_overall.csv",
        "by_week": out / f"{prefix}_metrics_by_week.csv",
        "by_history": out / f"{prefix}_metrics_by_history.csv",
    }
    result.predictions.to_parquet(paths["predictions"], index=False)
    result.overall.to_csv(paths["overall"], index=False)
    result.by_week.to_csv(paths["by_week"], index=False)
    result.by_history.to_csv(paths["by_history"], index=False)
    return paths
