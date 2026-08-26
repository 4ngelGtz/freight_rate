"""Paths and loaders for pipeline artifacts under ``models/``."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from freight_rates.evaluation import dual_regime_headline, summarize_predictions
from freight_rates.splits import (
    DIAGNOSTIC_TEST_START,
    DIAGNOSTIC_VAL_END,
    DIAGNOSTIC_VAL_START,
)

DEFAULT_MODELS_DIR = Path("models")
NESTED_HPO_DIR = DEFAULT_MODELS_DIR / "nested_hpo"
WALKFORWARD_DIR = DEFAULT_MODELS_DIR / "walkforward_gbm"
MANIFEST_PATH = DEFAULT_MODELS_DIR / "run_manifest.json"


@dataclass(frozen=True)
class HpoArtifacts:
    val_summary: pd.DataFrame
    best_params: pd.Series
    val_folds_best: pd.DataFrame
    best_params_path: Path


@dataclass(frozen=True)
class WalkForwardArtifacts:
    predictions: pd.DataFrame
    overall: pd.DataFrame
    by_week: pd.DataFrame
    by_history: pd.DataFrame
    dual_regime: pd.DataFrame
    output_dir: Path


def models_root(root: Path | None = None) -> Path:
    return root if root is not None else DEFAULT_MODELS_DIR


def require_hpo_artifacts(root: Path | None = None) -> HpoArtifacts:
    """Load HPO outputs; raise if missing."""
    base = models_root(root) / "nested_hpo"
    val_summary_path = base / "hpo_val_summary.csv"
    best_params_path = base / "best_params.json"
    val_folds_path = base / "hpo_val_folds_best.csv"
    missing = [p for p in (val_summary_path, best_params_path, val_folds_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "HPO artifacts missing. Run: python scripts/run_nested_hpo_gbm.py\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    return HpoArtifacts(
        val_summary=pd.read_csv(val_summary_path),
        best_params=pd.read_json(best_params_path, typ="series"),
        val_folds_best=pd.read_csv(val_folds_path, parse_dates=["date"]),
        best_params_path=best_params_path,
    )


def require_walkforward_artifacts(root: Path | None = None) -> WalkForwardArtifacts:
    """Load walk-forward outputs; raise if missing."""
    out = models_root(root) / "walkforward_gbm"
    pred_path = out / "walkforward_gbm_predictions.parquet"
    week_path = out / "walkforward_gbm_metrics_by_week.csv"
    hist_path = out / "walkforward_gbm_metrics_by_history.csv"
    overall_path = out / "walkforward_gbm_metrics_overall.csv"
    dual_path = out / "walkforward_gbm_metrics_dual_regime.csv"
    missing = [p for p in (pred_path, week_path, hist_path, overall_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Walk-forward artifacts missing. Run: python scripts/train_walkforward_gbm.py\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    preds = pd.read_parquet(pred_path)
    by_week = pd.read_csv(week_path, parse_dates=["date"])
    by_history = pd.read_csv(hist_path)
    overall = pd.read_csv(overall_path)
    if dual_path.exists():
        dual_regime = pd.read_csv(dual_path)
    else:
        dual_regime = dual_regime_headline(overall, by_history, label="full")
    return WalkForwardArtifacts(
        predictions=preds,
        overall=overall,
        by_week=by_week,
        by_history=by_history,
        dual_regime=dual_regime,
        output_dir=out,
    )


def filter_predictions_window(
    preds: pd.DataFrame,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Slice predictions by forecast ``date`` (inclusive bounds)."""
    work = preds.copy()
    work["date"] = pd.to_datetime(work["date"])
    if start is not None:
        work = work.loc[work["date"] >= pd.Timestamp(start)]
    if end is not None:
        work = work.loc[work["date"] <= pd.Timestamp(end)]
    return work


def metrics_for_window(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute metric tables for a prediction subset."""
    return summarize_predictions(preds)


def val_test_headlines(preds: pd.DataFrame) -> pd.DataFrame:
    """Dual-regime headlines for val, test, and full forecast windows."""
    rows = []
    for window, start, end in (
        ("val", DIAGNOSTIC_VAL_START, DIAGNOSTIC_VAL_END),
        ("test", DIAGNOSTIC_TEST_START, None),
        ("full", None, None),
    ):
        subset = filter_predictions_window(preds, start=start, end=end)
        if subset.empty:
            continue
        summary = summarize_predictions(subset)
        headline = dual_regime_headline(summary["overall"], summary["by_history"], label=window)
        headline["window"] = window
        rows.append(headline)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(
    *,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    best_params_path: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_manifest.json`` describing the latest pipeline run."""
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    hpo_rel = "nested_hpo/best_params.json"
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "diesel_features": "level",
        "residual_target": True,
        "hpo": {
            "val_window": [
                str(DIAGNOSTIC_VAL_START.date()),
                str(DIAGNOSTIC_VAL_END.date()),
            ],
            "test_window_from": str(DIAGNOSTIC_TEST_START.date()),
            "best_params_path": hpo_rel,
        },
        "walkforward": {
            "output_dir": "walkforward_gbm",
            "params_source": hpo_rel,
        },
    }
    if best_params_path is not None:
        payload["hpo"]["best_params_path"] = str(
            Path(best_params_path).relative_to(root)
            if Path(best_params_path).is_absolute()
            else best_params_path
        )
    if extra:
        payload.update(extra)
    path = root / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
