#!/usr/bin/env python3
"""Score one forecast week with fixed θ* (weekly operational pipeline).

Re-fits the global GBM on all history ``date < t`` using hyperparameters from
nested HPO, then predicts lane-week rows for ``date == t``.

Usage (from repository root)::

    make score
    make score-no-download
    python scripts/score_week.py --date 2025-08-26
    python scripts/score_week.py --download
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freight_rates.ingestion import DEFAULT_RAW_DIR, load_raw_snapshot  # noqa: E402
from freight_rates.preprocessing import build_modeling_panel  # noqa: E402
from freight_rates.splits import filter_model_window, resolve_forecast_date  # noqa: E402
from freight_rates.walkforward import (  # noqa: E402
    default_gbm,
    load_hpo_best_params,
    run_walkforward_gbm,
    save_walkforward_outputs,
)

DEFAULT_HPO_PARAMS = _ROOT / "models" / "nested_hpo" / "best_params.json"
DEFAULT_OUTPUT_DIR = _ROOT / "models" / "score"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one forecast week with fixed θ* (Case B, weekly re-fit)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_ROOT / DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--hpo-params",
        type=Path,
        default=DEFAULT_HPO_PARAMS,
        help="JSON with tuned GBM hyperparameters (from run_nested_hpo_gbm.py).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Forecast week-ending Tuesday (default: latest date in panel).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh data/raw/ snapshots before scoring.",
    )
    parser.add_argument(
        "--allow-defaults",
        action="store_true",
        help="Use default_gbm() when --hpo-params is missing (smoke only).",
    )
    return parser.parse_args(argv)


def _run_download(py: str) -> None:
    for script in ("download_usda_data.py", "download_diesel_data.py"):
        cmd = [py, f"scripts/{script}"]
        print("$", " ".join(cmd))
        subprocess.run(cmd, cwd=_ROOT, check=True)


def _resolve_model(hpo_path: Path, *, allow_defaults: bool):
    if hpo_path.exists():
        params = load_hpo_best_params(hpo_path)
        print(f"Using θ* from {hpo_path}: {params}")
        return default_gbm(**params), params
    if allow_defaults:
        print(f"No {hpo_path}; using default_gbm() (--allow-defaults).")
        return default_gbm(), None
    raise FileNotFoundError(
        f"HPO params not found: {hpo_path}\n"
        "Run: python scripts/run_nested_hpo_gbm.py\n"
        "Or pass --allow-defaults for smoke tests."
    )


def _write_latest_metadata(
    output_dir: Path,
    *,
    forecast_date: pd.Timestamp,
    hpo_params_path: Path,
    predictions_path: Path,
    n_predictions: int,
) -> Path:
    payload = {
        "forecast_date": str(forecast_date.date()),
        "created_at": datetime.now(UTC).isoformat(),
        "hpo_params_path": str(hpo_params_path),
        "predictions_path": predictions_path.name,
        "n_predictions": n_predictions,
    }
    path = output_dir / "latest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.download:
        _run_download(sys.executable)

    model, params = _resolve_model(args.hpo_params, allow_defaults=args.allow_defaults)

    print("Loading modeling panel...")
    raw = load_raw_snapshot(raw_dir=args.raw_dir)
    panel = filter_model_window(build_modeling_panel(raw, raw_dir=args.raw_dir))
    forecast_date = resolve_forecast_date(panel, forecast_date=args.date)
    print(
        f"panel: {len(panel):,} rows  "
        f"{panel['date'].min().date()} → {panel['date'].max().date()}  "
        f"forecast_date={forecast_date.date()}"
    )

    print(f"Scoring week {forecast_date.date()} (train date < t, fixed θ*)...")
    result = run_walkforward_gbm(
        panel,
        first_forecast_date=forecast_date,
        last_forecast_date=forecast_date,
        model=model,
        residual_target=True,
        diesel_features="level",
        verbose=False,
    )

    if result.predictions.empty:
        raise SystemExit(f"No score rows for forecast date {forecast_date.date()}")

    date_str = forecast_date.strftime("%Y-%m-%d")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = save_walkforward_outputs(result, args.output_dir, prefix=f"score_{date_str}")

    latest_predictions = args.output_dir / "latest_predictions.parquet"
    shutil.copy2(paths["predictions"], latest_predictions)
    latest_meta = _write_latest_metadata(
        args.output_dir,
        forecast_date=forecast_date,
        hpo_params_path=args.hpo_params,
        predictions_path=paths["predictions"],
        n_predictions=len(result.predictions),
    )

    print()
    print(f"forecast_date: {forecast_date.date()}")
    print(f"predictions:   {paths['predictions']} ({len(result.predictions):,} rows)")
    print(f"latest:        {latest_predictions}")
    print(f"metadata:      {latest_meta}")
    if params is not None:
        print(f"θ*:            {params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
