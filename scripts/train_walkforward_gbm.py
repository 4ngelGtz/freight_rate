#!/usr/bin/env python3
"""Run Case B expanding-window walk-forward GBM on the local USDA snapshot.

Usage (from repository root)::

    python scripts/train_walkforward_gbm.py --hpo-params models/nested_hpo/best_params.json
    python scripts/run_pipeline.py
    python scripts/train_walkforward_gbm.py --max-folds 5 --allow-defaults   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freight_rates.evaluation import (  # noqa: E402
    dual_regime_headline,
    format_dual_regime_report,
)
from freight_rates.ingestion import DEFAULT_RAW_DIR, load_raw_snapshot  # noqa: E402
from freight_rates.preprocessing import build_modeling_panel  # noqa: E402
from freight_rates.splits import FIRST_FORECAST_DATE, filter_model_window  # noqa: E402
from freight_rates.walkforward import (  # noqa: E402
    default_gbm,
    load_hpo_best_params,
    run_walkforward_gbm,
    save_walkforward_outputs,
)

DEFAULT_HPO_PARAMS = _ROOT / "models" / "nested_hpo" / "best_params.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward HistGradientBoostingRegressor (Case B, diesel level)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_ROOT / DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "models" / "walkforward_gbm",
    )
    parser.add_argument(
        "--hpo-params",
        type=Path,
        default=DEFAULT_HPO_PARAMS,
        help="JSON with tuned GBM hyperparameters (from run_nested_hpo_gbm.py).",
    )
    parser.add_argument(
        "--allow-defaults",
        action="store_true",
        help="Use default_gbm() when --hpo-params is missing (smoke only).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on forecast weeks (debug / smoke).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model, _ = _resolve_model(args.hpo_params, allow_defaults=args.allow_defaults)

    print("Loading modeling panel...")
    raw = load_raw_snapshot(raw_dir=args.raw_dir)
    panel = filter_model_window(build_modeling_panel(raw, raw_dir=args.raw_dir))
    print(
        f"panel: {len(panel):,} rows  "
        f"{panel['date'].min().date()} → {panel['date'].max().date()}  "
        f"first_forecast={pd.Timestamp(FIRST_FORECAST_DATE).date()}  "
        f"diesel=level"
    )

    print("Running expanding-window walk-forward GBM (Case B, residual target)...")
    result = run_walkforward_gbm(
        panel,
        max_folds=args.max_folds,
        model=model,
        residual_target=True,
        diesel_features="level",
        verbose=not args.quiet,
    )

    paths = save_walkforward_outputs(result, args.output_dir)
    headline = dual_regime_headline(result.overall, result.by_history, label="full")
    headline_path = Path(args.output_dir) / "walkforward_gbm_metrics_dual_regime.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    headline.to_csv(headline_path, index=False)

    print()
    print(format_dual_regime_report(headline))
    print()
    print("=== Overall ===")
    print(result.overall.to_string(index=False))
    print()
    print("=== By history bucket ===")
    print(result.by_history.to_string(index=False))
    print()
    print(f"predictions: {paths['predictions']}")
    print(f"metrics:     {paths['overall']}")
    print(f"             {paths['by_week']}")
    print(f"             {paths['by_history']}")
    print(f"             {headline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
