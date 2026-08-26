#!/usr/bin/env python3
"""Tune GBM hyperparameters on the validation walk-forward window.

Usage (from repository root)::

    python scripts/run_nested_hpo_gbm.py
    python scripts/run_nested_hpo_gbm.py --max-configs 2 --max-folds 3   # smoke
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

from freight_rates.hpo import run_nested_hpo  # noqa: E402
from freight_rates.ingestion import DEFAULT_RAW_DIR, load_raw_snapshot  # noqa: E402
from freight_rates.preprocessing import build_modeling_panel  # noqa: E402
from freight_rates.splits import (  # noqa: E402
    DIAGNOSTIC_VAL_END,
    DIAGNOSTIC_VAL_START,
    filter_model_window,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested HPO on val walk-forward (diesel level, Case B)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_ROOT / DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "models" / "nested_hpo",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Cap grid size (smoke).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Cap forecast weeks per config (smoke).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("Loading modeling panel...")
    raw = load_raw_snapshot(raw_dir=args.raw_dir)
    panel = filter_model_window(build_modeling_panel(raw, raw_dir=args.raw_dir))
    print(
        f"panel: {len(panel):,} rows  "
        f"{panel['date'].min().date()} → {panel['date'].max().date()}  "
        f"val={pd.Timestamp(DIAGNOSTIC_VAL_START).date()} → "
        f"{pd.Timestamp(DIAGNOSTIC_VAL_END).date()}"
    )

    result = run_nested_hpo(
        panel,
        output_dir=args.output_dir,
        val_first=DIAGNOSTIC_VAL_START,
        val_last=DIAGNOSTIC_VAL_END,
        max_configs=args.max_configs,
        max_folds=args.max_folds,
        verbose=not args.quiet,
    )

    print()
    print(f"best config: {result.best_label}")
    print(f"best_params: {args.output_dir / 'best_params.json'}")
    print(f"val summary: {args.output_dir / 'hpo_val_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
