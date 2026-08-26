#!/usr/bin/env python3
"""Diesel feature ablation under Case B walk-forward GBM.

Runs three residual-GBM variants on the same modeling panel:

- ``none``  — no diesel features
- ``level`` — ``diesel_us`` only (Case-B as-of)
- ``full``  — ``diesel_us`` + ``diesel_us_chg_1w``

Writes per-variant metrics under ``models/diesel_ablation/`` and a comparison
CSV focused on dual-regime headlines (overall MAE lift + cold-start 0-4 lift).

Usage (from repository root)::

    python scripts/ablate_diesel_walkforward.py
    python scripts/ablate_diesel_walkforward.py --max-folds 5   # smoke
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
from freight_rates.features import DIESEL_FEATURE_MODES  # noqa: E402
from freight_rates.ingestion import DEFAULT_RAW_DIR, load_raw_snapshot  # noqa: E402
from freight_rates.preprocessing import build_modeling_panel  # noqa: E402
from freight_rates.splits import FIRST_FORECAST_DATE, filter_model_window  # noqa: E402
from freight_rates.walkforward import (  # noqa: E402
    run_walkforward_gbm,
    save_walkforward_outputs,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate diesel features in walk-forward residual GBM (Case B)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_ROOT / DEFAULT_RAW_DIR,
        help="Directory with rates + diesel Parquet snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "models" / "diesel_ablation",
        help="Where to write per-variant metrics and the comparison CSV.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DIESEL_FEATURE_MODES),
        choices=list(DIESEL_FEATURE_MODES),
        help="Diesel modes to run (default: none level full).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on forecast weeks (smoke / debug).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-fold progress lines.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading modeling panel (rates + Case-B diesel as-of)...")
    raw = load_raw_snapshot(raw_dir=args.raw_dir)
    panel = filter_model_window(build_modeling_panel(raw, raw_dir=args.raw_dir))
    print(
        f"panel: {len(panel):,} rows  "
        f"{panel['date'].min().date()} → {panel['date'].max().date()}  "
        f"first_forecast={pd.Timestamp(FIRST_FORECAST_DATE).date()}"
    )
    print(f"modes: {args.modes}")
    print()

    headlines: list[pd.DataFrame] = []
    for mode in args.modes:
        print(f"=== Running diesel_features={mode!r} ===")
        result = run_walkforward_gbm(
            panel,
            max_folds=args.max_folds,
            residual_target=True,
            diesel_features=mode,
            verbose=not args.quiet,
        )
        prefix = f"diesel_{mode}"
        paths = save_walkforward_outputs(result, out_dir, prefix=prefix)
        headline = dual_regime_headline(
            result.overall,
            result.by_history,
            label=mode,
        )
        headlines.append(headline)
        print(format_dual_regime_report(headline))
        print(f"wrote: {paths['overall']}")
        print()

    comparison = pd.concat(headlines, ignore_index=True)
    # Delta vs no-diesel baseline when present.
    if "none" in set(comparison["label"]):
        base = comparison.loc[comparison["label"] == "none"].iloc[0]
        comparison["delta_mae_vs_none"] = comparison["mae"] - float(base["mae"])
        comparison["delta_cold_lift_vs_none"] = comparison["cold_mae_lift"] - float(
            base["cold_mae_lift"]
        )

    compare_path = out_dir / "diesel_ablation_comparison.csv"
    comparison.to_csv(compare_path, index=False)

    print("=== Ablation comparison (dual-regime) ===")
    print(comparison.to_string(index=False))
    print()
    print(f"comparison: {compare_path}")

    # Brief verdict
    if "none" in set(comparison["label"]) and "full" in set(comparison["label"]):
        full = comparison.loc[comparison["label"] == "full"].iloc[0]
        delta = float(full["delta_mae_vs_none"])
        if abs(delta) < 0.002:
            print(
                f"Verdict: |ΔMAE full vs none|={abs(delta):.4f} < 0.002 → "
                "diesel is optional for overall MAE."
            )
        elif delta < 0:
            print(f"Verdict: full diesel improves overall MAE by {-delta:.4f} vs none.")
        else:
            print(f"Verdict: full diesel worsens overall MAE by {delta:.4f} vs none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
