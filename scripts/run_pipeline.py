#!/usr/bin/env python3
"""Run download → HPO → walk-forward pipeline.

Usage (from repository root)::

    make pipeline
    make pipeline-no-download
    make pipeline-smoke

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-download --smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from freight_rates.artifacts import write_run_manifest  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full GBM training and eval pipeline.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume data/raw snapshots already exist.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small HPO grid and few walk-forward folds.",
    )
    return parser.parse_args(argv)


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    py = sys.executable

    if not args.skip_download:
        _run([py, "scripts/download_usda_data.py"])
        _run([py, "scripts/download_diesel_data.py"])

    hpo_cmd = [py, "scripts/run_nested_hpo_gbm.py"]
    wf_cmd = [
        py,
        "scripts/train_walkforward_gbm.py",
        "--hpo-params",
        str(_ROOT / "models" / "nested_hpo" / "best_params.json"),
    ]
    if args.smoke:
        hpo_cmd += ["--max-configs", "2", "--max-folds", "3"]
        wf_cmd += ["--max-folds", "3"]

    _run(hpo_cmd)
    _run(wf_cmd)

    manifest = write_run_manifest(models_dir=_ROOT / "models")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
