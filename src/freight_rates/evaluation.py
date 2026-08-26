"""Evaluation metrics and reporting for walk-forward freight-rate forecasts."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

HISTORY_BUCKETS: Final[tuple[str, ...]] = ("0-4", "5-19", "20-99", "100+")


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean absolute error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def medae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Median absolute error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs(yt[mask] - yp[mask])))


def mape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    eps: float = 1e-8,
) -> float:
    """Mean absolute percentage error; rows with ``|y_true| < eps`` are skipped."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) >= eps)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def assign_history_bucket(lane_history_n: pd.Series | np.ndarray) -> pd.Series:
    """Bucket lane history counts into cold-start / sparse / mature bands."""
    n = pd.Series(lane_history_n, dtype=float)
    out = pd.Series(pd.NA, index=n.index, dtype="object")
    out[(n >= 0) & (n <= 4)] = "0-4"
    out[(n >= 5) & (n <= 19)] = "5-19"
    out[(n >= 20) & (n <= 99)] = "20-99"
    out[n >= 100] = "100+"
    return out


def summarize_predictions(
    preds: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    baseline_col: str = "y_pred_baseline",
) -> dict[str, pd.DataFrame]:
    """Build overall / by-week / by-history-bucket metric tables.

    Expects columns including ``date``, ``lane_history_n``, and prediction cols.
    Returns a dict with keys ``overall``, ``by_week``, ``by_history``.
    """
    work = preds.copy()
    if "history_bucket" not in work.columns:
        work["history_bucket"] = assign_history_bucket(work["lane_history_n"])

    def _row(yt: pd.Series, yp: pd.Series, yb: pd.Series | None) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "n": int(yt.notna().sum()),
            "mae": mae(yt, yp),
            "medae": medae(yt, yp),
            "mape": mape(yt, yp),
        }
        if yb is not None:
            row["mae_baseline"] = mae(yt, yb)
            row["medae_baseline"] = medae(yt, yb)
            row["mae_lift"] = float(row["mae_baseline"] - row["mae"])
        return row

    has_baseline = baseline_col in work.columns
    yt_all = work[y_true_col]
    yp_all = work[y_pred_col]
    yb_all = work[baseline_col] if has_baseline else None

    overall = pd.DataFrame([_row(yt_all, yp_all, yb_all)])

    by_week_rows = []
    for date, g in work.groupby("date", sort=True):
        r = _row(
            g[y_true_col],
            g[y_pred_col],
            g[baseline_col] if has_baseline else None,
        )
        r["date"] = date
        by_week_rows.append(r)
    by_week = pd.DataFrame(by_week_rows)
    if not by_week.empty:
        by_week = by_week[
            ["date", "n", "mae", "medae", "mape"]
            + (["mae_baseline", "medae_baseline", "mae_lift"] if has_baseline else [])
        ]

    by_hist_rows = []
    for bucket in HISTORY_BUCKETS:
        g = work.loc[work["history_bucket"] == bucket]
        if g.empty:
            continue
        r = _row(
            g[y_true_col],
            g[y_pred_col],
            g[baseline_col] if has_baseline else None,
        )
        r["history_bucket"] = bucket
        by_hist_rows.append(r)
    by_history = pd.DataFrame(by_hist_rows)
    if not by_history.empty:
        cols = ["history_bucket", "n", "mae", "medae", "mape"]
        if has_baseline:
            cols += ["mae_baseline", "medae_baseline", "mae_lift"]
        by_history = by_history[cols]

    return {"overall": overall, "by_week": by_week, "by_history": by_history}
