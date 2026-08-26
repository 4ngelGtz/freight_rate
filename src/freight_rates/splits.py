"""Temporal splits and expanding-window walk-forward for lane-week panels.

Official evaluation uses :func:`iter_walk_forward` (Case B: train on
``date < t``, score ``date == t``). Fixed train/val/test labels from
:func:`assign_temporal_split` are **diagnostic only** — not the headline protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final, NamedTuple

import pandas as pd

from freight_rates.ingestion import DEFAULT_START_DATE

# Modeling window / walk-forward (aligned with notebook 03).
API_START_DATE: Final[pd.Timestamp] = pd.Timestamp(DEFAULT_START_DATE)  # 2024-07-01
FIRST_FORECAST_DATE: Final[pd.Timestamp] = pd.Timestamp("2025-01-07")

# Diagnostic fixed cutoffs only — not official eval.
DIAGNOSTIC_TRAIN_END: Final[pd.Timestamp] = pd.Timestamp("2024-12-31")
DIAGNOSTIC_VAL_START: Final[pd.Timestamp] = FIRST_FORECAST_DATE  # 2025-01-07
DIAGNOSTIC_VAL_END: Final[pd.Timestamp] = pd.Timestamp("2025-06-24")
DIAGNOSTIC_TEST_START: Final[pd.Timestamp] = pd.Timestamp("2025-07-01")


class WalkForwardFold(NamedTuple):
    """One expanding-window forecast step.

    Attributes
    ----------
    t:
        Forecast week-ending Tuesday.
    train:
        Rows with ``date < t`` (past only; no leakage from week ``t``).
    score:
        Rows with ``date == t``.
    """

    t: pd.Timestamp
    train: pd.DataFrame
    score: pd.DataFrame


def filter_model_window(
    panel: pd.DataFrame,
    *,
    start_date: pd.Timestamp | str = API_START_DATE,
) -> pd.DataFrame:
    """Keep rows with ``date >= start_date`` (default API / modeling window start)."""
    if "date" not in panel.columns:
        raise KeyError("panel must include a 'date' column")
    start = pd.Timestamp(start_date)
    dates = pd.to_datetime(panel["date"])
    return panel.loc[dates >= start].copy().reset_index(drop=True)


def assign_temporal_split(dates: pd.Series) -> pd.Series:
    """Map week-ending dates to diagnostic train/val/test labels.

    Cutoffs (illustrative only; prefer :func:`iter_walk_forward` for eval):

    - ``train``: ``date <= 2024-12-31``
    - ``val``: ``2025-01-07 <= date <= 2025-06-24``
    - ``test``: ``date >= 2025-07-01``

    Dates outside these ranges (if any) remain missing (``NA``).
    """
    d = pd.to_datetime(dates)
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    out[d <= DIAGNOSTIC_TRAIN_END] = "train"
    out[(d >= DIAGNOSTIC_VAL_START) & (d <= DIAGNOSTIC_VAL_END)] = "val"
    out[d >= DIAGNOSTIC_TEST_START] = "test"
    return out


def iter_walk_forward(
    panel: pd.DataFrame,
    *,
    first_forecast_date: pd.Timestamp | str = FIRST_FORECAST_DATE,
) -> Iterator[WalkForwardFold]:
    """Yield expanding-window folds for each forecast Tuesday ``t``.

    For each unique ``date`` in the panel with ``t >= first_forecast_date``:

    - **train**: rows with ``date < t``
    - **score**: rows with ``date == t``

    Skips ``t`` when the score set is empty. Assumes ``panel`` is already
    filtered to the modeling window (see :func:`filter_model_window`).
    """
    if "date" not in panel.columns:
        raise KeyError("panel must include a 'date' column")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    first_t = pd.Timestamp(first_forecast_date)

    forecast_dates = sorted(d for d in work["date"].dropna().unique() if d >= first_t)

    for t in forecast_dates:
        t = pd.Timestamp(t)
        score = work.loc[work["date"] == t]
        if score.empty:
            continue
        train = work.loc[work["date"] < t]
        yield WalkForwardFold(t=t, train=train, score=score)
