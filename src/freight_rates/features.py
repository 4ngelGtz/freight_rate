"""Feature engineering for lane-week freight rate modeling."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from freight_rates.preprocessing import LANE_WEEK_KEY

TARGET_COLUMN: Final[str] = "rpm"

# Allowed model inputs — excludes leakage columns handled in preprocessing.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("origin", "destination")
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "distance",
    "availability",
    "year",
    "month",
    "week",
    "quarter",
    "week_of_year",
    "lane_history_n",
    "weeks_since_first_seen",
    "rpm_lag_1",
    "rpm_rolling_4",
)
DERIVED_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "week_of_year",
    "lane_history_n",
    "weeks_since_first_seen",
    "rpm_lag_1",
    "rpm_rolling_4",
)


class LaneWeekFeatureBuilder(BaseEstimator, TransformerMixin):
    """Fit/transform feature builder for lane-week panels.

    Fits one-hot encoders for ``origin`` and ``destination``. During
    ``transform``, derives lane history and lagged RPM features using only
    past observations within each lane (sorted by ``date``).

    Lag features require ``rpm`` on historical rows. At scoring time, pass
    the panel rows to score together with prior history for each lane.

    The transformer is serializable with :func:`joblib.dump` /
    :func:`joblib.load` like other scikit-learn estimators.
    """

    def __init__(
        self,
        *,
        include_lags: bool = True,
        rolling_window: int = 4,
    ) -> None:
        self.include_lags = include_lags
        self.rolling_window = rolling_window

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LaneWeekFeatureBuilder:
        """Learn category encoders and the global RPM mean (cold-start fallback)."""
        panel = self._validate_panel(X)
        self.global_rpm_mean_ = float(panel[TARGET_COLUMN].mean())
        self._encoder = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(CATEGORICAL_FEATURES),
                )
            ],
            remainder="drop",
        )
        self._encoder.fit(panel[list(CATEGORICAL_FEATURES)])
        self.feature_names_ = self._build_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a model-ready feature matrix as a DataFrame."""
        if not hasattr(self, "_encoder"):
            raise RuntimeError("LaneWeekFeatureBuilder has not been fitted yet.")

        panel = self._add_history_features(self._validate_panel(X))
        numeric = panel[list(NUMERIC_FEATURES)].astype(float)
        encoded = self._encoder.transform(panel[list(CATEGORICAL_FEATURES)])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self._encoder.get_feature_names_out(),
            index=panel.index,
        )

        features = pd.concat([numeric, encoded_df], axis=1)
        features = features.reindex(columns=self.feature_names_, fill_value=0.0)
        return features

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        return self.fit(X, y=y).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("LaneWeekFeatureBuilder has not been fitted yet.")
        return np.asarray(self.feature_names_, dtype=object)

    def _build_feature_names(self) -> list[str]:
        cat_names = list(self._encoder.get_feature_names_out())
        return list(NUMERIC_FEATURES) + cat_names

    def _validate_panel(self, X: pd.DataFrame) -> pd.DataFrame:
        required = set(LANE_WEEK_KEY) | {TARGET_COLUMN, "distance", "availability"}
        required |= {"year", "month", "week", "quarter"}
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Lane-week panel is missing required columns: {missing}")

        panel = X.copy()
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        for col in ("year", "month", "week", "quarter", "distance", "availability", TARGET_COLUMN):
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
        return panel

    def _add_history_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)

        sort_cols = ["origin", "destination", "date"]
        out = out.sort_values(sort_cols, kind="mergesort")
        lane_key = ["origin", "destination"]

        out["lane_history_n"] = out.groupby(lane_key, sort=False).cumcount()
        first_seen = out.groupby(lane_key, sort=False)["date"].transform("min")
        out["weeks_since_first_seen"] = (
            (out["date"] - first_seen).dt.days / 7.0
        ).astype(float)

        if self.include_lags:
            grouped_rpm = out.groupby(lane_key, sort=False)[TARGET_COLUMN]
            out["rpm_lag_1"] = grouped_rpm.shift(1).fillna(self.global_rpm_mean_)
            out["rpm_rolling_4"] = grouped_rpm.transform(
                lambda s: s.shift(1).rolling(window=self.rolling_window, min_periods=1).mean()
            ).fillna(self.global_rpm_mean_)
        else:
            out["rpm_lag_1"] = self.global_rpm_mean_
            out["rpm_rolling_4"] = self.global_rpm_mean_

        return out.sort_index()
