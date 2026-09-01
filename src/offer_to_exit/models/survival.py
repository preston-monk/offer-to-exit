"""Discrete-time logistic sale-hazard model with right censoring."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ._common import build_preprocessor, require_columns, validate_feature_groups


def expand_discrete_time_rows(
    frame: pd.DataFrame,
    *,
    duration_col: str = "listing_periods",
    event_col: str = "event_observed",
    max_periods: int | None = None,
) -> pd.DataFrame:
    """Create one binary-hazard row for each period a listing is at risk."""

    require_columns(frame, (duration_col, event_col))
    durations = frame[duration_col].to_numpy(dtype=int)
    events = frame[event_col].to_numpy(dtype=int)
    if np.any(durations < 1):
        raise ValueError("all durations must be positive")
    if not set(np.unique(events)).issubset({0, 1}):
        raise ValueError("event indicator must be binary")
    if max_periods is not None:
        if max_periods < 1:
            raise ValueError("max_periods must be positive")
        observed_durations = np.minimum(durations, max_periods)
        observable_events = events * (durations <= max_periods)
    else:
        observed_durations = durations
        observable_events = events

    positions = np.repeat(np.arange(len(frame)), observed_durations)
    periods = np.concatenate([np.arange(1, duration + 1) for duration in observed_durations])
    panel = frame.iloc[positions].reset_index(drop=True)
    panel["_period"] = [f"period_{period:03d}" for period in periods]
    panel["_period_number"] = periods
    is_last_observed = periods == np.repeat(observed_durations, observed_durations)
    panel["_event"] = (
        is_last_observed & np.repeat(observable_events.astype(bool), observed_durations)
    ).astype(int)
    return panel


class DiscreteTimeHazardModel(BaseEstimator):
    """Logistic discrete-time survival model for a listing's sale hazard."""

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        *,
        max_periods: int = 12,
        regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.numeric_features, self.categorical_features = validate_feature_groups(
            numeric_features, categorical_features
        )
        if max_periods < 1:
            raise ValueError("max_periods must be positive")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.max_periods = int(max_periods)
        self.regularization = float(regularization)
        self.random_state = int(random_state)

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        duration_col: str = "listing_periods",
        event_col: str = "event_observed",
    ) -> DiscreteTimeHazardModel:
        require_columns(frame, (*self.features, duration_col, event_col))
        panel = expand_discrete_time_rows(
            frame,
            duration_col=duration_col,
            event_col=event_col,
            max_periods=self.max_periods,
        )
        if panel["_event"].nunique() < 2:
            raise ValueError("survival training data must include an observed event")

        categorical_with_period = (*self.categorical_features, "_period")
        self.pipeline_ = Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(self.numeric_features, categorical_with_period),
                ),
                (
                    "logistic",
                    LogisticRegression(
                        C=1.0 / self.regularization,
                        max_iter=1_000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(panel, panel["_event"])
        self.n_panel_rows_ = len(panel)
        self.n_events_ = int(panel["_event"].sum())
        return self

    def predict_hazard(
        self,
        frame: pd.DataFrame,
        *,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Return conditional event probabilities with shape ``(n, horizon)``."""

        check_is_fitted(self, ["pipeline_"])
        require_columns(frame, self.features)
        prediction_horizon = self.max_periods if horizon is None else int(horizon)
        if not 1 <= prediction_horizon <= self.max_periods:
            raise ValueError("horizon must be between one and max_periods")

        period_blocks: list[pd.DataFrame] = []
        for period in range(1, prediction_horizon + 1):
            block = frame.loc[:, self.features].copy()
            block["_period"] = f"period_{period:03d}"
            period_blocks.append(block)
        prediction_frame = pd.concat(period_blocks, ignore_index=True)
        flat_hazard = self.pipeline_.predict_proba(prediction_frame)[:, 1]
        return flat_hazard.reshape(prediction_horizon, len(frame)).T

    def predict_survival(
        self,
        frame: pd.DataFrame,
        *,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Return probability of remaining unsold after each period."""

        hazard = self.predict_hazard(frame, horizon=horizon)
        return np.cumprod(1.0 - hazard, axis=1)

    def predict_sale_probability(
        self,
        frame: pd.DataFrame,
        *,
        horizon: int | None = None,
    ) -> np.ndarray:
        survival = self.predict_survival(frame, horizon=horizon)
        return 1.0 - survival[:, -1]

    def predict_median_period(self, frame: pd.DataFrame) -> np.ndarray:
        sale_cdf = 1.0 - self.predict_survival(frame)
        crosses_median = sale_cdf >= 0.5
        first_crossing = crosses_median.argmax(axis=1) + 1
        return np.where(crosses_median.any(axis=1), first_crossing, self.max_periods + 1)

    def coefficient_table(self) -> pd.DataFrame:
        """Return hazard-logit coefficients on the transformed feature scale."""

        check_is_fitted(self, ["pipeline_"])
        preprocessor = self.pipeline_.named_steps["preprocess"]
        logistic = self.pipeline_.named_steps["logistic"]
        return pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "coefficient": logistic.coef_[0],
            }
        ).sort_values("coefficient", ascending=False, ignore_index=True)
