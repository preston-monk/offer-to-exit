"""Transparent linear valuation baseline with split-conformal intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ._common import build_preprocessor, require_columns, validate_feature_groups


class CalibratedLinearValuation(BaseEstimator):
    """Hedonic linear baseline with a finite-sample conformal interval.

    The model is fit on a proper training split and the interval radius is selected
    from held-out absolute residuals.  With ``log_target=True`` (the default), both
    the regression and interval are on log-price scale before being transformed
    back to currency units.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        *,
        alpha: float = 0.10,
        calibration_fraction: float = 0.20,
        log_target: bool = True,
        random_state: int = 0,
    ) -> None:
        self.numeric_features, self.categorical_features = validate_feature_groups(
            numeric_features, categorical_features
        )
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if not 0 < calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must lie between zero and 0.5")
        self.alpha = float(alpha)
        self.calibration_fraction = float(calibration_fraction)
        self.log_target = bool(log_target)
        self.random_state = int(random_state)

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def _new_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(self.numeric_features, self.categorical_features),
                ),
                ("regression", LinearRegression()),
            ]
        )

    def fit(
        self,
        frame: pd.DataFrame,
        target: Sequence[float] | np.ndarray | pd.Series,
    ) -> CalibratedLinearValuation:
        require_columns(frame, self.features)
        target_array = np.asarray(target, dtype=float)
        if target_array.ndim != 1 or len(target_array) != len(frame):
            raise ValueError("target must be one-dimensional with one value per row")
        if not np.all(np.isfinite(target_array)):
            raise ValueError("target values must be finite")
        if self.log_target and np.any(target_array <= 0):
            raise ValueError("log-target valuation requires strictly positive targets")
        if len(frame) < 20:
            raise ValueError("at least 20 observations are required for calibration")

        transformed_target = np.log(target_array) if self.log_target else target_array
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(frame))
        calibration_size = max(10, int(np.ceil(len(frame) * self.calibration_fraction)))
        calibration_positions = order[:calibration_size]
        training_positions = order[calibration_size:]

        self.pipeline_ = self._new_pipeline()
        self.pipeline_.fit(frame.iloc[training_positions], transformed_target[training_positions])
        calibration_prediction = self.pipeline_.predict(frame.iloc[calibration_positions])
        residuals = np.abs(transformed_target[calibration_positions] - calibration_prediction)
        # Split-conformal finite-sample correction: ceil((n + 1)(1-alpha))/n.
        quantile_level = min(
            1.0,
            np.ceil((len(residuals) + 1) * (1.0 - self.alpha)) / len(residuals),
        )
        self.interval_radius_ = float(np.quantile(residuals, quantile_level, method="higher"))
        self.n_training_ = len(training_positions)
        self.n_calibration_ = len(calibration_positions)
        self.calibration_residuals_ = residuals
        return self

    def _raw_prediction(self, frame: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["pipeline_", "interval_radius_"])
        require_columns(frame, self.features)
        return np.asarray(self.pipeline_.predict(frame), dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw_prediction(frame)
        return np.exp(raw) if self.log_target else raw

    def predict_interval(self, frame: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, 2)`` array containing lower and upper bounds."""

        raw = self._raw_prediction(frame)
        lower = raw - self.interval_radius_
        upper = raw + self.interval_radius_
        if self.log_target:
            lower, upper = np.exp(lower), np.exp(upper)
        return np.column_stack([lower, upper])

    def predict_with_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
        prediction = self.predict(frame)
        interval = self.predict_interval(frame)
        return pd.DataFrame(
            {
                "prediction": prediction,
                "lower": interval[:, 0],
                "upper": interval[:, 1],
            },
            index=frame.index,
        )

    def coefficient_table(self) -> pd.DataFrame:
        """Return coefficients on the transformed (standardized) feature scale."""

        check_is_fitted(self, ["pipeline_"])
        preprocessor = self.pipeline_.named_steps["preprocess"]
        regression = self.pipeline_.named_steps["regression"]
        return pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "coefficient": np.asarray(regression.coef_, dtype=float),
            }
        ).sort_values("coefficient", ascending=False, ignore_index=True)
