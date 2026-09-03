"""Parcel-safe valuation estimation and geographic transport evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from offer_to_exit.models._common import require_columns, validate_feature_groups

from .panels import (
    ORLANDO_MARKET,
    TAMPA_MARKET,
    ChronologicalGroupedSplit,
    GeographicSplit,
    chronological_grouped_split,
    geographic_holdout,
)

OUTCOME_DERIVED_FIELDS = frozenset(
    {
        "sale_price",
        "adjusted_sale_price",
        "nominal_price_per_sqft",
        "resale_price",
        "gross_spread",
        "gross_return",
    }
)

DEFAULT_VALUATION_NUMERIC_FEATURES = (
    "log_prior_relative_to_market",
    "years_since_prior_sale",
    "sale_quarter",
)

DEFAULT_VALUATION_CATEGORICAL_FEATURES = ("property_type_code",)


@dataclass(frozen=True)
class ValuationMetrics:
    """Prediction metrics in which each parcel receives equal total weight."""

    n_transactions: int
    n_parcels: int
    mean_absolute_error: float
    root_mean_squared_error: float
    median_absolute_error: float
    mean_absolute_percentage_error: float
    median_absolute_percentage_error: float
    mean_error: float
    weighted_r_squared: float
    interval_coverage: float | None = None
    median_interval_width: float | None = None


@dataclass(frozen=True)
class GeographicValuationStudy:
    """Fitted Tampa model and Orlando external-market evaluation."""

    model: GroupedConformalValuation
    split: GeographicSplit
    tampa_temporal_split: ChronologicalGroupedSplit
    tampa_calibration_metrics: ValuationMetrics
    tampa_out_of_time_metrics: ValuationMetrics
    orlando_holdout_metrics: ValuationMetrics
    tampa_out_of_time_baseline_metrics: ValuationMetrics
    orlando_holdout_baseline_metrics: ValuationMetrics
    tampa_calibration_predictions: pd.DataFrame
    tampa_out_of_time_predictions: pd.DataFrame
    orlando_holdout_predictions: pd.DataFrame

    def metrics_payload(self) -> dict[str, object]:
        """Return the release metrics without transaction-level predictions."""

        return {
            "design": self.tampa_temporal_split.audit_summary(),
            "tampa_calibration": asdict(self.tampa_calibration_metrics),
            "tampa_out_of_time": asdict(self.tampa_out_of_time_metrics),
            "orlando_common_window": asdict(self.orlando_holdout_metrics),
            "tampa_rolled_prior_baseline": asdict(self.tampa_out_of_time_baseline_metrics),
            "orlando_rolled_prior_baseline": asdict(self.orlando_holdout_baseline_metrics),
        }


class GroupedConformalValuation:
    """Nonlinear valuation model with a parcel-grouped conformal interval.

    A gradient-boosted tree model captures nonlinearities and interactions in
    the supplied ex-ante features.  The conformal residual sample is separated
    from the proper training sample by parcel, so repeat sales of the same home
    cannot cross the model-development/calibration boundary.  The Florida
    transport benchmark uses only fields shared by both county deed systems.
    Prices are expressed relative to the lagged local market median, which
    removes level differences before transporting the Tampa relationship.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        *,
        alpha: float = 0.10,
        calibration_fraction: float = 0.20,
        log_target: bool = True,
        target_scale_feature: str | None = None,
        learning_rate: float = 0.06,
        max_iter: int = 180,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.numeric_features, self.categorical_features = validate_feature_groups(
            numeric_features, categorical_features
        )
        invalid = OUTCOME_DERIVED_FIELDS.intersection(self.features)
        if invalid:
            raise ValueError(f"valuation features contain sale outcomes: {sorted(invalid)}")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if not 0 < calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must lie between zero and 0.5")
        if learning_rate <= 0 or max_iter < 1 or max_leaf_nodes < 2 or l2_regularization < 0:
            raise ValueError("gradient-boosting parameters are outside their supported ranges")
        self.alpha = float(alpha)
        self.calibration_fraction = float(calibration_fraction)
        self.log_target = bool(log_target)
        self.target_scale_feature = target_scale_feature
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.l2_regularization = float(l2_regularization)
        self.random_state = int(random_state)

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def _new_pipeline(self) -> Pipeline:
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if self.numeric_features:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="median", keep_empty_features=True),
                            )
                        ]
                    ),
                    list(self.numeric_features),
                )
            )
        if self.categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "one_hot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                    min_frequency=5,
                                ),
                            ),
                        ]
                    ),
                    list(self.categorical_features),
                )
            )
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        regression = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        return Pipeline([("preprocess", preprocessor), ("regression", regression)])

    def fit(
        self,
        frame: pd.DataFrame,
        target: Sequence[float] | np.ndarray | pd.Series,
        *,
        groups: Sequence[str] | np.ndarray | pd.Series,
    ) -> GroupedConformalValuation:
        require_columns(frame, self.features)
        target_array, group_array = self._validate_fit_arrays(frame, target, groups)
        unique_groups = np.unique(group_array)
        if len(frame) < 30 or len(unique_groups) < 15:
            raise ValueError("at least 30 rows and 15 parcels are required")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
        )
        proper_positions, calibration_positions = next(
            splitter.split(frame, target_array, groups=group_array)
        )
        if set(group_array[proper_positions]).intersection(group_array[calibration_positions]):
            raise RuntimeError("parcel-grouped split unexpectedly leaked a parcel")

        self._fit_preseparated(
            frame.iloc[proper_positions],
            target_array[proper_positions],
            group_array[proper_positions],
            frame.iloc[calibration_positions],
            target_array[calibration_positions],
            group_array[calibration_positions],
        )
        self.proper_training_positions_ = proper_positions
        self.calibration_positions_ = calibration_positions
        return self

    def fit_with_calibration(
        self,
        training_frame: pd.DataFrame,
        training_target: Sequence[float] | np.ndarray | pd.Series,
        *,
        training_groups: Sequence[str] | np.ndarray | pd.Series,
        calibration_frame: pd.DataFrame,
        calibration_target: Sequence[float] | np.ndarray | pd.Series,
        calibration_groups: Sequence[str] | np.ndarray | pd.Series,
    ) -> GroupedConformalValuation:
        """Fit on an earlier period and conformally calibrate on the next period."""

        require_columns(training_frame, self.features)
        require_columns(calibration_frame, self.features)
        training_y, training_group_array = self._validate_fit_arrays(
            training_frame, training_target, training_groups
        )
        calibration_y, calibration_group_array = self._validate_fit_arrays(
            calibration_frame, calibration_target, calibration_groups
        )
        if len(training_frame) < 30 or len(np.unique(training_group_array)) < 15:
            raise ValueError("at least 30 training rows and 15 training parcels are required")
        if len(calibration_frame) < 10 or len(np.unique(calibration_group_array)) < 5:
            raise ValueError("at least 10 calibration rows and 5 calibration parcels are required")
        if set(training_group_array).intersection(calibration_group_array):
            raise ValueError("training and calibration parcels must be disjoint")
        self._fit_preseparated(
            training_frame,
            training_y,
            training_group_array,
            calibration_frame,
            calibration_y,
            calibration_group_array,
        )
        self.proper_training_positions_ = np.arange(len(training_frame))
        self.calibration_positions_ = np.arange(len(calibration_frame))
        return self

    def _fit_preseparated(
        self,
        training_frame: pd.DataFrame,
        training_target: np.ndarray,
        training_groups: np.ndarray,
        calibration_frame: pd.DataFrame,
        calibration_target: np.ndarray,
        calibration_groups: np.ndarray,
    ) -> None:
        transformed_training = self._transform_target(training_target, training_frame)
        transformed_calibration = self._transform_target(calibration_target, calibration_frame)
        self.pipeline_ = self._new_pipeline()
        self.pipeline_.fit(training_frame, transformed_training)
        calibration_prediction = np.asarray(self.pipeline_.predict(calibration_frame), dtype=float)
        residuals = np.abs(transformed_calibration - calibration_prediction)
        # The parcel, not the transaction row, is the independent sampling
        # unit. Use one conservative score per parcel so repeat transactions do
        # not receive extra weight in conformal calibration.
        parcel_scores = (
            pd.DataFrame({"group": calibration_groups, "residual": residuals})
            .groupby("group", dropna=False, sort=False)["residual"]
            .max()
            .to_numpy(dtype=float)
        )
        quantile_level = min(
            1.0,
            np.ceil((len(parcel_scores) + 1) * (1.0 - self.alpha)) / len(parcel_scores),
        )
        self.interval_radius_ = float(np.quantile(parcel_scores, quantile_level, method="higher"))
        self.n_training_rows_ = len(training_frame)
        self.n_calibration_rows_ = len(calibration_frame)
        self.n_training_parcels_ = len(np.unique(training_groups))
        self.n_calibration_parcels_ = len(np.unique(calibration_groups))
        self.n_calibration_scores_ = len(parcel_scores)

    def _transform_target(self, target: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        scaled_target = target / self._target_scale(frame)
        transformed: np.ndarray = np.asarray(
            np.log(scaled_target) if self.log_target else scaled_target, dtype=float
        )
        return transformed

    def _target_scale(self, frame: pd.DataFrame) -> np.ndarray:
        if self.target_scale_feature is None:
            return np.ones(len(frame), dtype=float)
        require_columns(frame, (self.target_scale_feature,))
        scale: np.ndarray = np.asarray(
            frame[self.target_scale_feature].to_numpy(dtype=float), dtype=float
        )
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("target scale values must be finite and positive")
        return scale

    def _validate_fit_arrays(
        self,
        frame: pd.DataFrame,
        target: Sequence[float] | np.ndarray | pd.Series,
        groups: Sequence[str] | np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_array = np.asarray(target, dtype=float)
        group_array = np.asarray(groups)
        if target_array.ndim != 1 or len(target_array) != len(frame):
            raise ValueError("target must be one-dimensional with one value per row")
        if group_array.ndim != 1 or len(group_array) != len(frame):
            raise ValueError("groups must be one-dimensional with one value per row")
        if not np.all(np.isfinite(target_array)):
            raise ValueError("target values must be finite")
        if self.log_target and np.any(target_array <= 0):
            raise ValueError("log-target valuation requires strictly positive targets")
        return target_array, group_array

    def _raw_prediction(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        require_columns(frame, self.features)
        return np.asarray(self.pipeline_.predict(frame), dtype=float)

    def _check_fitted(self) -> None:
        if not hasattr(self, "pipeline_") or not hasattr(self, "interval_radius_"):
            raise RuntimeError("valuation model must be fit before prediction")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw_prediction(frame)
        normalized = np.exp(raw) if self.log_target else raw
        prediction: np.ndarray = np.asarray(normalized * self._target_scale(frame), dtype=float)
        return prediction

    def predict_interval(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw_prediction(frame)
        lower = raw - self.interval_radius_
        upper = raw + self.interval_radius_
        if self.log_target:
            lower, upper = np.exp(lower), np.exp(upper)
        scale = self._target_scale(frame)
        lower, upper = lower * scale, upper * scale
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


def parcel_weighted_valuation_metrics(
    frame: pd.DataFrame,
    *,
    actual_col: str = "sale_price",
    prediction_col: str = "prediction",
    parcel_col: str = "parcel_id",
    lower_col: str | None = "lower",
    upper_col: str | None = "upper",
) -> ValuationMetrics:
    """Score transactions while giving each unique parcel equal total weight."""

    required = [actual_col, prediction_col, parcel_col]
    if (lower_col is None) != (upper_col is None):
        raise ValueError("lower_col and upper_col must both be supplied or both be omitted")
    if lower_col is not None and upper_col is not None:
        required.extend([lower_col, upper_col])
    require_columns(frame, required)
    if frame.empty:
        raise ValueError("valuation metrics require at least one row")

    actual = frame[actual_col].to_numpy(dtype=float)
    prediction = frame[prediction_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(prediction)):
        raise ValueError("actual and predicted values must be finite")
    if np.any(actual <= 0):
        raise ValueError("actual values must be positive")
    weights = _parcel_weights(frame[parcel_col])
    error = prediction - actual
    absolute_error = np.abs(error)
    percentage_error = absolute_error / actual
    weighted_mean = float(np.sum(weights * actual))
    denominator = float(np.sum(weights * np.square(actual - weighted_mean)))
    weighted_r_squared = (
        1.0 - float(np.sum(weights * np.square(error))) / denominator
        if denominator > 0
        else float("nan")
    )

    coverage: float | None = None
    interval_width: float | None = None
    if lower_col is not None and upper_col is not None:
        lower = frame[lower_col].to_numpy(dtype=float)
        upper = frame[upper_col].to_numpy(dtype=float)
        if np.any(lower > upper):
            raise ValueError("interval lower bounds cannot exceed upper bounds")
        coverage = float(
            np.clip(np.sum(weights * ((actual >= lower) & (actual <= upper))), 0.0, 1.0)
        )
        interval_width = _weighted_quantile(upper - lower, weights, 0.5)

    return ValuationMetrics(
        n_transactions=len(frame),
        n_parcels=frame[parcel_col].nunique(dropna=False),
        mean_absolute_error=float(np.sum(weights * absolute_error)),
        root_mean_squared_error=float(np.sqrt(np.sum(weights * np.square(error)))),
        median_absolute_error=_weighted_quantile(absolute_error, weights, 0.5),
        mean_absolute_percentage_error=float(np.sum(weights * percentage_error)),
        median_absolute_percentage_error=_weighted_quantile(percentage_error, weights, 0.5),
        mean_error=float(np.sum(weights * error)),
        weighted_r_squared=weighted_r_squared,
        interval_coverage=coverage,
        median_interval_width=interval_width,
    )


def run_geographic_valuation_study(
    panel: pd.DataFrame,
    *,
    numeric_features: Sequence[str] = DEFAULT_VALUATION_NUMERIC_FEATURES,
    categorical_features: Sequence[str] = DEFAULT_VALUATION_CATEGORICAL_FEATURES,
    development_market: str = TAMPA_MARKET,
    holdout_market: str = ORLANDO_MARKET,
    target_col: str = "sale_price",
    parcel_col: str = "parcel_id",
    alpha: float = 0.10,
    random_state: int = 0,
    calibration_fraction: float = 0.20,
    test_fraction: float = 0.20,
    calibration_start: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
) -> GeographicValuationStudy:
    """Use sequential Tampa periods, then evaluate the same future window in Orlando."""

    target_scale = "lagged_market_median_price"
    baseline_col = "rolled_forward_prior_baseline"
    require_columns(panel, (target_scale, baseline_col))
    eligible_panel = panel.loc[
        pd.to_numeric(panel[target_scale], errors="coerce").gt(0)
        & pd.to_numeric(panel[baseline_col], errors="coerce").gt(0)
    ].copy()
    split = geographic_holdout(
        eligible_panel,
        development_market=development_market,
        holdout_market=holdout_market,
        parcel_col=parcel_col,
    )
    require_columns(
        split.development,
        (*numeric_features, *categorical_features, target_col, "sale_date"),
    )
    temporal = chronological_grouped_split(
        split.development,
        date_col="sale_date",
        parcel_col=parcel_col,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        calibration_start=calibration_start,
        test_start=test_start,
    )
    common_window_holdout = split.holdout.loc[
        pd.to_datetime(split.holdout["sale_date"]).ge(temporal.test_start)
    ].copy()
    split = GeographicSplit(
        development=split.development,
        holdout=common_window_holdout,
        development_market=split.development_market,
        holdout_market=split.holdout_market,
    )
    model = GroupedConformalValuation(
        numeric_features,
        categorical_features,
        alpha=alpha,
        target_scale_feature=target_scale,
        random_state=random_state,
    ).fit_with_calibration(
        temporal.proper_training,
        temporal.proper_training[target_col],
        training_groups=temporal.proper_training[parcel_col],
        calibration_frame=temporal.calibration,
        calibration_target=temporal.calibration[target_col],
        calibration_groups=temporal.calibration[parcel_col],
    )

    calibration = temporal.calibration.copy()
    tampa_test = temporal.out_of_time_test.copy()
    calibration_prediction = calibration.join(model.predict_with_interval(calibration))
    tampa_test_prediction = tampa_test.join(model.predict_with_interval(tampa_test))
    holdout_prediction = split.holdout.join(model.predict_with_interval(split.holdout))
    return GeographicValuationStudy(
        model=model,
        split=split,
        tampa_temporal_split=temporal,
        tampa_calibration_metrics=parcel_weighted_valuation_metrics(
            calibration_prediction, actual_col=target_col, parcel_col=parcel_col
        ),
        tampa_out_of_time_metrics=parcel_weighted_valuation_metrics(
            tampa_test_prediction, actual_col=target_col, parcel_col=parcel_col
        ),
        orlando_holdout_metrics=parcel_weighted_valuation_metrics(
            holdout_prediction, actual_col=target_col, parcel_col=parcel_col
        ),
        tampa_out_of_time_baseline_metrics=parcel_weighted_valuation_metrics(
            tampa_test_prediction,
            actual_col=target_col,
            prediction_col=baseline_col,
            parcel_col=parcel_col,
            lower_col=None,
            upper_col=None,
        ),
        orlando_holdout_baseline_metrics=parcel_weighted_valuation_metrics(
            holdout_prediction,
            actual_col=target_col,
            prediction_col=baseline_col,
            parcel_col=parcel_col,
            lower_col=None,
            upper_col=None,
        ),
        tampa_calibration_predictions=calibration_prediction,
        tampa_out_of_time_predictions=tampa_test_prediction,
        orlando_holdout_predictions=holdout_prediction,
    )


def _parcel_weights(parcels: pd.Series) -> np.ndarray:
    counts = parcels.groupby(parcels, dropna=False).transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    normalized: np.ndarray = np.asarray(weights / float(np.sum(weights)), dtype=float)
    return normalized


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    position = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(sorted_values[position])
