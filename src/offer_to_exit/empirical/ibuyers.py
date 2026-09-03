"""Descriptive and predictive evidence from named iBuyer inventory episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from offer_to_exit.models._common import require_columns, validate_feature_groups
from offer_to_exit.models.survival import expand_discrete_time_rows

from .panels import (
    ORLANDO_MARKET,
    TAMPA_MARKET,
    ChronologicalGroupedSplit,
    GeographicSplit,
    chronological_grouped_split,
    geographic_holdout,
)

EXIT_OUTCOME_FIELDS = frozenset(
    {
        "resale_date",
        "resale_price",
        "gross_spread",
        "gross_return",
        "hold_days",
        "duration_weeks",
        "event_observed",
    }
)

DEFAULT_EXIT_NUMERIC_FEATURES = (
    "log_acquisition_price",
    "acquisition_year",
    "acquisition_quarter",
)

DEFAULT_EXIT_CATEGORICAL_FEATURES: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExitHazardMetrics:
    """Person-period scoring metrics for a discrete-time exit hazard."""

    n_episodes: int
    n_parcels: int
    n_risk_rows: int
    n_events_within_horizon: int
    person_period_brier_score: float
    constant_hazard_brier_score: float
    brier_skill_score: float
    person_period_log_loss: float
    observed_person_period_hazard: float
    mean_predicted_person_period_hazard: float
    risk_row_auc: float


@dataclass(frozen=True)
class GeographicExitStudy:
    """Associational Tampa duration model and its Orlando transport test."""

    model: NamedIBuyerExitHazard
    split: GeographicSplit
    tampa_temporal_split: ChronologicalGroupedSplit
    tampa_training_metrics: ExitHazardMetrics
    tampa_out_of_time_metrics: ExitHazardMetrics
    orlando_holdout_metrics: ExitHazardMetrics
    episode_summary: pd.DataFrame
    training_followup_cutoff: pd.Timestamp
    n_training_spells_truncated: int
    n_training_spells_dropped_without_complete_week: int
    latest_training_observed_exit: pd.Timestamp | None

    def metrics_payload(self) -> dict[str, object]:
        """Return JSON-ready duration metrics and split audit information."""

        design = self.tampa_temporal_split.audit_summary()
        design.update(
            {
                "training_followup_cutoff": self.training_followup_cutoff.isoformat(),
                "training_spells_truncated_at_cutoff": self.n_training_spells_truncated,
                "training_spells_dropped_without_complete_week": (
                    self.n_training_spells_dropped_without_complete_week
                ),
                "latest_observed_training_exit": (
                    self.latest_training_observed_exit.isoformat()
                    if self.latest_training_observed_exit is not None
                    else None
                ),
            }
        )
        return {
            "design": design,
            "tampa_training": asdict(self.tampa_training_metrics),
            "tampa_out_of_time": asdict(self.tampa_out_of_time_metrics),
            "orlando_common_window": asdict(self.orlando_holdout_metrics),
        }


class NamedIBuyerExitHazard:
    """Penalized discrete-time logit with week and operator indicators.

    Operator indicators are descriptive fixed effects, regularized toward zero
    for stability.  They summarize conditional associations in observed deed
    episodes and do not identify a causal effect of operator behavior or list
    price on time to exit.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        *,
        operator_col: str = "operator",
        max_weeks: int = 156,
        regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        numeric, categorical = validate_feature_groups(numeric_features, categorical_features)
        if operator_col in numeric:
            raise ValueError("operator must be categorical")
        self.numeric_features = numeric
        self.categorical_features = tuple(
            feature for feature in categorical if feature != operator_col
        )
        invalid = EXIT_OUTCOME_FIELDS.intersection(
            (*self.numeric_features, *self.categorical_features)
        )
        if invalid:
            raise ValueError(f"exit-hazard features contain realized outcomes: {sorted(invalid)}")
        if not operator_col:
            raise ValueError("operator_col must be non-empty")
        if max_weeks < 1:
            raise ValueError("max_weeks must be positive")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.operator_col = operator_col
        self.max_weeks = int(max_weeks)
        self.regularization = float(regularization)
        self.random_state = int(random_state)

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features + (self.operator_col,)

    def _new_pipeline(self) -> Pipeline:
        categorical = (*self.categorical_features, self.operator_col, "_period")
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
                            ),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    list(self.numeric_features),
                )
            )
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", drop="first"),
                        ),
                    ]
                ),
                list(categorical),
            )
        )
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        logistic = LogisticRegression(
            C=1.0 / self.regularization,
            max_iter=1_000,
            random_state=self.random_state,
        )
        return Pipeline([("preprocess", preprocessor), ("logistic", logistic)])

    def fit(
        self,
        episodes: pd.DataFrame,
        *,
        parcel_col: str = "parcel_id",
        duration_col: str = "duration_weeks",
        event_col: str = "event_observed",
    ) -> NamedIBuyerExitHazard:
        require_columns(episodes, (*self.features, parcel_col, duration_col, event_col))
        panel = expand_discrete_time_rows(
            episodes,
            duration_col=duration_col,
            event_col=event_col,
            max_periods=self.max_weeks,
        )
        if panel["_event"].nunique() < 2:
            raise ValueError("exit-hazard data must contain event and non-event risk rows")
        self.pipeline_ = self._new_pipeline()
        self.pipeline_.fit(panel, panel["_event"])
        self.n_episodes_ = len(episodes)
        self.n_risk_rows_ = len(panel)
        self.n_events_within_horizon_ = int(panel["_event"].sum())
        self.training_hazard_rate_ = float(panel["_event"].mean())
        self.parcel_col_ = parcel_col
        self.duration_col_ = duration_col
        self.event_col_ = event_col
        return self

    def predict_hazard_panel(self, expanded_panel: pd.DataFrame) -> np.ndarray:
        """Predict hazards for rows produced by ``expand_discrete_time_rows``."""

        self._check_fitted()
        require_columns(expanded_panel, (*self.features, "_period"))
        return np.asarray(self.pipeline_.predict_proba(expanded_panel)[:, 1], dtype=float)

    def predict_hazard(self, episodes: pd.DataFrame, *, horizon: int | None = None) -> np.ndarray:
        """Return conditional weekly exit probabilities with shape ``(n, horizon)``."""

        self._check_fitted()
        require_columns(episodes, self.features)
        prediction_horizon = self.max_weeks if horizon is None else int(horizon)
        if not 1 <= prediction_horizon <= self.max_weeks:
            raise ValueError("horizon must be between one and max_weeks")
        blocks: list[pd.DataFrame] = []
        for week in range(1, prediction_horizon + 1):
            block = episodes.loc[:, self.features].copy()
            block["_period"] = f"period_{week:03d}"
            blocks.append(block)
        prediction_frame = pd.concat(blocks, ignore_index=True)
        flat = self.predict_hazard_panel(prediction_frame)
        return flat.reshape(prediction_horizon, len(episodes)).T

    def predict_exit_probability(
        self, episodes: pd.DataFrame, *, horizon: int | None = None
    ) -> np.ndarray:
        hazard = self.predict_hazard(episodes, horizon=horizon)
        return 1.0 - np.prod(1.0 - hazard, axis=1)

    def coefficient_table(self) -> pd.DataFrame:
        """Return penalized log-hazard coefficients on the transformed scale."""

        self._check_fitted()
        preprocessor = self.pipeline_.named_steps["preprocess"]
        logistic = self.pipeline_.named_steps["logistic"]
        return pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "log_hazard_coefficient": np.asarray(logistic.coef_[0], dtype=float),
            }
        ).sort_values("log_hazard_coefficient", ascending=False, ignore_index=True)

    def operator_effects(self) -> pd.DataFrame:
        """Return operator log-odds differences and odds ratios from the model."""

        self._check_fitted()
        preprocessor = self.pipeline_.named_steps["preprocess"]
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["one_hot"]
        categorical = (*self.categorical_features, self.operator_col, "_period")
        operator_position = categorical.index(self.operator_col)
        categories = [str(value) for value in encoder.categories_[operator_position]]
        drop_position = int(encoder.drop_idx_[operator_position])
        reference = categories[drop_position]
        coefficients = self.coefficient_table().set_index("feature")["log_hazard_coefficient"]
        rows: list[dict[str, object]] = []
        for operator in categories:
            feature = f"categorical__{self.operator_col}_{operator}"
            effect = 0.0 if operator == reference else float(coefficients.loc[feature])
            rows.append(
                {
                    "operator": operator,
                    "reference_operator": operator == reference,
                    "log_hazard_difference": effect,
                    "hazard_odds_ratio": float(np.exp(effect)),
                }
            )
        return pd.DataFrame(rows).sort_values("operator", ignore_index=True)

    def _check_fitted(self) -> None:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("exit-hazard model must be fit before prediction")


def summarize_ibuyer_episodes(
    episodes: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("market", "operator"),
    duration_col: str = "hold_days",
    event_col: str = "event_observed",
) -> pd.DataFrame:
    """Summarize observed inventory duration and pre-cost gross spreads.

    Kaplan-Meier exit rates account for right censoring.  Gross spreads remain
    unadjusted for renovation, financing, taxes, maintenance, transaction fees,
    and any bundled service fee; they must not be interpreted as profit.
    """

    require_columns(
        episodes,
        (*group_cols, "parcel_id", duration_col, event_col, "acquisition_date"),
    )
    rows: list[dict[str, object]] = []
    grouping_key: str | list[str] = list(group_cols)
    if len(group_cols) == 1:
        grouping_key = group_cols[0]
    for raw_key, group in episodes.groupby(grouping_key, dropna=False, sort=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        duration = group[duration_col].to_numpy(dtype=float)
        event = group[event_col].to_numpy(dtype=int)
        if np.any(duration < 0) or not set(np.unique(event)).issubset({0, 1}):
            raise ValueError("episode durations and event indicators are invalid")
        realized = group.loc[group[event_col].eq(1)]
        row: dict[str, object] = dict(zip(group_cols, key, strict=True))
        row.update(
            {
                "n_acquisitions": len(group),
                "n_unique_parcels": group["parcel_id"].nunique(dropna=False),
                "n_observed_exits": int(event.sum()),
                "observed_exit_share": float(event.mean()),
                "acquisition_start": pd.to_datetime(group["acquisition_date"]).min(),
                "acquisition_end": pd.to_datetime(group["acquisition_date"]).max(),
                "median_observed_hold_days": _safe_median(realized[duration_col]),
                "km_median_exit_days": _kaplan_meier_median(duration, event),
                "km_exit_probability_90d": _kaplan_meier_exit_probability(duration, event, 90),
                "km_exit_probability_120d": _kaplan_meier_exit_probability(duration, event, 120),
                "median_gross_spread": _safe_median(realized.get("gross_spread")),
                "median_gross_return": _safe_median(realized.get("gross_return")),
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("episode summaries require at least one row")
    return pd.DataFrame(rows)


def person_period_exit_hazard_metrics(
    model: NamedIBuyerExitHazard,
    episodes: pd.DataFrame,
    *,
    parcel_col: str = "parcel_id",
    duration_col: str = "duration_weeks",
    event_col: str = "event_observed",
) -> ExitHazardMetrics:
    """Evaluate the ordinary person-period hazard likelihood and predictions."""

    require_columns(episodes, (parcel_col, duration_col, event_col, *model.features))
    if episodes.empty:
        raise ValueError("exit-hazard metrics require at least one episode")
    panel = expand_discrete_time_rows(
        episodes,
        duration_col=duration_col,
        event_col=event_col,
        max_periods=model.max_weeks,
    )
    probability = np.clip(model.predict_hazard_panel(panel), 1e-9, 1 - 1e-9)
    outcome = panel["_event"].to_numpy(dtype=float)
    brier = float(np.mean(np.square(probability - outcome)))
    baseline_brier = float(np.mean(np.square(model.training_hazard_rate_ - outcome)))
    log_loss = float(
        -np.mean(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability))
    )
    auc = (
        float(roc_auc_score(outcome, probability)) if len(np.unique(outcome)) == 2 else float("nan")
    )
    return ExitHazardMetrics(
        n_episodes=len(episodes),
        n_parcels=episodes[parcel_col].nunique(dropna=False),
        n_risk_rows=len(panel),
        n_events_within_horizon=int(panel["_event"].sum()),
        person_period_brier_score=brier,
        constant_hazard_brier_score=baseline_brier,
        brier_skill_score=1.0 - brier / baseline_brier if baseline_brier > 0 else float("nan"),
        person_period_log_loss=log_loss,
        observed_person_period_hazard=float(np.mean(outcome)),
        mean_predicted_person_period_hazard=float(np.mean(probability)),
        risk_row_auc=auc,
    )


def run_geographic_exit_study(
    episodes: pd.DataFrame,
    *,
    numeric_features: Sequence[str] = DEFAULT_EXIT_NUMERIC_FEATURES,
    categorical_features: Sequence[str] = DEFAULT_EXIT_CATEGORICAL_FEATURES,
    development_market: str = TAMPA_MARKET,
    holdout_market: str = ORLANDO_MARKET,
    max_weeks: int = 156,
    regularization: float = 1.0,
    random_state: int = 0,
    calibration_fraction: float = 0.20,
    test_fraction: float = 0.20,
    calibration_start: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
) -> GeographicExitStudy:
    """Estimate on past Tampa episodes and test a common future window in both markets."""

    split = geographic_holdout(
        episodes,
        development_market=development_market,
        holdout_market=holdout_market,
    )
    temporal = chronological_grouped_split(
        split.development,
        date_col="acquisition_date",
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        calibration_start=calibration_start,
        test_start=test_start,
    )
    uncensored_tampa_training = pd.concat(
        [temporal.proper_training, temporal.calibration], ignore_index=True
    )
    tampa_training, n_training_spells_truncated = administratively_censor_episode_followup(
        uncensored_tampa_training,
        cutoff=temporal.test_start,
    )
    n_training_spells_dropped_without_complete_week = len(uncensored_tampa_training) - len(
        tampa_training
    )
    common_window_holdout = split.holdout.loc[
        pd.to_datetime(split.holdout["acquisition_date"]).ge(temporal.test_start)
    ].copy()
    split = GeographicSplit(
        development=split.development,
        holdout=common_window_holdout,
        development_market=split.development_market,
        holdout_market=split.holdout_market,
    )
    model = NamedIBuyerExitHazard(
        numeric_features,
        categorical_features,
        max_weeks=max_weeks,
        regularization=regularization,
        random_state=random_state,
    ).fit(tampa_training)
    observed_training = tampa_training.loc[tampa_training["event_observed"].eq(1)]
    observed_training_exits = (
        pd.to_datetime(observed_training["acquisition_date"], errors="coerce")
        + pd.to_timedelta(observed_training["hold_days"], unit="D")
    ).dropna()
    latest_training_observed_exit = (
        pd.Timestamp(observed_training_exits.max()) if len(observed_training_exits) else None
    )
    return GeographicExitStudy(
        model=model,
        split=split,
        tampa_temporal_split=temporal,
        tampa_training_metrics=person_period_exit_hazard_metrics(model, tampa_training),
        tampa_out_of_time_metrics=person_period_exit_hazard_metrics(
            model, temporal.out_of_time_test
        ),
        orlando_holdout_metrics=person_period_exit_hazard_metrics(model, split.holdout),
        episode_summary=summarize_ibuyer_episodes(episodes),
        training_followup_cutoff=temporal.test_start,
        n_training_spells_truncated=n_training_spells_truncated,
        n_training_spells_dropped_without_complete_week=(
            n_training_spells_dropped_without_complete_week
        ),
        latest_training_observed_exit=latest_training_observed_exit,
    )


def administratively_censor_episode_followup(
    episodes: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Remove post-cutoff disposition information from pre-cutoff cohorts."""

    require_columns(
        episodes,
        ("acquisition_date", "hold_days", "duration_weeks", "event_observed"),
    )
    result = episodes.copy()
    cutoff_timestamp = pd.Timestamp(cutoff)
    acquisition_date = pd.to_datetime(result["acquisition_date"], errors="coerce")
    followup_to_cutoff = (cutoff_timestamp - acquisition_date).dt.days
    if followup_to_cutoff.isna().any() or followup_to_cutoff.le(0).any():
        raise ValueError("training acquisitions must occur strictly before the follow-up cutoff")

    original_hold_days = pd.to_numeric(result["hold_days"], errors="raise")
    crosses_cutoff = original_hold_days.ge(followup_to_cutoff)
    result.loc[crosses_cutoff, "hold_days"] = followup_to_cutoff.loc[crosses_cutoff]
    result.loc[crosses_cutoff, "duration_weeks"] = np.floor(
        followup_to_cutoff.loc[crosses_cutoff] / 7
    ).astype(int)
    result.loc[crosses_cutoff, "event_observed"] = 0
    for column in ("resale_date", "resale_price", "gross_spread", "gross_return"):
        if column in result:
            result.loc[crosses_cutoff, column] = pd.NaT if column == "resale_date" else np.nan

    result = result.loc[result["event_observed"].eq(1) | result["duration_weeks"].ge(1)].copy()
    return result.reset_index(drop=True), int(crosses_cutoff.sum())


def _safe_median(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else float("nan")


def _kaplan_meier_exit_probability(
    duration: np.ndarray, event: np.ndarray, horizon: float
) -> float:
    survival = 1.0
    for time in np.sort(np.unique(duration[event == 1])):
        if time > horizon:
            break
        at_risk = int(np.sum(duration >= time))
        events_at_time = int(np.sum((duration == time) & (event == 1)))
        if at_risk:
            survival *= 1.0 - events_at_time / at_risk
    return float(1.0 - survival)


def _kaplan_meier_median(duration: np.ndarray, event: np.ndarray) -> float:
    survival = 1.0
    for time in np.sort(np.unique(duration[event == 1])):
        at_risk = int(np.sum(duration >= time))
        events_at_time = int(np.sum((duration == time) & (event == 1)))
        if at_risk:
            survival *= 1.0 - events_at_time / at_risk
        if survival <= 0.5:
            return float(time)
    return float("nan")
