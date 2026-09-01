"""Interpretable seller-acceptance probability baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ._common import build_preprocessor, require_columns, validate_feature_groups


class SellerAcceptanceModel(BaseEstimator):
    """Regularized logistic model for the probability a seller accepts an offer."""

    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str] = (),
        *,
        regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.numeric_features, self.categorical_features = validate_feature_groups(
            numeric_features, categorical_features
        )
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = float(regularization)
        self.random_state = int(random_state)

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def fit(
        self,
        frame: pd.DataFrame,
        accepted: Sequence[int] | np.ndarray | pd.Series,
    ) -> SellerAcceptanceModel:
        require_columns(frame, self.features)
        outcome = np.asarray(accepted, dtype=int)
        if outcome.ndim != 1 or len(outcome) != len(frame):
            raise ValueError("accepted must have one value per row")
        if not set(np.unique(outcome)).issubset({0, 1}) or len(np.unique(outcome)) < 2:
            raise ValueError("accepted must contain both binary outcome classes")

        self.pipeline_ = Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(self.numeric_features, self.categorical_features),
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
        self.pipeline_.fit(frame, outcome)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["pipeline_"])
        require_columns(frame, self.features)
        return self.pipeline_.predict_proba(frame)[:, 1]

    def predict(self, frame: pd.DataFrame, *, threshold: float = 0.5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one")
        return (self.predict_proba(frame) >= threshold).astype(int)

    def coefficient_table(self) -> pd.DataFrame:
        """Return logit coefficients on the transformed feature scale."""

        check_is_fitted(self, ["pipeline_"])
        preprocessor = self.pipeline_.named_steps["preprocess"]
        logistic = self.pipeline_.named_steps["logistic"]
        return pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "coefficient": logistic.coef_[0],
            }
        ).sort_values("coefficient", ascending=False, ignore_index=True)

    def offer_curve(
        self,
        frame: pd.DataFrame,
        offer_ratios: Sequence[float],
        *,
        offer_feature: str = "offer_ratio",
    ) -> pd.DataFrame:
        """Score counterfactual offer ratios for one representative row."""

        if len(frame) != 1:
            raise ValueError("offer_curve expects exactly one representative row")
        if offer_feature not in self.features:
            raise ValueError(f"{offer_feature!r} is not a model feature")
        ratios = np.asarray(offer_ratios, dtype=float)
        if ratios.ndim != 1 or len(ratios) == 0:
            raise ValueError("offer_ratios must be a non-empty one-dimensional sequence")
        counterfactual = pd.concat([frame] * len(ratios), ignore_index=True)
        counterfactual[offer_feature] = ratios
        return pd.DataFrame(
            {
                offer_feature: ratios,
                "acceptance_probability": self.predict_proba(counterfactual),
            }
        )
