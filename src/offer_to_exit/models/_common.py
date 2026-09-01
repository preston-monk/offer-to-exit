"""Shared preprocessing helpers for interpretable baseline models."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_feature_groups(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(numeric_features)
    categorical = tuple(categorical_features)
    if not numeric and not categorical:
        raise ValueError("at least one model feature is required")
    overlap = set(numeric).intersection(categorical)
    if overlap:
        raise ValueError(f"features cannot be both numeric and categorical: {overlap}")
    return numeric, categorical


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("features must be supplied as a pandas DataFrame")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"missing model columns: {sorted(missing)}")


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_features)))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_features)))
    return ColumnTransformer(transformers=transformers, remainder="drop")
