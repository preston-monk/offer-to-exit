"""Temporal contracts that prevent future information from entering a decision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping


Timestamp = datetime | date


def as_utc(value: Timestamp) -> datetime:
    """Normalize a date or datetime to an aware UTC datetime."""

    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class AsOfFeature:
    """A feature value paired with the first instant it was decision-available."""

    name: str
    value: Any
    available_at: Timestamp
    source: str

    def assert_available(self, decision_at: Timestamp) -> None:
        if as_utc(self.available_at) > as_utc(decision_at):
            raise FutureInformationError(
                f"Feature {self.name!r} became available at {self.available_at!s}, "
                f"after decision time {decision_at!s}."
            )


class FutureInformationError(ValueError):
    """Raised when a feature contains information unavailable at decision time."""


def validate_as_of(features: Iterable[AsOfFeature], decision_at: Timestamp) -> None:
    """Reject a feature set containing any future information."""

    for feature in features:
        feature.assert_available(decision_at)


def values_as_of(
    features: Iterable[AsOfFeature], decision_at: Timestamp
) -> dict[str, Any]:
    """Validate features and return their values keyed by feature name."""

    materialized = list(features)
    validate_as_of(materialized, decision_at)
    return {feature.name: feature.value for feature in materialized}


REQUIRED_HOME_FIELDS = frozenset(
    {"parcel_id", "sale_date", "sale_price", "living_area_sqft", "year_built"}
)


def validate_home_record(record: Mapping[str, Any]) -> None:
    """Apply lightweight structural checks before expensive model work."""

    missing = REQUIRED_HOME_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Missing required home fields: {sorted(missing)}")
    if float(record["sale_price"]) <= 0:
        raise ValueError("sale_price must be positive")
    if float(record["living_area_sqft"]) <= 0:
        raise ValueError("living_area_sqft must be positive")
    year_built = int(record["year_built"])
    if year_built < 1800 or year_built > datetime.now(tz=timezone.utc).year:
        raise ValueError("year_built is outside the supported range")
