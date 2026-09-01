from datetime import UTC, datetime, timedelta

import pytest

from offer_to_exit.data.contracts import (
    AsOfFeature,
    FutureInformationError,
    validate_as_of,
    validate_home_record,
    values_as_of,
)


def test_future_features_are_rejected() -> None:
    decision = datetime(2026, 1, 15, tzinfo=UTC)
    feature = AsOfFeature(
        name="future_comp",
        value=500_000,
        available_at=decision + timedelta(days=1),
        source="fixture",
    )
    with pytest.raises(FutureInformationError):
        validate_as_of([feature], decision)


def test_available_features_materialize() -> None:
    decision = datetime(2026, 1, 15, tzinfo=UTC)
    feature = AsOfFeature(
        name="mortgage_rate",
        value=6.2,
        available_at=decision - timedelta(days=7),
        source="fixture",
    )
    assert values_as_of([feature], decision) == {"mortgage_rate": 6.2}


def test_home_contract_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="sale_price"):
        validate_home_record(
            {
                "parcel_id": "123",
                "sale_date": "2025-01-01",
                "sale_price": 0,
                "living_area_sqft": 1_500,
                "year_built": 1990,
            }
        )
