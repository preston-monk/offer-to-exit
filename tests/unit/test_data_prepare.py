from pathlib import Path

import pandas as pd
import pytest

from offer_to_exit.data.prepare import assert_safe_columns, parcel_hash


def test_parcel_hash_is_stable_and_namespaced() -> None:
    assert parcel_hash("123-45-678") == parcel_hash(" 123-45-678 ")
    assert parcel_hash("123-45-678") != "123-45-678"
    assert len(parcel_hash("123-45-678")) == 20


def test_pii_columns_fail_closed() -> None:
    with pytest.raises(ValueError, match="PII"):
        assert_safe_columns(["ParcelNumber", "OwnerName"])


def test_market_fixture_shape(tmp_path: Path) -> None:
    frame = pd.DataFrame({"cbsa_code": [38060], "median_days_on_market": [64.0]})
    target = tmp_path / "market.csv"
    frame.to_csv(target, index=False)
    loaded = pd.read_csv(target)
    assert loaded.loc[0, "median_days_on_market"] == 64.0
