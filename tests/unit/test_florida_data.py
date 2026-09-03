from __future__ import annotations

import gzip
import json
import struct
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from offer_to_exit.data.florida import (
    EPISODE_COLUMNS,
    HILLSBOROUGH_MARKET,
    ORANGE_MARKET,
    TRANSACTION_COLUMNS,
    classify_classic_ibuyer,
    iter_orange_jsonl,
    link_ibuyer_episodes,
    normalize_hillsborough_rows,
    normalize_orange_records,
    read_sanitized_transactions,
    sanitize_hillsborough_archive,
    sanitize_orange_jsonl,
)


@pytest.mark.parametrize(
    ("party", "expected"),
    [
        ("OPENDOOR PROPERTY J LLC", "opendoor"),
        ("OPEN DOOR PROPERTY TRUST I", "opendoor"),
        ("OFFERPAD SPVBORROWER1 LLC", "offerpad"),
        ("OFFER PAD LLC", "offerpad"),
        ("OFFERPADLLC", "offerpad"),
        ("ZILLOW HOMES PROPERTY TRUST", "zillow_offers"),
        ("REDFINNOW BORROWER LLC", "redfinnow"),
        ("REDFIN NOW BORROWER LLC", "redfinnow"),
        ("OPEN DOOR COMMUNITY CHURCH", None),
        ("ZILLOW GROUP INC", None),
        ("SOME RANDOM HOUSE BUYER LLC", None),
        (None, None),
    ],
)
def test_classic_ibuyer_classifier_is_strict(party: object, expected: str | None) -> None:
    assert classify_classic_ibuyer(party) == expected


def test_hillsborough_normalization_hashes_ids_and_discards_party_names() -> None:
    raw = {
        "PIN": "A-01-29-18-ZZZ-000001-00001.0",
        "S_DATE": "2024-03-15",
        "S_AMT": "350,000",
        "QU": "Q",
        "DOR_CODE": "01",
        "VI": "I",
        "S_TYPE": "WD",
        "NBHC": "102001.00",
        "SUB": "SUNNY ACRES",
        "GRANTOR": "SMITH JANE",
        "GRANTEE": "OPENDOOR LABS INC",
    }

    frame = normalize_hillsborough_rows([raw])

    assert tuple(frame.columns) == TRANSACTION_COLUMNS
    assert frame.loc[0, "market"] == HILLSBOROUGH_MARKET
    assert frame.loc[0, "parcel_id"] != raw["PIN"]
    assert len(frame.loc[0, "parcel_id"]) == 20
    assert frame.loc[0, "buyer_operator"] == "opendoor"
    assert pd.isna(frame.loc[0, "seller_operator"])
    assert frame.loc[0, "sale_price"] == 350_000
    assert "GRANTOR" not in frame.columns
    assert "GRANTEE" not in frame.columns


def test_orange_normalization_handles_arcgis_dates_and_attributes() -> None:
    raw = {
        "attributes": {
            "PARCEL_ID": "012345678900001",
            "SALE_DATE": 1_704_067_200_000,
            "SALE_AMOUNT": 425000,
            "ADJ_SALE_AMOUNT": 420000,
            "SALE_TYPE": "SINGLE FAMILY RESIDENTIAL",
            "SALE_DESCRIPTION": "SALE QUALIFIED",
            "DOR_CODE": "0100",
            "DEED_CODE": "WD",
            "AYB": 1998,
            "LIVING_AREA": 1840,
            "BEDS": 3,
            "BATH": 2,
            "STYS": 1,
            "POOL": "Y",
            "NBHD_CODE": "N001",
            "SUBDIVISION_NAME": "LAKE VIEW",
            "BLKGROUPID": "120950101001",
            "GRANTOR": "OFFERPAD LLC",
            "GRANTEE": "DOE JOHN",
            "SITUS": "123 PRIVATE STREET",
            "X": 123.45,
            "Y": 678.90,
        }
    }

    frame = normalize_orange_records([raw])

    assert tuple(frame.columns) == TRANSACTION_COLUMNS
    assert frame.loc[0, "market"] == ORANGE_MARKET
    assert frame.loc[0, "seller_operator"] == "offerpad"
    assert frame.loc[0, "qualified"] is True or frame.loc[0, "qualified"] == True  # noqa: E712
    assert frame.loc[0, "property_type_code"] == "SINGLE FAMILY RESIDENTIAL"
    assert frame.loc[0, "dor_code"] == "0100"
    assert frame.loc[0, "living_area_sqft"] == 1840
    assert frame.loc[0, "pool"] is True or frame.loc[0, "pool"] == True  # noqa: E712
    assert frame.loc[0, "sale_date"] == pd.Timestamp("2024-01-01")
    assert "SITUS" not in frame.columns
    assert "X" not in frame.columns


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        (None, None),
        ("", None),
        ("SALE QUALIFIED", True),
        ("Sale Qualified by Property Appraiser", True),
        ("MULTI-PARCEL SALE", False),
        ("DEED CHANGE ONLY", False),
    ],
)
def test_orange_qualification_comes_from_sale_description(
    description: str | None, expected: bool | None
) -> None:
    record = {
        "PARCEL": f"parcel-{description}",
        "SALE_DATE": "2024-01-01",
        "SALE_AMT": 300_000,
        "SALE_TYPE": "SINGLE FAMILY RESIDENTIAL",
        "DOR_CODE": "0100",
        "SALE_DESCRIPTION": description,
    }
    frame = normalize_orange_records([record])
    if expected is None:
        assert pd.isna(frame.loc[0, "qualified"])
    else:
        assert bool(frame.loc[0, "qualified"]) is expected
    assert frame.loc[0, "property_type_code"] == "SINGLE FAMILY RESIDENTIAL"
    assert frame.loc[0, "dor_code"] == "0100"


def test_episode_linkage_handles_completed_cross_operator_and_censored_sales() -> None:
    rows = [
        _transaction("parcel-a", "2023-01-01", 300_000, buyer="opendoor"),
        _transaction(
            "parcel-a",
            "2023-05-01",
            340_000,
            buyer="offerpad",
            seller="opendoor",
        ),
        _transaction("parcel-a", "2023-08-15", 360_000, seller="offerpad"),
        _transaction("parcel-b", "2024-02-01", 250_000, buyer="opendoor"),
    ]
    transactions = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)

    episodes = link_ibuyer_episodes(transactions, as_of="2024-04-01")

    assert tuple(episodes.columns) == EPISODE_COLUMNS
    assert len(episodes) == 3
    opendoor_complete = episodes.loc[
        (episodes["operator"] == "opendoor") & episodes["event_observed"]
    ].iloc[0]
    assert opendoor_complete["hold_days"] == 120
    assert opendoor_complete["gross_spread"] == 40_000
    assert opendoor_complete["gross_return"] == pytest.approx(40_000 / 300_000)
    assert opendoor_complete["linkage_status"] == "completed"
    offerpad = episodes.loc[episodes["operator"] == "offerpad"].iloc[0]
    assert offerpad["hold_days"] == 106
    censored = episodes.loc[~episodes["event_observed"]].iloc[0]
    assert censored["parcel_id"] == "parcel-b"
    assert censored["hold_days"] == 60
    assert censored["linkage_status"] == "right_censored"
    assert pd.isna(censored["resale_price"])


def test_episode_linkage_accepts_mixed_county_date_representations() -> None:
    transactions = pd.DataFrame(
        [
            _transaction("tampa", "2023-01-01", 200_000, buyer="opendoor"),
            _transaction("tampa", "2023-04-01", 230_000, seller="opendoor"),
            {
                **_transaction("orlando", "2023-02-01 05:00:00", 250_000, buyer="offerpad"),
                "market": ORANGE_MARKET,
            },
            {
                **_transaction("orlando", "2023-06-01 04:00:00", 280_000, seller="offerpad"),
                "market": ORANGE_MARKET,
            },
        ],
        columns=TRANSACTION_COLUMNS,
    )

    episodes = link_ibuyer_episodes(transactions)

    assert set(episodes["market"]) == {HILLSBOROUGH_MARKET, ORANGE_MARKET}
    assert len(episodes) == 2


def test_episode_linkage_uses_each_markets_observation_end() -> None:
    tampa = _transaction("tampa-open", "2024-01-01", 200_000, buyer="opendoor")
    orlando = {
        **_transaction("orlando-open", "2024-01-01", 250_000, buyer="offerpad"),
        "market": ORANGE_MARKET,
    }
    transactions = pd.DataFrame([tampa, orlando], columns=TRANSACTION_COLUMNS)

    episodes = link_ibuyer_episodes(
        transactions,
        as_of={
            HILLSBOROUGH_MARKET: "2024-03-01",
            ORANGE_MARKET: "2024-04-01",
        },
    ).set_index("market")

    assert episodes.loc[HILLSBOROUGH_MARKET, "hold_days"] == 60
    assert episodes.loc[ORANGE_MARKET, "hold_days"] == 91


def test_episode_linkage_requires_cutoff_for_every_observed_market() -> None:
    transactions = pd.DataFrame(
        [
            _transaction("tampa-open", "2024-01-01", 200_000, buyer="opendoor"),
            {
                **_transaction("orlando-open", "2024-01-01", 250_000, buyer="offerpad"),
                "market": ORANGE_MARKET,
            },
        ],
        columns=TRANSACTION_COLUMNS,
    )

    with pytest.raises(ValueError, match="Missing as-of dates"):
        link_ibuyer_episodes(
            transactions,
            as_of={HILLSBOROUGH_MARKET: "2024-03-01"},
        )


def test_episode_linkage_reports_repeat_acquisition_and_tops_out_horizon() -> None:
    transactions = pd.DataFrame(
        [
            _transaction("repeat", "2020-01-01", 200_000, buyer="opendoor"),
            _transaction("repeat", "2020-03-01", 210_000, buyer="opendoor"),
            _transaction("repeat", "2020-06-01", 240_000, seller="opendoor"),
            _transaction("old-open", "2019-01-01", 180_000, buyer="offerpad"),
            _transaction("future", "2025-01-01", 300_000, buyer="opendoor"),
        ],
        columns=TRANSACTION_COLUMNS,
    )

    episodes = link_ibuyer_episodes(
        transactions,
        as_of="2024-01-01",
        maximum_hold_days=365,
    )

    assert set(episodes["linkage_status"]) == {
        "repeat_acquisition_before_exit",
        "completed",
        "administrative_horizon",
    }
    repeated = episodes.loc[episodes["linkage_status"] == "repeat_acquisition_before_exit"].iloc[0]
    assert repeated["hold_days"] == 60
    horizon = episodes.loc[episodes["linkage_status"] == "administrative_horizon"].iloc[0]
    assert horizon["hold_days"] == 365
    assert "future" not in set(episodes["parcel_id"])


def test_orange_jsonl_sanitization_is_streamed_and_privacy_safe(tmp_path: Path) -> None:
    raw = tmp_path / "orange.jsonl.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "PARCEL": "private-parcel",
                    "SALE_DATE": "2024-01-10",
                    "SALE_AMOUNT": 300000,
                    "GRANTOR": "PRIVATE PERSON",
                    "GRANTEE": "ZILLOW HOMES INC",
                    "SITUS": "PRIVATE ADDRESS",
                }
            )
            + "\n"
        )
    assert next(iter(iter_orange_jsonl(raw)))["PARCEL"] == "private-parcel"

    output = tmp_path / "orange_safe.csv.gz"
    summary = sanitize_orange_jsonl(raw, output, batch_size=1)
    safe = pd.read_csv(output)

    assert summary.input_rows == summary.output_rows == 1
    assert tuple(safe.columns) == TRANSACTION_COLUMNS
    assert safe.loc[0, "parcel_id"] != "private-parcel"
    assert safe.loc[0, "buyer_operator"] == "zillow_offers"
    assert "PRIVATE PERSON" not in output.read_bytes().decode("latin1", errors="ignore")
    loaded = read_sanitized_transactions(output)
    assert loaded.loc[0, "sale_date"] == pd.Timestamp("2024-01-10")


def test_plain_jsonl_rejects_non_object_records(tmp_path: Path) -> None:
    raw = tmp_path / "orange.jsonl"
    raw.write_text('\n{"PARCEL": "one"}\n[1, 2]\n', encoding="utf-8")
    iterator = iter_orange_jsonl(raw)
    assert next(iterator)["PARCEL"] == "one"
    with pytest.raises(ValueError, match="Expected an object"):
        next(iterator)


def test_hillsborough_dbf_archive_is_streamed_and_sanitized(tmp_path: Path) -> None:
    archive = tmp_path / "allsales.zip"
    fields = [
        ("PIN", 16, "parcel-private"),
        ("S_DATE", 10, "2024-02-01"),
        ("S_AMT", 10, "325000"),
        ("QU", 1, "Q"),
        ("DOR_CODE", 4, "0100"),
        ("VI", 1, "I"),
        ("GRANTOR", 24, "PRIVATE HOUSEHOLD"),
        ("GRANTEE", 24, "OPENDOOR LABS INC"),
    ]
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("allsales.dbf", _dbf_fixture(fields))

    output = tmp_path / "hillsborough_safe.csv.gz"
    summary = sanitize_hillsborough_archive(archive, output, batch_size=1)
    frame = pd.read_csv(output)

    assert summary.to_dict()["market"] == HILLSBOROUGH_MARKET
    assert summary.input_rows == summary.output_rows == 1
    assert frame.loc[0, "buyer_operator"] == "opendoor"
    assert frame.loc[0, "parcel_id"] != "parcel-private"
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        released = stream.read()
    assert "PRIVATE HOUSEHOLD" not in released
    assert "parcel-private" not in released


def test_hillsborough_archive_requires_a_dbf_member(tmp_path: Path) -> None:
    archive = tmp_path / "not-a-dbf.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("readme.txt", "no table")
    with pytest.raises(ValueError, match="No DBF"):
        sanitize_hillsborough_archive(archive, tmp_path / "safe.csv.gz")


def test_invalid_batch_size_fails_before_materialization(tmp_path: Path) -> None:
    raw = tmp_path / "orange.jsonl.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        stream.write("")
    with pytest.raises(ValueError, match="batch_size"):
        sanitize_orange_jsonl(raw, tmp_path / "safe.csv.gz", batch_size=0)


def test_episode_linkage_validates_inputs_and_can_drop_censored_rows() -> None:
    with pytest.raises(ValueError, match="required columns"):
        link_ibuyer_episodes(pd.DataFrame({"sale_date": []}))
    transactions = pd.DataFrame(
        [_transaction("one", "2024-01-01", 200_000, buyer="opendoor")],
        columns=TRANSACTION_COLUMNS,
    )
    with pytest.raises(ValueError, match="maximum_hold_days"):
        link_ibuyer_episodes(transactions, maximum_hold_days=0)
    assert link_ibuyer_episodes(transactions, include_censored=False).empty
    assert link_ibuyer_episodes(transactions, as_of="2020-01-01").empty


def test_normalizers_drop_records_without_parcel_ids() -> None:
    assert normalize_hillsborough_rows([{"S_AMT": 100_000}]).empty
    assert normalize_orange_records([{"attributes": []}, {"SALE_AMT": 100_000}]).empty


def _transaction(
    parcel: str,
    sale_date: str,
    sale_price: float,
    *,
    buyer: str | None = None,
    seller: str | None = None,
) -> dict[str, object]:
    row = dict.fromkeys(TRANSACTION_COLUMNS, pd.NA)
    row.update(
        {
            "parcel_id": parcel,
            "market": HILLSBOROUGH_MARKET,
            "sale_date": pd.Timestamp(sale_date),
            "sale_price": sale_price,
            "buyer_operator": buyer,
            "seller_operator": seller,
            "living_area_sqft": 1_800,
            "year_built": 2001,
        }
    )
    return row


def _dbf_fixture(fields: list[tuple[str, int, str]]) -> bytes:
    record_count = 1
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(width for _, width, _ in fields)
    header = bytearray(32)
    header[0] = 3
    struct.pack_into("<I", header, 4, record_count)
    struct.pack_into("<H", header, 8, header_length)
    struct.pack_into("<H", header, 10, record_length)
    descriptors = bytearray()
    for name, width, _ in fields:
        descriptor = bytearray(32)
        encoded_name = name.encode("ascii")[:10]
        descriptor[: len(encoded_name)] = encoded_name
        descriptor[11] = ord("C")
        descriptor[16] = width
        descriptors.extend(descriptor)
    record = b" " + b"".join(
        value.encode("cp1252")[:width].ljust(width, b" ") for _, width, value in fields
    )
    return bytes(header) + bytes(descriptors) + b"\r" + record + b"\x1a"
