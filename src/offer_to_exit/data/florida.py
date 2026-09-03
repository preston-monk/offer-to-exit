"""Privacy-safe Florida transactions and housing-intermediary episodes.

The raw county records contain names, addresses, exact coordinates, and parcel
identifiers.  This module uses party names only transiently to classify a small,
pre-specified set of classic inventory iBuyers.  Released tables contain namespaced
parcel hashes and operator labels, never raw party names or parcel identifiers.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import struct
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import IO, Final, TextIO

import pandas as pd

from offer_to_exit.data.prepare import parcel_hash

HILLSBOROUGH_MARKET: Final = "tampa_hillsborough"
ORANGE_MARKET: Final = "orlando_orange"

CLASSIC_IBUYERS: Final[tuple[str, ...]] = (
    "opendoor",
    "offerpad",
    "zillow_offers",
    "redfinnow",
)

TRANSACTION_COLUMNS: Final[tuple[str, ...]] = (
    "parcel_id",
    "market",
    "sale_date",
    "sale_price",
    "adjusted_sale_price",
    "qualified",
    "property_type_code",
    "dor_code",
    "improved",
    "instrument_type",
    "year_built",
    "effective_year_built",
    "living_area_sqft",
    "gross_area_sqft",
    "beds",
    "baths",
    "stories",
    "pool",
    "neighborhood",
    "subdivision",
    "census_block_group",
    "buyer_operator",
    "seller_operator",
)

EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "parcel_id",
    "market",
    "operator",
    "acquisition_date",
    "acquisition_price",
    "resale_date",
    "resale_price",
    "hold_days",
    "event_observed",
    "linkage_status",
    "gross_spread",
    "gross_return",
    "qualified",
    "property_type_code",
    "dor_code",
    "improved",
    "year_built",
    "effective_year_built",
    "living_area_sqft",
    "gross_area_sqft",
    "beds",
    "baths",
    "stories",
    "pool",
    "neighborhood",
    "subdivision",
    "census_block_group",
)

_EPISODE_ATTRIBUTES: Final[tuple[str, ...]] = (
    "qualified",
    "property_type_code",
    "dor_code",
    "improved",
    "year_built",
    "effective_year_built",
    "living_area_sqft",
    "gross_area_sqft",
    "beds",
    "baths",
    "stories",
    "pool",
    "neighborhood",
    "subdivision",
    "census_block_group",
)


@dataclass(frozen=True, slots=True)
class FloridaPreparationSummary:
    """Provenance for one privacy-safe Florida transaction table."""

    market: str
    input_file: str
    output_file: str
    input_rows: int
    output_rows: int
    columns: tuple[str, ...]
    created_at: str
    privacy_contract: str = (
        "raw party names and parcel identifiers excluded; parcel IDs namespaced and hashed"
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_classic_ibuyer(party_name: object) -> str | None:
    """Classify a party with strict legal-name roots, not fuzzy LLC matching.

    The classifier intentionally excludes generic investors, institutional
    landlords, brokers, and strings such as ``OPEN DOOR CHURCH``.  Its economic
    population is named classic iBuyers that acquired homes onto their own
    balance sheets and later resold them.
    """

    if party_name is None or _is_missing(party_name):
        return None
    raw_upper = str(party_name).upper()
    # Almost every deed party is outside the reviewed registry.  Avoid the
    # comparatively expensive normalization regex for those millions of rows.
    if not any(
        token in raw_upper
        for token in ("OPENDOOR", "OPEN DOOR", "OFFERPAD", "OFFER PAD", "ZILLOW HOME", "REDFIN")
    ):
        return None
    normalized = re.sub(r"[^A-Z0-9]+", " ", raw_upper).strip()
    if re.match(r"^OPENDOOR(?:\s|$)", normalized) or re.match(
        r"^OPEN DOOR PROPERTY TRUST(?:\s|$)", normalized
    ):
        return "opendoor"
    if re.match(r"^OFFER ?PAD(?:\s|LLC|INC|SOLUTIONS|SPV|$)", normalized):
        return "offerpad"
    if re.match(r"^ZILLOW HOMES?(?:\s|$)", normalized):
        return "zillow_offers"
    if re.match(r"^REDFIN\s*NOW(?:\s|$)", normalized):
        return "redfinnow"
    return None


def normalize_hillsborough_rows(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Normalize Hillsborough DBF rows and irreversibly remove direct identifiers."""

    normalized = [_normalize_hillsborough_row(row) for row in rows]
    return _canonical_frame(record for record in normalized if record is not None)


def normalize_orange_records(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Normalize Orange ArcGIS attributes and remove names, addresses, and geometry."""

    normalized = [_normalize_orange_record(record) for record in records]
    return _canonical_frame(record for record in normalized if record is not None)


def iter_hillsborough_dbf(archive: Path) -> Iterator[dict[str, object]]:
    """Stream the largest DBF member from Hillsborough's all-sales ZIP archive."""

    with zipfile.ZipFile(archive) as zipped:
        candidates = [item for item in zipped.infolist() if item.filename.lower().endswith(".dbf")]
        if not candidates:
            raise ValueError(f"No DBF member found in {archive}")
        member = max(candidates, key=lambda item: item.file_size)
        with zipped.open(member) as stream:
            yield from _iter_dbf_records(stream)


def iter_orange_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Stream attributes written by :func:`fetch_arcgis_source`."""

    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield from _iter_json_stream(stream, path)
    else:
        with path.open("rt", encoding="utf-8") as stream:
            yield from _iter_json_stream(stream, path)


def _iter_json_stream(stream: TextIO, path: Path) -> Iterator[dict[str, object]]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object on line {line_number} of {path}")
        yield payload


def sanitize_hillsborough_archive(
    archive: Path,
    output: Path,
    *,
    batch_size: int = 100_000,
) -> FloridaPreparationSummary:
    """Stream a Hillsborough archive to a privacy-safe compressed CSV."""

    return _sanitize_batches(
        iter_hillsborough_dbf(archive),
        output,
        normalizer=normalize_hillsborough_rows,
        market=HILLSBOROUGH_MARKET,
        input_file=archive,
        batch_size=batch_size,
    )


def sanitize_orange_jsonl(
    raw_jsonl: Path,
    output: Path,
    *,
    batch_size: int = 100_000,
) -> FloridaPreparationSummary:
    """Stream an Orange raw ArcGIS extract to a privacy-safe compressed CSV."""

    return _sanitize_batches(
        iter_orange_jsonl(raw_jsonl),
        output,
        normalizer=normalize_orange_records,
        market=ORANGE_MARKET,
        input_file=raw_jsonl,
        batch_size=batch_size,
    )


def read_sanitized_transactions(path: Path) -> pd.DataFrame:
    """Read a released transaction table using the canonical date types."""

    frame = pd.read_csv(path, low_memory=False)
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], format="mixed", errors="coerce")
    return frame


def link_ibuyer_episodes(
    transactions: pd.DataFrame,
    *,
    as_of: (
        str
        | date
        | datetime
        | pd.Timestamp
        | Mapping[str, str | date | datetime | pd.Timestamp]
        | None
    ) = None,
    minimum_price: float = 10_000.0,
    maximum_hold_days: int | None = 1_095,
    include_censored: bool = True,
) -> pd.DataFrame:
    """Link named iBuyer acquisitions to the first subsequent operator resale.

    An acquisition is a deed on which the named operator is the grantee and
    recorded consideration meets ``minimum_price``. Its exit is the earliest
    later deed on the same hashed parcel for which that operator is the grantor
    and consideration meets the same threshold. The screen excludes nominal and
    administrative transfers that do not provide interpretable purchase or
    resale prices. Resale deeds are consumed at most once. Rows with no eligible
    exit are retained as right-censored observations when requested.
    ``gross_spread`` is before repairs, transaction costs, financing, taxes, and
    operating costs and must not be interpreted as profit.
    """

    required = {
        "parcel_id",
        "market",
        "sale_date",
        "sale_price",
        "buyer_operator",
        "seller_operator",
    }
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise ValueError(f"Transactions are missing required columns: {missing}")
    if maximum_hold_days is not None and maximum_hold_days < 1:
        raise ValueError("maximum_hold_days must be positive or None")

    sales = transactions.copy()
    # County exports use different ISO representations. Hillsborough publishes
    # dates while Orange's ArcGIS service publishes local-midnight timestamps.
    # ``format="mixed"`` prevents pandas from inferring the first county's
    # representation and silently coercing the other county to missing values.
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], format="mixed", errors="coerce")
    sales["sale_price"] = pd.to_numeric(sales["sale_price"], errors="coerce")
    sales = sales.loc[
        sales["sale_date"].notna()
        & sales["sale_price"].ge(minimum_price)
        & (sales["buyer_operator"].notna() | sales["seller_operator"].notna())
    ].copy()
    if sales.empty:
        return _empty_episode_frame()

    markets = tuple(str(value) for value in sales["market"].dropna().unique())
    if isinstance(as_of, Mapping):
        missing_cutoffs = sorted(set(markets).difference(as_of))
        if missing_cutoffs:
            raise ValueError(f"Missing as-of dates for markets: {missing_cutoffs}")
        market_cutoffs = {market: pd.Timestamp(as_of[market]) for market in markets}
    elif as_of is None:
        market_cutoffs = {
            str(market): pd.Timestamp(cutoff)
            for market, cutoff in sales.groupby("market", sort=False)["sale_date"].max().items()
        }
    else:
        cutoff = pd.Timestamp(as_of)
        market_cutoffs = {market: cutoff for market in markets}
    if any(pd.isna(cutoff) for cutoff in market_cutoffs.values()):
        raise ValueError("Cannot determine an as-of date from the transactions")
    row_cutoffs = sales["market"].astype("string").map(market_cutoffs)
    sales = sales.loc[sales["sale_date"].le(row_cutoffs)].copy()
    if sales.empty:
        return _empty_episode_frame()

    dedupe_columns = [
        "parcel_id",
        "market",
        "sale_date",
        "sale_price",
        "buyer_operator",
        "seller_operator",
    ]
    sales = sales.sort_values(["market", "parcel_id", "sale_date", "sale_price"]).drop_duplicates(
        dedupe_columns
    )

    episodes: list[dict[str, object]] = []
    for (market, _), parcel_sales in sales.groupby(["market", "parcel_id"], sort=False):
        cutoff = market_cutoffs[str(market)]
        parcel_sales = parcel_sales.sort_values(["sale_date", "sale_price"]).reset_index(drop=True)
        for operator in CLASSIC_IBUYERS:
            open_acquisition: pd.Series | None = None
            operator_sales = parcel_sales.loc[
                parcel_sales["buyer_operator"].eq(operator)
                | parcel_sales["seller_operator"].eq(operator)
            ]
            for _, transaction in operator_sales.iterrows():
                transaction_date = pd.Timestamp(transaction["sale_date"])
                is_exit = _operator_equal(transaction["seller_operator"], operator)
                is_acquisition = _operator_equal(transaction["buyer_operator"], operator)

                if is_exit and open_acquisition is not None:
                    acquisition_date = pd.Timestamp(open_acquisition["sale_date"])
                    if transaction_date > acquisition_date:
                        elapsed = int((transaction_date - acquisition_date).days)
                        if maximum_hold_days is None or elapsed <= maximum_hold_days:
                            episodes.append(
                                _episode_record(
                                    open_acquisition,
                                    operator=operator,
                                    end=transaction_date,
                                    resale=transaction,
                                    status="completed",
                                )
                            )
                            open_acquisition = None
                        else:
                            if include_censored:
                                episodes.append(
                                    _episode_record(
                                        open_acquisition,
                                        operator=operator,
                                        end=acquisition_date + pd.Timedelta(days=maximum_hold_days),
                                        resale=None,
                                        status="administrative_horizon",
                                    )
                                )
                            open_acquisition = None

                if is_acquisition:
                    if open_acquisition is not None and include_censored:
                        prior_date = pd.Timestamp(open_acquisition["sale_date"])
                        repeat_end = transaction_date
                        if maximum_hold_days is not None:
                            repeat_end = min(
                                repeat_end,
                                prior_date + pd.Timedelta(days=maximum_hold_days),
                            )
                        episodes.append(
                            _episode_record(
                                open_acquisition,
                                operator=operator,
                                end=repeat_end,
                                resale=None,
                                status="repeat_acquisition_before_exit",
                            )
                        )
                    open_acquisition = transaction

            if open_acquisition is not None and include_censored:
                acquisition_date = pd.Timestamp(open_acquisition["sale_date"])
                censor_end = cutoff
                status = "right_censored"
                if maximum_hold_days is not None:
                    horizon = acquisition_date + pd.Timedelta(days=maximum_hold_days)
                    if horizon <= cutoff:
                        censor_end = horizon
                        status = "administrative_horizon"
                episodes.append(
                    _episode_record(
                        open_acquisition,
                        operator=operator,
                        end=censor_end,
                        resale=None,
                        status=status,
                    )
                )

    if not episodes:
        return _empty_episode_frame()
    return pd.DataFrame.from_records(episodes, columns=EPISODE_COLUMNS).sort_values(
        ["market", "operator", "acquisition_date", "parcel_id"], ignore_index=True
    )


def _episode_record(
    acquisition: pd.Series,
    *,
    operator: str,
    end: pd.Timestamp,
    resale: pd.Series | None,
    status: str,
) -> dict[str, object]:
    acquisition_date = pd.Timestamp(acquisition["sale_date"])
    acquisition_price = float(acquisition["sale_price"])
    resale_date = pd.NaT if resale is None else pd.Timestamp(resale["sale_date"])
    resale_price = math.nan if resale is None else float(resale["sale_price"])
    gross_spread = math.nan if resale is None else resale_price - acquisition_price
    gross_return = math.nan if resale is None else gross_spread / acquisition_price
    episode: dict[str, object] = {
        "parcel_id": acquisition["parcel_id"],
        "market": acquisition["market"],
        "operator": operator,
        "acquisition_date": acquisition_date,
        "acquisition_price": acquisition_price,
        "resale_date": resale_date,
        "resale_price": resale_price,
        "hold_days": max(0, int((end - acquisition_date).days)),
        "event_observed": resale is not None,
        "linkage_status": status,
        "gross_spread": gross_spread,
        "gross_return": gross_return,
    }
    for column in _EPISODE_ATTRIBUTES:
        episode[column] = acquisition.get(column, pd.NA)
    return episode


def _operator_equal(value: object, operator: str) -> bool:
    """Compare nullable CSV string values without invoking ``pd.NA`` truthiness."""

    return not _is_missing(value) and str(value) == operator


def _normalize_hillsborough_row(row: Mapping[str, object]) -> dict[str, object] | None:
    values = _casefold_mapping(row)
    raw_parcel = _value(values, "PIN", "FOLIO", "PARCEL_ID", "PARCEL")
    if raw_parcel is None or not str(raw_parcel).strip():
        return None
    return {
        "parcel_id": _market_parcel_hash(HILLSBOROUGH_MARKET, raw_parcel),
        "market": HILLSBOROUGH_MARKET,
        # DBF dates arrive as YYYYMMDD strings.  Leave them scalar-free here;
        # `_canonical_frame` parses the full batch in one vectorized operation.
        "sale_date": _value(values, "S_DATE", "SALE_DATE", "SALEDATE"),
        "sale_price": _parse_number(_value(values, "S_AMT", "SALE_AMOUNT", "SALE_PRICE")),
        "adjusted_sale_price": math.nan,
        "qualified": _parse_qualified(_value(values, "QU", "QUALIFIED", "QUAL", "QUAL_FLAG")),
        "property_type_code": _safe_text(_value(values, "DOR_CODE", "DOR")),
        "dor_code": _safe_text(_value(values, "DOR_CODE", "DOR")),
        "improved": _parse_improved(_value(values, "VI", "IMP_VAC", "IMPROVED")),
        "instrument_type": _safe_text(_value(values, "S_TYPE", "INSTRUMENT", "INSTR_TYPE", "STR")),
        "year_built": math.nan,
        "effective_year_built": math.nan,
        "living_area_sqft": math.nan,
        "gross_area_sqft": math.nan,
        "beds": math.nan,
        "baths": math.nan,
        "stories": math.nan,
        "pool": pd.NA,
        "neighborhood": _safe_text(_value(values, "NBHC", "NBRHD", "NEIGHBORHOOD")),
        "subdivision": _safe_text(_value(values, "SUB", "S_DIV", "SUBDIVISION")),
        "census_block_group": None,
        "buyer_operator": classify_classic_ibuyer(_value(values, "GRANTEE")),
        "seller_operator": classify_classic_ibuyer(_value(values, "GRANTOR")),
    }


def _normalize_orange_record(record: Mapping[str, object]) -> dict[str, object] | None:
    attributes = record.get("attributes", record)
    if not isinstance(attributes, Mapping):
        return None
    values = _casefold_mapping(attributes)
    raw_parcel = _value(
        values,
        "PARCEL",
        "PARCEL_ID",
        "PARCELID",
        "PARCELNO",
        "PARCEL_NUMBER",
        "PID",
    )
    if raw_parcel is None or not str(raw_parcel).strip():
        return None
    return {
        "parcel_id": _market_parcel_hash(ORANGE_MARKET, raw_parcel),
        "market": ORANGE_MARKET,
        "sale_date": _parse_date(
            _value(values, "SALE_DATE", "SALEDATE", "SALE_DT", "DATE_OF_SALE")
        ),
        "sale_price": _parse_number(
            _value(values, "SALE_AMOUNT", "SALE_AMT", "SALE_PRICE", "SALEPRICE")
        ),
        "adjusted_sale_price": _parse_number(
            _value(
                values,
                "ADJUSTED_SALE_AMOUNT",
                "ADJ_SALE_AMOUNT",
                "ADJ_SALE_AMT",
                "SALE_ADJ_AMOUNT",
            )
        ),
        "qualified": _parse_orange_qualified(_value(values, "SALE_DESCRIPTION")),
        "property_type_code": _safe_text(
            _value(
                values,
                "SALE_TYPE",
                "PROPERTY_TYPE",
                "PROPERTY_TYPE_CODE",
                "DOR_CODE",
                "DOR",
            )
        ),
        "dor_code": _safe_text(_value(values, "DOR_CODE", "DOR")),
        "improved": _parse_improved(_value(values, "VAC_IMPR_CODE", "IMPROVED", "IMP_VAC")),
        "instrument_type": _safe_text(_value(values, "DEED_CODE", "DEED_TYPE")),
        "year_built": _parse_number(
            _value(
                values,
                "AYB",
                "ACTUAL_YEAR_BUILT",
                "ACTUAL_YR_BUILT",
                "YEAR_BUILT",
                "YR_BUILT",
            )
        ),
        "effective_year_built": _parse_number(
            _value(
                values,
                "EYB",
                "EFFECTIVE_YEAR_BUILT",
                "EFFECTIVE_YR_BUILT",
                "EFF_YEAR_BUILT",
            )
        ),
        "living_area_sqft": _parse_number(
            _value(values, "LIVING_AREA", "LIVING_AREA_SQFT", "LIV_AREA", "HEATED_AREA")
        ),
        "gross_area_sqft": _parse_number(
            _value(values, "GROSS_AREA", "GROSS_AREA_SQFT", "TOTAL_AREA")
        ),
        "beds": _parse_number(_value(values, "BEDROOMS", "BEDS", "BED_COUNT")),
        "baths": _parse_number(_value(values, "BATH", "BATHS", "BATHROOMS", "BATH_COUNT")),
        "stories": _parse_number(_value(values, "STYS", "STORIES", "STORY_COUNT")),
        "pool": _parse_boolean(_value(values, "POOL", "HAS_POOL", "POOL_IND")),
        "neighborhood": _safe_text(_value(values, "NBHD_CODE", "NEIGHBORHOOD", "NBHD", "NBRHD")),
        "subdivision": _safe_text(
            _value(
                values,
                "SUBDIVISION_NAME",
                "SUBDIVISION",
                "SUBDIV",
                "SUBDIV_NAME",
            )
        ),
        "census_block_group": _safe_text(
            _value(
                values,
                "BLKGROUPID",
                "CENSUS_BLOCK_GROUP",
                "BLOCK_GROUP",
                "BLOCKGROUP",
            )
        ),
        "buyer_operator": classify_classic_ibuyer(_value(values, "GRANTEE")),
        "seller_operator": classify_classic_ibuyer(_value(values, "GRANTOR")),
    }


def _canonical_frame(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=TRANSACTION_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)
    raw_dates = frame["sale_date"]
    compact_dates = raw_dates.astype("string").str.fullmatch(r"\d{8}", na=False)
    if bool(compact_dates.all()):
        frame["sale_date"] = pd.to_datetime(
            raw_dates.astype("string"), format="%Y%m%d", errors="coerce"
        )
    else:
        frame["sale_date"] = pd.to_datetime(raw_dates, format="mixed", errors="coerce")
    # A deed date is a calendar date, not a time-of-day outcome. Normalizing
    # also gives both county files the same serialized representation.
    frame["sale_date"] = frame["sale_date"].dt.normalize()
    numeric = (
        "sale_price",
        "adjusted_sale_price",
        "year_built",
        "effective_year_built",
        "living_area_sqft",
        "gross_area_sqft",
        "beds",
        "baths",
        "stories",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _sanitize_batches(
    records: Iterable[Mapping[str, object]],
    output: Path,
    *,
    normalizer: Callable[[Iterable[Mapping[str, object]]], pd.DataFrame],
    market: str,
    input_file: Path,
    batch_size: int,
) -> FloridaPreparationSummary:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    input_rows = 0
    output_rows = 0
    batch: list[Mapping[str, object]] = []
    wrote_header = False
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as sink:
            for record in records:
                input_rows += 1
                batch.append(record)
                if len(batch) >= batch_size:
                    frame = normalizer(batch)
                    frame.to_csv(sink, index=False, header=not wrote_header)
                    output_rows += len(frame)
                    wrote_header = True
                    batch.clear()
            if batch or not wrote_header:
                frame = normalizer(batch)
                frame.to_csv(sink, index=False, header=not wrote_header)
                output_rows += len(frame)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return FloridaPreparationSummary(
        market=market,
        input_file=str(input_file),
        output_file=str(output),
        input_rows=input_rows,
        output_rows=output_rows,
        columns=TRANSACTION_COLUMNS,
        created_at=datetime.now(tz=UTC).isoformat(),
    )


def _iter_dbf_records(stream: IO[bytes]) -> Iterator[dict[str, object]]:
    header = stream.read(32)
    if len(header) != 32:
        raise ValueError("Truncated DBF header")
    record_count = struct.unpack("<I", header[4:8])[0]
    header_length = struct.unpack("<H", header[8:10])[0]
    record_length = struct.unpack("<H", header[10:12])[0]
    descriptors = stream.read(header_length - 32)
    fields: list[tuple[str, int]] = []
    cursor = 0
    while cursor < len(descriptors) and descriptors[cursor] != 0x0D:
        descriptor = descriptors[cursor : cursor + 32]
        if len(descriptor) < 32:
            raise ValueError("Truncated DBF field descriptor")
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        fields.append((name, descriptor[16]))
        cursor += 32
    if not fields or sum(length for _, length in fields) + 1 > record_length:
        raise ValueError("Invalid DBF field layout")

    for _ in range(record_count):
        raw = stream.read(record_length)
        if len(raw) < record_length:
            break
        if raw[:1] == b"*":
            continue
        position = 1
        record: dict[str, object] = {}
        for name, length in fields:
            value = raw[position : position + length]
            position += length
            record[name] = value.decode("cp1252", errors="replace").strip()
        yield record


def _casefold_mapping(row: Mapping[str, object]) -> dict[str, object]:
    return {_normalized_key(str(key)): value for key, value in row.items()}


def _normalized_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _value(row: Mapping[str, object], *aliases: str) -> object | None:
    for alias in aliases:
        key = _normalized_key(alias)
        if key in row:
            return row[key]
    return None


def _market_parcel_hash(market: str, value: object) -> str:
    return parcel_hash(f"{market}:{str(value).strip().upper()}")


def _parse_number(value: object) -> float:
    if value is None or _is_missing(value):
        return math.nan
    text = re.sub(r"[$,\s]", "", str(value))
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _parse_date(value: object) -> object:
    if value is None or _is_missing(value):
        return pd.NaT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric > 10_000_000_000:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if 19_000_000 <= numeric <= 21_001_231:
            return pd.to_datetime(str(int(numeric)), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def _parse_qualified(value: object) -> object:
    if value is None or _is_missing(value):
        return pd.NA
    normalized = str(value).strip().upper()
    if normalized in {"Q", "QUALIFIED", "Y", "YES", "TRUE", "1", "A", "ARMS LENGTH"}:
        return True
    if normalized in {"U", "UNQUALIFIED", "N", "NO", "FALSE", "0"}:
        return False
    return pd.NA


def _parse_orange_qualified(value: object) -> object:
    """Interpret the OCPA sale description without confusing it for property type."""

    if value is None or _is_missing(value):
        return pd.NA
    normalized = str(value).strip().upper()
    if not normalized:
        return pd.NA
    return normalized.startswith("SALE QUALIFIED")


def _parse_improved(value: object) -> object:
    if value is None or _is_missing(value):
        return pd.NA
    normalized = str(value).strip().upper()
    if normalized in {"I", "IMPROVED", "Y", "YES", "TRUE", "1"}:
        return True
    if normalized in {"V", "VACANT", "N", "NO", "FALSE", "0"}:
        return False
    return pd.NA


def _parse_boolean(value: object) -> object:
    if value is None or _is_missing(value):
        return pd.NA
    normalized = str(value).strip().upper()
    if normalized in {"Y", "YES", "TRUE", "1", "T"}:
        return True
    if normalized in {"N", "NO", "FALSE", "0", "F"}:
        return False
    return pd.NA


def _safe_text(value: object) -> str | None:
    if value is None or _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _empty_episode_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_COLUMNS)
