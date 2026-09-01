"""PII-minimizing preparation of the real Phoenix/Maricopa inputs.

The county files include owner and street-address fields.  This module never
extracts those fields: it streams selected columns from each ZIP archive,
hashes the parcel join key, and writes only sanitized analytical tables.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd


RESIDENTIAL_MEMBER = "Data/Residential_Master.txt"
SALES_MEMBER = "Data/Sales_Affidavits.txt"
PHOENIX_CBSA = "38060"
HASH_NAMESPACE = "offer-to-exit-public-v1"

RESIDENTIAL_COLUMNS = (
    "ParcelNumber",
    "ProportionComplete",
    "Class",
    "StoryCount",
    "AirConditioningType",
    "HeatingType",
    "BathroomFixtures",
    "ExteriorWallMaterial",
    "RoofMaterial",
    "RoofStyle",
    "ConstructionYear",
    "TotalLivingSqFt",
    "1stFloor",
    "2ndFloor",
    "3rdFloor",
    "Basement",
    "ParkCode",
    "Patios",
    "PoolSqFt",
    "SalePrice",
    "SaleDate",
    "AddedSqFt",
    "DetachSqFt",
    "PUC",
    "SitusCity",
    "SitusZipCode",
)

SALES_COLUMNS = (
    "PARCELNUMBER",
    "SALEDATE_MMYYYY",
    "SALEPRICE",
    "DEEDDATE_MMDDYYYY",
    "DEEDSTATUS",
    "DEEDTYPE",
    "PROPERTYTYPECODE",
    "PROPERTYTYPEDESCRIPTION",
    "SITUSCITY",
    "SITUSZIP",
    "DOWNPAYMENT",
    "PARTIALINTERESTINDICATOR",
    "MULTIPARCELINDICATOR",
    "BUY_SELLRELATIONSHIPINDICATOR",
    "ASSESSORCODE",
)

PII_TOKENS = frozenset(
    {
        "owner",
        "grantor",
        "grantee",
        "address",
        "streetnum",
        "streetname",
        "deednumber",
    }
)


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    table: str
    input_file: str
    output_file: str
    input_rows: int
    output_rows: int
    columns: tuple[str, ...]
    created_at: str


def parcel_hash(value: object) -> str:
    """Create a stable, namespaced join key without retaining the raw APN."""

    normalized = str(value).strip().upper()
    return hashlib.sha256(f"{HASH_NAMESPACE}:{normalized}".encode()).hexdigest()[:20]


def assert_safe_columns(columns: Iterable[str]) -> None:
    """Fail closed if a selected column name resembles known PII."""

    unsafe = [
        column
        for column in columns
        if any(token in column.lower().replace("_", "") for token in PII_TOKENS)
        and column.lower() not in {"parcelnumber"}
    ]
    if unsafe:
        raise ValueError(f"Refusing to materialize potential PII columns: {unsafe}")


def sanitize_residential(
    archive: Path, output: Path, *, chunk_size: int = 100_000
) -> PreparationSummary:
    """Stream safe residential attributes and hash the parcel join key."""

    return _sanitize_zip_table(
        archive,
        RESIDENTIAL_MEMBER,
        output,
        source_columns=RESIDENTIAL_COLUMNS,
        rename={"ParcelNumber": "parcel_id"},
        transform=_clean_residential,
        chunk_size=chunk_size,
        table="residential",
    )


def sanitize_sales(
    archive: Path, output: Path, *, chunk_size: int = 100_000
) -> PreparationSummary:
    """Stream safe recorded-sale fields and discard obvious non-market rows."""

    return _sanitize_zip_table(
        archive,
        SALES_MEMBER,
        output,
        source_columns=SALES_COLUMNS,
        rename={"PARCELNUMBER": "parcel_id"},
        transform=_clean_sales,
        chunk_size=chunk_size,
        table="sales",
    )


def prepare_market_series(raw_dir: Path, processed_dir: Path) -> list[Path]:
    """Filter the aggregate market files to Phoenix and normalize time fields."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    realtor = pd.read_csv(raw_dir / "RDC_Inventory_Core_Metrics_Metro_History.csv")
    realtor["cbsa_code"] = realtor["cbsa_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    realtor = realtor.loc[realtor["cbsa_code"] == PHOENIX_CBSA].copy()
    realtor["period"] = pd.to_datetime(
        realtor["month_date_yyyymm"].astype(str), format="%Y%m"
    )
    realtor_output = processed_dir / "phoenix_realtor_market.csv"
    realtor.sort_values("period").to_csv(realtor_output, index=False)
    outputs.append(realtor_output)

    hpi = pd.read_csv(raw_dir / "fhfa_hpi_master.csv", low_memory=False)
    hpi = hpi.loc[
        (hpi["place_id"].astype(str) == PHOENIX_CBSA)
        | hpi["place_name"].astype(str).str.contains("Phoenix-Mesa", na=False)
    ].copy()
    hpi["period_date"] = pd.to_datetime(
        hpi["yr"].astype(str)
        + "-"
        + hpi["period"].astype(str)
        + "-01",
        errors="coerce",
    )
    hpi_output = processed_dir / "phoenix_fhfa_hpi.csv"
    hpi.sort_values(["frequency", "yr", "period"]).to_csv(hpi_output, index=False)
    outputs.append(hpi_output)

    mortgage = pd.read_csv(raw_dir / "MORTGAGE30US.csv")
    mortgage["observation_date"] = pd.to_datetime(mortgage["observation_date"])
    mortgage_output = processed_dir / "mortgage30us.csv"
    mortgage.sort_values("observation_date").to_csv(mortgage_output, index=False)
    outputs.append(mortgage_output)
    return outputs


def write_preparation_manifest(
    processed_dir: Path, summaries: Iterable[PreparationSummary], market_files: Iterable[Path]
) -> Path:
    """Record the exact sanitized outputs without exposing raw identifiers."""

    path = processed_dir / "preparation_manifest.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "tables": [asdict(summary) for summary in summaries],
        "market_files": [str(file) for file in market_files],
        "privacy": {
            "raw_county_files_committed": False,
            "owner_names_retained": False,
            "street_addresses_retained": False,
            "parcel_ids_hashed": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sanitize_zip_table(
    archive: Path,
    member: str,
    output: Path,
    *,
    source_columns: tuple[str, ...],
    rename: Mapping[str, str],
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int,
    table: str,
) -> PreparationSummary:
    assert_safe_columns(source_columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    output_rows = 0
    written_columns: tuple[str, ...] = ()
    temporary = output.with_suffix(output.suffix + ".tmp")

    with zipfile.ZipFile(archive) as zipped, zipped.open(member) as stream:
        chunks = pd.read_csv(
            stream,
            sep="|",
            # Maricopa publishes these legacy extracts as Windows-1252.  In
            # particular, free-text categorical fields can contain byte 0x96
            # (an en dash), which is invalid UTF-8.
            encoding="cp1252",
            usecols=list(source_columns),
            dtype=str,
            chunksize=chunk_size,
            keep_default_na=False,
            quoting=csv.QUOTE_MINIMAL,
            low_memory=False,
        )
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as sink:
            for index, chunk in enumerate(chunks):
                input_rows += len(chunk)
                clean = transform(chunk.rename(columns=rename))
                if "parcel_id" in clean:
                    clean["parcel_id"] = clean["parcel_id"].map(parcel_hash)
                clean.to_csv(sink, index=False, header=index == 0)
                output_rows += len(clean)
                written_columns = tuple(clean.columns)

    temporary.replace(output)
    return PreparationSummary(
        table=table,
        input_file=str(archive),
        output_file=str(output),
        input_rows=input_rows,
        output_rows=output_rows,
        columns=written_columns,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _clean_residential(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame["parcel_id"].str.len() > 0].copy()
    for column in (
        "ProportionComplete",
        "StoryCount",
        "BathroomFixtures",
        "ConstructionYear",
        "TotalLivingSqFt",
        "1stFloor",
        "2ndFloor",
        "3rdFloor",
        "Basement",
        "Patios",
        "PoolSqFt",
        "SalePrice",
        "AddedSqFt",
        "DetachSqFt",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["SaleDate"] = pd.to_datetime(frame["SaleDate"], errors="coerce")
    return frame.loc[
        frame["TotalLivingSqFt"].between(300, 15_000, inclusive="both")
        & frame["ConstructionYear"].between(1850, 2030, inclusive="both")
    ]


def _clean_sales(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame["parcel_id"].str.len() > 0].copy()
    frame["SALEPRICE"] = pd.to_numeric(frame["SALEPRICE"], errors="coerce")
    frame["DOWNPAYMENT"] = pd.to_numeric(frame["DOWNPAYMENT"], errors="coerce")
    frame["sale_month"] = pd.to_datetime(
        frame["SALEDATE_MMYYYY"], format="%m%Y", errors="coerce"
    )
    frame["deed_date"] = pd.to_datetime(
        frame["DEEDDATE_MMDDYYYY"], format="%m%d%Y", errors="coerce"
    )
    description = frame["PROPERTYTYPEDESCRIPTION"].str.lower()
    plausible_type = description.str.contains("single|residential|condo|town", na=False)
    plausible_price = frame["SALEPRICE"].between(25_000, 10_000_000, inclusive="both")
    return frame.loc[plausible_type & plausible_price & frame["sale_month"].notna()].drop(
        columns=["SALEDATE_MMYYYY", "DEEDDATE_MMDDYYYY"]
    )
