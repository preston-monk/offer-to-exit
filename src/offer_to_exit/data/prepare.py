"""Shared privacy and provenance utilities for county-record preparation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

HASH_NAMESPACE = "offer-to-exit-public-v1"
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


class PreparationSummaryLike(Protocol):
    """Structural type required by the combined preparation manifest."""

    def to_dict(self) -> dict[str, object]: ...


def parcel_hash(value: object) -> str:
    """Create a stable, namespaced join key without retaining the raw parcel ID."""

    normalized = str(value).strip().upper()
    return hashlib.sha256(f"{HASH_NAMESPACE}:{normalized}".encode()).hexdigest()[:20]


def assert_safe_columns(columns: Iterable[str]) -> None:
    """Fail closed if a released column name resembles a direct identifier."""

    unsafe = [
        column
        for column in columns
        if any(token in column.lower().replace("_", "") for token in PII_TOKENS)
    ]
    if unsafe:
        raise ValueError(f"Refusing to materialize potential PII columns: {unsafe}")


def write_preparation_manifest(
    processed_dir: Path,
    summaries: Iterable[PreparationSummaryLike],
    market_files: Iterable[Path],
    *,
    analysis: Mapping[str, object] | None = None,
) -> Path:
    """Record privacy-safe outputs without exposing raw identifiers."""

    path = processed_dir / "preparation_manifest.json"
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "tables": [summary.to_dict() for summary in summaries],
        "market_files": [str(file) for file in market_files],
        "analysis": dict(analysis or {}),
        "privacy": {
            "raw_county_files_committed": False,
            "party_names_retained": False,
            "street_addresses_retained": False,
            "coordinates_retained": False,
            "raw_parcel_ids_retained": False,
            "market_namespaced_parcel_ids_hashed": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
