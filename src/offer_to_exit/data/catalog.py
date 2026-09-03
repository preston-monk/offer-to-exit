"""Machine-readable catalog of the public datasets used by the project.

Raw third-party data are intentionally excluded from version control.  Each
source below is fetched from its publisher and recorded in a local manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class DataSource:
    """A reproducible external data source and its publication constraints."""

    key: str
    title: str
    publisher: str
    url: str
    landing_page: str
    filename: str
    grain: str
    purpose: str
    update_cadence: str
    approximate_bytes: int | None
    redistribution: str
    attribution: str
    contains_pii: bool = False
    transport: Literal["file", "arcgis", "aspnet_postback"] = "file"
    query_order_field: str | None = None
    page_size: int | None = None
    out_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCES: Final[dict[str, DataSource]] = {
    "hillsborough_sales": DataSource(
        key="hillsborough_sales",
        title="Hillsborough County All Sales",
        publisher="Hillsborough County Property Appraiser",
        url="https://downloads.hcpafl.org/",
        landing_page="https://downloads.hcpafl.org/",
        filename="hillsborough_allsales.zip",
        grain="recorded property transfer",
        purpose=(
            "Tampa development-market transaction history and named housing-"
            "intermediary acquisition-resale episodes"
        ),
        update_cadence="weekly; downloader records the discovered release filename",
        approximate_bytes=68_000_000,
        redistribution=(
            "Do not commit the raw archive or party names; keep analytical inputs "
            "direct-identifier-reduced and publish only aggregates and hashes."
        ),
        attribution="Hillsborough County Property Appraiser",
        contains_pii=True,
        transport="aspnet_postback",
    ),
    "orange_sales": DataSource(
        key="orange_sales",
        title="Orange County All Sales",
        publisher="Orange County Property Appraiser",
        url=("https://vgispublic.ocpafl.org/server/rest/services/Webmap/SALES/MapServer/5/query"),
        landing_page=(
            "https://vgispublic.ocpafl.org/server/rest/services/Webmap/SALES/MapServer/5"
        ),
        filename="orange_all_sales.jsonl.gz",
        grain="recorded property transfer",
        purpose=(
            "Orlando external-market evaluation with property attributes and named "
            "housing-intermediary acquisition-resale episodes"
        ),
        update_cadence="live ArcGIS feature service",
        approximate_bytes=None,
        redistribution=(
            "Do not commit the raw response or party names; keep analytical inputs "
            "direct-identifier-reduced and publish only aggregates and hashes."
        ),
        attribution="Orange County Property Appraiser",
        contains_pii=True,
        transport="arcgis",
        query_order_field="OBJECTID",
        page_size=1_000,
        out_fields=(
            "OBJECTID",
            "PARCEL",
            "SALE_DATE",
            "SALE_AMT",
            "ADJ_SALE_AMT",
            "DEED_CODE",
            "VAC_IMPR_CODE",
            "DOR_CODE",
            "GRANTOR",
            "GRANTEE",
            "SALES_ID",
            "AYB",
            "EYB",
            "STYS",
            "BATH",
            "BEDS",
            "LIVING_AREA",
            "BLDG_NUM",
            "POOL",
            "NBHD_CODE",
            "BLKGROUPID",
            "BLOCKID",
            "SALE_TYPE",
            "SALE_DESCRIPTION",
            "SUBDIVISION_NAME",
            "GROSS_AREA",
        ),
    ),
}


def get_source(key: str) -> DataSource:
    """Return a catalog entry or raise an error listing valid source keys."""

    try:
        return SOURCES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(SOURCES))
        raise KeyError(f"Unknown data source {key!r}. Choose one of: {valid}") from exc
