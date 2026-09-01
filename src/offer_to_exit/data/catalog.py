"""Machine-readable catalog of the public datasets used by the project.

Raw third-party data are intentionally excluded from version control.  Each
source below is fetched from its publisher and recorded in a local manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCES: Final[dict[str, DataSource]] = {
    "maricopa_residential": DataSource(
        key="maricopa_residential",
        title="Residential Master",
        publisher="Maricopa County Assessor",
        url=(
            "https://www.arcgis.com/sharing/rest/content/items/"
            "e22983d41d91490d90965544b718a120/data"
        ),
        landing_page="https://www.mcassessor.maricopa.gov/page/data_sales/",
        filename="Residential_Master.zip",
        grain="current residential property component",
        purpose="physical home attributes for valuation and market segmentation",
        update_cadence="twice monthly",
        approximate_bytes=58_424_627,
        redistribution="Do not commit raw files; publisher data are supplied as-is.",
        attribution="Maricopa County Assessor's Office",
        contains_pii=True,
    ),
    "maricopa_sales": DataSource(
        key="maricopa_sales",
        title="Sales Affidavits",
        publisher="Maricopa County Assessor",
        url=(
            "https://www.arcgis.com/sharing/rest/content/items/"
            "f3484c72a938497286adc4e5de7e9963/data"
        ),
        landing_page="https://www.mcassessor.maricopa.gov/page/data_sales/",
        filename="Sales_Affidavits.zip",
        grain="recorded property transfer",
        purpose="historical transaction prices and dates",
        update_cadence="weekly",
        approximate_bytes=61_382_362,
        redistribution="Do not commit raw files; publisher data are supplied as-is.",
        attribution="Maricopa County Assessor's Office",
        contains_pii=True,
    ),
    "maricopa_parcel_points": DataSource(
        key="maricopa_parcel_points",
        title="Parcel Points",
        publisher="Maricopa County Assessor",
        url=(
            "https://www.arcgis.com/sharing/rest/content/items/"
            "dbf139379db946e1b10a2f15672c142d/data"
        ),
        landing_page="https://www.mcassessor.maricopa.gov/page/data_sales/",
        filename="Parcel_Points.zip",
        grain="active parcel point geometry",
        purpose="spatial joins without publishing street addresses",
        update_cadence="publisher schedule varies",
        approximate_bytes=85_721_909,
        redistribution="Do not commit raw geometry; publish only aggregated geography.",
        attribution="Maricopa County Assessor's Office",
    ),
    "realtor_metro_history": DataSource(
        key="realtor_metro_history",
        title="Monthly Housing Inventory - Metro History",
        publisher="Realtor.com Economic Research",
        url=(
            "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/"
            "RDC_Inventory_Core_Metrics_Metro_History.csv"
        ),
        landing_page="https://www.realtor.com/research/data/",
        filename="RDC_Inventory_Core_Metrics_Metro_History.csv",
        grain="metro-month",
        purpose="inventory, list price, price cuts, pending ratio, and days on market",
        update_cadence="monthly; full history may be revised",
        approximate_bytes=32_912_319,
        redistribution="Fetch from publisher; do not vendor the complete raw history.",
        attribution="Realtor.com Economic Research",
    ),
    "fhfa_hpi": DataSource(
        key="fhfa_hpi",
        title="FHFA House Price Index Master Data",
        publisher="Federal Housing Finance Agency",
        url="https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv",
        landing_page="https://www.fhfa.gov/house-price-index",
        filename="fhfa_hpi_master.csv",
        grain="geography-period-index type",
        purpose="repeat-sales market appreciation and stress regimes",
        update_cadence="monthly and quarterly",
        approximate_bytes=None,
        redistribution="Government data; preserve source and retrieval metadata.",
        attribution="Federal Housing Finance Agency",
    ),
    "mortgage30us": DataSource(
        key="mortgage30us",
        title="30-Year Fixed Rate Mortgage Average in the United States",
        publisher="Federal Reserve Bank of St. Louis (FRED)",
        url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
        landing_page="https://fred.stlouisfed.org/series/MORTGAGE30US",
        filename="MORTGAGE30US.csv",
        grain="week",
        purpose="financing-cost and demand-regime covariate",
        update_cadence="weekly",
        approximate_bytes=None,
        redistribution="Fetch from publisher and display the required source notice.",
        attribution=(
            "Federal Reserve Bank of St. Louis; this project is not endorsed or "
            "certified by the Federal Reserve Bank of St. Louis"
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
