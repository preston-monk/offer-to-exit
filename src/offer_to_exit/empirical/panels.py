"""Privacy-safe analytical panels for the two-market Florida study.

The functions in this module accept the normalized outputs of the county data
pipeline.  They never require party names, addresses, coordinates, or raw
parcel identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from offer_to_exit.models._common import require_columns

TAMPA_MARKET = "tampa_hillsborough"
ORLANDO_MARKET = "orlando_orange"

VALUATION_REQUIRED_COLUMNS = (
    "parcel_id",
    "market",
    "sale_date",
    "sale_price",
)

EPISODE_REQUIRED_COLUMNS = (
    "parcel_id",
    "market",
    "operator",
    "acquisition_date",
    "acquisition_price",
    "hold_days",
    "event_observed",
)


@dataclass(frozen=True)
class GeographicSplit:
    """Development-market observations and a never-pooled geographic holdout."""

    development: pd.DataFrame
    holdout: pd.DataFrame
    development_market: str
    holdout_market: str

    def __post_init__(self) -> None:
        if self.development.empty or self.holdout.empty:
            raise ValueError("both geographic samples must contain observations")


@dataclass(frozen=True)
class ChronologicalGroupedSplit:
    """Chronological partitions with parcels purged across boundaries."""

    proper_training: pd.DataFrame
    calibration: pd.DataFrame
    out_of_time_test: pd.DataFrame
    calibration_start: pd.Timestamp
    test_start: pd.Timestamp
    raw_training_rows: int
    raw_calibration_rows: int
    raw_test_rows: int
    purged_training_rows: int
    purged_calibration_rows: int
    purged_training_parcels: int
    purged_calibration_parcels: int

    def __post_init__(self) -> None:
        if self.proper_training.empty or self.calibration.empty or self.out_of_time_test.empty:
            raise ValueError("all chronological samples must contain observations")

    def audit_summary(self, *, parcel_col: str = "parcel_id") -> dict[str, object]:
        """Return JSON-ready cutoffs, denominators, and parcel-purge counts."""

        return {
            "calibration_start": self.calibration_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "raw_rows": {
                "training": self.raw_training_rows,
                "calibration": self.raw_calibration_rows,
                "out_of_time_test": self.raw_test_rows,
            },
            "analysis_rows": {
                "training": len(self.proper_training),
                "calibration": len(self.calibration),
                "out_of_time_test": len(self.out_of_time_test),
            },
            "analysis_parcels": {
                "training": self.proper_training[parcel_col].nunique(dropna=False),
                "calibration": self.calibration[parcel_col].nunique(dropna=False),
                "out_of_time_test": self.out_of_time_test[parcel_col].nunique(dropna=False),
            },
            "purged_rows": {
                "training": self.purged_training_rows,
                "calibration": self.purged_calibration_rows,
            },
            "purged_parcels": {
                "training": self.purged_training_parcels,
                "calibration": self.purged_calibration_parcels,
            },
        }


def build_valuation_panel(
    transactions: pd.DataFrame,
    *,
    minimum_price: float = 25_000.0,
    maximum_price: float = 10_000_000.0,
    minimum_area: float = 300.0,
    maximum_area: float = 15_000.0,
    require_qualified: bool = True,
    require_improved: bool = True,
) -> pd.DataFrame:
    """Clean qualified arms-length sales and create decision-time features.

    Sale outcomes enter the target and diagnostics only.  The returned
    ``nominal_price_per_sqft`` is intentionally named as an outcome diagnostic;
    it must not be included in a valuation feature set.
    """

    require_columns(transactions, VALUATION_REQUIRED_COLUMNS)
    if minimum_price <= 0 or maximum_price <= minimum_price:
        raise ValueError("price bounds must be positive and ordered")
    if minimum_area <= 0 or maximum_area <= minimum_area:
        raise ValueError("area bounds must be positive and ordered")

    panel = transactions.copy()
    panel["market"] = panel["market"].astype("string").str.strip().str.lower()
    panel["sale_date"] = pd.to_datetime(panel["sale_date"], format="mixed", errors="coerce")
    for column in (
        "sale_price",
        "adjusted_sale_price",
        "living_area_sqft",
        "gross_area_sqft",
        "year_built",
        "effective_year_built",
        "beds",
        "baths",
        "stories",
        "pool",
    ):
        if column in panel:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")

    valid = (
        panel["parcel_id"].notna()
        & panel["market"].notna()
        & panel["sale_date"].notna()
        & panel["sale_price"].between(minimum_price, maximum_price, inclusive="both")
    )
    if "living_area_sqft" in panel:
        valid &= panel["living_area_sqft"].isna() | panel["living_area_sqft"].between(
            minimum_area, maximum_area, inclusive="both"
        )
    if require_qualified and "qualified" in panel:
        valid &= _as_boolean(panel["qualified"])
    if require_improved and "improved" in panel:
        valid &= _as_boolean(panel["improved"])
    panel = panel.loc[valid].copy()

    duplicate_key = ["market", "parcel_id", "sale_date", "sale_price"]
    panel = (
        panel.drop_duplicates(duplicate_key, keep="last")
        .sort_values(["sale_date", "market", "parcel_id"], kind="stable")
        .reset_index(drop=True)
    )
    panel["sale_year"] = panel["sale_date"].dt.year.astype(int)
    panel["sale_quarter"] = panel["sale_date"].dt.quarter.astype(int)
    panel["prior_sale_date"] = panel.groupby(["market", "parcel_id"])["sale_date"].shift(1)
    panel["prior_sale_price"] = panel.groupby(["market", "parcel_id"])["sale_price"].shift(1)
    panel["years_since_prior_sale"] = (
        panel["sale_date"] - panel["prior_sale_date"]
    ).dt.days / 365.25
    panel["log_prior_sale_price"] = np.log(panel["prior_sale_price"])

    panel["_calendar_quarter"] = panel["sale_date"].dt.to_period("Q")
    panel["_prior_calendar_quarter"] = panel["prior_sale_date"].dt.to_period("Q")
    panel["_strict_prior_eligible"] = panel["_prior_calendar_quarter"].isna() | panel[
        "_prior_calendar_quarter"
    ].lt(panel["_calendar_quarter"])
    quarterly = (
        panel.groupby(["market", "_calendar_quarter"], observed=True)["sale_price"]
        .median()
        .rename("market_quarter_median")
        .reset_index()
        .sort_values(["market", "_calendar_quarter"])
    )
    quarterly["lagged_market_median_price"] = quarterly.groupby("market")[
        "market_quarter_median"
    ].shift(1)
    panel = panel.merge(
        quarterly[["market", "_calendar_quarter", "lagged_market_median_price"]],
        on=["market", "_calendar_quarter"],
        how="left",
        validate="many_to_one",
    )
    prior_quarterly = quarterly[["market", "_calendar_quarter", "market_quarter_median"]].rename(
        columns={
            "_calendar_quarter": "_prior_calendar_quarter",
            "market_quarter_median": "prior_market_median_price",
        }
    )
    panel = panel.merge(
        prior_quarterly,
        on=["market", "_prior_calendar_quarter"],
        how="left",
        validate="many_to_one",
    )
    panel["log_lagged_market_median"] = np.log(panel["lagged_market_median_price"])
    panel["log_prior_relative_to_market"] = panel["log_prior_sale_price"] - np.log(
        panel["prior_market_median_price"]
    )
    panel["rolled_forward_prior_baseline"] = (
        panel["prior_sale_price"]
        * panel["lagged_market_median_price"]
        / panel["prior_market_median_price"]
    )
    panel = panel.loc[panel["_strict_prior_eligible"]].drop(
        columns=["_calendar_quarter", "_prior_calendar_quarter", "_strict_prior_eligible"]
    )

    if "year_built" in panel:
        effective_year = (
            panel["effective_year_built"]
            if "effective_year_built" in panel
            else pd.Series(np.nan, index=panel.index)
        )
        construction_year = effective_year.where(effective_year.between(1800, panel["sale_year"]))
        construction_year = construction_year.fillna(panel["year_built"])
        panel["property_age_years"] = panel["sale_year"] - construction_year
    if "living_area_sqft" in panel:
        positive_area = panel["living_area_sqft"].where(panel["living_area_sqft"] > 0)
        panel["log_living_area"] = np.log(positive_area)
        panel["nominal_price_per_sqft"] = panel["sale_price"] / positive_area
    return panel


def build_episode_panel(
    episodes: pd.DataFrame,
    *,
    accepted_linkage_statuses: tuple[str, ...] = (
        "completed",
        "right_censored",
        "administrative_horizon",
    ),
) -> pd.DataFrame:
    """Validate observed iBuyer inventory episodes and derive ex-ante features.

    Right-censored and administrative-horizon episodes are retained as
    non-events. Resale prices and gross spreads are realized outcomes, not
    decision-time features and not measures of profit.
    """

    require_columns(episodes, EPISODE_REQUIRED_COLUMNS)
    panel = episodes.copy()
    if "linkage_status" in panel:
        panel["linkage_status"] = panel["linkage_status"].astype("string").str.strip().str.lower()
        panel = panel.loc[panel["linkage_status"].isin(accepted_linkage_statuses)].copy()
    panel["market"] = panel["market"].astype("string").str.strip().str.lower()
    panel["operator"] = panel["operator"].astype("string").str.strip()
    panel["acquisition_date"] = pd.to_datetime(
        panel["acquisition_date"], format="mixed", errors="coerce"
    )
    if "resale_date" in panel:
        panel["resale_date"] = pd.to_datetime(panel["resale_date"], format="mixed", errors="coerce")
    for column in (
        "acquisition_price",
        "resale_price",
        "hold_days",
        "year_built",
        "effective_year_built",
        "living_area_sqft",
        "gross_area_sqft",
        "beds",
        "baths",
        "stories",
        "pool",
    ):
        if column in panel:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["event_observed"] = _as_boolean(panel["event_observed"]).astype(int)

    valid = (
        panel["parcel_id"].notna()
        & panel["market"].notna()
        & panel["operator"].notna()
        & panel["acquisition_date"].notna()
        & panel["acquisition_price"].between(10_000, 20_000_000, inclusive="both")
        & panel["hold_days"].between(0, 3_650, inclusive="both")
    )
    if "resale_date" in panel:
        observed = panel["event_observed"].eq(1)
        valid &= ~observed | (
            panel["resale_date"].notna() & panel["resale_date"].ge(panel["acquisition_date"])
        )
    if "resale_price" in panel:
        valid &= panel["event_observed"].eq(0) | panel["resale_price"].gt(0)
    panel = panel.loc[valid].copy()

    observed = panel["event_observed"].eq(1)
    observed_weeks = np.maximum(1, np.ceil(panel["hold_days"] / 7))
    censored_weeks = np.floor(panel["hold_days"] / 7)
    panel["duration_weeks"] = np.where(observed, observed_weeks, censored_weeks).astype(int)
    panel = panel.loc[observed | panel["duration_weeks"].ge(1)].copy()
    panel["acquisition_year"] = panel["acquisition_date"].dt.year.astype(int)
    panel["acquisition_quarter"] = panel["acquisition_date"].dt.quarter.astype(int)
    panel["log_acquisition_price"] = np.log(panel["acquisition_price"])
    if "living_area_sqft" in panel:
        panel["log_living_area"] = np.log(panel["living_area_sqft"].where(lambda x: x > 0))
    if "year_built" in panel:
        effective_year = (
            panel["effective_year_built"]
            if "effective_year_built" in panel
            else pd.Series(np.nan, index=panel.index)
        )
        construction_year = effective_year.where(
            effective_year.between(1800, panel["acquisition_year"])
        )
        construction_year = construction_year.fillna(panel["year_built"])
        panel["property_age_years"] = panel["acquisition_year"] - construction_year

    if "resale_price" in panel:
        realized = panel["event_observed"].eq(1)
        calculated_spread = panel["resale_price"] - panel["acquisition_price"]
        calculated_return = calculated_spread / panel["acquisition_price"]
        panel["gross_spread"] = calculated_spread.where(realized)
        panel["gross_return"] = calculated_return.where(realized)

    return panel.sort_values(
        ["acquisition_date", "market", "operator", "parcel_id"], kind="stable"
    ).reset_index(drop=True)


def geographic_holdout(
    frame: pd.DataFrame,
    *,
    development_market: str = TAMPA_MARKET,
    holdout_market: str = ORLANDO_MARKET,
    market_col: str = "market",
    parcel_col: str = "parcel_id",
) -> GeographicSplit:
    """Keep Tampa for development and reserve Orlando as a geographic test."""

    require_columns(frame, (market_col, parcel_col))
    if development_market == holdout_market:
        raise ValueError("development and holdout markets must differ")
    normalized_market = frame[market_col].astype("string").str.strip().str.lower()
    development_name = development_market.strip().lower()
    holdout_name = holdout_market.strip().lower()
    development = frame.loc[normalized_market.eq(development_name)].copy()
    holdout = frame.loc[normalized_market.eq(holdout_name)].copy()
    split = GeographicSplit(development, holdout, development_name, holdout_name)

    development_keys = set(
        zip(split.development[market_col], split.development[parcel_col], strict=True)
    )
    holdout_keys = set(zip(split.holdout[market_col], split.holdout[parcel_col], strict=True))
    overlap = development_keys.intersection(holdout_keys)
    if overlap:
        raise ValueError("geographic split contains overlapping market-parcel keys")
    return split


def chronological_grouped_split(
    frame: pd.DataFrame,
    *,
    date_col: str,
    parcel_col: str = "parcel_id",
    calibration_fraction: float = 0.20,
    test_fraction: float = 0.20,
    calibration_start: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
) -> ChronologicalGroupedSplit:
    """Create chronological partitions and prevent parcel reuse across them.

    When a parcel appears in multiple raw time windows, it is retained only in
    the latest window.  This allows a later transaction to use its lagged deed
    history as an ex-ante feature without letting that property's earlier sale
    outcome enter model estimation.
    """

    require_columns(frame, (date_col, parcel_col))
    if frame.empty:
        raise ValueError("chronological split requires at least one row")
    if not 0 < calibration_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("calibration and test fractions must lie between zero and 0.5")
    if calibration_fraction + test_fraction >= 0.8:
        raise ValueError("at least 20 percent of dates must remain for proper training")

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_col} contains missing or invalid dates")
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 5:
        raise ValueError("chronological split requires at least five distinct dates")

    if calibration_start is None:
        calibration_position = int(
            np.floor(len(unique_dates) * (1.0 - calibration_fraction - test_fraction))
        )
        calibration_position = min(max(calibration_position, 1), len(unique_dates) - 2)
        calibration_cutoff = pd.Timestamp(unique_dates[calibration_position])
    else:
        calibration_cutoff = pd.Timestamp(calibration_start)
    if test_start is None:
        test_position = int(np.floor(len(unique_dates) * (1.0 - test_fraction)))
        test_position = min(max(test_position, 2), len(unique_dates) - 1)
        test_cutoff = pd.Timestamp(unique_dates[test_position])
    else:
        test_cutoff = pd.Timestamp(test_start)
    if calibration_cutoff >= test_cutoff:
        raise ValueError("calibration_start must precede test_start")

    raw_training = frame.loc[dates.lt(calibration_cutoff)].copy()
    raw_calibration = frame.loc[dates.ge(calibration_cutoff) & dates.lt(test_cutoff)].copy()
    test = frame.loc[dates.ge(test_cutoff)].copy()

    test_parcels = set(test[parcel_col])
    calibration = raw_calibration.loc[~raw_calibration[parcel_col].isin(test_parcels)].copy()
    later_parcels = test_parcels.union(calibration[parcel_col])
    training = raw_training.loc[~raw_training[parcel_col].isin(later_parcels)].copy()
    purged_training = raw_training.loc[raw_training[parcel_col].isin(later_parcels)]
    purged_calibration = raw_calibration.loc[raw_calibration[parcel_col].isin(test_parcels)]
    split = ChronologicalGroupedSplit(
        proper_training=training,
        calibration=calibration,
        out_of_time_test=test,
        calibration_start=calibration_cutoff,
        test_start=test_cutoff,
        raw_training_rows=len(raw_training),
        raw_calibration_rows=len(raw_calibration),
        raw_test_rows=len(test),
        purged_training_rows=len(purged_training),
        purged_calibration_rows=len(purged_calibration),
        purged_training_parcels=purged_training[parcel_col].nunique(dropna=False),
        purged_calibration_parcels=purged_calibration[parcel_col].nunique(dropna=False),
    )
    _assert_disjoint_parcels(split, parcel_col)
    return split


def _assert_disjoint_parcels(split: ChronologicalGroupedSplit, parcel_col: str) -> None:
    training = set(split.proper_training[parcel_col])
    calibration = set(split.calibration[parcel_col])
    test = set(split.out_of_time_test[parcel_col])
    if (
        training.intersection(calibration)
        or training.intersection(test)
        or calibration.intersection(test)
    ):
        raise RuntimeError("chronological split unexpectedly reused a parcel")


def _as_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y", "qualified", "improved"})
