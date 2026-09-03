from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from offer_to_exit.empirical import (
    ORLANDO_MARKET,
    TAMPA_MARKET,
    GroupedConformalValuation,
    NamedIBuyerExitHazard,
    administratively_censor_episode_followup,
    build_episode_panel,
    build_valuation_panel,
    parcel_weighted_valuation_metrics,
    run_geographic_exit_study,
    run_geographic_valuation_study,
    summarize_ibuyer_episodes,
)


def _transactions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(47)
    for market, level in ((TAMPA_MARKET, 220_000), (ORLANDO_MARKET, 255_000)):
        for parcel_number in range(70):
            property_effect = rng.normal(0, 12_000)
            property_type = "single_family" if parcel_number % 4 else "townhome"
            target_year = 2021 if parcel_number < 30 else 2022 if parcel_number < 50 else 2023
            for transaction_number, year in enumerate((target_year - 1, target_year)):
                quarter = parcel_number % 4 + 1
                price = (
                    level
                    + property_effect
                    + (year - 2020) * 14_000
                    + transaction_number * 2_500
                    + rng.normal(0, 3_000)
                )
                rows.append(
                    {
                        "parcel_id": f"{market}-{parcel_number:03d}",
                        "market": market,
                        "sale_date": pd.Timestamp(year, quarter * 3 - 1, 15),
                        "sale_price": price,
                        "property_type_code": property_type,
                        "qualified": True,
                        "improved": True,
                    }
                )
    return pd.DataFrame(rows)


def _episodes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(73)
    operator_days = {"opendoor": 105, "offerpad": 125, "zillow_offers": 145}
    for market, market_shift in ((TAMPA_MARKET, 0), (ORLANDO_MARKET, 12)):
        for operator, base_days in operator_days.items():
            for number in range(28):
                acquisition_date = pd.Timestamp(2021 + number % 4, number % 12 + 1, 1)
                hold_days = max(14, int(base_days + market_shift + rng.normal(0, 18)))
                event = int(number % 9 != 0)
                acquisition_price = 220_000 + 2_000 * number + rng.normal(0, 4_000)
                resale_price = acquisition_price * (1.07 + rng.normal(0, 0.01)) if event else np.nan
                rows.append(
                    {
                        "parcel_id": f"{market}-{operator}-{number:03d}",
                        "market": market,
                        "operator": operator,
                        "acquisition_date": acquisition_date,
                        "acquisition_price": acquisition_price,
                        "resale_date": acquisition_date + pd.Timedelta(days=hold_days)
                        if event
                        else pd.NaT,
                        "resale_price": resale_price,
                        "hold_days": hold_days,
                        "event_observed": event,
                    }
                )
    return pd.DataFrame(rows)


def test_valuation_panel_uses_shared_repeat_sale_and_lagged_market_features() -> None:
    transactions = pd.DataFrame(
        {
            "parcel_id": ["a", "b", "a", "bad"],
            "market": [TAMPA_MARKET] * 4,
            "sale_date": ["2020-01-15", "2020-02-01", "2020-04-15", "2020-05-01"],
            "sale_price": [100_000, 200_000, 120_000, 140_000],
            "qualified": [True, True, True, False],
            "improved": [True, True, True, True],
        }
    )

    panel = build_valuation_panel(transactions)
    repeat = panel.loc[(panel["parcel_id"] == "a") & panel["prior_sale_price"].notna()].iloc[0]

    assert len(panel) == 3
    assert repeat["prior_sale_price"] == 100_000
    assert repeat["years_since_prior_sale"] == pytest.approx(91 / 365.25)
    assert repeat["lagged_market_median_price"] == 150_000
    assert repeat["prior_market_median_price"] == 150_000
    assert repeat["log_prior_relative_to_market"] == pytest.approx(np.log(100_000 / 150_000))
    assert repeat["rolled_forward_prior_baseline"] == pytest.approx(100_000)
    assert panel.loc[panel["sale_date"].dt.quarter.eq(1), "lagged_market_median_price"].isna().all()
    assert "living_area_sqft" not in panel.columns


def test_valuation_panel_requires_prior_sale_from_a_strictly_earlier_quarter() -> None:
    transactions = pd.DataFrame(
        {
            "parcel_id": [
                "same-quarter",
                "comparison",
                "same-quarter",
                "comparison",
                "same-quarter",
            ],
            "market": [TAMPA_MARKET] * 5,
            "sale_date": [
                "2020-01-15",
                "2020-02-01",
                "2020-03-20",
                "2020-04-01",
                "2020-04-15",
            ],
            "sale_price": [100_000, 200_000, 110_000, 220_000, 120_000],
            "qualified": [True] * 5,
            "improved": [True] * 5,
        }
    )

    panel = build_valuation_panel(transactions)

    retained_dates = set(panel.loc[panel["parcel_id"] == "same-quarter", "sale_date"])
    assert retained_dates == {pd.Timestamp("2020-01-15"), pd.Timestamp("2020-04-15")}
    april_repeat = panel.loc[
        (panel["parcel_id"] == "same-quarter") & panel["sale_date"].eq(pd.Timestamp("2020-04-15"))
    ].iloc[0]
    assert april_repeat["prior_sale_date"] == pd.Timestamp("2020-03-20")
    assert april_repeat["prior_sale_price"] == 110_000
    assert april_repeat["prior_market_median_price"] == 110_000


def test_geographic_valuation_is_parcel_grouped_and_orlando_is_never_fit() -> None:
    panel = build_valuation_panel(_transactions())
    study = run_geographic_valuation_study(
        panel,
        random_state=19,
        calibration_start="2022-01-01",
        test_start="2023-01-01",
        numeric_features=(
            "log_prior_sale_price",
            "years_since_prior_sale",
            "log_lagged_market_median",
            "sale_year",
            "sale_quarter",
        ),
        categorical_features=("property_type_code",),
    )
    temporal = study.tampa_temporal_split
    proper_parcels = set(temporal.proper_training["parcel_id"])
    calibration_parcels = set(temporal.calibration["parcel_id"])
    test_parcels = set(temporal.out_of_time_test["parcel_id"])

    assert proper_parcels.isdisjoint(calibration_parcels)
    assert proper_parcels.isdisjoint(test_parcels)
    assert calibration_parcels.isdisjoint(test_parcels)
    assert set(study.split.development["market"]) == {TAMPA_MARKET}
    assert set(study.split.holdout["market"]) == {ORLANDO_MARKET}
    assert study.orlando_holdout_metrics.n_parcels > 0
    assert study.orlando_holdout_metrics.mean_absolute_percentage_error < 0.20
    assert 0 <= study.orlando_holdout_metrics.interval_coverage <= 1
    assert (study.orlando_holdout_predictions["lower"] > 0).all()
    assert study.tampa_out_of_time_metrics.n_transactions == len(temporal.out_of_time_test)
    assert temporal.purged_training_parcels == 0
    assert study.model.n_calibration_scores_ == study.model.n_calibration_parcels_
    payload = study.metrics_payload()
    assert payload["design"]["test_start"] == temporal.test_start.isoformat()
    assert payload["orlando_common_window"]["n_transactions"] == len(study.split.holdout)
    assert "orlando_rolled_prior_baseline" in payload


def test_parcel_weighted_metrics_do_not_let_repeat_sales_dominate() -> None:
    scored = pd.DataFrame(
        {
            "parcel_id": ["repeat", "repeat", "repeat", "single"],
            "sale_price": [100.0, 100.0, 100.0, 100.0],
            "prediction": [100.0, 100.0, 130.0, 130.0],
        }
    )
    metrics = parcel_weighted_valuation_metrics(scored, lower_col=None, upper_col=None)

    assert metrics.mean_absolute_error == pytest.approx(20.0)
    assert metrics.n_transactions == 4
    assert metrics.n_parcels == 2


def test_episode_summary_accounts_for_censoring_and_labels_gross_spreads() -> None:
    episodes = _episodes()
    episodes["linkage_status"] = "completed"
    episodes.loc[episodes.index[-1], "linkage_status"] = "administrative_horizon"
    panel = build_episode_panel(episodes)
    summary = summarize_ibuyer_episodes(panel)

    assert len(summary) == 6
    assert summary["n_acquisitions"].sum() == len(episodes)
    assert set(panel["linkage_status"]) == {"completed", "administrative_horizon"}
    administrative = panel.loc[panel["linkage_status"].eq("administrative_horizon")].iloc[0]
    assert administrative["event_observed"] == 0
    assert summary["km_exit_probability_120d"].between(0, 1).all()
    assert summary["median_gross_spread"].notna().all()
    assert panel.loc[panel["event_observed"].eq(0), "gross_spread"].isna().all()
    assert (panel.loc[panel["event_observed"].eq(1), "gross_return"] > 0).all()


def test_episode_panel_uses_complete_weeks_for_censoring_and_keeps_same_day_exit() -> None:
    episodes = pd.DataFrame(
        [
            {
                "parcel_id": "short-censored",
                "market": TAMPA_MARKET,
                "operator": "opendoor",
                "acquisition_date": "2024-01-01",
                "acquisition_price": 250_000,
                "resale_date": pd.NaT,
                "resale_price": np.nan,
                "hold_days": 6,
                "event_observed": 0,
            },
            {
                "parcel_id": "one-week-censored",
                "market": TAMPA_MARKET,
                "operator": "opendoor",
                "acquisition_date": "2024-01-01",
                "acquisition_price": 250_000,
                "resale_date": pd.NaT,
                "resale_price": np.nan,
                "hold_days": 7,
                "event_observed": 0,
            },
            {
                "parcel_id": "same-day-exit",
                "market": TAMPA_MARKET,
                "operator": "offerpad",
                "acquisition_date": "2024-02-01",
                "acquisition_price": 300_000,
                "resale_date": "2024-02-01",
                "resale_price": 315_000,
                "hold_days": 0,
                "event_observed": 1,
            },
            {
                "parcel_id": "eight-day-exit",
                "market": TAMPA_MARKET,
                "operator": "offerpad",
                "acquisition_date": "2024-03-01",
                "acquisition_price": 300_000,
                "resale_date": "2024-03-09",
                "resale_price": 315_000,
                "hold_days": 8,
                "event_observed": 1,
            },
        ]
    )

    panel = build_episode_panel(episodes)

    durations = panel.set_index("parcel_id")["duration_weeks"].to_dict()
    assert "short-censored" not in durations
    assert durations["one-week-censored"] == 1
    assert durations["same-day-exit"] == 1
    assert durations["eight-day-exit"] == 2


def test_training_followup_is_censored_at_the_test_boundary() -> None:
    episodes = pd.DataFrame(
        [
            {
                "parcel_id": "before",
                "acquisition_date": pd.Timestamp("2021-11-01"),
                "resale_date": pd.Timestamp("2021-12-01"),
                "resale_price": 330_000.0,
                "gross_spread": 30_000.0,
                "gross_return": 0.10,
                "hold_days": 30,
                "duration_weeks": 5,
                "event_observed": 1,
            },
            {
                "parcel_id": "on-cutoff",
                "acquisition_date": pd.Timestamp("2021-12-01"),
                "resale_date": pd.Timestamp("2022-01-01"),
                "resale_price": 330_000.0,
                "gross_spread": 30_000.0,
                "gross_return": 0.10,
                "hold_days": 31,
                "duration_weeks": 5,
                "event_observed": 1,
            },
            {
                "parcel_id": "after",
                "acquisition_date": pd.Timestamp("2021-12-15"),
                "resale_date": pd.Timestamp("2022-01-15"),
                "resale_price": 330_000.0,
                "gross_spread": 30_000.0,
                "gross_return": 0.10,
                "hold_days": 31,
                "duration_weeks": 5,
                "event_observed": 1,
            },
            {
                "parcel_id": "already-censored",
                "acquisition_date": pd.Timestamp("2021-12-01"),
                "resale_date": pd.NaT,
                "resale_price": np.nan,
                "gross_spread": np.nan,
                "gross_return": np.nan,
                "hold_days": 60,
                "duration_weeks": 8,
                "event_observed": 0,
            },
            {
                "parcel_id": "incomplete-week",
                "acquisition_date": pd.Timestamp("2021-12-29"),
                "resale_date": pd.Timestamp("2022-01-15"),
                "resale_price": 330_000.0,
                "gross_spread": 30_000.0,
                "gross_return": 0.10,
                "hold_days": 17,
                "duration_weeks": 3,
                "event_observed": 1,
            },
        ]
    )

    censored, n_truncated = administratively_censor_episode_followup(
        episodes,
        cutoff="2022-01-01",
    )
    indexed = censored.set_index("parcel_id")

    assert n_truncated == 4
    assert set(indexed.index) == {"before", "on-cutoff", "after", "already-censored"}
    assert indexed.loc["before", "event_observed"] == 1
    assert indexed.loc["before", "resale_date"] == pd.Timestamp("2021-12-01")
    assert indexed.loc["on-cutoff", "event_observed"] == 0
    assert indexed.loc["on-cutoff", "duration_weeks"] == 4
    assert pd.isna(indexed.loc["on-cutoff", "resale_date"])
    assert indexed.loc["after", "hold_days"] == 17
    assert indexed.loc["after", "duration_weeks"] == 2
    assert pd.isna(indexed.loc["after", "resale_price"])
    assert indexed.loc["already-censored", "hold_days"] == 31
    assert indexed.loc["already-censored", "duration_weeks"] == 4


def test_operator_duration_model_transports_from_tampa_to_orlando() -> None:
    panel = build_episode_panel(_episodes())
    study = run_geographic_exit_study(
        panel,
        numeric_features=(
            "log_acquisition_price",
            "acquisition_year",
            "acquisition_quarter",
        ),
        categorical_features=(),
        max_weeks=30,
        regularization=0.5,
        random_state=31,
    )
    effects = study.model.operator_effects()
    probability = study.model.predict_exit_probability(study.split.holdout.iloc[:4], horizon=20)

    assert set(effects["operator"]) == {"offerpad", "opendoor", "zillow_offers"}
    assert effects["reference_operator"].sum() == 1
    assert np.isfinite(effects["hazard_odds_ratio"]).all()
    assert np.all((probability > 0) & (probability < 1))
    assert study.orlando_holdout_metrics.n_episodes > 0
    assert study.tampa_out_of_time_metrics.n_episodes > 0
    assert study.tampa_temporal_split.test_start <= study.split.holdout["acquisition_date"].min()
    assert np.isfinite(study.orlando_holdout_metrics.person_period_brier_score)
    assert 0 <= study.orlando_holdout_metrics.risk_row_auc <= 1
    assert study.metrics_payload()["tampa_out_of_time"]["n_episodes"] > 0
    assert study.training_followup_cutoff == study.tampa_temporal_split.test_start
    assert study.n_training_spells_truncated > 0
    assert study.n_training_spells_dropped_without_complete_week >= 0
    assert study.latest_training_observed_exit is not None
    assert study.latest_training_observed_exit < study.training_followup_cutoff
    assert (
        study.metrics_payload()["design"]["training_followup_cutoff"]
        == study.tampa_temporal_split.test_start.isoformat()
    )


def test_empirical_models_reject_realized_outcome_leakage() -> None:
    with pytest.raises(ValueError, match="sale outcomes"):
        GroupedConformalValuation(("sale_price",))
    with pytest.raises(ValueError, match="realized outcomes"):
        NamedIBuyerExitHazard(("gross_return",))
