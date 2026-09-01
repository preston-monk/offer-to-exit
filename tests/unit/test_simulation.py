import numpy as np
import pandas as pd

from offer_to_exit.simulation import (
    CausalParameters,
    EnvironmentConfig,
    simulate_environment,
    simulate_train_evaluation,
    true_acceptance_probability,
    true_listing_hazard_probability,
)


def _log_odds(probability: np.ndarray) -> np.ndarray:
    return np.log(probability / (1.0 - probability))


def test_environment_generation_is_deterministic() -> None:
    config = EnvironmentConfig(
        name="repeatable",
        seed=19,
        n_homes=450,
        max_listing_periods=9,
    )

    first = simulate_environment(config)
    second = simulate_environment(config)

    pd.testing.assert_frame_equal(first.homes, second.homes)
    pd.testing.assert_frame_equal(first.offers, second.offers)
    pd.testing.assert_frame_equal(first.listings, second.listings)
    pd.testing.assert_frame_equal(first.survival_panel, second.survival_panel)


def test_train_and_evaluation_are_independent_shifted_environments() -> None:
    simulation = simulate_train_evaluation(
        seed=503,
        n_train=600,
        n_evaluation=400,
        max_listing_periods=8,
    )
    repeated = simulate_train_evaluation(
        seed=503,
        n_train=600,
        n_evaluation=400,
        max_listing_periods=8,
    )

    assert simulation.train.config.seed != simulation.evaluation.config.seed
    assert set(simulation.train.homes["home_id"]).isdisjoint(simulation.evaluation.homes["home_id"])
    assert simulation.train.homes["environment"].eq("train").all()
    assert simulation.evaluation.homes["environment"].eq("evaluation").all()
    assert (
        simulation.train.homes["market_heat"].mean()
        > simulation.evaluation.homes["market_heat"].mean()
    )
    assert (
        simulation.train.homes["mortgage_rate"].mean()
        < simulation.evaluation.homes["mortgage_rate"].mean()
    )
    pd.testing.assert_frame_equal(simulation.train.offers, repeated.train.offers)
    pd.testing.assert_frame_equal(simulation.evaluation.listings, repeated.evaluation.listings)


def test_price_treatments_have_known_counterfactual_response() -> None:
    truth = CausalParameters()
    acceptance_frame = pd.DataFrame(
        {
            "offer_ratio": [0.86, 0.96],
            "seller_urgency": [0.2, 0.2],
            "market_heat": [-0.1, -0.1],
            "repair_cost_fraction": [0.04, 0.04],
        }
    )
    acceptance_probability = true_acceptance_probability(acceptance_frame, truth)

    assert acceptance_probability[1] > acceptance_probability[0]
    assert np.isclose(
        np.diff(_log_odds(acceptance_probability)).item(),
        truth.acceptance_log_odds_per_ten_pct,
    )

    listing_frame = pd.DataFrame(
        {
            "list_price_premium": [-0.02, 0.08],
            "market_heat": [0.1, 0.1],
            "condition_score": [3, 3],
            "mortgage_rate": [0.06, 0.06],
            "truth_demand_shock": [0.0, 0.0],
        }
    )
    hazard_probability = true_listing_hazard_probability(listing_frame, period=2, truth=truth)

    assert hazard_probability[1] < hazard_probability[0]
    assert np.isclose(
        np.diff(_log_odds(hazard_probability)).item(),
        truth.hazard_log_odds_per_ten_pct_overpricing,
    )


def test_survival_panel_preserves_events_and_right_censoring() -> None:
    environment = simulate_environment(
        EnvironmentConfig(
            name="censoring_check",
            seed=827,
            n_homes=1_200,
            max_listing_periods=10,
        )
    )
    listings = environment.listings
    panel = environment.survival_panel

    assert listings["event_observed"].eq(1).any()
    assert listings["censored"].eq(1).any()
    assert listings.loc[listings["event_observed"].eq(0), "exit_sale_price"].isna().all()
    assert listings.loc[listings["event_observed"].eq(1), "exit_sale_price"].notna().all()

    panel_rows = panel.groupby("listing_id", sort=False).size()
    events = panel.groupby("listing_id", sort=False)["event_in_period"].sum()
    indexed_listings = listings.set_index("listing_id")
    np.testing.assert_array_equal(
        panel_rows.loc[indexed_listings.index], indexed_listings["listing_periods"]
    )
    np.testing.assert_array_equal(
        events.loc[indexed_listings.index], indexed_listings["event_observed"]
    )
    assert panel["true_hazard_probability"].between(0.0, 1.0).all()
