import numpy as np
import pandas as pd
import pytest

from offer_to_exit.models import (
    CalibratedLinearValuation,
    DiscreteTimeHazardModel,
    SellerAcceptanceModel,
    expand_discrete_time_rows,
)
from offer_to_exit.simulation import TrainEvaluationSimulation, simulate_train_evaluation

VALUATION_NUMERIC_FEATURES = (
    "square_feet",
    "bedrooms",
    "bathrooms",
    "age_years",
    "condition_score",
    "quality_score",
    "lot_square_feet",
    "market_heat",
    "mortgage_rate",
)
ACCEPTANCE_NUMERIC_FEATURES = (
    "offer_ratio",
    "seller_urgency",
    "market_heat",
    "repair_cost_fraction",
)
HAZARD_NUMERIC_FEATURES = (
    "list_price_premium",
    "market_heat",
    "condition_score",
    "mortgage_rate",
)


@pytest.fixture(scope="module")
def simulation() -> TrainEvaluationSimulation:
    return simulate_train_evaluation(
        seed=4_079,
        n_train=2_400,
        n_evaluation=1_100,
        max_listing_periods=10,
    )


def test_valuation_baseline_returns_useful_calibrated_intervals(
    simulation: TrainEvaluationSimulation,
) -> None:
    model = CalibratedLinearValuation(
        VALUATION_NUMERIC_FEATURES,
        ("submarket",),
        alpha=0.10,
        random_state=13,
    ).fit(
        simulation.train.homes,
        simulation.train.homes["observed_sale_price"],
    )

    evaluation = simulation.evaluation.homes
    prediction = model.predict_with_interval(evaluation)
    target = evaluation["observed_sale_price"].to_numpy()
    covered = (target >= prediction["lower"]) & (target <= prediction["upper"])
    absolute_percentage_error = np.abs(prediction["prediction"] - target) / target

    assert model.n_training_ + model.n_calibration_ == len(simulation.train.homes)
    assert (prediction["lower"] < prediction["prediction"]).all()
    assert (prediction["prediction"] < prediction["upper"]).all()
    assert (prediction > 0).all().all()
    assert covered.mean() >= 0.75
    assert np.median(absolute_percentage_error) < 0.10
    assert set(model.coefficient_table().columns) == {"feature", "coefficient"}


def test_acceptance_model_learns_offer_response_out_of_environment(
    simulation: TrainEvaluationSimulation,
) -> None:
    model = SellerAcceptanceModel(
        ACCEPTANCE_NUMERIC_FEATURES,
        ("submarket",),
        regularization=0.25,
        random_state=29,
    ).fit(
        simulation.train.offers,
        simulation.train.offers["seller_accepted"],
    )

    evaluation = simulation.evaluation.offers
    probability = model.predict_proba(evaluation)
    outcome = evaluation["seller_accepted"].to_numpy()
    prevalence_prediction = np.full(
        len(evaluation), simulation.train.offers["seller_accepted"].mean()
    )
    brier_score = np.mean((probability - outcome) ** 2)
    prevalence_brier_score = np.mean((prevalence_prediction - outcome) ** 2)
    curve = model.offer_curve(evaluation.iloc[[0]], [0.84, 0.90, 0.96, 1.02])
    coefficients = model.coefficient_table().set_index("feature")["coefficient"]

    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert brier_score < prevalence_brier_score
    assert curve["acceptance_probability"].is_monotonic_increasing
    assert coefficients["numeric__offer_ratio"] > 0


def test_discrete_time_expansion_handles_censoring_and_truncation() -> None:
    listings = pd.DataFrame(
        {
            "listing_id": ["sold", "censored", "late_sale"],
            "listing_periods": [2, 5, 7],
            "event_observed": [1, 0, 1],
        }
    )

    panel = expand_discrete_time_rows(listings, max_periods=5)

    assert len(panel) == 12
    assert panel.groupby("listing_id")["_event"].sum().to_dict() == {
        "censored": 0,
        "late_sale": 0,
        "sold": 1,
    }
    assert panel.groupby("listing_id")["_period_number"].max().to_dict() == {
        "censored": 5,
        "late_sale": 5,
        "sold": 2,
    }


def test_hazard_model_learns_price_response_and_valid_survival_curve(
    simulation: TrainEvaluationSimulation,
) -> None:
    model = DiscreteTimeHazardModel(
        HAZARD_NUMERIC_FEATURES,
        ("submarket",),
        max_periods=10,
        regularization=0.25,
        random_state=41,
    ).fit(simulation.train.listings)

    representative = simulation.evaluation.listings.iloc[[0]].copy()
    counterfactual = pd.concat([representative, representative], ignore_index=True)
    counterfactual["list_price_premium"] = [-0.03, 0.12]
    hazard = model.predict_hazard(counterfactual)
    survival = model.predict_survival(counterfactual)
    sale_probability = model.predict_sale_probability(counterfactual)
    coefficients = model.coefficient_table().set_index("feature")["coefficient"]

    assert hazard.shape == (2, 10)
    assert np.all((hazard > 0.0) & (hazard < 1.0))
    assert np.all(np.diff(survival, axis=1) <= 0.0)
    assert sale_probability[0] > sale_probability[1]
    assert coefficients["numeric__list_price_premium"] < 0
    assert model.n_panel_rows_ >= len(simulation.train.listings)
    assert model.n_events_ == simulation.train.listings["event_observed"].sum()
