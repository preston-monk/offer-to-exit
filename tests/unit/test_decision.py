from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from offer_to_exit.decision import (
    AcquisitionOfferOptimizer,
    ConstantElasticitySaleModel,
    CostStructure,
    DynamicPricingOptimizer,
    FittedHazardSaleOutcomeAdapter,
    FittedSellerAcceptanceAdapter,
    FixedSpreadOfferPolicy,
    HoldPricePolicy,
    HomeContext,
    LogisticAcceptanceModel,
    MarketScenarioSpec,
    PriceAction,
    PricingOptimizerConfig,
    PricingState,
    ProfitOutcome,
    SaleScenario,
    ScheduledMarkdownPolicy,
    TargetAcceptanceOfferPolicy,
    ValueScenarioSpec,
    downside_cvar_loss,
    expected_profit,
    lower_tail_mean,
    solve_worked_decision_cases,
    weighted_quantile,
)
from offer_to_exit.decision.risk import compress_outcomes, normalize_outcomes


class _ThresholdSaleModel:
    def sale_scenarios(self, context: HomeContext, state: PricingState) -> tuple[SaleScenario, ...]:
        del context
        list_price = state.list_price
        probability = 0.95 if list_price <= 95 else 0.01
        return (SaleScenario("base", 1.0, probability, list_price),)


class _RiskyAtFullPriceModel:
    def sale_scenarios(self, context: HomeContext, state: PricingState) -> tuple[SaleScenario, ...]:
        del context
        list_price = state.list_price
        if list_price >= 99.5:
            return (
                SaleScenario("upside", 0.9, 1.0, 115.0),
                SaleScenario("loss", 0.1, 1.0, 0.0),
            )
        return (SaleScenario("safe", 1.0, 1.0, list_price + 1.0),)


class _CertainHundredDollarSale:
    def sale_scenarios(self, context: HomeContext, state: PricingState) -> tuple[SaleScenario, ...]:
        del context, state
        return (SaleScenario("certain", 1.0, 1.0, 100.0),)


class _GridAcceptanceModel:
    def acceptance_probability(self, context: HomeContext, offer: float) -> float:
        del context
        return 0.1 if offer <= 60 else 0.9


class _SpyAcceptanceModel:
    features = ("offer_ratio", "seller_urgency")

    def __init__(self) -> None:
        self.calls: list[pd.DataFrame] = []

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls.append(frame.copy())
        return frame["offer_ratio"].to_numpy(dtype=float) / 2


class _SpyHazardModel:
    features = ("list_price_premium", "market_heat")
    max_periods = 17

    def __init__(self) -> None:
        self.calls: list[tuple[pd.DataFrame, int]] = []

    def predict_hazard(self, frame: pd.DataFrame, *, horizon: int | None = None) -> np.ndarray:
        assert horizon is not None
        self.calls.append((frame.copy(), horizon))
        base = np.clip(0.25 - 0.5 * frame["list_price_premium"].to_numpy(), 0.01, 0.99)
        return np.repeat(base[:, None], horizon, axis=1)


def test_price_actions_and_cost_structure() -> None:
    assert PriceAction.HOLD.apply(100_000) == 100_000
    assert PriceAction.CUT_1.apply(100_000) == 99_000
    assert PriceAction.CUT_2_5.apply(100_000) == 97_500
    assert PriceAction.CUT_5.apply(100_000) == 95_000

    costs = CostStructure(
        repair_cost=10,
        weekly_holding_cost=2,
        transaction_cost_rate=0.10,
        fixed_transaction_cost=5,
        acquisition_cost=3,
        prelisting_holding_weeks=1,
    )
    # Sold in listing week 1 means three charged weeks: one pre-listing and two listed.
    assert costs.profit(sale_price=100, acquisition_offer=50, sale_week=1) == 16


def test_dynamic_optimizer_cuts_when_carry_is_expensive() -> None:
    context = HomeContext("threshold-home", 100)
    optimizer = DynamicPricingOptimizer(
        _ThresholdSaleModel(),
        PricingOptimizerConfig(
            horizon_weeks=2,
            risk_aversion=0,
            terminal_liquidation_discount=0.8,
            min_list_price_ratio=0.5,
        ),
    )

    result = optimizer.optimize(
        context=context,
        acquisition_offer=70,
        initial_list_price=100,
        costs=CostStructure(weekly_holding_cost=10),
    )

    assert result.first_decision.action is PriceAction.CUT_5
    assert result.first_decision.new_list_price == 95
    assert [decision.state.week for decision in result.no_sale_path()] == [0, 1]


def test_downside_penalty_can_change_the_first_action() -> None:
    context = HomeContext("risk-home", 100)
    common = dict(
        context=context,
        acquisition_offer=80,
        initial_list_price=100,
        costs=CostStructure(),
    )
    risk_neutral = DynamicPricingOptimizer(
        _RiskyAtFullPriceModel(),
        PricingOptimizerConfig(horizon_weeks=1, risk_aversion=0, min_list_price_ratio=0.5),
    ).optimize(**common)
    risk_averse = DynamicPricingOptimizer(
        _RiskyAtFullPriceModel(),
        PricingOptimizerConfig(horizon_weeks=1, risk_aversion=1, min_list_price_ratio=0.5),
    ).optimize(**common)

    assert risk_neutral.first_decision.action is PriceAction.HOLD
    assert risk_neutral.expected_profit == pytest.approx(23.5)
    assert risk_neutral.downside_cvar_loss == pytest.approx(80)
    assert risk_averse.first_decision.action is PriceAction.CUT_1
    assert risk_averse.downside_cvar_loss == 0


def test_outer_offer_grid_uses_acceptance_and_exit_value() -> None:
    context = HomeContext("offer-home", 100)
    pricing_optimizer = DynamicPricingOptimizer(
        _CertainHundredDollarSale(),
        PricingOptimizerConfig(horizon_weeks=1, risk_aversion=0),
    )
    result = AcquisitionOfferOptimizer(
        pricing_optimizer,
        _GridAcceptanceModel(),
    ).optimize(
        context=context,
        offer_grid=(60, 80),
        initial_list_price=100,
        costs=CostStructure(),
    )

    assert result.selected.offer == 80
    assert result.selected.acceptance_probability == pytest.approx(0.9)
    candidate_value = {
        candidate.offer: candidate.expected_profit_per_lead for candidate in result.candidates
    }
    assert candidate_value == pytest.approx({60.0: 4.0, 80.0: 18.0})


def test_rule_based_baselines_are_explicit_and_evaluable() -> None:
    context = HomeContext("baseline-home", 100)
    state = PricingState(week=2, list_price=100)
    acceptance = LogisticAcceptanceModel(midpoint_offer_ratio=0.9, slope=20)

    assert HoldPricePolicy().choose_action(context, state) is PriceAction.HOLD
    scheduled = ScheduledMarkdownPolicy(
        action=PriceAction.CUT_2_5,
        every_n_weeks=2,
        first_markdown_week=2,
    )
    assert scheduled.choose_action(context, PricingState(1, 100)) is PriceAction.HOLD
    assert scheduled.choose_action(context, state) is PriceAction.CUT_2_5
    assert scheduled.choose_action(context, PricingState(3, 100)) is PriceAction.HOLD
    assert FixedSpreadOfferPolicy(0.1).offer(context) == 90
    target = TargetAcceptanceOfferPolicy(acceptance, (80, 90, 100), 0.5)
    assert target.offer(context) == 90

    optimizer = DynamicPricingOptimizer(
        _CertainHundredDollarSale(), PricingOptimizerConfig(horizon_weeks=1)
    )
    evaluated = optimizer.evaluate_policy(
        HoldPricePolicy(),
        context=context,
        acquisition_offer=80,
        initial_list_price=100,
        costs=CostStructure(),
    )
    assert evaluated.first_decision.action is PriceAction.HOLD


def test_reference_models_are_monotone_and_scenario_aware() -> None:
    context = HomeContext("model-home", 100)
    acceptance = LogisticAcceptanceModel(midpoint_offer_ratio=0.9, slope=20)
    assert acceptance.acceptance_probability(context, 95) > acceptance.acceptance_probability(
        context, 85
    )

    model = ConstantElasticitySaleModel(
        base_weekly_sale_probability=0.2,
        price_elasticity=10,
        scenarios=(MarketScenarioSpec("soft", 1, value_multiplier=0.9, demand_multiplier=0.8),),
    )
    expensive = model.sale_scenarios(context, PricingState(0, 110))[0]
    affordable = model.sale_scenarios(context, PricingState(0, 90))[0]
    assert affordable.sale_probability > expensive.sale_probability
    assert affordable.sale_price <= 90


def test_fitted_adapters_invoke_prediction_methods_and_rewrite_only_price_treatments() -> None:
    acceptance_model = _SpyAcceptanceModel()
    acceptance_features = pd.DataFrame({"offer_ratio": [0.50], "seller_urgency": [0.25]})
    acceptance = FittedSellerAcceptanceAdapter(
        acceptance_model,
        acceptance_features,
        offer_ratio_support=(0.82, 1.02),
    )
    context = HomeContext(
        "LAB-SPY-001",
        100,
        features={
            "preoffer_reference_value": 120,
            "prelisting_reference_value": 100,
        },
    )

    assert acceptance.acceptance_probability(context, 108) == pytest.approx(0.45)
    assert len(acceptance_model.calls) == 1
    assert acceptance_model.calls[0].loc[0, "offer_ratio"] == pytest.approx(0.90)
    assert acceptance_model.calls[0].loc[0, "seller_urgency"] == pytest.approx(0.25)
    assert acceptance_features.loc[0, "offer_ratio"] == pytest.approx(0.50)

    with pytest.raises(KeyError, match="observable offer reference"):
        acceptance.acceptance_probability(HomeContext("missing-reference", 100), 90)
    with pytest.raises(ValueError, match="outside fitted support"):
        acceptance.acceptance_probability(context, 80)

    hazard_model = _SpyHazardModel()
    hazard_features = pd.DataFrame({"list_price_premium": [0.0], "market_heat": [0.1]})
    outcome = FittedHazardSaleOutcomeAdapter(
        hazard_model,
        hazard_features,
        (
            ValueScenarioSpec("downside", 0.2, 90),
            ValueScenarioSpec("base", 0.6, 100),
            ValueScenarioSpec("upside", 0.2, 120),
        ),
        list_price_premium_support=(-0.30, 0.15),
    )
    scenarios = outcome.sale_scenarios(context, PricingState(3, 110))

    assert len(hazard_model.calls) == 1
    scored_frame, horizon = hazard_model.calls[0]
    assert horizon == 17
    assert scored_frame["list_price_premium"].to_numpy() == pytest.approx([0.10])
    assert scored_frame["market_heat"].eq(0.1).all()
    assert [scenario.sale_price for scenario in scenarios] == pytest.approx([90, 100, 109.45])
    assert outcome.scored_price_premiums == pytest.approx((0.10,))
    assert outcome.sale_scenarios(context, PricingState(3, 110)) == scenarios
    assert len(hazard_model.calls) == 1
    with pytest.raises(ValueError, match="outside fitted support"):
        outcome.sale_scenarios(context, PricingState(3, 116))


def test_weighted_tail_risk_and_compression_preserve_economics() -> None:
    outcomes = (
        ProfitOutcome(0.05, -100),
        ProfitOutcome(0.45, 0),
        ProfitOutcome(0.50, 100),
    )
    assert expected_profit(outcomes) == pytest.approx(45)
    assert lower_tail_mean(outcomes, tail_probability=0.1) == pytest.approx(-50)
    assert downside_cvar_loss(outcomes, tail_probability=0.1) == pytest.approx(50)
    assert weighted_quantile(outcomes, 0.5) == 0
    compressed = compress_outcomes(outcomes, max_points=2)
    assert sum(item.probability for item in compressed) == pytest.approx(1)
    assert expected_profit(compressed) == pytest.approx(expected_profit(outcomes))
    assert normalize_outcomes((ProfitOutcome(2, 10),))[0].probability == 1


def test_three_worked_cases_run_through_the_same_interfaces() -> None:
    results = solve_worked_decision_cases()

    assert {result.case.name for result in results} == {
        "healthy-demand-margin-protection",
        "stale-inventory-high-carry",
        "sparse-support-downside-protection",
    }
    assert all(result.selected_offer in result.case.offer_grid for result in results)
    assert all(result.result.selected.exit_result.no_sale_path() for result in results)
    assert all(result.case.pricing_config.horizon_weeks == 17 for result in results)
    assert all(result.case.context.home_id.startswith("LAB-") for result in results)
