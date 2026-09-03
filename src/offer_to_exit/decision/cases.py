"""Three worked cases that exercise distinct pricing trade-offs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .models import (
    ConstantElasticitySaleModel,
    LogisticAcceptanceModel,
    MarketScenarioSpec,
)
from .optimizer import (
    AcquisitionOfferOptimizer,
    AcquisitionOptimizerConfig,
    DynamicPricingOptimizer,
    PricingOptimizerConfig,
)
from .protocols import AcceptanceModel, SaleOutcomeModel
from .types import (
    AcquisitionOptimizationResult,
    CostStructure,
    HomeContext,
    PriceAction,
)


@dataclass(frozen=True)
class WorkedDecisionCase:
    """Inputs and explanation for a reproducible decision example."""

    name: str
    question: str
    context: HomeContext
    costs: CostStructure
    initial_list_price: float
    offer_grid: tuple[float, ...]
    outcome_model: SaleOutcomeModel
    acceptance_model: AcceptanceModel
    pricing_config: PricingOptimizerConfig


@dataclass(frozen=True)
class WorkedCaseModels:
    """Model adapters and observable context for one worked decision profile."""

    context: HomeContext
    outcome_model: SaleOutcomeModel
    acceptance_model: AcceptanceModel


@dataclass(frozen=True)
class WorkedDecisionResult:
    """Solved case with the headline decisions exposed for presentation."""

    case: WorkedDecisionCase
    result: AcquisitionOptimizationResult

    @property
    def selected_offer(self) -> float:
        return self.result.selected.offer

    @property
    def first_action(self) -> PriceAction:
        return self.result.selected.exit_result.first_decision.action


def worked_decision_cases() -> tuple[WorkedDecisionCase, ...]:
    """Return healthy, stale, and downside-risk pricing cases."""

    healthy = WorkedDecisionCase(
        name="healthy-demand-margin-protection",
        question="How much price should we preserve when demand is healthy?",
        context=HomeContext(
            home_id="LAB-HEALTHY-001",
            reference_value=400_000,
            features={"demand_log_odds_shift": 0.35},
        ),
        costs=CostStructure(
            repair_cost=11_000,
            weekly_holding_cost=650,
            transaction_cost_rate=0.035,
            fixed_transaction_cost=1_500,
            acquisition_cost=2_000,
            prelisting_holding_weeks=2,
        ),
        initial_list_price=410_000,
        offer_grid=(348_000, 356_000, 364_000, 372_000, 380_000),
        outcome_model=ConstantElasticitySaleModel(
            base_weekly_sale_probability=0.22,
            price_elasticity=8.0,
            dom_log_odds_per_week=0.02,
            scenarios=(
                MarketScenarioSpec("soft", 0.20, 0.97, 0.80),
                MarketScenarioSpec("base", 0.60, 1.00, 1.00),
                MarketScenarioSpec("strong", 0.20, 1.03, 1.20),
            ),
        ),
        acceptance_model=LogisticAcceptanceModel(
            midpoint_offer_ratio=0.91,
            slope=28,
        ),
        pricing_config=PricingOptimizerConfig(
            horizon_weeks=17,
            risk_aversion=0.20,
            terminal_liquidation_discount=0.94,
        ),
    )

    stale = WorkedDecisionCase(
        name="stale-inventory-high-carry",
        question="When does a deeper markdown beat another costly week on market?",
        context=HomeContext(
            home_id="LAB-STALE-002",
            reference_value=600_000,
            features={"demand_log_odds_shift": -0.30},
        ),
        costs=CostStructure(
            repair_cost=18_000,
            weekly_holding_cost=1_800,
            transaction_cost_rate=0.04,
            fixed_transaction_cost=2_000,
            acquisition_cost=3_000,
            prelisting_holding_weeks=3,
        ),
        initial_list_price=645_000,
        offer_grid=(510_000, 525_000, 540_000, 555_000, 570_000),
        outcome_model=ConstantElasticitySaleModel(
            base_weekly_sale_probability=0.09,
            price_elasticity=14.0,
            dom_log_odds_per_week=-0.01,
            scenarios=(
                MarketScenarioSpec("soft", 0.35, 0.94, 0.65),
                MarketScenarioSpec("base", 0.50, 0.98, 0.90),
                MarketScenarioSpec("rebound", 0.15, 1.02, 1.10),
            ),
        ),
        acceptance_model=LogisticAcceptanceModel(
            midpoint_offer_ratio=0.90,
            slope=26,
            intercept=0.10,
        ),
        pricing_config=PricingOptimizerConfig(
            horizon_weeks=17,
            risk_aversion=0.45,
            terminal_liquidation_discount=0.90,
        ),
    )

    downside = WorkedDecisionCase(
        name="sparse-support-downside-protection",
        question="How should uncertainty change both the offer and exit policy?",
        context=HomeContext(
            home_id="LAB-SPARSE-SUPPORT-003",
            reference_value=325_000,
            features={
                "demand_log_odds_shift": -0.10,
                "seller_urgency_log_odds": 0.15,
            },
        ),
        costs=CostStructure(
            repair_cost=15_000,
            weekly_holding_cost=900,
            transaction_cost_rate=0.04,
            fixed_transaction_cost=1_250,
            acquisition_cost=2_250,
            prelisting_holding_weeks=2,
        ),
        initial_list_price=338_000,
        offer_grid=(266_500, 276_250, 286_000, 295_750, 305_500),
        outcome_model=ConstantElasticitySaleModel(
            base_weekly_sale_probability=0.14,
            price_elasticity=10.0,
            dom_log_odds_per_week=0.0,
            scenarios=(
                MarketScenarioSpec("downside", 0.30, 0.86, 0.55),
                MarketScenarioSpec("base", 0.50, 1.00, 1.00),
                MarketScenarioSpec("upside", 0.20, 1.08, 1.20),
            ),
        ),
        acceptance_model=LogisticAcceptanceModel(
            midpoint_offer_ratio=0.89,
            slope=24,
        ),
        pricing_config=PricingOptimizerConfig(
            horizon_weeks=17,
            tail_probability=0.10,
            risk_aversion=0.90,
            terminal_liquidation_discount=0.88,
        ),
    )
    return healthy, stale, downside


def model_driven_worked_decision_cases(
    model_inputs: Mapping[str, WorkedCaseModels],
    *,
    offer_ratio_support: tuple[float, float],
    list_price_premium_support: tuple[float, float],
) -> tuple[WorkedDecisionCase, ...]:
    """Bind fitted-model adapters to the three economic decision profiles."""

    expected = {
        "healthy-demand-margin-protection",
        "stale-inventory-high-carry",
        "sparse-support-downside-protection",
    }
    if set(model_inputs) != expected:
        raise ValueError(f"model_inputs must contain exactly these profiles: {sorted(expected)}")

    healthy_models = model_inputs["healthy-demand-margin-protection"]
    healthy_value = healthy_models.context.reference_value
    stale_models = model_inputs["stale-inventory-high-carry"]
    stale_value = stale_models.context.reference_value
    downside_models = model_inputs["sparse-support-downside-protection"]
    downside_value = downside_models.context.reference_value
    minimum_list_price_ratio = 1.0 + list_price_premium_support[0]

    return (
        WorkedDecisionCase(
            name="healthy-demand-margin-protection",
            question="How much price should we preserve when demand is healthy?",
            context=healthy_models.context,
            costs=CostStructure(
                repair_cost=_scaled(healthy_value, 0.0275),
                weekly_holding_cost=_scaled(healthy_value, 0.001625),
                transaction_cost_rate=0.035,
                fixed_transaction_cost=_scaled(healthy_value, 0.00375),
                acquisition_cost=_scaled(healthy_value, 0.005),
                prelisting_holding_weeks=2,
            ),
            initial_list_price=_fitted_initial_list_price(
                healthy_models.context,
                1.025,
                list_price_premium_support,
            ),
            offer_grid=_fitted_acceptance_offer_grid(
                healthy_models.context,
                (0.87, 0.89, 0.91, 0.93, 0.95),
                offer_ratio_support,
            ),
            outcome_model=healthy_models.outcome_model,
            acceptance_model=healthy_models.acceptance_model,
            pricing_config=PricingOptimizerConfig(
                horizon_weeks=17,
                risk_aversion=0.20,
                terminal_liquidation_discount=0.94,
                min_list_price_ratio=minimum_list_price_ratio,
            ),
        ),
        WorkedDecisionCase(
            name="stale-inventory-high-carry",
            question="When does a deeper markdown beat another costly week on market?",
            context=stale_models.context,
            costs=CostStructure(
                repair_cost=_scaled(stale_value, 0.03),
                weekly_holding_cost=_scaled(stale_value, 0.003),
                transaction_cost_rate=0.04,
                fixed_transaction_cost=_scaled(stale_value, 0.0033),
                acquisition_cost=_scaled(stale_value, 0.005),
                prelisting_holding_weeks=3,
            ),
            initial_list_price=_fitted_initial_list_price(
                stale_models.context,
                1.075,
                list_price_premium_support,
            ),
            offer_grid=_fitted_acceptance_offer_grid(
                stale_models.context,
                (0.85, 0.875, 0.90, 0.925, 0.95),
                offer_ratio_support,
            ),
            outcome_model=stale_models.outcome_model,
            acceptance_model=stale_models.acceptance_model,
            pricing_config=PricingOptimizerConfig(
                horizon_weeks=17,
                risk_aversion=0.45,
                terminal_liquidation_discount=0.90,
                min_list_price_ratio=minimum_list_price_ratio,
            ),
        ),
        WorkedDecisionCase(
            name="sparse-support-downside-protection",
            question="How should uncertainty change both the offer and exit policy?",
            context=downside_models.context,
            costs=CostStructure(
                repair_cost=_scaled(downside_value, 0.046),
                weekly_holding_cost=_scaled(downside_value, 0.00277),
                transaction_cost_rate=0.04,
                fixed_transaction_cost=_scaled(downside_value, 0.00385),
                acquisition_cost=_scaled(downside_value, 0.0069),
                prelisting_holding_weeks=2,
            ),
            initial_list_price=_fitted_initial_list_price(
                downside_models.context,
                1.04,
                list_price_premium_support,
            ),
            offer_grid=_fitted_acceptance_offer_grid(
                downside_models.context,
                (0.83, 0.85, 0.88, 0.91, 0.94),
                offer_ratio_support,
            ),
            outcome_model=downside_models.outcome_model,
            acceptance_model=downside_models.acceptance_model,
            pricing_config=PricingOptimizerConfig(
                horizon_weeks=17,
                tail_probability=0.10,
                risk_aversion=0.90,
                terminal_liquidation_discount=0.88,
                min_list_price_ratio=minimum_list_price_ratio,
            ),
        ),
    )


def solve_worked_decision_cases(
    cases: Sequence[WorkedDecisionCase] | None = None,
) -> tuple[WorkedDecisionResult, ...]:
    """Solve all examples through the same offer-to-exit interfaces."""

    solved: list[WorkedDecisionResult] = []
    for case in worked_decision_cases() if cases is None else cases:
        pricing_optimizer = DynamicPricingOptimizer(case.outcome_model, case.pricing_config)
        offer_optimizer = AcquisitionOfferOptimizer(
            pricing_optimizer,
            case.acceptance_model,
            AcquisitionOptimizerConfig(
                tail_probability=case.pricing_config.tail_probability,
                risk_aversion=case.pricing_config.risk_aversion,
            ),
        )
        solved.append(
            WorkedDecisionResult(
                case=case,
                result=offer_optimizer.optimize(
                    context=case.context,
                    offer_grid=case.offer_grid,
                    initial_list_price=case.initial_list_price,
                    costs=case.costs,
                ),
            )
        )
    return tuple(solved)


def _scaled(reference_value: float, ratio: float) -> float:
    return round(reference_value * ratio, 2)


def _offer_grid(reference_value: float, ratios: Sequence[float]) -> tuple[float, ...]:
    return tuple(_scaled(reference_value, ratio) for ratio in ratios)


def _fitted_acceptance_offer_grid(
    context: HomeContext,
    ratios: Sequence[float],
    support: tuple[float, float],
) -> tuple[float, ...]:
    """Express fitted-model candidate bids in the acceptance model's denominator."""

    try:
        offer_reference = float(context.features["preoffer_reference_value"])
    except KeyError as error:
        raise KeyError("fitted decision context is missing preoffer_reference_value") from error
    if offer_reference <= 0:
        raise ValueError("preoffer_reference_value must be positive")
    support_lower, support_upper = support
    if any(not support_lower <= ratio <= support_upper for ratio in ratios):
        raise ValueError("fitted acquisition-offer grid lies outside randomized support")
    return _offer_grid(offer_reference, ratios)


def _fitted_initial_list_price(
    context: HomeContext,
    ratio: float,
    premium_support: tuple[float, float],
) -> float:
    """Express the initial ask in the fitted hazard's observable denominator."""

    try:
        list_price_reference = float(context.features["prelisting_reference_value"])
    except KeyError as error:
        raise KeyError("fitted decision context is missing prelisting_reference_value") from error
    if list_price_reference <= 0:
        raise ValueError("prelisting_reference_value must be positive")
    support_lower, support_upper = premium_support
    if not support_lower <= ratio - 1.0 <= support_upper:
        raise ValueError("fitted initial list price lies outside randomized support")
    return _scaled(list_price_reference, ratio)
