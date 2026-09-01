"""Three worked cases that exercise distinct pricing trade-offs."""

from __future__ import annotations

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
    outcome_model: ConstantElasticitySaleModel
    acceptance_model: LogisticAcceptanceModel
    pricing_config: PricingOptimizerConfig


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
            home_id="PHX-HEALTHY-001",
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
            horizon_weeks=12,
            risk_aversion=0.20,
            terminal_liquidation_discount=0.94,
        ),
    )

    stale = WorkedDecisionCase(
        name="stale-inventory-high-carry",
        question="When does a deeper markdown beat another costly week on market?",
        context=HomeContext(
            home_id="PHX-STALE-002",
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
            horizon_weeks=12,
            risk_aversion=0.45,
            terminal_liquidation_discount=0.90,
        ),
    )

    downside = WorkedDecisionCase(
        name="thin-comps-downside-protection",
        question="How should uncertainty change both the offer and exit policy?",
        context=HomeContext(
            home_id="PHX-THIN-COMPS-003",
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
            horizon_weeks=12,
            tail_probability=0.15,
            risk_aversion=0.90,
            terminal_liquidation_discount=0.88,
        ),
    )
    return healthy, stale, downside


def solve_worked_decision_cases() -> tuple[WorkedDecisionResult, ...]:
    """Solve all examples through the same offer-to-exit interfaces."""

    solved: list[WorkedDecisionResult] = []
    for case in worked_decision_cases():
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
