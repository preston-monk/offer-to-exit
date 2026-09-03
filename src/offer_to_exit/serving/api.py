"""FastAPI boundary for location-neutral controlled-experiment illustrations.

The API deliberately reuses the worked-case decision models. It does not load a
model binary, call a remote service, or claim to score a real property. Request
fields first pass the public Pydantic contract and then a narrower v0.2 support
gate before any recommendation can be returned.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from math import log, prod
from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from offer_to_exit import __version__
from offer_to_exit.decision import (
    AcquisitionOfferOptimizer,
    AcquisitionOptimizerConfig,
    ConstantElasticitySaleModel,
    CostStructure,
    DynamicPricingOptimizer,
    HomeContext,
    WorkedDecisionResult,
    solve_worked_decision_cases,
)
from offer_to_exit.simulation import LIST_PRICE_PREMIUM_SUPPORT, OFFER_RATIO_SUPPORT

from .schemas import PricingRequest, PricingResponse, WeeklyAction

_EVIDENCE_HEADER = "X-Offer-to-Exit-Evidence"
_EVIDENCE_VALUE = "controlled-experiment"
_MAX_INTERVAL_WIDTH_RATE = 0.20


class HealthResponse(BaseModel):
    """Small operational response that does not execute a pricing decision."""

    status: Literal["ok"]
    version: str
    evidence: Literal["controlled-experiment"]
    worked_case_count: int


@lru_cache(maxsize=1)
def _solved_cases() -> tuple[WorkedDecisionResult, ...]:
    """Solve and cache the transparent examples used as API templates."""

    return solve_worked_decision_cases()


def _select_case(request: PricingRequest) -> WorkedDecisionResult:
    """Choose the nearest value-scale worked case with deterministic tie-breaking."""

    return min(
        _solved_cases(),
        key=lambda item: (
            abs(log(request.estimated_market_value / item.case.context.reference_value)),
            item.case.name,
        ),
    )


def _support_violations(request: PricingRequest) -> list[str]:
    """Return valid-schema inputs that fall outside the narrow v0.2 domain."""

    violations: list[str] = []
    if not 100_000 <= request.estimated_market_value <= 1_250_000:
        violations.append("estimated_market_value outside v0.2 support [$100k, $1.25m]")
    if request.repair_cost / request.estimated_market_value > 0.20:
        violations.append("repair_cost exceeds 20% of estimated market value")
    if request.weekly_holding_cost / request.estimated_market_value > 0.01:
        violations.append("weekly_holding_cost exceeds 1% of estimated market value")
    if request.risk_aversion > 2.0:
        violations.append("risk_aversion outside declared illustrative support [0, 2]")
    return violations


def _scenario_interval_width_rate(template: WorkedDecisionResult) -> float:
    """Use the worked case's explicit value scenarios as an interval-width proxy."""

    outcome_model = template.case.outcome_model
    if not isinstance(outcome_model, ConstantElasticitySaleModel):
        raise TypeError("the illustrative API requires explicit worked-case scenarios")
    multipliers = tuple(scenario.value_multiplier for scenario in outcome_model.scenarios)
    return max(multipliers) - min(multipliers)


def _adapt_and_solve(
    request: PricingRequest,
    template: WorkedDecisionResult,
) -> WorkedDecisionResult:
    """Re-solve one worked case at the request's scale and transparent costs."""

    case = template.case
    scale = request.estimated_market_value / case.context.reference_value
    context = HomeContext(
        home_id=request.property_id,
        reference_value=request.estimated_market_value,
        features={
            **case.context.features,
            "preoffer_reference_value": request.estimated_market_value,
            "prelisting_reference_value": request.estimated_market_value,
        },
    )
    template_offer_reference = float(
        case.context.features.get("preoffer_reference_value", case.context.reference_value)
    )
    offer_ratios = tuple(offer / template_offer_reference for offer in case.offer_grid)
    support_lower, support_upper = OFFER_RATIO_SUPPORT
    if any(not support_lower <= ratio <= support_upper for ratio in offer_ratios):
        raise ValueError("worked-case offer grid falls outside controlled treatment support")
    offer_grid_values: list[float] = []
    for ratio in offer_ratios:
        offer = round(request.estimated_market_value * ratio, 2)
        realized_ratio = offer / request.estimated_market_value
        if realized_ratio < support_lower:
            offer = request.estimated_market_value * support_lower
        elif realized_ratio > support_upper:
            offer = request.estimated_market_value * support_upper
        offer_grid_values.append(offer)
    offer_grid = tuple(offer_grid_values)
    template_list_reference = float(
        case.context.features.get("prelisting_reference_value", case.context.reference_value)
    )
    initial_list_price_ratio = case.initial_list_price / template_list_reference
    list_support_lower, list_support_upper = LIST_PRICE_PREMIUM_SUPPORT
    initial_list_price_premium = initial_list_price_ratio - 1.0
    if not list_support_lower <= initial_list_price_premium <= list_support_upper:
        raise ValueError(
            "worked-case initial list price falls outside controlled treatment support"
        )
    initial_list_price = request.estimated_market_value * initial_list_price_ratio
    costs = CostStructure(
        repair_cost=request.repair_cost,
        weekly_holding_cost=request.weekly_holding_cost,
        transaction_cost_rate=case.costs.transaction_cost_rate,
        fixed_transaction_cost=case.costs.fixed_transaction_cost * scale,
        acquisition_cost=case.costs.acquisition_cost * scale,
        prelisting_holding_weeks=case.costs.prelisting_holding_weeks,
    )
    pricing_config = replace(
        case.pricing_config,
        risk_aversion=request.risk_aversion,
        min_list_price_ratio=1.0 + list_support_lower,
    )
    pricing_optimizer = DynamicPricingOptimizer(case.outcome_model, pricing_config)
    offer_optimizer = AcquisitionOfferOptimizer(
        pricing_optimizer,
        case.acceptance_model,
        AcquisitionOptimizerConfig(
            tail_probability=pricing_config.tail_probability,
            risk_aversion=request.risk_aversion,
        ),
    )
    result = offer_optimizer.optimize(
        context=context,
        offer_grid=offer_grid,
        initial_list_price=initial_list_price,
        costs=costs,
    )
    adapted_case = replace(
        case,
        context=context,
        costs=costs,
        initial_list_price=initial_list_price,
        offer_grid=offer_grid,
        pricing_config=pricing_config,
    )
    return WorkedDecisionResult(case=adapted_case, result=result)


def _probability_of_loss_per_lead(result: WorkedDecisionResult) -> float:
    """Approximate loss probability from the compressed lead-level outcome grid."""

    outcomes = result.result.selected.outcomes
    return sum(outcome.probability for outcome in outcomes if outcome.profit < 0)


def _sale_probability_on_no_sale_path(result: WorkedDecisionResult) -> float:
    path = result.result.selected.exit_result.no_sale_path()
    probability_unsold = prod(1.0 - item.weekly_sale_probability for item in path)
    return 1.0 - probability_unsold


def _policy(result: WorkedDecisionResult) -> list[WeeklyAction]:
    return [
        WeeklyAction(
            week=item.state.week,
            action=item.action.value,
            list_price=round(item.new_list_price, 2),
            sale_probability=round(item.weekly_sale_probability, 6),
        )
        for item in result.result.selected.exit_result.no_sale_path()
    ]


def _non_action_response(
    request: PricingRequest,
    *,
    recommendation: Literal["review", "abstain"],
    reasons: list[str],
) -> PricingResponse:
    return PricingResponse(
        property_id=request.property_id,
        recommendation=recommendation,
        acquisition_offer=None,
        expected_profit=0.0,
        probability_of_loss=0.0,
        sale_by_120_days=0.0,
        policy=[],
        reasons=[
            "controlled-experiment illustration; not a live appraisal or transaction offer",
            *reasons,
        ],
    )


def price_request(request: PricingRequest) -> PricingResponse:
    """Return a deterministic illustration or a structured no-action decision."""

    violations = _support_violations(request)
    if violations:
        return _non_action_response(
            request,
            recommendation="review",
            reasons=["outside location-neutral v0.2 illustrative support", *violations],
        )

    template = _select_case(request)
    interval_width_rate = _scenario_interval_width_rate(template)
    if interval_width_rate > _MAX_INTERVAL_WIDTH_RATE:
        return _non_action_response(
            request,
            recommendation="abstain",
            reasons=[
                f"worked_case={template.case.name}",
                (
                    "illustrative scenario interval is too wide: "
                    f"{interval_width_rate:.1%} > {_MAX_INTERVAL_WIDTH_RATE:.1%}"
                ),
            ],
        )

    solved = _adapt_and_solve(request, template)
    selected = solved.result.selected
    if selected.objective_value <= 0:
        return _non_action_response(
            request,
            recommendation="abstain",
            reasons=[
                f"worked_case={template.case.name}",
                (
                    "best supported offer has non-positive risk-adjusted value: "
                    f"{selected.objective_value:,.2f}"
                ),
            ],
        )

    conditional_expected_profit = selected.exit_result.expected_profit
    conditional_expected_margin = conditional_expected_profit / selected.offer
    if conditional_expected_margin < request.target_margin:
        return _non_action_response(
            request,
            recommendation="review",
            reasons=[
                f"worked_case={template.case.name}",
                (
                    "illustrative conditional expected margin does not meet target: "
                    f"{conditional_expected_margin:.2%} < {request.target_margin:.2%}"
                ),
            ],
        )

    return PricingResponse(
        property_id=request.property_id,
        recommendation="price",
        acquisition_offer=round(selected.offer, 2),
        expected_profit=round(selected.expected_profit_per_lead, 2),
        probability_of_loss=round(_probability_of_loss_per_lead(solved), 6),
        sale_by_120_days=round(_sale_probability_on_no_sale_path(solved), 6),
        policy=_policy(solved),
        reasons=[
            "controlled-experiment illustration; not a live appraisal or transaction offer",
            f"worked_case={template.case.name}",
            f"illustrative_scenario_interval_width={interval_width_rate:.1%}",
            "expected_profit and approximate probability_of_loss are measured per quoted lead",
            "sale_by_120_days is conditional on acquisition",
            "recommendation uses only transparent worked-case models and request costs",
        ],
    )


def create_app() -> FastAPI:
    """Build the stateless HTTP application."""

    application = FastAPI(
        title="Offer-to-Exit illustrative pricing API",
        summary="A deterministic boundary over controlled-experiment worked cases.",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["operations"],
    )
    def health(response: Response) -> HealthResponse:
        response.headers[_EVIDENCE_HEADER] = _EVIDENCE_VALUE
        return HealthResponse(
            status="ok",
            version=__version__,
            evidence="controlled-experiment",
            worked_case_count=len(_solved_cases()),
        )

    @application.post(
        "/v1/price",
        response_model=PricingResponse,
        status_code=status.HTTP_200_OK,
        tags=["pricing"],
    )
    def price(request: PricingRequest, response: Response) -> PricingResponse:
        response.headers[_EVIDENCE_HEADER] = _EVIDENCE_VALUE
        return price_request(request)

    return application


app = create_app()


__all__ = ["HealthResponse", "app", "create_app", "price_request"]
