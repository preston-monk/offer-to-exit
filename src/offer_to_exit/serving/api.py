"""FastAPI boundary for deterministic, semi-synthetic pricing illustrations.

The API deliberately reuses the worked-case decision models. It does not load a
model binary, call a remote service, or claim to score a real property. Request
fields first pass the public Pydantic contract and then a narrower v0.1 support
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
    CostStructure,
    DynamicPricingOptimizer,
    HomeContext,
    WorkedDecisionResult,
    solve_worked_decision_cases,
)

from .schemas import PricingRequest, PricingResponse, WeeklyAction

_EVIDENCE_HEADER = "X-Offer-to-Exit-Evidence"
_EVIDENCE_VALUE = "semi-synthetic"
_MAX_INTERVAL_WIDTH_RATE = 0.20


class HealthResponse(BaseModel):
    """Small operational response that does not execute a pricing decision."""

    status: Literal["ok"]
    version: str
    evidence: Literal["semi-synthetic"]
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
    """Return valid-schema inputs that fall outside the narrow v0.1 domain."""

    violations: list[str] = []
    if not 100_000 <= request.estimated_market_value <= 1_250_000:
        violations.append("estimated_market_value outside v0.1 support [$100k, $1.25m]")
    if not 700 <= request.living_area_sqft <= 5_000:
        violations.append("living_area_sqft outside v0.1 support [700, 5,000]")
    if not 1940 <= request.year_built <= 2026:
        violations.append("year_built outside v0.1 support [1940, 2026]")
    if request.repair_cost / request.estimated_market_value > 0.20:
        violations.append("repair_cost exceeds 20% of estimated market value")
    if request.weekly_holding_cost / request.estimated_market_value > 0.01:
        violations.append("weekly_holding_cost exceeds 1% of estimated market value")
    if request.risk_aversion > 2.0:
        violations.append("risk_aversion outside calibrated illustrative support [0, 2]")
    return violations


def _scenario_interval_width_rate(template: WorkedDecisionResult) -> float:
    """Use the worked case's explicit value scenarios as an interval-width proxy."""

    multipliers = tuple(
        scenario.value_multiplier for scenario in template.case.outcome_model.scenarios
    )
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
        features=case.context.features,
    )
    costs = CostStructure(
        repair_cost=request.repair_cost,
        weekly_holding_cost=request.weekly_holding_cost,
        transaction_cost_rate=case.costs.transaction_cost_rate,
        fixed_transaction_cost=case.costs.fixed_transaction_cost * scale,
        acquisition_cost=case.costs.acquisition_cost * scale,
        prelisting_holding_weeks=case.costs.prelisting_holding_weeks,
    )
    pricing_config = replace(case.pricing_config, risk_aversion=request.risk_aversion)
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
        offer_grid=tuple(offer * scale for offer in case.offer_grid),
        initial_list_price=case.initial_list_price * scale,
        costs=costs,
    )
    adapted_case = replace(
        case,
        context=context,
        costs=costs,
        initial_list_price=case.initial_list_price * scale,
        offer_grid=tuple(offer * scale for offer in case.offer_grid),
        pricing_config=pricing_config,
    )
    return WorkedDecisionResult(case=adapted_case, result=result)


def _probability_of_loss(result: WorkedDecisionResult) -> float:
    outcomes = result.result.selected.exit_result.outcomes
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
            "semi-synthetic illustration; not a live appraisal or transaction offer",
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
            reasons=["outside Phoenix/Maricopa v0.1 illustrative support", *violations],
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
    expected_profit = selected.exit_result.expected_profit
    expected_margin = expected_profit / selected.offer
    if expected_margin < request.target_margin:
        return _non_action_response(
            request,
            recommendation="review",
            reasons=[
                f"worked_case={template.case.name}",
                (
                    "illustrative expected margin does not meet target: "
                    f"{expected_margin:.2%} < {request.target_margin:.2%}"
                ),
            ],
        )

    return PricingResponse(
        property_id=request.property_id,
        recommendation="price",
        acquisition_offer=round(selected.offer, 2),
        expected_profit=round(expected_profit, 2),
        probability_of_loss=round(_probability_of_loss(solved), 6),
        sale_by_120_days=round(_sale_probability_on_no_sale_path(solved), 6),
        policy=_policy(solved),
        reasons=[
            "semi-synthetic illustration; not a live appraisal or transaction offer",
            f"worked_case={template.case.name}",
            f"illustrative_scenario_interval_width={interval_width_rate:.1%}",
            "recommendation uses only transparent worked-case models and request costs",
        ],
    )


def create_app() -> FastAPI:
    """Build the stateless HTTP application."""

    application = FastAPI(
        title="Offer-to-Exit illustrative pricing API",
        summary="A deterministic boundary over semi-synthetic worked cases.",
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
            evidence="semi-synthetic",
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
