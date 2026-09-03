"""Finite-horizon offer-to-exit optimization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isclose
from types import MappingProxyType

from .protocols import AcceptanceModel, PricingPolicy, SaleOutcomeModel
from .risk import (
    compress_outcomes,
    downside_cvar_loss,
    expected_profit,
    normalize_outcomes,
)
from .types import (
    AcquisitionCandidate,
    AcquisitionOptimizationResult,
    CostStructure,
    HomeContext,
    PolicyDecision,
    PriceAction,
    PricingOptimizationResult,
    PricingState,
    ProfitOutcome,
    SaleScenario,
    StateKey,
    state_key,
)


@dataclass(frozen=True)
class PricingOptimizerConfig:
    """Controls the weekly dynamic program."""

    horizon_weeks: int = 17
    actions: tuple[PriceAction, ...] = (
        PriceAction.HOLD,
        PriceAction.CUT_1,
        PriceAction.CUT_2_5,
        PriceAction.CUT_5,
    )
    tail_probability: float = 0.1
    risk_aversion: float = 0.25
    terminal_liquidation_discount: float = 0.92
    min_list_price_ratio: float = 0.70
    max_distribution_points: int = 200

    def __post_init__(self) -> None:
        if self.horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive")
        if not self.actions:
            raise ValueError("at least one action is required")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        if not 0 < self.tail_probability <= 1:
            raise ValueError("tail_probability must be in (0, 1]")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion cannot be negative")
        if not 0 < self.terminal_liquidation_discount <= 1:
            raise ValueError("terminal_liquidation_discount must be in (0, 1]")
        if not 0 < self.min_list_price_ratio <= 1:
            raise ValueError("min_list_price_ratio must be in (0, 1]")
        if self.max_distribution_points <= 0:
            raise ValueError("max_distribution_points must be positive")


@dataclass(frozen=True)
class AcquisitionOptimizerConfig:
    """Controls acquisition-offer grid scoring."""

    tail_probability: float = 0.1
    risk_aversion: float = 0.25
    quote_cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.tail_probability <= 1:
            raise ValueError("tail_probability must be in (0, 1]")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion cannot be negative")
        if self.quote_cost < 0:
            raise ValueError("quote_cost cannot be negative")


@dataclass(frozen=True)
class _NodeValue:
    outcomes: tuple[ProfitOutcome, ...]
    expected_profit: float
    downside_cvar_loss: float
    objective_value: float


class DynamicPricingOptimizer:
    """Optimize weekly list-price actions by backward induction.

    The optimizer consumes weighted scenarios from ``SaleOutcomeModel``.
    It does not know whether those scenarios came from a simulator, a Bayesian
    model, an API, or a table fixture.
    """

    def __init__(
        self,
        outcome_model: SaleOutcomeModel,
        config: PricingOptimizerConfig | None = None,
    ) -> None:
        self.outcome_model = outcome_model
        self.config = config or PricingOptimizerConfig()

    def optimize(
        self,
        *,
        context: HomeContext,
        acquisition_offer: float,
        initial_list_price: float,
        costs: CostStructure,
    ) -> PricingOptimizationResult:
        """Return the risk-adjusted optimal policy for one acquired home."""

        return self._run(
            context=context,
            acquisition_offer=acquisition_offer,
            initial_list_price=initial_list_price,
            costs=costs,
            policy=None,
        )

    def evaluate_policy(
        self,
        policy: PricingPolicy,
        *,
        context: HomeContext,
        acquisition_offer: float,
        initial_list_price: float,
        costs: CostStructure,
    ) -> PricingOptimizationResult:
        """Evaluate a fixed baseline policy under the same outcome model."""

        return self._run(
            context=context,
            acquisition_offer=acquisition_offer,
            initial_list_price=initial_list_price,
            costs=costs,
            policy=policy,
        )

    def _run(
        self,
        *,
        context: HomeContext,
        acquisition_offer: float,
        initial_list_price: float,
        costs: CostStructure,
        policy: PricingPolicy | None,
    ) -> PricingOptimizationResult:
        if acquisition_offer < 0:
            raise ValueError("acquisition_offer cannot be negative")
        if initial_list_price <= 0:
            raise ValueError("initial_list_price must be positive")

        initial_state = PricingState(week=0, list_price=initial_list_price)
        memo: dict[StateKey, _NodeValue] = {}
        decisions: dict[StateKey, PolicyDecision] = {}

        def solve(state: PricingState) -> _NodeValue:
            key = state_key(state)
            if key in memo:
                return memo[key]
            if state.week >= self.config.horizon_weeks:
                raise RuntimeError("terminal states are resolved by their parent")

            if policy is None:
                candidate_actions = self.config.actions
            else:
                candidate_actions = (policy.choose_action(context, state),)

            best_value: _NodeValue | None = None
            best_decision: PolicyDecision | None = None
            list_price_reference = float(
                context.features.get("prelisting_reference_value", context.reference_value)
            )
            if list_price_reference <= 0:
                raise ValueError("prelisting_reference_value must be positive")
            price_floor = list_price_reference * self.config.min_list_price_ratio

            for action in candidate_actions:
                if action not in self.config.actions:
                    raise ValueError(f"policy returned disallowed action: {action}")
                new_price = action.apply(state.list_price)
                if action is not PriceAction.HOLD and new_price < price_floor:
                    continue

                post_action_state = PricingState(
                    week=state.week,
                    list_price=new_price,
                    demand_state=state.demand_state,
                )
                scenarios = _normalize_scenarios(
                    self.outcome_model.sale_scenarios(context, post_action_state)
                )
                weekly_sale_probability = sum(
                    scenario.probability * scenario.sale_probability for scenario in scenarios
                )

                outcomes: list[ProfitOutcome] = []
                for scenario in scenarios:
                    sale_mass = scenario.probability * scenario.sale_probability
                    if sale_mass > 0:
                        outcomes.append(
                            ProfitOutcome(
                                probability=sale_mass,
                                profit=costs.profit(
                                    sale_price=scenario.sale_price,
                                    acquisition_offer=acquisition_offer,
                                    sale_week=state.week,
                                ),
                            )
                        )

                if state.week + 1 < self.config.horizon_weeks:
                    no_sale_probability = max(0.0, 1.0 - weekly_sale_probability)
                    if no_sale_probability > 0:
                        next_state = PricingState(
                            week=state.week + 1,
                            list_price=new_price,
                            demand_state=state.demand_state,
                        )
                        continuation = solve(next_state)
                        outcomes.extend(
                            ProfitOutcome(
                                probability=no_sale_probability * outcome.probability,
                                profit=outcome.profit,
                            )
                            for outcome in continuation.outcomes
                        )
                else:
                    # Any inventory still unsold at the horizon is liquidated.
                    # Scenario-specific recovery preserves downside variation.
                    for scenario in scenarios:
                        no_sale_mass = scenario.probability * (1.0 - scenario.sale_probability)
                        if no_sale_mass <= 0:
                            continue
                        # ``sale_price`` is already the scenario's modeled
                        # conditional headline proceeds and cannot exceed the
                        # posted price under the fitted adapter.
                        liquidation_price = (
                            scenario.sale_price * self.config.terminal_liquidation_discount
                        )
                        outcomes.append(
                            ProfitOutcome(
                                probability=no_sale_mass,
                                profit=costs.profit(
                                    sale_price=liquidation_price,
                                    acquisition_offer=acquisition_offer,
                                    sale_week=state.week,
                                ),
                            )
                        )

                distribution = compress_outcomes(
                    normalize_outcomes(outcomes),
                    max_points=self.config.max_distribution_points,
                )
                mean = expected_profit(distribution)
                cvar_loss = downside_cvar_loss(
                    distribution,
                    tail_probability=self.config.tail_probability,
                )
                objective = mean - self.config.risk_aversion * cvar_loss
                node_value = _NodeValue(distribution, mean, cvar_loss, objective)
                decision = PolicyDecision(
                    state=state,
                    action=action,
                    new_list_price=new_price,
                    weekly_sale_probability=weekly_sale_probability,
                    expected_profit=mean,
                    downside_cvar_loss=cvar_loss,
                    objective_value=objective,
                )

                # Preserve action ordering on an economically immaterial tie,
                # which favors HOLD before successively larger markdowns.
                if best_value is None or objective > best_value.objective_value + 1e-9:
                    best_value = node_value
                    best_decision = decision

            if best_value is None or best_decision is None:
                raise ValueError("no feasible pricing action at state")
            memo[key] = best_value
            decisions[key] = best_decision
            return best_value

        root = solve(initial_state)
        return PricingOptimizationResult(
            initial_state=initial_state,
            expected_profit=root.expected_profit,
            downside_cvar_loss=root.downside_cvar_loss,
            objective_value=root.objective_value,
            outcomes=root.outcomes,
            decisions=MappingProxyType(dict(decisions)),
        )


class AcquisitionOfferOptimizer:
    """Choose an offer grid point using the optimal exit continuation value."""

    def __init__(
        self,
        pricing_optimizer: DynamicPricingOptimizer,
        acceptance_model: AcceptanceModel,
        config: AcquisitionOptimizerConfig | None = None,
    ) -> None:
        self.pricing_optimizer = pricing_optimizer
        self.acceptance_model = acceptance_model
        self.config = config or AcquisitionOptimizerConfig(
            tail_probability=pricing_optimizer.config.tail_probability,
            risk_aversion=pricing_optimizer.config.risk_aversion,
        )

    def optimize(
        self,
        *,
        context: HomeContext,
        offer_grid: Iterable[float],
        initial_list_price: float,
        costs: CostStructure,
    ) -> AcquisitionOptimizationResult:
        offers = tuple(sorted(set(float(offer) for offer in offer_grid)))
        if not offers:
            raise ValueError("offer_grid cannot be empty")
        if offers[0] < 0:
            raise ValueError("offers cannot be negative")

        candidates: list[AcquisitionCandidate] = []
        for offer in offers:
            acceptance = self.acceptance_model.acceptance_probability(context, offer)
            if not 0 <= acceptance <= 1:
                raise ValueError("acceptance model must return a probability in [0, 1]")
            exit_result = self.pricing_optimizer.optimize(
                context=context,
                acquisition_offer=offer,
                initial_list_price=initial_list_price,
                costs=costs,
            )
            outcomes = [
                ProfitOutcome(
                    probability=acceptance * outcome.probability,
                    profit=outcome.profit - self.config.quote_cost,
                )
                for outcome in exit_result.outcomes
            ]
            if acceptance < 1:
                outcomes.append(
                    ProfitOutcome(
                        probability=1.0 - acceptance,
                        profit=-self.config.quote_cost,
                    )
                )
            distribution = compress_outcomes(
                normalize_outcomes(outcomes),
                max_points=self.pricing_optimizer.config.max_distribution_points,
            )
            mean = expected_profit(distribution)
            cvar_loss = downside_cvar_loss(
                distribution,
                tail_probability=self.config.tail_probability,
            )
            objective = mean - self.config.risk_aversion * cvar_loss
            candidates.append(
                AcquisitionCandidate(
                    offer=offer,
                    acceptance_probability=acceptance,
                    expected_profit_per_lead=mean,
                    downside_cvar_loss=cvar_loss,
                    objective_value=objective,
                    exit_result=exit_result,
                    outcomes=distribution,
                )
            )

        selected = max(
            candidates,
            key=lambda candidate: (
                candidate.objective_value,
                candidate.expected_profit_per_lead,
                -candidate.offer,
            ),
        )
        return AcquisitionOptimizationResult(selected=selected, candidates=tuple(candidates))


def _normalize_scenarios(scenarios: Sequence[SaleScenario]) -> tuple[SaleScenario, ...]:
    items = tuple(scenario for scenario in scenarios if scenario.probability > 0)
    if not items:
        raise ValueError("outcome model must return a positive-probability scenario")
    total = sum(item.probability for item in items)
    if total <= 0:
        raise ValueError("scenario probabilities must have positive total")
    if isclose(total, 1.0, abs_tol=1e-9):
        return items
    return tuple(
        SaleScenario(
            name=item.name,
            probability=item.probability / total,
            sale_probability=item.sale_probability,
            sale_price=item.sale_price,
        )
        for item in items
    )
