"""Core value objects for offer-to-exit decisions.

The decision package intentionally depends only on these small, immutable data
objects.  A simulator, fitted statistical model, or production scoring service
can therefore supply demand scenarios without depending on optimizer internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class PriceAction(StrEnum):
    """Allowed weekly resale-price actions."""

    HOLD = "hold"
    CUT_1 = "cut_1_pct"
    CUT_2_5 = "cut_2_5_pct"
    CUT_5 = "cut_5_pct"

    @property
    def cut_fraction(self) -> float:
        return {
            PriceAction.HOLD: 0.0,
            PriceAction.CUT_1: 0.01,
            PriceAction.CUT_2_5: 0.025,
            PriceAction.CUT_5: 0.05,
        }[self]

    def apply(self, list_price: float) -> float:
        """Return the post-action price, rounded to cents."""

        if list_price <= 0:
            raise ValueError("list_price must be positive")
        return round(list_price * (1.0 - self.cut_fraction), 2)


@dataclass(frozen=True)
class HomeContext:
    """Information known about a home at the decision timestamp.

    ``features`` is deliberately generic.  Downstream model adapters can read
    whatever as-of property or market features they require; the optimizer only
    relies on ``reference_value``.
    """

    home_id: str
    reference_value: float
    features: Mapping[str, float] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.home_id:
            raise ValueError("home_id must be non-empty")
        if self.reference_value <= 0:
            raise ValueError("reference_value must be positive")


@dataclass(frozen=True)
class PricingState:
    """Observable state immediately before a weekly pricing action."""

    week: int
    list_price: float
    demand_state: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.week < 0:
            raise ValueError("week must be non-negative")
        if self.list_price <= 0:
            raise ValueError("list_price must be positive")


@dataclass(frozen=True)
class SaleScenario:
    """A conditional outcome supplied by a demand model.

    ``probability`` is the scenario weight. ``sale_probability`` is the chance
    of selling during the current week within that scenario. ``sale_price`` is
    expected net headline proceeds conditional on that sale, before transaction
    and inventory costs are applied.
    """

    name: str
    probability: float
    sale_probability: float
    sale_price: float

    def __post_init__(self) -> None:
        if self.probability < 0:
            raise ValueError("scenario probability cannot be negative")
        if not 0.0 <= self.sale_probability <= 1.0:
            raise ValueError("sale_probability must be in [0, 1]")
        if self.sale_price < 0:
            raise ValueError("sale_price cannot be negative")


@dataclass(frozen=True)
class ProfitOutcome:
    """One point in a discrete profit distribution."""

    probability: float
    profit: float

    def __post_init__(self) -> None:
        if self.probability < 0:
            raise ValueError("outcome probability cannot be negative")


@dataclass(frozen=True)
class CostStructure:
    """Transparent per-home economics used by the optimizer."""

    repair_cost: float = 0.0
    weekly_holding_cost: float = 0.0
    transaction_cost_rate: float = 0.0
    fixed_transaction_cost: float = 0.0
    acquisition_cost: float = 0.0
    prelisting_holding_weeks: int = 0

    def __post_init__(self) -> None:
        numeric_costs = (
            self.repair_cost,
            self.weekly_holding_cost,
            self.transaction_cost_rate,
            self.fixed_transaction_cost,
            self.acquisition_cost,
        )
        if any(value < 0 for value in numeric_costs):
            raise ValueError("cost inputs cannot be negative")
        if self.transaction_cost_rate >= 1:
            raise ValueError("transaction_cost_rate must be less than 1")
        if self.prelisting_holding_weeks < 0:
            raise ValueError("prelisting_holding_weeks cannot be negative")

    def transaction_cost(self, sale_price: float) -> float:
        if sale_price < 0:
            raise ValueError("sale_price cannot be negative")
        return self.fixed_transaction_cost + self.transaction_cost_rate * sale_price

    def profit(
        self,
        *,
        sale_price: float,
        acquisition_offer: float,
        sale_week: int,
    ) -> float:
        """Contribution profit for a sale in a zero-indexed listing week."""

        if acquisition_offer < 0:
            raise ValueError("acquisition_offer cannot be negative")
        if sale_week < 0:
            raise ValueError("sale_week must be non-negative")
        held_weeks = self.prelisting_holding_weeks + sale_week + 1
        return (
            sale_price
            - acquisition_offer
            - self.repair_cost
            - self.acquisition_cost
            - self.transaction_cost(sale_price)
            - held_weeks * self.weekly_holding_cost
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Optimal or policy-prescribed decision at one state."""

    state: PricingState
    action: PriceAction
    new_list_price: float
    weekly_sale_probability: float
    expected_profit: float
    downside_cvar_loss: float
    objective_value: float


StateKey = tuple[int, int, tuple[float, ...]]


def state_key(state: PricingState) -> StateKey:
    """Stable memoization key using integer cents for the price."""

    return state.week, round(state.list_price * 100), state.demand_state


@dataclass(frozen=True)
class PricingOptimizationResult:
    """Finite-horizon policy and its root profit distribution."""

    initial_state: PricingState
    expected_profit: float
    downside_cvar_loss: float
    objective_value: float
    outcomes: tuple[ProfitOutcome, ...]
    decisions: Mapping[StateKey, PolicyDecision]

    @property
    def first_decision(self) -> PolicyDecision:
        return self.decisions[state_key(self.initial_state)]

    def decision_for(self, state: PricingState) -> PolicyDecision:
        return self.decisions[state_key(state)]

    def no_sale_path(self) -> tuple[PolicyDecision, ...]:
        """Trace recommended actions if the home remains unsold each week."""

        state = self.initial_state
        path: list[PolicyDecision] = []
        while state_key(state) in self.decisions:
            decision = self.decision_for(state)
            path.append(decision)
            state = PricingState(
                week=state.week + 1,
                list_price=decision.new_list_price,
                demand_state=state.demand_state,
            )
        return tuple(path)


@dataclass(frozen=True)
class AcquisitionCandidate:
    """Evaluation of one acquisition offer and its optimal exit policy."""

    offer: float
    acceptance_probability: float
    expected_profit_per_lead: float
    downside_cvar_loss: float
    objective_value: float
    exit_result: PricingOptimizationResult
    outcomes: tuple[ProfitOutcome, ...]


@dataclass(frozen=True)
class AcquisitionOptimizationResult:
    """Chosen acquisition offer plus all grid candidates."""

    selected: AcquisitionCandidate
    candidates: tuple[AcquisitionCandidate, ...]
