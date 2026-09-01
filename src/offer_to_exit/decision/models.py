"""Small reference models used by examples and smoke tests.

They are intentionally transparent and replaceable. Real fitted models should
implement the protocols in :mod:`offer_to_exit.decision.protocols`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from .types import HomeContext, PricingState, SaleScenario


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-8), 1.0 - 1e-8)
    return log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class MarketScenarioSpec:
    name: str
    probability: float
    value_multiplier: float = 1.0
    demand_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.probability < 0:
            raise ValueError("probability cannot be negative")
        if self.value_multiplier <= 0 or self.demand_multiplier <= 0:
            raise ValueError("scenario multipliers must be positive")


@dataclass(frozen=True)
class ConstantElasticitySaleModel:
    """Interpretable weekly demand model for worked examples."""

    base_weekly_sale_probability: float
    price_elasticity: float
    dom_log_odds_per_week: float = 0.0
    negotiation_discount: float = 0.005
    scenarios: tuple[MarketScenarioSpec, ...] = (MarketScenarioSpec("base", 1.0),)

    def __post_init__(self) -> None:
        if not 0 < self.base_weekly_sale_probability < 1:
            raise ValueError("base_weekly_sale_probability must be in (0, 1)")
        if self.price_elasticity < 0:
            raise ValueError("price_elasticity cannot be negative")
        if not 0 <= self.negotiation_discount < 1:
            raise ValueError("negotiation_discount must be in [0, 1)")
        if not self.scenarios:
            raise ValueError("at least one market scenario is required")

    def sale_scenarios(self, context: HomeContext, state: PricingState) -> tuple[SaleScenario, ...]:
        context_shift = float(context.features.get("demand_log_odds_shift", 0.0))
        outcomes: list[SaleScenario] = []
        for scenario in self.scenarios:
            market_value = context.reference_value * scenario.value_multiplier
            relative_price_gap = state.list_price / market_value - 1.0
            log_odds = (
                _logit(self.base_weekly_sale_probability)
                - self.price_elasticity * relative_price_gap
                + self.dom_log_odds_per_week * state.week
                + log(scenario.demand_multiplier)
                + context_shift
            )
            sale_probability = _sigmoid(log_odds)
            sale_price = min(
                state.list_price * (1.0 - self.negotiation_discount),
                market_value,
            )
            outcomes.append(
                SaleScenario(
                    name=scenario.name,
                    probability=scenario.probability,
                    sale_probability=sale_probability,
                    sale_price=sale_price,
                )
            )
        return tuple(outcomes)


@dataclass(frozen=True)
class LogisticAcceptanceModel:
    """Monotone seller response as a function of offer/reference value."""

    midpoint_offer_ratio: float = 0.92
    slope: float = 30.0
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if self.midpoint_offer_ratio <= 0:
            raise ValueError("midpoint_offer_ratio must be positive")
        if self.slope <= 0:
            raise ValueError("slope must be positive")

    def acceptance_probability(self, context: HomeContext, offer: float) -> float:
        if offer < 0:
            raise ValueError("offer cannot be negative")
        urgency = float(context.features.get("seller_urgency_log_odds", 0.0))
        ratio = offer / context.reference_value
        return _sigmoid(self.intercept + urgency + self.slope * (ratio - self.midpoint_offer_ratio))
