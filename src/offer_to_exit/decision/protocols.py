"""Interfaces that separate fitted models and simulators from decisions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import HomeContext, PriceAction, PricingState, SaleScenario


@runtime_checkable
class SaleOutcomeModel(Protocol):
    """Produces calibrated weekly sale scenarios for a post-action state."""

    def sale_scenarios(
        self, context: HomeContext, state: PricingState
    ) -> Sequence[SaleScenario]: ...


@runtime_checkable
class AcceptanceModel(Protocol):
    """Estimates causal seller acceptance for an offered acquisition price."""

    def acceptance_probability(self, context: HomeContext, offer: float) -> float: ...


@runtime_checkable
class PricingPolicy(Protocol):
    """Rule or learned policy used for baseline evaluation."""

    def choose_action(self, context: HomeContext, state: PricingState) -> PriceAction: ...


@runtime_checkable
class OfferPolicy(Protocol):
    """Rule or learned policy that returns an acquisition offer."""

    def offer(self, context: HomeContext) -> float: ...
