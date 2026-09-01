"""Auditable rule-based offer and resale baselines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .protocols import AcceptanceModel
from .types import HomeContext, PriceAction, PricingState


@dataclass(frozen=True)
class HoldPricePolicy:
    """Never change the list price."""

    def choose_action(self, context: HomeContext, state: PricingState) -> PriceAction:
        del context, state
        return PriceAction.HOLD


@dataclass(frozen=True)
class ScheduledMarkdownPolicy:
    """Cut by a fixed amount on a fixed cadence after an initial wait."""

    action: PriceAction = PriceAction.CUT_2_5
    every_n_weeks: int = 2
    first_markdown_week: int = 2

    def __post_init__(self) -> None:
        if self.action is PriceAction.HOLD:
            raise ValueError("scheduled markdown action must reduce price")
        if self.every_n_weeks <= 0:
            raise ValueError("every_n_weeks must be positive")
        if self.first_markdown_week < 0:
            raise ValueError("first_markdown_week cannot be negative")

    def choose_action(self, context: HomeContext, state: PricingState) -> PriceAction:
        del context
        if state.week < self.first_markdown_week:
            return PriceAction.HOLD
        if (state.week - self.first_markdown_week) % self.every_n_weeks == 0:
            return self.action
        return PriceAction.HOLD


@dataclass(frozen=True)
class FixedSpreadOfferPolicy:
    """Offer a fixed discount to the as-of reference value."""

    spread: float = 0.08

    def __post_init__(self) -> None:
        if not 0 <= self.spread < 1:
            raise ValueError("spread must be in [0, 1)")

    def offer(self, context: HomeContext) -> float:
        return round(context.reference_value * (1.0 - self.spread), 2)


@dataclass(frozen=True)
class TargetAcceptanceOfferPolicy:
    """Choose the least expensive grid offer meeting an acceptance target."""

    acceptance_model: AcceptanceModel
    offer_grid: tuple[float, ...]
    target_probability: float = 0.5

    def __init__(
        self,
        acceptance_model: AcceptanceModel,
        offer_grid: Iterable[float],
        target_probability: float = 0.5,
    ) -> None:
        grid = tuple(sorted(set(float(offer) for offer in offer_grid)))
        if not grid or grid[0] < 0:
            raise ValueError("offer_grid must contain non-negative offers")
        if not 0 <= target_probability <= 1:
            raise ValueError("target_probability must be in [0, 1]")
        object.__setattr__(self, "acceptance_model", acceptance_model)
        object.__setattr__(self, "offer_grid", grid)
        object.__setattr__(self, "target_probability", target_probability)

    def offer(self, context: HomeContext) -> float:
        for offer in self.offer_grid:
            if (
                self.acceptance_model.acceptance_probability(context, offer)
                >= self.target_probability
            ):
                return offer
        return self.offer_grid[-1]
