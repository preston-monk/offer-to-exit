"""Reference models and adapters used by the pricing optimizer.

The fitted adapters translate statistical-model inputs into the deliberately
small protocols in :mod:`offer_to_exit.decision.protocols`.  They contain no
simulator truth and make the price transformations visible in one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import exp, log

import numpy as np
import pandas as pd

from offer_to_exit.models import DiscreteTimeHazardModel, SellerAcceptanceModel

from .types import HomeContext, PriceAction, PricingState, SaleScenario


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


@dataclass(frozen=True)
class ValueScenarioSpec:
    """One weighted stress point derived from a fitted valuation interval.

    The weight is a decision-laboratory design choice. A conformal endpoint is
    not a fitted quantile, so these objects must not be presented as a calibrated
    predictive distribution.
    """

    name: str
    probability: float
    resale_value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be non-empty")
        if self.probability < 0:
            raise ValueError("scenario probability cannot be negative")
        if self.resale_value <= 0:
            raise ValueError("scenario resale value must be positive")


@dataclass
class FittedSellerAcceptanceAdapter:
    """Score candidate offers with a fitted seller-acceptance model.

    ``feature_row`` contains only the fitted model's observed covariates.  The
    adapter replaces the randomized ``offer_ratio`` treatment for each offer on
    the acquisition grid. Its denominator is the same observable pre-offer value
    reference used to assign offers in the controlled generator. It never reads
    the simulator's latent market value or acceptance truth.
    """

    model: SellerAcceptanceModel
    feature_row: pd.DataFrame
    offer_ratio_support: tuple[float, float]
    offer_feature: str = "offer_ratio"
    reference_feature: str = "preoffer_reference_value"

    def __post_init__(self) -> None:
        if len(self.feature_row) != 1:
            raise ValueError("acceptance adapter requires exactly one feature row")
        if self.offer_feature not in self.model.features:
            raise ValueError("offer feature is not present in the fitted acceptance model")
        missing = set(self.model.features).difference(self.feature_row.columns)
        if missing:
            raise KeyError(f"acceptance feature row is missing columns: {sorted(missing)}")
        lower, upper = self.offer_ratio_support
        if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
            raise ValueError("offer_ratio_support must contain finite increasing positive bounds")
        self.feature_row = self.feature_row.loc[:, self.model.features].copy()

    def acceptance_probability(self, context: HomeContext, offer: float) -> float:
        if offer < 0:
            raise ValueError("offer cannot be negative")
        if self.reference_feature not in context.features:
            raise KeyError(
                f"home context is missing observable offer reference {self.reference_feature!r}"
            )
        reference_value = float(context.features[self.reference_feature])
        if not np.isfinite(reference_value) or reference_value <= 0:
            raise ValueError("observable offer reference must be finite and positive")
        offer_ratio = offer / reference_value
        lower, upper = self.offer_ratio_support
        if not lower <= offer_ratio <= upper:
            raise ValueError(
                f"offer ratio {offer_ratio:.6f} is outside fitted support [{lower}, {upper}]"
            )
        counterfactual = self.feature_row.copy()
        counterfactual[self.offer_feature] = offer_ratio
        probability = float(self.model.predict_proba(counterfactual)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("fitted acceptance model returned an invalid probability")
        return probability


@dataclass
class FittedHazardSaleOutcomeAdapter:
    """Combine a fitted weekly hazard with weighted proceeds stress points.

    The posted list price becomes a counterfactual premium relative to the same
    observable pre-listing reference used to assign prices in the controlled
    generator. The fitted hazard supplies one sale-incidence probability for the
    current week. Valuation stress points affect conditional headline proceeds,
    not the price-treatment denominator. This keeps incidence and proceeds
    separate while preserving valuation downside. Stress weights are heuristic
    and are remixed at each week by the current simplified optimizer; they are
    not persistent latent states or posterior beliefs.
    """

    model: DiscreteTimeHazardModel
    feature_row: pd.DataFrame
    value_scenarios: tuple[ValueScenarioSpec, ...]
    list_price_premium_support: tuple[float, float]
    price_feature: str = "list_price_premium"
    reference_feature: str = "prelisting_reference_value"
    negotiation_discount: float = 0.005
    horizon_weeks: int = 17
    _cache: dict[tuple[int, int, int], tuple[SaleScenario, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _primed_reference_keys: set[int] = field(default_factory=set, init=False, repr=False)
    _scored_price_premiums: set[float] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.feature_row) != 1:
            raise ValueError("sale-outcome adapter requires exactly one feature row")
        if self.price_feature not in self.model.features:
            raise ValueError("price feature is not present in the fitted hazard model")
        missing = set(self.model.features).difference(self.feature_row.columns)
        if missing:
            raise KeyError(f"hazard feature row is missing columns: {sorted(missing)}")
        if not self.value_scenarios:
            raise ValueError("at least one valuation scenario is required")
        probability_total = sum(scenario.probability for scenario in self.value_scenarios)
        if probability_total <= 0:
            raise ValueError("valuation scenarios must have positive probability")
        if not np.isclose(probability_total, 1.0):
            raise ValueError("valuation scenario probabilities must sum to one")
        if not 0 <= self.negotiation_discount < 1:
            raise ValueError("negotiation_discount must lie in [0, 1)")
        if not 1 <= self.horizon_weeks <= self.model.max_periods:
            raise ValueError("adapter horizon must lie within the fitted hazard horizon")
        lower, upper = self.list_price_premium_support
        if not np.isfinite([lower, upper]).all() or not -1 < lower < upper:
            raise ValueError(
                "list_price_premium_support must contain finite increasing bounds above -1"
            )
        self.feature_row = self.feature_row.loc[:, self.model.features].copy()

    def sale_scenarios(self, context: HomeContext, state: PricingState) -> tuple[SaleScenario, ...]:
        if state.week >= self.model.max_periods:
            raise ValueError("pricing state exceeds the fitted hazard horizon")
        reference_value = self._reference_value(context)
        self._price_premium(state.list_price, reference_value)
        cache_key = self._cache_key(state.week, state.list_price, reference_value)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        reference_key = round(reference_value * 100)
        if state.week == 0 and reference_key not in self._primed_reference_keys:
            self._prime_price_lattice(state.list_price, reference_value)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        self._score_states(((state.week, state.list_price),), reference_value)
        return self._cache[cache_key]

    @property
    def scored_price_premiums(self) -> tuple[float, ...]:
        """Return every unique price treatment sent to the fitted hazard."""

        return tuple(sorted(self._scored_price_premiums))

    def _reference_value(self, context: HomeContext) -> float:
        if self.reference_feature not in context.features:
            raise KeyError(
                f"home context is missing observable list-price reference {self.reference_feature!r}"
            )
        reference_value = float(context.features[self.reference_feature])
        if not np.isfinite(reference_value) or reference_value <= 0:
            raise ValueError("observable list-price reference must be finite and positive")
        return reference_value

    def _price_premium(self, list_price: float, reference_value: float) -> float:
        premium = list_price / reference_value - 1.0
        lower, upper = self.list_price_premium_support
        tolerance = 1e-9
        if premium < lower - tolerance or premium > upper + tolerance:
            raise ValueError(
                f"list-price premium {premium:.6f} is outside fitted support [{lower}, {upper}]"
            )
        return min(max(premium, lower), upper)

    def _cache_key(
        self,
        week: int,
        list_price: float,
        reference_value: float,
    ) -> tuple[int, int, int]:
        return week, round(list_price * 100), round(reference_value * 100)

    def _prime_price_lattice(self, initial_list_price: float, reference_value: float) -> None:
        """Batch-score reachable weekly prices that remain inside fitted support."""

        before_action = {initial_list_price}
        states: list[tuple[int, float]] = []
        for week in range(self.horizon_weeks):
            candidate_prices = {
                action.apply(list_price) for list_price in before_action for action in PriceAction
            }
            after_action = {
                list_price
                for list_price in candidate_prices
                if self._price_is_supported(list_price, reference_value)
            }
            states.extend((week, list_price) for list_price in sorted(after_action))
            before_action = after_action
        self._score_states(states, reference_value)
        self._primed_reference_keys.add(round(reference_value * 100))

    def _price_is_supported(self, list_price: float, reference_value: float) -> bool:
        premium = list_price / reference_value - 1.0
        lower, upper = self.list_price_premium_support
        return lower - 1e-9 <= premium <= upper + 1e-9

    def _score_states(
        self,
        states: Sequence[tuple[int, float]],
        reference_value: float,
    ) -> None:
        if not states:
            return
        price_premiums = [
            self._price_premium(list_price, reference_value) for _, list_price in states
        ]
        self._scored_price_premiums.update(price_premiums)
        counterfactuals = pd.concat([self.feature_row] * len(states), ignore_index=True)
        counterfactuals[self.price_feature] = price_premiums
        hazards = self.model.predict_hazard(counterfactuals, horizon=self.horizon_weeks)
        for row_number, (week, list_price) in enumerate(states):
            probability = float(hazards[row_number, week])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("fitted hazard model returned an invalid probability")
            key = self._cache_key(week, list_price, reference_value)
            self._cache[key] = tuple(
                SaleScenario(
                    name=scenario.name,
                    probability=scenario.probability,
                    sale_probability=probability,
                    sale_price=min(
                        list_price * (1.0 - self.negotiation_discount),
                        scenario.resale_value,
                    ),
                )
                for scenario in self.value_scenarios
            )
