"""Small dependency-free utilities for weighted profit distributions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .types import ProfitOutcome

_PROBABILITY_TOLERANCE = 1e-9


def normalize_outcomes(outcomes: Iterable[ProfitOutcome]) -> tuple[ProfitOutcome, ...]:
    """Normalize a non-empty discrete distribution to unit probability."""

    items = tuple(outcome for outcome in outcomes if outcome.probability > 0)
    if not items:
        raise ValueError("at least one positive-probability outcome is required")
    total = sum(outcome.probability for outcome in items)
    if total <= 0:
        raise ValueError("total outcome probability must be positive")
    if abs(total - 1.0) <= _PROBABILITY_TOLERANCE:
        return items
    return tuple(ProfitOutcome(item.probability / total, item.profit) for item in items)


def expected_profit(outcomes: Sequence[ProfitOutcome]) -> float:
    normalized = normalize_outcomes(outcomes)
    return sum(item.probability * item.profit for item in normalized)


def lower_tail_mean(outcomes: Sequence[ProfitOutcome], *, tail_probability: float = 0.1) -> float:
    """Return the weighted mean profit in the worst probability tail."""

    if not 0 < tail_probability <= 1:
        raise ValueError("tail_probability must be in (0, 1]")
    ordered = sorted(normalize_outcomes(outcomes), key=lambda item: item.profit)
    remaining = tail_probability
    weighted_profit = 0.0
    for item in ordered:
        if remaining <= _PROBABILITY_TOLERANCE:
            break
        weight = min(item.probability, remaining)
        weighted_profit += weight * item.profit
        remaining -= weight
    return weighted_profit / tail_probability


def downside_cvar_loss(
    outcomes: Sequence[ProfitOutcome], *, tail_probability: float = 0.1
) -> float:
    """CVaR-style expected positive loss in the worst profit tail.

    Loss is defined as ``max(-profit, 0)``. This avoids rewarding a policy for
    having a positive lower-tail profit while still directly penalizing the
    severe negative outcomes that matter in an inventory business.
    """

    if not 0 < tail_probability <= 1:
        raise ValueError("tail_probability must be in (0, 1]")
    ordered = sorted(normalize_outcomes(outcomes), key=lambda item: item.profit)
    remaining = tail_probability
    weighted_loss = 0.0
    for item in ordered:
        if remaining <= _PROBABILITY_TOLERANCE:
            break
        weight = min(item.probability, remaining)
        weighted_loss += weight * max(-item.profit, 0.0)
        remaining -= weight
    return weighted_loss / tail_probability


def weighted_quantile(outcomes: Sequence[ProfitOutcome], quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(normalize_outcomes(outcomes), key=lambda item: item.profit)
    if quantile == 0:
        return ordered[0].profit
    cumulative = 0.0
    for item in ordered:
        cumulative += item.probability
        if cumulative + _PROBABILITY_TOLERANCE >= quantile:
            return item.profit
    return ordered[-1].profit


def compress_outcomes(
    outcomes: Sequence[ProfitOutcome], *, max_points: int
) -> tuple[ProfitOutcome, ...]:
    """Compress a distribution into equal-mass bins using bin means.

    Recursively mixing weekly sale and continuation distributions can otherwise
    grow exponentially. Equal-mass compression preserves total probability and
    the overall mean while retaining useful tail resolution.
    """

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    ordered = sorted(normalize_outcomes(outcomes), key=lambda item: item.profit)
    if len(ordered) <= max_points:
        return tuple(ordered)

    target_mass = 1.0 / max_points
    bins: list[ProfitOutcome] = []
    bin_mass = 0.0
    bin_profit_mass = 0.0

    for item in ordered:
        remaining_item_mass = item.probability
        while remaining_item_mass > _PROBABILITY_TOLERANCE:
            capacity = target_mass - bin_mass
            allocated = min(capacity, remaining_item_mass)
            bin_mass += allocated
            bin_profit_mass += allocated * item.profit
            remaining_item_mass -= allocated
            if bin_mass + _PROBABILITY_TOLERANCE >= target_mass:
                bins.append(ProfitOutcome(bin_mass, bin_profit_mass / bin_mass))
                bin_mass = 0.0
                bin_profit_mass = 0.0

    if bin_mass > _PROBABILITY_TOLERANCE:
        bins.append(ProfitOutcome(bin_mass, bin_profit_mass / bin_mass))
    return normalize_outcomes(bins)
