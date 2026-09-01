"""Thin orchestration interface for simulator-backed backtests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from offer_to_exit.decision.protocols import OfferPolicy, PricingPolicy
from offer_to_exit.decision.types import HomeContext

from .metrics import BacktestRecord, PolicyMetrics, summarize_backtest


@runtime_checkable
class EpisodeSimulator(Protocol):
    """Adapter contract for a real or synthetic episode generator."""

    def run_episode(
        self,
        *,
        context: HomeContext,
        offer_policy: OfferPolicy,
        pricing_policy: PricingPolicy,
        policy_name: str,
        seed: int,
    ) -> BacktestRecord: ...


def run_backtest(
    simulator: EpisodeSimulator,
    *,
    contexts: Sequence[HomeContext],
    offer_policy: OfferPolicy,
    pricing_policy: PricingPolicy,
    policy_name: str,
    seeds: Iterable[int] | None = None,
    tail_probability: float = 0.1,
) -> tuple[tuple[BacktestRecord, ...], PolicyMetrics]:
    """Run one policy across contexts and immediately produce its scorecard."""

    if not contexts:
        raise ValueError("contexts cannot be empty")
    episode_seeds = tuple(seeds) if seeds is not None else tuple(range(len(contexts)))
    if len(episode_seeds) != len(contexts):
        raise ValueError("seeds must contain exactly one value per context")
    records = tuple(
        simulator.run_episode(
            context=context,
            offer_policy=offer_policy,
            pricing_policy=pricing_policy,
            policy_name=policy_name,
            seed=seed,
        )
        for context, seed in zip(contexts, episode_seeds, strict=True)
    )
    return records, summarize_backtest(records, tail_probability=tail_probability)
