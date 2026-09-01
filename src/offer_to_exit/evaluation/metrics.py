"""Business-first backtest metrics for realized policy episodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean, median

from offer_to_exit.decision.risk import downside_cvar_loss, weighted_quantile
from offer_to_exit.decision.types import ProfitOutcome

WEEKS_IN_90_DAYS = 13
WEEKS_IN_120_DAYS = 18


@dataclass(frozen=True)
class BacktestRecord:
    """One lead's realized outcome under one complete decision policy.

    A rejected lead should have ``accepted=False``, ``sold=False``, and zero
    deployed capital. Its profit may be negative if lead/quote cost is modeled.
    """

    lead_id: str
    policy_name: str
    acquisition_offer: float
    accepted: bool
    sold: bool
    weeks_held: int
    sale_price: float
    profit: float
    markdown_count: int = 0
    capital_deployed: float | None = None

    def __post_init__(self) -> None:
        if not self.lead_id or not self.policy_name:
            raise ValueError("lead_id and policy_name must be non-empty")
        if self.acquisition_offer < 0 or self.sale_price < 0:
            raise ValueError("prices cannot be negative")
        if self.weeks_held < 0 or self.markdown_count < 0:
            raise ValueError("weeks_held and markdown_count cannot be negative")
        if self.sold and not self.accepted:
            raise ValueError("a rejected lead cannot be sold")
        if self.capital_deployed is not None and self.capital_deployed < 0:
            raise ValueError("capital_deployed cannot be negative")

    @property
    def effective_capital(self) -> float:
        if not self.accepted:
            return 0.0
        if self.capital_deployed is not None:
            return self.capital_deployed
        return self.acquisition_offer


@dataclass(frozen=True)
class PolicyMetrics:
    policy_name: str
    n_leads: int
    n_accepted: int
    n_sold: int
    acceptance_rate: float
    sell_through_rate: float
    sell_through_90d: float
    sell_through_120d: float
    mean_profit_per_lead: float
    mean_profit_per_accepted_home: float
    contribution_margin: float
    contribution_margin_bps: float
    median_weeks_held: float
    loss_rate: float
    downside_cvar_loss: float
    p10_profit: float
    total_capital_days: float
    mean_capital_days_per_accepted_home: float
    mean_markdowns_per_accepted_home: float


@dataclass(frozen=True)
class PolicyComparison:
    """Directional lift of a candidate relative to a named baseline."""

    policy_name: str
    baseline_name: str
    profit_per_lead_lift: float
    contribution_margin_bps_lift: float
    acceptance_rate_lift: float
    sell_through_rate_lift: float
    median_weeks_held_change: float
    loss_rate_change: float
    downside_cvar_loss_change: float


def summarize_backtest(
    records: Iterable[BacktestRecord], *, tail_probability: float = 0.1
) -> PolicyMetrics:
    """Aggregate one policy's lead-level outcomes without external libraries."""

    items = tuple(records)
    if not items:
        raise ValueError("at least one backtest record is required")
    policy_names = {item.policy_name for item in items}
    if len(policy_names) != 1:
        raise ValueError("summarize_backtest expects records for exactly one policy")
    if not 0 < tail_probability <= 1:
        raise ValueError("tail_probability must be in (0, 1]")

    accepted = tuple(item for item in items if item.accepted)
    sold = tuple(item for item in accepted if item.sold)
    n_leads = len(items)
    n_accepted = len(accepted)
    n_sold = len(sold)

    acceptance_rate = n_accepted / n_leads
    sell_through_rate = n_sold / n_accepted if n_accepted else 0.0
    sell_through_90d = (
        sum(item.sold and item.weeks_held <= WEEKS_IN_90_DAYS for item in accepted) / n_accepted
        if n_accepted
        else 0.0
    )
    sell_through_120d = (
        sum(item.sold and item.weeks_held <= WEEKS_IN_120_DAYS for item in accepted) / n_accepted
        if n_accepted
        else 0.0
    )

    mean_profit_per_lead = fmean(item.profit for item in items)
    mean_profit_per_accepted = fmean(item.profit for item in accepted) if accepted else 0.0
    realized_revenue = sum(item.sale_price for item in sold)
    # Include every accepted home's realized P&L in the numerator. This keeps
    # losses on aged or liquidated inventory visible rather than conditioning
    # the margin on successful exits only.
    accepted_profit = sum(item.profit for item in accepted)
    contribution_margin = accepted_profit / realized_revenue if realized_revenue else 0.0

    weeks = [item.weeks_held for item in accepted]
    median_weeks = float(median(weeks)) if weeks else 0.0
    loss_rate = sum(item.profit < 0 for item in accepted) / n_accepted if n_accepted else 0.0

    equal_weight = 1.0 / n_leads
    profit_distribution = tuple(ProfitOutcome(equal_weight, item.profit) for item in items)
    cvar_loss = downside_cvar_loss(profit_distribution, tail_probability=tail_probability)
    p10_profit = weighted_quantile(profit_distribution, 0.1)

    total_capital_days = sum(item.effective_capital * item.weeks_held * 7 for item in accepted)
    mean_capital_days = total_capital_days / n_accepted if n_accepted else 0.0
    mean_markdowns = fmean(item.markdown_count for item in accepted) if accepted else 0.0

    return PolicyMetrics(
        policy_name=next(iter(policy_names)),
        n_leads=n_leads,
        n_accepted=n_accepted,
        n_sold=n_sold,
        acceptance_rate=acceptance_rate,
        sell_through_rate=sell_through_rate,
        sell_through_90d=sell_through_90d,
        sell_through_120d=sell_through_120d,
        mean_profit_per_lead=mean_profit_per_lead,
        mean_profit_per_accepted_home=mean_profit_per_accepted,
        contribution_margin=contribution_margin,
        contribution_margin_bps=contribution_margin * 10_000,
        median_weeks_held=median_weeks,
        loss_rate=loss_rate,
        downside_cvar_loss=cvar_loss,
        p10_profit=p10_profit,
        total_capital_days=total_capital_days,
        mean_capital_days_per_accepted_home=mean_capital_days,
        mean_markdowns_per_accepted_home=mean_markdowns,
    )


def summarize_by_policy(
    records: Iterable[BacktestRecord], *, tail_probability: float = 0.1
) -> Mapping[str, PolicyMetrics]:
    """Group an interleaved record stream and summarize each policy."""

    grouped: dict[str, list[BacktestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.policy_name].append(record)
    if not grouped:
        raise ValueError("at least one backtest record is required")
    return {
        name: summarize_backtest(items, tail_probability=tail_probability)
        for name, items in sorted(grouped.items())
    }


def compare_to_baseline(candidate: PolicyMetrics, baseline: PolicyMetrics) -> PolicyComparison:
    """Return candidate-minus-baseline changes in decision metrics."""

    return PolicyComparison(
        policy_name=candidate.policy_name,
        baseline_name=baseline.policy_name,
        profit_per_lead_lift=(candidate.mean_profit_per_lead - baseline.mean_profit_per_lead),
        contribution_margin_bps_lift=(
            candidate.contribution_margin_bps - baseline.contribution_margin_bps
        ),
        acceptance_rate_lift=candidate.acceptance_rate - baseline.acceptance_rate,
        sell_through_rate_lift=(candidate.sell_through_rate - baseline.sell_through_rate),
        median_weeks_held_change=(candidate.median_weeks_held - baseline.median_weeks_held),
        loss_rate_change=candidate.loss_rate - baseline.loss_rate,
        downside_cvar_loss_change=(candidate.downside_cvar_loss - baseline.downside_cvar_loss),
    )
