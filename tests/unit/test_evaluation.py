from __future__ import annotations

import pytest

from offer_to_exit.decision import (
    FixedSpreadOfferPolicy,
    HoldPricePolicy,
    HomeContext,
    OfferPolicy,
    PricingPolicy,
)
from offer_to_exit.evaluation import (
    BacktestRecord,
    compare_to_baseline,
    run_backtest,
    summarize_backtest,
    summarize_by_policy,
)


def _records(policy_name: str = "candidate") -> tuple[BacktestRecord, ...]:
    return (
        BacktestRecord("lead-1", policy_name, 80, True, True, 10, 100, 10, 1),
        BacktestRecord("lead-2", policy_name, 80, True, True, 15, 100, -5, 2),
        BacktestRecord("lead-3", policy_name, 80, True, False, 18, 0, -20, 3),
        BacktestRecord("lead-4", policy_name, 80, False, False, 0, 0, 0, 0),
    )


def test_backtest_scorecard_uses_lead_and_inventory_denominators() -> None:
    metrics = summarize_backtest(_records(), tail_probability=0.25)

    assert metrics.n_leads == 4
    assert metrics.n_accepted == 3
    assert metrics.n_sold == 2
    assert metrics.acceptance_rate == pytest.approx(0.75)
    assert metrics.sell_through_rate == pytest.approx(2 / 3)
    assert metrics.sell_through_90d == pytest.approx(1 / 3)
    assert metrics.sell_through_120d == pytest.approx(2 / 3)
    assert metrics.mean_profit_per_lead == pytest.approx(-3.75)
    assert metrics.mean_profit_per_accepted_home == pytest.approx(-5)
    assert metrics.contribution_margin == pytest.approx(-0.075)
    assert metrics.contribution_margin_bps == pytest.approx(-750)
    assert metrics.median_weeks_held == 15
    assert metrics.loss_rate == pytest.approx(2 / 3)
    assert metrics.downside_cvar_loss == pytest.approx(20)
    assert metrics.p10_profit == -20
    assert metrics.total_capital_days == pytest.approx(80 * (10 + 15 + 18) * 7)
    assert metrics.mean_markdowns_per_accepted_home == 2


def test_grouped_metrics_and_baseline_comparison() -> None:
    baseline = _records("baseline")
    candidate = tuple(
        BacktestRecord(
            record.lead_id,
            "candidate",
            record.acquisition_offer,
            record.accepted,
            record.sold,
            max(0, record.weeks_held - 1),
            record.sale_price,
            record.profit + 5,
            record.markdown_count,
        )
        for record in baseline
    )
    grouped = summarize_by_policy((*baseline, *candidate), tail_probability=0.25)
    comparison = compare_to_baseline(grouped["candidate"], grouped["baseline"])

    assert comparison.profit_per_lead_lift == pytest.approx(5)
    assert comparison.median_weeks_held_change == pytest.approx(-1)
    assert comparison.downside_cvar_loss_change < 0


class _DeterministicSimulator:
    def run_episode(
        self,
        *,
        context: HomeContext,
        offer_policy: OfferPolicy,
        pricing_policy: PricingPolicy,
        policy_name: str,
        seed: int,
    ) -> BacktestRecord:
        del pricing_policy
        offer = offer_policy.offer(context)
        return BacktestRecord(
            lead_id=context.home_id,
            policy_name=policy_name,
            acquisition_offer=offer,
            accepted=True,
            sold=True,
            weeks_held=seed + 1,
            sale_price=context.reference_value,
            profit=context.reference_value - offer,
        )


def test_run_backtest_accepts_a_simulator_adapter() -> None:
    contexts = (HomeContext("a", 100), HomeContext("b", 200))
    records, metrics = run_backtest(
        _DeterministicSimulator(),
        contexts=contexts,
        offer_policy=FixedSpreadOfferPolicy(0.1),
        pricing_policy=HoldPricePolicy(),
        policy_name="fixed-spread",
        seeds=(0, 1),
    )

    assert len(records) == 2
    assert metrics.n_sold == 2
    assert metrics.mean_profit_per_lead == pytest.approx(15)


def test_evaluation_rejects_ambiguous_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="exactly one policy"):
        summarize_backtest(
            (
                BacktestRecord("a", "one", 0, False, False, 0, 0, 0),
                BacktestRecord("b", "two", 0, False, False, 0, 0, 0),
            )
        )
    with pytest.raises(ValueError, match="exactly one value"):
        run_backtest(
            _DeterministicSimulator(),
            contexts=(HomeContext("a", 100), HomeContext("b", 200)),
            offer_policy=FixedSpreadOfferPolicy(0.1),
            pricing_policy=HoldPricePolicy(),
            policy_name="bad-seeds",
            seeds=(0,),
        )
    with pytest.raises(ValueError, match="rejected lead"):
        BacktestRecord("bad", "policy", 0, False, True, 0, 10, 0)
