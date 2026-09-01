"""Risk-aware acquisition and dynamic resale-price decisions."""

from .baselines import (
    FixedSpreadOfferPolicy,
    HoldPricePolicy,
    ScheduledMarkdownPolicy,
    TargetAcceptanceOfferPolicy,
)
from .cases import (
    WorkedDecisionCase,
    WorkedDecisionResult,
    solve_worked_decision_cases,
    worked_decision_cases,
)
from .models import (
    ConstantElasticitySaleModel,
    LogisticAcceptanceModel,
    MarketScenarioSpec,
)
from .optimizer import (
    AcquisitionOfferOptimizer,
    AcquisitionOptimizerConfig,
    DynamicPricingOptimizer,
    PricingOptimizerConfig,
)
from .protocols import AcceptanceModel, OfferPolicy, PricingPolicy, SaleOutcomeModel
from .risk import downside_cvar_loss, expected_profit, lower_tail_mean, weighted_quantile
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
)

__all__ = [
    "AcceptanceModel",
    "AcquisitionCandidate",
    "AcquisitionOfferOptimizer",
    "AcquisitionOptimizationResult",
    "AcquisitionOptimizerConfig",
    "ConstantElasticitySaleModel",
    "CostStructure",
    "DynamicPricingOptimizer",
    "FixedSpreadOfferPolicy",
    "HoldPricePolicy",
    "HomeContext",
    "LogisticAcceptanceModel",
    "MarketScenarioSpec",
    "OfferPolicy",
    "PolicyDecision",
    "PriceAction",
    "PricingOptimizationResult",
    "PricingOptimizerConfig",
    "PricingPolicy",
    "PricingState",
    "ProfitOutcome",
    "SaleOutcomeModel",
    "SaleScenario",
    "ScheduledMarkdownPolicy",
    "TargetAcceptanceOfferPolicy",
    "WorkedDecisionCase",
    "WorkedDecisionResult",
    "downside_cvar_loss",
    "expected_profit",
    "lower_tail_mean",
    "solve_worked_decision_cases",
    "weighted_quantile",
    "worked_decision_cases",
]
