"""Backtest protocols, business metrics, and baseline comparisons."""

from .backtest import EpisodeSimulator, run_backtest
from .metrics import (
    WEEKS_IN_90_DAYS,
    WEEKS_IN_120_DAYS,
    BacktestRecord,
    PolicyComparison,
    PolicyMetrics,
    compare_to_baseline,
    summarize_backtest,
    summarize_by_policy,
)

__all__ = [
    "WEEKS_IN_90_DAYS",
    "WEEKS_IN_120_DAYS",
    "BacktestRecord",
    "EpisodeSimulator",
    "PolicyComparison",
    "PolicyMetrics",
    "compare_to_baseline",
    "run_backtest",
    "summarize_backtest",
    "summarize_by_policy",
]
