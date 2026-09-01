from __future__ import annotations

import pytest

from offer_to_exit.decision.risk import (
    compress_outcomes,
    downside_cvar_loss,
    lower_tail_mean,
    normalize_outcomes,
    weighted_quantile,
)
from offer_to_exit.decision.types import ProfitOutcome


def test_risk_helpers_reject_invalid_distributions_and_parameters() -> None:
    with pytest.raises(ValueError, match="positive-probability"):
        normalize_outcomes(())
    with pytest.raises(ValueError, match="positive-probability"):
        normalize_outcomes((ProfitOutcome(0, 1),))
    with pytest.raises(ValueError, match="tail_probability"):
        lower_tail_mean((ProfitOutcome(1, 1),), tail_probability=0)
    with pytest.raises(ValueError, match="tail_probability"):
        downside_cvar_loss((ProfitOutcome(1, 1),), tail_probability=1.1)
    with pytest.raises(ValueError, match="quantile"):
        weighted_quantile((ProfitOutcome(1, 1),), -0.1)
    with pytest.raises(ValueError, match="max_points"):
        compress_outcomes((ProfitOutcome(1, 1),), max_points=0)


def test_quantile_endpoints_and_uncompressed_distribution() -> None:
    outcomes = (ProfitOutcome(0.25, -2), ProfitOutcome(0.75, 4))
    assert weighted_quantile(outcomes, 0) == -2
    assert weighted_quantile(outcomes, 1) == 4
    assert compress_outcomes(outcomes, max_points=3) == outcomes
