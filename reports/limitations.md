# Limitations and Risk Register

## Claims boundary

The most important limitation is not model size or sample size. It is the boundary
between public observation and simulated intervention.

| Claim | Supported? | Why |
|---|---|---|
| The code executes a linked acquisition/exit policy | Yes, when release tests pass | Directly inspectable and reproducible |
| A value model achieves stated out-of-time metrics | Only after a versioned result exists | Empirical predictive claim |
| A behavioral model recovers simulator-known response | Only inside the documented simulator | Ground truth is generated |
| A policy improves simulated risk-adjusted contribution | Only against named baselines/environment | Offline simulator result |
| Phoenix sellers or buyers have the reported elasticity | No | Real treatments and counterfactuals are unobserved |
| A real company would realize the simulated lift | No | Operations, selection, costs, and behavior differ |
| Results generalize outside Maricopa County | No | The current scope is intentionally single-market |

## Risk register

### 1. Temporal leakage

**Failure:** future comparable sales, revised assessor fields, final days on
market, or post-action demand enter a decision row.
**Consequence:** implausibly strong prediction and policy results.
**Controls:** feature-level availability timestamps, as-of joins, explicit
forbidden columns, chronological tests, and a 120-day maturity buffer.

### 2. Entity leakage

**Failure:** repeat observations of the same parcel cross train and test.
**Consequence:** memorized home effects masquerade as generalization.
**Controls:** parcel-grouped splits, duplicate-resolution report, and split
disjointness tests.

### 3. Treatment endogeneity

**Failure:** historical markdowns are treated as random even though weak latent
demand caused them.
**Consequence:** biased price-response estimates and unsafe recommendations.
**Controls:** causal-response claims are restricted to logged simulator data with
controlled exploration; observational price-change regressions are not given a
causal label.

### 4. Selection and censoring

**Failure:** resale outcomes are learned only from accepted/sold homes, or unsold
homes are mislabeled.
**Consequence:** optimistic proceeds and sell-through.
**Controls:** explicit acceptance selection, right-censored weekly panels,
separate sale-hazard and proceeds models, and matured outcomes.

### 5. Simulator overfitting

**Failure:** fitted models mirror the generator equation or share evaluation
seeds/parameters.
**Consequence:** near-oracle results with no robustness meaning.
**Controls:** richer nonlinear generator, simpler fitted models, independent
seeds, hidden evaluation parameters, and shifted stress environments.

### 6. Miscalibrated uncertainty

**Failure:** intervals achieve average coverage but fail for expensive, sparse,
or shifted homes.
**Consequence:** risk optimizer treats false precision as opportunity.
**Controls:** time and segment calibration plots, interval-width reporting,
conditional diagnostics, abstention, and stress tests.

### 7. Cost-model error

**Failure:** repair, transaction, financing, or holding costs are understated.
**Consequence:** offers are too high and simulated margins are overstated.
**Controls:** transparent schedules, uncertainty distributions, reconciliation
tests, and repair/holding/rate stress scenarios.

### 8. Objective misspecification

**Failure:** the selected CVaR weight or margin/sell-through target does not
represent the actual operator's preferences.
**Consequence:** a mathematically optimal policy is operationally wrong.
**Controls:** report the full frontier, compare risk-neutral and risk-aware
policies, and avoid presenting one parameter setting as universally optimal.

### 9. Distribution shift

**Failure:** mortgage rates, inventory, seasonality, or buyer preferences leave
training support.
**Consequence:** valuation, hazard, and policy failure occur together.
**Controls:** newest-period testing, stress environments, out-of-domain checks,
calibration monitoring, and higher abstention.

### 10. Geographic proxy and uneven data quality

**Failure:** geography or property attributes proxy for protected characteristics,
or sparse areas receive systematically worse estimates.
**Consequence:** unequal errors or coverage and misleading neighborhood signals.
**Controls:** no protected-class inputs, coarse location in examples, slice-level
error/uncertainty/abstention diagnostics, feature review, and no lending or
consumer-eligibility use.

### 11. Operational simplification

**Failure:** The current scope omits renovation queues, listing operations, portfolio capital,
and market concentration.
**Consequence:** individual-home policies cannot be interpreted as a portfolio
operating plan.
**Controls:** label results as property-level contribution economics and reserve
portfolio optimization for a later system scope.

### 12. Metric gaming

**Failure:** a policy appears safe by abstaining almost everywhere, or profitable
by accepting very few favorable homes.
**Consequence:** impressive conditional metrics with negligible usefulness.
**Controls:** always report coverage, acceptance, capital deployed, sell-through,
and the complete profit-risk frontier alongside conditional profit.

## What would be needed for real-world validation?

- Logged quote and acceptance data with defensible exploration or another source
  of identification.
- Listing histories with action-time demand signals and reliable censoring.
- Operationally complete repair, financing, transaction, and holding costs.
- Prospective shadow-mode evaluation and pre-specified guardrails.
- Human-review workflow, model-risk ownership, auditability, and rollback.
- Multi-market and regime-shift validation.
- Controlled online experimentation before economic-impact claims.

Those requirements are intentionally outside the current scope. They mark the
boundary between a research prototype and a production decision system.
