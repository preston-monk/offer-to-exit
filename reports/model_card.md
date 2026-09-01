# Model Card

## System, not a single model

Offer-to-Exit is a linked decision system. Its components answer different
questions and do not share the same interpretation.

| Component | Output | Interpretation |
|---|---|---|
| Exit valuation | Quantiles/distribution of value at a candidate exit horizon | Predictive |
| Seller acceptance | Acceptance probability over acquisition offers | Causal-response model inside the simulator |
| Exit hazard | Weekly probability of sale at a chosen price, conditional on remaining unsold | Causal-response model inside the simulator |
| Sale proceeds | Conditional distribution of proceeds when a sale occurs | Conditional predictive/response model inside the simulator |
| Cost model | Distribution of repair, selling, financing, and holding cost | Transparent scenario model |
| Optimizer | Offer and weekly resale-price policy | Decision rule evaluated in the simulator |

The components are joined through continuation value: the acquisition offer uses
the value and risk of the best feasible exit policy that follows acceptance.

## Status

**v0.1 research prototype.** The release fits three transparent components and
evaluates them on one independently seeded, covariate-shifted semi-synthetic
environment. The measured results and manifest are linked from
[results.md](results.md). Repeated policy comparisons and a stress suite are not
implemented or claimed.

## Intended use

- Research and education about pricing under uncertainty.
- Comparison of transparent baselines with constrained predictive/decision
  methods.
- Evaluation of temporal leakage controls, probability calibration, and
  abstention behavior.
- Reproducible technical analysis of an end-to-end pricing system.

## Not intended for

- Real offers, appraisals, or automated real-estate transactions.
- Decisions about lending, insurance, taxation, tenants, or consumers.
- Estimates of real seller or buyer price elasticity.
- Production use or generalization beyond the documented market/environment.

## Component specifications

### Exit valuation

**Baseline:** pooled training-period median.
**Primary model:** log-linear hedonic regression.
**Outputs:** point estimate and split-conformal 90% interval.
**Features:** structure, property age, coarse geography, lagged local-market
state, seasonality, and time.
**Key risk:** leakage from future comparable sales, revised records, or repeat
parcels across splits.

### Seller acceptance

**Baseline:** pooled acceptance rate.
**Primary model:** regularized logistic response with offer ratio, seller
urgency, market heat, repair-cost fraction, and submarket.
**Treatment:** acquisition offer divided by contemporaneous estimated value.
**Output:** calibrated acceptance probability over a bounded offer grid.
**Key risk:** causal interpretation is limited to the simulator and depends on
controlled exploration/overlap.

### Exit hazard

**Baseline:** period-specific observed hazard.
**Primary model:** discrete-time survival model using price-to-current-value,
weeks on market, lagged market state, and bounded heterogeneity.
**Output:** weekly sale probability conditional on remaining unsold.
**Censoring:** homes unsold after week 17 are right-censored.
**Key risk:** historical price cuts are endogenous; simulator exploration is
required for response recovery.

### Sale proceeds

**v0.1 treatment:** explicit downside/base/upside scenario distributions inside
the worked decision models. No learned proceeds model is claimed.
**Output:** scenario-weighted proceeds conditional on sale in a given week.
**Key risk:** mixing sale occurrence and proceeds into one regression can bias
both; they remain separate.

### Calibration and abstention

Probability and interval calibration use validation data only. The system may
abstain or route to review when:

- value intervals are wider than the configured threshold;
- behavioral overlap is weak;
- required features or temporal provenance are invalid;
- a property falls outside training support;
- no offer satisfies margin, sell-through, and loss constraints.

Coverage is therefore a model outcome, not a fixed entitlement.

## Evaluation

### Predictive

- Valuation: mean/median absolute error, WAPE, and interval coverage/width.
- Acceptance: log loss, Brier score, expected calibration error, calibration
  intercept/slope.
- Exit hazard: person-period log loss/Brier score and horizon sell-through.
- Proceeds: not fitted or separately scored in v0.1.

### Response recovery

- Error for simulator-known log-odds treatment effects.
- Monotonicity violations.
- Recovery in an independent shifted evaluation environment.

### Decision

- Contribution profit and margin conditional on accepted homes.
- Acceptance and coverage/abstention.
- Holding time, capital-days, sell-through, and markdown count.
- Probability of loss, worst-decile loss, and CVaR.
- Regret relative to the simulator oracle.

Slice calibration, overlap-stratified recovery, bootstrap intervals, and repeated
stress environments are not implemented in v0.1 and are not implied by the
reported evidence.

## Implemented comparisons

- Pooled-median versus regularized hedonic valuation.
- Pooled-rate versus logistic acceptance.
- Period-rate versus price-aware survival hazard.
- Fixed discount acquisition offer.
- Static list price with no cuts.
- Fixed scheduled markdown.
- Risk-aware continuation-value policy and three worked cases.

## Evidence still needed

- Fixed acceptance-target and myopic acquisition offers.
- Risk-neutral versus risk-aware dynamic pricing across evaluation episodes.
- Stress environments, uncertainty intervals, and simulator-oracle regret.

## Technical limitations

- A flexible predictive model can be calibrated and still fail under regime
  shift.
- Split conformal intervals do not guarantee conditional coverage for every home
  or geography.
- Simulated exploration does not establish that real operations could or should
  use the same exploration policy.
- Logistic response models may smooth important heterogeneous effects.
- Coarse cost schedules omit operational bottlenecks and renovation decisions.
- Optimizer quality is bounded by the model and simulator, not just search.

## Reproducibility

The release result must name its configuration, data manifest, simulator seeds,
dependency lock, code revision, and artifact checksum. Seeded determinism is
tested on the generated fixture; statistical results should still be interpreted
with uncertainty across evaluation draws.
