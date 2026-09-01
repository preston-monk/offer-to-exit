# Decision-Policy Card

## Policy objective

Offer-to-Exit chooses an acquisition offer and, conditional on acceptance, a
weekly resale-price policy. The policy maximizes risk-adjusted contribution value,
not valuation accuracy, offer acceptance, or sell-through in isolation.

For offer \(a\), exit policy \(\pi\), and contribution profit \(\Pi\):

\[
U(a,\pi) = \mathbb{E}[\Pi(a,\pi)]
- \lambda\operatorname{CVaR}_{0.95}[-\Pi(a,\pi)].
\]

The risk-aversion parameter \(\lambda\) is configured and disclosed. Results must
include the risk-neutral comparison rather than imply one setting is universally
correct.

## Contribution-profit boundary

Included in v0.1:

- acquisition price;
- repair/preparation-cost uncertainty;
- selling/transaction costs;
- weekly financing and holding costs;
- negotiated exit proceeds;
- sale timing and horizon censoring.

Excluded in v0.1:

- corporate overhead allocations;
- portfolio concentration and capital opportunity cost beyond a simple bound;
- operational capacity constraints;
- taxes or fees not present in the configured schedule;
- unmodeled renovation scope decisions.

This is contribution economics inside the documented simulator, not a real
operator profit-and-loss statement.

## Exit policy

### State

- week and remaining horizon;
- current list price;
- property and lagged market features;
- current exit-value distribution;
- lagged demand summary when enabled;
- accumulated holding cost and markdown count.

### Actions

- keep list price;
- reduce by 1%;
- reduce by 2.5%;
- reduce by 5%.

Price increases are excluded from v0.1. Prices must remain inside operational
bounds, and the policy cannot exceed the configured markdown count or cadence.

### Transition and value

For each feasible action, the policy integrates:

1. probability of sale this week;
2. conditional proceeds distribution if sold;
3. current cost and loss distribution;
4. holding cost and continuation value if unsold.

Finite-horizon dynamic programming or an equivalent backward recursion is
preferred over reinforcement learning because the state/action space is small,
the horizon is finite, and constraints should remain inspectable.

## Acquisition policy

Each candidate offer is evaluated using:

- modeled seller-acceptance probability;
- optimized exit continuation value if accepted;
- contribution-cost distribution;
- downside-risk penalty;
- margin, sell-through, and loss constraints.

The selected offer is the feasible candidate with the largest risk-adjusted
expected value. The policy abstains when no candidate is feasible or when data,
uncertainty, domain, or overlap checks fail.

## Constraints

The v0.1 worked-case configuration includes:

- minimum expected contribution-margin rate;
- minimum probability of sale by the 17-week horizon;
- maximum probability of loss;
- maximum interval-width and minimum-overlap rules;
- bounded offer/value ratios;
- bounded markdown actions and frequency.

These are scenario assumptions, not universal business rules. Every report must
print their actual configured values.

## Human review and abstention

The system returns a structured reason instead of an offer when:

- a temporal/data contract fails;
- the property is outside supported market or property type;
- the exit-value interval is too wide;
- behavioral treatment overlap is too weak;
- no candidate satisfies the economic constraints;
- the optimizer encounters an invalid state.

Review is not a loophole for silently overriding constraints. A review output
must retain the relevant diagnostic and must not be counted as an automated
offer in coverage metrics.

## Policy baselines

### Acquisition

- Fixed discount to estimated value.
- Fixed seller-acceptance target.
- Myopic acceptance probability times static expected margin.
- Risk-neutral continuation-value policy.

### Exit

- List at estimated value and never cut.
- Fixed markup with fixed markdown every two weeks.
- One-week proceeds maximizer without continuation value.
- Sell-through-only policy.
- Dynamic policy without uncertainty or CVaR.

### Upper bound

The simulator oracle observes the true environment parameters and provides a
regret benchmark. It is not a deployable competitor.

## Invariants

Automated tests should enforce at least these relationships, holding other inputs
fixed:

- higher repair cost cannot increase the acquisition offer;
- higher weekly holding cost cannot increase the acquisition offer;
- a higher required margin cannot increase the acquisition offer;
- greater risk aversion cannot increase the acquisition offer;
- an action may not increase list price;
- a policy cannot act after sale or after the horizon;
- invalid/out-of-domain rows cannot receive an automated offer;
- profit calculations reconcile to proceeds minus every configured cost.

These invariants are often more decision-relevant than reproducing an exact
floating-point prediction.

## Future policy evaluation

Any population policy comparison should be labeled **simulated** and evaluated
across independent environments. A valid future result would name:

- baseline and policy version;
- simulator version and seed family;
- evaluation sample and maturity rule;
- mean and downside metrics;
- abstention/coverage;
- sensitivity to core cost and behavior assumptions;
- regret versus the oracle.

“Improved simulated risk-adjusted contribution under the documented environment”
is acceptable. “Would improve a real operator's profit” is not supported.
