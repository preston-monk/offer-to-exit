# System Specification

## Decision contract

Offer-to-Exit studies a hypothetical single-market home acquisition and resale
operator. For each supported single-family home, the system chooses the highest
acquisition offer worth making and a weekly resale-pricing policy conditional on
acceptance.

| Field | Contract |
|---|---|
| Market shape | Phoenix/Maricopa County, Arizona |
| Acquisition action | Recommend an offer from a bounded grid, request human review, or decline to price |
| Resale action | Hold list price or reduce it by 1%, 2.5%, or 5% |
| Horizon | 17 weekly resale decisions, approximately 120 days |
| Objective | Expected contribution profit less a configurable 95% CVaR downside penalty |
| Constraints | Margin, sell-through, loss probability, price bounds, and markdown cadence |
| Quote-time information | Property and market data available when the offer is made |
| Weekly information | Only state and demand information observed before the next pricing action |

Human review and declining to price are distinct. Review is appropriate when an
input or support problem may be resolved by a person. Declining to price is the
economic action when valid inputs still produce no supported offer with positive
risk-adjusted value.

## Economic object

For property $i$, acquisition offer $a_i$, seller-acceptance indicator
$S_i(a_i)$, closing proceeds $Y_{i\tau}$, and resale policy $\pi$,
contribution profit is

```math
\Pi_i(a_i,\pi) = S_i(a_i)\left[
Y_{i\tau} - a_i - R_i - C_i(Y_{i\tau})
- \sum_{t=0}^{\tau}H_{it}
\right],
```

where $R_i$, $C_i$, and $H_{it}$ denote repair, closing, and weekly holding
costs. Each acquisition offer is evaluated using the continuation value of the
best feasible resale policy that follows acceptance. Optimizing acceptance and
resale pricing as unrelated tasks would violate this contract.

## Estimands and interpretation

| Object | Target | Interpretation in this release |
|---|---|---|
| Resale value | Distribution of future sale value conditional on quote-time features | Predictive |
| Seller acceptance | Acceptance probability under an acquisition-offer intervention | Causal response inside the simulator only |
| Weekly sale hazard | Sale probability this week, conditional on remaining unsold and a list-price intervention | Causal response inside the simulator only |
| Conditional proceeds | Closing proceeds conditional on a sale | Transparent simulated scenario distribution |
| Policy value | Expected contribution, loss probability, CVaR, and resale path | Diagnostic simulator result |

## Evidence boundary

The public-data pipeline handles property records, historical transactions, and
market series. The released experiment does not fit models on those downloaded
records. It uses generated, Phoenix-shaped data because public sources do not
contain randomized acquisition offers, randomized resale prices, buyer-arrival
counterfactuals, or complete proprietary cost histories.

Training and evaluation use independent seeds. The evaluation environment also
shifts market heat, mortgage rates, appreciation, home values, square footage,
and submarket composition. This is one documented covariate-shift check, not a
claim of real-market transportability or comprehensive robustness.

## Information-time rules

At decision time $t$, every feature $x_j$ must satisfy

```math
\operatorname{available\_at}(x_j) \leq t.
```

Quote-time models cannot use repair or listing information learned after an
offer. Weekly decisions use only prior-week demand. Comparable sales must be
available by the decision timestamp, repeated parcels may not cross evaluation
partitions, and outcomes must mature before labeling.

## Executable invariants

Holding other inputs fixed, tests enforce the following economic and operational
relationships:

1. higher repair or holding cost cannot raise the recommended acquisition offer;
2. greater risk aversion cannot raise the recommended acquisition offer;
3. a resale action cannot increase list price;
4. a policy cannot act after sale or after the horizon;
5. invalid inputs cannot receive an automated offer; and
6. contribution profit reconciles to proceeds less acquisition and every
   configured cost.

## Current scope and exclusions

The current release is a property-level research prototype. It excludes
real-world economic-impact claims, cross-market generalization, portfolio capital
allocation, renovation operations, learned conditional proceeds, unconstrained
dynamic pricing, reinforcement learning, production service levels, and any use
for lending, insurance, taxation, appraisal, or consumer eligibility.
