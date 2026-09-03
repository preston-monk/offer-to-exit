# Offer-to-Exit

**A public GitHub project on the property-level economics of a housing market maker.**

[![CI](https://github.com/preston-monk/offer-to-exit/actions/workflows/ci.yml/badge.svg)](https://github.com/preston-monk/offer-to-exit/actions/workflows/ci.yml)
[![Tested on Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)

[Open the Florida evidence report](https://preston-monk.github.io/offer-to-exit/artifacts/release/florida_evidence.html) ·
[Open the controlled decision laboratory](https://preston-monk.github.io/offer-to-exit/artifacts/release/demo.html)

Offer-to-Exit asks one economic question:

> How much should a housing market maker pay for a home, given the value,
> duration, and downside risk of the best feasible resale policy after acquisition?

The acquisition offer is the bid. The resale price is the ask. The expected
spread must compensate the operator for transaction costs, carrying costs,
valuation error, market exposure, and the financing and operating costs of an
illiquid unit of inventory. Because an acquisition is worth only what the best exit policy makes
it worth, the bid and the sequence of asks belong in one decision problem.

The repository contains two complementary sources of evidence:

1. **A real Florida transaction study.** The pipeline harmonizes 2,657,072
   recorded parcel-transfer rows from Hillsborough County, which contains Tampa, and Orange
   County, which contains Orlando. It estimates a repeat-sale valuation model,
   reconstructs 5,060 named iBuyer ownership spells, estimates a recorded
   disposition hazard, and applies Tampa-only specifications to an Orlando
   external-market evaluation without fitting or calibrating on Orlando.
2. **A controlled policy experiment.** Public deeds do not reveal offers that
   sellers rejected or demand at list prices that were never posted. A separate
   generated experiment randomizes acquisition-offer ratios relative to an
   observable pre-offer reference and list-price premia relative to an observable
   pre-listing reference. It fits the behavioral models, passes those fitted
   models into a finite-horizon optimizer, and tests whether the behavioral
   models recover the known responses and how the linked decision system behaves.

Those layers answer different questions. The Florida records support predictive
and descriptive claims about recorded prices and title duration. The controlled
experiment supports causal-response recovery and policy diagnostics inside its
known data-generating process. Neither layer estimates real seller or buyer
elasticities, real operator profit, or real-market policy lift.

## The economic setting

An iBuyer is a balance-sheet intermediary. It posts an acquisition offer, owns
the property if the seller accepts, and then chooses a sequence of resale list
prices until the home sells or reaches a terminal liquidation date.

| Stage | Choice | Economic tradeoff |
|---|---|---|
| Acquisition | Offer $a$ | A higher bid raises seller acceptance but reduces margin conditional on purchase. |
| Initial resale | List price $p_0$ | A higher ask raises proceeds conditional on sale but can lower the probability of a quick exit. |
| Unsold inventory | Hold or mark down | Waiting preserves upside but incurs another week of cost and market exposure. |
| Terminal week | Liquidate | A finite recovery value prevents the model from treating delay as free. |

A house is a large, indivisible, illiquid unit of inventory. A \$400,000 sale in
three weeks and a \$400,000 sale in seventeen weeks are not economically
equivalent. The objective is not the highest eventual sale price. It is
risk-adjusted contribution value after time, costs, liquidity, and uncertainty
are priced.

### The buyer side is a sale hazard

The model does not follow a named buyer making an individual yes-or-no choice.
It represents aggregate buyer demand as a weekly sale hazard. For state $s_t$
and posted list price $p_t$,

```math
h_t(p_t \mid s_t)
=
\Pr\!\left(
\text{sale in week }t
\mid
\text{unsold at }t,\;p_t,\;s_t
\right).
```

A high list price increases proceeds if the home sells, but may reduce
$h_t$. No sale adds carrying cost and leaves a risky asset on the balance sheet
for another week. The remaining horizon determines how valuable waiting is.

Let realized contribution profit from a sale in week $t$ be

```math
G_t(a,P_t)
=
P_t-a-R-K-C(P_t)-(w_0+t+1)H,
```

where $P_t$ is the realized transaction price, $R$ is repair cost, $K$ is
acquisition cost, $C(P_t)$ is resale transaction cost, $w_0$ is pre-listing
holding time, and $H$ is weekly holding cost. For a risk-neutral operator, the
familiar Bellman recursion is

```math
\begin{aligned}
V_t(s_t)=\max_{p_t\in\mathcal P(s_t)}\Big\{&
h_t(p_t\mid s_t)\,
\mathbb E[G_t(a,P_t)\mid\text{sale}] \\
&+\left[1-h_t(p_t\mid s_t)\right]
V_{t+1}(s_{t+1})
\Big\}.
\end{aligned}
```

The current week's holding cost is already inside $G_t$ in every realized sale
or liquidation outcome. Remaining unsold raises the cumulative holding term in
the later payoff, so the recursion does not subtract $H$ a second time on the
no-sale branch. Offer-to-Exit extends this setup by propagating a discrete
weighted set of stress outcomes at every state and applying a transparent
downside rule.

### The downside rule is recursive, not a static CVaR commitment

For profit distribution $\Pi$ and worst-tail mass $\alpha$, define

```math
D_\alpha(\Pi)
=
\frac{1}{\alpha}
\int_0^\alpha
\max\{-Q_\Pi(u),0\}\,du,
```

where $Q_\Pi(u)$ is the profit quantile. The released cases use
$\alpha=0.10$. Thus $D_{0.10}$ is the average positive loss in the worst decile
of conditional outcomes. At each unsold-inventory state, the algorithm chooses
the feasible action that maximizes

```math
\mathbb E[\Pi_t\mid s_t]
-
\lambda D_{0.10}(\Pi_t\mid s_t).
```

This is a recursive, state-contingent downside rule. It is deliberately not
described as the solution to one static ex-ante mean-CVaR commitment problem.
That distinction matters because static tail-risk criteria need not be time
consistent when they are reoptimized after new information arrives.

### Acquisition depends on continuation value

Before choosing the acquisition offer, the operator solves the resale problem
backward. For every candidate offer $a$, the system constructs the weighted
exit-profit stress outcomes under the best feasible pricing path, combines them with
seller acceptance $q(a\mid x)$ and the rejection outcome, then applies the same
mean-minus-downside rule to profit per lead.

A higher offer:

- raises the probability of acquiring the home;
- reduces the spread conditional on acquisition;
- leaves less protection against value and cost error; and
- makes the exit policy more fragile when demand is weak.

The maximum bid worth making therefore depends on the liquidity and downside
risk of the best available exit path.

## Evidence layer 1: real Florida deeds

### Why Tampa and Orlando

Hillsborough and Orange Counties expose complementary official record systems.
The [Hillsborough County Property Appraiser](https://downloads.hcpafl.org/)
publishes a large all-sales archive with long transaction histories. The
[Orange County Property Appraiser](https://vgispublic.ocpafl.org/server/rest/services/Webmap/SALES/MapServer/5)
publishes a paginated ArcGIS sales layer with property attributes and party
names. Both contain enough named intermediary activity to study the same
economic objects, and their different schemas make the data-engineering test
real rather than cosmetic.

The pipeline:

- downloads 2,450,504 Hillsborough deed rows and 206,568 Orange deed rows;
- records source URLs, retrieval time, file hashes, query fields, and pagination;
- uses party names transiently to classify a narrow registry of Opendoor,
  Offerpad, Zillow Offers, and RedfinNow legal-name patterns;
- removes names, addresses, coordinates, and raw parcel identifiers;
- hashes parcel identifiers in a market-specific namespace;
- preserves property-use codes as strings, including leading zeros; and
- links each named acquisition deed with at least \$10,000 in recorded
  consideration to the first later deed on which the same operator is the seller
  and consideration meets the same threshold,
  retaining open spells as right-censored observations.

Raw files and transaction-level analytical files remain outside version
control. Only aggregate results, figures, and manifests are released.

### Valuation estimand and target population

The common Hillsborough-Orange file does not contain a rich, symmetric set of
structural housing characteristics. A cross-sectional automated valuation model
would therefore ask unobserved property quality to do too much. The released
estimand is narrower and more defensible:

> Recorded sale price among high-turnover, improved, single-family repeat sales
> whose prior eligible sale occurred no more than 4.5 years earlier.

The prior eligible sale must fall in a strictly earlier calendar quarter than the target
sale. This ensures that the full-quarter median used to normalize the prior
price is completely observable before the target quarter begins. A separate
30-day minimum-gap sensitivity tests whether very rapid cross-quarter resales
drive the comparison.

Hillsborough observations require the county's qualified-sale flag. Orange's
`SALE_DESCRIPTION` is absent in the older portion of its published layer, so
the Orange sample uses the county-qualified flag when observed and a warranty
deed (`WD`) proxy only when that flag is missing. This is a practical
comparability rule, not proof that every retained Orange transfer is arm's
length. The released sample funnel reports 27,251 officially qualified Orange
rows and 45,223 rows admitted by the warranty-deed proxy before the repeat-sale
restriction.

For the current sale of parcel $i$ in market $m$ and quarter $q$, define the
normalized target

```math
y_{imq}
=
\log\!\left(\frac{P_{imq}}{M_{m,q-1}}\right),
```

where $M_{m,q-1}$ is the median retained sale price in the preceding county
quarter. If the previous sale occurred in quarter $r(i)$, the transparent
repeat-sale benchmark is

```math
\widehat P^{\mathrm{roll}}_{imq}
=
P_{im,r(i)}
\frac{M_{m,q-1}}{M_{m,r(i)}}.
```

This rolls the same home's prior price forward by the county market. The
gradient-boosted challenger uses
$\log(P_{im,r(i)}/M_{m,r(i)})$, elapsed years since the prior eligible sale, and calendar
quarter, then converts its normalized prediction back to dollars with
$M_{m,q-1}$. The prior eligible sale supplies a property-specific price anchor and proxy
for persistent quality, while the lagged current-quarter median supplies an
as-of market scale. This is a
repeat-sale benchmark, not a full hedonic AVM.

The Hillsborough sample is chronological:

- proper training: eligible Tampa repeat sales before January 2022;
- conformal calibration: January 2022 through December 2023; and
- out-of-time test: January 2024 through the August 2026 snapshot.

Parcels appearing in a later partition are purged from earlier model-fitting
partitions. The prediction interval uses one conservative calibration score per
parcel, defined as that parcel's maximum absolute log residual. The fitted Tampa
pipeline is then applied to Orange County sales from January 2024 onward without
fitting or interval calibration on Orlando. This is an external-market
evaluation, not a preregistered untouched holdout.

### Valuation results

| Test | Transactions | Model MAE | Rolled-prior MAE | Reduction in MAE | Mean absolute percentage error | Nominal 90% interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| Tampa, out of time | 10,514 | \$76,503 | \$78,268 | 2.3% | 13.7% | 85.3% |
| Orlando, external market | 6,036 | \$129,039 | \$147,902 | 12.8% | 18.5% | 80.5% |

The nonlinear challenger improves MAE only modestly over the strong repeat-sale
benchmark in Tampa and more clearly in Orlando. It does not dominate on every
loss function: the rolled-prior benchmark has lower Tampa RMSE and median
absolute error. Mean errors are \$18,953 below recorded prices in Tampa and
\$31,401 below in Orlando. Interval coverage falls from 85.3% in Tampa to 80.5%
in Orlando, below the nominal 90% in both places. The decision implication is
recalibration or review, not automatic deployment. That deterioration is
evidence of temporal and geographic model risk, not a result to hide.

As a qualification sensitivity, the 4,124 Orlando test transactions with an
observed county-qualified description produce model MAE of \$144,459 versus
\$162,374 for the rolled-prior benchmark, an 11.0% reduction. This subset spans
January 2, 2025 through August 19, 2026, so the diagnostic changes both
qualification observability and calendar composition; it does not isolate the
warranty-deed proxy's measurement error. The interval still undercovers at
79.8%, so the operational conclusion does not change.

The 30-day seasoning sensitivity leaves 10,506 Tampa and 6,021 Orlando test
transactions. Model MAE remains below the rolled-prior benchmark in both
markets: \$76,074 versus \$78,310 in Tampa and \$127,896 versus \$146,593 in
Orlando. The main comparison is therefore not being driven by a small set of
very rapid cross-quarter resales.

### Named iBuyer ownership spells

The linkage stage finds 5,060 recorded ownership spells, 3,395 of them involving
Opendoor as the acquirer. The duration-eligible population contains 4,185
improved, single-family spells for Opendoor and Offerpad, the two operators with
adequate support in both markets. After parcel purges, 4,162 episodes remain.
Fourteen pre-2022 training spells were acquired too near the test boundary to
contribute one complete pre-boundary risk week, leaving 4,148 episodes in model
fitting or scoring. The direct descriptive comparison is restricted to 2,391
acquisitions in the common January 2022 onward window. County deed qualification
is not required because the outcome is title-holding duration rather than an
arm's-length price estimand.

The duration is

```math
T_i
=
\text{recorded resale date}_i
-
\text{recorded acquisition date}_i.
```

It is not MLS days on market. It includes renovation, pre-listing work, contract
time, and recording lags. Open spells inside the three-year administrative
horizon remain right censored. Unresolved spells that reach that horizon also
enter as censored because their non-exit status is fully observed throughout
the 52-week model window. Observed linked exits are assigned to the ceiling of
deed-to-deed days divided by seven. Censored spells contribute only complete
observed weeks, so those
with fewer than seven observed days do not enter either the weekly risk panel
or the model-eligible Kaplan-Meier table below.

| Opendoor slice | Comparable acquisitions | Recorded exits | Kaplan-Meier median exit | Exit by 120 days |
|---|---:|---:|---:|---:|
| Hillsborough County | 1,190 | 1,132 | 135 days | 43.9% |
| Orange County | 935 | 881 | 120 days | 50.1% |

A penalized discrete-time logit estimated in historical Tampa episodes ranks
future recorded exits better than chance: risk-row AUC is 0.630 in Tampa's
out-of-time sample and 0.649 in Orlando. Its probabilities are not well
calibrated under the shift. Brier skill relative to a constant training hazard
is -0.373 in Tampa and -0.253 in Orlando. The operational conclusion is
abstention or local recalibration, not deployment. Rank discrimination without
probability calibration is not enough for a pricing decision.

The discrete-time logit maximizes the ordinary person-period likelihood: every
week in which a home is at risk contributes one Bernoulli observation. Training
acquisitions all occur before January 2022, and their outcomes are
administratively censored at that boundary. No disposition recorded in the
2022-forward test period can enter model fitting. Brier score, log loss, and AUC
are evaluated over the corresponding person-period risk rows.

### What the deed study does and does not identify

The public records identify:

- recorded sale prices and dates;
- repeat-sale price prediction in the documented target population;
- named-operator acquisition-to-resale title duration; and
- descriptive differences across operators and counties.

They do not identify:

- rejected acquisition offers;
- list prices, markdown histories, or buyer demand at unposted prices;
- days on market;
- renovation spending, concessions, financing, taxes, or selling costs;
- net contribution profit;
- causal performance differences between operators; or
- a pricing policy's real-market lift.

Gross deed-price differences are not called profit. Operator indicators in the
duration model are descriptive because firms select different homes, sellers,
locations, and dates.

## Evidence layer 2: the controlled policy experiment

The decision system requires counterfactual price response. Historical offers
and list prices are endogenous: operators choose them in response to property
quality, seller information, latent demand, inventory age, and other signals.
Regressing outcomes on those chosen prices would mix the response to price with
the process that selected the price.

The location-neutral controlled experiment resolves that problem inside a known
data-generating process. It first constructs appraisal-like pre-offer and
pre-listing value references from observable property and market characteristics
plus independent measurement noise. It then independently randomizes:

- the acquisition offer as a share of that observable pre-offer reference; and
- the resale list-price premium relative to that observable pre-listing reference.

It then generates seller acceptance, weekly sale events with right censoring,
and conditional proceeds from known response functions. Training and evaluation
environments use different random seeds and different covariate distributions.
Simulator truth is used only for scoring response recovery.

The fitted components are:

1. a contemporaneous log-price anchor with a split-conformal interval;
2. a regularized seller-acceptance logit;
3. a discrete-time sale-hazard logit; and
4. transparent cost and terminal-recovery scenarios.

The fitted valuation, acceptance, and hazard models are adapted to the decision
protocols and passed into the 17-week optimizer. They are not replaced by the
known simulator equations at decision time. For every candidate bid, the
acceptance adapter recomputes offer divided by the same observable pre-offer
reference used in treatment assignment. The hazard adapter likewise computes
posted list price relative to the same observable pre-listing reference used in
its randomized treatment. Neither adapter divides by latent value. The lower
bound, point prediction, and upper bound of the conformal interval are mapped
heuristically into 17-week proceeds stress points. This is a contemporaneous
home-value anchor, not a learned terminal proceeds forecast. The stress weights
are explicit modeling choices, not calibrated probabilities, because conformal
endpoints are not distributional quantiles.

### Controlled-experiment results

The evaluation environment contains 480 generated homes excluded from fitting.
The hazard exercise retains right-censored listings.

| Component | Transparent baseline | Result | Interpretation |
|---|---:|---:|---|
| Contemporaneous value anchor | Pooled-median MAE \$123,567 | MAE \$43,536 | 64.8% lower MAE under the documented shift |
| 90% value interval | None | 90.0% coverage | Marginal coverage in the generated evaluation population |
| Seller acceptance | Pooled-rate Brier 0.2440 | Brier 0.1837 | Probability accuracy inside the experiment |
| Weekly sale hazard | Period-only log loss 0.5316 | Log loss 0.4353 | Price-aware hazard improves on the period baseline |
| Offer response | Truth +1.600 log-odds per +10 points | Fitted +1.667 | Absolute response-recovery error 0.067 |
| Resale-price response | Truth -1.200 log-odds per +10 points | Fitted -1.288 | Absolute response-recovery error 0.088 |

These numbers validate implementation and recovery of known randomized
responses. They do not estimate Florida response functions.

## Decision outputs and abstention

For each property-level case, the optimizer evaluates an acquisition-offer grid
and the feasible resale actions `hold`, `cut 1%`, `cut 2.5%`, and `cut 5%` over
17 weeks. Acquisition candidates are fractions of the observable pre-offer
reference and remain within the randomized offer-ratio support $[0.82,1.02]$;
the fitted acceptance adapter refuses extrapolation. Initial and reachable list
prices remain within the randomized premium support $[-0.30,0.15]$ relative to
the observable pre-listing reference, and the fitted hazard adapter also fails
closed outside support. The terminal state applies a case-specific liquidation
discount. In this version, the optimizer remixes the stress weights each week after a no-sale
outcome rather than carrying a persistent latent regime and updating beliefs.
That simplified state transition is inspectable and intentionally listed as a
limitation; posterior learning is a natural extension. The public output
includes:

- the selected offer and estimated acceptance probability;
- the recommended no-sale markdown path;
- expected contribution per quoted lead, including the zero contribution from rejection;
- approximate loss probability from the compressed outcome grid and
  worst-decile mean positive loss; and
- an explicit `price`, `review`, or `abstain` state.

The three worked cases are scenario analyses drawn from the controlled
evaluation environment. They test routine demand, high carrying cost, and weak
valuation support. They are not estimates of how any real Florida home should be
priced. Inspect every path in the
[self-contained decision laboratory](https://preston-monk.github.io/offer-to-exit/artifacts/release/demo.html).

The controlled listings hold one randomized price premium fixed over each
generated spell. Using the fitted common response to score an adaptive markdown
path assumes away price-history and carryover effects. It is an explicit
laboratory extension, not evidence that the dynamic response has been identified.

## What the code demonstrates

- Two reproducible adapters for materially different county data systems
- Streaming preparation of millions of records without releasing direct identifiers
- Explicit date, property-use, censoring, linkage, and claim-scope contracts
- Chronological and parcel-separated evaluation
- A nonlinear repeat-sale model with parcel-grouped conformal calibration
- A discrete-time duration model that retains right-censored ownership spells
- An explicit external-market transport test without Orlando refitting
- Randomized treatment assignment for otherwise endogenous price-response objects
- Fitted-model adapters into a finite-horizon stochastic optimizer
- Typed decision objects, a FastAPI boundary, and versioned aggregate evidence
- Tests for leakage, accounting identities, edge cases, privacy, reproducibility, and failure states

The repository reports negative evidence when it matters. In particular, the
real-data duration model's probability calibration fails out of time and across
geography even though its ranking remains informative.

## Reproduce the project

Install the locked Python environment and run the complete quality suite:

```bash
make install
make check
```

Rebuild the controlled experiment and decision laboratory:

```bash
make reproduce
```

Rebuild the Florida study from the current official county files:

```bash
make florida-data
make florida-study
```

The county downloads are large and live publisher sources can be revised. The
checked-in aggregate artifacts are the September 3, 2026 snapshot identified by
the hashes in `artifacts/release/florida_manifest.v2.json`. Raw files and
transaction-level outputs remain under git-ignored `data/raw/` and
`data/processed/` directories.

Run the illustrative API locally:

```bash
uv run uvicorn offer_to_exit.serving:app --reload
```

## Repository map

```text
configs/                    Executed controlled-experiment configurations
data/                       Source contract and ignored local data zones
docs/DATA_DESIGN.md         Florida samples, estimands, and privacy rules
docs/METHODS.md             Statistical and decision methods
src/offer_to_exit/data/     County download, normalization, and linkage
src/offer_to_exit/empirical Real-data valuation and duration studies
src/offer_to_exit/models/   Controlled-experiment statistical models
src/offer_to_exit/decision/ Dynamic pricing and acquisition optimization
src/offer_to_exit/serving/  Typed API boundary
tests/                      Unit, contract, and end-to-end checks
reports/                    Data, model, policy, results, and limitations cards
artifacts/release/          Versioned aggregate tables, figures, and HTML reports
```

## Go deeper

- [Inspect the Florida evidence](artifacts/release/florida_evidence.html)
- [Read the data design](docs/DATA_DESIGN.md)
- [Read the methods](docs/METHODS.md)
- [Inspect the complete results](reports/results.md)
- [Review assumptions and failure modes](reports/limitations.md)
- [Read the decision contract](SYSTEM_SPEC.md)

Code is released under the [MIT License](LICENSE). External datasets retain
their publishers' terms and are not relicensed by this repository. Citation
metadata are in [CITATION.cff](CITATION.cff).
