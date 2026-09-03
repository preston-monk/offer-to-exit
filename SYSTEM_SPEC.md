# System Specification

## Economic decision

Offer-to-Exit represents a property-level housing intermediary. The operator
posts an acquisition offer and, if the seller accepts, holds the home as
inventory while choosing a weekly resale price. The acquisition bid is therefore
valued through the continuation value of the best feasible exit policy.

The central question is:

> How much should the operator pay for a home, given the value, duration, and
> downside risk of the resale policy it expects to follow after acquisition?

## Two evidence layers

The release keeps observed-market evidence separate from experimental decision
evidence.

| Layer | Data | Role |
|---|---|---|
| Florida transaction study | Hillsborough County and Orange County recorded sales | Predict recorded repeat-sale prices, describe named iBuyer ownership duration, and test geographic transport |
| Controlled decision experiment | Location-neutral generated homes, offers, and weekly listings | Recover known seller and buyer price responses and pass fitted models into the optimizer |

The Florida records do not identify rejected acquisition offers, weekly list
prices, buyer arrivals, repairs, operating costs, or counterfactual demand. The
controlled experiment does not estimate Florida behavioral elasticities or
real-market policy lift.

## Decision contract

| Field | Implemented contract |
|---|---|
| Acquisition action | Choose one offer from a finite grid |
| Resale action | Hold, cut 1 percent, cut 2.5 percent, or cut 5 percent |
| Resale horizon | 17 weekly decisions, followed by terminal liquidation |
| Markdown floor | Markdown actions cannot push the list price below 70 percent of the observable pre-listing reference value; a pre-existing lower price may still be held |
| Profit accounting | Sale proceeds less the acquisition offer, repairs, acquisition closing costs, resale transaction costs, and holding costs |
| Decision output | Return an offer only when the best grid point has positive risk-adjusted value; otherwise abstain |
| Statistical inputs | Fitted valuation, seller-acceptance, and discrete-time sale-hazard models from the controlled experiment |

The worked cases use five candidate offers expressed as fractions of the
observable pre-offer reference. Every fitted-case ratio lies inside the
generator's randomized support of $[0.82,1.02]$, and the acceptance adapter
fails closed outside those bounds. Case-specific cost schedules and
terminal-liquidation discounts are declared in code. The optimizer does not
currently impose separate hard constraints on margin, sell-through, or loss
probability.

## Profit and downside objective

For acquisition offer $a$, sale price $P_\tau$, sale week $\tau$, repair cost
$R$, acquisition cost $K$, transaction cost $C(P_\tau)$, and weekly holding
cost $H$, contribution profit is

```math
\Pi(a,P_\tau,\tau)
=
P_\tau-a-R-K-C(P_\tau)-(w_0+\tau+1)H,
```

where $w_0$ is the number of pre-listing holding weeks.

For a discrete profit distribution $\Pi$ and tail mass $\alpha$, define the
average positive loss in the worst tail as

```math
D_\alpha(\Pi)
=
\frac{1}{\alpha}
\int_0^\alpha \max\{-Q_\Pi(u),0\}\,du.
```

The released cases use $\alpha=0.10$. At each state, the dynamic program
maximizes

```math
\mathbb E[\Pi\mid s_t]-\lambda D_{0.10}(\Pi\mid s_t).
```

This is a recursive, state-contingent downside rule. It is not one global
ex-ante mean-CVaR commitment.

## Weekly resale recursion

For state $s_t$ and feasible action $u_t$, the fitted hazard supplies the
probability of sale in week $t$. Conditional sale proceeds and the no-sale
continuation distribution are combined by backward induction. If the home is
still unsold after the seventeenth weekly decision, which is zero-indexed week
16, the code applies a case-specific liquidation discount.

The implementation propagates a compressed discrete profit distribution rather
than only an expected value. Scenario weights are reused at later states. The
current state does not preserve a latent market scenario or update beliefs after
observing no sale. That independent scenario remix is a simplifying assumption,
not an estimated law of motion.

## Fitted-model boundary

The controlled release fits:

1. a contemporaneous linear log-value anchor with a split-conformal interval;
2. a regularized logit for seller acceptance as a function of offer divided by
   an observable pre-offer value reference and observed covariates; and
3. a regularized discrete-time logit for weekly sale incidence as a function of
   list price divided by an observable pre-listing value reference and observed
   covariates.

Three profiles are selected from the independent evaluation sample using
observed covariates and fitted valuation outputs, never simulator truth or
realized outcomes. For each profile, the fitted contemporaneous value point estimate and interval
endpoints are mapped heuristically to three 17-week proceeds stress points. This
is not a learned terminal proceeds forecast. Their scenario weights are
transparent heuristics, not probabilities calibrated from the conformal
interval. The fitted acceptance and hazard models then score every candidate
offer and reachable weekly price state.

The generator constructs the pre-offer reference from observable property and
market characteristics plus independent measurement noise before randomizing
the offer ratio. The fitted acceptance adapter uses that same observable generated reference
when it converts each candidate dollar offer into a ratio. Latent market value
is never an input to treatment assignment or decision-time acceptance scoring.
An analogous pre-listing reference defines the randomized list-price premium.
The fitted hazard uses that one observable generated denominator for sale incidence, while
valuation scenarios affect only conditional proceeds. Its allowed premium range
is $[-0.30,0.15]$, which contains the initial prices and 70 percent markdown
floor. The controlled hazard equation contains no omitted latent-demand term, so
its randomized price coefficient and fitted coefficient share the same
conditional log-odds estimand.

Each generated listing holds one randomized list-price premium fixed during its
spell. Applying the fitted common price slope to changing weekly prices assumes
away price-history and carryover effects. The adaptive path is therefore a
declared decision-model extension, not a dynamically identified response.

## Scope

This is a research prototype for a single home. It does not model portfolio
capital allocation, correlated inventory risk, geographic concentration,
renovation operations, a learned proceeds distribution, belief updating,
production service levels, or realized business impact. It is not intended for
appraisal, lending, insurance, taxation, or consumer eligibility.
