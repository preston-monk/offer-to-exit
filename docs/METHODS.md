# Methods

## 1. Economic sequence

The model follows the order in which a housing intermediary receives
information and acts:

```text
home information -> acquisition offer -> seller response -> ownership
-> weekly resale prices -> recorded sale or terminal liquidation
```

Let $q(a\mid x)$ be the probability that a seller accepts acquisition offer
$a$. After acceptance, let $s_t$ denote the state at the start of resale week
$t$ and $p_t$ the price after the weekly action. Aggregate buyer demand is a
discrete-time sale hazard:

```math
h_t(p_t\mid s_t)
=
\Pr(\text{sale in week }t\mid\text{unsold at }t,p_t,s_t).
```

Let contribution profit when the home sells in week $t$ at realized transaction
price $P_t$ be

```math
G_t(a,P_t)
=
P_t-a-R-K-C(P_t)-(w_0+t+1)H,
```

where $R$ is repair cost, $K$ is acquisition cost, $C(P_t)$ is resale
transaction cost, $w_0$ is pre-listing holding time, and $H$ is weekly holding
cost. The risk-neutral recursion is

```math
\begin{aligned}
V_t(s_t)=\max_{p_t\in\mathcal P(s_t)}\Big\{&
h_t(p_t\mid s_t)\,
\mathbb E[G_t(a,P_t)\mid\text{sale}]\\
&+[1-h_t(p_t\mid s_t)]V_{t+1}(s_{t+1})
\Big\}.
\end{aligned}
```

The current week's holding cost is embedded in every realized sale or terminal
liquidation payoff through $G_t$. If the home remains unsold, the later payoff
contains more cumulative holding weeks. Writing a separate $-H$ on the no-sale
branch would therefore double count holding cost relative to the implementation.

Offer-to-Exit propagates a compressed discrete approximation to the profit
distribution and replaces the expected-value criterion with the recursive
downside objective described below.

## 2. Florida repeat-sale valuation

### Population

The estimator uses improved single-family repeat sales with an observed prior
sale no more than 4.5 years earlier. Sale prices must lie between \$25,000 and \$10
million. Hillsborough requires the county-qualified flag. Orange uses its sale
description where populated and uses a warranty-deed code as a proxy when the
historical description is missing.

The prior eligible sale must occur in a strictly earlier calendar quarter than the
target sale. The prior-quarter normalization is therefore fully observed before
the target quarter. The release also reruns the complete design after requiring
at least 30 days between deeds.

This is a high-turnover repeat-sale estimand, not an automated valuation model
for every home. The prior eligible sale provides a property-specific proxy for persistent
quality that cannot be measured symmetrically in the two county sources;
renovations and other unobserved changes may still remain.

### Estimator

For current price $P_i$, prior price $P_i^-$, the county median $M_i^-$ in the
prior eligible sale's quarter, and the lagged county median $M_{q-1}$ before the current
sale, the model predicts

```math
\log(P_i/M_{q-1})
```

with three inputs:

```math
\log(P_i^-/M_i^-),
\qquad
\text{years since prior sale},
\qquad
\text{calendar quarter}.
```

The estimator is a `HistGradientBoostingRegressor` with learning rate 0.06, 180
iterations, at most 31 leaves, and L2 regularization 1.0. Numeric missing values
are median-imputed. No Orange outcome or Orange structural attribute enters
fitting.

The fair benchmark is

```math
\widehat P_i^{\mathrm{base}}
=
P_i^-\frac{M_{q-1}}{M_i^-},
```

which rolls the home's own prior eligible price forward with the same county market
movement available to the fitted model.

### Time and geography

Hillsborough sales before 2022 form the proper training period. Sales in 2022
and 2023 form the interval-calibration period. Sales from 2024 onward form the
Tampa out-of-time test. Parcels assigned to a later period are purged from
earlier target samples.

A 90 percent split-conformal interval is calibrated on absolute normalized-log
residuals. To avoid transaction-weighting a parcel, each calibration parcel
contributes one conservative score: its maximum residual. The same fitted model
and interval radius are applied to Orange sales from 2024 onward. Orange is
excluded from fitting and calibration, but this is not presented as a
preregistered untouched holdout.

Metrics give each parcel equal total weight and report transaction and parcel
counts, MAE, RMSE, median absolute error, mean and median percentage error, mean
error, weighted $R^2$, interval coverage, and median interval width.

## 3. Florida recorded-disposition model

Named iBuyer episodes begin with a classified operator acquisition and end with
the first later deed where the same operator is grantor. Both deeds must report
at least \$10,000 in consideration. The screen excludes nominal or
administrative transfers that do not reveal interpretable purchase or resale
prices. Completed, open
right-censored, and administrative-horizon spells enter the modeling panel;
repeat-acquisition-before-exit flags do not. Administrative-horizon spells have
at least 1,095 observed non-exit days and are therefore fully observed through
the 52-week model horizon.

Observed linked exits use the ceiling of deed-to-deed days divided by seven.
Censored spells contribute only complete
observed weeks, so a censored spell with fewer than seven days of follow-up does
not enter the weekly risk panel.

The study uses improved single-family acquisitions from 2016 onward and keeps
operators with at least 25 episodes in both counties. The current supported set
is Opendoor and Offerpad. Qualification is not required for this title-duration
estimand.

For episode $i$, the model expands one row per week at risk and estimates

```math
\Pr(D_{it}=1\mid D_{i1}=\cdots=D_{i,t-1}=0,X_i,t)
```

with a penalized logistic regression. Inputs are log acquisition price,
acquisition year, acquisition quarter, operator indicators, and week indicators.
No resale price, realized duration, gross spread, or Florida list price is a
predictor.

Fitting maximizes the ordinary person-period likelihood: every week in which a
home remains at risk contributes one Bernoulli observation. This is the
discrete-time hazard likelihood, so weights do not depend on the subsequently
realized duration or censoring time.

The model fits Tampa episodes acquired before January 2022. Follow-up for every
training spell is administratively censored at January 1, 2022, so a future
disposition cannot leak across the test boundary. The common future evaluation
window contains Tampa and Orlando acquisitions from January 2022 onward. The
risk window is capped at 52 weeks; a later disposition is treated as censored at
that horizon. Metrics report person-period Brier score against a constant
training hazard, Brier skill, log loss, observed and predicted hazard, and
risk-row AUC.

Kaplan-Meier summaries by market and operator preserve open spells with at
least one complete week of observed follow-up. The outcome
is deed-to-deed title duration, not listing duration. Operator coefficients are
regularized associations, not causal effects.

## 4. Location-neutral controlled experiment

County deeds reveal only chosen transactions. They cannot identify the response
to offers sellers did not receive or list prices the operator did not post. A
separate generated experiment therefore constructs appraisal-like pre-offer and
pre-listing value references from observable property and market characteristics
plus independent measurement noise, then randomizes:

- acquisition offer divided by that observable pre-offer reference over a
  bounded range; and
- list-price premium relative to the observable pre-listing reference.

Training and evaluation environments use independent seeds. The evaluation
environment shifts home values, square footage, market heat, mortgage rates,
appreciation, and synthetic submarket shares. No Florida record or Florida-fitted
parameter seeds this generator.

### Fitted models

The controlled experiment fits:

1. a linear log-price model with a 90 percent split-conformal interval, using
   generated structural, market, and submarket features;
2. a regularized seller-acceptance logit using offer divided by the observable
   pre-offer reference, seller urgency, market heat, repair-cost fraction, and
   submarket; and
3. a regularized discrete-time sale-hazard logit using list-price premium,
   market heat, condition, mortgage rate, submarket, and week indicators.

The pooled acceptance rate, pooled median price, and period-specific sale hazard
are the reported component baselines. Known offer and list-price log-odds
coefficients are used for evaluation only. The controlled hazard equation has no
omitted latent-demand shock, so the fitted and known randomized price
coefficients refer to the same conditional log-odds estimand.

Each generated listing holds its randomized list-price premium fixed throughout
the spell. Applying the fitted common slope to a weekly markdown path assumes no
price-history or carryover effects. That extension is a transparent decision
laboratory assumption, not an experimentally validated dynamic response.

### Fitted-model decision cases

Three cases are selected from the independent evaluation environment using
observed covariates and fitted valuation outputs: strong demand, weak demand
with high carrying costs, and sparse training support with a consequential
lower valuation stress. The selected rows are scored by all three fitted
models. Simulator truth and realized outcomes are excluded from selection and
the adapters.

For each candidate acquisition price, the acceptance adapter divides the dollar
offer by the selected home's observable pre-offer reference. This is the same
denominator used to assign the randomized offer in the generator. The fitted
contemporaneous home-value anchor is used heuristically for 17-week proceeds
stress points and acquisition economics, but it is not a learned terminal
proceeds forecast and is not substituted for the acceptance treatment denominator.
Candidate acquisition bids are constructed as fractions of the pre-offer
reference and remain inside the randomized support $[0.82,1.02]$. The fitted
acceptance adapter rejects any attempted extrapolation beyond those bounds.

For each case, the fitted valuation lower endpoint, point estimate, and upper
endpoint become three proceeds-cap scenarios. The assigned weights differ by
case and are declared in code. A split-conformal interval is a coverage set, not
a three-point probability distribution, so these weights are heuristic stress
weights rather than calibrated posterior probabilities.

At each reachable price state, the list price is divided by the selected home's
observable pre-listing reference and passed once to the fitted hazard. This is the
same denominator used to assign the randomized list-price premium in the
generator. Every scored premium remains inside $[-0.30,0.15]$; the 70 percent
markdown floor is defined against that same reference, and the adapter rejects
extrapolation. Conditional proceeds still vary across valuation scenarios and
equal the smaller of the post-negotiation list price and scenario value. This
separates sale incidence from proceeds while keeping the proceeds model
deliberately simple.

## 5. Dynamic decision rule

The resale policy considers hold, 1 percent cut, 2.5 percent cut, and 5 percent
cut actions over 17 weeks. It enforces a floor of 70 percent of the observable pre-listing reference
and applies a declared terminal-liquidation discount if inventory remains. For
fitted cases, that floor is 70 percent of the observable pre-listing reference.

For profit quantile function $Q_\Pi(u)$, the downside statistic is

```math
D_{0.10}(\Pi)
=
10\int_0^{0.10}\max\{-Q_\Pi(u),0\}\,du.
```

Every state chooses the action that maximizes expected profit minus the
case-specific risk weight times $D_{0.10}$. The acquisition layer combines the
optimized exit distribution with fitted seller acceptance and the rejection
outcome, then applies the same criterion to each offer in the five-point grid.
The public action is the best offer when its objective is positive and abstention
otherwise.

The risk criterion is reoptimized at every state. It is therefore a recursive
downside rule rather than one global static mean-CVaR commitment. The scenario
identity is also not carried through time: the current code remixes the same
scenario weights after a no-sale transition and does not update beliefs from
failure to sell.

## 6. Interpretation

The Florida layer supports predictive and descriptive claims for the stated
samples. The controlled layer supports recovery of known generated responses and
inspection of a fitted-model decision system. Neither layer identifies a real
seller-acceptance curve, a causal Florida list-price elasticity, net operator
profit, or realized policy lift.
