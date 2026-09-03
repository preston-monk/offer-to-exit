# Model Card

## System structure

Offer-to-Exit is a linked decision system rather than one model. The empirical
Florida models and the controlled decision models have different evidentiary
roles.

| Component | Method | Interpretation |
|---|---|---|
| Florida repeat-sale value | Histogram gradient boosting plus grouped split-conformal interval | Prediction for the stated high-turnover repeat-sale population |
| Florida recorded disposition | Penalized discrete-time logit | Association and prediction for deed-to-deed ownership duration |
| Controlled contemporaneous value anchor | Linear log-price model plus split-conformal interval | Prediction inside the generated experiment |
| Controlled seller acceptance | Regularized logit | Causal-response recovery inside the randomized generator |
| Controlled weekly sale | Regularized discrete-time logit | Causal-response recovery inside the randomized generator |
| Optimizer | Finite-horizon backward induction over discrete profit distributions | Decision behavior under fitted controlled models and declared assumptions |

Florida models do not currently feed the optimizer. The controlled experiment is
location-neutral and is not calibrated to Florida.

## Florida repeat-sale model

**Population:** improved single-family repeat sales with a prior eligible sale no
more than 4.5 years earlier. The prior eligible sale must be in a strictly earlier
calendar quarter; a separate sensitivity requires at least 30 days between
deeds.

**Target:** log current price relative to the lagged current county-quarter
median.

**Features:** log prior eligible price relative to the county median in the prior-sale
quarter, years since the prior eligible sale, and current calendar quarter.

**Estimator:** `HistGradientBoostingRegressor` with median imputation, learning
rate 0.06, 180 iterations, at most 31 leaves, and L2 regularization 1.0.

**Interval:** 90 percent split conformal. Hillsborough 2022 and 2023 sales form
the calibration period; each parcel contributes its maximum absolute normalized
log residual as one calibration score.

**Baseline:** the home's own prior eligible sale price rolled forward by the county median
change between its prior-sale quarter and the quarter preceding its current
sale.

**Evaluation:** fit before 2022, evaluate in Tampa from 2024 onward, and apply
the same fitted model and interval radius to Orlando from 2024 onward. Parcels
are purged across Tampa target partitions. Metrics weight parcels equally.

**Interpretation:** Orange outcomes are excluded from fitting and calibration.
The exercise is an external-market evaluation, not a preregistered untouched
holdout and not evidence of national transport.

## Florida recorded-disposition model

**Population:** improved single-family Opendoor and Offerpad ownership spells
from 2016 onward, including completed and censored episodes.

**Target:** probability that a recorded disposition occurs in week $t$,
conditional on no recorded disposition before week $t$.

**Features:** log acquisition price, acquisition year, acquisition quarter,
operator, and risk-week indicators.

**Estimator:** L2-regularized logistic regression on one row per week at risk,
using the ordinary person-period likelihood. Sample weights do not depend on
the realized duration or censoring time.

**Horizon:** 52 weeks. A recorded exit after week 52 is censored at that horizon
for model fitting and scoring. Observed exits use ceiling weeks, while censored
spells contribute completed weeks only and require at least one complete week.

**Evaluation:** fit on Tampa acquisitions before 2022, with every training
spell administratively censored at January 1, 2022; score Tampa and Orlando
acquisitions from 2022 onward. Metrics report a constant training-hazard
baseline as well as person-period Brier score, Brier skill, log loss, observed
and predicted hazard, and risk-row AUC.

**Interpretation:** the outcome is title duration, not days on market. Operator
coefficients are regularized conditional associations. They do not estimate a
causal operator effect or a response to list price.

## Controlled contemporaneous value model

**Estimator:** linear regression of log generated sale price on square feet,
bedrooms, bathrooms, age, condition, quality, lot size, market heat, mortgage
rate, and synthetic submarket.

**Interval:** 90 percent split conformal using a random calibration subset of
the controlled training environment.

**Baseline:** pooled median sale price from the training environment.

**Risk:** this is a deliberately transparent specification scored on an
independent shifted generated environment, not a model of a real county.

## Controlled seller-acceptance model

**Estimator:** regularized logistic regression.

**Features:** randomized offer divided by an observable pre-offer value
reference, generated pre-offer seller urgency, market heat, repair-cost
fraction, and synthetic submarket.

**Baseline:** pooled training acceptance rate.

**Evaluation:** Brier score, log loss, 10-bin calibration error, monotonicity,
and recovery of the known log-odds effect of a 10-percentage-point offer change.

**Interpretation:** the generator first records an appraisal-like reference
constructed from observable property and market characteristics plus independent
measurement noise. The offer ratio is randomized against that reference. The
decision adapter uses the same observable generated denominator for candidate offers and
never reads latent market value. The causal claim stops at that generated
environment.

**Decision-time support:** candidate acquisition offers are divided by the same
pre-offer reference and must remain within the randomized interval
$[0.82,1.02]$. The fitted adapter raises an error rather than extrapolating.

## Controlled weekly sale-hazard model

**Estimator:** regularized discrete-time logistic regression with period
indicators and right censoring.

**Features:** randomized list-price premium relative to an observable
pre-listing value reference, market heat, property condition, mortgage rate,
and synthetic submarket.

**Baseline:** training sale rate by listing period.

**Evaluation:** person-period Brier score and log loss, horizon sell-through,
monotonicity, and recovery of the known log-odds effect of a
10-percentage-point increase in overpricing.

**Interpretation:** the price coefficient is causal only inside the randomized
generator. The hazard equation contains no omitted latent-demand term, so the
known and fitted price coefficients share the same conditional log-odds
estimand. Observed Florida markdowns are not used or claimed.

**Decision-time support:** every posted price is divided by the same observable
generated pre-listing reference used in treatment assignment and must produce a premium
inside $[-0.30,0.15]$. The adapter raises an error rather than extrapolating.

## Decision adapters

Three evaluation rows are selected using observed covariates and fitted
valuation outputs for strong demand, weak demand with high carrying cost, and
sparse training support with a consequential lower valuation stress. Each row
is scored by the fitted controlled models. Truth columns and realized outcomes
are not used for selection or passed to the adapters.

The fitted contemporaneous value lower endpoint, point estimate, and upper
endpoint are mapped heuristically to three 17-week proceeds stress points. This
is not a learned terminal proceeds forecast. Their assigned weights are declared
stress weights, not probabilities implied by conformal coverage. The fitted
hazard scores each reachable posted price once relative to the observable
pre-listing reference. The value scenarios do not change the price-treatment
denominator.

Conditional headline proceeds are capped at the lower of post-negotiation list
price and scenario value. There is no separately trained proceeds model.

## Decision criterion

The 17-week optimizer chooses among hold, 1 percent cut, 2.5 percent cut, and 5
percent cut actions. It propagates a compressed discrete contribution-profit
distribution and subtracts a case-specific multiple of mean positive loss in
the worst 10 percent of outcomes.

This criterion is recomputed at every state. It is not one global static
mean-CVaR commitment. Scenario identities are not persistent through time, and
no belief update follows a week without sale.

The acquisition layer combines each optimized exit distribution with fitted
seller acceptance and the rejection outcome. It selects the best five-point
offer-grid value and publishes it only if risk-adjusted value is positive.

## Reported comparisons

The release implements component baselines for Florida repeat-sale value,
Florida recorded disposition, controlled value, controlled acceptance, and
controlled sale hazard. It also publishes three fitted-model worked decisions.
It does not publish a population policy backtest, oracle regret calculation,
profit-risk frontier, or realized-market lift.

## Intended use and status

This is an inspectable research prototype for learning about housing inventory
pricing, model transport, censoring, uncertainty, and dynamic decisions. It is
not a production model and must not be used for real offers, appraisals, lending,
insurance, taxation, or consumer decisions.
