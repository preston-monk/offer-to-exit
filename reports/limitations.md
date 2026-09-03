# Limitations and Risk Register

## Claims ledger

| Statement | Status | Correct interpretation |
|---|---|---|
| The code fits a Tampa repeat-sale model and scores 2024-forward Tampa and Orlando sales | Supported by the Florida artifact | Predictive performance for the documented high-turnover repeat-sale samples |
| Orange outcomes are excluded from fitting and interval calibration | Supported by code | External-market evaluation, not a preregistered untouched holdout |
| Named Opendoor or Offerpad episodes have the reported holding distribution | Descriptively supported | Recorded deed-to-deed ownership duration, conditional on classifier and linkage rules |
| An operator coefficient is a causal effect | Not supported | Operator selection, timing, homes, and strategies differ |
| The controlled models recover known price responses | Supported inside the generated experiment | Implementation evidence, not a Florida elasticity |
| The worked cases are produced by fitted models | Supported by adapters and tests | Decision behavior under generated covariates and declared assumptions |
| The project estimates real operator profit or policy lift | Not supported | County records omit key costs, actions, and counterfactuals |

## Florida data risks

### Orange qualification proxy

The Orange layer's sale description is missing for the retrieved 2022 through
2024 history. When qualification is unknown, the valuation screen treats a
warranty deed (`WD`) as a proxy. Warranty-deed form does not establish an
arm's-length transaction. The Orlando external sample can therefore contain
transfers that Hillsborough's official qualification flag would exclude.

This is measurement asymmetry between counties. It can affect apparent model
transport even when the predictive relationship is stable.

### Narrow repeat-sale population

The valuation study requires a usable prior transaction no more than 4.5 years
earlier. It therefore overrepresents high-turnover properties and cannot be
interpreted as coverage of the full housing stock. The prior-deed design helps
control property quality, but it selects on resale and requires accurate parcel
linkage. The main sample requires the prior eligible sale to fall in a strictly earlier
quarter, and the release reports a 30-day minimum-gap sensitivity. Neither rule
resolves every rapid resale, bundled transfer, or deed-correction ambiguity.

The Orange service begins in 2022 in the retrieved snapshot. Even under the same
4.5-year rule, the observed history and distribution of prior eligible-sale gaps differ
from Hillsborough. The release reports those gap distributions rather than
calling the markets identical.

### Repeat-sale baseline

The fair baseline uses the home's own prior eligible price and county median change. It is
substantially stronger than a pooled or lagged-median benchmark. A fitted model
that does not improve on this baseline has not demonstrated incremental value,
even if its raw prediction error appears moderate.

### Time and entity leakage

The model uses lagged county medians and chronological Tampa splits. Parcels in
later Tampa target periods are purged from earlier target periods. This reduces,
but does not eliminate, leakage risk from revised county records, deed timing,
same-day transactions, or unobserved administrative corrections.

### Named-operator classification

The classifier uses narrow anchored patterns for four classic iBuyer brands.
That favors precision over recall. Uncoded affiliates can be missed, and a
matching legal-name root can still be misclassified. There is no exhaustive
corporate-family registry with validity dates in this release.

### Episode linkage

The linker infers ownership from deed-party roles. It does not observe contract
dates, listing dates, renovations, concessions, affiliate transfers, or
operational status. Both the acquisition deed and the first later eligible
operator-as-grantor deed must have at least \$10,000 in recorded consideration;
the resulting linked pair may not always be
the economically relevant retail exit.

Spells with a repeated acquisition before exit are excluded from the modeling
panel because their identity is ambiguous. Spells hitting the 1,095-day
administrative linkage horizon remain as censored observations because their
non-exit status is observed throughout the 52-week model horizon. The 52-week
hazard also treats later observed dispositions as censored at week 52. Results
depend on these choices. Censored spells observed for fewer than seven days contribute
no complete risk week and are excluded. The reusable panelizer maps externally
supplied same-day exits to week one, while the Florida linker requires a
strictly later exit deed.

### Selection and interpretation

Operator episodes are conditional on a classified firm acquiring the home.
Completed-only summaries would select again on disposition, so the release uses
Kaplan-Meier estimates and retains open spells. Still, deed-to-deed duration is
not days on market, and a deed-price difference is not profit.

## Model risks

### Geographic shift

Orange is excluded from model fitting and conformal calibration, but the design
was not preregistered and Orange data were observable during development. The
external result is a transparent cross-market evaluation, not a pristine trial
of a specification fixed before anyone saw Orange.

Prediction error and conformal coverage can change because housing composition,
transaction selection, source measurement, or regimes differ. Two Florida
counties do not establish statewide or national transport.

### Conformal coverage

The 90 percent interval uses one maximum residual per Hillsborough calibration
parcel. Its finite-sample logic relies on exchangeability that may fail over time
or geography. Marginal coverage does not imply conditional coverage for price
tiers, neighborhoods, or unusual homes.

### Recorded-disposition calibration

The Florida duration model uses only acquisition price, acquisition year,
acquisition quarter, operator, and week. It omits listing strategy, renovations,
market conditions, and property quality. A model can rank risk rows while still
producing poorly calibrated probabilities. Brier skill against the constant
training hazard must be read alongside AUC and mean predicted versus observed
hazard.

Operator indicators are associational. They combine selection, geography,
timing, operational practice, and measurement.

## Controlled-experiment risks

### External validity

The controlled generator is location-neutral and independent of the Florida
records. Randomized price variation identifies the generator's known response,
not the response of real sellers or buyers. An independently seeded and shifted
evaluation sample is a useful software test, not external validation in a real
market.

### Valuation scenarios

The lower conformal endpoint, point estimate, and upper endpoint are assigned
heuristic scenario weights. Conformal coverage does not identify the probability
mass at those three values. The resulting decision distribution should be read
as a stress construction, not a calibrated forecast distribution.

### No belief updating

The dynamic program reuses scenario weights at every state. A no-sale outcome
does not increase the inferred probability of weak demand or low value, and a
latent scenario is not preserved along a path. This independent remix can
understate or misstate serial inventory risk.

### Static-price experimental spells

Each controlled listing receives one randomized list-price premium that remains
fixed for its generated spell. The fitted common price-response slope is then
used to score adaptive weekly markdown paths in the decision laboratory. The
experiment therefore omits price-history, markdown-announcement, and carryover
effects. It validates the code against a known contemporaneous response, not the
full behavioral response to a sequence of changing prices.

### Recursive risk objective

The optimizer applies mean profit minus average positive loss in the worst
decile at each state. Reoptimization makes this a recursive downside rule. It is
not equivalent to committing at acquisition to one global static mean-CVaR
objective, and it should not be described as such.

### Cost and terminal assumptions

Repair, acquisition, transaction, weekly holding, negotiation, and terminal
liquidation parameters are scenarios. They are not estimated from Florida
operations. Different assumptions can change both the offer and markdown path.

### Missing operational dimensions

The model covers one home. It omits portfolio capital, correlated market risk,
geographic concentration, inventory capacity, renovation queues, staffing,
strategic interactions, and learning from incoming demand signals.

## What real validation would require

- Quote-level offers and seller responses with defensible treatment variation.
- Listing histories with action timestamps, exposure, buyer demand, and proper
  censoring.
- Property-level repairs, concessions, financing, taxes, and operating costs.
- A calibrated joint value and demand-state model with explicit belief updates.
- Prospective shadow-mode evaluation, prespecified guardrails, and operational
  baselines.
- Multi-market and regime-shift evaluation before economic-impact claims.

Until then, Offer-to-Exit is an auditable research prototype, not a production
pricing system.
