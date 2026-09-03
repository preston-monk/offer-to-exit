# Release Evidence

This page is an index to the generated evidence for Offer-to-Exit 0.2.0. It does
not copy headline metrics because the public county sources can change and the
artifacts are regenerated from the current retrieval.

## Florida transaction study

The Florida workflow downloads and sanitizes Hillsborough and Orange County
sales, builds repeat-sale and named-iBuyer episode panels, fits the Tampa models,
and scores the common future samples.

The primary outputs are:

- [Florida evidence report](../artifacts/release/florida_evidence.html)
- [Florida metrics](../artifacts/release/florida_metrics.v2.json)
- [operator duration summary](../artifacts/release/florida_operator_summary.v2.csv)
- [operator model effects](../artifacts/release/florida_operator_effects.v2.csv)
- [valuation transport figure](../artifacts/release/florida_valuation_transport.v2.png)
- [inventory duration figure](../artifacts/release/florida_inventory_duration.v2.png)
- [Florida artifact manifest](../artifacts/release/florida_manifest.v2.json)

### How to read the valuation evidence

The relevant comparison is the fitted repeat-sale model against the
rolled-forward prior-sale baseline. Both observe the home's own prior eligible price and
county market movement. Metrics are parcel weighted, and the table reports both
transactions and unique parcels.

Tampa 2024-forward results are out of time. Orlando 2024-forward results use the
Tampa-fitted model and Tampa conformal radius without Orange fitting or
calibration. The Orlando exercise is an external-market evaluation, not a
preregistered untouched holdout.

The target population is high-turnover repeat sales with a prior eligible sale no more
than 4.5 years earlier and in a strictly earlier calendar quarter. A 30-day
minimum-gap sensitivity is reported separately. Orange's missing historical
qualification description is handled with a disclosed warranty-deed proxy.
Results do not apply to every home or every sale in either county.

### How to read the duration evidence

The operator summary reports Kaplan-Meier title-exit quantities for the common
2022-forward acquisition window. It retains open right-censored spells. The
interval is acquisition deed to disposition deed, not MLS days on market.

The hazard metrics should be read jointly. AUC measures ranking, while
person-period Brier score, constant-hazard Brier score, Brier skill, and
observed versus predicted hazard reveal probability calibration. Positive
discrimination does not excuse negative Brier skill.

Operator coefficient rows are regularized conditional associations. They are
not causal comparisons of Opendoor and Offerpad.

## Controlled decision experiment

The location-neutral controlled workflow generates independent training and
shifted evaluation environments, fits valuation, seller-acceptance, and weekly
sale-hazard models, and passes those fitted models into three 17-week decisions.

The outputs are:

- [controlled decision laboratory](../artifacts/release/demo.html)
- [controlled metrics](../artifacts/release/metrics.v2.json)
- [worked decisions](../artifacts/release/decision_cases.v2.csv)
- [controlled run summary](../artifacts/release/summary.v2.json)
- [controlled artifact manifest](../artifacts/release/run_manifest.v2.json)
- [valuation figure](../artifacts/release/valuation_evaluation.v2.png)
- [seller-response figure](../artifacts/release/acceptance_response.v2.png)
- [sale-hazard figure](../artifacts/release/hazard_sell_through.v2.png)

The `v2` suffix on these filenames is the controlled artifact-schema version,
not the software release number.

### How to read the controlled evidence

Seller offer ratios relative to an observable pre-offer value reference and
list-price premia relative to an observable pre-listing reference
are randomized in the generator, so the fitted log-odds responses can be
compared with known coefficients. Candidate offers and list prices are scored
against those same respective denominators and remain within randomized support.
The controlled hazard has no omitted latent-demand term. This comparison
validates response recovery inside the controlled environment. It does not
estimate a real seller-acceptance or buyer-demand elasticity.

The three case profiles are chosen using observed evaluation covariates and
fitted valuation outputs, then scored by the fitted models. The valuation
interval endpoints are assigned heuristic stress weights. They are not a
calibrated distribution. The decision outputs show what the implemented
objective does under those assumptions; they are not a population policy
backtest or estimate of economic lift.

## Reproduction

Install the locked environment and run the checks:

```bash
uv sync --all-extras
make check
```

Rebuild the Florida inputs and aggregate evidence:

```bash
make florida-data
make florida-study
```

Rebuild the controlled experiment:

```bash
make reproduce
```

Or rebuild both evidence layers:

```bash
make release
```

Raw and processed county transaction files remain local and Git-ignored. The
committed manifests contain hashes that connect each evidence bundle to its
inputs and outputs.

## Results not claimed

The release does not contain a real-market seller response, causal Florida list
price elasticity, causal operator comparison, net operator profit, population
policy lift, oracle regret study, portfolio allocator, or production deployment.
