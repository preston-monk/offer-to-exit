# Data Contract

This directory documents the evidence boundary for Offer-to-Exit. Raw and
derived data are local build inputs, not repository artifacts.

## Data zones

```text
data/
├── README.md       This contract
├── sample/         Small generated, de-identified fixtures allowed in Git
├── raw/            Immutable source downloads; ignored
├── external/       Optional licensed reference data; ignored
├── interim/        Validated and joined tables; ignored
├── processed/      Model-ready as-of tables; ignored
└── cache/          Replaceable computation cache; ignored
```

Only `README.md` and deliberately small files in `sample/` may be committed.
Files in the other zones must remain ignored even when their upstream source is
public. Public availability does not automatically grant redistribution rights.

## Evidence layers

| Layer | Examples | Status | Interpretation |
|---|---|---|---|
| Property context | living area, beds/baths, year built, lot size, coarse location | Public or generated equivalent | Predictive/descriptive |
| Transaction context | prior sales, repeat-sale behavior, time-filtered comparable sales | Public or generated equivalent | Predictive/descriptive |
| Market context | lagged price index, mortgage rate, inventory, seasonality | Public aggregate or generated equivalent | Predictive/descriptive |
| Acquisition behavior | offer, seller acceptance, reservation value | Semi-synthetic | Causal only inside the simulator |
| Exit behavior | weekly price action, buyer arrival, sale hazard, negotiated proceeds | Semi-synthetic | Causal only inside the simulator |
| Operations | repair, selling, financing, and holding costs | Transparent schedules plus simulated uncertainty | Scenario assumptions |

The generated quickstart fixture contains no owner names, real addresses, or
exact coordinates. If real property records are added locally, public examples
must replace identifiers with stable random IDs and coarsen geography.

## Units of observation

The pipeline produces two principal analytical tables.

### Acquisition table

One row represents one quote-time decision for one property.

Required concepts include:

- stable parcel group identifier;
- `quote_date` and feature-level `available_at` timestamps;
- structure and coarse geographic attributes;
- lagged market state;
- value target or simulator truth;
- candidate acquisition offers and simulated acceptance outcomes.

### Property-week table

One row represents one property still at risk of sale at the beginning of one
listing week.

Required concepts include:

- week start and weeks on market;
- list price before the weekly action;
- action chosen that week;
- only demand information observed before that action;
- sale/censoring indicator and conditional proceeds when sold.

Rows after sale do not exist. Homes still unsold after week 17 are right-censored;
they are not silently labeled as sales or zero-valued outcomes.

## Temporal contract

Every feature must carry or inherit an availability timestamp. For a quote-time
decision, `available_at <= quote_date`. For a week-\(t\) resale decision, only
information available before that action may be used.

The following are prohibited predictors:

- comparable sales that closed after the decision;
- final days on market, total markdowns, or closing price;
- same-week demand observed after the price action;
- assessor fields revised after the target transaction;
- inspection or repair facts learned only after acquisition at quote time;
- future or revised macroeconomic series unavailable on the decision date.

## Split contract

The real-data design uses chronological train, validation, and test periods
grouped by parcel.

1. No parcel or repeat sale may cross partitions.
2. Model selection and policy tuning use validation only.
3. The newest test outcomes remain untouched until the pipeline is frozen.
4. The dataset ends at least 120 days before the test labeling cutoff so active
   listings are not mislabeled as failures.
5. Simulator training and evaluation use independent seeds and evaluation
   parameters are not visible to the policy.

## Validation gates

- Schema and type checks pass.
- Keys are unique at their stated grain.
- Prices, areas, dates, and rates satisfy plausible configured bounds.
- No future feature enters a decision row.
- Split membership is disjoint at the parcel level.
- Missingness is measured by period, geography, and price tier.
- Duplicate/near-duplicate records are either resolved or reported.
- Sample fixtures contain no direct identifiers.

## Provenance manifest

Every non-generated source must record:

- source organization and direct URL;
- dataset title and version or retrieval date;
- license/terms and redistribution decision;
- original filename and cryptographic checksum;
- transformations required before use;
- known coverage breaks or restatements;
- earliest date the field was actually available.

The run manifest records the exact input manifest, configuration, random seeds,
package version, and code revision. A result without that provenance is not a
release result.

## Responsible use

County property data can include information that is public yet unnecessary for
this decision. Owner names, contact information, exact addresses in examples, and
other direct identifiers are excluded. Geography can still proxy for protected
characteristics, so error and abstention behavior must be inspected across coarse
areas and data-density strata. This project is not a lending, insurance, tenant,
or consumer-eligibility system.
