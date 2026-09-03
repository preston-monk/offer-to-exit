# Data Card

## Summary

Offer-to-Exit has two independent data layers.

| Layer | Population | Purpose |
|---|---|---|
| Florida transactions | Hillsborough County sales, Orange County sales, and named iBuyer ownership spells | Repeat-sale prediction, geographic evaluation, and recorded inventory duration |
| Controlled experiment | Location-neutral generated homes, offers, and listing weeks | Known-response recovery and fitted-model decision diagnostics |

The generated layer is not drawn from, calibrated to, or seeded by the Florida
data. Keeping these layers separate prevents simulated behavioral responses from
being read as estimates for Florida households or buyers.

## Florida sources

| Source | Retrieval | Fields used |
|---|---|---|
| [Hillsborough County Property Appraiser All Sales](https://downloads.hcpafl.org/) | Current bulk archive discovered through the publisher's download form | Parcel, sale date and amount, qualification, improvement, property use, instrument, grantor, grantee |
| [Orange County Property Appraiser All Sales](https://vgispublic.ocpafl.org/server/rest/services/Webmap/SALES/MapServer/5) | ArcGIS query ordered by `OBJECTID`, 1,000 records per page, no geometry | Parcel, sale date and amount, sale description, deed and property-use codes, selected property attributes, grantor, grantee |

These are mutable public sources. The local raw-data manifest records retrieval
metadata and file hashes. The public release manifest records hashes for the
local direct-identifier-reduced inputs and aggregate artifacts.

## Privacy treatment

County files may contain direct identifiers. The preparation code:

1. reads party names transiently;
2. assigns a canonical operator label only for anchored Opendoor, Offerpad,
   Zillow Homes, or RedfinNow name patterns;
3. hashes parcel numbers with a market namespace; and
4. omits names, addresses, geometry, document IDs, and raw parcel numbers from
   the analytical outputs.

Raw and processed transaction tables are Git-ignored. The repository publishes
aggregate counts, metrics, figures, and checksums rather than transaction rows.
The operator matcher favors precision. Unrecognized affiliates can therefore be
missed.

## Florida transaction table

One row is a recorded transfer. The common fields include market, parcel hash,
sale date and price, qualification, property-use and improvement indicators,
instrument type, optional property attributes, and classified buyer or seller
operator. Property-use codes are stored as strings so leading zeros survive the
county join.

Exact duplicates on market, parcel, date, and price are removed before modeling.
The current pipeline does not claim to identify every related-party,
multi-parcel, or administrative deed.

## Repeat-sale analytical sample

The target is recorded price among improved single-family repeat sales with a
prior eligible sale no more than 4.5 years earlier. This high-turnover target is
used because the current Orange service begins in 2022 and because the two
county sources do not share a sufficiently rich historical structural-feature
set for a symmetric cross-sectional model.

The initial screen requires:

- sale date on or after January 1, 2010;
- sale price from \$25,000 through \$10 million;
- an improved-property indicator; and
- a single-family DOR or property-type code.

Hillsborough sales must be county-qualified. Orange uses
`SALE_DESCRIPTION` when present. That field is missing in the retrieved 2022
through 2024 history, so `DEED_CODE == "WD"` is used as a warranty-deed proxy
when qualification is unknown. A warranty deed is not proof of an arm's-length
sale. Residual non-market transfers are a material Orange measurement risk.

The released sample funnel and prior-sale gap summaries are in
[`florida_metrics.v2.json`](../artifacts/release/florida_metrics.v2.json).
The prior eligible sale must be in a strictly earlier calendar quarter than the target
sale, and the release includes a 30-day minimum-gap sensitivity.

## Valuation split and baseline

Hillsborough targets are split by sale date:

- before 2022: proper training;
- 2022 and 2023: conformal calibration; and
- 2024 onward: out-of-time test.

Parcels in a later target period are purged from earlier target periods. Orange
sales from 2024 onward are scored without fitting or interval calibration on
Orange outcomes. This is an external-market evaluation, not a preregistered
untouched holdout.

The baseline rolls the home's prior eligible price forward by the change from the county
median in its prior-sale quarter to the lagged county median before the current
sale. It observes the same own-price history and market movement as the fitted
repeat-sale model.

## Named iBuyer episodes

An acquisition is a deed on which a classified operator is grantee and recorded
consideration is at least \$10,000. A completed episode ends at the first later
deed for the same parcel on which that operator is grantor and consideration
meets the same threshold. The screen excludes nominal or administrative
transfers that do not reveal interpretable purchase or resale prices. Open episodes are right-censored at each county's latest valid
transaction date at or before that source's retrieval timestamp.

The linker records four statuses: `completed`, `right_censored`,
`administrative_horizon`, and `repeat_acquisition_before_exit`. The modeling
panel retains the first three. A potential link longer than 1,095 days is
assigned the administrative-horizon status rather than treated as an observed
exit; because that episode has a fully observed non-exit through the 52-week
model horizon, it enters as censored at week 52. Repeat-acquisition-before-exit
episodes remain excluded because their spell identity is ambiguous.

The duration study uses improved single-family acquisitions from 2016 onward
and requires at least 25 observations for an operator in each county. Opendoor
and Offerpad satisfy that common-support rule in the current release. The model
uses a 52-week risk window, with later dispositions treated as censored at week
52. The common descriptive comparison covers acquisitions from January 2022
onward.

Observed linked exits use the ceiling of deed-to-deed days divided by seven.
Censored spells use completed weeks only;
those with less than seven days of follow-up do not enter the weekly panel or
the model-eligible Kaplan-Meier summary. Every pre-2022 Tampa training spell is
administratively censored at January 1, 2022 before weekly expansion.

Qualification is not required for the duration sample because the outcome is
recorded title duration. The interval from acquisition deed to disposition deed
is not MLS days on market.

## Controlled experiment

The controlled generator creates a location-neutral housing population. It
constructs observable appraisal-like pre-offer and pre-listing references,
randomizes acquisition offer ratios relative to the first and list-price premia
relative to the second, generates responses from known log-odds equations, and
retains right-censored listing histories. The controlled hazard equation
contains no latent-demand term; random event draws provide outcome noise.

Training and evaluation use independent random seeds. The evaluation environment
changes synthetic home values, square footage, market heat, mortgage rates,
appreciation, and submarket shares. Known response coefficients are excluded
from fitting and used only to score response recovery.

The causal interpretation is internal to the generated experiment. It does not
describe real seller acceptance, buyer demand, or economic impact.

## Intended uses

- Evaluate a Tampa-trained repeat-sale predictor over time and in Orlando.
- Describe and predict deed-to-deed ownership duration for named iBuyers.
- Inspect known response recovery in a controlled price experiment.
- Exercise a linked acquisition and resale decision rule under declared costs
  and uncertainty scenarios.

## Excluded uses

- Real offers, appraisals, transactions, lending, insurance, taxation, or
  consumer eligibility.
- Causal comparisons of operators.
- Claims that deed-to-deed duration is listing duration.
- Claims that a recorded price difference is net profit.
- Claims that generated price responses apply to Florida or any real market.

The complete construction and estimands are documented in
[`docs/DATA_DESIGN.md`](../docs/DATA_DESIGN.md).
