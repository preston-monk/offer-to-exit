# Florida Data Design

## Why the evidence is split

Offer-to-Exit studies a balance-sheet intermediary that buys a home, holds it as
inventory, and later resells it. No public deed file identifies every object in
that decision. The release therefore separates two kinds of evidence.

| Evidence layer | Unit | Supported interpretation |
|---|---|---|
| Florida housing market | Recorded transfer and named-operator ownership spell | Predictive evidence for recorded repeat-sale prices; descriptive and predictive evidence for deed-to-deed inventory duration |
| Controlled decision experiment | Generated home, offer, and property-week | Recovery of known price responses and behavior of the linked acquisition and resale optimizer inside the generated environment |

The Florida records do not show rejected offers, unchosen acquisition bids,
weekly list-price paths, buyer arrivals, repairs, operating costs, or potential
outcomes at prices that were not posted. The generated experiment is
location-neutral and is not calibrated from the Florida sample.

## Official Florida sources

The empirical pipeline uses two sources and no private data.

### Hillsborough County

The [Hillsborough County Property Appraiser All Sales
archive](https://downloads.hcpafl.org/) is the Tampa-area source. The downloader
discovers the current file through the publisher's web form and stores the
archive locally. Its sales table supplies parcel identifiers, sale dates and
amounts, qualification and improvement flags, property-use and instrument
codes, and grantor and grantee names.

### Orange County

The [Orange County Property Appraiser All Sales ArcGIS
layer](https://vgispublic.ocpafl.org/server/rest/services/Webmap/SALES/MapServer/5)
is the Orlando-area source. The downloader requests a fixed allowlist of fields,
orders on `OBJECTID`, pages in blocks of 1,000, records progress for restart,
and requests no geometry. The layer supplies sales, deed and property-use
fields, limited property attributes, and grantor and grantee names.

The raw files are mutable publisher snapshots. A local fetch manifest records
the source URL, retrieval time, byte count, and SHA-256 hash. The committed
Florida release manifest records hashes for the local direct-identifier-reduced input tables and
the aggregate output artifacts. It also records the exact per-market observation
end used to right-censor open ownership spells.

## Privacy boundary

Raw county records remain outside version control. During preparation, the code
uses party names only to classify a narrow set of named iBuyers. It then drops
the names and replaces each raw parcel number with a 20-character (80-bit)
truncated SHA-256 digest that includes the county namespace. The combined analytical table contains no names,
addresses, coordinates, document numbers, or raw parcel identifiers.

The operator classifier uses anchored legal-name roots for Opendoor, Offerpad,
Zillow Homes, and RedfinNow. It is deliberately not a fuzzy search for generic
investors. This improves precision but can miss affiliates whose names do not
match the coded patterns.

## Canonical transaction table

One row is one recorded county transfer. The shared table contains:

- namespaced parcel hash and market;
- sale date, recorded consideration, qualification, improvement, property-use,
  and instrument fields;
- county fields such as living area or neighborhood when available; and
- canonical buyer and seller operator labels when the narrow classifier matches.

Exact duplicates on market, parcel, sale date, and price are removed before the
valuation panel is formed. The Florida release does not claim to resolve every
multi-parcel, related-party, or same-day deed ambiguity.

## Repeat-sale valuation population

The common county feature set is too sparse for a symmetric cross-sectional
hedonic model. The implemented target is therefore deliberately narrower:

> Recorded sale price for an improved, single-family, high-turnover repeat sale
> with a usable prior eligible sale no more than 4.5 years earlier.

The transaction screen requires a date on or after January 1, 2010, a recorded
price from \$25,000 through \$10 million, an improved-property flag, and a
single-family use code. It then applies a market-specific transfer screen:

- Hillsborough observations must carry the county's qualified-sale flag.
- Orange observations use `SALE_DESCRIPTION` when that field is populated.
  The retrieved layer does not populate that description for its 2022 through
  2024 history. For those unknown observations, a warranty-deed code (`WD`) is
  used as a transparent proxy.

A warranty deed is not equivalent to a county-qualified arm's-length sale. The
Orange screen can therefore retain related-party or otherwise non-market
transfers, and the external result must be interpreted with that measurement
risk in view.

Requiring an observed prior eligible sale and imposing the 4.5-year limit aligns the
observable resale history available in Orange with a comparable high-turnover
population in Hillsborough. It does not represent the full stock of homes or all
transactions in either county.

The prior eligible sale must also fall in a strictly earlier calendar quarter than the
target. The median for the prior eligible sale's quarter is then fully known before the
target quarter starts. The released sensitivity adds a minimum 30-day gap to
test the influence of unusually rapid cross-quarter resales.

## Valuation variables and split

Let $P_{imq}$ be the current sale price for parcel $i$ in market $m$ and quarter
$q$. Let $P_{imq_i^-}$ be its prior eligible sale price, $M_{mq_i^-}$ the county
median in the prior eligible sale's quarter, and $M_{m,q-1}$ the median in the quarter
before the current sale. The implemented model uses

```math
x_i=\log\left(\frac{P_{imq_i^-}}{M_{mq_i^-}}\right),
\qquad
y_i=\log\left(\frac{P_{imq}}{M_{m,q-1}}\right).
```

The other predictors are elapsed years since the prior eligible sale and current calendar
quarter. Sale price is recovered by multiplying the predicted normalized value
by $M_{m,q-1}$.

Hillsborough is partitioned chronologically:

- proper training: before January 1, 2022;
- conformal calibration: January 1, 2022 through December 31, 2023; and
- out-of-time test: January 1, 2024 onward.

If a parcel appears in a later partition, its earlier target row is purged from
the earlier partition. The prior eligible sale remains an ex-ante feature, but the same
parcel's earlier outcome is not also used as a model-fitting row.

Orange observations from January 1, 2024 onward are scored with the Tampa-fitted
model and Tampa calibration radius. Orange outcomes are excluded from fitting
and interval calibration. This is an external-market evaluation, not a
preregistered untouched holdout. Design choices were developed while the public
Orange source and its results were observable.

## Fair repeat-sale baseline

The comparison carries the home's prior eligible sale forward by the change in its own
county median:

```math
\widehat P_i^{\mathrm{base}}
=
P_{imq_i^-}\frac{M_{m,q-1}}{M_{mq_i^-}}.
```

Both the model and the baseline therefore observe the home's prior price and the
same county-level market movement. The baseline is not a pooled median or a
lagged county median alone.

All reported valuation errors weight parcels equally. When a parcel contributes
multiple scored transactions, its total metric weight remains one parcel's
share. The 90 percent split-conformal radius uses one score per calibration
parcel, equal to that parcel's maximum absolute residual on the normalized log
scale.

## Named iBuyer ownership spells

An episode begins when a classified operator is the grantee on a deed with at
least \$10,000 in recorded consideration. A completed episode ends at the first
later deed for the same parcel on which the same operator is the grantor and
consideration meets the same threshold.
A resale deed is consumed at most once. If another
acquisition by that operator appears before a recorded exit, the earlier spell
is flagged rather than treated as completed.

Open spells are right-censored at each county's latest valid transaction date at
or before that source's retrieval timestamp. A potential link more than 1,095
days after acquisition is assigned an administrative-horizon status. The
modeling panel keeps `completed`, `right_censored`, and
`administrative_horizon` episodes; the last group is observed without an exit
through the entire 52-week model horizon. It excludes only
repeat-acquisition-before-exit statuses.
Observed linked exits enter the ceiling week.
Censored spells contribute only complete weeks; those observed for fewer than
seven days are excluded from the weekly risk panel and model-eligible
Kaplan-Meier table. Pre-2022 Tampa training spells are administratively censored
at January 1, 2022 before weekly expansion, so post-boundary disposition labels
cannot enter training.

The current duration study then requires:

- acquisition on or after January 1, 2016;
- an improved single-family property;
- an operator with at least 25 episodes in each county; and
- a maximum modeled risk window of 52 weeks.

The supported operators in the current release are Opendoor and Offerpad.
Qualification is not required because the outcome is title duration rather than
an arm's-length price. A completed disposition after week 52 is treated as
censored at week 52 for this model.

The observed duration is the number of days between recorded acquisition and
resale deeds. It is not MLS days on market. Public descriptive comparisons use
the common January 2022 onward acquisition window in both counties and use
Kaplan-Meier estimates to retain right-censored spells.

## Controlled decision experiment

The controlled experiment generates a separate location-neutral population.
Offer-to-value ratios and list-price premia are randomized by construction.
Training and evaluation use independent seeds, and the evaluation environment
shifts the distributions of values, square footage, market heat, mortgage rates,
appreciation, and synthetic submarkets.

The experiment permits comparison of fitted offer and list-price log-odds
effects with known generator coefficients. That is an implementation and
response-recovery test. It does not estimate how Florida sellers or buyers react
to price.

The three worked decisions use observable profiles selected from the generated
evaluation sample. Simulator truth and realized outcomes are not passed to the
decision adapters. The fitted valuation interval endpoints become stress values
with explicitly chosen scenario weights; those weights are heuristic, not
conformal probabilities.

## Released evidence

The released Florida bundle contains only aggregate outputs:

- [`florida_metrics.v2.json`](../artifacts/release/florida_metrics.v2.json),
  including source counts, sample selection, prior-sale gaps, split counts, and
  model metrics;
- [`florida_operator_summary.v2.csv`](../artifacts/release/florida_operator_summary.v2.csv),
  containing aggregate duration summaries;
- [`florida_operator_effects.v2.csv`](../artifacts/release/florida_operator_effects.v2.csv),
  containing regularized conditional operator associations;
- two aggregate figures and a self-contained
  [Florida evidence report](../artifacts/release/florida_evidence.html); and
- [`florida_manifest.v2.json`](../artifacts/release/florida_manifest.v2.json),
  containing input and artifact checksums.

The controlled bundle has its own summary, metrics, cases, figures, and
checksum manifest. Counts and performance claims should be read from the
generated artifacts rather than copied into this design contract.

## Claim boundary

The release supports conditional predictive statements about the stated
repeat-sale population, external-market scoring between these two samples,
descriptive ownership-duration statements, and response recovery inside the
controlled generator. It does not identify real offer acceptance, causal list
price effects, causal operator effects, net operator profit, statewide or
national transport, or real-world policy lift.
