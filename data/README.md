# Data Contract

The project uses public Florida deeds for predictive and descriptive evidence
and a separate location-neutral generator for controlled decision experiments.
Raw and transaction-level Florida files are local build inputs, not repository
artifacts.

## Local data zones

```text
data/
├── README.md
├── raw/          publisher downloads and local fetch manifest; ignored
└── processed/    direct-identifier-reduced local analytical tables; ignored
```

The code catalog contains exactly two current external sources:

- `hillsborough_sales`: Hillsborough County Property Appraiser All Sales bulk
  archive; and
- `orange_sales`: Orange County Property Appraiser All Sales ArcGIS layer.

No Florida property record is bundled with the package.

## Fetch and prepare

```bash
uv run offer-to-exit fetch --raw-dir data/raw
uv run offer-to-exit prepare \
  --raw-dir data/raw \
  --processed-dir data/processed
```

The fetch step writes `data/raw/manifest.json` with source metadata and SHA-256
hashes. The prepare step writes:

- `hillsborough_transactions_safe.csv.gz`;
- `orange_transactions_safe.csv.gz`;
- `florida_transactions_safe.csv.gz`;
- `named_ibuyer_episodes_safe.csv.gz`; and
- `preparation_manifest.json`.

Publisher files are mutable. Repeating a download later can produce different
rows and therefore different hashes or metrics.

## Privacy rule

Raw county inputs can contain party names and parcel identifiers. Preparation
uses names only to classify four narrow iBuyer brands, hashes parcel IDs with a
market namespace, and drops direct identifiers before writing processed tables.

Processed outputs contain no raw party names, street addresses, coordinates,
document numbers, or raw parcel IDs. They remain ignored because a hashed parcel
history is still row-level property data and does not need to be public.

## Canonical transaction grain

One transaction row is one recorded parcel transfer. The common columns include:

- `parcel_id`: 20-character (80-bit) truncated SHA-256 digest with a market namespace;
- `market`: `tampa_hillsborough` or `orlando_orange`;
- sale date and recorded consideration;
- qualification, improvement, DOR, property-type, and instrument fields;
- optional county property attributes; and
- canonical buyer and seller operator labels.

Property-use fields are read as strings to preserve leading zeros. County dates
are normalized to calendar dates.

## Valuation panel

The Florida release builds the valuation panel from improved single-family
transactions priced between \$25,000 and \$10 million and dated from 2010 onward.
Hillsborough requires its official qualification flag. Orange uses its sale
description when observed and a `WD` warranty-deed proxy when that description
is missing.

Only repeat sales with a prior observation no more than 4.5 years earlier enter
the final target population. The panel derives prior eligible sale price, prior-sale gap,
county-quarter medians, a rolled-forward prior-price baseline, and the normalized
variables used by the repeat-sale estimator.
The prior eligible sale must be in a strictly earlier calendar quarter than the target;
the release also reports a 30-day minimum-gap sensitivity.

Hillsborough target rows are split before 2022, 2022 through 2023, and 2024
onward. Parcels assigned to later targets are purged from earlier target samples.
Orange 2024-forward rows are used only for external-market scoring.

## Operator episode grain

One episode begins when a classified operator is grantee on a deed with at least
\$10,000 in recorded consideration. It ends at the first later deed where the same
operator is grantor and consideration meets the same threshold. This screen
excludes nominal or administrative transfers that do not reveal interpretable
purchase or resale prices. Open episodes are right-censored.
The episode table records acquisition and disposition fields, holding days,
event status, linkage status, property fields from acquisition, and operator.

The four possible linkage statuses are:

- `completed`;
- `right_censored`;
- `administrative_horizon`; and
- `repeat_acquisition_before_exit`.

The modeling panel accepts `completed`, `right_censored`, and
`administrative_horizon`. The last group is retained as censored because its
1,095 observed non-exit days cover the full 52-week model window.
`repeat_acquisition_before_exit` remains excluded because spell identity is
ambiguous. Observed exits use ceiling weeks; any externally supplied zero-day
duration maps to week one. Censored spells use completed weeks and need at least
seven days of follow-up to contribute a risk row or the model-eligible
Kaplan-Meier summary. Pre-2022 Tampa training spells are administratively
censored at January 1, 2022 before weekly expansion.

Recorded duration is deed to deed. It is not listing time. Recorded acquisition
and resale price can be used to compute a gross deed-price difference locally,
but that difference is not profit.

## Controlled generated data

The generated experiment is built in memory by the simulation module rather
than downloaded into `data/`. It creates separate training and evaluation
environments with independent seeds. An appraisal-like
`preoffer_reference_value` is constructed from observed property and market
characteristics plus independent measurement noise. A separate
`prelisting_reference_value` updates that observable signal with observed
appreciation and new measurement noise. Offer ratios are randomized relative to
the pre-offer reference, list-price premia are randomized relative to the
pre-listing reference, and seller acceptance and weekly sale incidence follow
known response equations. The hazard equation has no omitted latent-demand
shock; event-level Bernoulli draws still supply outcome noise.

The generator is location-neutral and does not ingest Florida records. Its
causal interpretation applies only to its own randomized price treatments.

## Public output rule

Only aggregate Florida metrics, operator summaries, figures, HTML evidence, and
manifests are committed. No claim should rely on a local transaction table that
is absent from the generated public evidence.
