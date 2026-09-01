# Data Card

## Dataset summary

**Name:** Offer-to-Exit Phoenix/Maricopa v0.1
**Market:** Maricopa County, Arizona
**Property type:** Single-family homes
**Decision cadence:** Quote time plus weekly resale decisions
**Maximum resale horizon:** 17 weeks
**Construction:** Public or generated market context plus semi-synthetic behavior
**Current status:** v0.1 public-data preparation audited; release experiment uses
the generated fixture so behavioral ground truth remains known

## Local public-data audit (2026-09-01)

The reproducible fetch/prepare commands were executed against all six cataloged
sources. Raw downloads and sanitized full tables remain Git-ignored.

| Source | Download | SHA-256 prefix | Prepared Phoenix/Maricopa result |
|---|---:|---|---:|
| Maricopa Residential Master | 58.4 MB | `8b6fdc8c31fe` | 1,416,644 of 1,416,731 rows retained |
| Maricopa Sales Affidavits | 61.4 MB | `4ac71e880880` | 740,561 of 912,807 plausible-sale rows retained |
| Maricopa Parcel Points | 85.7 MB | `e6acc936bcbc` | Downloaded; raw geometry intentionally not materialized in v0.1 |
| Realtor.com metro history | 32.9 MB | `258e84edc705` | 121 Phoenix metro-month rows |
| FHFA HPI master | 17.2 MB | `f7eca3f886f0` | 623 Phoenix-area index rows across published frequencies |
| FRED 30-year mortgage rate | 46.9 KB | `c3cbce71ddda` | 2,892 weekly observations |

The county extracts contain fields that are public but unnecessary for this
decision. Preparation streams only an allowlist, decodes the publisher's legacy
Windows-1252 text, hashes parcel identifiers, rejects owner/address-like column
names, and writes no owner names, grantor/grantee names, street addresses, deed
numbers, or exact parcel geometry. A header audit found none of those fields in
the two sanitized tables.

These real tables establish the ingest, provenance, privacy, and market-context
layer. The v0.1 metric bundle does not mix them into its behavioral claims; its
fitted experiment is explicitly semi-synthetic.

## Why a semi-synthetic dataset is necessary

A public deed or assessor record can describe a property and a completed sale.
It usually cannot reveal:

- the acquisition offers a seller did not accept;
- the seller's unobserved reservation value or convenience preference;
- a randomized set of list prices that could have been chosen;
- buyer arrivals, views, or conversions under counterfactual prices;
- proprietary repair estimates and operating costs.

Treating historical list-price changes as random would create a false causal
story: prices are often cut because latent demand is weak. The released workflow uses a
documented simulator for seller acceptance, weekly buyer demand, negotiated
proceeds, and cost uncertainty. Controlled exploration creates treatment support,
and simulator truth makes response-recovery error measurable.

This design supports a statement such as “the fitted model recovered simulated
price response with this error.” It does **not** support a statement such as
“Phoenix sellers have this real acceptance elasticity.”

## Intended uses

- Train and compare time-aware home-value models.
- Estimate known behavioral response functions inside a simulator.
- Evaluate acquisition and resale policies under controlled uncertainty.
- Test leakage prevention, calibration, abstention, and stress behavior.
- Reproduce a complete pricing-system workflow and its evidence bundle.

## Out-of-scope uses

- Real property offers or automated transactions.
- Lending, insurance, taxation, appraisal, or consumer eligibility.
- Inference about individual homeowners, buyers, or neighborhoods.
- Claims about markets outside Phoenix/Maricopa.
- Claims that simulated seller or buyer behavior describes real people.

## Data-generating process

The environment contains four linked sources of uncertainty:

1. **Exit value.** Property attributes, time, coarse geography, and market state
   determine a latent value distribution.
2. **Seller acceptance.** Acceptance rises monotonically with offer-to-value
   ratio but varies with latent reservation value and convenience preference.
3. **Buyer demand.** Weekly arrival and sale hazard fall as list-price premium
   rises and vary with desirability, seasonality, mortgage rates, inventory, and
   weeks on market.
4. **Economics.** Repairs, selling costs, financing, and holding costs convert
   transactions into contribution profit.

The data generator is intentionally more nonlinear and heterogeneous than the
fitted behavioral models. Training and evaluation environments use different
seeds, and stress scenarios shift selected parameters.

## Records and labels

| Table | Grain | Principal labels |
|---|---|---|
| Acquisition | One property at quote time | Exit value, candidate-offer acceptance |
| Property-week | One property at risk at the start of a listing week | Weekly sale event, conditional sale proceeds |
| Policy episode | One accepted acquisition through sale or censoring | Profit, holding time, markdowns, loss, sell-through |

Unsold homes are right-censored after the configured horizon. Conditional sale
proceeds are defined only for sale events. The model must not impute a completed
sale for a censored listing merely to simplify evaluation.

## Temporal and entity boundaries

- Every feature has an `available_at` timestamp or a documented inherited
  availability rule.
- Quote-time rows cannot contain inspection, repair, listing, or sale information
  learned after acquisition.
- Weekly rows contain only lagged demand observations available before the action.
- Comparable sales and market aggregates are constructed strictly as of the
  decision date.
- Splits are chronological and grouped by parcel so repeat observations of the
  same home cannot cross partitions.
- A 120-day maturity buffer prevents unresolved outcomes from being labeled as
  failures.

## Known sources of bias

- Public transaction data omit off-market activity and may revise attributes.
- Repeat-sale homes are selected and may not represent the full housing stock.
- Recorded structure and condition fields may be stale or missing.
- Coarse geography captures amenities and access but can also proxy for protected
  characteristics.
- Simulator calibration to aggregate market distributions cannot validate
  unobserved individual seller or buyer response.
- A single market cannot establish transportability to another city or regime.

## Privacy and governance

Sample data must contain stable random identifiers, fictional display labels,
and coarse geography. Owner names, contact details, exact addresses, and exact
coordinates are unnecessary and prohibited in committed fixtures. Any locally
downloaded raw records remain ignored.

Protected-class attributes are not model inputs. Their omission does not by
itself eliminate fairness concerns because geography and property characteristics
can act as proxies. A real-data evaluation would need error, uncertainty, and
automation coverage by coarse geography and data-density tier. The v0.1 release
does not report those slice results.

## Versioning and reproducibility

Each run must write a manifest containing source checksums, configuration path,
random seeds, dependency state, package version, and code revision. Release
tables and figures must be traceable to that manifest. The repository does not
version raw data or large fitted model objects.
