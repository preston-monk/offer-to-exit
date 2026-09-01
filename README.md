# Offer-to-Exit

**A transparent decision system for pricing a home from acquisition through resale.**

[![CI](https://github.com/preston-monk/offer-to-exit/actions/workflows/ci.yml/badge.svg)](https://github.com/preston-monk/offer-to-exit/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](https://www.python.org/)

Offer-to-Exit answers one decision question:

> What is the highest acquisition offer worth making today when seller
> acceptance, resale value, time to sale, operating costs, and downside risk are
> uncertain?

The output is not just a value estimate. The system either recommends an offer,
requests human review when the inputs are invalid or poorly supported, or
declines to price when no supported offer clears the economic and risk
constraints. Conditional on acquisition, it also returns a constrained weekly
resale-pricing policy over a 17-week horizon.

> [!IMPORTANT]
> **Evidence boundary:** the public-data pipeline is real; the released
> experiment and every reported model or policy metric are semi-synthetic.
> Public records do not reveal the randomized acquisition offers and resale-price
> interventions needed to identify seller and buyer response. The release is
> therefore evidence about implementation, recovery of known simulated
> responses, and decision behavior—not evidence about any real operator or
> housing market.

## The decision in two minutes

An acquisition offer creates value only through the resale policy that follows
acceptance. Offer-to-Exit therefore connects four uncertain objects:

1. a distribution of resale value at exit;
2. seller acceptance under each candidate acquisition offer;
3. weekly sale probability under each candidate resale list price; and
4. closing proceeds, repair cost, transaction cost, and cost of holding inventory.

For acquisition offer \(a\) and resale policy \(\pi\), the system searches for

\[
\max_{a,\pi}\; \mathbb{E}[\Pi(a,\pi)]
- \lambda\,\operatorname{CVaR}_{0.95}[-\Pi(a,\pi)].
\]

The continuation value of \(\pi\)—the risk-adjusted value of the best feasible
resale path after acquisition—links the two decisions. CVaR is the average loss
in the worst 5% of simulated outcomes; \(\lambda\) expresses a risk preference,
not a statistical truth.

```mermaid
flowchart LR
    A[As-of property and market data] --> B[Resale-value distribution]
    A --> C[Seller-acceptance model]
    O[Candidate acquisition offer] --> C
    A --> D[Weekly sale-hazard model]
    P[Candidate resale list price] --> D
    E[Conditional proceeds and costs] --> F[Finite-horizon resale optimizer]
    B --> F
    D --> F
    F --> G[Exit continuation value]
    C --> H[Acquisition-offer optimizer]
    G --> H
    H --> I[Recommend offer, request review, or decline to price]
```

### One traceable decision

The worked cases are hand-constructed diagnostic scenarios, separate from the
480-home model-evaluation holdout. They test whether the decision layer behaves
sensibly under routine, high-carry, and weak-evidence conditions.

| Scenario | System action | Best supported offer | Acceptance | Expected profit / lead | Loss probability | First resale action |
|---|---|---:|---:|---:|---:|---|
| Healthy demand | **Recommend offer** | `$348,000` | `24.6%` | `$4,829` | `0.6%` | Hold |
| Stale inventory | **Decline to price** | `$510,000` diagnostic only | `23.1%` | `-$6,199` | `14.1%` | Cut `5%` if acquired |
| Thin comparables | **Decline to price** | `$266,500` diagnostic only | `17.8%` | `$852` | `3.6%` | Cut `2.5%` if acquired |

The thin-comparables case is intentionally instructive: a positive mean profit
does not imply an automated offer when uncertainty and tail loss make the
risk-adjusted objective non-positive. Inspect the full paths in the
[self-contained decision explorer](https://preston-monk.github.io/offer-to-exit/artifacts/release/demo.html).

## What is observed, generated, and identified?

| Layer | Release treatment | Supported interpretation |
|---|---|---|
| Public property and market pipeline | Six Phoenix/Maricopa sources with provenance, privacy filtering, and untracked raw data | Data engineering and descriptive market context |
| Released model experiment | Phoenix-shaped generated homes in independent training and evaluation environments | Predictive performance under the documented covariate shift |
| Seller acceptance | Randomized offer-to-value ratios with heterogeneous simulated preferences | Causal-response recovery inside the simulator |
| Weekly buyer demand | Randomized list-price premiums with right-censored sale outcomes | Causal-response recovery inside the simulator |
| Proceeds and costs | Transparent scenario distributions | Sensitivity and decision diagnostics |
| Policy outcomes | Three hand-constructed worked cases | Software and economic-behavior checks, not population lift |

The evaluation environment is one independently seeded covariate-shift check,
not broad robustness evidence. Relative to training, mean market heat moves from
`0.20` to `-0.18`, mean mortgage rates from `5.5%` to `6.9%`, and mean annual
appreciation from `3.0%` to `-0.5%`; value, square-footage, and submarket mixes
also shift.

## v0.1 evidence

The fitted components are evaluated on 480 generated homes held outside model
fitting. The weekly hazard evaluation retains 213 listings and 1,389
right-censored person-period observations.

| Component | Transparent baseline | v0.1 result | Interpretation |
|---|---:|---:|---|
| Resale valuation | Pooled-median MAE `$123,567` | Mean absolute error `$43,536`; median absolute error `$32,777` | `64.8%` lower mean absolute error under the shift |
| 90% value interval | None | `90.0%` coverage | Marginal coverage is calibrated; median width is `35.7%` of the prediction |
| Seller acceptance | Pooled-rate Brier `0.2469` | Brier `0.1965` | Probability accuracy inside the simulator |
| Weekly sale hazard | Period-only log loss `0.4025` | Log loss `0.3922` | Price-aware censored hazard improves on the period baseline |
| Offer response | Truth `+1.600` log-odds per +10 points | Fitted `+1.337` | Known simulated response recovered with `0.263` absolute error |
| Resale-price response | Truth `-1.200` log-odds per +10 points | Fitted `-1.177` | Known simulated response recovered with `0.023` absolute error |

The important result is not a spectacular simulated lift. It is that each link
is inspectable: what was known at decision time, what was predicted, what was
simulated, how uncertainty entered the decision, and when the system refused to
act. See the [complete evidence table](reports/results.md) and
[versioned manifest](artifacts/release/run_manifest.v1.json).

## Why these methods?

- **Split-conformal valuation intervals** expose uncertainty to the optimizer
  instead of reducing value to one point estimate.
- **Discrete-time survival modeling** represents weekly sale probability while
  retaining homes that remain unsold at the horizon.
- **Separate sale hazard and proceeds** preserve the distinction between whether
  a home sells and the amount realized conditional on sale.
- **Finite-horizon dynamic programming** is auditable for a small action grid and
  handles hard pricing and cadence constraints directly.
- **Structured response models** make extrapolation and monotonicity easier to
  inspect than a flexible black box entering an optimizer.
- **Review and decline states** reduce automated coverage when data quality,
  support, or risk is inadequate.

## Reproduce the release

The complete released experiment runs locally from a locked environment:

```bash
make install
make reproduce
```

Run the full quality suite—formatting, linting, static types, tests with coverage,
and the deterministic quickstart—with:

```bash
make check
```

The repository also exposes a typed API over the worked cases:

```bash
uv run uvicorn offer_to_exit.serving:app --reload
```

The public-data path is deliberately separate from the released generated
experiment:

```bash
uv run offer-to-exit catalog
uv run offer-to-exit fetch
uv run offer-to-exit prepare
```

Preparation streams allowlisted Maricopa fields, hashes parcel identifiers, and
fails closed on owner- or address-like columns. Raw and full processed data stay
outside version control.

## Current scope

- Phoenix/Maricopa-shaped single-family homes;
- one acquisition decision and weekly resale decisions for 17 weeks;
- hold or reduce list price by 1%, 2.5%, or 5%;
- calibrated linear valuation, regularized logistic seller acceptance, and a
  discrete-time logistic sale hazard;
- transparent proceeds and cost scenarios rather than a learned proceeds model;
- one independently seeded covariate-shifted evaluation environment; and
- three diagnostic decision cases with executable economic invariants.

The release does not report real-market elasticity or economic lift, repeated
policy backtests, bootstrap uncertainty, geography-slice results, stress-suite
results, oracle regret, portfolio allocation, or production readiness.

## Repository map

```text
configs/              Executed quickstart and release configurations
data/                 Public-source contract and untracked local data zones
docs/                  Methodology for the linked decision system
src/offer_to_exit/    Simulation, modeling, optimization, evaluation, and serving
tests/                 Temporal, privacy, decision, and integration checks
reports/               Data/model/policy cards, evidence, and limitations
artifacts/release/     Deterministic tables, figures, manifest, and static explorer
```

## Go deeper

- [Understand the method](docs/METHODS.md)
- [Inspect the evidence](reports/results.md)
- [Review assumptions and failure modes](reports/limitations.md)
- [Read the decision contract](SYSTEM_SPEC.md)

Code is released under the [MIT License](LICENSE). External datasets retain
their original terms and are not relicensed by this repository. Citation metadata
is in [CITATION.cff](CITATION.cff).
