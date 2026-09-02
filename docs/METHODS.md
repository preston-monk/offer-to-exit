# Methods

## 1. Decision sequence

The system respects the order in which information becomes available.

```text
quote -> acquisition offer -> seller response -> acquisition -> repair -> list
      -> weekly resale-price decisions -> sale or censoring at week 17
```

At quote time, the state contains only property and market facts available before
the offer. After acquisition, the weekly state adds the current list price,
remaining horizon, accumulated costs, and prior-week demand. Same-week demand is
never used to choose the price that generated it.

The output vocabulary is deliberately operational:

- **recommend an offer** when a supported action clears the economic and risk
  constraints;
- **request human review** when data validity or model support may be resolvable;
  and
- **decline to price** when no supported offer has positive risk-adjusted value.

## 2. Linked acquisition and resale economics

Let $q(a)$ denote the probability that a seller accepts acquisition offer
$a$. Let $J_0^*(a)$ denote the optimized continuation value of resale after
the home is acquired. The acquisition problem is

```math
a^*=\arg\max_{a\in\mathcal A}
q(a)J_0^*(a)-\text{downside penalty},
```

subject to margin, sell-through, loss, and support constraints. Continuation
value is the expected risk-adjusted contribution from the best feasible resale
path, conditional on owning the home. This is the link that prevents the offer
model from treating liquidity, carrying cost, and markdown flexibility as an
afterthought.

For weekly resale state $s_t$ and price action $u_t$, backward recursion uses

```math
Q_t(s_t,u_t)=
h_t(s_t,u_t)\,\mathbb{E}[\text{net proceeds}\mid\text{sale}]
+(1-h_t(s_t,u_t))\left[-H_t+V_{t+1}(s_{t+1})\right],
```

```math
V_t(s_t)=\max_{u_t\in\mathcal U(s_t)}Q_t(s_t,u_t).
```

Here $h_t$ is the probability of selling during week $t$, conditional on
remaining unsold. The feasible actions are hold, cut 1%, cut 2.5%, and cut 5%.
The optimizer carries an explicit terminal value at the 17-week horizon.

## 3. Released model components

### Resale valuation

The baseline is the pooled median observed sale price in the training
environment. The fitted model is a log-linear hedonic regression over structure,
age, condition, quality, lot size, coarse submarket, and market state. A
split-conformal calibration step produces a marginal 90% prediction interval.

The interval does not solve uncertainty. Marginal coverage can conceal
conditional miscalibration, and a distribution shift weakens exchangeability.
The interval is still decision-relevant because equal point estimates with very
different uncertainty should not automatically support equal offers.

### Seller acceptance

The baseline is the pooled acceptance rate. The fitted model is a regularized
logistic response using offer-to-value ratio, a simulated pre-offer seller-timing
signal, market heat, repair-cost fraction, and coarse submarket. The timing signal
represents information collected before an offer is chosen; latent reservation
value is not exposed to the fitted model.

Offer ratios are deliberately varied over a bounded range in the simulator. This
creates treatment support and makes the known log-odds response recoverable.
That response has a causal interpretation only inside the simulator.

### Weekly sale hazard

The baseline is the observed sale rate by listing period without price. The
fitted discrete-time logistic hazard uses list-price premium, weeks on market,
market heat, property condition, mortgage rate, and coarse submarket. Homes that
remain unsold at week 17 are retained as right-censored observations.

A regression on completed days on market would discard censored listings and
condition on future information. A price-aware hazard instead matches the weekly
decision and distinguishes the probability of selling from proceeds conditional
on sale.

### Conditional proceeds and costs

The current release does not fit a proceeds model. The worked decisions use
explicit downside, base, and upside proceeds scenarios with transparent repair,
closing, financing, and holding-cost distributions. This keeps an unmeasured
component visible rather than presenting an unsupported fitted result.

## 4. Why structured models and dynamic programming?

Behavioral response enters an optimizer, so monotonicity, support, and
extrapolation matter more than a small average gain from an opaque model. The
regularized response models are intentionally simpler than the simulator that
generated their training data.

The resale horizon and action grid are also small. Backward recursion is
reproducible, auditable, and compatible with hard action constraints. A
reinforcement-learning layer would add off-policy evaluation and stability
problems without addressing a current requirement.

## 5. Evaluation design

The released experiment creates 720 training homes and 480 evaluation homes with
different seeds. The evaluation environment is intentionally harder:

| Quantity | Training mean or setting | Evaluation mean or setting |
|---|---:|---:|
| Market heat | `0.20` | `-0.18` |
| Mortgage rate | `5.5%` | `6.9%` |
| Annual appreciation | `3.0%` | `-0.5%` |
| Log value | Reference | `+0.035` shift |
| Log square footage | Reference | `+0.045` shift |
| Submarket mix | Training distribution | Reweighted toward zones D–F |

This is one independently seeded covariate-shifted holdout. It supports an
initial shift check, not a confidence interval over environments.

Evaluation is layered:

1. **Prediction:** value error, interval coverage and width, Brier score, and
   right-censored hazard log loss.
2. **Response recovery:** fitted offer and resale-price log-odds effects against
   known simulator truth, plus monotonicity.
3. **Decision diagnostics:** three separately constructed worked cases exposing
   offer, acceptance, resale path, expected contribution, loss probability, and
   abstention behavior.
4. **Engineering controls:** deterministic artifacts, temporal contracts,
   optimizer invariants, API validation, cost reconciliation, and privacy gates.

The worked decisions are not sampled policy outcomes for the 480-home holdout.
They are diagnostic scenarios designed to make decision behavior traceable.

## 6. Claim boundary

The release supports claims about reproducibility, temporal controls, fitted
performance on the documented generated holdout, recovery of known simulated
responses, and the behavior of three diagnostic decisions. It does not support
real-market elasticity, real economic lift, repeated-policy superiority,
cross-market transportability, or production readiness.
