# Decision-Policy Card

## Decision object

Offer-to-Exit chooses an acquisition offer and, conditional on acceptance, a
17-week resale-price policy for one home. The acquisition bid depends on the
continuation value of the best exit policy. Seller acceptance, resale liquidity,
carrying cost, and downside risk therefore enter one decision rather than four
unrelated scores.

## Contribution profit

For sale price $P$, acquisition offer $a$, sale week $t$, pre-listing holding
weeks $w_0$, repair cost $R$, acquisition cost $K$, fixed transaction cost $F$,
transaction rate $c$, and weekly holding cost $H$, the code computes

```math
\Pi(P,a,t)
=
P-a-R-K-F-cP-(w_0+t+1)H.
```

These are transparent scenario inputs. They are not estimated from county deeds
and they are not a complete operator profit-and-loss statement. The calculation
omits corporate overhead, capital opportunity cost, renovation queues,
concentration risk, and other portfolio constraints.

## Downside statistic

For discrete profit distribution $\Pi$, quantile function $Q_\Pi(u)$, and tail
mass $\alpha$, define

```math
D_\alpha(\Pi)
=
\frac{1}{\alpha}\int_0^\alpha
\max\{-Q_\Pi(u),0\}\,du.
```

The released cases use $\alpha=0.10$. Thus $D_{0.10}$ is the average positive
loss in the worst decile, with profitable outcomes inside that tail contributing
zero loss. The objective at a state is

```math
U(\Pi\mid s_t)
=
\mathbb E[\Pi\mid s_t]-\lambda D_{0.10}(\Pi\mid s_t).
```

The case-specific parameter $\lambda$ is 0.20, 0.45, or 0.90. This statistic is
CVaR-style, but the policy is not one global ex-ante mean-CVaR commitment. The
objective is recomputed recursively at every unsold-inventory state.

## Resale policy

### State

The implemented state contains:

- zero-indexed resale week;
- current list price; and
- an optional demand-state tuple, empty in the released cases.

The home context contains a fitted contemporaneous value anchor, an observable
pre-listing price reference, and case covariates. The current state does not
include a persistent latent market regime.

### Actions

The feasible actions are:

- hold the current list price;
- reduce it by 1 percent;
- reduce it by 2.5 percent; or
- reduce it by 5 percent.

The policy cannot increase price. A markdown that would take price below 70
percent of the observable pre-listing reference is infeasible. There is no separate cooldown or
maximum-markdown-count constraint.

### Transition

For each post-action price, the fitted discrete-time hazard receives one premium
relative to the observable pre-listing reference and returns the probability of
sale in that week. Every scored premium lies in the randomized support
$[-0.30,0.15]$. The same sale probability is paired with each valuation stress
scenario, which changes conditional proceeds but not the treatment definition.
A sale realizes conditional headline proceeds and the applicable costs. No sale
moves to the next week at the new price.

After the seventeenth weekly decision, which is zero-indexed week 16, unsold
inventory is liquidated at a case-specific fraction of modeled conditional
headline proceeds. In the fitted adapter, those proceeds are the lower of 99.5
percent of posted price and the valuation stress value. The released
liquidation discounts are 0.94, 0.90, and 0.88.

The profit distribution is compressed into at most 200 equal-mass bins as the
tree is propagated backward. Compression preserves the mean while approximating
the tail.

## Scenario interpretation

The fitted controlled valuation supplies a point estimate and a conformal lower
and upper endpoint. The three values are assigned case-specific weights and used
as stress scenarios. A conformal interval is not a probability distribution, so
those weights are heuristic and should not be read as calibrated posterior
probabilities.

The same value-scenario weights are used again at later states. The code does not
condition future weights on the scenario that generated the current transition
and does not update beliefs after failure to sell. This independent scenario
remix is a tractability assumption.

Conditional sale price is the smaller of the scenario value and posted price
after a 0.5 percent negotiation discount. This is a transparent proceeds rule,
not a fitted proceeds model.

## Acquisition policy

For each candidate offer $a$, the fitted acceptance model returns
$q(a\mid x)$. The optimizer combines:

- the accepted branch, weighted by $q(a\mid x)$, with its optimized resale
  profit distribution; and
- the rejected branch, weighted by $1-q(a\mid x)$, with the configured quote
  cost, which is zero in the released cases.

The same mean-minus-worst-decile-loss criterion scores the resulting profit per
lead. The fitted-case offer grids contain five values expressed as fractions of
the observable pre-offer value reference. Every candidate lies inside the
controlled treatment support $[0.82,1.02]$. The fitted acceptance adapter
refuses to extrapolate beyond those bounds. The highest-objective supported grid
point is selected. The public action is `price` when that objective is positive
and `abstain` otherwise.

The core optimizer and the three released worked cases do not apply separate
hard thresholds for margin, sell-through, loss probability, or interval width.
Behavioral overlap is enforced by limiting acquisition bids to the randomized
offer-ratio support. The cases choose the best supported grid point and return
`abstain` when its objective is not positive. The illustrative FastAPI wrapper
preserves the same offer-ratio support and adds operational review gates for
missing support, scenario width, extreme cost inputs, and target margin. Those
API gates are declared product rules, not estimated or calibrated decision
boundaries. The wrapper still returns `abstain` whenever the core risk-adjusted
objective is non-positive. Its
`expected_profit` is measured per quoted lead, including seller rejection.
`probability_of_loss` is the approximate negative-profit mass in the compressed
lead-level outcome grid. `sale_by_120_days` is conditional on acquisition.

## Fitted inputs

The worked cases use fitted controlled-experiment models rather than simulator
truth or hand-coded response curves:

- a fitted linear valuation and split-conformal interval;
- a fitted regularized seller-acceptance logit; and
- a fitted regularized weekly sale-hazard logit.

The three case profiles are selected from the independent generated evaluation
sample using observed covariates and fitted valuation outputs. The healthy case
maximizes a demand-oriented score, the stale case maximizes a weak-demand and
high-rate score after excluding the first profile, and the sparse-support case
favors rare generated submarkets, unusual size, and a wide fitted interval after
excluding the first two. Its lower valuation stress must bind initial sale
proceeds while the point prediction does not.

The fitted contemporaneous value anchor supplies heuristic 17-week proceeds
stress values and scales costs; it is not a learned terminal proceeds forecast.
The pre-listing reference scales list-price states and
is the denominator of the sale-hazard treatment. The separate pre-offer
reference scales the acquisition grid and is the denominator of the acceptance
treatment. Keeping those objects distinct avoids silently replacing either
randomized treatment definition with a model prediction.

## What the tests establish

The automated decision tests check cost reconciliation, action arithmetic,
finite-horizon traversal, response to an expensive-carry fixture, response to a
downside-risk fixture, integration of fitted model adapters, risk-distribution
arithmetic, and execution of all three 17-week cases. These are implementation
tests, not evidence that the policy improves real-market outcomes.

## Current exclusions

The release does not implement portfolio allocation, correlated property risk,
capital constraints, geographic concentration, belief updating, endogenous
repair choice, learned proceeds, online learning, reinforcement learning, or a
population policy experiment against operational baselines.
