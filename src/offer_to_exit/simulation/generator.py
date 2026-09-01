"""Deterministic semi-synthetic data for decision-science experiments.

The generator deliberately makes the two price treatments random:

* ``offer_ratio`` changes seller acceptance with a known log-odds effect.
* ``list_price_premium`` changes the sale hazard with a known log-odds effect.

Everything in this module is synthetic. The values are useful for testing causal,
predictive, and decision code; they are not estimates of any real operator or market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_SUBMARKETS = ("zone_a", "zone_b", "zone_c", "zone_d", "zone_e", "zone_f")
_SUBMARKET_LOG_VALUE = {
    "zone_a": -0.16,
    "zone_b": -0.08,
    "zone_c": 0.00,
    "zone_d": 0.07,
    "zone_e": 0.14,
    "zone_f": 0.22,
}


def _sigmoid(value: np.ndarray | pd.Series | float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


@dataclass(frozen=True)
class CausalParameters:
    """Ground-truth response parameters used by the simulator.

    Coefficients are on a log-odds scale.  Price variables are expressed as
    fractions, so multiplying a price coefficient by ``0.10`` gives the effect of
    a ten-percentage-point price change.
    """

    acceptance_intercept: float = -0.45
    acceptance_offer_ratio: float = 16.0
    acceptance_seller_urgency: float = 0.90
    acceptance_market_heat: float = 0.45
    acceptance_repair_fraction: float = -3.0

    hazard_intercept: float = -1.55
    hazard_log_period: float = 0.38
    hazard_list_price_premium: float = -12.0
    hazard_market_heat: float = 0.70
    hazard_condition: float = 0.12
    hazard_mortgage_rate: float = -8.0
    hazard_latent_demand: float = 0.35

    exit_price_list_premium: float = 0.18

    @property
    def acceptance_log_odds_per_ten_pct(self) -> float:
        return 0.10 * self.acceptance_offer_ratio

    @property
    def hazard_log_odds_per_ten_pct_overpricing(self) -> float:
        return 0.10 * self.hazard_list_price_premium


@dataclass(frozen=True)
class EnvironmentConfig:
    """Distributional settings for one independent synthetic environment."""

    name: str
    seed: int
    n_homes: int = 3_000
    max_listing_periods: int = 12
    market_heat_mean: float = 0.15
    mortgage_rate_mean: float = 0.058
    annual_appreciation_mean: float = 0.025
    log_value_shift: float = 0.0
    log_sqft_shift: float = 0.0
    submarket_probabilities: tuple[float, ...] = (
        0.18,
        0.18,
        0.18,
        0.17,
        0.16,
        0.13,
    )

    def __post_init__(self) -> None:
        if self.n_homes < 1:
            raise ValueError("n_homes must be positive")
        if self.max_listing_periods < 2:
            raise ValueError("max_listing_periods must be at least two")
        if len(self.submarket_probabilities) != len(_SUBMARKETS):
            raise ValueError("submarket_probabilities must have six entries")
        if any(probability < 0 for probability in self.submarket_probabilities):
            raise ValueError("submarket probabilities cannot be negative")
        if not np.isclose(sum(self.submarket_probabilities), 1.0):
            raise ValueError("submarket probabilities must sum to one")


@dataclass
class SimulatedEnvironment:
    """Related tables generated for a single environment."""

    config: EnvironmentConfig
    truth: CausalParameters
    homes: pd.DataFrame
    offers: pd.DataFrame
    listings: pd.DataFrame
    survival_panel: pd.DataFrame


@dataclass
class TrainEvaluationSimulation:
    """Independent training and evaluation environments."""

    train: SimulatedEnvironment
    evaluation: SimulatedEnvironment
    truth: CausalParameters = field(repr=False)


def true_acceptance_probability(
    frame: pd.DataFrame,
    truth: CausalParameters | None = None,
) -> np.ndarray:
    """Return the simulator's seller-acceptance probability for each row."""

    parameters = truth or CausalParameters()
    required = {
        "offer_ratio",
        "seller_urgency",
        "market_heat",
        "repair_cost_fraction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing acceptance columns: {sorted(missing)}")

    linear_predictor = (
        parameters.acceptance_intercept
        + parameters.acceptance_offer_ratio * (frame["offer_ratio"].to_numpy() - 0.90)
        + parameters.acceptance_seller_urgency * frame["seller_urgency"].to_numpy()
        + parameters.acceptance_market_heat * frame["market_heat"].to_numpy()
        + parameters.acceptance_repair_fraction * frame["repair_cost_fraction"].to_numpy()
    )
    return _sigmoid(linear_predictor)


def true_listing_hazard_probability(
    frame: pd.DataFrame,
    period: int | np.ndarray | pd.Series,
    truth: CausalParameters | None = None,
) -> np.ndarray:
    """Return the known conditional sale probability in a listing period."""

    parameters = truth or CausalParameters()
    required = {
        "list_price_premium",
        "market_heat",
        "condition_score",
        "mortgage_rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing hazard columns: {sorted(missing)}")

    period_array = np.asarray(period, dtype=float)
    if period_array.ndim == 0:
        period_array = np.full(len(frame), float(period_array))
    if len(period_array) != len(frame):
        raise ValueError("period must be scalar or have one value per row")
    if np.any(period_array < 1):
        raise ValueError("period values must be positive")

    latent_demand = (
        frame["truth_demand_shock"].to_numpy()
        if "truth_demand_shock" in frame
        else np.zeros(len(frame))
    )
    linear_predictor = (
        parameters.hazard_intercept
        + parameters.hazard_log_period * np.log(period_array)
        + parameters.hazard_list_price_premium * frame["list_price_premium"].to_numpy()
        + parameters.hazard_market_heat * frame["market_heat"].to_numpy()
        + parameters.hazard_condition * (frame["condition_score"].to_numpy() - 3.0)
        + parameters.hazard_mortgage_rate * (frame["mortgage_rate"].to_numpy() - 0.06)
        + parameters.hazard_latent_demand * latent_demand
    )
    return _sigmoid(linear_predictor)


def make_survival_panel(
    listings: pd.DataFrame,
    truth: CausalParameters | None = None,
) -> pd.DataFrame:
    """Expand listing outcomes into one at-risk row per observed period.

    Censored listings contribute zero-event rows through their censoring period.
    Sold listings contribute a single event in their final observed period.
    """

    required = {"listing_periods", "event_observed", "listing_id"}
    missing = required.difference(listings.columns)
    if missing:
        raise KeyError(f"missing survival columns: {sorted(missing)}")
    if listings.empty:
        return pd.DataFrame(columns=[*listings.columns, "period", "event_in_period", "at_risk"])

    durations = listings["listing_periods"].to_numpy(dtype=int)
    if np.any(durations < 1):
        raise ValueError("listing_periods must be positive")
    row_positions = np.repeat(np.arange(len(listings)), durations)
    periods = np.concatenate([np.arange(1, duration + 1) for duration in durations])
    panel = listings.iloc[row_positions].reset_index(drop=True)
    panel["period"] = periods
    final_period = periods == np.repeat(durations, durations)
    panel["event_in_period"] = (
        final_period & np.repeat(listings["event_observed"].to_numpy(dtype=bool), durations)
    ).astype(int)
    panel["at_risk"] = 1
    panel["true_hazard_probability"] = true_listing_hazard_probability(
        panel, panel["period"], truth
    )
    return panel


def simulate_environment(
    config: EnvironmentConfig,
    truth: CausalParameters | None = None,
) -> SimulatedEnvironment:
    """Generate one deterministic, entirely semi-synthetic environment."""

    parameters = truth or CausalParameters()
    rng = np.random.default_rng(config.seed)
    n_homes = config.n_homes

    submarket = rng.choice(
        _SUBMARKETS,
        size=n_homes,
        p=np.asarray(config.submarket_probabilities),
    )
    square_feet = np.clip(
        np.exp(rng.normal(7.48 + config.log_sqft_shift, 0.34, n_homes)),
        650,
        5_000,
    ).round()
    bedrooms = np.clip(np.rint(square_feet / 590 + rng.normal(0.15, 0.62, n_homes)), 1, 7).astype(
        int
    )
    bathrooms = np.clip(
        np.round((0.52 * bedrooms + rng.normal(0.25, 0.55, n_homes)) * 2) / 2,
        1,
        6,
    )
    age_years = np.clip(rng.gamma(2.3, 16.0, n_homes), 0, 120).round().astype(int)
    condition_score = rng.choice(np.arange(1, 6), size=n_homes, p=[0.06, 0.16, 0.42, 0.27, 0.09])
    quality_score = rng.choice(np.arange(1, 6), size=n_homes, p=[0.05, 0.19, 0.43, 0.25, 0.08])
    lot_square_feet = np.clip(
        square_feet * rng.lognormal(1.15, 0.42, n_homes), 1_200, 35_000
    ).round()
    market_heat = rng.normal(config.market_heat_mean, 0.34, n_homes)
    mortgage_rate = np.clip(rng.normal(config.mortgage_rate_mean, 0.0045, n_homes), 0.025, 0.12)
    annual_appreciation = rng.normal(config.annual_appreciation_mean, 0.022, n_homes)
    submarket_effect = np.array([_SUBMARKET_LOG_VALUE[value] for value in submarket])
    stable_home_noise = rng.normal(0.0, 0.075, n_homes)
    log_market_value = (
        13.02
        + config.log_value_shift
        + 0.50 * np.log(square_feet / 1_800)
        + 0.025 * (bedrooms - 3)
        + 0.050 * (bathrooms - 2)
        - 0.0030 * age_years
        + 0.045 * (condition_score - 3)
        + 0.075 * (quality_score - 3)
        + 0.10 * np.log(lot_square_feet / 6_000)
        + submarket_effect
        + 0.060 * market_heat
        - 1.50 * (mortgage_rate - 0.06)
        + stable_home_noise
    )
    true_market_value = np.exp(log_market_value)
    observed_sale_price = true_market_value * np.exp(rng.normal(0.0, 0.045, n_homes))

    home_ids = [f"{config.name}_home_{index:06d}" for index in range(n_homes)]
    homes = pd.DataFrame(
        {
            "environment": config.name,
            "home_id": home_ids,
            "submarket": submarket,
            "square_feet": square_feet.astype(int),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "condition_score": condition_score,
            "quality_score": quality_score,
            "lot_square_feet": lot_square_feet.astype(int),
            "market_heat": market_heat,
            "mortgage_rate": mortgage_rate,
            "annual_appreciation": annual_appreciation,
            "observed_sale_price": observed_sale_price,
            "true_market_value": true_market_value,
        }
    )

    offers = homes.copy()
    offers["offer_id"] = [f"{config.name}_offer_{index:06d}" for index in range(n_homes)]
    # Randomized independently of home and market features by construction.
    offers["offer_ratio"] = rng.uniform(0.82, 1.02, n_homes)
    offers["offer_price"] = offers["true_market_value"] * offers["offer_ratio"]
    offers["seller_urgency"] = rng.normal(0.0, 1.0, n_homes)
    offers["repair_cost_fraction"] = np.clip(rng.beta(2.0, 12.0, n_homes) * 0.30, 0.0, 0.18)
    offers["true_acceptance_probability"] = true_acceptance_probability(offers, parameters)
    offers["seller_accepted"] = (
        rng.random(n_homes) < offers["true_acceptance_probability"]
    ).astype(int)

    listings = offers.loc[offers["seller_accepted"].eq(1)].copy().reset_index(drop=True)
    n_listings = len(listings)
    listings["listing_id"] = [f"{config.name}_listing_{index:06d}" for index in range(n_listings)]
    listings["true_exit_value"] = listings["true_market_value"] * np.exp(
        0.50 * listings["annual_appreciation"].to_numpy() + rng.normal(0.0, 0.025, n_listings)
    )
    # This randomized treatment is the known causal price response in the exit model.
    listings["list_price_premium"] = rng.uniform(-0.04, 0.14, n_listings)
    listings["list_price"] = listings["true_exit_value"] * (1.0 + listings["list_price_premium"])
    listings["truth_demand_shock"] = rng.normal(0.0, 1.0, n_listings)

    max_periods = config.max_listing_periods
    period_grid = np.arange(1, max_periods + 1)
    hazard = np.column_stack(
        [true_listing_hazard_probability(listings, period, parameters) for period in period_grid]
    )
    event_draws = rng.random((n_listings, max_periods))
    event_matrix = event_draws < hazard
    has_potential_sale = event_matrix.any(axis=1)
    potential_sale_period = np.where(
        has_potential_sale,
        event_matrix.argmax(axis=1) + 1,
        max_periods + 1,
    )
    censor_floor = max(3, max_periods - 4)
    censor_period = rng.integers(censor_floor, max_periods + 1, n_listings)
    event_observed = potential_sale_period <= censor_period
    observed_periods = np.where(event_observed, potential_sale_period, censor_period)

    listings["censor_period"] = censor_period
    listings["listing_periods"] = observed_periods.astype(int)
    listings["event_observed"] = event_observed.astype(int)
    listings["censored"] = (~event_observed).astype(int)
    listings["days_on_market"] = listings["listing_periods"] * 14
    listings["true_hazard_period_1"] = hazard[:, 0]
    sale_noise = rng.normal(0.0, 0.020, n_listings)
    realized_exit_price = listings["true_exit_value"].to_numpy() * np.exp(
        parameters.exit_price_list_premium * listings["list_price_premium"].to_numpy() + sale_noise
    )
    listings["exit_sale_price"] = np.where(event_observed, realized_exit_price, np.nan)

    survival_panel = make_survival_panel(listings, parameters)
    return SimulatedEnvironment(
        config=config,
        truth=parameters,
        homes=homes,
        offers=offers,
        listings=listings,
        survival_panel=survival_panel,
    )


def simulate_train_evaluation(
    *,
    seed: int = 2_026,
    n_train: int = 3_000,
    n_evaluation: int = 1_500,
    max_listing_periods: int = 12,
    truth: CausalParameters | None = None,
) -> TrainEvaluationSimulation:
    """Generate independent train/evaluation environments with covariate shift."""

    parameters = truth or CausalParameters()
    seed_sequence = np.random.SeedSequence(seed)
    train_seed, evaluation_seed = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(2)
    ]
    train_config = EnvironmentConfig(
        name="train",
        seed=train_seed,
        n_homes=n_train,
        max_listing_periods=max_listing_periods,
        market_heat_mean=0.20,
        mortgage_rate_mean=0.055,
        annual_appreciation_mean=0.030,
    )
    evaluation_config = EnvironmentConfig(
        name="evaluation",
        seed=evaluation_seed,
        n_homes=n_evaluation,
        max_listing_periods=max_listing_periods,
        market_heat_mean=-0.18,
        mortgage_rate_mean=0.069,
        annual_appreciation_mean=-0.005,
        log_value_shift=0.035,
        log_sqft_shift=0.045,
        submarket_probabilities=(0.10, 0.13, 0.16, 0.19, 0.21, 0.21),
    )
    return TrainEvaluationSimulation(
        train=simulate_environment(train_config, parameters),
        evaluation=simulate_environment(evaluation_config, parameters),
        truth=parameters,
    )
