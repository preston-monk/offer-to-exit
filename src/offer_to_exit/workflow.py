"""Deterministic end-to-end workflow for the controlled policy experiment.

The workflow is deliberately small enough to reproduce locally. It generates two
independent environments, fits three transparent models, evaluates them under a
documented covariate shift, solves three diagnostic decision cases, and publishes
auditable artifacts. Generated behavior and policy outcomes are not real-market claims.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from offer_to_exit import __version__
from offer_to_exit.decision import (
    FittedHazardSaleOutcomeAdapter,
    FittedSellerAcceptanceAdapter,
    HomeContext,
    ValueScenarioSpec,
    WorkedCaseModels,
    model_driven_worked_decision_cases,
    solve_worked_decision_cases,
)
from offer_to_exit.models import (
    CalibratedLinearValuation,
    DiscreteTimeHazardModel,
    SellerAcceptanceModel,
)
from offer_to_exit.simulation import (
    LIST_PRICE_PREMIUM_SUPPORT,
    OFFER_RATIO_SUPPORT,
    CausalParameters,
    EnvironmentConfig,
    SimulatedEnvironment,
    TrainEvaluationSimulation,
    simulate_environment,
    true_acceptance_probability,
)

plt.switch_backend("Agg")


ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_VERSION = "v2"
CLAIM_SCOPE = (
    "All behavioral responses, counterfactuals, policy economics, and model metrics "
    "in this run come from a controlled generated experiment; they are not estimates of any real operator, "
    "homeowner population, or housing market."
)

VALUATION_NUMERIC_FEATURES = (
    "square_feet",
    "bedrooms",
    "bathrooms",
    "age_years",
    "condition_score",
    "quality_score",
    "lot_square_feet",
    "market_heat",
    "mortgage_rate",
)
ACCEPTANCE_NUMERIC_FEATURES = (
    "offer_ratio",
    "seller_urgency",
    "market_heat",
    "repair_cost_fraction",
)
HAZARD_NUMERIC_FEATURES = (
    "list_price_premium",
    "market_heat",
    "condition_score",
    "mortgage_rate",
)
_CATEGORICAL_FEATURES = ("submarket",)
DECISION_HORIZON_WEEKS = 17


def run_experiment(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the configured controlled experiment and publish its artifacts.

    The signature intentionally matches :mod:`offer_to_exit.cli`: the CLI passes an
    already-parsed YAML mapping and the path used to obtain it. The returned mapping
    contains only JSON-compatible values.
    """

    run_config = _section(config, "run")
    data_config = _section(config, "data")
    simulator_config = _section(config, "simulator")
    market_config = _section(config, "market")

    mode = str(run_config.get("mode", "controlled_experiment"))
    if mode not in {"controlled_experiment", "controlled-experiment"}:
        raise ValueError("the public workflow supports controlled_experiment mode only")
    if str(data_config.get("source_mode", "generated_fixture")) != "generated_fixture":
        raise ValueError("the public workflow accepts generated fixtures only")

    run_name = str(run_config.get("name", "quickstart"))
    run_seed = int(run_config.get("seed", 2_026_090_1))
    output_dir = Path(str(run_config.get("output_dir", "artifacts/quickstart")))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _artifact_paths(output_dir)
    _guard_output_targets(
        artifact_paths,
        overwrite=bool(run_config.get("overwrite", False)),
    )

    simulation = _build_simulation(
        data_config=data_config,
        simulator_config=simulator_config,
        market_config=market_config,
        fallback_seed=run_seed,
    )
    models = _fit_models(simulation)
    metrics, plot_data = _evaluate_models(simulation, models)
    decision_cases = _solve_decision_cases(simulation, models)

    _write_figures(plot_data, artifact_paths["figures"])

    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    config_sha256 = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
    run_id = f"{_slug(run_name)}-seed-{run_seed}-schema-{ARTIFACT_SCHEMA_VERSION}"
    artifacts: dict[str, str | list[str]] = {
        "output_dir": str(output_dir),
        "manifest": str(artifact_paths["manifest"]),
        "metrics": str(artifact_paths["metrics"]),
        "decision_cases": str(artifact_paths["decision_cases"]),
        "summary": str(artifact_paths["summary"]),
        "figures": [str(path) for path in artifact_paths["figures"]],
        "demo": str(artifact_paths["demo"]),
    }
    result: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "package_version": __version__,
        "run_id": run_id,
        "run_name": run_name,
        "mode": "controlled_experiment",
        "claim_scope": CLAIM_SCOPE,
        "market_shape": str(market_config.get("name", "unspecified synthetic market")),
        "reproducibility": {
            "run_seed": run_seed,
            "training_seed": simulation.train.config.seed,
            "evaluation_seed": simulation.evaluation.config.seed,
            "config_sha256": config_sha256,
            "config_path": str(config_path) if config_path is not None else None,
            "independent_evaluation_environment": True,
            "evaluation_truth_excluded_from_model_fitting": True,
            "decision_cases_use_fitted_models": True,
            "decision_case_truth_excluded": True,
            "acquisition_offer_reference_observable": True,
            "list_price_reference_observable": True,
            "behavioral_treatment_support_enforced": True,
            "hazard_truth_matches_fitted_estimand": True,
            "valuation_scenario_weights_are_heuristic": True,
            "weekly_stress_scenarios_are_remixed": True,
        },
        "data": {
            "source": "controlled generated fixture",
            "n_train_homes": len(simulation.train.homes),
            "n_evaluation_homes": len(simulation.evaluation.homes),
            "n_train_listings": len(simulation.train.listings),
            "n_evaluation_listings": len(simulation.evaluation.listings),
            "listing_period_days": 7,
            "max_listing_periods": simulation.train.config.max_listing_periods,
            "acquisition_offer_ratio_support": list(OFFER_RATIO_SUPPORT),
            "list_price_premium_support": list(LIST_PRICE_PREMIUM_SUPPORT),
        },
        "metrics": metrics,
        "decision_cases": decision_cases,
        "artifacts": artifacts,
    }

    metrics_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "claim_scope": CLAIM_SCOPE,
        "evaluation_environment": "independent shifted generated evaluation environment",
        "metrics": metrics,
    }
    _write_json(artifact_paths["metrics"], metrics_payload)
    _write_decision_csv(artifact_paths["decision_cases"], decision_cases)
    render_demo(result, artifact_paths["demo"])
    _write_json(artifact_paths["summary"], result)
    manifest_artifacts: dict[str, str | list[str]] = {}
    for key, artifact_value in artifacts.items():
        if key == "output_dir":
            continue
        if isinstance(artifact_value, list):
            manifest_artifacts[key] = [Path(item).name for item in artifact_value]
        else:
            manifest_artifacts[key] = Path(artifact_value).name

    emitted_paths = [
        artifact_paths["metrics"],
        artifact_paths["decision_cases"],
        artifact_paths["summary"],
        *artifact_paths["figures"],
        artifact_paths["demo"],
    ]

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "package_version": __version__,
        "run_id": run_id,
        "mode": "controlled_experiment",
        "claim_scope": CLAIM_SCOPE,
        "config_sha256": config_sha256,
        "training_seed": simulation.train.config.seed,
        "evaluation_seed": simulation.evaluation.config.seed,
        "artifacts": manifest_artifacts,
        "artifact_sha256": {path.name: _sha256_file(path) for path in emitted_paths},
    }
    _write_json(artifact_paths["manifest"], manifest)
    return result


def render_demo(result: Mapping[str, Any], output: Path) -> Path:
    """Render a self-contained static HTML decision-science demonstration."""

    metrics = _section(result, "metrics")
    valuation = _section(metrics, "valuation")
    acceptance = _section(metrics, "seller_acceptance")
    hazard = _section(metrics, "sale_hazard")
    data = _section(result, "data")
    reproducibility = _section(result, "reproducibility")
    decision_cases = result.get("decision_cases", [])
    if not isinstance(decision_cases, Sequence) or isinstance(decision_cases, str):
        raise TypeError("decision_cases must be a sequence")

    artifacts = _section(result, "artifacts")
    figures_value = artifacts.get("figures", [])
    figure_paths = (
        [Path(str(path)) for path in figures_value]
        if isinstance(figures_value, Sequence) and not isinstance(figures_value, str)
        else []
    )
    figure_titles = (
        "Contemporaneous value anchor on shifted evaluation homes",
        "Seller response: fitted versus simulator truth",
        "List-price response and cumulative sell-through",
    )
    figure_cards = "".join(
        f"""
        <figure class="figure-card">
          <img src="{_image_data_uri(path)}" alt="{html.escape(title)}">
          <figcaption>{html.escape(title)} · controlled evaluation environment</figcaption>
        </figure>
        """
        for path, title in zip(figure_paths, figure_titles, strict=False)
        if path.exists()
    )

    case_cards = "".join(_case_html(case) for case in decision_cases if isinstance(case, Mapping))
    claim_scope = html.escape(str(result.get("claim_scope", CLAIM_SCOPE)))
    run_id = html.escape(str(result.get("run_id", "controlled-experiment-run")))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Offer-to-Exit · Controlled Decision Lab</title>
  <style>
    :root {{
      --ink: #14232b; --muted: #5d6d74; --paper: #f5f2eb; --card: #fffdf8;
      --teal: #087e78; --teal-dark: #075853; --coral: #db694d; --line: #d9d7cf;
      --shadow: 0 16px 42px rgba(20,35,43,.09); --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper); font-family: Inter, ui-sans-serif,
      -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; }}
    .topline {{ height: 7px; background: linear-gradient(90deg,var(--teal),#23a49c,var(--coral)); }}
    main {{ width: min(1160px, calc(100% - 36px)); margin: 0 auto; padding: 54px 0 80px; }}
    .hero {{ display: grid; grid-template-columns: 1.45fr .75fr; gap: 28px; align-items: stretch; }}
    .hero-copy, .provenance {{ background: var(--card); border: 1px solid var(--line);
      border-radius: var(--radius); box-shadow: var(--shadow); }}
    .hero-copy {{ padding: clamp(30px,5vw,62px); }}
    .eyebrow {{ color: var(--teal-dark); font-size: .77rem; font-weight: 800; letter-spacing: .13em;
      text-transform: uppercase; }}
    h1 {{ margin: 12px 0 16px; font-size: clamp(2.5rem,6vw,5.2rem); line-height: .94;
      letter-spacing: -.055em; max-width: 820px; }}
    .lede {{ margin: 0; color: var(--muted); font-size: 1.08rem; max-width: 720px; }}
    .badge {{ display: inline-flex; margin-top: 24px; padding: 9px 13px; border-radius: 999px;
      background: #e5f4f1; color: var(--teal-dark); font-size: .76rem; font-weight: 850;
      letter-spacing: .07em; }}
    .provenance {{ padding: 30px; display: flex; flex-direction: column; justify-content: space-between;
      background: var(--ink); color: #f8f4eb; }}
    .provenance h2 {{ margin: 0 0 18px; font-size: 1.15rem; }}
    .provenance dl {{ margin: 0; display: grid; gap: 16px; }}
    .provenance dt {{ color: #9fb6ba; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; }}
    .provenance dd {{ margin: 2px 0 0; font-size: .92rem; overflow-wrap: anywhere; }}
    section {{ margin-top: 56px; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 24px; align-items: end;
      margin-bottom: 20px; }}
    .section-head h2 {{ margin: 0; font-size: clamp(1.65rem,3vw,2.45rem); letter-spacing: -.035em; }}
    .section-head p {{ margin: 0; color: var(--muted); max-width: 560px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }}
    .metric {{ padding: 22px; background: var(--card); border: 1px solid var(--line); border-radius: 17px; }}
    .metric .value {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -.035em; }}
    .metric .label {{ color: var(--muted); font-size: .78rem; margin-top: 5px; }}
    .metric .context {{ color: var(--teal-dark); font-size: .7rem; margin-top: 12px; font-weight: 700; }}
    .figures {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }}
    .figure-card {{ margin: 0; overflow: hidden; background: var(--card); border: 1px solid var(--line);
      border-radius: 18px; box-shadow: 0 8px 22px rgba(20,35,43,.05); }}
    .figure-card img {{ width: 100%; aspect-ratio: 4/3; display: block; object-fit: contain; background: white; }}
    .figure-card figcaption {{ padding: 13px 16px 15px; color: var(--muted); font-size: .76rem; }}
    .cases {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }}
    details.case {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px;
      padding: 0 20px 20px; }}
    details.case[open] {{ box-shadow: var(--shadow); border-color: #a8cbc6; }}
    details.case summary {{ cursor: pointer; list-style: none; padding: 22px 0 14px; }}
    details.case summary::-webkit-details-marker {{ display: none; }}
    .case-kicker {{ color: var(--coral); text-transform: uppercase; letter-spacing: .1em;
      font-size: .68rem; font-weight: 850; }}
    .case h3 {{ margin: 5px 0 8px; font-size: 1.12rem; line-height: 1.25; }}
    .case-question {{ color: var(--muted); font-size: .84rem; min-height: 48px; }}
    .case-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }}
    .case-stat {{ background: #f4f7f5; border-radius: 11px; padding: 11px; }}
    .case-stat b {{ display: block; font-size: 1rem; }}
    .case-stat span {{ color: var(--muted); font-size: .68rem; }}
    .path {{ margin-top: 14px; padding-top: 13px; border-top: 1px solid var(--line); font-size: .76rem; }}
    .disclaimer {{ margin-top: 56px; padding: 24px 28px; border-left: 5px solid var(--coral);
      border-radius: 4px 15px 15px 4px; background: #fff7f2; }}
    .disclaimer strong {{ display: block; margin-bottom: 5px; }}
    footer {{ color: var(--muted); font-size: .74rem; margin-top: 30px; }}
    @media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2,1fr); }} .figures,.cases {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 520px) {{ main {{ width: min(100% - 22px,1160px); padding-top: 24px; }}
      .metrics {{ grid-template-columns: 1fr; }} .section-head {{ display: block; }} }}
  </style>
</head>
<body>
<div class="topline"></div>
<main>
  <div class="hero">
    <div class="hero-copy">
      <div class="eyebrow">Residential pricing decision system</div>
      <h1>Offer to exit,<br>as one decision.</h1>
      <p class="lede">A transparent laboratory connecting valuation uncertainty, seller conversion,
      buyer price response, inventory risk, and the acquisition offer. This evidence layer is
      location-neutral and built entirely from generated behavior.</p>
      <span class="badge">CONTROLLED EXPERIMENT · AUDITABLE · REPRODUCIBLE</span>
    </div>
    <aside class="provenance">
      <div><h2>Locked run</h2><dl>
        <div><dt>Run ID</dt><dd>{run_id}</dd></div>
        <div><dt>Training seed</dt><dd>{html.escape(str(reproducibility.get("training_seed", "—")))}</dd></div>
        <div><dt>Evaluation seed</dt><dd>{html.escape(str(reproducibility.get("evaluation_seed", "—")))}</dd></div>
        <div><dt>Evaluation homes</dt><dd>{_integer(data.get("n_evaluation_homes", 0))}</dd></div>
      </dl></div>
      <small>Evaluation is independently generated, distribution-shifted, and excluded from model fitting.</small>
    </aside>
  </div>

  <section>
    <div class="section-head"><h2>Evidence, before optimization</h2>
      <p>Predictive quality and behavioral recovery are measured only on the independent evaluation environment.</p></div>
    <div class="metrics">
      {_metric_card(_money(valuation.get("median_absolute_error", 0)), "Home-value anchor median absolute error", "Lower is better")}
      {_metric_card(_percent(valuation.get("interval_coverage", 0)), "90% interval coverage", "Split conformal")}
      {_metric_card(_decimal(acceptance.get("brier_score", 0), 3), "Seller-acceptance Brier", "Versus pooled-rate baseline")}
      {_metric_card(_decimal(hazard.get("person_period_log_loss", 0), 3), "Sale-hazard log loss", "Right censoring retained")}
    </div>
  </section>

  <section>
    <div class="section-head"><h2>What the models learned</h2>
      <p>Fitted responses are shown beside known simulator truth; truth is used for evaluation, never model fitting.</p></div>
    <div class="figures">{figure_cards}</div>
  </section>

  <section>
    <div class="section-head"><h2>Three decisions under uncertainty</h2>
      <p>Open a case to inspect the selected acquisition offer and the no-sale markdown path.</p></div>
    <div class="cases">{case_cards}</div>
  </section>

  <aside class="disclaimer"><strong>Scope of evidence</strong>{claim_scope}</aside>
  <footer>Offer-to-Exit package {html.escape(str(result.get("package_version", "unknown")))} ·
    schema {html.escape(str(result.get("schema_version", "—")))} · static file, no external assets or scripts.</footer>
</main>
</body>
</html>
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    output.write_text(clean_document, encoding="utf-8")
    return output


def _build_simulation(
    *,
    data_config: Mapping[str, Any],
    simulator_config: Mapping[str, Any],
    market_config: Mapping[str, Any],
    fallback_seed: int,
) -> TrainEvaluationSimulation:
    n_total = int(data_config.get("generated_property_count", 1_200))
    if n_total < 200:
        raise ValueError("generated_property_count must be at least 200")
    split_config = _section(data_config, "split")
    train_fraction = float(split_config.get("train_fraction", 0.60))
    if not 0.25 <= train_fraction <= 0.85:
        raise ValueError("train_fraction must be between 0.25 and 0.85")
    n_train = max(100, round(n_total * train_fraction))
    n_evaluation = n_total - n_train
    if n_evaluation < 80:
        n_evaluation = 80

    horizon_weeks = int(market_config.get("exit_horizon_weeks", 17))
    max_periods = max(DECISION_HORIZON_WEEKS, horizon_weeks)
    training_seed = int(simulator_config.get("training_seed", fallback_seed))
    evaluation_seed = int(simulator_config.get("evaluation_seed", fallback_seed + 1))
    if training_seed == evaluation_seed:
        raise ValueError("training_seed and evaluation_seed must differ")

    truth = CausalParameters()
    train = simulate_environment(
        EnvironmentConfig(
            name="train",
            seed=training_seed,
            n_homes=n_train,
            max_listing_periods=max_periods,
            market_heat_mean=0.20,
            mortgage_rate_mean=0.055,
            annual_appreciation_mean=0.030,
        ),
        truth,
    )
    evaluation = simulate_environment(
        EnvironmentConfig(
            name="evaluation",
            seed=evaluation_seed,
            n_homes=n_evaluation,
            max_listing_periods=max_periods,
            market_heat_mean=-0.18,
            mortgage_rate_mean=0.069,
            annual_appreciation_mean=-0.005,
            log_value_shift=0.035,
            log_sqft_shift=0.045,
            submarket_probabilities=(0.10, 0.13, 0.16, 0.19, 0.21, 0.21),
        ),
        truth,
    )
    return TrainEvaluationSimulation(train=train, evaluation=evaluation, truth=truth)


def _fit_models(simulation: TrainEvaluationSimulation) -> dict[str, Any]:
    valuation = CalibratedLinearValuation(
        VALUATION_NUMERIC_FEATURES,
        _CATEGORICAL_FEATURES,
        alpha=0.10,
        random_state=73,
    ).fit(simulation.train.homes, simulation.train.homes["observed_sale_price"])
    acceptance = SellerAcceptanceModel(
        ACCEPTANCE_NUMERIC_FEATURES,
        _CATEGORICAL_FEATURES,
        regularization=0.25,
        random_state=79,
    ).fit(simulation.train.offers, simulation.train.offers["seller_accepted"])
    hazard = DiscreteTimeHazardModel(
        HAZARD_NUMERIC_FEATURES,
        _CATEGORICAL_FEATURES,
        max_periods=simulation.train.config.max_listing_periods,
        regularization=0.25,
        random_state=83,
    ).fit(simulation.train.listings)
    return {"valuation": valuation, "acceptance": acceptance, "hazard": hazard}


def _evaluate_models(
    simulation: TrainEvaluationSimulation,
    models: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    valuation: CalibratedLinearValuation = models["valuation"]
    acceptance: SellerAcceptanceModel = models["acceptance"]
    hazard: DiscreteTimeHazardModel = models["hazard"]
    evaluation = simulation.evaluation

    value_target = evaluation.homes["observed_sale_price"].to_numpy(dtype=float)
    value_prediction = valuation.predict(evaluation.homes)
    value_interval = valuation.predict_interval(evaluation.homes)
    value_error = value_prediction - value_target
    naive_value = float(simulation.train.homes["observed_sale_price"].median())
    naive_absolute_error = np.abs(value_target - naive_value)
    interval_covered = (value_target >= value_interval[:, 0]) & (
        value_target <= value_interval[:, 1]
    )
    valuation_metrics = {
        "evaluation_rows": len(value_target),
        "median_absolute_error": _rounded(np.median(np.abs(value_error)), 2),
        "mean_absolute_error": _rounded(np.mean(np.abs(value_error)), 2),
        "wape": _rounded(np.sum(np.abs(value_error)) / np.sum(value_target), 6),
        "median_absolute_percentage_error": _rounded(
            np.median(np.abs(value_error) / value_target), 6
        ),
        "interval_nominal_coverage": 0.90,
        "interval_coverage": _rounded(np.mean(interval_covered), 6),
        "median_interval_width_rate": _rounded(
            np.median((value_interval[:, 1] - value_interval[:, 0]) / value_prediction),
            6,
        ),
        "pooled_median_baseline_mae": _rounded(np.mean(naive_absolute_error), 2),
        "mae_improvement_over_pooled_median": _rounded(
            1.0 - np.mean(np.abs(value_error)) / np.mean(naive_absolute_error), 6
        ),
    }

    acceptance_target = evaluation.offers["seller_accepted"].to_numpy(dtype=float)
    acceptance_probability = acceptance.predict_proba(evaluation.offers)
    pooled_acceptance = float(simulation.train.offers["seller_accepted"].mean())
    acceptance_probe = evaluation.offers.iloc[[0]].copy()
    fitted_acceptance_curve = acceptance.offer_curve(acceptance_probe, [0.86, 0.96])[
        "acceptance_probability"
    ].to_numpy()
    fitted_acceptance_effect = _log_odds(fitted_acceptance_curve[1]) - _log_odds(
        fitted_acceptance_curve[0]
    )
    acceptance_metrics = {
        "evaluation_rows": len(acceptance_target),
        "brier_score": _rounded(np.mean((acceptance_probability - acceptance_target) ** 2), 6),
        "pooled_rate_baseline_brier": _rounded(
            np.mean((pooled_acceptance - acceptance_target) ** 2), 6
        ),
        "log_loss": _rounded(_binary_log_loss(acceptance_target, acceptance_probability), 6),
        "expected_calibration_error_10_bin": _rounded(
            _calibration_error(acceptance_target, acceptance_probability), 6
        ),
        "estimated_log_odds_effect_per_10pp_offer": _rounded(fitted_acceptance_effect, 6),
        "truth_log_odds_effect_per_10pp_offer": _rounded(
            simulation.truth.acceptance_log_odds_per_ten_pct, 6
        ),
        "absolute_response_recovery_error": _rounded(
            abs(fitted_acceptance_effect - simulation.truth.acceptance_log_odds_per_ten_pct),
            6,
        ),
        "counterfactual_response_monotone": bool(
            fitted_acceptance_curve[1] > fitted_acceptance_curve[0]
        ),
    }

    evaluation_hazard = hazard.predict_hazard(evaluation.listings)
    durations = evaluation.listings["listing_periods"].to_numpy(dtype=int)
    panel_rows = np.repeat(np.arange(len(evaluation.listings)), durations)
    panel_periods = evaluation.survival_panel["period"].to_numpy(dtype=int) - 1
    panel_probability = evaluation_hazard[panel_rows, panel_periods]
    panel_target = evaluation.survival_panel["event_in_period"].to_numpy(dtype=float)
    training_period_rate = simulation.train.survival_panel.groupby("period")[
        "event_in_period"
    ].mean()
    fallback_hazard = float(simulation.train.survival_panel["event_in_period"].mean())
    panel_baseline = np.array(
        [float(training_period_rate.get(period + 1, fallback_hazard)) for period in panel_periods]
    )

    hazard_probe = evaluation.listings.iloc[[0]].copy()
    low_premium = hazard_probe.copy()
    high_premium = hazard_probe.copy()
    low_premium["list_price_premium"] = -0.02
    high_premium["list_price_premium"] = 0.08
    fitted_low = hazard.predict_hazard(low_premium, horizon=1)[0, 0]
    fitted_high = hazard.predict_hazard(high_premium, horizon=1)[0, 0]
    fitted_hazard_effect = _log_odds(fitted_high) - _log_odds(fitted_low)
    hazard_metrics = {
        "evaluation_listings": len(evaluation.listings),
        "evaluation_person_periods": len(panel_target),
        "person_period_brier_score": _rounded(np.mean((panel_probability - panel_target) ** 2), 6),
        "period_rate_baseline_brier": _rounded(np.mean((panel_baseline - panel_target) ** 2), 6),
        "person_period_log_loss": _rounded(_binary_log_loss(panel_target, panel_probability), 6),
        "period_rate_baseline_log_loss": _rounded(
            _binary_log_loss(panel_target, panel_baseline), 6
        ),
        "mean_predicted_sell_through_by_horizon": _rounded(
            np.mean(hazard.predict_sale_probability(evaluation.listings)), 6
        ),
        "estimated_log_odds_effect_per_10pp_overpricing": _rounded(fitted_hazard_effect, 6),
        "truth_log_odds_effect_per_10pp_overpricing": _rounded(
            simulation.truth.hazard_log_odds_per_ten_pct_overpricing, 6
        ),
        "absolute_response_recovery_error": _rounded(
            abs(fitted_hazard_effect - simulation.truth.hazard_log_odds_per_ten_pct_overpricing),
            6,
        ),
        "counterfactual_response_monotone": bool(fitted_high < fitted_low),
    }

    metrics = {
        "evaluation_contract": {
            "environment": "independent shifted generated evaluation environment",
            "used_for_fitting": False,
            "right_censoring_preserved": True,
            "behavioral_truth_used_for_scoring_only": True,
        },
        "valuation": valuation_metrics,
        "seller_acceptance": acceptance_metrics,
        "sale_hazard": hazard_metrics,
    }
    plot_data = {
        "value_target": value_target,
        "value_prediction": value_prediction,
        "evaluation": evaluation,
        "acceptance_model": acceptance,
        "hazard_model": hazard,
        "truth": simulation.truth,
    }
    return metrics, plot_data


def _solve_decision_cases(
    simulation: TrainEvaluationSimulation,
    models: Mapping[str, Any],
) -> list[dict[str, Any]]:
    model_inputs = _decision_case_model_inputs(simulation, models)
    fitted_cases = model_driven_worked_decision_cases(
        model_inputs,
        offer_ratio_support=OFFER_RATIO_SUPPORT,
        list_price_premium_support=LIST_PRICE_PREMIUM_SUPPORT,
    )
    cases: list[dict[str, Any]] = []
    for solved in solve_worked_decision_cases(fitted_cases):
        selected = solved.result.selected
        path = selected.exit_result.no_sale_path()
        recommendation = "price" if selected.objective_value > 0 else "abstain"
        recommendation_reason = (
            "best supported offer has positive risk-adjusted value"
            if recommendation == "price"
            else "every supported offer has non-positive risk-adjusted value"
        )
        approximate_loss_probability = sum(
            outcome.probability for outcome in selected.outcomes if outcome.profit < 0
        )
        preoffer_reference_value = float(solved.case.context.features["preoffer_reference_value"])
        prelisting_reference_value = float(
            solved.case.context.features["prelisting_reference_value"]
        )
        if not isinstance(solved.case.outcome_model, FittedHazardSaleOutcomeAdapter):
            raise TypeError("controlled decision case must use the fitted hazard adapter")
        scored_price_premiums = solved.case.outcome_model.scored_price_premiums
        if not scored_price_premiums:
            raise ValueError("fitted hazard adapter did not score any list-price states")
        cases.append(
            {
                "name": solved.case.name,
                "question": solved.case.question,
                "home_id": solved.case.context.home_id,
                "reference_value": _rounded(solved.case.context.reference_value, 2),
                "preoffer_reference_value": _rounded(preoffer_reference_value, 2),
                "prelisting_reference_value": _rounded(prelisting_reference_value, 2),
                "valuation_interval_lower": _rounded(
                    solved.case.context.features["valuation_interval_lower"], 2
                ),
                "valuation_interval_upper": _rounded(
                    solved.case.context.features["valuation_interval_upper"], 2
                ),
                "initial_list_price": _rounded(solved.case.initial_list_price, 2),
                "initial_list_price_premium": _rounded(
                    solved.case.initial_list_price / prelisting_reference_value - 1.0,
                    6,
                ),
                "recommendation": recommendation,
                "recommendation_reason": recommendation_reason,
                "selected_offer": _rounded(selected.offer, 2),
                "automated_offer": (
                    _rounded(selected.offer, 2) if recommendation == "price" else None
                ),
                "selected_offer_to_value_anchor_ratio": _rounded(
                    selected.offer / solved.case.context.reference_value, 6
                ),
                "acceptance_offer_ratio": _rounded(
                    selected.offer / preoffer_reference_value,
                    6,
                ),
                "evaluated_acceptance_offer_ratios": [
                    _rounded(offer / preoffer_reference_value, 6)
                    for offer in solved.case.offer_grid
                ],
                "scored_list_price_premium_min": _rounded(min(scored_price_premiums), 6),
                "scored_list_price_premium_max": _rounded(max(scored_price_premiums), 6),
                "scored_list_price_premium_count": len(scored_price_premiums),
                "acceptance_probability": _rounded(selected.acceptance_probability, 6),
                "first_action": solved.first_action.value,
                "first_post_action_list_price": _rounded(
                    selected.exit_result.first_decision.new_list_price, 2
                ),
                "path_length_weeks": len(path),
                "expected_profit_per_lead": _rounded(selected.expected_profit_per_lead, 2),
                "worst_tail_positive_loss": _rounded(selected.downside_cvar_loss, 2),
                "loss_probability": _rounded(approximate_loss_probability, 6),
                "loss_probability_note": (
                    "approximate negative-profit mass after outcome-grid compression"
                ),
                "risk_adjusted_objective": _rounded(selected.objective_value, 2),
                "no_sale_path": [
                    {
                        "week": decision.state.week,
                        "action": decision.action.value,
                        "post_action_list_price": _rounded(decision.new_list_price, 2),
                        "weekly_sale_probability": _rounded(decision.weekly_sale_probability, 6),
                    }
                    for decision in path
                ],
                "model_source": (
                    "fitted contemporaneous value anchor, seller-acceptance model, and "
                    "sale-hazard model from the controlled experiment"
                ),
                "evidence_scope": (
                    "illustrative controlled-experiment decision; not real-market performance"
                ),
            }
        )
    return cases


def _decision_case_model_inputs(
    simulation: TrainEvaluationSimulation,
    models: Mapping[str, Any],
) -> dict[str, WorkedCaseModels]:
    """Create three observable-feature cases scored by the fitted models.

    Representative rows come from the independent evaluation environment.
    Selection uses observed covariates and fitted valuation outputs. The
    feature frames passed to adapters are projected onto each model's declared
    features, excluding all simulator truth and realized-outcome columns by
    construction.
    """

    valuation: CalibratedLinearValuation = models["valuation"]
    acceptance: SellerAcceptanceModel = models["acceptance"]
    hazard: DiscreteTimeHazardModel = models["hazard"]
    offers = simulation.evaluation.offers
    valuation_rows = offers.loc[:, valuation.features]
    all_value_predictions = valuation.predict(valuation_rows)
    all_value_intervals = valuation.predict_interval(valuation_rows)
    positions = _representative_case_positions(
        simulation,
        value_predictions=all_value_predictions,
        value_intervals=all_value_intervals,
    )
    profiles = tuple(positions)
    selected_rows = offers.loc[[positions[profile] for profile in profiles]].copy()
    selected_offsets = [offers.index.get_loc(positions[profile]) for profile in profiles]
    value_prediction = all_value_predictions[selected_offsets]
    value_interval = all_value_intervals[selected_offsets]

    scenario_design = {
        "healthy-demand-margin-protection": (
            ("soft", 0.20),
            ("base", 0.60),
            ("strong", 0.20),
        ),
        "stale-inventory-high-carry": (
            ("soft", 0.35),
            ("base", 0.50),
            ("rebound", 0.15),
        ),
        "sparse-support-downside-protection": (
            ("downside", 0.30),
            ("base", 0.50),
            ("upside", 0.20),
        ),
    }
    home_ids = {
        "healthy-demand-margin-protection": "LAB-HEALTHY-001",
        "stale-inventory-high-carry": "LAB-STALE-002",
        "sparse-support-downside-protection": "LAB-SPARSE-SUPPORT-003",
    }

    result: dict[str, WorkedCaseModels] = {}
    for row_number, profile in enumerate(profiles):
        row = selected_rows.iloc[row_number]
        reference_value = float(value_prediction[row_number])
        preoffer_reference_value = float(row["preoffer_reference_value"])
        prelisting_reference_value = float(row["prelisting_reference_value"])
        lower = min(float(value_interval[row_number, 0]), reference_value)
        upper = max(float(value_interval[row_number, 1]), reference_value)
        if (
            lower <= 0
            or not np.isfinite(
                [
                    lower,
                    reference_value,
                    upper,
                    preoffer_reference_value,
                    prelisting_reference_value,
                ]
            ).all()
        ):
            raise ValueError("valuation model returned invalid decision stress points")
        if preoffer_reference_value <= 0:
            raise ValueError("observable pre-offer reference must be positive")
        if prelisting_reference_value <= 0:
            raise ValueError("observable pre-listing reference must be positive")
        scenario_values = (lower, reference_value, upper)
        value_scenarios = tuple(
            ValueScenarioSpec(name=name, probability=weight, resale_value=value)
            for (name, weight), value in zip(scenario_design[profile], scenario_values, strict=True)
        )

        acceptance_row = _project_model_features(row, acceptance.features)
        hazard_row = _project_model_features(
            row,
            hazard.features,
            replacements={"list_price_premium": 0.0},
        )
        context = HomeContext(
            home_id=home_ids[profile],
            reference_value=reference_value,
            features={
                "market_heat": float(row["market_heat"]),
                "condition_score": float(row["condition_score"]),
                "mortgage_rate": float(row["mortgage_rate"]),
                "seller_urgency": float(row["seller_urgency"]),
                "repair_cost_fraction": float(row["repair_cost_fraction"]),
                "preoffer_reference_value": preoffer_reference_value,
                "prelisting_reference_value": prelisting_reference_value,
                "valuation_interval_lower": lower,
                "valuation_interval_upper": upper,
            },
        )
        result[profile] = WorkedCaseModels(
            context=context,
            acceptance_model=FittedSellerAcceptanceAdapter(
                acceptance,
                acceptance_row,
                offer_ratio_support=OFFER_RATIO_SUPPORT,
            ),
            outcome_model=FittedHazardSaleOutcomeAdapter(
                hazard,
                hazard_row,
                value_scenarios,
                list_price_premium_support=LIST_PRICE_PREMIUM_SUPPORT,
            ),
        )
    return result


def _representative_case_positions(
    simulation: TrainEvaluationSimulation,
    *,
    value_predictions: np.ndarray,
    value_intervals: np.ndarray,
) -> dict[str, int]:
    """Select healthy, stale, and consequential-uncertainty evaluation profiles.

    Selection uses only observed covariates and fitted valuation outputs.  For
    the sparse-support profile, the lower fitted stress point must bind sale
    proceeds at the initial supported list price while the point prediction
    does not.  That makes the valuation interval economically consequential in
    the worked decision rather than a label attached to an otherwise identical
    set of proceeds.
    """

    evaluation = simulation.evaluation.offers
    required = {
        "market_heat",
        "condition_score",
        "mortgage_rate",
        "square_feet",
        "submarket",
    }
    missing = required.difference(evaluation.columns)
    if missing:
        raise KeyError(f"evaluation features are missing: {sorted(missing)}")
    if len(evaluation) < 3:
        raise ValueError("at least three evaluation homes are required for worked cases")
    if len(value_predictions) != len(evaluation) or value_intervals.shape != (len(evaluation), 2):
        raise ValueError("valuation outputs must align one-for-one with evaluation homes")

    heat = _standardized(evaluation["market_heat"])
    condition = _standardized(evaluation["condition_score"])
    mortgage = _standardized(evaluation["mortgage_rate"])
    healthy_score = heat + 0.25 * condition - 0.25 * mortgage
    healthy_position = int(healthy_score.idxmax())

    stale_score = -heat - 0.20 * condition + 0.35 * mortgage
    stale_score = stale_score.drop(index=healthy_position)
    stale_position = int(stale_score.idxmax())

    support_counts = simulation.train.homes["submarket"].value_counts()
    training_support = evaluation["submarket"].map(support_counts).fillna(0).astype(float)
    size_distance = np.abs(_standardized(np.log(evaluation["square_feet"])))
    prediction = pd.Series(value_predictions, index=evaluation.index, dtype=float)
    lower = pd.Series(value_intervals[:, 0], index=evaluation.index, dtype=float)
    upper = pd.Series(value_intervals[:, 1], index=evaluation.index, dtype=float)
    initial_price = 1.04 * evaluation["prelisting_reference_value"].astype(float)
    initial_proceeds_cap = 0.995 * initial_price
    interval_width_rate = (upper - lower) / prediction
    sparse_score = (
        -np.log1p(training_support)
        + 0.35 * size_distance
        + 0.50 * _standardized(interval_width_rate)
    )
    sparse_score = sparse_score.drop(index=[healthy_position, stale_position])
    consequential = (lower < initial_proceeds_cap) & (initial_proceeds_cap < prediction)
    sparse_score = sparse_score.loc[consequential.reindex(sparse_score.index, fill_value=False)]
    if sparse_score.empty:
        raise ValueError(
            "no independent-evaluation row makes fitted valuation uncertainty "
            "consequential at the sparse-support initial list price"
        )
    sparse_position = int(sparse_score.idxmax())
    return {
        "healthy-demand-margin-protection": healthy_position,
        "stale-inventory-high-carry": stale_position,
        "sparse-support-downside-protection": sparse_position,
    }


def _project_model_features(
    row: pd.Series,
    features: Sequence[str],
    *,
    replacements: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    replacement_values = replacements or {}
    values: dict[str, Any] = {}
    for feature in features:
        if feature in replacement_values:
            values[feature] = replacement_values[feature]
        elif feature in row.index:
            values[feature] = row[feature]
        else:
            raise KeyError(f"representative row is missing model feature {feature!r}")
    forbidden = [name for name in values if name.startswith(("true_", "truth_"))]
    if forbidden:
        raise ValueError(f"truth features cannot enter a decision adapter: {forbidden}")
    return pd.DataFrame([values], columns=list(features))


def _standardized(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    scale = float(numeric.std(ddof=0))
    if scale == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.mean())) / scale


def _write_figures(plot_data: Mapping[str, Any], paths: Sequence[Path]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#9eaaa9",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    value_target = np.asarray(plot_data["value_target"], dtype=float)
    value_prediction = np.asarray(plot_data["value_prediction"], dtype=float)
    sample_size = min(500, len(value_target))
    sample_positions = np.linspace(0, len(value_target) - 1, sample_size, dtype=int)
    lower = min(value_target[sample_positions].min(), value_prediction[sample_positions].min())
    upper = max(value_target[sample_positions].max(), value_prediction[sample_positions].max())

    figure, axis = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    axis.scatter(
        value_target[sample_positions] / 1_000,
        value_prediction[sample_positions] / 1_000,
        s=17,
        alpha=0.55,
        color="#087e78",
        edgecolors="none",
    )
    axis.plot([lower / 1_000, upper / 1_000], [lower / 1_000, upper / 1_000], "--", color="#db694d")
    axis.set(
        title="Contemporaneous value anchor on a shifted environment",
        xlabel="Observed generated price ($000s)",
        ylabel="Predicted price ($000s)",
    )
    axis.text(
        0.02, 0.98, "Controlled experiment", transform=axis.transAxes, va="top", color="#5d6d74"
    )
    _save_figure(figure, paths[0])

    evaluation: SimulatedEnvironment = plot_data["evaluation"]
    acceptance_model: SellerAcceptanceModel = plot_data["acceptance_model"]
    truth: CausalParameters = plot_data["truth"]
    acceptance_sample = evaluation.offers.iloc[: min(300, len(evaluation.offers))].copy()
    ratios = np.linspace(*OFFER_RATIO_SUPPORT, 31)
    fitted_curve: list[float] = []
    truth_curve: list[float] = []
    for ratio in ratios:
        counterfactual = acceptance_sample.copy()
        counterfactual["offer_ratio"] = ratio
        fitted_curve.append(float(acceptance_model.predict_proba(counterfactual).mean()))
        truth_curve.append(float(true_acceptance_probability(counterfactual, truth).mean()))
    figure, axis = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    axis.plot(ratios, truth_curve, color="#14232b", linewidth=2.4, label="Simulator truth")
    axis.plot(
        ratios,
        fitted_curve,
        color="#087e78",
        linewidth=2.4,
        linestyle="--",
        label="Fitted logistic",
    )
    axis.set(
        title="Seller acceptance response",
        xlabel="Offer / observable pre-offer reference",
        ylabel="Acceptance probability",
        ylim=(0, 1),
    )
    axis.legend(frameon=False, loc="upper left")
    _save_figure(figure, paths[1])

    hazard_model: DiscreteTimeHazardModel = plot_data["hazard_model"]
    hazard_probe = evaluation.listings.iloc[[0]].copy()
    periods = np.arange(1, hazard_model.max_periods + 1)
    figure, axis = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    for premium, label, color in (
        (-0.02, "2% below reference", "#087e78"),
        (0.05, "5% above reference", "#d99a3b"),
        (0.12, "12% above reference", "#db694d"),
    ):
        counterfactual = hazard_probe.copy()
        counterfactual["list_price_premium"] = premium
        fitted_hazard = hazard_model.predict_hazard(counterfactual)[0]
        sell_through = 1.0 - np.cumprod(1.0 - fitted_hazard)
        axis.plot(periods, sell_through, marker="o", markersize=3, color=color, label=label)
    axis.set(
        title="List price changes cumulative sell-through",
        xlabel="Weekly listing period",
        ylabel="Predicted cumulative sell-through",
        ylim=(0, 1),
    )
    axis.legend(frameon=False, loc="lower right")
    _save_figure(figure, paths[2])


def _save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        facecolor="white",
        metadata={"Software": f"offer-to-exit {__version__}"},
    )
    plt.close(figure)


def _write_decision_csv(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "name",
        "question",
        "home_id",
        "reference_value",
        "preoffer_reference_value",
        "prelisting_reference_value",
        "initial_list_price",
        "initial_list_price_premium",
        "recommendation",
        "recommendation_reason",
        "selected_offer",
        "automated_offer",
        "selected_offer_to_value_anchor_ratio",
        "acceptance_offer_ratio",
        "scored_list_price_premium_min",
        "scored_list_price_premium_max",
        "scored_list_price_premium_count",
        "acceptance_probability",
        "first_action",
        "first_post_action_list_price",
        "expected_profit_per_lead",
        "worst_tail_positive_loss",
        "loss_probability",
        "risk_adjusted_objective",
        "evidence_scope",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cases)


def _artifact_paths(output_dir: Path) -> dict[str, Any]:
    return {
        "manifest": output_dir / f"run_manifest.{ARTIFACT_VERSION}.json",
        "metrics": output_dir / f"metrics.{ARTIFACT_VERSION}.json",
        "decision_cases": output_dir / f"decision_cases.{ARTIFACT_VERSION}.csv",
        "summary": output_dir / f"summary.{ARTIFACT_VERSION}.json",
        "figures": [
            output_dir / f"valuation_evaluation.{ARTIFACT_VERSION}.png",
            output_dir / f"acceptance_response.{ARTIFACT_VERSION}.png",
            output_dir / f"hazard_sell_through.{ARTIFACT_VERSION}.png",
        ],
        "demo": output_dir / "demo.html",
    }


def _guard_output_targets(paths: Mapping[str, Any], *, overwrite: bool) -> None:
    targets = [
        path for value in paths.values() for path in (value if isinstance(value, list) else [value])
    ]
    existing = [Path(path) for path in targets if Path(path).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing run artifacts: {names}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _binary_log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    observed = np.asarray(target, dtype=float)
    return float(-np.mean(observed * np.log(clipped) + (1.0 - observed) * np.log(1.0 - clipped)))


def _calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    observed = np.asarray(target, dtype=float)
    predicted = np.asarray(probability, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(pairwise(boundaries)):
        in_bin = (predicted >= lower) & (
            predicted <= upper if index == bins - 1 else predicted < upper
        )
        if in_bin.any():
            error += float(in_bin.mean()) * abs(
                float(predicted[in_bin].mean()) - float(observed[in_bin].mean())
            )
    return error


def _log_odds(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    return math.log(clipped / (1.0 - clipped))


def _rounded(value: float | np.floating[Any], digits: int) -> float:
    return round(float(value), digits)


def _slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part) or "run"


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _metric_card(value: str, label: str, context: str) -> str:
    return (
        '<article class="metric">'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="context">{html.escape(context)}</div>'
        "</article>"
    )


def _case_html(case: Mapping[str, Any]) -> str:
    path_value = case.get("no_sale_path", [])
    actions = []
    if isinstance(path_value, Sequence) and not isinstance(path_value, str):
        actions = [
            f"W{item.get('week', '—')}: {_display_action(item.get('action', '—'))}"
            for item in path_value
            if isinstance(item, Mapping)
        ]
    path_text = " → ".join(actions)
    recommendation = str(case.get("recommendation", "review"))
    offer_label = "automated offer" if recommendation == "price" else "best grid candidate"
    return f"""
    <details class="case">
      <summary>
        <div class="case-kicker">{html.escape(recommendation.upper())}</div>
        <h3>{html.escape(str(case.get("name", "Worked case")).replace("-", " ").title())}</h3>
        <div class="case-question">{html.escape(str(case.get("question", "")))}</div>
      </summary>
      <div class="case-grid">
        <div class="case-stat"><b>{_money(case.get("selected_offer", 0))}</b><span>{offer_label}</span></div>
        <div class="case-stat"><b>{_percent(case.get("acceptance_probability", 0))}</b><span>acceptance probability</span></div>
        <div class="case-stat"><b>{_money(case.get("expected_profit_per_lead", 0))}</b><span>expected profit / lead</span></div>
        <div class="case-stat"><b>{_percent(case.get("loss_probability", 0))}</b><span>approx. loss probability</span></div>
      </div>
      <div class="path"><strong>Decision gate:</strong> {html.escape(str(case.get("recommendation_reason", "")))}</div>
      <div class="path"><strong>No-sale path:</strong> {html.escape(path_text)}</div>
    </details>
    """


def _display_action(value: Any) -> str:
    action = str(value)
    labels = {
        "hold": "hold",
        "cut_1_pct": "cut 1%",
        "cut_2_5_pct": "cut 2.5%",
        "cut_5_pct": "cut 5%",
    }
    return labels.get(action, action.replace("_", " "))


def _money(value: Any) -> str:
    return f"${float(value):,.0f}"


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _decimal(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    return f"{int(value):,}"
