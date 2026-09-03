"""Versioned evidence bundle for the Tampa development and Orlando holdout study."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from offer_to_exit import __version__
from offer_to_exit.data.catalog import SOURCES
from offer_to_exit.empirical import (
    ORLANDO_MARKET,
    TAMPA_MARKET,
    build_episode_panel,
    build_valuation_panel,
    parcel_weighted_valuation_metrics,
    run_geographic_exit_study,
    run_geographic_valuation_study,
    summarize_ibuyer_episodes,
)

FLORIDA_ARTIFACT_SCHEMA_VERSION = 2
FLORIDA_RELEASE_VERSION = "v2"
FLORIDA_CLAIM_SCOPE = (
    "County deeds support prediction of recorded prices and descriptive ownership duration. "
    "They do not reveal rejected offers, weekly "
    "list-price paths, repairs, operating costs, days on market, or causal price response."
)


def run_florida_release(
    transactions_path: Path,
    episodes_path: Path,
    output_dir: Path,
    *,
    minimum_date: str = "2010-01-01",
    maximum_years_since_prior_sale: float = 4.5,
    valuation_calibration_start: str = "2022-01-01",
    valuation_test_start: str = "2024-01-01",
    duration_calibration_start: str = "2021-01-01",
    duration_test_start: str = "2022-01-01",
    max_duration_weeks: int = 52,
    random_state: int = 20260903,
    raw_manifest_path: Path | None = Path("data/raw/manifest.json"),
) -> dict[str, Any]:
    """Fit the real-data studies and write a small, privacy-safe evidence bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    transactions, source_counts, selection_audit = _load_analysis_transactions(
        transactions_path, minimum_date=minimum_date
    )
    valuation_panel = build_valuation_panel(transactions, require_qualified=False)
    # The county-common valuation estimand is deliberately restricted to repeat
    # sales. Hillsborough's all-sales file does not contain the structural
    # attributes needed for a defensible cross-sectional AVM. Conditioning on a
    # prior deed makes the property's own lagged price the quality control and
    # defines a transparent, reproducible target population in both counties.
    valuation_panel = valuation_panel.loc[
        valuation_panel["prior_sale_price"].gt(0)
        & valuation_panel["prior_market_median_price"].gt(0)
        & valuation_panel["lagged_market_median_price"].gt(0)
        & valuation_panel["rolled_forward_prior_baseline"].gt(0)
        & valuation_panel["years_since_prior_sale"].between(
            0, maximum_years_since_prior_sale, inclusive="both"
        )
    ].copy()
    valuation = run_geographic_valuation_study(
        valuation_panel,
        categorical_features=(),
        calibration_start=valuation_calibration_start,
        test_start=valuation_test_start,
        random_state=random_state,
    )
    seasoned_valuation_panel = valuation_panel.loc[
        valuation_panel["years_since_prior_sale"].ge(30 / 365.25)
    ].copy()
    seasoned_valuation = run_geographic_valuation_study(
        seasoned_valuation_panel,
        categorical_features=(),
        calibration_start=valuation_calibration_start,
        test_start=valuation_test_start,
        random_state=random_state,
    )

    raw_episodes = pd.read_csv(
        episodes_path,
        low_memory=False,
        dtype={
            "parcel_id": "string",
            "market": "string",
            "operator": "string",
            "property_type_code": "string",
            "dor_code": "string",
        },
    )
    raw_episodes["acquisition_date"] = pd.to_datetime(
        raw_episodes["acquisition_date"], format="mixed", errors="coerce"
    )
    episode_status_counts = _status_counts(raw_episodes)
    linked_episode_counts = _market_counts(raw_episodes)
    episode_panel = build_episode_panel(raw_episodes)
    episode_panel = episode_panel.loc[
        episode_panel["acquisition_date"].ge(pd.Timestamp("2016-01-01"))
    ].copy()
    episode_panel, supported_operators = _filter_duration_population(episode_panel)
    duration = run_geographic_exit_study(
        episode_panel,
        max_weeks=max_duration_weeks,
        calibration_start=duration_calibration_start,
        test_start=duration_test_start,
        random_state=random_state,
    )

    common_duration_panel = episode_panel.loc[
        episode_panel["acquisition_date"].ge(duration.tampa_temporal_split.test_start)
    ].copy()
    operator_summary = summarize_ibuyer_episodes(common_duration_panel)
    operator_summary = _serialize_dates(operator_summary)
    operator_summary = operator_summary.drop(
        columns=["median_gross_spread", "median_gross_return"], errors="ignore"
    )
    operator_effects = duration.model.operator_effects()
    valuation_payload = _round_nested(valuation.metrics_payload())
    seasoned_payload = seasoned_valuation.metrics_payload()
    valuation_payload["minimum_30_day_prior_gap_sensitivity"] = _round_nested(
        {
            "rule": (
                "Require at least 30 days between the prior and target deed in addition to "
                "requiring the prior deed to fall in an earlier calendar quarter."
            ),
            "panel_rows": _market_counts(seasoned_valuation_panel),
            "tampa_out_of_time": seasoned_payload["tampa_out_of_time"],
            "tampa_rolled_prior_baseline": seasoned_payload["tampa_rolled_prior_baseline"],
            "orlando_common_window": seasoned_payload["orlando_common_window"],
            "orlando_rolled_prior_baseline": seasoned_payload["orlando_rolled_prior_baseline"],
        }
    )
    observed_qualified_orlando = valuation.orlando_holdout_predictions.loc[
        _boolean_series(valuation.orlando_holdout_predictions.get("qualified"))
    ].copy()
    if not observed_qualified_orlando.empty:
        observed_qualified_dates = pd.to_datetime(observed_qualified_orlando["sale_date"])
        valuation_payload["orlando_observed_qualified_sensitivity"] = _round_nested(
            asdict(
                parcel_weighted_valuation_metrics(
                    observed_qualified_orlando,
                    actual_col="sale_price",
                    parcel_col="parcel_id",
                )
            )
        )
        valuation_payload["orlando_observed_qualified_window"] = {
            "start": observed_qualified_dates.min().date().isoformat(),
            "end": observed_qualified_dates.max().date().isoformat(),
            "interpretation": (
                "This diagnostic changes both qualification observability and calendar period; "
                "it does not isolate warranty-deed proxy error."
            ),
        }
        valuation_payload["orlando_observed_qualified_baseline"] = _round_nested(
            asdict(
                parcel_weighted_valuation_metrics(
                    observed_qualified_orlando,
                    actual_col="sale_price",
                    prediction_col="rolled_forward_prior_baseline",
                    parcel_col="parcel_id",
                    lower_col=None,
                    upper_col=None,
                )
            )
        )
    duration_payload = _round_nested(duration.metrics_payload())
    opendoor_summary = operator_summary.loc[operator_summary["operator"].eq("opendoor")].to_dict(
        orient="records"
    )

    valuation_figure = output_dir / "florida_valuation_transport.v2.png"
    duration_figure = output_dir / "florida_inventory_duration.v2.png"
    _write_valuation_figure(valuation, valuation_figure)
    _write_duration_figure(common_duration_panel, duration_figure)

    run_id = f"florida-{datetime.now(tz=UTC).date().isoformat()}-schema-2"
    preparation_analysis = _preparation_analysis(
        transactions_path.parent / "preparation_manifest.json"
    )
    analysis_provenance: dict[str, object] = {
        "random_state": random_state,
        "minimum_sale_date": minimum_date,
        "maximum_years_since_prior_sale": maximum_years_since_prior_sale,
        "seasoning_sensitivity_minimum_prior_gap_days": 30,
        "valuation_calibration_start": valuation_calibration_start,
        "valuation_test_start": valuation_test_start,
        "duration_calibration_start": duration_calibration_start,
        "duration_test_start": duration_test_start,
        "max_duration_weeks": max_duration_weeks,
    }
    analysis_provenance.update(preparation_analysis)
    payload: dict[str, Any] = {
        "schema_version": FLORIDA_ARTIFACT_SCHEMA_VERSION,
        "release_version": FLORIDA_RELEASE_VERSION,
        "package_version": __version__,
        "run_id": run_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "claim_scope": FLORIDA_CLAIM_SCOPE,
        "provenance": {
            "source_snapshot": _source_provenance(raw_manifest_path),
            "analysis": analysis_provenance,
        },
        "design": {
            "development_market": "Hillsborough County (Tampa), Florida",
            "external_holdout": "Orange County (Orlando), Florida",
            "valuation_target": "recorded sale price",
            "valuation_population": (
                "high-turnover improved single-family repeat sales whose prior eligible sale "
                f"occurred no more than {maximum_years_since_prior_sale:g} years earlier"
            ),
            "valuation_qualification_rule": (
                "Hillsborough county-qualified deeds; Orange county-qualified deeds when the "
                "description is present, otherwise warranty deeds as a documented proxy"
            ),
            "valuation_normalization": (
                "prior eligible sale price relative to its county-quarter median; the prior "
                "eligible sale must be in an earlier quarter; prediction scaled by the lagged "
                "current-quarter county median"
            ),
            "external_evaluation_status": (
                "Orange is excluded from fitting and interval calibration. This is an "
                "external-market evaluation, not a preregistered untouched holdout."
            ),
            "duration_target": "weeks from recorded acquisition deed to recorded resale deed",
            "duration_horizon_weeks": max_duration_weeks,
            "linked_operator_registry": [
                "Opendoor",
                "Offerpad",
                "Zillow Offers",
                "RedfinNow",
            ],
            "duration_model_operators": [
                operator.replace("_", " ").title() for operator in supported_operators
            ],
            "duration_population": (
                "improved single-family acquisition spells for Opendoor and Offerpad; model "
                "training uses historical Tampa spells and descriptive comparisons use the "
                "common January 2022 onward window; deed qualification is not required"
            ),
        },
        "data": {
            "raw_safe_transaction_rows": source_counts,
            "eligible_market_rows": _market_counts(transactions),
            "valuation_panel_rows": _market_counts(valuation_panel),
            "valuation_selection_audit": selection_audit,
            "valuation_prior_gap_years": _numeric_market_summary(
                valuation_panel, "years_since_prior_sale"
            ),
            "linked_episode_rows": linked_episode_counts,
            "duration_model_population_rows": _market_counts(episode_panel, market_col="market"),
            "duration_common_window_rows": _market_counts(
                common_duration_panel, market_col="market"
            ),
            "episode_linkage_status": episode_status_counts,
            "transactions_sha256": _sha256_file(transactions_path),
            "episodes_sha256": _sha256_file(episodes_path),
        },
        "valuation": valuation_payload,
        "inventory_duration": duration_payload,
        "opendoor_slice": _round_nested(opendoor_summary),
        "operator_summary": _round_nested(operator_summary.to_dict(orient="records")),
        "operator_effects": _round_nested(operator_effects.to_dict(orient="records")),
        "artifacts": {
            "metrics": "florida_metrics.v2.json",
            "operator_summary": "florida_operator_summary.v2.csv",
            "operator_effects": "florida_operator_effects.v2.csv",
            "valuation_figure": valuation_figure.name,
            "duration_figure": duration_figure.name,
            "report": "florida_evidence.html",
            "manifest": "florida_manifest.v2.json",
        },
    }

    metrics_path = output_dir / "florida_metrics.v2.json"
    summary_path = output_dir / "florida_operator_summary.v2.csv"
    effects_path = output_dir / "florida_operator_effects.v2.csv"
    report_path = output_dir / "florida_evidence.html"
    manifest_path = output_dir / "florida_manifest.v2.json"
    _write_json(metrics_path, payload)
    operator_summary.to_csv(summary_path, index=False)
    operator_effects.to_csv(effects_path, index=False)
    render_florida_report(payload, report_path)

    emitted = [
        metrics_path,
        summary_path,
        effects_path,
        valuation_figure,
        duration_figure,
        report_path,
    ]
    manifest = {
        "schema_version": FLORIDA_ARTIFACT_SCHEMA_VERSION,
        "release_version": FLORIDA_RELEASE_VERSION,
        "run_id": run_id,
        "package_version": __version__,
        "provenance": payload["provenance"],
        "source": {
            "transactions_sha256": payload["data"]["transactions_sha256"],
            "episodes_sha256": payload["data"]["episodes_sha256"],
        },
        "artifact_sha256": {path.name: _sha256_file(path) for path in emitted},
    }
    _write_json(manifest_path, manifest)
    return payload


def render_florida_report(payload: dict[str, Any], output: Path) -> Path:
    """Render a self-contained public report with no transaction-level records."""

    data = payload["data"]
    valuation = payload["valuation"]
    duration = payload["inventory_duration"]
    orlando_value = valuation["orlando_common_window"]
    orlando_baseline = valuation["orlando_rolled_prior_baseline"]
    orlando_duration = duration["orlando_common_window"]
    tampa_duration = duration["tampa_out_of_time"]
    opendoor_rows = payload.get("opendoor_slice", [])

    transaction_total = sum(int(value) for value in data["raw_safe_transaction_rows"].values())
    episode_total = sum(int(value) for value in data["linked_episode_rows"].values())
    duration_common_window_total = sum(
        int(value) for value in data["duration_common_window_rows"].values()
    )
    valuation_gain = 1 - (
        float(orlando_value["mean_absolute_error"]) / float(orlando_baseline["mean_absolute_error"])
    )
    valuation_comparison = (
        f"{_percent(abs(valuation_gain))} lower than the rolled-forward prior-sale baseline"
        if valuation_gain >= 0
        else f"{_percent(abs(valuation_gain))} higher than the rolled-forward prior-sale baseline"
    )
    observation_ends = (
        payload.get("provenance", {})
        .get("analysis", {})
        .get("episode_observation_end_by_market", {})
    )
    observation_end_note = ""
    if isinstance(observation_ends, dict) and observation_ends:
        tampa_end = html.escape(str(observation_ends.get(TAMPA_MARKET, "not reported")))
        orlando_end = html.escape(str(observation_ends.get(ORLANDO_MARKET, "not reported")))
        observation_end_note = (
            f" Open spells are censored at each county's observation end: Tampa {tampa_end} "
            f"and Orlando {orlando_end}."
        )
    qualified_sensitivity = valuation.get("orlando_observed_qualified_sensitivity")
    qualified_baseline = valuation.get("orlando_observed_qualified_baseline")
    sensitivity_note = ""
    if isinstance(qualified_sensitivity, dict) and isinstance(qualified_baseline, dict):
        sensitivity_window = valuation.get("orlando_observed_qualified_window", {})
        sensitivity_start = str(sensitivity_window.get("start", "the reported start date"))
        sensitivity_end = str(sensitivity_window.get("end", "the reported end date"))
        sensitivity_note = (
            '<div class="boundary"><strong>Orange qualification sensitivity</strong>'
            f"The subset with an observed county-qualified description contains "
            f"{int(qualified_sensitivity['n_transactions']):,} test transactions. Its model "
            f"MAE is {_money(qualified_sensitivity['mean_absolute_error'])} versus "
            f"{_money(qualified_baseline['mean_absolute_error'])} for the rolled-prior benchmark. "
            f"The subset spans {html.escape(sensitivity_start)} through "
            f"{html.escape(sensitivity_end)}, so it changes both qualification observability "
            "and calendar period rather than isolating proxy error.</div>"
        )
    seasoning = valuation.get("minimum_30_day_prior_gap_sensitivity")
    seasoning_note = ""
    if isinstance(seasoning, dict):
        seasoned_tampa = seasoning["tampa_out_of_time"]
        seasoned_tampa_baseline = seasoning["tampa_rolled_prior_baseline"]
        seasoned_orlando = seasoning["orlando_common_window"]
        seasoned_orlando_baseline = seasoning["orlando_rolled_prior_baseline"]
        tampa_gain = 1 - float(seasoned_tampa["mean_absolute_error"]) / float(
            seasoned_tampa_baseline["mean_absolute_error"]
        )
        orlando_gain = 1 - float(seasoned_orlando["mean_absolute_error"]) / float(
            seasoned_orlando_baseline["mean_absolute_error"]
        )
        seasoning_note = (
            '<div class="boundary"><strong>Repeat-sale seasoning sensitivity</strong>'
            "The main design already excludes prior deeds from the target sale's quarter. "
            "Requiring an additional 30-day gap leaves "
            f"{int(seasoned_tampa['n_transactions']):,} Tampa and "
            f"{int(seasoned_orlando['n_transactions']):,} Orlando test transactions. "
            f"Model MAE remains {_percent(tampa_gain)} below the rolled-prior benchmark "
            f"in Tampa and {_percent(orlando_gain)} below it in Orlando.</div>"
        )
    opendoor_cards = "".join(_opendoor_card(row) for row in opendoor_rows if isinstance(row, dict))
    operator_rows = "".join(
        _operator_table_row(row) for row in _read_operator_summary_for_report(payload)
    )
    valuation_image = _image_data_uri(output.parent / payload["artifacts"]["valuation_figure"])
    duration_image = _image_data_uri(output.parent / payload["artifacts"]["duration_figure"])
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Offer-to-Exit · Florida Evidence</title>
  <style>
    :root {{ --ink:#17262d; --muted:#5f6d72; --paper:#f4f1e9; --card:#fffdf8;
      --teal:#087e78; --coral:#d66a4d; --line:#d8d5cc; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink);
      font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      line-height:1.55 }} .top {{ height:7px; background:linear-gradient(90deg,var(--teal),#25a69c,var(--coral)) }}
    main {{ width:min(1120px,calc(100% - 32px)); margin:auto; padding:52px 0 76px }}
    .hero,.card,figure,.boundary {{ background:var(--card); border:1px solid var(--line);
      border-radius:20px }} .hero {{ padding:clamp(28px,6vw,68px) }} .eyebrow {{ color:var(--teal);
      font-size:.76rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase }}
    h1 {{ margin:12px 0 18px; max-width:900px; font-size:clamp(2.5rem,6vw,5.2rem);
      line-height:.96; letter-spacing:-.055em }} h2 {{ margin:0; font-size:clamp(1.65rem,3.5vw,2.6rem);
      letter-spacing:-.035em }} h3 {{ margin:0 0 8px }} p {{ color:var(--muted) }} .lede {{ max-width:760px;
      font-size:1.08rem }} section {{ margin-top:54px }} .section-head {{ display:grid;
      grid-template-columns:.8fr 1.2fr; gap:28px; align-items:end; margin-bottom:20px }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:13px }} .card {{ padding:21px }}
    .value {{ font-size:1.65rem; font-weight:800; letter-spacing:-.04em }} .label {{ color:var(--muted);
      font-size:.75rem; margin-top:5px }} .context {{ color:var(--teal); font-size:.7rem;
      font-weight:700; margin-top:10px }} .figures {{ display:grid; grid-template-columns:1fr 1fr; gap:16px }}
    figure {{ margin:0; overflow:hidden }} figure img {{ display:block; width:100%; background:white }}
    figcaption {{ padding:12px 15px; color:var(--muted); font-size:.75rem }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line) }}
    th,td {{ padding:12px 14px; border-bottom:1px solid var(--line); text-align:left; font-size:.82rem }}
    th {{ color:var(--teal); text-transform:uppercase; letter-spacing:.07em; font-size:.68rem }}
    .operator-scroll {{ overflow-x:auto; border-radius:16px }} .boundary {{ padding:24px 27px;
      border-left:5px solid var(--coral) }} .boundary strong {{ display:block; margin-bottom:5px }}
    .boundary + .boundary {{ margin-top:12px }}
    .operator-cards {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:16px }}
    footer {{ margin-top:26px; color:var(--muted); font-size:.72rem }}
    @media(max-width:800px) {{ .section-head,.figures {{ grid-template-columns:1fr }}
      .metrics {{ grid-template-columns:1fr 1fr }} }} @media(max-width:520px) {{ .metrics,.operator-cards {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body><div class="top"></div><main>
  <header class="hero"><div class="eyebrow">Real Florida transaction evidence · v2</div>
    <h1>Tampa for development.<br>Orlando for the test.</h1>
    <p class="lede">Offer-to-Exit reconstructs named iBuyer ownership spells from county deeds,
    fits only on Hillsborough County, and measures what transports to Orange County without
    refitting on Orlando outcomes.</p></header>

  <section><div class="section-head"><h2>Evidence at a glance</h2><p>Valuation errors give each parcel
    equal metric weight. Hazard fit and evaluation use the ordinary person-period likelihood.
    Kaplan-Meier summaries use ownership episodes.</p></div>
    <div class="metrics">
      {_metric(f"{transaction_total:,}", "privacy-safe deed rows", "Two official county systems")}
      {_metric(f"{episode_total:,}", "linked iBuyer ownership spells", f"{duration_common_window_total:,} common-window descriptive episodes")}
      {_metric(_money(orlando_value["mean_absolute_error"]), "Orlando repeat-sale MAE", valuation_comparison)}
      {_metric(_decimal(orlando_duration["risk_row_auc"], 3), "Orlando disposition-hazard AUC", "Tampa-only specification; no Orlando refit")}
    </div></section>

  <section><div class="section-head"><h2>Geographic transport</h2><p>The repeat-sale valuation
    exercise uses the prior eligible sale price relative to its own county-quarter market median, elapsed
    time, and sale quarter. The prediction is scaled by the lagged current-quarter median. The
    duration model uses acquisition-time information and operator indicators.</p></div>
    <div class="figures"><figure><img src="{valuation_image}" alt="Orlando recorded prices versus Tampa-model predictions">
      <figcaption>Tampa-only repeat-sale model applied to the Orlando common-window test without refitting.</figcaption></figure>
      <figure><img src="{duration_image}" alt="Kaplan-Meier inventory exit curves for Tampa and Orlando">
      <figcaption>Recorded acquisition-to-resale duration, which is not MLS days on market.</figcaption></figure></div>
    {sensitivity_note}{seasoning_note}
  </section>

  <section><div class="section-head"><h2>The Opendoor slice</h2><p>Opendoor remains visible as the
    motivating operator, while Offerpad enlarges the common-market empirical support for
    the shared inventory problem. Operator comparisons are descriptive.</p></div>
    <div class="operator-cards">{opendoor_cards}</div></section>

  <section><div class="section-head"><h2>Named intermediary episodes</h2><p>The comparable duration
    sample requires an improved single-family home, operator support in both counties, and an
    acquisition in January 2022 or later. Deed-to-deed duration is not MLS days on market.{observation_end_note}</p></div>
    <div class="operator-scroll"><table><thead><tr><th>Market</th><th>Operator</th><th>Acquisitions</th>
    <th>Observed exits</th><th>KM exit by 120d</th><th>Median observed hold</th></tr></thead>
    <tbody>{operator_rows}</tbody></table></div></section>

  <section><div class="section-head"><h2>What transported</h2><p>An external-market test is a hard
    test of a Tampa-only specification, not proof of national generalizability.</p></div>
    <div class="metrics">
      {_metric(_money(orlando_baseline["mean_absolute_error"]), "Orlando baseline MAE", "Prior sale rolled forward with the county market")}
      {_metric(_percent(orlando_value["interval_coverage"]), "Orlando 90% interval coverage", "Coverage under geographic shift")}
      {_metric(_decimal(tampa_duration["brier_skill_score"], 3), "Tampa future Brier skill", "Against constant training hazard")}
      {_metric(_decimal(orlando_duration["brier_skill_score"], 3), "Orlando Brier skill", "Same Tampa-only model and time window")}
    </div></section>

  <section class="boundary"><strong>Identification boundary</strong>{html.escape(str(payload["claim_scope"]))}
    The separate randomized policy experiment identifies response only inside its known data-generating
    process. No Florida seller or buyer elasticity is claimed here.</section>
  <footer>{html.escape(str(payload["run_id"]))} · offer-to-exit {html.escape(str(payload["package_version"]))} ·
    aggregate evidence only; no names, addresses, coordinates, or raw parcel identifiers.</footer>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output


def _load_analysis_transactions(
    path: Path, *, minimum_date: str
) -> tuple[pd.DataFrame, dict[str, int], dict[str, dict[str, int]]]:
    retained: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    selection_audit: dict[str, dict[str, int]] = {}
    cutoff = pd.Timestamp(minimum_date)
    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
        low_memory=False,
        dtype={
            "parcel_id": "string",
            "market": "string",
            "property_type_code": "string",
            "dor_code": "string",
        },
    ):
        chunk["market"] = chunk["market"].astype("string")
        for market, count in chunk["market"].value_counts(dropna=False).items():
            key = "missing" if pd.isna(market) else str(market)
            source_counts[key] = source_counts.get(key, 0) + int(count)
        chunk["sale_date"] = pd.to_datetime(chunk["sale_date"], format="mixed", errors="coerce")
        chunk["sale_price"] = pd.to_numeric(chunk["sale_price"], errors="coerce")
        qualified = _nullable_boolean_series(chunk.get("qualified"))
        improved = _boolean_series(chunk.get("improved"))
        dor = chunk.get("dor_code", pd.Series("", index=chunk.index)).astype("string")
        property_type = chunk.get("property_type_code", pd.Series("", index=chunk.index)).astype(
            "string"
        )
        single_family = dor.str.startswith("01", na=False) | property_type.str.contains(
            "single family", case=False, na=False
        )
        instrument = chunk.get("instrument_type", pd.Series("", index=chunk.index)).astype("string")
        base_valid = (
            chunk["sale_date"].ge(cutoff)
            & chunk["sale_price"].between(25_000, 10_000_000, inclusive="both")
            & improved
            & single_family
        )
        official_qualified = qualified.eq(True).fillna(False)
        orange_warranty_proxy = (
            chunk["market"].eq(ORLANDO_MARKET)
            & qualified.isna()
            & instrument.str.strip().str.upper().eq("WD")
        ).fillna(False)
        valid = base_valid & (official_qualified.astype(bool) | orange_warranty_proxy.astype(bool))
        for market in (TAMPA_MARKET, ORLANDO_MARKET):
            market_mask = chunk["market"].eq(market)
            audit = selection_audit.setdefault(
                market,
                {
                    "date_price_improved_single_family": 0,
                    "officially_qualified": 0,
                    "orange_warranty_deed_proxy": 0,
                    "selected": 0,
                },
            )
            audit["date_price_improved_single_family"] += int((market_mask & base_valid).sum())
            audit["officially_qualified"] += int(
                (market_mask & base_valid & official_qualified).sum()
            )
            audit["orange_warranty_deed_proxy"] += int(
                (market_mask & base_valid & orange_warranty_proxy).sum()
            )
            audit["selected"] += int((market_mask & valid).sum())
        if valid.any():
            selected = chunk.loc[valid].copy()
            selected["property_type_code"] = "single_family"
            retained.append(selected)
    if not retained:
        raise ValueError("no eligible Florida market transactions were found")
    return pd.concat(retained, ignore_index=True), source_counts, selection_audit


def _filter_duration_population(
    episodes: pd.DataFrame,
    *,
    minimum_operator_market_episodes: int = 25,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Define a comparable single-family duration population in both counties."""

    improved = _boolean_series(episodes.get("improved"))
    dor = episodes.get("dor_code", pd.Series("", index=episodes.index)).astype("string")
    property_type = episodes.get("property_type_code", pd.Series("", index=episodes.index)).astype(
        "string"
    )
    single_family = dor.str.startswith("01", na=False) | property_type.str.contains(
        "single family", case=False, na=False
    )
    comparable = episodes.loc[improved & single_family].copy()
    support = comparable.groupby(["operator", "market"], observed=True).size().unstack(fill_value=0)
    required_markets = {TAMPA_MARKET, ORLANDO_MARKET}
    for market in required_markets.difference(support.columns):
        support[market] = 0
    supported = tuple(
        sorted(
            support.index[
                support.loc[:, sorted(required_markets)]
                .min(axis=1)
                .ge(minimum_operator_market_episodes)
            ].astype(str)
        )
    )
    if not supported:
        raise ValueError("no operator has sufficient duration support in both Florida markets")
    return comparable.loc[comparable["operator"].isin(supported)].copy(), supported


def _boolean_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        raise ValueError("Florida transactions must contain qualification and improvement flags")
    return _nullable_boolean_series(values).fillna(False).astype(bool)


def _nullable_boolean_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        raise ValueError("Florida transactions must contain qualification and improvement flags")
    normalized = values.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result.loc[normalized.isin({"1", "true", "t", "yes", "y", "qualified", "improved"})] = True
    result.loc[normalized.isin({"0", "false", "f", "no", "n", "unqualified", "vacant"})] = False
    return result


def _numeric_market_summary(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for market, group in frame.groupby("market", observed=True):
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        result[str(market)] = {
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "maximum": float(values.max()),
        }
    return result


def _source_provenance(path: Path | None) -> dict[str, dict[str, object]]:
    """Return only the public provenance fields for the two used county sources."""

    local_files: dict[str, object] = {}
    if path is not None and path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        raw_files = loaded.get("files", {}) if isinstance(loaded, dict) else {}
        if isinstance(raw_files, dict):
            local_files = raw_files

    result: dict[str, dict[str, object]] = {}
    for key in ("hillsborough_sales", "orange_sales"):
        source = SOURCES[key]
        record = local_files.get(key, {})
        record = record if isinstance(record, dict) else {}
        public: dict[str, object] = {
            "title": source.title,
            "publisher": source.publisher,
            "landing_page": source.landing_page,
            "transport": source.transport,
        }
        for field in ("url", "retrieved_at", "sha256", "bytes", "records"):
            value = record.get(field)
            if value is not None:
                public[field] = value
        result[key] = public
    return result


def _preparation_analysis(path: Path) -> dict[str, object]:
    """Read non-sensitive episode-construction provenance when available."""

    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    analysis = loaded.get("analysis", {}) if isinstance(loaded, dict) else {}
    return dict(analysis) if isinstance(analysis, dict) else {}


def _market_counts(frame: pd.DataFrame, *, market_col: str = "market") -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[market_col].value_counts(dropna=False).sort_index().items()
    }


def _status_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    if "linkage_status" not in frame:
        return {}
    grouped = frame.groupby(["market", "linkage_status"], dropna=False).size()
    result: dict[str, dict[str, int]] = {}
    for (market, status), count in grouped.items():
        result.setdefault(str(market), {})[str(status)] = int(count)
    return result


def _write_valuation_figure(study: Any, path: Path) -> None:
    plt.switch_backend("Agg")
    frame = study.orlando_holdout_predictions
    positions = np.linspace(0, len(frame) - 1, min(2_000, len(frame)), dtype=int)
    actual = frame.iloc[positions]["sale_price"].to_numpy(dtype=float) / 1_000
    predicted = frame.iloc[positions]["prediction"].to_numpy(dtype=float) / 1_000
    lower = min(float(np.min(actual)), float(np.min(predicted)))
    upper = max(float(np.max(actual)), float(np.max(predicted)))
    figure, axis = plt.subplots(figsize=(6.0, 4.3), constrained_layout=True)
    axis.scatter(actual, predicted, s=10, alpha=0.25, color="#087e78", edgecolors="none")
    axis.plot([lower, upper], [lower, upper], "--", color="#d66a4d", linewidth=1.6)
    axis.set(
        title="Tampa repeat-sale model in the Orlando test",
        xlabel="Recorded Orlando sale price ($000s)",
        ylabel="Predicted price ($000s)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _write_duration_figure(episodes: pd.DataFrame, path: Path) -> None:
    plt.switch_backend("Agg")
    figure, axis = plt.subplots(figsize=(6.0, 4.3), constrained_layout=True)
    for market, label, color in (
        (TAMPA_MARKET, "Tampa / Hillsborough", "#087e78"),
        (ORLANDO_MARKET, "Orlando / Orange", "#d66a4d"),
    ):
        group = episodes.loc[episodes["market"].eq(market) & episodes["operator"].eq("opendoor")]
        weeks, exit_cdf = _kaplan_meier_curve(group, horizon_days=365)
        axis.step(weeks, exit_cdf, where="post", label=label, color=color, linewidth=2.2)
    axis.axvline(17, color="#788589", linewidth=1.1, linestyle=":", label="17-week policy horizon")
    axis.set(
        title="Opendoor recorded inventory-exit distribution",
        xlabel="Weeks since acquisition deed",
        ylabel="Kaplan-Meier probability of recorded resale",
        xlim=(0, 52),
        ylim=(0, 1),
    )
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def _kaplan_meier_curve(frame: pd.DataFrame, *, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        return np.array([0.0]), np.array([0.0])
    duration = frame["hold_days"].to_numpy(dtype=float)
    event = frame["event_observed"].to_numpy(dtype=int)
    survival = 1.0
    times = [0.0]
    cdf = [0.0]
    for day in np.sort(np.unique(duration[(event == 1) & (duration <= horizon_days)])):
        at_risk = int(np.sum(duration >= day))
        exits = int(np.sum((duration == day) & (event == 1)))
        if at_risk:
            survival *= 1.0 - exits / at_risk
        times.append(float(day / 7))
        cdf.append(float(1.0 - survival))
    return np.asarray(times), np.asarray(cdf)


def _serialize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("acquisition_start", "acquisition_end"):
        if column in result:
            result[column] = pd.to_datetime(result[column]).dt.date.astype(str)
    return result


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else round(float(value), 6)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _read_operator_summary_for_report(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("operator_summary")
    return rows if isinstance(rows, list) else payload.get("opendoor_slice", [])


def _opendoor_card(row: dict[str, Any]) -> str:
    market = "Tampa" if row.get("market") == TAMPA_MARKET else "Orlando"
    return (
        '<div class="card">'
        f'<div class="eyebrow">{market}</div><div class="value">{int(row.get("n_acquisitions", 0)):,}</div>'
        '<div class="label">Opendoor acquisition spells</div>'
        f'<div class="context">KM exit by 120 days: {_percent(row.get("km_exit_probability_120d", 0))}</div>'
        "</div>"
    )


def _operator_table_row(row: dict[str, Any]) -> str:
    market = "Tampa" if row.get("market") == TAMPA_MARKET else "Orlando"
    operator = str(row.get("operator", "")).replace("_", " ").title()
    return (
        f"<tr><td>{market}</td><td>{html.escape(operator)}</td>"
        f"<td>{int(row.get('n_acquisitions', 0)):,}</td>"
        f"<td>{int(row.get('n_observed_exits', 0)):,}</td>"
        f"<td>{_percent(row.get('km_exit_probability_120d', 0))}</td>"
        f"<td>{_days(row.get('median_observed_hold_days'))}</td></tr>"
    )


def _metric(value: object, label: str, context: str) -> str:
    return (
        f'<div class="card"><div class="value">{value}</div>'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="context">{html.escape(context)}</div></div>'
    )


def _money(value: object) -> str:
    return f"${float(str(value)):,.0f}"


def _percent(value: object) -> str:
    if value is None:
        return "not estimable"
    numeric = float(str(value))
    if not math.isfinite(numeric):
        return "not estimable"
    return f"{numeric * 100:.1f}%"


def _decimal(value: object, places: int) -> str:
    return f"{float(str(value)):.{places}f}"


def _days(value: object) -> str:
    if value is None or not math.isfinite(float(str(value))):
        return "not estimable"
    return f"{float(str(value)):,.0f} days"


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
