"""Public command-line interface for reproducible project workflows."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml

from offer_to_exit.data.catalog import SOURCES
from offer_to_exit.data.fetch import fetch_source
from offer_to_exit.data.florida import (
    HILLSBOROUGH_MARKET,
    ORANGE_MARKET,
    FloridaPreparationSummary,
    link_ibuyer_episodes,
    sanitize_hillsborough_archive,
    sanitize_orange_jsonl,
)
from offer_to_exit.data.prepare import (
    write_preparation_manifest,
)

FLORIDA_SOURCE_KEYS = ("hillsborough_sales", "orange_sales")
FLORIDA_MARKETS = ("hillsborough", "orange")
FLORIDA_STRING_COLUMNS = {
    "parcel_id": "string",
    "market": "string",
    "property_type_code": "string",
    "dor_code": "string",
    "instrument_type": "string",
    "neighborhood": "string",
    "subdivision": "string",
    "census_block_group": "string",
    "buyer_operator": "string",
    "seller_operator": "string",
}

app = typer.Typer(
    name="offer-to-exit",
    help="Risk-aware home acquisition and resale pricing under uncertainty.",
    no_args_is_help=True,
)


@app.command("catalog")
def catalog_command(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the machine-readable catalog.")
    ] = False,
) -> None:
    """List external sources and their publication constraints."""

    if as_json:
        typer.echo(json.dumps({k: v.to_dict() for k, v in SOURCES.items()}, indent=2))
        return
    for key, source in SOURCES.items():
        size = (
            f"{source.approximate_bytes / 1_000_000:.1f} MB"
            if source.approximate_bytes
            else "size varies"
        )
        typer.echo(f"{key:26} {size:12} {source.title} — {source.publisher}")


@app.command("fetch")
def fetch_command(
    sources: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="Catalog key; repeat to fetch several."),
    ] = None,
    raw_dir: Annotated[Path, typer.Option(help="Git-ignored destination.")] = Path("data/raw"),
    overwrite: Annotated[bool, typer.Option(help="Replace an existing local file.")] = False,
) -> None:
    """Download the Tampa and Orlando county records with SHA-256 provenance."""

    selected = sources or list(FLORIDA_SOURCE_KEYS)
    for key in selected:
        result = fetch_source(key, raw_dir, overwrite=overwrite)
        typer.echo(f"{result.source}: {result.bytes:,} bytes, sha256={result.sha256[:12]}…")


@app.command("prepare")
def prepare_command(
    raw_dir: Annotated[Path, typer.Option(help="Directory created by fetch.")] = Path("data/raw"),
    processed_dir: Annotated[
        Path, typer.Option(help="Git-ignored sanitized output directory.")
    ] = Path("data/processed"),
    markets: Annotated[
        list[str] | None,
        typer.Option(
            "--market",
            "-m",
            help="Market to prepare: hillsborough or orange; repeat to select both.",
        ),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option(help="Administrative censoring date for open iBuyer episodes (YYYY-MM-DD)."),
    ] = None,
    maximum_hold_days: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum observable holding period before administrative censoring.",
        ),
    ] = 1_095,
) -> None:
    """Build privacy-safe Florida transactions and named-iBuyer episodes."""

    selected = list(dict.fromkeys(markets or FLORIDA_MARKETS))
    invalid = sorted(set(selected).difference(FLORIDA_MARKETS))
    if invalid:
        choices = ", ".join(FLORIDA_MARKETS)
        raise typer.BadParameter(f"Unknown market(s): {', '.join(invalid)}. Choose from {choices}.")

    raw_inputs = {
        "hillsborough": raw_dir / SOURCES["hillsborough_sales"].filename,
        "orange": raw_dir / SOURCES["orange_sales"].filename,
    }
    missing = [raw_inputs[market] for market in selected if not raw_inputs[market].exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise typer.BadParameter(
            f"Missing raw input(s): {missing_text}. Run `offer-to-exit fetch` first."
        )

    summaries: list[FloridaPreparationSummary] = []
    transaction_files: list[Path] = []
    if "hillsborough" in selected:
        output = processed_dir / "hillsborough_transactions_safe.csv.gz"
        summaries.append(sanitize_hillsborough_archive(raw_inputs["hillsborough"], output))
        transaction_files.append(output)
    if "orange" in selected:
        output = processed_dir / "orange_transactions_safe.csv.gz"
        summaries.append(sanitize_orange_jsonl(raw_inputs["orange"], output))
        transaction_files.append(output)

    combined = processed_dir / "florida_transactions_safe.csv.gz"
    transaction_rows = _combine_transaction_tables(transaction_files, combined)
    retrieval_cutoffs = None if as_of is not None else _source_retrieval_cutoffs(raw_dir, selected)
    operator_transactions, source_observation_ends = _read_operator_transactions(
        combined,
        latest_allowed_dates=retrieval_cutoffs,
    )
    episodes = link_ibuyer_episodes(
        operator_transactions,
        as_of=as_of if as_of is not None else source_observation_ends,
        maximum_hold_days=maximum_hold_days,
    )
    episode_observation_ends = (
        {market: pd.Timestamp(as_of) for market in source_observation_ends}
        if as_of is not None
        else source_observation_ends
    )
    episodes_output = processed_dir / "named_ibuyer_episodes_safe.csv.gz"
    _write_frame_atomically(episodes, episodes_output)
    manifest = write_preparation_manifest(
        processed_dir,
        summaries,
        [combined, episodes_output],
        analysis={
            "episode_observation_end_by_market": {
                market: cutoff.date().isoformat()
                for market, cutoff in sorted(episode_observation_ends.items())
            },
            "maximum_episode_linkage_days": maximum_hold_days,
        },
    )
    typer.echo(
        f"Prepared {transaction_rows:,} privacy-safe transactions and "
        f"{len(episodes):,} named-iBuyer episodes; {manifest}"
    )


def _combine_transaction_tables(inputs: list[Path], output: Path) -> int:
    """Combine sanitized county tables without materializing all rows in memory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = 0
    wrote_header = False
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as sink:
            for source in inputs:
                for chunk in pd.read_csv(
                    source,
                    chunksize=100_000,
                    low_memory=False,
                    dtype=FLORIDA_STRING_COLUMNS,
                ):
                    chunk.to_csv(sink, index=False, header=not wrote_header)
                    wrote_header = True
                    rows += len(chunk)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def _read_operator_transactions(
    path: Path,
    *,
    latest_allowed_dates: dict[str, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    """Load operator deeds and derive each market's latest observable sale date."""

    chunks: list[pd.DataFrame] = []
    market_observation_ends: dict[str, pd.Timestamp] = {}
    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
        low_memory=False,
        dtype=FLORIDA_STRING_COLUMNS,
    ):
        dates = pd.to_datetime(chunk["sale_date"], format="mixed", errors="coerce")
        prices = pd.to_numeric(chunk["sale_price"], errors="coerce")
        eligible_date = dates.notna() & prices.ge(10_000)
        if latest_allowed_dates is not None:
            allowed = chunk["market"].astype("string").map(latest_allowed_dates)
            eligible_date &= dates.le(allowed)
        for market, latest in (
            dates.loc[eligible_date].groupby(chunk.loc[eligible_date, "market"]).max().items()
        ):
            if pd.isna(market) or pd.isna(latest):
                continue
            key = str(market)
            observed = pd.Timestamp(latest)
            previous = market_observation_ends.get(key)
            if previous is None or observed > previous:
                market_observation_ends[key] = observed
        operator_rows = chunk.loc[
            chunk["buyer_operator"].notna() | chunk["seller_operator"].notna()
        ].copy()
        if not operator_rows.empty:
            chunks.append(operator_rows)
    if chunks:
        return pd.concat(chunks, ignore_index=True), market_observation_ends
    return (
        pd.DataFrame(
            columns=[
                "parcel_id",
                "market",
                "sale_date",
                "sale_price",
                "buyer_operator",
                "seller_operator",
            ]
        ),
        market_observation_ends,
    )


def _source_retrieval_cutoffs(raw_dir: Path, selected: list[str]) -> dict[str, pd.Timestamp]:
    """Read per-source retrieval dates so future-dated records cannot extend follow-up."""

    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(
            "Raw fetch manifest is required when --as-of is omitted; run `offer-to-exit fetch` "
            "or pass an explicit --as-of date."
        )
    try:
        files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        source_market = {
            "hillsborough": ("hillsborough_sales", HILLSBOROUGH_MARKET),
            "orange": ("orange_sales", ORANGE_MARKET),
        }
        cutoffs = {
            market: pd.Timestamp(files[source]["retrieved_at"]).tz_localize(None).normalize()
            for selected_market in selected
            for source, market in (source_market[selected_market],)
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(
            "Raw fetch manifest does not contain valid Florida retrieval timestamps; "
            "run `offer-to-exit fetch` or pass an explicit --as-of date."
        ) from error
    if any(pd.isna(cutoff) for cutoff in cutoffs.values()):
        raise typer.BadParameter("Raw fetch manifest contains a missing retrieval timestamp.")
    return cutoffs


def _write_frame_atomically(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@app.command("run")
def run_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Experiment YAML configuration.")
    ] = Path("configs/quickstart.yaml"),
) -> None:
    """Run simulation, fit decision components, and write release artifacts."""

    from offer_to_exit.workflow import run_experiment

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    result = run_experiment(payload, config_path=config)
    typer.echo(json.dumps(result, indent=2))


@app.command("florida-study")
def florida_study_command(
    transactions: Annotated[
        Path,
        typer.Option(help="Privacy-safe combined transactions created by prepare."),
    ] = Path("data/processed/florida_transactions_safe.csv.gz"),
    episodes: Annotated[
        Path,
        typer.Option(help="Privacy-safe named-iBuyer episodes created by prepare."),
    ] = Path("data/processed/named_ibuyer_episodes_safe.csv.gz"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Destination for aggregate, versioned evidence."),
    ] = Path("artifacts/release"),
) -> None:
    """Fit Tampa models and score an Orlando external-market evaluation."""

    if not transactions.exists() or not episodes.exists():
        raise typer.BadParameter(
            "Florida processed inputs are missing. Run `offer-to-exit prepare` first."
        )
    from offer_to_exit.florida_release import run_florida_release

    result = run_florida_release(transactions, episodes, output_dir)
    typer.echo(json.dumps(result, indent=2))


@app.command("demo")
def demo_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Experiment YAML configuration.")
    ] = Path("configs/quickstart.yaml"),
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional alternate path for the static HTML decision explorer.",
        ),
    ] = None,
) -> None:
    """Build a static, dependency-free decision explorer from a seeded run."""

    from offer_to_exit.workflow import render_demo, run_experiment

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    result = run_experiment(payload, config_path=config)
    generated_output = Path(str(result["artifacts"]["demo"]))
    destination = generated_output if output is None else output
    if destination != generated_output:
        render_demo(result, destination)
    typer.echo(f"Wrote {destination}")
