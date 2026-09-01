"""Public command-line interface for reproducible project workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from offer_to_exit.data.catalog import SOURCES
from offer_to_exit.data.fetch import fetch_source
from offer_to_exit.data.prepare import (
    prepare_market_series,
    sanitize_residential,
    sanitize_sales,
    write_preparation_manifest,
)

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
    """Download official raw sources and record SHA-256 provenance."""

    selected = sources or list(SOURCES)
    for key in selected:
        result = fetch_source(key, raw_dir, overwrite=overwrite)
        typer.echo(f"{result.source}: {result.bytes:,} bytes, sha256={result.sha256[:12]}…")


@app.command("prepare")
def prepare_command(
    raw_dir: Annotated[Path, typer.Option(help="Directory created by fetch.")] = Path("data/raw"),
    processed_dir: Annotated[
        Path, typer.Option(help="Git-ignored sanitized output directory.")
    ] = Path("data/processed"),
    market_only: Annotated[
        bool,
        typer.Option("--market-only", help="Skip the larger county files during a quick audit."),
    ] = False,
) -> None:
    """Create Phoenix-only analytical inputs while dropping names and addresses."""

    summaries = []
    if not market_only:
        summaries.append(
            sanitize_residential(
                raw_dir / "Residential_Master.zip",
                processed_dir / "residential_safe.csv.gz",
            )
        )
        summaries.append(
            sanitize_sales(
                raw_dir / "Sales_Affidavits.zip",
                processed_dir / "sales_safe.csv.gz",
            )
        )
    market_files = prepare_market_series(raw_dir, processed_dir)
    manifest = write_preparation_manifest(processed_dir, summaries, market_files)
    typer.echo(f"Prepared {len(summaries) + len(market_files)} tables; {manifest}")


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


@app.command("demo")
def demo_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Experiment YAML configuration.")
    ] = Path("configs/quickstart.yaml"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Static HTML decision explorer.")
    ] = Path("artifacts/release/demo.html"),
) -> None:
    """Build a static, dependency-free decision explorer from a seeded run."""

    from offer_to_exit.workflow import render_demo, run_experiment

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    result = run_experiment(payload, config_path=config)
    render_demo(result, output)
    typer.echo(f"Wrote {output}")
