from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from offer_to_exit import cli
from offer_to_exit.data.catalog import DataSource
from offer_to_exit.data.florida import FloridaPreparationSummary

runner = CliRunner()


def _source(key: str, *, approximate_bytes: int | None = 1_500_000) -> DataSource:
    return DataSource(
        key=key,
        title=f"{key} title",
        publisher="Test Publisher",
        url=f"https://example.test/{key}",
        landing_page="https://example.test",
        filename=f"{key}.csv",
        grain="test row",
        purpose="unit testing",
        update_cadence="never",
        approximate_bytes=approximate_bytes,
        redistribution="test only",
        attribution="Test Publisher",
    )


def test_catalog_supports_human_and_json_output(monkeypatch: object) -> None:
    sources = {
        "fixed": _source("fixed"),
        "variable": _source("variable", approximate_bytes=None),
    }
    monkeypatch.setattr(cli, "SOURCES", sources)  # type: ignore[attr-defined]

    human = runner.invoke(cli.app, ["catalog"])
    assert human.exit_code == 0
    assert "1.5 MB" in human.stdout
    assert "size varies" in human.stdout
    assert "Test Publisher" in human.stdout

    machine = runner.invoke(cli.app, ["catalog", "--json"])
    assert machine.exit_code == 0
    payload = json.loads(machine.stdout)
    assert payload["fixed"]["filename"] == "fixed.csv"


def test_fetch_command_uses_requested_and_default_sources(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "SOURCES",
        {
            "hillsborough_sales": _source("hillsborough_sales"),
            "orange_sales": _source("orange_sales"),
            "legacy": _source("legacy"),
        },
    )
    calls: list[tuple[str, Path, bool]] = []

    def fake_fetch(source: str, raw_dir: Path, *, overwrite: bool = False) -> object:
        calls.append((source, raw_dir, overwrite))
        return SimpleNamespace(source=source, bytes=1_234, sha256="a" * 64)

    monkeypatch.setattr(cli, "fetch_source", fake_fetch)  # type: ignore[attr-defined]
    requested = runner.invoke(
        cli.app,
        [
            "fetch",
            "--source",
            "orange_sales",
            "--raw-dir",
            str(tmp_path),
            "--overwrite",
        ],
    )
    assert requested.exit_code == 0
    assert calls == [("orange_sales", tmp_path, True)]
    assert "1,234 bytes" in requested.stdout
    assert "aaaaaaaaaaaa…" in requested.stdout

    calls.clear()
    defaulted = runner.invoke(cli.app, ["fetch", "--raw-dir", str(tmp_path)])
    assert defaulted.exit_code == 0
    assert [call[0] for call in calls] == ["hillsborough_sales", "orange_sales"]


def test_prepare_command_builds_combined_transactions_episodes_and_manifest(
    monkeypatch: object, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    sources = {
        "hillsborough_sales": _source("hillsborough_sales"),
        "orange_sales": _source("orange_sales"),
    }
    monkeypatch.setattr(cli, "SOURCES", sources)  # type: ignore[attr-defined]
    for source in sources.values():
        (raw_dir / source.filename).write_bytes(b"fixture")

    calls: list[tuple[str, Path, Path]] = []

    def write_fixture(
        market: str,
        output: Path,
        rows: list[dict[str, object]],
    ) -> FloridaPreparationSummary:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output, index=False)
        return FloridaPreparationSummary(
            market=market,
            input_file="fixture",
            output_file=str(output),
            input_rows=len(frame),
            output_rows=len(frame),
            columns=tuple(frame.columns),
            created_at="2026-09-03T00:00:00+00:00",
        )

    def fake_hillsborough(source: Path, destination: Path) -> FloridaPreparationSummary:
        calls.append(("hillsborough", source, destination))
        return write_fixture(
            "tampa_hillsborough",
            destination,
            [
                _florida_transaction(
                    "tampa-home", "tampa_hillsborough", "2023-01-01", 300_000, buyer="opendoor"
                ),
                _florida_transaction(
                    "tampa-home", "tampa_hillsborough", "2023-05-01", 340_000, seller="opendoor"
                ),
            ],
        )

    def fake_orange(source: Path, destination: Path) -> FloridaPreparationSummary:
        calls.append(("orange", source, destination))
        return write_fixture(
            "orlando_orange",
            destination,
            [
                _florida_transaction(
                    "orlando-home", "orlando_orange", "2023-02-01", 250_000, buyer="offerpad"
                ),
                _florida_transaction(
                    "orlando-home", "orlando_orange", "2023-06-01", 280_000, seller="offerpad"
                ),
            ],
        )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "sanitize_hillsborough_archive", fake_hillsborough
    )
    monkeypatch.setattr(cli, "sanitize_orange_jsonl", fake_orange)  # type: ignore[attr-defined]

    full = runner.invoke(
        cli.app,
        [
            "prepare",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--as-of",
            "2024-01-01",
            "--maximum-hold-days",
            "730",
        ],
    )
    assert full.exit_code == 0
    assert "Prepared 4 privacy-safe transactions and 2 named-iBuyer episodes" in full.stdout
    assert [call[0] for call in calls] == ["hillsborough", "orange"]
    combined = pd.read_csv(processed_dir / "florida_transactions_safe.csv.gz")
    episodes = pd.read_csv(
        processed_dir / "named_ibuyer_episodes_safe.csv.gz", dtype={"dor_code": "string"}
    )
    assert len(combined) == 4
    assert set(episodes["operator"]) == {"opendoor", "offerpad"}
    assert set(episodes["dor_code"]) == {"0100"}
    manifest = json.loads((processed_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["privacy"]["party_names_retained"] is False
    assert len(manifest["tables"]) == 2
    assert manifest["analysis"]["episode_observation_end_by_market"] == {
        "orlando_orange": "2024-01-01",
        "tampa_hillsborough": "2024-01-01",
    }
    assert manifest["analysis"]["maximum_episode_linkage_days"] == 730


def test_prepare_command_rejects_unknown_markets_and_reports_missing_inputs(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "SOURCES",
        {
            "hillsborough_sales": _source("hillsborough_sales"),
            "orange_sales": _source("orange_sales"),
        },
    )
    unknown = runner.invoke(cli.app, ["prepare", "--market", "miami"])
    assert unknown.exit_code == 2
    assert "Unknown market(s): miami" in unknown.stderr

    missing = runner.invoke(cli.app, ["prepare", "--raw-dir", str(tmp_path)])
    assert missing.exit_code == 2
    assert "Missing raw input(s)" in missing.stderr
    assert "fetch` first" in missing.stderr


def _florida_transaction(
    parcel_id: str,
    market: str,
    sale_date: str,
    sale_price: float,
    *,
    buyer: str | None = None,
    seller: str | None = None,
) -> dict[str, object]:
    return {
        "parcel_id": parcel_id,
        "market": market,
        "sale_date": sale_date,
        "sale_price": sale_price,
        "property_type_code": "0100",
        "dor_code": "0100",
        "qualified": True,
        "improved": True,
        "buyer_operator": buyer,
        "seller_operator": seller,
    }


def test_run_and_demo_commands_delegate_to_workflow(monkeypatch: object, tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text("seed: 7\nlabel: test\n", encoding="utf-8")
    output = tmp_path / "demo.html"
    calls: list[tuple[str, object, Path]] = []

    fake_workflow = ModuleType("offer_to_exit.workflow")

    generated_demo = tmp_path / "generated" / "demo.html"

    def fake_run(payload: object, *, config_path: Path) -> dict[str, object]:
        calls.append(("run", payload, config_path))
        generated_demo.parent.mkdir(parents=True, exist_ok=True)
        generated_demo.write_text("<html>generated</html>", encoding="utf-8")
        return {
            "status": "ok",
            "seed": 7,
            "artifacts": {"demo": str(generated_demo)},
        }

    def fake_render(result: object, destination: Path) -> None:
        calls.append(("render", result, destination))
        destination.write_text("<html>demo</html>", encoding="utf-8")

    fake_workflow.run_experiment = fake_run  # type: ignore[attr-defined]
    fake_workflow.render_demo = fake_render  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "offer_to_exit.workflow", fake_workflow)  # type: ignore[attr-defined]

    run_result = runner.invoke(cli.app, ["run", "--config", str(config)])
    assert run_result.exit_code == 0
    assert json.loads(run_result.stdout) == {
        "status": "ok",
        "seed": 7,
        "artifacts": {"demo": str(generated_demo)},
    }

    demo_result = runner.invoke(
        cli.app,
        ["demo", "--config", str(config), "--output", str(output)],
    )
    assert demo_result.exit_code == 0
    assert f"Wrote {output}" in demo_result.stdout
    assert output.read_text(encoding="utf-8") == "<html>demo</html>"
    assert [call[0] for call in calls] == ["run", "run", "render"]

    calls.clear()
    default_demo = runner.invoke(cli.app, ["demo", "--config", str(config)])
    assert default_demo.exit_code == 0
    assert f"Wrote {generated_demo}" in default_demo.stdout
    assert [call[0] for call in calls] == ["run"]
