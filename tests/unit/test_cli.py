from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from typer.testing import CliRunner

from offer_to_exit import cli
from offer_to_exit.data.catalog import DataSource

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
        cli, "SOURCES", {"one": _source("one"), "two": _source("two")}
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
            "two",
            "--raw-dir",
            str(tmp_path),
            "--overwrite",
        ],
    )
    assert requested.exit_code == 0
    assert calls == [("two", tmp_path, True)]
    assert "1,234 bytes" in requested.stdout
    assert "aaaaaaaaaaaa…" in requested.stdout

    calls.clear()
    defaulted = runner.invoke(cli.app, ["fetch", "--raw-dir", str(tmp_path)])
    assert defaulted.exit_code == 0
    assert [call[0] for call in calls] == ["one", "two"]


def test_prepare_command_covers_full_and_market_only_paths(
    monkeypatch: object, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    sanitizers: list[tuple[str, Path, Path]] = []

    def fake_residential(source: Path, destination: Path) -> str:
        sanitizers.append(("residential", source, destination))
        return "residential-summary"

    def fake_sales(source: Path, destination: Path) -> str:
        sanitizers.append(("sales", source, destination))
        return "sales-summary"

    def fake_market(source: Path, destination: Path) -> list[Path]:
        assert source == raw_dir
        assert destination == processed_dir
        return [destination / "market-a.csv", destination / "market-b.csv"]

    manifests: list[tuple[Path, list[object], list[Path]]] = []

    def fake_manifest(destination: Path, summaries: list[object], market_files: list[Path]) -> Path:
        manifests.append((destination, list(summaries), list(market_files)))
        return destination / "manifest.json"

    monkeypatch.setattr(cli, "sanitize_residential", fake_residential)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "sanitize_sales", fake_sales)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "prepare_market_series", fake_market)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "write_preparation_manifest", fake_manifest)  # type: ignore[attr-defined]

    full = runner.invoke(
        cli.app,
        [
            "prepare",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
        ],
    )
    assert full.exit_code == 0
    assert "Prepared 4 tables" in full.stdout
    assert [call[0] for call in sanitizers] == ["residential", "sales"]
    assert manifests[-1][1] == ["residential-summary", "sales-summary"]

    sanitizers.clear()
    market_only = runner.invoke(
        cli.app,
        [
            "prepare",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--market-only",
        ],
    )
    assert market_only.exit_code == 0
    assert "Prepared 2 tables" in market_only.stdout
    assert sanitizers == []
    assert manifests[-1][1] == []


def test_run_and_demo_commands_delegate_to_workflow(monkeypatch: object, tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text("seed: 7\nlabel: test\n", encoding="utf-8")
    output = tmp_path / "demo.html"
    calls: list[tuple[str, object, Path]] = []

    fake_workflow = ModuleType("offer_to_exit.workflow")

    def fake_run(payload: object, *, config_path: Path) -> dict[str, object]:
        calls.append(("run", payload, config_path))
        return {"status": "ok", "seed": 7}

    def fake_render(result: object, destination: Path) -> None:
        calls.append(("render", result, destination))
        destination.write_text("<html>demo</html>", encoding="utf-8")

    fake_workflow.run_experiment = fake_run  # type: ignore[attr-defined]
    fake_workflow.render_demo = fake_render  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "offer_to_exit.workflow", fake_workflow)  # type: ignore[attr-defined]

    run_result = runner.invoke(cli.app, ["run", "--config", str(config)])
    assert run_result.exit_code == 0
    assert json.loads(run_result.stdout) == {"status": "ok", "seed": 7}

    demo_result = runner.invoke(
        cli.app,
        ["demo", "--config", str(config), "--output", str(output)],
    )
    assert demo_result.exit_code == 0
    assert f"Wrote {output}" in demo_result.stdout
    assert output.read_text(encoding="utf-8") == "<html>demo</html>"
    assert [call[0] for call in calls] == ["run", "run", "render"]
