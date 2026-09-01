from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from offer_to_exit.workflow import CLAIM_SCOPE, render_demo, run_experiment


def _config(output_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {
            "name": "integration-quickstart",
            "mode": "semi_synthetic",
            "seed": 2_026_090_1,
            "output_dir": str(output_dir),
            "overwrite": True,
        },
        "market": {
            "name": "Phoenix-shaped generated fixture",
            "exit_horizon_weeks": 12,
        },
        "data": {
            "source_mode": "generated_fixture",
            "generated_property_count": 500,
            "split": {"train_fraction": 0.60},
        },
        "simulator": {"training_seed": 1_129, "evaluation_seed": 4_079},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_quickstart_is_reproducible_and_publishes_a_static_demo(tmp_path: Path) -> None:
    output_dir = tmp_path / "release"
    config = _config(output_dir)

    first = run_experiment(config, config_path=tmp_path / "quickstart.yaml")

    json.dumps(first, allow_nan=False)
    assert first["mode"] == "semi_synthetic"
    assert first["claim_scope"] == CLAIM_SCOPE
    assert first["reproducibility"]["independent_evaluation_environment"] is True
    assert first["metrics"]["evaluation_contract"]["used_for_fitting"] is False
    assert first["data"]["n_train_homes"] == 300
    assert first["data"]["n_evaluation_homes"] == 200
    assert len(first["decision_cases"]) == 3
    assert {case["first_action"] for case in first["decision_cases"]} >= {
        "hold",
        "cut_5_pct",
    }
    assert {case["recommendation"] for case in first["decision_cases"]} == {
        "price",
        "abstain",
    }
    assert all(
        (case["automated_offer"] is not None) == (case["recommendation"] == "price")
        for case in first["decision_cases"]
    )

    artifacts = first["artifacts"]
    manifest_path = Path(artifacts["manifest"])
    metrics_path = Path(artifacts["metrics"])
    cases_path = Path(artifacts["decision_cases"])
    summary_path = Path(artifacts["summary"])
    demo_path = Path(artifacts["demo"])
    figure_paths = [Path(path) for path in artifacts["figures"]]
    for path in [manifest_path, metrics_path, cases_path, summary_path, demo_path, *figure_paths]:
        assert path.is_file()
        assert path.stat().st_size > 100
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figure_paths)

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["metrics"] == first["metrics"]
    with cases_path.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    demo = demo_path.read_text(encoding="utf-8")
    assert "SEMI-SYNTHETIC" in demo
    assert "data:image/png;base64," in demo
    assert "no external assets or scripts" in demo

    stable_files = [metrics_path, cases_path, *figure_paths]
    first_hashes = {path.name: _sha256(path) for path in stable_files}
    second = run_experiment(config, config_path=tmp_path / "quickstart.yaml")
    second_hashes = {path.name: _sha256(path) for path in stable_files}
    assert second["metrics"] == first["metrics"]
    assert second["decision_cases"] == first["decision_cases"]
    assert second_hashes == first_hashes

    alternate_demo = render_demo(first, tmp_path / "standalone-demo.html")
    assert alternate_demo.is_file()
    assert "Offer to exit" in alternate_demo.read_text(encoding="utf-8")
