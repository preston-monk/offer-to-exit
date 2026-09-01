from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from offer_to_exit import __version__
from offer_to_exit.serving import app

pytestmark = pytest.mark.integration

client = TestClient(app)


def _healthy_payload() -> dict[str, object]:
    return {
        "property_id": "API-HEALTHY-001",
        "estimated_market_value": 400_000,
        "living_area_sqft": 1_900,
        "year_built": 1998,
        "repair_cost": 11_000,
        "weekly_holding_cost": 650,
        "risk_aversion": 0.20,
        "target_margin": 0.04,
    }


def test_health_is_small_and_explicit_about_evidence() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Offer-to-Exit-Evidence"] == "semi-synthetic"
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "evidence": "semi-synthetic",
        "worked_case_count": 3,
    }


def test_price_returns_deterministic_worked_case_policy() -> None:
    first = client.post("/v1/price", json=_healthy_payload())
    second = client.post("/v1/price", json=_healthy_payload())

    assert first.status_code == 200
    assert first.headers["X-Offer-to-Exit-Evidence"] == "semi-synthetic"
    assert first.json() == second.json()
    body = first.json()
    assert body["property_id"] == "API-HEALTHY-001"
    assert body["recommendation"] == "price"
    assert body["acquisition_offer"] == 348_000
    assert body["expected_profit"] > 0
    assert 0 <= body["probability_of_loss"] <= 1
    assert 0 <= body["sale_by_120_days"] <= 1
    assert len(body["policy"]) == 12
    assert body["policy"][0]["action"] == "hold"
    assert any("semi-synthetic" in reason for reason in body["reasons"])
    assert "worked_case=healthy-demand-margin-protection" in body["reasons"]


def test_wide_scenario_interval_abstains_without_an_offer() -> None:
    payload = {
        **_healthy_payload(),
        "property_id": "API-THIN-003",
        "estimated_market_value": 325_000,
        "repair_cost": 15_000,
        "weekly_holding_cost": 900,
        "risk_aversion": 0.90,
        "target_margin": 0.01,
    }

    response = client.post("/v1/price", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"] == "abstain"
    assert body["acquisition_offer"] is None
    assert body["policy"] == []
    assert any("interval is too wide" in reason for reason in body["reasons"])
    assert "worked_case=thin-comps-downside-protection" in body["reasons"]


def test_valid_schema_but_unsupported_home_routes_to_review() -> None:
    payload = {
        **_healthy_payload(),
        "property_id": "API-OOD-004",
        "living_area_sqft": 7_000,
    }

    response = client.post("/v1/price", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"] == "review"
    assert body["acquisition_offer"] is None
    assert body["policy"] == []
    assert any("outside Phoenix/Maricopa" in reason for reason in body["reasons"])
    assert any("living_area_sqft" in reason for reason in body["reasons"])


def test_request_schema_rejects_impossible_market_value() -> None:
    payload = {**_healthy_payload(), "estimated_market_value": 20_000}

    response = client.post("/v1/price", json=payload)

    assert response.status_code == 422
