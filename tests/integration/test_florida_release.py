from __future__ import annotations

from pathlib import Path

import pandas as pd

from offer_to_exit.data.florida import TRANSACTION_COLUMNS
from offer_to_exit.florida_release import run_florida_release


def test_florida_release_writes_aggregate_external_evaluation(tmp_path: Path) -> None:
    transactions = _transactions()
    episodes = _episodes()
    transaction_path = tmp_path / "transactions.csv.gz"
    episode_path = tmp_path / "episodes.csv.gz"
    transactions.to_csv(transaction_path, index=False, compression="gzip")
    episodes.to_csv(episode_path, index=False, compression="gzip")

    output = tmp_path / "release"
    result = run_florida_release(transaction_path, episode_path, output)

    assert result["design"]["development_market"].startswith("Hillsborough")
    assert result["valuation"]["orlando_common_window"]["n_transactions"] > 0
    assert result["valuation"]["orlando_observed_qualified_window"]["start"] == "2024-07-01"
    assert (
        "does not isolate"
        in result["valuation"]["orlando_observed_qualified_window"]["interpretation"]
    )
    assert result["inventory_duration"]["orlando_common_window"]["n_episodes"] > 0
    assert len(result["opendoor_slice"]) == 2
    assert (output / "florida_metrics.v2.json").exists()
    assert (output / "florida_manifest.v2.json").exists()
    report = (output / "florida_evidence.html").read_text(encoding="utf-8")
    assert "Tampa for development" in report
    assert "No Florida seller or buyer elasticity" in report
    assert "PRIVATE PERSON" not in report


def _transactions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = (
        ("2021-07-01", 80),
        ("2022-07-01", 40),
        ("2024-07-01", 44),
    )
    for market, market_shift in (("tampa_hillsborough", 0), ("orlando_orange", 18_000)):
        row_number = 0
        for start, count in periods:
            for offset in range(count):
                sale_date = pd.Timestamp(start) + pd.Timedelta(days=offset * 6)
                parcel_id = f"{market}-parcel-{row_number:04d}"
                target_price = 225_000 + market_shift + row_number * 1_350
                for transaction_date, transaction_price in (
                    (sale_date - pd.DateOffset(months=6), target_price / 1.06),
                    (sale_date, target_price),
                ):
                    row = dict.fromkeys(TRANSACTION_COLUMNS, pd.NA)
                    row.update(
                        {
                            "parcel_id": parcel_id,
                            "market": market,
                            "sale_date": transaction_date,
                            "sale_price": transaction_price,
                            "qualified": True,
                            "property_type_code": "0100",
                            "dor_code": "0100",
                            "improved": True,
                        }
                    )
                    rows.append(row)
                row_number += 1
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)


def _episodes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = (
        ("2018-01-01", 34),
        ("2021-02-01", 20),
        ("2022-03-01", 28),
    )
    for market, market_shift in (("tampa_hillsborough", 0), ("orlando_orange", 9)):
        row_number = 0
        for start, count in periods:
            for offset in range(count):
                acquisition_date = pd.Timestamp(start) + pd.Timedelta(days=offset * 5)
                event = offset % 4 != 0
                hold_days = 48 + ((offset + market_shift) % 16) * 7
                acquisition_price = 210_000 + market_shift * 1_000 + row_number * 1_100
                rows.append(
                    {
                        "parcel_id": f"{market}-episode-{row_number:04d}",
                        "market": market,
                        "operator": "opendoor" if offset % 2 == 0 else "offerpad",
                        "acquisition_date": acquisition_date,
                        "acquisition_price": acquisition_price,
                        "resale_date": (
                            acquisition_date + pd.Timedelta(days=hold_days) if event else pd.NaT
                        ),
                        "resale_price": acquisition_price * 1.06 if event else pd.NA,
                        "hold_days": hold_days,
                        "event_observed": event,
                        "linkage_status": "completed" if event else "right_censored",
                        "gross_spread": acquisition_price * 0.06 if event else pd.NA,
                        "gross_return": 0.06 if event else pd.NA,
                        "qualified": True,
                        "property_type_code": "0100",
                        "dor_code": "0100",
                        "improved": True,
                    }
                )
                row_number += 1
    return pd.DataFrame(rows)
