from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from offer_to_exit.data import fetch as fetch_module
from offer_to_exit.data.catalog import DataSource
from offer_to_exit.data.fetch import fetch_source, sha256_file


def _source(key: str = "fixture", filename: str = "fixture.bin") -> DataSource:
    return DataSource(
        key=key,
        title="Fixture bytes",
        publisher="Unit Tests",
        url=f"https://example.test/{filename}",
        landing_page="https://example.test/data",
        filename=filename,
        grain="byte",
        purpose="exercise atomic retrieval",
        update_cadence="never",
        approximate_bytes=12,
        redistribution="test only",
        attribution="Unit Tests",
    )


def test_sha256_file_streams_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abcdefghij")
    assert sha256_file(path, chunk_size=3) == hashlib.sha256(b"abcdefghij").hexdigest()


def test_fetch_source_writes_bytes_checksum_and_manifest(tmp_path: Path) -> None:
    payload = b"downloaded fixture"
    observed_requests: list[tuple[str, str | None, int]] = []

    def opener(request: Request, *, timeout: int) -> io.BytesIO:
        observed_requests.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(payload)

    source = _source()
    result = fetch_source(source, tmp_path, opener=opener)

    assert (tmp_path / source.filename).read_bytes() == payload
    assert result.bytes == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert observed_requests == [(source.url, "offer-to-exit/0.1", 120)]
    assert not list(tmp_path.glob(f".{source.filename}.*"))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["files"][source.key]
    assert recorded["sha256"] == result.sha256
    assert recorded["publisher"] == "Unit Tests"
    assert recorded["contains_pii"] is False


def test_existing_file_is_reused_and_manifest_is_extended(
    monkeypatch: object, tmp_path: Path
) -> None:
    first = _source("first", "first.bin")
    existing = tmp_path / first.filename
    existing.write_bytes(b"already here")

    def forbidden_opener(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("existing file should not trigger the opener")

    monkeypatch.setattr(fetch_module, "get_source", lambda key: first)  # type: ignore[attr-defined]
    reused = fetch_source("first", tmp_path, opener=forbidden_opener)
    assert reused.bytes == len(b"already here")
    assert reused.sha256 == hashlib.sha256(b"already here").hexdigest()

    second = _source("second", "second.bin")
    fetch_source(second, tmp_path, opener=lambda request, timeout: io.BytesIO(b"second"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"first", "second"}


def test_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    source = _source()
    destination = tmp_path / source.filename
    destination.write_bytes(b"old")

    result = fetch_source(
        source,
        tmp_path,
        overwrite=True,
        opener=lambda request, timeout: io.BytesIO(b"new payload"),
    )

    assert destination.read_bytes() == b"new payload"
    assert result.sha256 == hashlib.sha256(b"new payload").hexdigest()


def test_failed_fetch_removes_temporary_file(tmp_path: Path) -> None:
    source = _source()

    def failing_opener(request: Request, *, timeout: int) -> object:
        del request, timeout
        raise OSError("simulated network failure")

    with pytest.raises(OSError, match="simulated network failure"):
        fetch_source(source, tmp_path, opener=failing_opener)

    assert not (tmp_path / source.filename).exists()
    assert list(tmp_path.iterdir()) == []
