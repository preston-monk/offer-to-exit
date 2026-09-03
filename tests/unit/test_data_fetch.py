from __future__ import annotations

import hashlib
import io
import json
from gzip import open as gzip_open
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from offer_to_exit.data import fetch as fetch_module
from offer_to_exit.data.catalog import DataSource
from offer_to_exit.data.fetch import (
    discover_hillsborough_download,
    fetch_arcgis_source,
    fetch_hillsborough_source,
    fetch_source,
    sha256_file,
)


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
    assert observed_requests == [(source.url, "offer-to-exit/0.2", 120)]
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
    original_retrieval = "2026-08-31T12:34:56+00:00"
    original_url = "https://example.test/signed-snapshot"
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    first.key: {
                        "sha256": hashlib.sha256(b"already here").hexdigest(),
                        "bytes": len(b"already here"),
                        "retrieved_at": original_retrieval,
                        "url": original_url,
                        "records": 73,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def forbidden_opener(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("existing file should not trigger the opener")

    monkeypatch.setattr(fetch_module, "get_source", lambda key: first)  # type: ignore[attr-defined]
    reused = fetch_source("first", tmp_path, opener=forbidden_opener)
    assert reused.bytes == len(b"already here")
    assert reused.sha256 == hashlib.sha256(b"already here").hexdigest()
    assert reused.retrieved_at == original_retrieval
    assert reused.url == original_url
    assert reused.records == 73

    second = _source("second", "second.bin")
    fetch_source(second, tmp_path, opener=lambda request, timeout: io.BytesIO(b"second"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"first", "second"}
    assert manifest["files"]["first"]["retrieved_at"] == original_retrieval
    assert manifest["files"]["first"]["url"] == original_url
    assert manifest["files"]["first"]["records"] == 73


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


def test_arcgis_fetch_pages_to_gzip_json_lines(tmp_path: Path) -> None:
    source = DataSource(
        **{
            **_source("orange", "orange.jsonl.gz").to_dict(),
            "transport": "arcgis",
            "query_order_field": "OBJECTID",
            "page_size": 2,
        }
    )
    pages = {
        0: [{"attributes": {"OBJECTID": 1}}, {"attributes": {"OBJECTID": 2}}],
        2: [{"attributes": {"OBJECTID": 3}}],
    }
    requests: list[dict[str, list[str]]] = []

    def opener(request: Request, *, timeout: int) -> io.BytesIO:
        assert timeout == 120
        query = parse_qs(urlparse(request.full_url).query)
        requests.append(query)
        offset = int(query["resultOffset"][0])
        payload = {"features": pages[offset], "exceededTransferLimit": offset == 0}
        return io.BytesIO(json.dumps(payload).encode())

    result = fetch_arcgis_source(source, tmp_path, opener=opener)

    assert result.records == 3
    with gzip_open(tmp_path / source.filename, "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["OBJECTID"] for line in stream] == [1, 2, 3]
    assert [request["resultOffset"] for request in requests] == [["0"], ["2"]]
    assert requests[0]["returnGeometry"] == ["false"]
    assert requests[0]["orderByFields"] == ["OBJECTID ASC"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["orange"]["query"]["return_geometry"] is False


def test_arcgis_fetch_retries_and_resumes_from_page_checkpoint(tmp_path: Path) -> None:
    source = DataSource(
        **{
            **_source("resumable", "resumable.jsonl.gz").to_dict(),
            "transport": "arcgis",
            "query_order_field": "OBJECTID",
            "page_size": 2,
        }
    )
    first_offsets: list[int] = []

    def interrupted(request: Request, *, timeout: int) -> io.BytesIO:
        del timeout
        offset = int(parse_qs(urlparse(request.full_url).query)["resultOffset"][0])
        first_offsets.append(offset)
        if offset == 2:
            raise TimeoutError("transient handshake timeout")
        return io.BytesIO(
            json.dumps(
                {
                    "features": [
                        {"attributes": {"OBJECTID": 1}},
                        {"attributes": {"OBJECTID": 2}},
                    ],
                    "exceededTransferLimit": True,
                }
            ).encode()
        )

    with pytest.raises(TimeoutError):
        fetch_arcgis_source(source, tmp_path, opener=interrupted, max_retries=0)
    assert first_offsets == [0, 2]
    assert (tmp_path / ".resumable.jsonl.gz.part").exists()
    assert (tmp_path / ".resumable.jsonl.gz.part.json").exists()

    resumed_offsets: list[int] = []

    def resumed(request: Request, *, timeout: int) -> io.BytesIO:
        del timeout
        offset = int(parse_qs(urlparse(request.full_url).query)["resultOffset"][0])
        resumed_offsets.append(offset)
        return io.BytesIO(json.dumps({"features": [{"attributes": {"OBJECTID": 3}}]}).encode())

    result = fetch_arcgis_source(source, tmp_path, opener=resumed, overwrite=True)
    assert resumed_offsets == [2]
    assert result.records == 3
    with gzip_open(tmp_path / source.filename, "rt", encoding="utf-8") as stream:
        assert [json.loads(line)["OBJECTID"] for line in stream] == [1, 2, 3]
    assert not (tmp_path / ".resumable.jsonl.gz.part.json").exists()


def test_arcgis_pii_source_requires_explicit_allowlist(tmp_path: Path) -> None:
    source = DataSource(
        **{
            **_source("pii", "pii.jsonl.gz").to_dict(),
            "transport": "arcgis",
            "contains_pii": True,
        }
    )
    with pytest.raises(ValueError, match="allowlist"):
        fetch_arcgis_source(source, tmp_path)


def test_hillsborough_discovery_and_postback_download(tmp_path: Path) -> None:
    html = """
    <html><body><form>
      <input type="hidden" name="__VIEWSTATE" value="state-token" />
      <input type="hidden" name="__EVENTVALIDATION" value="validation-token" />
      <a href="javascript:__doPostBack('old$row','')">allsales_08_21_2026.zip</a>
      <a href="javascript:__doPostBack('grdFiles$ctl00$ctl04$ctl00','')">
        allsales_08_28_2026.zip
      </a>
    </form></body></html>
    """
    filename, target, hidden = discover_hillsborough_download(html)
    assert filename == "allsales_08_28_2026.zip"
    assert target == "grdFiles$ctl00$ctl04$ctl00"
    assert hidden["__VIEWSTATE"] == "state-token"

    source = DataSource(
        **{
            **_source("hillsborough", "hillsborough.zip").to_dict(),
            "transport": "aspnet_postback",
        }
    )
    observed_posts: list[dict[str, list[str]]] = []

    def opener(request: Request, *, timeout: int) -> io.BytesIO:
        assert timeout == 120
        if request.data is None:
            return io.BytesIO(html.encode())
        observed_posts.append(parse_qs(request.data.decode()))
        return io.BytesIO(b"PK\x03\x04fixture archive")

    result = fetch_hillsborough_source(source, tmp_path, opener=opener)

    assert (tmp_path / "hillsborough.zip").read_bytes() == b"PK\x03\x04fixture archive"
    assert result.url.endswith("#allsales_08_28_2026.zip")
    assert observed_posts[0]["__EVENTTARGET"] == ["grdFiles$ctl00$ctl04$ctl00"]
    assert observed_posts[0]["__VIEWSTATE"] == ["state-token"]
