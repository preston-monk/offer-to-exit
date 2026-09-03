"""Streaming retrieval with local checksums and provenance manifests."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from offer_to_exit.data.catalog import DataSource, get_source


@dataclass(frozen=True, slots=True)
class FetchResult:
    source: str
    path: str
    bytes: int
    sha256: str
    retrieved_at: str
    url: str
    records: int | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source(
    source: str | DataSource,
    raw_dir: Path,
    *,
    overwrite: bool = False,
    opener: Callable[..., object] = urlopen,
) -> FetchResult:
    """Fetch one catalog source atomically and append its local manifest."""

    entry = get_source(source) if isinstance(source, str) else source
    if entry.transport == "arcgis":
        return fetch_arcgis_source(
            entry,
            raw_dir,
            overwrite=overwrite,
            opener=opener,
        )
    if entry.transport == "aspnet_postback":
        return fetch_hillsborough_source(
            entry,
            raw_dir,
            overwrite=overwrite,
            opener=opener,
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / entry.filename
    if destination.exists() and not overwrite:
        result = _existing_result(entry, destination, raw_dir=raw_dir)
        _write_manifest(raw_dir, result, entry)
        return result

    request = Request(entry.url, headers={"User-Agent": "offer-to-exit/0.2"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{entry.filename}.", dir=raw_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with opener(request, timeout=120) as response:  # type: ignore[attr-defined]
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    result = FetchResult(
        source=entry.key,
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        retrieved_at=datetime.now(tz=UTC).isoformat(),
        url=entry.url,
    )
    _write_manifest(raw_dir, result, entry)
    return result


def fetch_arcgis_source(
    source: str | DataSource,
    raw_dir: Path,
    *,
    where: str = "1=1",
    out_fields: tuple[str, ...] | str | None = None,
    overwrite: bool = False,
    max_retries: int = 4,
    retry_backoff_seconds: float = 0.5,
    opener: Callable[..., object] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Page an ArcGIS feature layer into an atomic gzip JSON-lines file.

    The raw response can contain party names and parcel identifiers, so callers
    should place ``raw_dir`` under the repository's ignored ``data/raw`` tree.
    Each line contains one feature's ``attributes`` object.  Geometry is never
    requested because the released analytical data use coarse geographies only.
    """

    entry = get_source(source) if isinstance(source, str) else source
    if entry.transport != "arcgis":
        raise ValueError(f"Source {entry.key!r} is not an ArcGIS source")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / entry.filename
    page_size = entry.page_size or 2_000
    order_field = entry.query_order_field
    requested_fields = entry.out_fields if out_fields is None else out_fields
    selected_fields = (
        requested_fields if isinstance(requested_fields, str) else ",".join(requested_fields)
    )
    if not selected_fields:
        selected_fields = "*"
    if selected_fields == "*" and entry.contains_pii:
        raise ValueError(
            f"Source {entry.key!r} contains PII and requires an explicit out-fields allowlist"
        )
    query_metadata = {
        "where": where,
        "out_fields": selected_fields.split(","),
        "return_geometry": False,
        "page_size": page_size,
        "order_by": None if order_field is None else f"{order_field} ASC",
    }
    if destination.exists() and not overwrite:
        result = _existing_result(entry, destination, raw_dir=raw_dir)
        _write_manifest(raw_dir, result, entry, query=query_metadata)
        return result

    partial = raw_dir / f".{entry.filename}.part"
    checkpoint = raw_dir / f".{entry.filename}.part.json"
    signature = {
        "source": entry.key,
        "url": entry.url,
        **query_metadata,
    }
    offset = 0
    record_count = 0
    if partial.exists() and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            offset = int(saved.get("next_offset", 0))
            record_count = int(saved.get("records", 0))
        else:
            partial.unlink()
            checkpoint.unlink()
    elif partial.exists() or checkpoint.exists():
        partial.unlink(missing_ok=True)
        checkpoint.unlink(missing_ok=True)

    mode = "at" if offset else "wt"
    complete = False
    with cast(TextIO, gzip.open(partial, mode, encoding="utf-8", newline="\n")) as sink:
        while True:
            parameters = {
                "f": "json",
                "where": where,
                "outFields": selected_fields,
                "returnGeometry": "false",
                "resultOffset": str(offset),
                "resultRecordCount": str(page_size),
            }
            if order_field:
                parameters["orderByFields"] = f"{order_field} ASC"
            separator = "&" if "?" in entry.url else "?"
            request = Request(
                f"{entry.url}{separator}{urlencode(parameters)}",
                headers={"User-Agent": "offer-to-exit/0.2"},
            )
            payload = _arcgis_page_with_retry(
                request,
                entry=entry,
                opener=opener,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                sleep=sleep,
            )

            features = payload.get("features", [])
            if not isinstance(features, list):
                raise RuntimeError(
                    f"ArcGIS query for {entry.key} returned a non-list features value"
                )
            for feature in features:
                attributes = feature.get("attributes", feature)
                sink.write(json.dumps(attributes, separators=(",", ":"), default=str))
                sink.write("\n")
            sink.flush()
            record_count += len(features)

            if not features:
                complete = True
                break
            offset += len(features)
            _write_arcgis_checkpoint(
                checkpoint,
                signature=signature,
                next_offset=offset,
                records=record_count,
            )
            if len(features) < page_size and not payload.get("exceededTransferLimit", False):
                complete = True
                break

    if not complete:
        raise RuntimeError(f"ArcGIS query for {entry.key} ended before completion")
    os.replace(partial, destination)
    checkpoint.unlink(missing_ok=True)

    result = FetchResult(
        source=entry.key,
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        retrieved_at=datetime.now(tz=UTC).isoformat(),
        url=entry.url,
        records=record_count,
    )
    _write_manifest(raw_dir, result, entry, query=query_metadata)
    return result


def _arcgis_page_with_retry(
    request: Request,
    *,
    entry: DataSource,
    opener: Callable[..., object],
    max_retries: int,
    retry_backoff_seconds: float,
    sleep: Callable[[float], None],
) -> dict[str, object]:
    for attempt in range(max_retries + 1):
        try:
            with opener(request, timeout=120) as response:  # type: ignore[attr-defined]
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"ArcGIS query for {entry.key} returned a non-object payload")
            if "error" in payload:
                error = payload["error"]
                message = (
                    error.get("message", "unknown ArcGIS error")
                    if isinstance(error, dict)
                    else str(error)
                )
                raise RuntimeError(f"ArcGIS query failed for {entry.key}: {message}")
            return payload
        except (TimeoutError, URLError, OSError, json.JSONDecodeError):
            if attempt >= max_retries:
                raise
            sleep(retry_backoff_seconds * (2**attempt))
    raise AssertionError("retry loop must return or raise")


def _write_arcgis_checkpoint(
    path: Path,
    *,
    signature: dict[str, object],
    next_offset: int,
    records: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "signature": signature,
                "next_offset": next_offset,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _HillsboroughDownloadParser(HTMLParser):
    """Extract the current all-sales postback and ASP.NET form state."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden: dict[str, str] = {}
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self.candidates: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "input" and (attributes.get("type") or "").lower() == "hidden":
            name = attributes.get("name")
            if name:
                self.hidden[name] = attributes.get("value", "") or ""
        if tag == "a":
            self._anchor_href = attributes.get("href")
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._anchor_href is None:
            return
        filename = "".join(self._anchor_text).strip()
        match = re.search(r"__doPostBack\('([^']+)'", self._anchor_href)
        if match and re.fullmatch(r"allsales_[\d_]+\.zip", filename, flags=re.IGNORECASE):
            self.candidates.append((filename, match.group(1)))
        self._anchor_href = None
        self._anchor_text = []


def discover_hillsborough_download(html: str) -> tuple[str, str, dict[str, str]]:
    """Return the newest all-sales filename, postback target, and hidden form state."""

    parser = _HillsboroughDownloadParser()
    parser.feed(html)
    if not parser.candidates:
        raise RuntimeError("Hillsborough downloads page did not expose an allsales ZIP")

    def release_key(candidate: tuple[str, str]) -> tuple[int, int, int]:
        match = re.search(r"(\d{2})_(\d{2})_(\d{4})", candidate[0])
        return (0, 0, 0) if match is None else (int(match[3]), int(match[1]), int(match[2]))

    filename, target = max(parser.candidates, key=release_key)
    return filename, target, parser.hidden


class _CallableSession:
    """Adapt an injected urlopen-like callable to urllib's opener interface."""

    def __init__(self, callback: Callable[..., object]) -> None:
        self.callback = callback

    def open(self, request: Request, *, timeout: int) -> object:
        return self.callback(request, timeout=timeout)


def fetch_hillsborough_source(
    source: str | DataSource,
    raw_dir: Path,
    *,
    overwrite: bool = False,
    opener: Callable[..., object] = urlopen,
) -> FetchResult:
    """Download Hillsborough's current all-sales archive via ASP.NET postback.

    The publisher does not expose a durable direct link.  This function reads
    the live downloads page, discovers the all-sales row and its postback
    target, preserves the ASP.NET cookie and hidden form state, and streams the
    resulting ZIP atomically.
    """

    entry = get_source(source) if isinstance(source, str) else source
    if entry.transport != "aspnet_postback":
        raise ValueError(f"Source {entry.key!r} is not an ASP.NET postback source")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / entry.filename
    if destination.exists() and not overwrite:
        result = _existing_result(entry, destination, raw_dir=raw_dir)
        _write_manifest(raw_dir, result, entry)
        return result

    session: Any
    if opener is urlopen:
        session = build_opener(HTTPCookieProcessor(CookieJar()))
    else:
        session = _CallableSession(opener)

    landing_request = Request(entry.url, headers={"User-Agent": "offer-to-exit/0.2"})
    with session.open(landing_request, timeout=120) as response:
        html = response.read().decode("utf-8", errors="replace")
    published_filename, event_target, hidden = discover_hillsborough_download(html)

    form = {
        **hidden,
        "__EVENTTARGET": event_target,
        "__EVENTARGUMENT": "",
    }
    post_request = Request(
        entry.url,
        data=urlencode(form).encode("ascii"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": entry.url,
            "User-Agent": "offer-to-exit/0.2",
        },
        method="POST",
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{entry.filename}.", dir=raw_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with session.open(post_request, timeout=120) as response:
                first = response.read(4)
                if not first.startswith(b"PK"):
                    raise RuntimeError(
                        "Hillsborough postback did not return a ZIP archive for "
                        f"{published_filename}"
                    )
                temporary.write(first)
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    result = FetchResult(
        source=entry.key,
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        retrieved_at=datetime.now(tz=UTC).isoformat(),
        url=f"{entry.url}#{published_filename}",
    )
    _write_manifest(raw_dir, result, entry)
    return result


def _existing_result(
    entry: DataSource,
    destination: Path,
    *,
    raw_dir: Path,
) -> FetchResult:
    digest = sha256_file(destination)
    byte_count = destination.stat().st_size
    manifest_path = raw_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        prior = files.get(entry.key, {}) if isinstance(files, dict) else {}
        if (
            isinstance(prior, dict)
            and prior.get("sha256") == digest
            and prior.get("bytes") == byte_count
            and prior.get("retrieved_at")
        ):
            records = prior.get("records")
            return FetchResult(
                source=entry.key,
                path=str(destination),
                bytes=byte_count,
                sha256=digest,
                retrieved_at=str(prior["retrieved_at"]),
                url=str(prior.get("url", entry.url)),
                records=int(records) if records is not None else None,
            )
    retrieved_at = datetime.fromtimestamp(destination.stat().st_mtime, tz=UTC).isoformat()
    return FetchResult(
        source=entry.key,
        path=str(destination),
        bytes=byte_count,
        sha256=digest,
        retrieved_at=retrieved_at,
        url=entry.url,
    )


def _write_manifest(
    raw_dir: Path,
    result: FetchResult,
    source: DataSource,
    *,
    query: dict[str, object] | None = None,
) -> None:
    manifest_path = raw_dir / "manifest.json"
    manifest: dict[str, object] = {"schema_version": 1, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.setdefault("files", {})
    assert isinstance(files, dict)
    record: dict[str, object] = {
        **asdict(result),
        "title": source.title,
        "publisher": source.publisher,
        "landing_page": source.landing_page,
        "attribution": source.attribution,
        "redistribution": source.redistribution,
        "contains_pii": source.contains_pii,
        "transport": source.transport,
        "query_order_field": source.query_order_field,
        "page_size": source.page_size,
        "catalog_out_fields": list(source.out_fields),
    }
    if query is not None:
        record["query"] = query
    files[source.key] = record
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
