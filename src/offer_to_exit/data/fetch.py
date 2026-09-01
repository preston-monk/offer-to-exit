"""Streaming retrieval with local checksums and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from offer_to_exit.data.catalog import DataSource, get_source


@dataclass(frozen=True, slots=True)
class FetchResult:
    source: str
    path: str
    bytes: int
    sha256: str
    retrieved_at: str
    url: str


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
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / entry.filename
    if destination.exists() and not overwrite:
        retrieved_at = datetime.fromtimestamp(
            destination.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        result = FetchResult(
            source=entry.key,
            path=str(destination),
            bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            retrieved_at=retrieved_at,
            url=entry.url,
        )
        _write_manifest(raw_dir, result, entry)
        return result

    request = Request(entry.url, headers={"User-Agent": "offer-to-exit/0.1"})
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
        retrieved_at=datetime.now(tz=timezone.utc).isoformat(),
        url=entry.url,
    )
    _write_manifest(raw_dir, result, entry)
    return result


def _write_manifest(raw_dir: Path, result: FetchResult, source: DataSource) -> None:
    manifest_path = raw_dir / "manifest.json"
    manifest: dict[str, object] = {"schema_version": 1, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.setdefault("files", {})
    assert isinstance(files, dict)
    files[source.key] = {
        **asdict(result),
        "title": source.title,
        "publisher": source.publisher,
        "landing_page": source.landing_page,
        "attribution": source.attribution,
        "redistribution": source.redistribution,
        "contains_pii": source.contains_pii,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
