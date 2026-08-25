from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from evolab_local.mining_platform.core.paths import display_path
from evolab_local.mining_platform.schemas.paper import Paper


YEAR_RE = re.compile(r"(19|20)\d{2}")


def paper_id_from_doi(doi: str) -> str:
    return quote(doi.strip().lower(), safe="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pdf_downloader_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "tasks", "papers"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def load_pdf_downloader_sqlite_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'download_tasks'"
        ).fetchone()
        if not table:
            return {}
        available_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(download_tasks)").fetchall()
        }
        wanted_columns = [
            column
            for column in (
                "doi",
                "status",
                "publisher",
                "journal",
                "pdf_path",
                "completed_at",
            )
            if column in available_columns
        ]
        if "doi" not in wanted_columns:
            return {}
        rows = conn.execute(f"SELECT {', '.join(wanted_columns)} FROM download_tasks").fetchall()
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        values = dict(row)
        doi = values.get("doi")
        if isinstance(doi, str):
            metadata[doi.strip().lower()] = values
    return metadata


def _resolve_pdf_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def _infer_year(item: dict[str, Any]) -> int | None:
    for key in ("year", "publication_year", "published_year"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for key in ("completed_at", "created_at", "updated_at"):
        value = item.get(key)
        if isinstance(value, str):
            match = YEAR_RE.search(value)
            if match:
                return int(match.group(0))
    return None


def papers_from_pdf_downloader_manifest(
    manifest_path: Path,
    project_root: Path,
    sqlite_path: Path | None = None,
    domain: str = "unknown",
) -> tuple[list[Paper], int]:
    papers: list[Paper] = []
    skipped_count = 0
    sqlite_metadata = load_pdf_downloader_sqlite_metadata(sqlite_path) if sqlite_path else {}
    for item in load_pdf_downloader_manifest(manifest_path):
        doi = item.get("doi")
        if not isinstance(doi, str):
            skipped_count += 1
            continue
        doi = doi.strip().lower()
        metadata = sqlite_metadata.get(doi, {})
        status = item.get("status") or metadata.get("status")
        if status != "completed":
            skipped_count += 1
            continue
        raw_pdf_path = item.get("pdf_path") or metadata.get("pdf_path")
        if not isinstance(raw_pdf_path, str):
            skipped_count += 1
            continue
        pdf_path = _resolve_pdf_path(raw_pdf_path, project_root)
        if not pdf_path.exists():
            skipped_count += 1
            continue
        journal = item.get("journal") or metadata.get("journal")
        publisher = item.get("publisher") or metadata.get("publisher")
        papers.append(
            Paper(
                paper_id=paper_id_from_doi(doi),
                doi=doi,
                title=item.get("title") if isinstance(item.get("title"), str) else None,
                journal=journal if isinstance(journal, str) else None,
                publisher=publisher if isinstance(publisher, str) else None,
                year=_infer_year(item),
                pdf_path=display_path(pdf_path, project_root),
                pdf_sha256=sha256_file(pdf_path),
                pdf_size_bytes=pdf_path.stat().st_size,
                source="pdf_downloader",
                download_status="completed",
                domain=domain,
            )
        )
    return papers, skipped_count
