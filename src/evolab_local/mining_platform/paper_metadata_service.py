from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from evolab_local.mining_platform.core.config import MiningPlatformConfig, PaperMetadataConfig
from evolab_local.mining_platform.library.paper_registry import write_paper_registry
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.paper import Paper
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import PaperRepository


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    source: str
    title: str | None = None
    journal: str | None = None
    publisher: str | None = None
    year: int | None = None
    raw_cache_path: str | None = None


@dataclass(frozen=True, slots=True)
class PaperMetadataEnrichmentResult:
    paper_id: str
    doi: str
    status: str
    source: str | None = None
    title: str | None = None
    journal: str | None = None
    publisher: str | None = None
    year: int | None = None
    updated_fields: tuple[str, ...] = ()
    error_message: str | None = None


class PaperMetadataClientProtocol(Protocol):
    def fetch(self, doi: str) -> PaperMetadata | None: ...


class OpenAlexCrossrefMetadataClient:
    def __init__(self, config: PaperMetadataConfig, *, cache_dir: Path) -> None:
        self.config = config
        self.cache_dir = cache_dir

    def fetch(self, doi: str) -> PaperMetadata | None:
        openalex = self._fetch_openalex(doi)
        crossref: PaperMetadata | None = None
        if not openalex or not _metadata_has_all_fields(openalex):
            crossref = self._fetch_crossref(doi)
        if openalex and crossref:
            return _merge_metadata(openalex, crossref)
        if openalex and _metadata_has_any_field(openalex):
            return openalex
        return crossref

    def _fetch_openalex(self, doi: str) -> PaperMetadata | None:
        params: dict[str, str] = {}
        if self.config.mailto:
            params["mailto"] = self.config.mailto
        if self.config.openalex_api_key:
            params["api_key"] = self.config.openalex_api_key
        url = f"{self.config.openalex_base_url.rstrip('/')}/works/doi:{doi}"
        try:
            with self._client() as client:
                response = client.get(url, params=params)
            cache_path = self._cache_path("openalex", doi)
            if response.status_code == httpx.codes.NOT_FOUND:
                self._write_cache(cache_path, {"source": "openalex", "doi": doi, "not_found": True})
                return None
            response.raise_for_status()
            data = response.json()
            self._write_cache(cache_path, data)
            return _parse_openalex_metadata(data, cache_path)
        except httpx.HTTPError:
            return None

    def _fetch_crossref(self, doi: str) -> PaperMetadata | None:
        headers = {"User-Agent": "evolab-local mining-platform"}
        params: dict[str, str] = {}
        if self.config.mailto:
            headers["User-Agent"] = f"evolab-local mining-platform (mailto:{self.config.mailto})"
            params["mailto"] = self.config.mailto
        url = f"{self.config.crossref_base_url.rstrip('/')}/works/{doi}"
        try:
            with self._client(headers=headers) as client:
                response = client.get(url, params=params)
            cache_path = self._cache_path("crossref", doi)
            if response.status_code == httpx.codes.NOT_FOUND:
                self._write_cache(cache_path, {"source": "crossref", "doi": doi, "not_found": True})
                return None
            response.raise_for_status()
            data = response.json()
            self._write_cache(cache_path, data)
            return _parse_crossref_metadata(data, cache_path)
        except httpx.HTTPError:
            return None

    def _client(self, headers: dict[str, str] | None = None) -> httpx.Client:
        return httpx.Client(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
            trust_env=False,
            headers=headers,
        )

    def _cache_path(self, source: str, doi: str) -> Path:
        safe = doi.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{source}_{safe}.json"

    @staticmethod
    def _write_cache(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class PaperMetadataEnrichmentService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        metadata_client: PaperMetadataClientProtocol | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.metadata_client = metadata_client or OpenAlexCrossrefMetadataClient(
            config.external_services.paper_metadata,
            cache_dir=config.paths.runtime_dir / "metadata_cache",
        )

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def enrich_paper(self, paper_id: str, *, force: bool = False) -> PaperMetadataEnrichmentResult:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        paper = self.papers.get(normalized_paper_id)
        if not paper:
            return PaperMetadataEnrichmentResult(
                paper_id=normalized_paper_id,
                doi=paper_id,
                status="not_found",
                error_message="Paper not found.",
            )
        if not force and _metadata_complete(paper):
            return _result_from_paper(paper, status="skipped_complete")
        try:
            metadata = self.metadata_client.fetch(paper.doi)
        except Exception as exc:
            return _result_from_paper(paper, status="failed", error_message=str(exc))
        if not metadata:
            return _result_from_paper(paper, status="not_found")

        updates: dict[str, object] = {}
        for field in ("title", "journal", "publisher", "year"):
            current = getattr(paper, field)
            candidate = getattr(metadata, field)
            if candidate is not None and (force or current in (None, "")):
                updates[field] = candidate
        if not updates:
            return _result_from_paper(paper, status="skipped_no_new_fields", source=metadata.source)
        updated = self.papers.upsert(paper.model_copy(update=updates))
        write_paper_registry(self.config.paths.paper_registry_path, self.papers.list())
        return _result_from_paper(
            updated,
            status="updated",
            source=metadata.source,
            updated_fields=tuple(sorted(updates)),
        )

    def enrich_missing(self, *, limit: int | None = None, force: bool = False) -> list[PaperMetadataEnrichmentResult]:
        self.init_runtime()
        candidates = [paper for paper in self.papers.list() if force or not _metadata_complete(paper)]
        if limit is not None:
            candidates = candidates[: max(0, limit)]
        return [self.enrich_paper(paper.paper_id, force=force) for paper in candidates]


def _metadata_complete(paper: Paper) -> bool:
    return bool(paper.title and paper.journal and paper.publisher and paper.year)


def _result_from_paper(
    paper: Paper,
    *,
    status: str,
    source: str | None = None,
    updated_fields: tuple[str, ...] = (),
    error_message: str | None = None,
) -> PaperMetadataEnrichmentResult:
    return PaperMetadataEnrichmentResult(
        paper_id=paper.paper_id,
        doi=paper.doi,
        status=status,
        source=source,
        title=paper.title,
        journal=paper.journal,
        publisher=paper.publisher,
        year=paper.year,
        updated_fields=updated_fields,
        error_message=error_message,
    )


def _metadata_has_any_field(metadata: PaperMetadata) -> bool:
    return any([metadata.title, metadata.journal, metadata.publisher, metadata.year])


def _metadata_has_all_fields(metadata: PaperMetadata) -> bool:
    return bool(metadata.title and metadata.journal and metadata.publisher and metadata.year)


def _merge_metadata(primary: PaperMetadata, fallback: PaperMetadata) -> PaperMetadata:
    return PaperMetadata(
        source=f"{primary.source}+{fallback.source}",
        title=primary.title or fallback.title,
        journal=primary.journal or fallback.journal,
        publisher=primary.publisher or fallback.publisher,
        year=primary.year or fallback.year,
        raw_cache_path=primary.raw_cache_path or fallback.raw_cache_path,
    )


def _parse_openalex_metadata(data: dict, cache_path: Path) -> PaperMetadata:
    primary = data.get("primary_location") or {}
    source = primary.get("source") or {}
    host_venue = data.get("host_venue") or {}
    journal = _text(source.get("display_name")) or _text(host_venue.get("display_name"))
    publisher = (
        _text(source.get("publisher"))
        or _text(source.get("host_organization_name"))
        or _text(host_venue.get("publisher"))
    )
    return PaperMetadata(
        source="openalex",
        title=_text(data.get("title")) or _text(data.get("display_name")),
        journal=journal,
        publisher=publisher,
        year=_int_or_none(data.get("publication_year")),
        raw_cache_path=cache_path.as_posix(),
    )


def _parse_crossref_metadata(data: dict, cache_path: Path) -> PaperMetadata:
    message = data.get("message") or data
    return PaperMetadata(
        source="crossref",
        title=_first_text(message.get("title")),
        journal=_first_text(message.get("container-title")) or _first_text(message.get("short-container-title")),
        publisher=_text(message.get("publisher")),
        year=_crossref_year(message),
        raw_cache_path=cache_path.as_posix(),
    )


def _crossref_year(message: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            year = _int_or_none(date_parts[0][0])
            if year:
                return year
    return None


def _first_text(value: object) -> str | None:
    if isinstance(value, list) and value:
        return _text(value[0])
    return _text(value)


def _text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
