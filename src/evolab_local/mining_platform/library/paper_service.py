from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from urllib.parse import unquote

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.ingest.pdf_downloader_adapter import (
    paper_id_from_doi,
    papers_from_pdf_downloader_manifest,
)
from evolab_local.mining_platform.library.paper_registry import write_paper_registry
from evolab_local.mining_platform.schemas.paper import Paper, PaperIngestResult
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import PaperRepository, ReviewEventRepository


class PaperService:
    _registry_init_lock = Lock()

    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.repository = PaperRepository(self.database)
        self.review_events = ReviewEventRepository(self.database)

    def init_runtime(self) -> None:
        self.config.ensure_dirs()
        self.database.init_db()
        registry_path = self.config.paths.paper_registry_path
        if registry_path.exists():
            return
        with self._registry_init_lock:
            if not registry_path.exists():
                write_paper_registry(registry_path, self.repository.list())

    def ingest_from_pdf_downloader(self, domain: str = "unknown") -> PaperIngestResult:
        self.init_runtime()
        papers, skipped_count = papers_from_pdf_downloader_manifest(
            self.config.pdf_downloader.manifest_path,
            self.config.project_root,
            sqlite_path=self.config.pdf_downloader.sqlite_path,
            domain=domain,
        )
        imported: list[Paper] = []
        for paper in papers:
            imported.append(self.repository.upsert(paper))
        write_paper_registry(self.config.paths.paper_registry_path, self.repository.list())
        return PaperIngestResult(
            imported_count=len(imported),
            skipped_count=skipped_count,
            papers=imported,
        )

    def list_papers(self) -> list[Paper]:
        self.init_runtime()
        return self.repository.list()

    def register_uploaded_pdf(
        self,
        *,
        paper_id: str,
        doi: str,
        pdf_path: Path,
        pdf_sha256: str,
        pdf_size_bytes: int,
        title: str | None,
        domain: str,
        source: str = "data_mining_agent_upload",
    ) -> Paper:
        self.init_runtime()
        paper = self.repository.upsert(
            Paper(
                paper_id=paper_id,
                doi=doi,
                title=title,
                pdf_path=pdf_path.as_posix(),
                pdf_sha256=pdf_sha256,
                pdf_size_bytes=pdf_size_bytes,
                source=source,
                download_status="completed",
                domain=domain,
            )
        )
        write_paper_registry(self.config.paths.paper_registry_path, self.repository.list())
        return paper

    def get_paper(self, paper_id: str) -> Paper | None:
        self.init_runtime()
        return self.repository.get(self.normalize_paper_id(paper_id))

    def get_pdf_path(self, paper_id: str) -> Path | None:
        paper = self.get_paper(paper_id)
        if not paper:
            return None
        path = Path(paper.pdf_path)
        if not path.is_absolute():
            path = self.config.project_root / path
        return path if path.exists() else None

    def exclude_review_article(
        self,
        paper_id: str,
        *,
        actor: str = "local_user",
        message: str | None = None,
    ) -> Paper | None:
        self.init_runtime()
        normalized_paper_id = self.normalize_paper_id(paper_id)
        before = self.repository.get(normalized_paper_id)
        if not before:
            return None
        if before.review_status == "confirmed":
            raise ValueError("A confirmed paper cannot be excluded without first removing its final record.")
        updated = self.repository.set_review_status(
            normalized_paper_id,
            "excluded",
            reason="review_article",
        )
        if not updated:
            return None
        self.review_events.add(
            paper_id=normalized_paper_id,
            event_type="paper_excluded",
            actor=actor,
            message=message or "Excluded review article from OLED extraction",
            before_json=json.dumps(before.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            after_json=json.dumps(updated.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
        )
        return updated

    def restore_excluded_paper(
        self,
        paper_id: str,
        *,
        actor: str = "local_user",
        message: str | None = None,
    ) -> Paper | None:
        self.init_runtime()
        normalized_paper_id = self.normalize_paper_id(paper_id)
        before = self.repository.get(normalized_paper_id)
        if not before:
            return None
        if before.review_status != "excluded":
            raise ValueError("Only an excluded paper can be restored.")
        updated = self.repository.set_review_status(
            normalized_paper_id,
            "needs_review",
            reason="restored_from_review_exclusion",
        )
        if not updated:
            return None
        self.review_events.add(
            paper_id=normalized_paper_id,
            event_type="paper_exclusion_restored",
            actor=actor,
            message=message or "Restored paper to the review queue",
            before_json=json.dumps(before.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            after_json=json.dumps(updated.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
        )
        return updated

    @staticmethod
    def normalize_paper_id(value: str) -> str:
        stripped = value.strip()
        if "/" in stripped or "%" in stripped:
            return paper_id_from_doi(unquote(stripped))
        return stripped
