from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    paper_id: str
    doi: str
    title: str | None = None
    journal: str | None = None
    publisher: str | None = None
    year: int | None = None
    pdf_path: str
    pdf_sha256: str
    pdf_size_bytes: int
    source: str = "pdf_downloader"
    download_status: str = "completed"
    parse_status: str = "pending"
    mining_status: str = "pending"
    review_status: str = "not_started"
    review_reason: str | None = None
    domain: str = "unknown"
    created_at: str | None = None
    updated_at: str | None = None


class PaperIngestResult(BaseModel):
    imported_count: int = 0
    skipped_count: int = 0
    papers: list[Paper] = Field(default_factory=list)
