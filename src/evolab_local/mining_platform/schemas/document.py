from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocumentBlock(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    paper_id: str
    block_id: str
    page_id: int
    block_index: int
    block_type: str = "text"
    text: str
    bbox: list[float] = Field(default_factory=list)
    source: str = "pymupdf"
    created_at: str | None = None


class ParseResult(BaseModel):
    paper_id: str
    parser: str
    page_count: int
    block_count: int
    document_path: str
    blocks_path: str
    status: str = "parsed"
    error_message: str | None = None
