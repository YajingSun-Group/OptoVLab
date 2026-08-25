from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvidenceAnchor(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    evidence_anchor_id: str
    paper_id: str
    page_id: int | None = None
    block_id: str | None = None
    bbox: list[float] = Field(default_factory=list)
    source_text: str | None = None
    source_type: str = "text"
    created_at: str | None = None
