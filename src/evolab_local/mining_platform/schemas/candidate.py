from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CandidateFieldValue(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    candidate_field_id: str
    paper_id: str
    record_scope: str = "device"
    record_id: str
    field_name: str
    field_label: str
    mined_value: str | None = None
    reviewed_value: str | None = None
    confidence: float | None = None
    confidence_json: dict[str, Any] = Field(default_factory=dict)
    evidence_anchor_id: str | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    field_status: str = "pending"
    created_at: str | None = None
    updated_at: str | None = None


class CandidateFieldUpdate(BaseModel):
    reviewed_value: str | None = None
    field_status: str | None = None
    actor: str = "local_user"
    message: str | None = None


class CandidateSeedResult(BaseModel):
    paper_id: str
    raw_record_count: int
    field_count: int
    evidence_count: int
