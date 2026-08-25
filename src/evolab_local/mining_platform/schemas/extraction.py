from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evolab_local.mining_platform.schemas.device_record import DeviceRecordFields


class FieldEvidence(BaseModel):
    value: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    page_id: int | None = None
    source_text: str | None = None


class ExtractionRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    paper_id: str
    extractor_name: str
    extractor_version: str
    status: str
    input_block_count: int = 0
    raw_record_count: int = 0
    error_message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class RawDeviceCandidate(DeviceRecordFields):
    evidence_block_ids: list[str] = Field(default_factory=list)
    field_evidence: dict[str, FieldEvidence] = Field(default_factory=dict)
    confidence: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class DeviceRecordRaw(RawDeviceCandidate):
    model_config = ConfigDict(from_attributes=True)

    raw_record_id: str
    run_id: str
    paper_id: str
    review_status: str = "pending"
    reviewed_record_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ExtractionResult(BaseModel):
    run: ExtractionRun
    raw_records: list[DeviceRecordRaw] = Field(default_factory=list)
