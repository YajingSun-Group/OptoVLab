from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evolab_local.mining_platform.schemas.evidence import EvidenceAnchor
from evolab_local.mining_platform.schemas.mining_result import MiningResultValidationReport


class CandidateIngestionRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    candidate_run_id: str
    paper_id: str
    template_id: str
    template_version: str
    source_name: str
    source_version: str | None = None
    status: str
    validation_report: dict[str, Any] = Field(default_factory=dict)
    mining_result: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


class CandidateEntity(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    candidate_entity_id: str
    candidate_run_id: str
    paper_id: str
    template_id: str
    entity_type: str
    entity_path: str
    entity_label: str | None = None
    parent_entity_id: str | None = None
    sort_order: int = 0
    source_json: dict[str, Any] = Field(default_factory=dict)
    review_status: str = "pending"
    created_at: str
    updated_at: str


class CandidateValue(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    candidate_value_id: str
    candidate_run_id: str
    candidate_entity_id: str
    paper_id: str
    template_id: str
    template_field_path: str
    concrete_path: str
    field_label: str
    data_type: str
    value_json: Any = None
    reviewed_value_json: Any = None
    display_value: str | None = None
    evidence_anchor_ids: list[str] = Field(default_factory=list)
    status: str = "pending"
    created_at: str
    updated_at: str


class CandidateIngestionResult(BaseModel):
    run: CandidateIngestionRun
    validation_report: MiningResultValidationReport
    entity_count: int = 0
    value_count: int = 0
    evidence_anchor_count: int = 0


class CandidateValueUpdate(BaseModel):
    reviewed_value_json: Any = None
    status: str | None = None
    actor: str = "local_user"
    message: str | None = None


class CandidateValueReviewEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    candidate_value_id: str
    candidate_run_id: str
    candidate_entity_id: str
    paper_id: str
    template_id: str
    template_field_path: str
    concrete_path: str
    action: str
    actor: str
    message: str | None = None
    original_value_json: Any = None
    before_reviewed_value_json: Any = None
    after_reviewed_value_json: Any = None
    before_status: str
    after_status: str
    created_at: str


class CandidateReviewV2Bundle(BaseModel):
    run: CandidateIngestionRun | None = None
    entities: list[CandidateEntity] = Field(default_factory=list)
    values: list[CandidateValue] = Field(default_factory=list)
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list)
    template: dict[str, Any] | None = None


class CandidateFinalRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    final_record_id: str
    paper_id: str
    candidate_run_id: str
    template_id: str
    template_version: str
    final_json: dict[str, Any] = Field(default_factory=dict)
    source_candidate_value_ids: list[str] = Field(default_factory=list)
    confirmed_by: str
    status: str = "confirmed"
    created_at: str
    updated_at: str
    confirmed_at: str


class CandidateFinalConfirmResult(BaseModel):
    paper_id: str
    final_record: CandidateFinalRecord
    final_value_count: int
