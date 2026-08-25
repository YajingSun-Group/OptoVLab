from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MaterialAgentRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_run_id: str
    paper_id: str
    status: str
    strategy: str = "foundation"
    source_candidate_run_id: str | None = None
    mineru_run_id: str | None = None
    material_count: int = 0
    visual_block_count: int = 0
    tool_summary: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class DocumentVisualBlock(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    visual_block_id: str
    paper_id: str
    mineru_run_id: str
    collected_by_agent_run_id: str | None = None
    content_index: int
    content_type: str
    sub_type: str | None = None
    page_idx: int | None = None
    page_id: int | None = None
    bbox: list[float] = Field(default_factory=list)
    img_path: str | None = None
    resolved_img_path: str | None = None
    image_exists: bool = False
    caption: str | None = None
    nearby_text: str | None = None
    source_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class FigureTriageResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    triage_result_id: str
    agent_run_id: str
    visual_block_id: str
    paper_id: str
    provider: str
    model: str
    contains_molecular_structures: bool = False
    image_role: str = "unknown"
    has_clean_structure_depictions: bool = False
    has_orbital_overlay: bool = False
    has_energy_level_diagram: bool = False
    has_device_stack: bool = False
    should_run_decimer_segmentation: bool = False
    label_candidates: list[str] = Field(default_factory=list)
    related_paper_material_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    error_message: str | None = None
    created_at: str
    updated_at: str


class MaterialAgentFoundationResult(BaseModel):
    run: MaterialAgentRun
    visual_blocks: list[DocumentVisualBlock] = Field(default_factory=list)


class FigureTriageBatchResult(BaseModel):
    run: MaterialAgentRun
    results: list[FigureTriageResult] = Field(default_factory=list)


class MoleculeCrop(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    crop_id: str
    paper_id: str
    agent_run_id: str
    triage_result_id: str
    visual_block_id: str
    segment_index: int
    bbox: list[float] = Field(default_factory=list)
    source_image_path: str
    crop_path: str
    width: int | None = None
    height: int | None = None
    segmentation_confidence: float | None = None
    validation_json: dict[str, Any] = Field(default_factory=dict)
    raw_segment: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending_validation"
    created_at: str
    updated_at: str


class DecimerSegmentationBatchResult(BaseModel):
    run: MaterialAgentRun
    crops: list[MoleculeCrop] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MoleculeCropValidation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    validation_id: str
    crop_id: str
    paper_id: str
    agent_run_id: str
    visual_block_id: str
    provider: str
    model: str
    is_molecular_depiction: bool = False
    is_single_molecule: bool = False
    is_complete_structure: bool = False
    has_benign_highlight: bool = False
    is_ocsr_readable: bool = False
    has_blocking_interference: bool = False
    has_orbital_overlay: bool = False
    has_excess_annotation: bool = False
    has_multiple_structures: bool = False
    has_reaction_arrow: bool = False
    has_non_structural_graphics: bool = False
    should_run_ocsr: bool = False
    confidence: float | None = None
    reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    error_message: str | None = None
    created_at: str
    updated_at: str


class MoleculeCropValidationBatchResult(BaseModel):
    run: MaterialAgentRun
    validations: list[MoleculeCropValidation] = Field(default_factory=list)


class MoleculeLabelBinding(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    binding_id: str
    paper_id: str
    candidate_run_id: str
    agent_run_id: str
    crop_id: str
    visual_block_id: str
    provider: str
    model: str
    source_figure_path: str
    highlighted_source_figure_path: str
    crop_path: str
    caption_text: str | None = None
    nearby_text: str | None = None
    triage_label_candidates: list[str] = Field(default_factory=list)
    candidate_materials: list[dict[str, Any]] = Field(default_factory=list)
    model_observed_label: str | None = None
    model_label_source: str = "unknown"
    model_proposed_paper_material_id: str | None = None
    model_alternative_paper_material_ids: list[str] = Field(default_factory=list)
    model_decision: str = "failed"
    model_confidence: float | None = None
    model_reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    error_message: str | None = None
    reviewed_paper_material_id: str | None = None
    reviewed_observed_label: str | None = None
    review_status: str = "pending_review"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None
    created_at: str
    updated_at: str


class MoleculeLabelBindingBatchResult(BaseModel):
    run: MaterialAgentRun
    bindings: list[MoleculeLabelBinding] = Field(default_factory=list)


class MaterialOCSRPipelineResult(BaseModel):
    run: MaterialAgentRun
    triage_count: int = 0
    crop_count: int = 0
    validation_count: int = 0
    binding_count: int = 0
    eligible_binding_count: int = 0
    ocsr_candidate_count: int = 0
    skipped_count: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MoleculeLabelBindingReviewAction(BaseModel):
    action: str
    actor: str = "local_user"
    reviewed_paper_material_id: str | None = None
    reviewed_observed_label: str | None = None
    message: str | None = None


class MoleculeLabelBindingReviewEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    binding_id: str
    paper_id: str
    crop_id: str
    action: str
    actor: str
    message: str | None = None
    before_reviewed_paper_material_id: str | None = None
    after_reviewed_paper_material_id: str | None = None
    before_observed_label: str | None = None
    after_observed_label: str | None = None
    before_review_status: str
    after_review_status: str
    created_at: str


class VLMCallLog(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vlm_call_id: str
    paper_id: str
    agent_run_id: str
    stage: str
    input_entity_type: str
    input_entity_id: str
    provider: str
    model: str
    prompt_version: str
    input_image_paths: list[str] = Field(default_factory=list)
    input_context: dict[str, Any] = Field(default_factory=dict)
    parsed_response: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    status: str = "running"
    error_message: str | None = None
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
