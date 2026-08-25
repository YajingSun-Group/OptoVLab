from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MaterialGlobal(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    global_material_id: str
    canonical_name: str | None = None
    material_class: str = "unknown"
    representation_type: str = "unknown"
    raw_smiles: str | None = None
    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    source: str = "manual"
    source_detail: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    review_status: str = "candidate"
    created_at: str
    updated_at: str
    confirmed_at: str | None = None


class MaterialAlias(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    alias_id: str
    global_material_id: str
    alias_text: str
    normalized_alias: str
    alias_type: str = "unknown"
    source_paper_id: str | None = None
    source: str = "manual"
    confidence: float | None = None
    review_status: str = "candidate"
    created_at: str
    updated_at: str


class PaperMaterialLink(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    paper_material_link_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    global_material_id: str | None = None
    match_method: str = "none"
    match_confidence: float | None = None
    match_status: str = "unresolved"
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    confirmed_at: str | None = None


class MaterialResolutionTask(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    material_mentions: list[str] = Field(default_factory=list)
    material_context: dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"
    status: str = "pending"
    assigned_strategy: str = "unresolved"
    current_stage: str = "unresolved"
    next_action: str = "resolve"
    retry_count: int = 0
    stage_timings: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class MaterialStructureTriageItem(BaseModel):
    paper_material_id: str
    material_label: str | None = None
    category: str
    requires_structure: bool = True
    requires_public_resolution: bool = True
    has_accepted_or_matched_structure: bool = False
    should_run_ocsr: bool = True
    confidence: float | None = None
    reason: str | None = None
    rule: str | None = None
    link_status: str | None = None
    matched_alias: str | None = None


class MaterialStructureTriageResult(BaseModel):
    paper_id: str
    candidate_run_id: str | None = None
    material_count: int = 0
    needs_ocsr_count: int = 0
    skipped_count: int = 0
    items: list[MaterialStructureTriageItem] = Field(default_factory=list)

    @property
    def should_run_ocsr(self) -> bool:
        return self.needs_ocsr_count > 0


class MaterialStructureCandidate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    structure_candidate_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    provider: str
    resolver_name: str
    query_text: str
    query_type: str = "name"
    source_identifier: str | None = None
    source_url: str | None = None
    canonical_name: str | None = None
    material_class: str = "unknown"
    representation_type: str = "small_molecule"
    raw_smiles: str | None = None
    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    synonyms: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    status: str = "pending_review"
    created_at: str
    updated_at: str


class MaterialIdentityJudgment(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    judgment_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    structure_candidate_id: str
    provider: str
    model: str
    prompt_version: str = "material_identity_judge_v1"
    verdict: str = "insufficient_evidence"
    confidence: float | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    recommended_action: str = "manual_review"
    deterministic_checks: dict[str, Any] = Field(default_factory=dict)
    input_context: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    error_message: str | None = None
    created_at: str
    updated_at: str


class MaterialIdentityEvidenceRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    evidence_run_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    trigger_judgment_id: str | None = None
    provider: str
    model: str
    prompt_version: str = "material_identity_evidence_v1"
    strategy: str = "targeted_web_text_enrichment"
    query_plan: list[str] = Field(default_factory=list)
    status: str = "running"
    generated_candidate_ids: list[str] = Field(default_factory=list)
    recommended_next_action: str = "manual_review"
    raw_response: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class MaterialIdentityEvidenceItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    evidence_item_id: str
    evidence_run_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    source_type: str = "web_search"
    source_tier: str = "C"
    source_title: str | None = None
    source_url: str | None = None
    query_text: str | None = None
    excerpt: str | None = None
    alias: str | None = None
    full_name: str | None = None
    cas_number: str | None = None
    pubchem_cid: str | None = None
    explicitly_linked: bool = False
    confidence: float | None = None
    extraction: dict[str, Any] = Field(default_factory=dict)
    raw_source: dict[str, Any] = Field(default_factory=dict)
    review_status: str = "pending_review"
    reviewed_by: str | None = None
    review_note: str | None = None
    reviewed_at: str | None = None
    created_at: str
    updated_at: str


class MaterialIdentityEvidenceReviewAction(BaseModel):
    decision: str
    actor: str = "local_user"
    message: str | None = None


class MaterialReviewAction(BaseModel):
    actor: str = "local_user"
    message: str | None = None
    global_material_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class MaterialStructureEditAction(MaterialReviewAction):
    smiles: str


class MaterialManualStructureAction(MaterialReviewAction):
    smiles: str
    reviewed_name: str | None = None
    full_name_in_paper: str | None = None
    material_class: str = "small_molecule_organic"
    representation_type: str = "small_molecule"
    source_note: str | None = None
    source_url: str | None = None


class PaperMaterialNameReviewAction(BaseModel):
    actor: str = "local_user"
    message: str | None = None
    reviewed_name: str | None = None
    reviewed_full_name_in_paper: str | None = None
    reviewed_abbreviation: str | None = None
    reviewed_normalized_name: str | None = None
    reviewed_canonical_name: str | None = None


class PaperMaterialNameReview(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    review_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    reviewed_name: str | None = None
    reviewed_full_name_in_paper: str | None = None
    reviewed_abbreviation: str | None = None
    reviewed_normalized_name: str | None = None
    reviewed_canonical_name: str | None = None
    review_status: str = "corrected"
    actor: str = "local_user"
    message: str | None = None
    source: str = "manual_review"
    source_detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class PaperMaterialNameSuggestion(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    suggestion_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    agent_name: str = "material_name_agent_v1"
    original_name: str | None = None
    suggested_name: str
    suggested_full_name_in_paper: str | None = None
    suggested_abbreviation: str | None = None
    suggested_normalized_name: str | None = None
    suggested_canonical_name: str | None = None
    confidence: float | None = None
    reason: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: str = "suggested"
    created_at: str
    updated_at: str




class MaterialPropertyCandidate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    property_candidate_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    global_material_id: str | None = None
    property_name: str
    property_category: str = "unknown"
    value_numeric: float | None = None
    value_text: str | None = None
    value_raw: str | None = None
    unit: str | None = None
    normalized_value_numeric: float | None = None
    normalized_unit: str | None = None
    condition: dict[str, Any] = Field(default_factory=dict)
    method: str | None = None
    source_type: str = "unknown"
    evidence_text: str | None = None
    llm_evidence_text: str | None = None
    source_block_text: str | None = None
    evidence_anchor: dict[str, Any] = Field(default_factory=dict)
    provider: str = "unknown"
    model: str | None = None
    prompt_version: str = "material_property_miner_v1"
    confidence: float | None = None
    status: str = "pending_review"
    error_message: str | None = None
    created_at: str
    updated_at: str


class MaterialPropertyReview(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    review_id: str
    property_candidate_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    decision: str
    reviewed_property_name: str
    reviewed_value_numeric: float | None = None
    reviewed_value_text: str | None = None
    reviewed_unit: str | None = None
    reviewed_condition: dict[str, Any] = Field(default_factory=dict)
    reviewed_evidence_anchor: dict[str, Any] = Field(default_factory=dict)
    actor: str = "local_user"
    message: str | None = None
    created_at: str


class MaterialPropertyReviewEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    property_candidate_id: str | None = None
    event_type: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    actor: str = "local_user"
    message: str | None = None
    created_at: str


class MaterialPropertyReviewAction(BaseModel):
    actor: str = "local_user"
    message: str | None = None
    reviewed_property_name: str | None = None
    reviewed_value_numeric: float | None = None
    reviewed_value_text: str | None = None
    reviewed_value_raw: str | None = None
    reviewed_unit: str | None = None
    reviewed_condition: dict[str, Any] = Field(default_factory=dict)
    reviewed_evidence_anchor: dict[str, Any] = Field(default_factory=dict)


class MaterialPropertyManualAddAction(BaseModel):
    actor: str = "local_user"
    message: str | None = None
    property_name: str
    property_category: str | None = None
    value_numeric: float | None = None
    value_text: str | None = None
    value_raw: str | None = None
    unit: str | None = None
    normalized_value_numeric: float | None = None
    normalized_unit: str | None = None
    condition: dict[str, Any] = Field(default_factory=dict)
    method: str | None = None
    source_type: str = "manual"
    evidence_text: str | None = None
    evidence_anchor: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = 1.0


class MaterialReviewEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    structure_candidate_id: str | None = None
    global_material_id: str | None = None
    action: str
    actor: str
    message: str | None = None
    before_candidate_status: str | None = None
    after_candidate_status: str | None = None
    before_link: dict[str, Any] | None = None
    after_link: dict[str, Any] | None = None
    before_task: dict[str, Any] | None = None
    after_task: dict[str, Any] | None = None
    before_candidate: dict[str, Any] | None = None
    after_candidate: dict[str, Any] | None = None
    created_global_material_id: str | None = None
    created_at: str


class MaterialAutoDecision(BaseModel):
    paper_material_id: str
    structure_candidate_id: str | None = None
    action: str
    reason: str
    verdict: str | None = None
    recommended_action: str | None = None
    confidence: float | None = None
    applied: bool = False
    error_message: str | None = None


class MaterialAutoDecisionResult(BaseModel):
    paper_id: str
    candidate_run_id: str | None = None
    dry_run: bool = False
    actor: str = "automation_policy"
    accepted_count: int = 0
    rejected_count: int = 0
    skipped_count: int = 0
    decisions: list[MaterialAutoDecision] = Field(default_factory=list)


class ChemicalFigureBlock(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    figure_block_id: str
    paper_id: str
    mineru_run_id: str
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
    heuristic_tags: list[str] = Field(default_factory=list)
    confidence: float | None = None
    status: str = "pending_classification"
    source_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class MaterialUsage(BaseModel):
    device_label: str | None = None
    layer_index: int | None = None
    layer_role: str | None = None
    layer_name: str | None = None
    component_role: str | None = None
    material_mention: str | None = None
    usage_type: str = "component"


class PaperLocalMaterial(BaseModel):
    paper_material_id: str
    entity_path: str
    entity_label: str | None = None
    mention_list: list[str] = Field(default_factory=list)
    full_name_in_paper: str | None = None
    normalized_name: str | None = None
    canonical_name: str | None = None
    abbreviation: str | None = None
    paper_specific_label: str | None = None
    material_class: str | None = None
    smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    structure_source: str | None = None
    structure_confidence: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    used_in: list[MaterialUsage] = Field(default_factory=list)


class PaperMaterialStructureBundle(BaseModel):
    paper_id: str
    candidate_run_id: str | None = None
    materials: list[PaperLocalMaterial] = Field(default_factory=list)
    material_name_reviews: list[PaperMaterialNameReview] = Field(default_factory=list)
    material_name_suggestions: list[PaperMaterialNameSuggestion] = Field(default_factory=list)
    links: list[PaperMaterialLink] = Field(default_factory=list)
    tasks: list[MaterialResolutionTask] = Field(default_factory=list)
    structure_candidates: list[MaterialStructureCandidate] = Field(default_factory=list)
    identity_judgments: list[MaterialIdentityJudgment] = Field(default_factory=list)
    identity_evidence_runs: list[MaterialIdentityEvidenceRun] = Field(default_factory=list)
    identity_evidence_items: list[MaterialIdentityEvidenceItem] = Field(default_factory=list)
    material_review_events: list[MaterialReviewEvent] = Field(default_factory=list)
    property_candidates: list[MaterialPropertyCandidate] = Field(default_factory=list)
    property_reviews: list[MaterialPropertyReview] = Field(default_factory=list)
    property_review_events: list[MaterialPropertyReviewEvent] = Field(default_factory=list)
    global_materials: list[MaterialGlobal] = Field(default_factory=list)


class MaterialIdentityJudgmentBatchResult(BaseModel):
    paper_id: str
    candidate_run_id: str | None = None
    judgments: list[MaterialIdentityJudgment] = Field(default_factory=list)


class DecimerOCSRBatchResult(BaseModel):
    paper_id: str
    agent_run_id: str
    eligible_binding_count: int = 0
    candidates: list[MaterialStructureCandidate] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
