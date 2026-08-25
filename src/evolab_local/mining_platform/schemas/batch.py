from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evolab_local.mining_platform.schemas.paper import Paper


class BatchJob(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    paper_id: str
    doi: str
    source_pdf_path: str | None = None
    inbox_pdf_path: str
    pdf_sha256: str
    pdf_size_bytes: int
    status: str = "registered"
    current_stage: str = "registered"
    last_completed_stage: str | None = None
    retry_count: int = 0
    max_retries: int = 2
    error_message: str | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class BatchScanResult(BaseModel):
    scanned_count: int = 0
    registered_count: int = 0
    existing_count: int = 0
    skipped_unstable_count: int = 0
    jobs: list[BatchJob] = Field(default_factory=list)


class BatchImportResult(BaseModel):
    requested_count: int = 0
    copied_count: int = 0
    existing_count: int = 0
    missing_count: int = 0
    skipped_count: int = 0
    registered_count: int = 0
    jobs: list[BatchJob] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class BatchWorkerRunResult(BaseModel):
    processed_count: int = 0
    review_ready_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    jobs: list[BatchJob] = Field(default_factory=list)
    phase_timings: dict[str, float] = Field(default_factory=dict)
    material_metrics: dict[str, Any] = Field(default_factory=dict)


class MaterialStage3PlanItem(BaseModel):
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    material_label: str | None = None
    category: str
    route: str
    link_status: str | None = None
    reason: str | None = None


class MaterialStage3Plan(BaseModel):
    paper_count: int = 0
    skipped_paper_count: int = 0
    material_count: int = 0
    core_material_count: int = 0
    terminal_scope_count: int = 0
    local_resolved_count: int = 0
    public_pending_count: int = 0
    items: list[MaterialStage3PlanItem] = Field(default_factory=list)


class BatchReportJob(BaseModel):
    batch_index: int
    batch_number: int
    position: int
    paper_id: str
    doi: str
    title: str | None = None
    journal: str | None = None
    publisher: str | None = None
    year: int | None = None
    status: str
    current_stage: str
    last_completed_stage: str | None = None
    retry_count: int = 0
    max_retries: int = 0
    error_message: str | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    total_seconds: float = 0.0
    is_confirmed: bool = False
    is_excluded: bool = False
    final_record_count: int = 0
    candidate_run_count: int = 0
    latest_candidate_status: str | None = None


class BatchReportSummary(BaseModel):
    batch_id: str
    batch_index: int
    batch_number: int
    batch_size: int
    total_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    processed_count: int = 0
    review_ready_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    confirmed_count: int = 0
    excluded_count: int = 0
    total_stage_seconds: float = 0.0
    average_stage_seconds: float = 0.0
    slowest_stage_seconds: float = 0.0
    failed_paper_ids: list[str] = Field(default_factory=list)


class BatchRunReport(BaseModel):
    report_id: str
    generated_at: str
    summary: BatchReportSummary
    run_options: dict[str, Any] = Field(default_factory=dict)
    phase_timings: dict[str, float] = Field(default_factory=dict)
    material_metrics: dict[str, Any] = Field(default_factory=dict)
    jobs: list[BatchReportJob] = Field(default_factory=list)
    json_path: str | None = None
    markdown_path: str | None = None


class BatchReviewSummary(BaseModel):
    batch_id: str
    batch_index: int
    batch_number: int
    batch_size: int
    offset: int
    total_count: int
    confirmed_count: int = 0
    excluded_count: int = 0
    review_ready_count: int = 0
    running_count: int = 0
    failed_count: int = 0
    registered_count: int = 0
    status: str = "pending"
    first_doi: str | None = None
    last_doi: str | None = None
    all_confirmed: bool = False
    all_resolved: bool = False


class BatchReviewPaper(BaseModel):
    batch_index: int
    position: int
    paper: Paper
    job: BatchJob
    is_confirmed: bool = False
    is_excluded: bool = False
    final_record_count: int = 0


class BatchReviewOverview(BaseModel):
    batch_size: int
    total_jobs: int
    total_batches: int
    current_batch_index: int | None = None
    confirmed_batch_count: int = 0
    resolved_batch_count: int = 0
    batches: list[BatchReviewSummary] = Field(default_factory=list)


class BatchReviewDetail(BaseModel):
    summary: BatchReviewSummary
    papers: list[BatchReviewPaper] = Field(default_factory=list)
