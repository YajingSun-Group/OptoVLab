from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import shutil
from threading import BoundedSemaphore, Lock
import time
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.core.paths import display_path
from evolab_local.mining_platform.ingest.pdf_downloader_adapter import (
    paper_id_from_doi,
    sha256_file,
)
from evolab_local.mining_platform.library.paper_registry import write_paper_registry
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_identity_judge_service import (
    MaterialIdentityJudgeService,
)
from evolab_local.mining_platform.material_auto_decision_service import MaterialAutoDecisionService
from evolab_local.mining_platform.material_completion_service import PaperMaterialCompletionService
from evolab_local.mining_platform.material_public_resolver_service import (
    MaterialPublicResolverService,
    is_plausible_public_structure_candidate,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
)
from evolab_local.mining_platform.material_stage3_planner_service import (
    MaterialStage3PlannerService,
)
from evolab_local.mining_platform.material_structure_triage_service import (
    MaterialStructureTriageService,
)
from evolab_local.mining_platform.mining.llm_mining_service import LLMMiningService
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService
from evolab_local.mining_platform.paper_metadata_service import PaperMetadataEnrichmentService
from evolab_local.mining_platform.paper_review_policy import no_device_review_reason
from evolab_local.mining_platform.schemas.batch import (
    BatchImportResult,
    BatchJob,
    BatchReportJob,
    BatchReportSummary,
    BatchReviewDetail,
    BatchReviewOverview,
    BatchReviewPaper,
    BatchReviewSummary,
    BatchRunReport,
    BatchScanResult,
    BatchWorkerRunResult,
    MaterialStage3PlanItem,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialIdentityJudgment,
    MaterialStructureCandidate,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.schemas.paper import Paper
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    BatchJobRepository,
    CandidateIngestionRepository,
    LLMMiningRunRepository,
    MinerUParseRunRepository,
    PaperRepository,
)


@dataclass
class _Stage3PaperFlowResult:
    paper_id: str
    public_attempted_count: int = 0
    public_errors: list[dict[str, str]] = field(default_factory=list)
    judge_material_count: int = 0
    judge_errors: list[dict[str, str]] = field(default_factory=list)
    auto_accepted_count: int = 0
    auto_rejected_count: int = 0
    auto_errors: list[dict[str, str]] = field(default_factory=list)
    residual_items: list[MaterialStage3PlanItem] = field(default_factory=list)
    public_human_items: list[MaterialStage3PlanItem] = field(default_factory=list)
    unavailable_items: list[MaterialStage3PlanItem] = field(default_factory=list)
    ocsr_errors: list[dict[str, str]] = field(default_factory=list)
    ocsr_human_review_count: int = 0


_VISUAL_FALLBACK_IDENTITY_VERDICTS = {
    "ambiguous",
    "insufficient_evidence",
    "conflict",
    "rejected",
}
_VISUAL_FALLBACK_IDENTITY_ACTIONS = {
    "search_more_evidence",
    "reject_candidate",
}


def _public_candidate_should_yield_to_visual_fallback(
    candidate: MaterialStructureCandidate,
    judgments: list[MaterialIdentityJudgment],
) -> bool:
    """Let an unresolved public lead coexist with an independent OCSR candidate."""
    candidate_judgments = [
        judgment
        for judgment in judgments
        if judgment.structure_candidate_id == candidate.structure_candidate_id
        and judgment.status == "completed"
    ]
    if not candidate_judgments:
        return False
    latest = max(
        candidate_judgments,
        key=lambda judgment: (judgment.updated_at, judgment.created_at, judgment.judgment_id),
    )
    return (
        latest.verdict in _VISUAL_FALLBACK_IDENTITY_VERDICTS
        or latest.recommended_action in _VISUAL_FALLBACK_IDENTITY_ACTIONS
    )


class _ConcurrentPhaseTimer:
    """Track both the overlapping wall window and cumulative task time."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._first_started: float | None = None
        self._last_finished: float | None = None
        self._task_seconds = 0.0
        self._task_count = 0

    def run(self, action):
        started = perf_counter()
        with self._lock:
            if self._first_started is None or started < self._first_started:
                self._first_started = started
            self._task_count += 1
        try:
            return action()
        finally:
            finished = perf_counter()
            with self._lock:
                self._last_finished = max(self._last_finished or finished, finished)
                self._task_seconds += finished - started

    @property
    def wall_seconds(self) -> float:
        if self._first_started is None or self._last_finished is None:
            return 0.0
        return round(self._last_finished - self._first_started, 3)

    @property
    def task_seconds(self) -> float:
        return round(self._task_seconds, 3)

    @property
    def task_count(self) -> int:
        return self._task_count


class BatchWorkerService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        mineru_service: MinerUParseService | None = None,
        llm_service: LLMMiningService | None = None,
        material_resolution_service: MaterialResolutionService | None = None,
        material_public_resolver_service: MaterialPublicResolverService | None = None,
        material_identity_judge_service: MaterialIdentityJudgeService | None = None,
        material_auto_decision_service: MaterialAutoDecisionService | None = None,
        material_agent_service: MaterialStructureAgentService | None = None,
        material_structure_triage_service: MaterialStructureTriageService | None = None,
        material_completion_service: PaperMaterialCompletionService | None = None,
        material_stage3_planner_service: MaterialStage3PlannerService | None = None,
        metadata_service: PaperMetadataEnrichmentService | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.jobs = BatchJobRepository(self.database)
        self.mineru_runs = MinerUParseRunRepository(self.database)
        self.llm_runs = LLMMiningRunRepository(self.database)
        self.candidate_runs = CandidateIngestionRepository(self.database)
        self.mineru_service = mineru_service or MinerUParseService(config)
        self.llm_service = llm_service or LLMMiningService(config)
        self.material_resolution_service = material_resolution_service or MaterialResolutionService(
            config
        )
        self.material_public_resolver_service = (
            material_public_resolver_service or MaterialPublicResolverService(config)
        )
        self.material_identity_judge_service = (
            material_identity_judge_service or MaterialIdentityJudgeService(config)
        )
        self.material_auto_decision_service = (
            material_auto_decision_service or MaterialAutoDecisionService(config)
        )
        self.material_agent_service = material_agent_service or MaterialStructureAgentService(
            config
        )
        self.material_structure_triage_service = (
            material_structure_triage_service or MaterialStructureTriageService(config)
        )
        self.material_completion_service = (
            material_completion_service or PaperMaterialCompletionService(config)
        )
        self.material_stage3_planner_service = (
            material_stage3_planner_service
            or MaterialStage3PlannerService(
                config,
                material_resolution_service=self.material_resolution_service,
                triage_service=self.material_structure_triage_service,
            )
        )
        self.metadata_service = metadata_service or PaperMetadataEnrichmentService(config)
        self._initialized = False

    def init_runtime(self) -> None:
        if self._initialized:
            return
        self.config.ensure_dirs()
        self.database.init_db()
        write_paper_registry(self.config.paths.paper_registry_path, self.papers.list())
        self._initialized = True

    def import_pdfs_from_csv(
        self,
        *,
        csv_path: Path,
        source_pdf_dir: Path,
        doi_column: str = "doi_encode",
        limit: int | None = None,
        overwrite: bool = False,
        domain: str | None = None,
    ) -> BatchImportResult:
        self.init_runtime()
        requested = _unique_csv_values(csv_path, doi_column=doi_column)
        if limit is not None:
            requested = requested[:limit]
        result = BatchImportResult(requested_count=len(requested))
        for encoded in requested:
            stem = _pdf_stem(encoded)
            source_path = source_pdf_dir / f"{stem}.pdf"
            if not source_path.exists():
                result.missing_count += 1
                result.missing.append(source_path.as_posix())
                continue
            dest_path = self.config.paths.inbox_pdfs_dir / source_path.name
            if dest_path.exists() and not overwrite:
                result.existing_count += 1
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                result.copied_count += 1
            job = self.register_inbox_pdf(
                dest_path,
                source_pdf_path=source_path,
                domain=domain or self.config.batch_worker.default_domain,
                write_registry=False,
            )
            result.jobs.append(job)
        result.registered_count = len(result.jobs)
        write_paper_registry(self.config.paths.paper_registry_path, self.papers.list())
        return result

    def scan_inbox_pdfs(
        self,
        *,
        limit: int | None = None,
        domain: str | None = None,
        stable_file_seconds: float | None = None,
    ) -> BatchScanResult:
        self.init_runtime()
        stable_seconds = (
            self.config.batch_worker.stable_file_seconds
            if stable_file_seconds is None
            else stable_file_seconds
        )
        pdf_paths = sorted(self.config.paths.inbox_pdfs_dir.glob("*.pdf"))
        if limit is not None:
            pdf_paths = pdf_paths[:limit]
        result = BatchScanResult(scanned_count=len(pdf_paths))
        for pdf_path in pdf_paths:
            if not _is_stable_file(pdf_path, stable_seconds):
                result.skipped_unstable_count += 1
                continue
            paper_id = paper_id_from_doi(_doi_from_pdf_name(pdf_path.name))
            existing_before = self.jobs.get_by_paper(paper_id)
            job = self.register_inbox_pdf(
                pdf_path,
                domain=domain or self.config.batch_worker.default_domain,
                write_registry=False,
            )
            if existing_before and existing_before.pdf_sha256 == job.pdf_sha256:
                result.existing_count += 1
            else:
                result.registered_count += 1
            result.jobs.append(job)
        write_paper_registry(self.config.paths.paper_registry_path, self.papers.list())
        return result

    def register_inbox_pdf(
        self,
        pdf_path: Path,
        *,
        source_pdf_path: Path | None = None,
        domain: str = "oled",
        write_registry: bool = True,
    ) -> BatchJob:
        self.init_runtime()
        resolved_pdf_path = pdf_path.resolve()
        doi = _doi_from_pdf_name(resolved_pdf_path.name)
        paper_id = paper_id_from_doi(doi)
        digest = sha256_file(resolved_pdf_path)
        size = resolved_pdf_path.stat().st_size
        paper = Paper(
            paper_id=paper_id,
            doi=doi,
            pdf_path=display_path(resolved_pdf_path, self.config.project_root),
            pdf_sha256=digest,
            pdf_size_bytes=size,
            source="inbox_watcher",
            download_status="completed",
            domain=domain,
        )
        self.papers.upsert(paper)
        timestamp = now_iso()
        job = BatchJob(
            job_id=uuid4().hex,
            paper_id=paper_id,
            doi=doi,
            source_pdf_path=(source_pdf_path.resolve().as_posix() if source_pdf_path else None),
            inbox_pdf_path=display_path(resolved_pdf_path, self.config.project_root),
            pdf_sha256=digest,
            pdf_size_bytes=size,
            status="registered",
            current_stage="registered",
            retry_count=0,
            max_retries=self.config.batch_worker.max_retries,
            options={"domain": domain},
            created_at=timestamp,
            updated_at=timestamp,
        )
        stored = self.jobs.upsert(job)
        if write_registry:
            write_paper_registry(self.config.paths.paper_registry_path, self.papers.list())
        return stored

    def list_jobs(self, status: str | None = None, limit: int | None = None) -> list[BatchJob]:
        self.init_runtime()
        return self.jobs.list(status=status, limit=limit)

    def list_review_batches(self, *, batch_size: int | None = None) -> BatchReviewOverview:
        self.init_runtime()
        size = self._review_batch_size(batch_size)
        jobs = self.jobs.list_for_review_batches()
        final_record_counts = self._final_record_counts()
        papers_by_id = {paper.paper_id: paper for paper in self.papers.list()}
        summaries: list[BatchReviewSummary] = []
        for batch_index, offset in enumerate(range(0, len(jobs), size)):
            group = jobs[offset : offset + size]
            summaries.append(
                self._batch_summary(
                    group,
                    batch_index=batch_index,
                    batch_size=size,
                    offset=offset,
                    final_record_counts=final_record_counts,
                    papers_by_id=papers_by_id,
                )
            )
        current = next((item.batch_index for item in summaries if not item.all_resolved), None)
        return BatchReviewOverview(
            batch_size=size,
            total_jobs=len(jobs),
            total_batches=len(summaries),
            current_batch_index=current,
            confirmed_batch_count=sum(1 for item in summaries if item.all_confirmed),
            resolved_batch_count=sum(1 for item in summaries if item.all_resolved),
            batches=summaries,
        )

    def get_review_batch(
        self,
        batch_index: int,
        *,
        batch_size: int | None = None,
    ) -> BatchReviewDetail:
        self.init_runtime()
        size = self._review_batch_size(batch_size)
        jobs = self.jobs.list_for_review_batches()
        if batch_index < 0:
            raise ValueError("Batch index must be non-negative.")
        offset = batch_index * size
        group = jobs[offset : offset + size]
        if not group:
            raise ValueError(f"Review batch not found: {batch_index}")
        final_record_counts = self._final_record_counts()
        papers_by_id = {paper.paper_id: paper for paper in self.papers.list()}
        summary = self._batch_summary(
            group,
            batch_index=batch_index,
            batch_size=size,
            offset=offset,
            final_record_counts=final_record_counts,
            papers_by_id=papers_by_id,
        )
        papers: list[BatchReviewPaper] = []
        for position, job in enumerate(group, start=1):
            paper = papers_by_id.get(job.paper_id) or self.papers.get(job.paper_id)
            if not paper:
                continue
            final_count = final_record_counts.get(job.paper_id, 0)
            papers.append(
                BatchReviewPaper(
                    batch_index=batch_index,
                    position=position,
                    paper=paper,
                    job=job,
                    is_confirmed=self._paper_is_confirmed(paper, final_count),
                    is_excluded=self._paper_is_excluded(paper),
                    final_record_count=final_count,
                )
            )
        return BatchReviewDetail(summary=summary, papers=papers)

    def run_next_review_batch(
        self,
        *,
        batch_size: int | None = None,
        include_failed_retries: bool = False,
        template_id: str = "oled_device_v1",
        provider: str | None = None,
        model: str | None = None,
        run_parse: bool = True,
        run_device_mining: bool = True,
        run_material_resolution: bool = True,
        run_metadata_enrichment: bool | None = None,
        run_public_resolver: bool | None = None,
        run_identity_judge: bool | None = None,
        run_material_auto_decision: bool | None = None,
        run_visual_prep: bool | None = None,
        run_material_ocsr: bool | None = None,
    ) -> BatchWorkerRunResult:
        overview = self.list_review_batches(batch_size=batch_size)
        if overview.current_batch_index is None:
            return BatchWorkerRunResult()
        detail = self.get_review_batch(
            overview.current_batch_index,
            batch_size=overview.batch_size,
        )
        return self._run_review_batch_detail(
            detail,
            include_failed_retries=include_failed_retries,
            template_id=template_id,
            provider=provider,
            model=model,
            run_parse=run_parse,
            run_device_mining=run_device_mining,
            run_material_resolution=run_material_resolution,
            run_metadata_enrichment=run_metadata_enrichment,
            run_public_resolver=run_public_resolver,
            run_identity_judge=run_identity_judge,
            run_material_auto_decision=run_material_auto_decision,
            run_visual_prep=run_visual_prep,
            run_material_ocsr=run_material_ocsr,
        )

    def run_review_batch(
        self,
        *,
        batch_index: int,
        batch_size: int | None = None,
        include_failed_retries: bool = False,
        template_id: str = "oled_device_v1",
        provider: str | None = None,
        model: str | None = None,
        run_parse: bool = True,
        run_device_mining: bool = True,
        run_material_resolution: bool = True,
        run_metadata_enrichment: bool | None = None,
        run_public_resolver: bool | None = None,
        run_identity_judge: bool | None = None,
        run_material_auto_decision: bool | None = None,
        run_visual_prep: bool | None = None,
        run_material_ocsr: bool | None = None,
    ) -> BatchWorkerRunResult:
        detail = self.get_review_batch(batch_index, batch_size=batch_size)
        return self._run_review_batch_detail(
            detail,
            include_failed_retries=include_failed_retries,
            template_id=template_id,
            provider=provider,
            model=model,
            run_parse=run_parse,
            run_device_mining=run_device_mining,
            run_material_resolution=run_material_resolution,
            run_metadata_enrichment=run_metadata_enrichment,
            run_public_resolver=run_public_resolver,
            run_identity_judge=run_identity_judge,
            run_material_auto_decision=run_material_auto_decision,
            run_visual_prep=run_visual_prep,
            run_material_ocsr=run_material_ocsr,
        )

    def run_review_batch_staged(
        self,
        *,
        batch_index: int,
        batch_size: int | None = None,
        include_failed_retries: bool = False,
        template_id: str = "oled_device_v1",
        provider: str | None = None,
        model: str | None = None,
        parse_concurrency: int = 1,
        llm_concurrency: int = 10,
        material_concurrency: int = 5,
        run_parse: bool = True,
        run_device_mining: bool = True,
        run_material_resolution: bool = True,
        run_metadata_enrichment: bool | None = None,
        run_public_resolver: bool | None = None,
        run_identity_judge: bool | None = None,
        run_material_auto_decision: bool | None = None,
        run_visual_prep: bool | None = None,
        run_material_ocsr: bool | None = None,
    ) -> BatchWorkerRunResult:
        """Run a review batch by stage barriers instead of paper-by-paper serial order.

        Stage 1 prepares metadata/MinerU parse; Stage 2 mines all parsed papers with LLM;
        Stage 3 runs material resolution/judgment/OCSR for papers with completed mining.
        """
        detail = self.get_review_batch(batch_index, batch_size=batch_size)
        run_metadata = (
            self.config.batch_worker.run_metadata_enrichment
            if run_metadata_enrichment is None
            else run_metadata_enrichment
        )
        runnable: list[BatchJob] = []
        for item in detail.papers:
            if item.is_confirmed:
                continue
            job = item.job
            if job.status == "registered":
                runnable.append(job)
            elif (
                include_failed_retries
                and job.status == "failed"
                and job.retry_count < job.max_retries
            ):
                runnable.append(job)
        result = BatchWorkerRunResult(
            processed_count=len(runnable),
            skipped_count=max(0, len(detail.papers) - len(runnable)),
        )
        if not runnable:
            metadata_metrics, metadata_seconds = self._enrich_missing_batch_metadata(
                [item.paper for item in detail.papers],
                enabled=run_metadata,
                max_workers=min(4, parse_concurrency),
            )
            result.phase_timings["metadata_enrichment"] = metadata_seconds
            result.material_metrics.update(metadata_metrics)
            result.jobs = [item.job for item in detail.papers]
            return result

        run_public = (
            self.config.batch_worker.run_public_resolver
            if run_public_resolver is None
            else run_public_resolver
        )
        run_identity = (
            self.config.batch_worker.run_identity_judge
            if run_identity_judge is None
            else run_identity_judge
        )
        run_auto_decision = (
            self.config.batch_worker.run_material_auto_decision
            if run_material_auto_decision is None
            else run_material_auto_decision
        )
        run_visual = (
            self.config.batch_worker.run_visual_prep if run_visual_prep is None else run_visual_prep
        )
        run_ocsr = (
            self.config.batch_worker.run_material_ocsr
            if run_material_ocsr is None
            else run_material_ocsr
        )

        prepared = self._run_staged_batch_phase_timed(
            result,
            "stage1_metadata_parse",
            runnable,
            max_workers=parse_concurrency,
            runner=lambda job: self._run_staged_metadata_parse(
                job,
                run_metadata_enrichment=run_metadata,
                run_parse=run_parse,
            ),
        )
        llm_inputs = [job for job in prepared if job.status != "failed"]
        mined = self._run_staged_batch_phase_timed(
            result,
            "stage2_device_mining",
            llm_inputs,
            max_workers=llm_concurrency,
            runner=lambda job: self._run_staged_device_mining(
                job,
                template_id=template_id,
                provider=provider,
                model=model,
                run_device_mining=run_device_mining,
            ),
        )
        material_inputs = [job for job in mined if job.status != "failed"]
        if material_inputs:
            stage3_result = self.refresh_review_batch_materials(
                batch_index=batch_index,
                batch_size=batch_size,
                provider=provider,
                model=model,
                material_concurrency=material_concurrency,
                run_metadata_enrichment=run_metadata,
                run_material_resolution=run_material_resolution,
                run_public_resolver=run_public,
                run_identity_judge=run_identity,
                run_material_auto_decision=run_auto_decision,
                run_visual_prep=run_visual,
                run_material_ocsr=run_ocsr,
            )
            result.phase_timings.update(stage3_result.phase_timings)
            result.phase_timings["stage3_material_pipeline"] = stage3_result.phase_timings.get(
                "stage3_material_refresh",
                0.0,
            )
            result.material_metrics = stage3_result.material_metrics

        final_jobs = []
        for item in detail.papers:
            job = self.jobs.get(item.job.job_id) or item.job
            final_jobs.append(job)
        result.jobs = final_jobs
        runnable_ids = {job.job_id for job in runnable}
        for job in final_jobs:
            if job.job_id not in runnable_ids:
                continue
            if job.status == "review_ready":
                result.review_ready_count += 1
            elif job.status == "failed":
                result.failed_count += 1
        return result

    def refresh_review_batch_materials(
        self,
        *,
        batch_index: int,
        batch_size: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        material_concurrency: int = 5,
        run_metadata_enrichment: bool | None = None,
        run_material_resolution: bool = True,
        run_public_resolver: bool | None = None,
        run_identity_judge: bool | None = None,
        run_material_auto_decision: bool | None = None,
        run_visual_prep: bool | None = None,
        run_material_ocsr: bool | None = None,
        public_concurrency: int | None = None,
        judge_concurrency: int | None = None,
        vlm_concurrency: int | None = None,
        ocsr_paper_concurrency: int | None = None,
        decimer_segmentation_concurrency: int | None = None,
        decimer_ocsr_concurrency: int | None = None,
    ) -> BatchWorkerRunResult:
        if material_concurrency <= 0:
            raise ValueError("material_concurrency must be positive.")
        for name, value in (
            ("decimer_segmentation_concurrency", decimer_segmentation_concurrency),
            ("decimer_ocsr_concurrency", decimer_ocsr_concurrency),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive.")
        detail = self.get_review_batch(batch_index, batch_size=batch_size)
        run_metadata = (
            self.config.batch_worker.run_metadata_enrichment
            if run_metadata_enrichment is None
            else run_metadata_enrichment
        )
        metadata_metrics, metadata_seconds = self._enrich_missing_batch_metadata(
            [item.paper for item in detail.papers],
            enabled=run_metadata,
            max_workers=min(4, material_concurrency),
        )
        run_public = (
            self.config.batch_worker.run_public_resolver
            if run_public_resolver is None
            else run_public_resolver
        )
        run_identity = (
            self.config.batch_worker.run_identity_judge
            if run_identity_judge is None
            else run_identity_judge
        )
        run_auto_decision = (
            self.config.batch_worker.run_material_auto_decision
            if run_material_auto_decision is None
            else run_material_auto_decision
        )
        run_visual = (
            self.config.batch_worker.run_visual_prep if run_visual_prep is None else run_visual_prep
        )
        run_ocsr = (
            self.config.batch_worker.run_material_ocsr
            if run_material_ocsr is None
            else run_material_ocsr
        )
        active_jobs: list[BatchJob] = []
        for item in detail.papers:
            if item.is_confirmed:
                job = self.jobs.get(item.job.job_id) or item.job
                if job.status != "review_ready":
                    self._mark_review_ready(job)
                continue
            if self._paper_has_no_reviewable_device_data(item.paper.paper_id):
                self._confirm_unreviewable_device_data(item.paper.paper_id)
                self._mark_review_ready(item.job)
                continue
            if not self._has_completed_candidate_run(item.paper.paper_id):
                continue
            job = self.jobs.get(item.job.job_id) or item.job
            active_jobs.append(job if job.status == "running" else self._mark_running(job))
        result = BatchWorkerRunResult(
            processed_count=len(active_jobs),
            skipped_count=max(0, len(detail.papers) - len(active_jobs)),
        )
        result.phase_timings["metadata_enrichment"] = metadata_seconds
        result.material_metrics.update(metadata_metrics)
        if not active_jobs:
            result.jobs = [self.jobs.get(item.job.job_id) or item.job for item in detail.papers]
            result.review_ready_count = sum(job.status == "review_ready" for job in result.jobs)
            result.failed_count = sum(job.status == "failed" for job in result.jobs)
            return result

        stage3_started = perf_counter()

        started = perf_counter()
        plan = self.material_stage3_planner_service.plan_papers(
            [job.paper_id for job in active_jobs],
            refresh_local=run_material_resolution,
            max_workers=min(
                material_concurrency,
                self.config.batch_worker.material_plan_concurrency,
            ),
        )
        result.phase_timings["stage3_plan_local"] = round(perf_counter() - started, 3)
        result.material_metrics.update(
            {
                "paper_count": plan.paper_count,
                "material_count": plan.material_count,
                "core_material_count": plan.core_material_count,
                "terminal_scope_count": plan.terminal_scope_count,
                "local_resolved_count": plan.local_resolved_count,
                "public_pending_count": plan.public_pending_count,
            }
        )
        public_items = [item for item in plan.items if item.route == "public_resolution"]

        public_by_paper: dict[str, list[MaterialStage3PlanItem]] = {}
        for item in public_items:
            public_by_paper.setdefault(item.paper_id, []).append(item)
        public_timer = _ConcurrentPhaseTimer()
        judge_timer = _ConcurrentPhaseTimer()
        auto_timer = _ConcurrentPhaseTimer()
        visual_timer = _ConcurrentPhaseTimer()
        prefetch_timer = _ConcurrentPhaseTimer()
        public_limit = max(
            1,
            public_concurrency or self.config.batch_worker.material_public_concurrency,
        )
        judge_limit = max(
            1,
            judge_concurrency or self.config.batch_worker.material_judge_concurrency,
        )
        ocsr_limit = max(
            1,
            ocsr_paper_concurrency or self.config.batch_worker.material_ocsr_paper_concurrency,
        )
        public_semaphore = BoundedSemaphore(public_limit)
        judge_semaphore = BoundedSemaphore(judge_limit)
        ocsr_semaphore = BoundedSemaphore(ocsr_limit)
        with ThreadPoolExecutor(
            max_workers=max(
                1,
                self.config.batch_worker.material_visual_prefetch_concurrency,
            )
        ) as prefetch_executor:
            paper_results = _run_concurrent_items(
                active_jobs,
                max_workers=min(material_concurrency, len(active_jobs)),
                runner=lambda job: self._run_stage3_paper_flow(
                    paper_id=job.paper_id,
                    public_items=public_by_paper.get(job.paper_id, []),
                    provider=provider,
                    model=model,
                    run_public=run_public,
                    run_identity=run_identity,
                    run_auto_decision=run_auto_decision,
                    run_visual=run_visual,
                    run_ocsr=run_ocsr,
                    public_semaphore=public_semaphore,
                    judge_semaphore=judge_semaphore,
                    ocsr_semaphore=ocsr_semaphore,
                    prefetch_executor=prefetch_executor,
                    public_timer=public_timer,
                    judge_timer=judge_timer,
                    auto_timer=auto_timer,
                    visual_timer=visual_timer,
                    prefetch_timer=prefetch_timer,
                    vlm_concurrency=(
                        vlm_concurrency or self.config.batch_worker.material_vlm_concurrency
                    ),
                    decimer_segmentation_concurrency=(
                        decimer_segmentation_concurrency
                        or self.config.batch_worker.material_decimer_segmentation_concurrency
                    ),
                    decimer_ocsr_concurrency=(
                        decimer_ocsr_concurrency
                        or self.config.batch_worker.material_decimer_ocsr_concurrency
                    ),
                ),
            )
        flows = [
            flow
            for _, flow, error, _ in paper_results
            if error is None and isinstance(flow, _Stage3PaperFlowResult)
        ]
        paper_flow_errors = _concurrent_errors(paper_results)
        result.phase_timings.update(
            {
                "stage3_public_resolution": public_timer.wall_seconds,
                "stage3_identity_judge": judge_timer.wall_seconds,
                "stage3_auto_decision": auto_timer.wall_seconds,
                "stage3_visual_prefetch": prefetch_timer.wall_seconds,
                "stage3_visual_ocsr": visual_timer.wall_seconds,
            }
        )
        result.material_metrics.update(
            {
                "public_attempted_count": sum(flow.public_attempted_count for flow in flows),
                "public_error_count": sum(len(flow.public_errors) for flow in flows),
                "judge_material_count": sum(flow.judge_material_count for flow in flows),
                "judge_error_count": sum(len(flow.judge_errors) for flow in flows),
                "auto_accepted_count": sum(flow.auto_accepted_count for flow in flows),
                "auto_rejected_count": sum(flow.auto_rejected_count for flow in flows),
                "auto_error_count": sum(len(flow.auto_errors) for flow in flows),
                "public_human_review_count": sum(len(flow.public_human_items) for flow in flows),
                "structure_unavailable_count": sum(len(flow.unavailable_items) for flow in flows),
                "ocsr_pending_count": sum(len(flow.residual_items) for flow in flows),
                "ocsr_error_count": sum(len(flow.ocsr_errors) for flow in flows),
                "ocsr_human_review_count": sum(flow.ocsr_human_review_count for flow in flows),
                "human_review_count": sum(
                    len(flow.public_human_items) + flow.ocsr_human_review_count for flow in flows
                ),
                "paper_flow_error_count": len(paper_flow_errors),
                "scheduler_mode": "paper_dag",
                "phase_timings_overlap": True,
                "stage3_phase_task_seconds": {
                    "public_resolution": public_timer.task_seconds,
                    "identity_judge": judge_timer.task_seconds,
                    "auto_decision": auto_timer.task_seconds,
                    "visual_pipeline": visual_timer.task_seconds,
                },
            }
        )

        started = perf_counter()
        batch_completion = getattr(
            self.material_completion_service,
            "confirm_papers_if_materials_complete",
            None,
        )
        if callable(batch_completion):
            batch_completion([job.paper_id for job in active_jobs])
        else:
            for job in active_jobs:
                self.material_completion_service.confirm_paper_if_materials_complete(job.paper_id)
        self._mark_review_ready_many([self.jobs.get(job.job_id) or job for job in active_jobs])
        result.phase_timings["stage3_completion"] = round(perf_counter() - started, 3)
        self._sync_material_task_timings(active_jobs)
        result.phase_timings["stage3_material_refresh"] = round(
            perf_counter() - stage3_started,
            3,
        )

        final_jobs = [self.jobs.get(item.job.job_id) or item.job for item in detail.papers]
        result.jobs = final_jobs
        for job in final_jobs:
            if job.status == "review_ready":
                result.review_ready_count += 1
            elif job.status == "failed":
                result.failed_count += 1
        return result

    def write_batch_report(
        self,
        *,
        batch_index: int,
        batch_size: int | None = None,
        run_result: BatchWorkerRunResult | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> BatchRunReport:
        detail = self.get_review_batch(batch_index, batch_size=batch_size)
        generated_at = now_iso()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_id = f"{detail.summary.batch_id}-{timestamp}"
        jobs = [self._batch_report_job(item) for item in detail.papers]
        status_counts = Counter(job.status for job in jobs)
        totals = [job.total_seconds for job in jobs]
        failed_paper_ids = [job.paper_id for job in jobs if job.status == "failed"]
        summary = BatchReportSummary(
            batch_id=detail.summary.batch_id,
            batch_index=detail.summary.batch_index,
            batch_number=detail.summary.batch_number,
            batch_size=detail.summary.batch_size,
            total_count=detail.summary.total_count,
            status_counts=dict(sorted(status_counts.items())),
            processed_count=(run_result.processed_count if run_result else 0),
            review_ready_count=status_counts.get("review_ready", 0),
            failed_count=status_counts.get("failed", 0),
            skipped_count=(run_result.skipped_count if run_result else 0),
            confirmed_count=sum(1 for job in jobs if job.is_confirmed),
            excluded_count=sum(1 for job in jobs if job.is_excluded),
            total_stage_seconds=round(sum(totals), 3),
            average_stage_seconds=round(sum(totals) / len(totals), 3) if totals else 0.0,
            slowest_stage_seconds=max(totals) if totals else 0.0,
            failed_paper_ids=failed_paper_ids,
        )
        report = BatchRunReport(
            report_id=report_id,
            generated_at=generated_at,
            summary=summary,
            run_options=run_options or {},
            phase_timings=(run_result.phase_timings if run_result else {}),
            material_metrics=(run_result.material_metrics if run_result else {}),
            jobs=jobs,
        )
        self.config.paths.batch_reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.config.paths.batch_reports_dir / f"{report_id}.json"
        markdown_path = self.config.paths.batch_reports_dir / f"{report_id}.md"
        report = report.model_copy(
            update={
                "json_path": display_path(json_path, self.config.project_root),
                "markdown_path": display_path(markdown_path, self.config.project_root),
            }
        )
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(_render_batch_report_markdown(report), encoding="utf-8")
        return report

    def _run_review_batch_detail(
        self,
        detail: BatchReviewDetail,
        *,
        include_failed_retries: bool,
        template_id: str,
        provider: str | None,
        model: str | None,
        run_parse: bool,
        run_device_mining: bool,
        run_material_resolution: bool,
        run_metadata_enrichment: bool | None,
        run_public_resolver: bool | None,
        run_identity_judge: bool | None,
        run_material_auto_decision: bool | None,
        run_visual_prep: bool | None,
        run_material_ocsr: bool | None,
    ) -> BatchWorkerRunResult:
        runnable: list[BatchJob] = []
        for item in detail.papers:
            if item.is_confirmed:
                continue
            job = item.job
            if job.status == "registered":
                runnable.append(job)
            elif (
                include_failed_retries
                and job.status == "failed"
                and job.retry_count < job.max_retries
            ):
                runnable.append(job)
        result = BatchWorkerRunResult(skipped_count=max(0, len(detail.papers) - len(runnable)))
        if not runnable:
            result.jobs = [item.job for item in detail.papers]
            return result
        run_metadata = (
            self.config.batch_worker.run_metadata_enrichment
            if run_metadata_enrichment is None
            else run_metadata_enrichment
        )
        run_public = (
            self.config.batch_worker.run_public_resolver
            if run_public_resolver is None
            else run_public_resolver
        )
        run_identity = (
            self.config.batch_worker.run_identity_judge
            if run_identity_judge is None
            else run_identity_judge
        )
        run_auto_decision = (
            self.config.batch_worker.run_material_auto_decision
            if run_material_auto_decision is None
            else run_material_auto_decision
        )
        run_visual = (
            self.config.batch_worker.run_visual_prep if run_visual_prep is None else run_visual_prep
        )
        run_ocsr = (
            self.config.batch_worker.run_material_ocsr
            if run_material_ocsr is None
            else run_material_ocsr
        )
        for job in runnable:
            result.processed_count += 1
            try:
                completed = self._run_job(
                    job,
                    template_id=template_id,
                    provider=provider,
                    model=model,
                    run_parse=run_parse,
                    run_device_mining=run_device_mining,
                    run_material_resolution=run_material_resolution,
                    run_metadata_enrichment=run_metadata,
                    run_public_resolver=run_public,
                    run_identity_judge=run_identity,
                    run_material_auto_decision=run_auto_decision,
                    run_visual_prep=run_visual,
                    run_material_ocsr=run_ocsr,
                )
                result.jobs.append(completed)
                if completed.status == "review_ready":
                    result.review_ready_count += 1
                elif completed.status == "failed":
                    result.failed_count += 1
                else:
                    result.skipped_count += 1
            except Exception:
                failed = self.jobs.get(job.job_id) or job
                result.jobs.append(failed)
                result.failed_count += 1
        return result

    def _batch_report_job(self, item: BatchReviewPaper) -> BatchReportJob:
        job = self.jobs.get(item.job.job_id) or item.job
        paper = self.papers.get(job.paper_id) or item.paper
        candidate_runs = self.candidate_runs.list_runs_by_paper(job.paper_id)
        latest_candidate_status = candidate_runs[0].status if candidate_runs else None
        total_seconds = round(sum(float(value) for value in job.stage_timings.values()), 3)
        return BatchReportJob(
            batch_index=item.batch_index,
            batch_number=item.batch_index + 1,
            position=item.position,
            paper_id=job.paper_id,
            doi=job.doi,
            title=paper.title,
            journal=paper.journal,
            publisher=paper.publisher,
            year=paper.year,
            status=job.status,
            current_stage=job.current_stage,
            last_completed_stage=job.last_completed_stage,
            retry_count=job.retry_count,
            max_retries=job.max_retries,
            error_message=job.error_message,
            stage_timings=job.stage_timings,
            stage_errors=job.stage_errors,
            total_seconds=total_seconds,
            is_confirmed=item.is_confirmed,
            is_excluded=item.is_excluded,
            final_record_count=item.final_record_count,
            candidate_run_count=len(candidate_runs),
            latest_candidate_status=latest_candidate_status,
        )

    def _run_staged_batch_phase_timed(
        self,
        result: BatchWorkerRunResult,
        phase_name: str,
        jobs: list[BatchJob],
        *,
        max_workers: int,
        runner,
    ) -> list[BatchJob]:
        started = perf_counter()
        phase_result = self._run_staged_batch_phase(
            jobs,
            max_workers=max_workers,
            runner=runner,
        )
        result.phase_timings[phase_name] = round(perf_counter() - started, 3)
        return phase_result

    def _run_staged_batch_phase(
        self,
        jobs: list[BatchJob],
        *,
        max_workers: int,
        runner,
    ) -> list[BatchJob]:
        if not jobs:
            return []
        worker_count = max(1, min(max_workers, len(jobs)))
        if worker_count == 1:
            results: list[BatchJob] = []
            for job in jobs:
                try:
                    results.append(runner(job))
                except Exception:
                    results.append(self.jobs.get(job.job_id) or job)
            return results
        results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {executor.submit(runner, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results.append(future.result())
                except Exception:
                    results.append(self.jobs.get(job.job_id) or job)
        return results

    def _run_staged_metadata_parse(
        self,
        job: BatchJob,
        *,
        run_metadata_enrichment: bool,
        run_parse: bool,
    ) -> BatchJob:
        job = self.jobs.get(job.job_id) or job
        if job.status != "running":
            job = self._mark_running(job)
        try:
            if run_metadata_enrichment:
                job = self._run_stage(
                    job,
                    "metadata_enrich",
                    lambda: self.metadata_service.enrich_paper(job.paper_id),
                    skip_if=lambda: self._paper_metadata_complete(job.paper_id),
                )
            if run_parse:
                job = self._run_stage(
                    job,
                    "parse_mineru",
                    lambda: self.mineru_service.parse_paper(job.paper_id),
                    skip_if=lambda: self._has_completed_mineru(job.paper_id),
                )
            return job
        except Exception as exc:
            failed = self.jobs.get(job.job_id) or job
            self._mark_failed(failed, exc)
            raise

    def _run_staged_device_mining(
        self,
        job: BatchJob,
        *,
        template_id: str,
        provider: str | None,
        model: str | None,
        run_device_mining: bool,
    ) -> BatchJob:
        job = self.jobs.get(job.job_id) or job
        if job.status != "running":
            job = self._mark_running(job)
        try:
            if run_device_mining:
                job = self._run_stage(
                    job,
                    "device_mining_llm",
                    lambda: self.llm_service.mine_paper(
                        job.paper_id,
                        template_id=template_id,
                        provider=provider,
                        model=model,
                    ),
                    skip_if=lambda: self._has_completed_candidate_run(job.paper_id),
                )
            job = self._run_stage(
                job,
                "device_review_auto_confirm",
                lambda: self._confirm_unreviewable_device_data(job.paper_id),
                skip_if=lambda: not self._paper_has_no_reviewable_device_data(job.paper_id),
            )
            return job
        except Exception as exc:
            failed = self.jobs.get(job.job_id) or job
            self._mark_failed(failed, exc)
            raise

    def _run_staged_material_pipeline(
        self,
        job: BatchJob,
        *,
        provider: str | None,
        model: str | None,
        run_material_resolution: bool,
        run_public_resolver: bool,
        run_identity_judge: bool,
        run_material_auto_decision: bool,
        run_visual_prep: bool,
        run_material_ocsr: bool,
    ) -> BatchJob:
        job = self.jobs.get(job.job_id) or job
        if job.status != "running":
            job = self._mark_running(job)
        try:
            if self._paper_has_no_reviewable_device_data(job.paper_id):
                self._confirm_unreviewable_device_data(job.paper_id)
                return self._mark_review_ready(job)
            if run_material_resolution:
                job = self._run_stage(
                    job,
                    "material_resolve_local",
                    lambda: self.material_resolution_service.resolve_paper_materials(job.paper_id),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            job = self._run_stage(
                job,
                "material_structure_triage",
                lambda: self.material_structure_triage_service.triage_paper(job.paper_id),
                skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
            )
            if run_public_resolver:
                job = self._run_stage(
                    job,
                    "material_resolve_public",
                    lambda: self.material_public_resolver_service.resolve_paper_public(
                        job.paper_id
                    ),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_identity_judge:
                job = self._run_stage(
                    job,
                    "material_identity_judge",
                    lambda: self.material_identity_judge_service.judge_paper_candidates(
                        job.paper_id,
                        provider=provider,
                        model=model,
                    ),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_material_auto_decision:
                job = self._run_stage(
                    job,
                    "material_auto_decision",
                    lambda: self.material_auto_decision_service.apply_paper_auto_decisions(
                        job.paper_id
                    ),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_visual_prep:
                job = self._run_stage(
                    job,
                    "material_visual_prep",
                    lambda: self.material_agent_service.run_foundation(job.paper_id),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_material_ocsr:
                job = self._run_stage(
                    job,
                    "material_ocsr",
                    lambda: self.material_agent_service.run_ocsr_pipeline(
                        job.paper_id,
                        allow_unreviewed_matches=True,
                        min_model_confidence=(
                            self.config.batch_worker.material_ocsr_auto_match_min_confidence
                        ),
                    ),
                    skip_if=lambda: (
                        not self._has_completed_candidate_run(job.paper_id)
                        or not self._paper_needs_material_ocsr(job.paper_id)
                    ),
                )
            job = self._run_stage(
                job,
                "material_completion_check",
                lambda: self.material_completion_service.confirm_paper_if_materials_complete(
                    job.paper_id
                ),
                skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
            )
            return self._mark_review_ready(job)
        except Exception as exc:
            failed = self.jobs.get(job.job_id) or job
            self._mark_failed(failed, exc)
            raise

    def run_once(
        self,
        *,
        max_jobs: int = 1,
        include_failed_retries: bool = False,
        template_id: str = "oled_device_v1",
        provider: str | None = None,
        model: str | None = None,
        run_parse: bool = True,
        run_device_mining: bool = True,
        run_material_resolution: bool = True,
        run_metadata_enrichment: bool | None = None,
        run_public_resolver: bool | None = None,
        run_identity_judge: bool | None = None,
        run_material_auto_decision: bool | None = None,
        run_visual_prep: bool | None = None,
        run_material_ocsr: bool | None = None,
    ) -> BatchWorkerRunResult:
        self.init_runtime()
        run_metadata = (
            self.config.batch_worker.run_metadata_enrichment
            if run_metadata_enrichment is None
            else run_metadata_enrichment
        )
        run_public = (
            self.config.batch_worker.run_public_resolver
            if run_public_resolver is None
            else run_public_resolver
        )
        run_identity = (
            self.config.batch_worker.run_identity_judge
            if run_identity_judge is None
            else run_identity_judge
        )
        run_auto_decision = (
            self.config.batch_worker.run_material_auto_decision
            if run_material_auto_decision is None
            else run_material_auto_decision
        )
        run_visual = (
            self.config.batch_worker.run_visual_prep if run_visual_prep is None else run_visual_prep
        )
        run_ocsr = (
            self.config.batch_worker.run_material_ocsr
            if run_material_ocsr is None
            else run_material_ocsr
        )
        jobs = self.jobs.next_runnable(
            limit=max_jobs,
            include_failed_retries=include_failed_retries,
        )
        result = BatchWorkerRunResult()
        for job in jobs:
            result.processed_count += 1
            try:
                completed = self._run_job(
                    job,
                    template_id=template_id,
                    provider=provider,
                    model=model,
                    run_parse=run_parse,
                    run_device_mining=run_device_mining,
                    run_material_resolution=run_material_resolution,
                    run_metadata_enrichment=run_metadata,
                    run_public_resolver=run_public,
                    run_identity_judge=run_identity,
                    run_material_auto_decision=run_auto_decision,
                    run_visual_prep=run_visual,
                    run_material_ocsr=run_ocsr,
                )
                result.jobs.append(completed)
                if completed.status == "review_ready":
                    result.review_ready_count += 1
                elif completed.status == "failed":
                    result.failed_count += 1
                else:
                    result.skipped_count += 1
            except Exception:
                # _run_job stores the failure on the job; keep processing the batch.
                failed = self.jobs.get(job.job_id) or job
                result.jobs.append(failed)
                result.failed_count += 1
        return result

    def watch(
        self,
        *,
        interval_seconds: float | None = None,
        max_cycles: int | None = None,
        max_jobs_per_cycle: int = 1,
        **run_options: Any,
    ) -> list[BatchWorkerRunResult]:
        interval = (
            self.config.batch_worker.scan_interval_seconds
            if interval_seconds is None
            else interval_seconds
        )
        results: list[BatchWorkerRunResult] = []
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            self.scan_inbox_pdfs()
            results.append(self.run_once(max_jobs=max_jobs_per_cycle, **run_options))
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                break
            time.sleep(interval)
        return results

    def _run_job(
        self,
        job: BatchJob,
        *,
        template_id: str,
        provider: str | None,
        model: str | None,
        run_parse: bool,
        run_device_mining: bool,
        run_material_resolution: bool,
        run_metadata_enrichment: bool,
        run_public_resolver: bool,
        run_identity_judge: bool,
        run_material_auto_decision: bool,
        run_visual_prep: bool,
        run_material_ocsr: bool,
    ) -> BatchJob:
        job = self._mark_running(job)
        try:
            if run_metadata_enrichment:
                job = self._run_stage(
                    job,
                    "metadata_enrich",
                    lambda: self.metadata_service.enrich_paper(job.paper_id),
                    skip_if=lambda: self._paper_metadata_complete(job.paper_id),
                )
            if run_parse:
                job = self._run_stage(
                    job,
                    "parse_mineru",
                    lambda: self.mineru_service.parse_paper(job.paper_id),
                    skip_if=lambda: self._has_completed_mineru(job.paper_id),
                )
            if run_device_mining:
                job = self._run_stage(
                    job,
                    "device_mining_llm",
                    lambda: self.llm_service.mine_paper(
                        job.paper_id,
                        template_id=template_id,
                        provider=provider,
                        model=model,
                    ),
                    skip_if=lambda: self._has_completed_candidate_run(job.paper_id),
                )
            job = self._run_stage(
                job,
                "device_review_auto_confirm",
                lambda: self._confirm_unreviewable_device_data(job.paper_id),
                skip_if=lambda: not self._paper_has_no_reviewable_device_data(job.paper_id),
            )
            if self._paper_has_no_reviewable_device_data(job.paper_id):
                return self._mark_review_ready(job)
            if run_material_resolution:
                job = self._run_stage(
                    job,
                    "material_resolve_local",
                    lambda: self.material_resolution_service.resolve_paper_materials(job.paper_id),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            job = self._run_stage(
                job,
                "material_structure_triage",
                lambda: self.material_structure_triage_service.triage_paper(job.paper_id),
                skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
            )
            if run_public_resolver:
                job = self._run_stage(
                    job,
                    "material_resolve_public",
                    lambda: self.material_public_resolver_service.resolve_paper_public(
                        job.paper_id
                    ),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_identity_judge:
                job = self._run_stage(
                    job,
                    "material_identity_judge",
                    lambda: self.material_identity_judge_service.judge_paper_candidates(
                        job.paper_id,
                        provider=provider,
                        model=model,
                    ),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_material_auto_decision:
                job = self._run_stage(
                    job,
                    "material_auto_decision",
                    lambda: self.material_auto_decision_service.apply_paper_auto_decisions(
                        job.paper_id
                    ),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_visual_prep:
                job = self._run_stage(
                    job,
                    "material_visual_prep",
                    lambda: self.material_agent_service.run_foundation(job.paper_id),
                    skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
                )
            if run_material_ocsr:
                job = self._run_stage(
                    job,
                    "material_ocsr",
                    lambda: self.material_agent_service.run_ocsr_pipeline(
                        job.paper_id,
                        allow_unreviewed_matches=True,
                        min_model_confidence=(
                            self.config.batch_worker.material_ocsr_auto_match_min_confidence
                        ),
                    ),
                    skip_if=lambda: (
                        not self._has_completed_candidate_run(job.paper_id)
                        or not self._paper_needs_material_ocsr(job.paper_id)
                    ),
                )
            job = self._run_stage(
                job,
                "material_completion_check",
                lambda: self.material_completion_service.confirm_paper_if_materials_complete(
                    job.paper_id
                ),
                skip_if=lambda: not self._has_completed_candidate_run(job.paper_id),
            )
            return self._mark_review_ready(job)
        except Exception as exc:
            self._mark_failed(job, exc)
            raise

    def _run_stage(
        self,
        job: BatchJob,
        stage: str,
        action,
        *,
        skip_if,
    ) -> BatchJob:
        if skip_if():
            timings = dict(job.stage_timings)
            timings.setdefault(stage, 0.0)
            return self._update_job(
                job.model_copy(
                    update={
                        "current_stage": stage,
                        "last_completed_stage": stage,
                        "stage_timings": timings,
                        "updated_at": now_iso(),
                    }
                )
            )
        job = self._update_job(
            job.model_copy(update={"current_stage": stage, "updated_at": now_iso()})
        )
        started = perf_counter()
        action()
        elapsed = perf_counter() - started
        timings = dict(job.stage_timings)
        timings[stage] = round(elapsed, 3)
        errors = dict(job.stage_errors)
        errors.pop(stage, None)
        return self._update_job(
            job.model_copy(
                update={
                    "last_completed_stage": stage,
                    "stage_timings": timings,
                    "stage_errors": errors,
                    "error_message": None,
                    "updated_at": now_iso(),
                }
            )
        )

    def _mark_running(self, job: BatchJob) -> BatchJob:
        timestamp = now_iso()
        return self._update_job(
            job.model_copy(
                update={
                    "status": "running",
                    "current_stage": "starting",
                    "started_at": job.started_at or timestamp,
                    "completed_at": None,
                    "updated_at": timestamp,
                }
            )
        )

    def _mark_review_ready(self, job: BatchJob) -> BatchJob:
        return self._update_job(
            job.model_copy(
                update={
                    "status": "review_ready",
                    "current_stage": "review_ready",
                    "last_completed_stage": "review_ready",
                    "error_message": None,
                    "updated_at": now_iso(),
                    "completed_at": now_iso(),
                }
            )
        )

    def _mark_review_ready_many(self, jobs: list[BatchJob]) -> list[BatchJob]:
        timestamp = now_iso()
        updated = [
            job.model_copy(
                update={
                    "status": "review_ready",
                    "current_stage": "review_ready",
                    "last_completed_stage": "review_ready",
                    "error_message": None,
                    "updated_at": timestamp,
                    "completed_at": timestamp,
                }
            )
            for job in jobs
        ]
        return self.jobs.update_many(updated)

    def _mark_failed(self, job: BatchJob, exc: Exception) -> BatchJob:
        stage = job.current_stage or "unknown"
        stage_errors = dict(job.stage_errors)
        stage_errors[stage] = str(exc)
        failed = job.model_copy(
            update={
                "status": "failed",
                "error_message": str(exc),
                "stage_errors": stage_errors,
                "retry_count": job.retry_count + 1,
                "updated_at": now_iso(),
                "completed_at": now_iso(),
            }
        )
        return self._update_job(failed)

    def _update_job(self, job: BatchJob) -> BatchJob:
        return self.jobs.update(job)

    def _review_batch_size(self, batch_size: int | None) -> int:
        size = batch_size or self.config.batch_worker.review_batch_size
        if size <= 0:
            raise ValueError("Batch size must be positive.")
        return size

    def _final_record_counts(self) -> dict[str, int]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, COUNT(*) AS count
                FROM candidate_final_records
                WHERE status = 'confirmed'
                GROUP BY paper_id
                """
            ).fetchall()
        return {str(row["paper_id"]): int(row["count"]) for row in rows}

    @staticmethod
    def _paper_is_confirmed(paper: Paper, final_record_count: int) -> bool:
        return paper.review_status == "confirmed" or final_record_count > 0

    @staticmethod
    def _paper_is_excluded(paper: Paper) -> bool:
        return paper.review_status == "excluded"

    def _batch_summary(
        self,
        jobs: list[BatchJob],
        *,
        batch_index: int,
        batch_size: int,
        offset: int,
        final_record_counts: dict[str, int],
        papers_by_id: dict[str, Paper],
    ) -> BatchReviewSummary:
        counts = Counter(job.status for job in jobs)
        confirmed_count = 0
        excluded_count = 0
        for job in jobs:
            paper = papers_by_id.get(job.paper_id) or self.papers.get(job.paper_id)
            if paper and self._paper_is_confirmed(paper, final_record_counts.get(job.paper_id, 0)):
                confirmed_count += 1
            elif paper and self._paper_is_excluded(paper):
                excluded_count += 1
        total = len(jobs)
        all_confirmed = total > 0 and confirmed_count == total
        all_resolved = total > 0 and confirmed_count + excluded_count == total
        if all_confirmed:
            status = "confirmed"
        elif all_resolved:
            status = "completed"
        elif counts.get("running", 0) > 0:
            status = "running"
        elif counts.get("failed", 0) > 0:
            status = "needs_attention"
        elif counts.get("review_ready", 0) > 0 or confirmed_count > 0 or excluded_count > 0:
            status = "in_review"
        elif counts.get("registered", 0) == total:
            status = "pending"
        else:
            status = "mixed"
        return BatchReviewSummary(
            batch_id=f"batch-{batch_index + 1:04d}",
            batch_index=batch_index,
            batch_number=batch_index + 1,
            batch_size=batch_size,
            offset=offset,
            total_count=total,
            confirmed_count=confirmed_count,
            excluded_count=excluded_count,
            review_ready_count=counts.get("review_ready", 0),
            running_count=counts.get("running", 0),
            failed_count=counts.get("failed", 0),
            registered_count=counts.get("registered", 0),
            status=status,
            first_doi=jobs[0].doi if jobs else None,
            last_doi=jobs[-1].doi if jobs else None,
            all_confirmed=all_confirmed,
            all_resolved=all_resolved,
        )

    def _has_completed_mineru(self, paper_id: str) -> bool:
        return self.mineru_runs.latest_completed_by_paper(paper_id) is not None

    def _has_completed_candidate_run(self, paper_id: str) -> bool:
        runs = self.candidate_runs.list_runs_by_paper(paper_id)
        return any(run.status in {"completed", "no_device"} for run in runs)

    def _paper_has_no_reviewable_device_data(self, paper_id: str) -> bool:
        return self._unreviewable_device_reason(paper_id) is not None

    def _unreviewable_device_reason(self, paper_id: str) -> str | None:
        runs = self.candidate_runs.list_runs_by_paper(paper_id)
        if not runs:
            return None
        run = runs[0]
        reason = no_device_review_reason(run.mining_result)
        if reason:
            return reason
        if run.status == "failed" and not self.candidate_runs.list_values_by_run(
            run.candidate_run_id
        ):
            return "device_data_validation_failed"
        return None

    def _confirm_unreviewable_device_data(self, paper_id: str) -> bool:
        reason = self._unreviewable_device_reason(paper_id)
        if reason is None:
            return False
        self.papers.set_review_status(
            paper_id,
            "confirmed",
            reason=reason,
        )
        return True

    def _paper_needs_material_ocsr(self, paper_id: str) -> bool:
        result = self.material_structure_triage_service.triage_paper(paper_id)
        return bool(result and result.should_run_ocsr)

    def _run_stage3_paper_flow(
        self,
        *,
        paper_id: str,
        public_items: list[MaterialStage3PlanItem],
        provider: str | None,
        model: str | None,
        run_public: bool,
        run_identity: bool,
        run_auto_decision: bool,
        run_visual: bool,
        run_ocsr: bool,
        public_semaphore: BoundedSemaphore,
        judge_semaphore: BoundedSemaphore,
        ocsr_semaphore: BoundedSemaphore,
        prefetch_executor: ThreadPoolExecutor,
        public_timer: _ConcurrentPhaseTimer,
        judge_timer: _ConcurrentPhaseTimer,
        auto_timer: _ConcurrentPhaseTimer,
        visual_timer: _ConcurrentPhaseTimer,
        prefetch_timer: _ConcurrentPhaseTimer,
        vlm_concurrency: int,
        decimer_segmentation_concurrency: int,
        decimer_ocsr_concurrency: int,
    ) -> _Stage3PaperFlowResult:
        flow = _Stage3PaperFlowResult(paper_id=paper_id)
        prefetch_lock = Lock()
        prefetch_future: Future[Any] | None = None

        def start_visual_prefetch() -> Future[Any] | None:
            nonlocal prefetch_future
            if not run_ocsr:
                return None
            with prefetch_lock:
                if prefetch_future is None:
                    prefetch_future = prefetch_executor.submit(
                        lambda: visual_timer.run(
                            lambda: prefetch_timer.run(
                                lambda: self._prepare_material_visual_assets(paper_id)
                            )
                        )
                    )
                return prefetch_future

        initial_bundle = (
            self.material_resolution_service.get_material_structure_bundle(paper_id)
            if not run_public
            else None
        )

        def process_item(item: MaterialStage3PlanItem) -> dict[str, Any]:
            item_result: dict[str, Any] = {
                "public_attempted": 0,
                "public_error": None,
                "judge_attempted": 0,
                "judge_error": None,
            }
            public_bundle = initial_bundle
            if run_public:
                item_result["public_attempted"] = 1
                try:
                    with public_semaphore:
                        public_bundle = public_timer.run(
                            lambda: self.material_stage3_planner_service.run_timed_task_stage(
                                item,
                                stage="public_resolution",
                                success_next_action="judge_candidates",
                                runner=lambda: (
                                    self.material_public_resolver_service.resolve_material_public(
                                        item.paper_id,
                                        item.paper_material_id,
                                    )
                                ),
                            )
                        )
                except Exception as exc:
                    item_result["public_error"] = str(exc)
            has_pending = self._bundle_has_pending_structure_candidates(
                item,
                public_bundle,
            )
            if not has_pending:
                if self._bundle_material_requires_visual(item, public_bundle):
                    start_visual_prefetch()
                return item_result
            if not run_identity:
                return item_result
            item_result["judge_attempted"] = 1
            try:
                with judge_semaphore:
                    judge_timer.run(
                        lambda: self.material_stage3_planner_service.run_timed_task_stage(
                            item,
                            stage="identity_judge",
                            success_next_action="auto_decision",
                            runner=lambda: (
                                self.material_identity_judge_service.judge_material_candidates(
                                    item.paper_id,
                                    item.paper_material_id,
                                    provider=provider,
                                    model=model,
                                )
                            ),
                        )
                    )
            except Exception as exc:
                item_result["judge_error"] = str(exc)
            return item_result

        if public_items:
            item_workers = max(
                1,
                min(
                    len(public_items),
                    self.config.batch_worker.material_public_concurrency,
                ),
            )
            with ThreadPoolExecutor(max_workers=item_workers) as item_executor:
                item_results = list(item_executor.map(process_item, public_items))
            for item, item_result in zip(public_items, item_results, strict=True):
                flow.public_attempted_count += int(item_result["public_attempted"])
                flow.judge_material_count += int(item_result["judge_attempted"])
                if item_result["public_error"]:
                    flow.public_errors.append(
                        {
                            "paper_id": item.paper_id,
                            "paper_material_id": item.paper_material_id,
                            "error": str(item_result["public_error"]),
                        }
                    )
                if item_result["judge_error"]:
                    flow.judge_errors.append(
                        {
                            "paper_id": item.paper_id,
                            "paper_material_id": item.paper_material_id,
                            "error": str(item_result["judge_error"]),
                        }
                    )

        if run_auto_decision:
            try:
                auto_result = auto_timer.run(
                    lambda: self.material_auto_decision_service.apply_paper_auto_decisions(paper_id)
                )
                if auto_result is not None:
                    flow.auto_accepted_count = auto_result.accepted_count
                    flow.auto_rejected_count = auto_result.rejected_count
            except Exception as exc:
                flow.auto_errors.append({"paper_id": paper_id, "error": str(exc)})

        (
            flow.residual_items,
            flow.public_human_items,
            flow.unavailable_items,
        ) = self._route_stage3_residuals(public_items)

        if flow.residual_items and run_ocsr:
            visual_future = start_visual_prefetch()
            try:
                if visual_future is not None:
                    visual_future.result()
                with ocsr_semaphore:
                    visual_timer.run(
                        lambda: self.material_agent_service.run_ocsr_pipeline(
                            paper_id,
                            allow_unreviewed_matches=True,
                            min_model_confidence=(
                                self.config.batch_worker.material_ocsr_auto_match_min_confidence
                            ),
                            vlm_concurrency=vlm_concurrency,
                            decimer_segmentation_concurrency=(decimer_segmentation_concurrency),
                            decimer_ocsr_concurrency=decimer_ocsr_concurrency,
                            target_paper_material_ids={
                                item.paper_material_id for item in flow.residual_items
                            },
                        )
                    )
            except Exception as exc:
                flow.ocsr_errors.append({"paper_id": paper_id, "error": str(exc)})
        elif flow.residual_items and run_visual:
            try:
                visual_timer.run(lambda: self.material_agent_service.run_foundation(paper_id))
            except Exception as exc:
                flow.ocsr_errors.append({"paper_id": paper_id, "error": str(exc)})

        flow.ocsr_human_review_count = self._finalize_stage3_ocsr_routes(
            flow.residual_items,
            ocsr_was_run=run_ocsr,
        )
        return flow

    def _prepare_material_visual_assets(self, paper_id: str):
        prepare = getattr(self.material_agent_service, "prepare_visual_assets", None)
        if callable(prepare):
            return prepare(paper_id)
        return self.material_agent_service.run_foundation(paper_id)

    def _bundle_has_pending_structure_candidates(
        self,
        item: MaterialStage3PlanItem,
        bundle: object,
    ) -> bool:
        if not isinstance(bundle, PaperMaterialStructureBundle):
            return self._material_has_pending_structure_candidates(item)
        return any(
            candidate.paper_material_id == item.paper_material_id
            and candidate.provider in {"pubchem", "opsin"}
            and candidate.status not in {"accepted", "rejected"}
            and bool(candidate.canonical_smiles or candidate.inchi_key)
            and is_plausible_public_structure_candidate(candidate)
            for candidate in bundle.structure_candidates
        )

    @staticmethod
    def _bundle_material_requires_visual(
        item: MaterialStage3PlanItem,
        bundle: object,
    ) -> bool:
        if not isinstance(bundle, PaperMaterialStructureBundle):
            return True
        material = next(
            (
                material
                for material in bundle.materials
                if material.paper_material_id == item.paper_material_id
            ),
            None,
        )
        return material is None or material.material_class != "proprietary"

    def _material_has_pending_structure_candidates(
        self,
        item: MaterialStage3PlanItem,
    ) -> bool:
        bundle = self.material_resolution_service.get_material_structure_bundle(item.paper_id)
        if bundle is None:
            return False
        return any(
            candidate.paper_material_id == item.paper_material_id
            and candidate.provider in {"pubchem", "opsin"}
            and candidate.status not in {"accepted", "rejected"}
            and bool(candidate.canonical_smiles or candidate.inchi_key)
            and is_plausible_public_structure_candidate(candidate)
            for candidate in bundle.structure_candidates
        )

    def _route_stage3_residuals(
        self,
        public_items: list[MaterialStage3PlanItem],
    ) -> tuple[
        list[MaterialStage3PlanItem],
        list[MaterialStage3PlanItem],
        list[MaterialStage3PlanItem],
    ]:
        residual: list[MaterialStage3PlanItem] = []
        human_review: list[MaterialStage3PlanItem] = []
        unavailable: list[MaterialStage3PlanItem] = []
        bundle_by_paper = {
            paper_id: self.material_resolution_service.get_material_structure_bundle(paper_id)
            for paper_id in sorted({item.paper_id for item in public_items})
        }
        triage_by_paper = {}
        for paper_id, bundle in bundle_by_paper.items():
            triage_bundle = getattr(
                self.material_structure_triage_service,
                "triage_bundle",
                None,
            )
            triage_by_paper[paper_id] = (
                triage_bundle(bundle)
                if callable(triage_bundle) and isinstance(bundle, PaperMaterialStructureBundle)
                else self.material_structure_triage_service.triage_paper(paper_id)
            )
        for item in public_items:
            triage = triage_by_paper.get(item.paper_id)
            triage_item = next(
                (
                    value
                    for value in (triage.items if triage else [])
                    if value.paper_material_id == item.paper_material_id
                ),
                None,
            )
            if triage_item is not None and not triage_item.should_run_ocsr:
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="completed",
                    next_action="none",
                    status="completed",
                )
                continue
            bundle = bundle_by_paper.get(item.paper_id)
            if bundle is None:
                residual.append(item)
                continue
            pending_public = [
                candidate
                for candidate in bundle.structure_candidates
                if candidate.paper_material_id == item.paper_material_id
                and candidate.provider in {"pubchem", "opsin"}
                and candidate.status not in {"accepted", "rejected"}
                and is_plausible_public_structure_candidate(candidate)
            ]
            blocking_public = [
                candidate
                for candidate in pending_public
                if not _public_candidate_should_yield_to_visual_fallback(
                    candidate,
                    bundle.identity_judgments,
                )
            ]
            if blocking_public:
                human_review.append(item)
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="human_review",
                    next_action="review_public_candidate",
                    status="needs_review",
                    assigned_strategy="public_candidate_human_review",
                )
                continue
            material = next(
                (
                    value
                    for value in bundle.materials
                    if value.paper_material_id == item.paper_material_id
                ),
                None,
            )
            if material and material.material_class == "proprietary":
                unavailable.append(item)
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="completed",
                    next_action="none",
                    status="completed",
                    assigned_strategy="structure_unavailable_proprietary",
                )
                continue
            residual.append(item)
            self.material_stage3_planner_service.update_task_stage(
                item,
                stage="visual_ocsr_pending",
                next_action="run_visual_ocsr",
                status="pending",
                assigned_strategy=(
                    "visual_ocsr_with_identity_uncertainty" if pending_public else "visual_ocsr"
                ),
            )
        return residual, human_review, unavailable

    def _finalize_stage3_ocsr_routes(
        self,
        items: list[MaterialStage3PlanItem],
        *,
        ocsr_was_run: bool,
    ) -> int:
        human_review_count = 0
        bundles = {
            paper_id: self.material_resolution_service.get_material_structure_bundle(paper_id)
            for paper_id in sorted({item.paper_id for item in items})
        }
        for item in items:
            if not ocsr_was_run:
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="visual_ocsr_pending",
                    next_action="run_visual_ocsr",
                    status="pending",
                )
                continue
            bundle = bundles.get(item.paper_id)
            accepted_candidates = [
                candidate
                for candidate in (bundle.structure_candidates if bundle else [])
                if candidate.paper_material_id == item.paper_material_id
                and candidate.status == "accepted"
                and (candidate.canonical_smiles or candidate.inchi_key)
            ]
            if accepted_candidates:
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="completed",
                    next_action="none",
                    status="completed",
                    assigned_strategy="accepted_ocsr_structure",
                )
                continue
            candidates = [
                candidate
                for candidate in (bundle.structure_candidates if bundle else [])
                if candidate.paper_material_id == item.paper_material_id
                and candidate.provider == "decimer_ocsr"
                and candidate.status not in {"accepted", "rejected"}
            ]
            human_review_count += 1
            if candidates:
                needs_correction = any(
                    candidate.status == "needs_correction" for candidate in candidates
                )
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="human_review",
                    next_action=(
                        "edit_ocsr_structure" if needs_correction else "review_ocsr_candidate"
                    ),
                    status="needs_review",
                    assigned_strategy=(
                        "ocsr_candidate_needs_correction"
                        if needs_correction
                        else "ocsr_candidate_review"
                    ),
                )
            else:
                self.material_stage3_planner_service.update_task_stage(
                    item,
                    stage="human_review",
                    next_action="manual_structure_input",
                    status="needs_review",
                    assigned_strategy="manual_structure_required",
                )
        return human_review_count

    def _sync_material_task_timings(self, jobs: list[BatchJob]) -> None:
        for original in jobs:
            runs = self.candidate_runs.list_runs_by_paper(original.paper_id)
            run = next((item for item in runs if item.status == "completed"), None)
            if run is None:
                continue
            task_timings: Counter[str] = Counter()
            task_errors: dict[str, str] = {}
            for task in self.material_stage3_planner_service.tasks.list_by_run(
                run.candidate_run_id
            ):
                for stage, seconds in task.stage_timings.items():
                    task_timings[f"material_task_{stage}"] += float(seconds)
                for stage, error in task.stage_errors.items():
                    task_errors[f"{task.paper_material_id}:{stage}"] = error
            current = self.jobs.get(original.job_id) or original
            self.jobs.upsert(
                current.model_copy(
                    update={
                        "stage_timings": {
                            **current.stage_timings,
                            **{stage: round(seconds, 3) for stage, seconds in task_timings.items()},
                        },
                        "stage_errors": {**current.stage_errors, **task_errors},
                        "updated_at": now_iso(),
                    }
                )
            )

    def _paper_metadata_complete(self, paper_id: str) -> bool:
        paper = self.papers.get(paper_id)
        return bool(paper and paper.title and paper.journal and paper.publisher and paper.year)

    def _enrich_missing_batch_metadata(
        self,
        papers: list[Paper],
        *,
        enabled: bool,
        max_workers: int,
    ) -> tuple[dict[str, int], float]:
        targets = [paper.paper_id for paper in papers if not _paper_metadata_complete(paper)]
        started = perf_counter()
        results: list[tuple[Any, Any | None, Exception | None, float]] = []
        if enabled and targets:
            results = _run_concurrent_items(
                targets,
                max_workers=max_workers,
                runner=self.metadata_service.enrich_paper,
            )

        updated = 0
        not_found = 0
        failed = 0
        for _, enrichment, error, _ in results:
            if error is not None or enrichment is None:
                failed += 1
                continue
            status = getattr(enrichment, "status", None)
            if status == "updated":
                updated += 1
            elif status == "not_found":
                not_found += 1
            elif status == "failed":
                failed += 1

        return (
            {
                "metadata_target_count": len(targets),
                "metadata_attempted_count": len(targets) if enabled else 0,
                "metadata_updated_count": updated,
                "metadata_not_found_count": not_found,
                "metadata_failed_count": failed,
            },
            round(perf_counter() - started, 3),
        )


def _run_concurrent_items(
    items: list[Any],
    *,
    max_workers: int,
    runner,
) -> list[tuple[Any, Any | None, Exception | None, float]]:
    if not items:
        return []
    worker_count = max(1, min(max_workers, len(items)))

    def run_one(item: Any) -> tuple[Any, Any | None, Exception | None, float]:
        started = perf_counter()
        try:
            return item, runner(item), None, perf_counter() - started
        except Exception as exc:
            return item, None, exc, perf_counter() - started

    if worker_count == 1:
        return [run_one(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(run_one, items))


def _concurrent_errors(
    results: list[tuple[Any, Any | None, Exception | None, float]],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for item, _, error, _ in results:
        if error is None:
            continue
        if isinstance(item, MaterialStage3PlanItem):
            item_id = f"{item.paper_id}:{item.paper_material_id}"
        elif isinstance(item, tuple) and item:
            item_id = str(item[0])
        else:
            item_id = str(getattr(item, "paper_id", item))
        errors.append({"item": item_id, "error": str(error)})
    return errors


def _paper_metadata_complete(paper: Paper) -> bool:
    return bool(paper.title and paper.journal and paper.publisher and paper.year)


def _render_batch_report_markdown(report: BatchRunReport) -> str:
    summary = report.summary
    lines = [
        f"# Batch Report: {summary.batch_id}",
        "",
        f"Generated: {report.generated_at}",
        f"Batch number: {summary.batch_number}",
        f"Batch size: {summary.batch_size}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| total papers | {summary.total_count} |",
        f"| processed in this run | {summary.processed_count} |",
        f"| review ready | {summary.review_ready_count} |",
        f"| confirmed | {summary.confirmed_count} |",
        f"| excluded reviews | {summary.excluded_count} |",
        f"| failed | {summary.failed_count} |",
        f"| skipped | {summary.skipped_count} |",
        f"| total stage seconds | {summary.total_stage_seconds:.3f} |",
        f"| average stage seconds | {summary.average_stage_seconds:.3f} |",
        f"| slowest paper seconds | {summary.slowest_stage_seconds:.3f} |",
        "",
    ]
    if report.run_options:
        lines.extend(["## Run Options", "", "| option | value |", "| --- | --- |"])
        for key, value in sorted(report.run_options.items()):
            lines.append(f"| {_md_cell(key)} | {_md_cell(value)} |")
        lines.append("")
    if report.phase_timings:
        lines.extend(["## Phase Wall Time", "", "| phase | seconds |", "| --- | ---: |"])
        for phase, seconds in report.phase_timings.items():
            lines.append(f"| {_md_cell(phase)} | {float(seconds):.3f} |")
        if report.material_metrics.get("phase_timings_overlap"):
            lines.extend(
                [
                    "",
                    "Stage 3 uses a paper-level DAG. These phase windows overlap and must not "
                    "be added together; `stage3_material_refresh` is the end-to-end wall time.",
                ]
            )
        lines.append("")

    if report.material_metrics:
        lines.extend(["## Material Stage 3 Funnel", "", "| metric | value |", "| --- | ---: |"])
        for key, value in report.material_metrics.items():
            if key == "stage3_phase_task_seconds":
                continue
            lines.append(f"| {_md_cell(key)} | {_md_cell(value)} |")
        lines.append("")
        task_seconds = report.material_metrics.get("stage3_phase_task_seconds")
        if isinstance(task_seconds, dict):
            lines.extend(
                [
                    "## Stage 3 Cumulative Task Time",
                    "",
                    "| phase | task seconds |",
                    "| --- | ---: |",
                ]
            )
            for phase, seconds in task_seconds.items():
                lines.append(f"| {_md_cell(phase)} | {float(seconds):.3f} |")
            lines.append("")

    lines.extend(["## Status Counts", "", "| status | count |", "| --- | ---: |"])
    for status, count in sorted(summary.status_counts.items()):
        lines.append(f"| {_md_cell(status)} | {count} |")
    lines.append("")

    stage_totals: Counter[str] = Counter()
    for job in report.jobs:
        for stage, seconds in job.stage_timings.items():
            stage_totals[stage] += float(seconds)
    lines.extend(["## Stage Timing Totals", "", "| stage | seconds |", "| --- | ---: |"])
    if stage_totals:
        for stage, seconds in stage_totals.most_common():
            lines.append(f"| {_md_cell(stage)} | {seconds:.3f} |")
    else:
        lines.append("| none | 0.000 |")
    lines.append("")

    lines.extend(
        [
            "## Jobs",
            "",
            "| # | DOI | journal | year | status | stage | seconds | error |",
            "| ---: | --- | --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    for job in report.jobs:
        error = job.error_message or ""
        lines.append(
            "| "
            f"{job.position} | {_md_cell(job.doi)} | {_md_cell(job.journal)} | "
            f"{job.year or ''} | {_md_cell(job.status)} | "
            f"{_md_cell(job.current_stage)} | {job.total_seconds:.3f} | "
            f"{_md_cell(error[:160])} |"
        )
    lines.append("")

    failed_jobs = [job for job in report.jobs if job.status == "failed" or job.stage_errors]
    if failed_jobs:
        lines.extend(["## Failed / Attention Needed", ""])
        for job in failed_jobs:
            lines.append(
                f"- `{job.paper_id}` `{job.current_stage}`: {job.error_message or 'stage error'}"
            )
            for stage, message in job.stage_errors.items():
                lines.append(f"  - `{stage}`: {message}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _md_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text


def _unique_csv_values(csv_path: Path, *, doi_column: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if doi_column not in (reader.fieldnames or []):
            raise ValueError(f"CSV column not found: {doi_column}")
        for row in reader:
            raw = (row.get(doi_column) or "").strip()
            if not raw:
                continue
            stem = _pdf_stem(raw)
            if stem in seen:
                continue
            seen.add(stem)
            values.append(stem)
    return values


def _pdf_stem(value: str) -> str:
    stripped = value.strip()
    return stripped[:-4] if stripped.lower().endswith(".pdf") else stripped


def _doi_from_pdf_name(name: str) -> str:
    stem = Path(name).stem
    return unquote(stem).strip().lower()


def _is_stable_file(path: Path, stable_seconds: float) -> bool:
    if stable_seconds <= 0:
        return True
    return (time.time() - path.stat().st_mtime) >= stable_seconds
