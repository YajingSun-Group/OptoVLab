from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pypdf import PdfReader

from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.domain_template_service import DomainTemplateService
from evolab_local.mining_platform.external.openai_compatible_client import (
    OpenAICompatibleLLMClient,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_auto_decision_service import (
    MaterialAutoDecisionService,
)
from evolab_local.mining_platform.material_identity_judge_service import (
    MaterialIdentityJudgeService,
)
from evolab_local.mining_platform.material_public_resolver_service import (
    MaterialPublicResolverService,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
)
from evolab_local.mining_platform.material_structure_triage_service import (
    MaterialStructureTriageService,
)
from evolab_local.mining_platform.mining.llm_mining_service import LLMMiningService
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService
from evolab_local.mining_platform.schemas.data_mining_agent import (
    AgentConversationTurn,
    AgentJob,
    AgentMessage,
    AgentResult,
    AgentSession,
    AgentSessionCreate,
    AgentTemplateSummary,
    AgentUploadedPaper,
    AgentWorkspace,
)
from evolab_local.mining_platform.storage.data_mining_agent_repository import (
    DataMiningAgentRepository,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import DocumentBlockRepository


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
CUSTOM_PLAN_PROMPT_VERSION = "data_mining_agent_plan_v1"
GENERIC_MINING_PROMPT_VERSION = "data_mining_agent_generic_extract_v1"


class DataMiningAgentService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        paper_service: PaperService | None = None,
        candidate_ingestion_service: CandidateIngestionService | None = None,
        material_resolution_service: MaterialResolutionService | None = None,
        material_public_resolver_service: MaterialPublicResolverService | None = None,
        material_identity_judge_service: MaterialIdentityJudgeService | None = None,
        material_auto_decision_service: MaterialAutoDecisionService | None = None,
        material_structure_agent_service: MaterialStructureAgentService | None = None,
        max_workers: int = 2,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.repository = DataMiningAgentRepository(self.database)
        self.paper_service = paper_service or PaperService(config)
        self.candidate_ingestion = candidate_ingestion_service or CandidateIngestionService(config)
        self.material_resolution = material_resolution_service or MaterialResolutionService(config)
        self.material_public = (
            material_public_resolver_service or MaterialPublicResolverService(config)
        )
        self.material_judge = material_identity_judge_service or MaterialIdentityJudgeService(config)
        self.material_auto = material_auto_decision_service or MaterialAutoDecisionService(config)
        self.material_agent = material_structure_agent_service or MaterialStructureAgentService(config)
        self.material_triage = MaterialStructureTriageService(config)
        self.mineru = MinerUParseService(config)
        self.llm_mining = LLMMiningService(config)
        self.templates = DomainTemplateService(config)
        self.blocks = DocumentBlockRepository(self.database)
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="data-mining-agent",
        )
        self._futures: dict[str, Future[None]] = {}
        self._future_lock = threading.Lock()

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()
        self.repository.init_schema()

    def recover_interrupted_jobs(self) -> int:
        self.init_runtime()
        return self.repository.interrupt_stale_jobs()

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def list_templates(self) -> list[AgentTemplateSummary]:
        return [
            AgentTemplateSummary(
                template_id="oled_device_v1",
                name="OLED Device and Material Mining",
                version="1.0",
                domain="oled",
                description=(
                    "Extract OLED device stacks, layer compositions, fabrication conditions, "
                    "performance metrics, evidence anchors, and resolvable molecular structures."
                ),
                capabilities=[
                    "MinerU document parsing",
                    "DeepSeek device extraction",
                    "field-level evidence anchors",
                    "local and public material resolution",
                    "LLM identity cross-check",
                    "VLM + DECIMER molecular structure recognition",
                    "human-editable review results",
                ],
                plan=self._oled_plan(),
            )
        ]

    def create_session(self, payload: AgentSessionCreate) -> AgentSession:
        self.init_runtime()
        timestamp = now_iso()
        session_id = uuid4().hex
        if payload.mode == "preset":
            if payload.template_id != "oled_device_v1":
                raise ValueError(f"Unknown preset template: {payload.template_id}")
            plan = self._oled_plan()
            session = AgentSession(
                session_id=session_id,
                title=payload.title or "OLED mining session",
                mode="preset",
                status="awaiting_plan_approval",
                domain="oled",
                template_id="oled_device_v1",
                plan_status="awaiting_user_approval",
                critic_status="not_configured",
                requirements={"initial_message": payload.initial_message or ""},
                plan=plan,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self.repository.create_session(session)
            if payload.initial_message:
                self.repository.add_message(
                    session_id=session_id,
                    role="user",
                    content=payload.initial_message,
                )
            self.repository.add_message(
                session_id=session_id,
                role="assistant",
                content=(
                    "I loaded the validated OLED Device and Material Mining preset. "
                    "Review the proposed fields, evidence policy, and tool sequence, "
                    "then approve the plan before uploading a PDF."
                ),
                message_type="plan",
                metadata={"plan_status": "awaiting_user_approval"},
            )
            return self._require_session(session_id)

        session = AgentSession(
            session_id=session_id,
            title=payload.title or "Custom mining session",
            mode="custom",
            status="collecting_requirements",
            domain="custom",
            template_id=None,
            plan_status="collecting_requirements",
            critic_status="not_configured",
            requirements={"initial_message": payload.initial_message or ""},
            plan={},
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.repository.create_session(session)
        if payload.initial_message:
            self.repository.add_message(
                session_id=session_id,
                role="user",
                content=payload.initial_message,
            )
        self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content=(
                "Before I draft the mining plan, describe the scientific domain, the entities "
                "and fields you need, expected units or value constraints, and what counts as "
                "acceptable source evidence. Include any required databases or validation rules."
            ),
            metadata={
                "questions": [
                    "What scientific domain and document type are in scope?",
                    "Which entities and fields must be extracted?",
                    "What units, ranges, or controlled vocabularies apply?",
                    "What evidence and cross-validation rules are required?",
                ]
            },
        )
        return self._require_session(session_id)

    def list_sessions(self, limit: int = 50) -> list[AgentSession]:
        self.init_runtime()
        return self.repository.list_sessions(limit=limit)

    def get_session(self, session_id: str) -> AgentSession | None:
        self.init_runtime()
        return self.repository.get_session(session_id)

    def add_message(
        self,
        session_id: str,
        *,
        content: str,
        auto_respond: bool = True,
    ) -> AgentConversationTurn:
        session = self._require_session(session_id)
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Message content must not be empty.")
        user_message = self.repository.add_message(
            session_id=session_id,
            role="user",
            content=clean_content,
        )
        assistant_message = None
        if auto_respond and session.mode == "custom" and session.plan_status != "approved":
            assistant_message = self._draft_custom_plan_turn(session_id)
        return AgentConversationTurn(
            session=self._require_session(session_id),
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def generate_plan(self, session_id: str) -> AgentSession:
        session = self._require_session(session_id)
        if session.mode == "preset":
            return session
        self._draft_custom_plan_turn(session_id, force_plan=True)
        return self._require_session(session_id)

    def update_plan(
        self,
        session_id: str,
        *,
        plan: dict[str, Any],
        actor: str,
        message: str | None,
    ) -> AgentSession:
        session = self._require_session(session_id)
        normalized = self._normalize_custom_plan(plan, fallback_domain=session.domain)
        updated = self.repository.update_session(
            session_id,
            plan=normalized,
            domain=str(normalized.get("domain") or session.domain),
            status="awaiting_plan_approval",
            plan_status="awaiting_user_approval",
        )
        self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content=message or f"The mining plan was revised by {actor}. Review and approve it.",
            message_type="plan",
            metadata={"action": "plan_updated", "actor": actor},
        )
        return updated or self._require_session(session_id)

    def approve_plan(
        self,
        session_id: str,
        *,
        actor: str,
        message: str | None,
    ) -> AgentSession:
        session = self._require_session(session_id)
        if not session.plan:
            raise ValueError("A mining plan must be generated before approval.")
        next_status = "ready_to_run" if session.paper_id else "awaiting_pdf"
        updated = self.repository.update_session(
            session_id,
            status=next_status,
            plan_status="approved",
        )
        self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content=message or "Plan approved. Upload one PDF to start the mining workflow.",
            message_type="status",
            metadata={"action": "plan_approved", "actor": actor},
        )
        return updated or self._require_session(session_id)

    def upload_pdf(
        self,
        session_id: str,
        *,
        filename: str,
        content: bytes,
    ) -> AgentUploadedPaper:
        session = self._require_session(session_id)
        if session.plan_status != "approved":
            raise ValueError("Approve the mining plan before uploading a PDF.")
        if not content:
            raise ValueError("The uploaded PDF is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("The uploaded PDF exceeds the 200 MB limit.")
        if b"%PDF" not in content[:1024]:
            raise ValueError("The uploaded file does not have a valid PDF header.")

        digest = hashlib.sha256(content).hexdigest()
        upload_dir = self.config.paths.runtime_dir / "data_mining_agent" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_path = upload_dir / f"{digest}.pdf"
        if not output_path.exists():
            temporary_path = output_path.with_suffix(".pdf.part")
            temporary_path.write_bytes(content)
            temporary_path.replace(output_path)
        try:
            reader = PdfReader(output_path)
            page_count = len(reader.pages)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise ValueError(f"The uploaded PDF cannot be parsed: {exc}") from exc
        if page_count < 1:
            output_path.unlink(missing_ok=True)
            raise ValueError("The uploaded PDF has no pages.")

        paper_id = f"upload-{digest[:24]}"
        safe_name = _safe_filename(filename)
        self.paper_service.register_uploaded_pdf(
            paper_id=paper_id,
            doi=f"upload:{digest}",
            pdf_path=output_path,
            pdf_sha256=digest,
            pdf_size_bytes=len(content),
            title=Path(safe_name).stem or "Uploaded paper",
            domain=session.domain,
        )
        updated = self.repository.update_session(
            session_id,
            paper_id=paper_id,
            status="ready_to_run",
        )
        self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content=(
                f"PDF received and validated: {safe_name} ({page_count} pages). "
                "The approved workflow is ready to run."
            ),
            message_type="file",
            metadata={
                "paper_id": paper_id,
                "filename": safe_name,
                "sha256": digest,
                "page_count": page_count,
                "size_bytes": len(content),
            },
        )
        return AgentUploadedPaper(
            session=updated or self._require_session(session_id),
            paper_id=paper_id,
            filename=safe_name,
            size_bytes=len(content),
            sha256=digest,
            page_count=page_count,
        )

    def start_job(self, session_id: str, *, force: bool = False) -> AgentJob:
        session = self._require_session(session_id)
        if session.plan_status != "approved":
            raise ValueError("Approve the mining plan before running it.")
        if not session.paper_id:
            raise ValueError("Upload a PDF before running the mining plan.")
        running = [
            job
            for job in self.repository.list_jobs(session_id)
            if job.status in {"queued", "running"}
        ]
        if running:
            return running[0]
        timestamp = now_iso()
        job = self.repository.create_job(
            AgentJob(
                job_id=uuid4().hex,
                session_id=session_id,
                paper_id=session.paper_id,
                status="queued",
                current_step="queued",
                progress=0.0,
                result_summary={"force": force},
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        self.repository.add_event(
            job_id=job.job_id,
            session_id=session_id,
            event_type="workflow",
            stage="queued",
            status="queued",
            title="Mining workflow queued",
            detail="The approved plan is waiting for a background worker.",
        )
        self.repository.update_session(session_id, status="running")
        self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content="The mining workflow has started. Tool calls and evidence checks appear below.",
            message_type="status",
            metadata={"job_id": job.job_id},
        )
        future = self.executor.submit(self._run_job, job.job_id, force)
        with self._future_lock:
            self._futures[job.job_id] = future
        future.add_done_callback(lambda _: self._drop_future(job.job_id))
        return job

    def get_job(self, job_id: str) -> AgentJob | None:
        self.init_runtime()
        return self.repository.get_job(job_id)

    def list_job_events(self, job_id: str) -> list[Any]:
        self.init_runtime()
        return self.repository.list_events(job_id=job_id)

    def get_workspace(self, session_id: str) -> AgentWorkspace:
        session = self._require_session(session_id)
        jobs = self.repository.list_jobs(session_id)
        paper = self.paper_service.get_paper(session.paper_id) if session.paper_id else None
        candidate_bundle = None
        material_bundle = None
        if session.paper_id and session.template_id == "oled_device_v1":
            bundle = self.candidate_ingestion.get_review_bundle(session.paper_id)
            if bundle and bundle.run:
                candidate_bundle = bundle.model_dump(mode="json")
            materials = self.material_resolution.get_material_structure_bundle(session.paper_id)
            if materials:
                material_bundle = materials.model_dump(mode="json")
        return AgentWorkspace(
            session=session,
            messages=self.repository.list_messages(session_id),
            jobs=jobs,
            events=self.repository.list_events(session_id=session_id),
            paper=paper.model_dump(mode="json") if paper else None,
            result=self.repository.latest_result(session_id),
            candidate_bundle=candidate_bundle,
            material_bundle=material_bundle,
        )

    def update_result(
        self,
        session_id: str,
        *,
        reviewed_result: dict[str, Any],
        actor: str,
        message: str | None,
    ) -> AgentResult:
        self._require_session(session_id)
        result = self.repository.latest_result(session_id)
        if not result:
            raise ValueError("No completed mining result exists for this session.")
        updated = self.repository.update_result(
            result.result_id,
            reviewed_result,
            actor=actor,
            message=message,
        )
        if not updated:
            raise ValueError("Mining result no longer exists.")
        self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content=message or "Reviewed result changes were saved with an audit event.",
            message_type="status",
            metadata={"action": "result_updated", "actor": actor},
        )
        return updated

    def _run_job(self, job_id: str, force: bool) -> None:
        job = self.repository.get_job(job_id)
        if not job:
            return
        session = self.repository.get_session(job.session_id)
        if not session:
            return
        self.repository.update_job(
            job_id,
            status="running",
            current_step="starting",
            started_at=now_iso(),
        )
        try:
            self._run_stage(
                job,
                stage="pdf_validation",
                title="Validate uploaded PDF",
                progress=5,
                detail="Confirming the uploaded file is still available and readable.",
                runner=lambda: self._validate_registered_pdf(job.paper_id),
            )
            self._run_stage(
                job,
                stage="document_parsing",
                title="MinerU document parsing",
                progress=25,
                detail="Extracting page text, tables, captions, coordinates, and evidence blocks.",
                runner=lambda: self._parse_paper(job.paper_id, force=force),
            )
            if session.template_id == "oled_device_v1":
                raw_result, summary = self._run_oled_pipeline(job, force=force)
                result_type = "oled_device_v1"
            else:
                raw_result, summary = self._run_generic_pipeline(job, session)
                result_type = "generic_structured"
            result = self.repository.upsert_result(
                session_id=job.session_id,
                job_id=job.job_id,
                paper_id=job.paper_id,
                result_type=result_type,
                raw_result=raw_result,
            )
            final_summary = {**summary, "result_id": result.result_id}
            self.repository.update_job(
                job_id,
                status="completed",
                current_step="completed",
                progress=100.0,
                result_summary=final_summary,
                completed_at=now_iso(),
            )
            self.repository.update_session(job.session_id, status="review_ready")
            self.repository.add_event(
                job_id=job.job_id,
                session_id=job.session_id,
                event_type="result",
                stage="completed",
                status="completed",
                title="Evidence-backed mining result ready",
                detail="Review and edit the extracted values before final confirmation.",
                metadata=final_summary,
            )
            self.repository.add_message(
                session_id=job.session_id,
                role="assistant",
                content=(
                    "Mining completed. The result workspace now shows editable values together "
                    "with their source evidence and material-resolution provenance."
                ),
                message_type="result",
                metadata=final_summary,
            )
        except Exception as exc:
            self.repository.update_job(
                job_id,
                status="failed",
                current_step="failed",
                error_message=str(exc),
                completed_at=now_iso(),
            )
            self.repository.update_session(job.session_id, status="failed")
            self.repository.add_event(
                job_id=job.job_id,
                session_id=job.session_id,
                event_type="error",
                stage="failed",
                status="failed",
                title="Mining workflow failed",
                detail=str(exc),
            )
            self.repository.add_message(
                session_id=job.session_id,
                role="assistant",
                content=f"The mining workflow stopped: {exc}",
                message_type="error",
                metadata={"job_id": job.job_id},
            )

    def _run_oled_pipeline(
        self,
        job: AgentJob,
        *,
        force: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = self.candidate_ingestion.get_review_bundle(job.paper_id)
        if force or not existing or not existing.run:
            mining_result = self._run_stage(
                job,
                stage="device_extraction",
                title="DeepSeek OLED data extraction",
                progress=43,
                detail=(
                    "Extracting devices, variable layer stacks, components, fabrication data, "
                    "performance metrics, and exact source evidence."
                ),
                runner=lambda: self.llm_mining.mine_paper(
                    job.paper_id,
                    template_id="oled_device_v1",
                ),
            )
            if not mining_result:
                raise ValueError("OLED extraction returned no mining result.")
            raw_result = mining_result.mining_result
        else:
            raw_result = existing.run.mining_result
            self._record_skip(
                job,
                stage="device_extraction",
                title="Reuse existing OLED extraction",
                progress=43,
                detail="A completed candidate-v2 extraction already exists for this PDF.",
            )

        self._run_stage(
            job,
            stage="local_material_resolution",
            title="Local material identity resolution",
            progress=52,
            detail="Matching paper-local material names and aliases against confirmed materials.",
            runner=lambda: self.material_resolution.resolve_paper_materials(job.paper_id),
        )
        self._run_stage(
            job,
            stage="public_material_resolution",
            title="Public database and web resolution",
            progress=64,
            detail="Querying PubChem, OPSIN, and configured web evidence sources.",
            runner=lambda: self.material_public.resolve_paper_public(job.paper_id),
        )
        self._run_stage(
            job,
            stage="material_identity_judge",
            title="LLM material identity cross-check",
            progress=74,
            detail="Comparing names, aliases, formulas, structures, and paper context.",
            runner=lambda: self.material_judge.judge_paper_candidates(job.paper_id),
        )
        first_decisions = self._run_stage(
            job,
            stage="material_auto_decision",
            title="Conservative automatic material decisions",
            progress=80,
            detail="Accepting or rejecting only high-confidence identity judgments.",
            runner=lambda: self.material_auto.apply_paper_auto_decisions(job.paper_id),
        )

        triage = self.material_triage.triage_paper(job.paper_id)
        targets = {
            item.paper_material_id
            for item in (triage.items if triage else [])
            if item.should_run_ocsr
        }
        if targets and self._tool_enabled(
            self._require_session(job.session_id).plan,
            "molecular_figure_ocsr",
        ):
            ocsr = self._run_stage(
                job,
                stage="molecular_figure_ocsr",
                title="VLM + DECIMER molecular structure recognition",
                progress=94,
                detail=(
                    f"Inspecting molecular figures for {len(targets)} unresolved material(s), "
                    "binding labels to crops, and converting readable structures to SMILES."
                ),
                runner=lambda: self.material_agent.run_ocsr_pipeline(
                    job.paper_id,
                    vision_provider="qwen",
                    allow_unreviewed_matches=True,
                    min_model_confidence=0.8,
                    vlm_concurrency=max(1, self.config.batch_worker.material_vlm_concurrency),
                    target_paper_material_ids=targets,
                ),
            )
            self._run_stage(
                job,
                stage="post_ocsr_auto_decision",
                title="Review OCSR candidates automatically",
                progress=97,
                detail="Applying the same conservative policy to newly recognized structures.",
                runner=lambda: self.material_auto.apply_paper_auto_decisions(job.paper_id),
            )
        else:
            ocsr = None
            self._record_skip(
                job,
                stage="molecular_figure_ocsr",
                title="Molecular figure OCSR not required",
                progress=94,
                detail=(
                    "No unresolved in-scope small-molecule material requires image recognition."
                    if not targets
                    else "The approved plan disabled molecular figure OCSR."
                ),
            )

        bundle = self.candidate_ingestion.get_review_bundle(job.paper_id)
        materials = self.material_resolution.get_material_structure_bundle(job.paper_id)
        summary = {
            "candidate_value_count": len(bundle.values) if bundle else 0,
            "evidence_anchor_count": len(bundle.evidence_anchors) if bundle else 0,
            "material_count": len(materials.materials) if materials else 0,
            "accepted_material_count": (
                sum(1 for candidate in materials.structure_candidates if candidate.status == "accepted")
                if materials
                else 0
            ),
            "auto_accepted_count": getattr(first_decisions, "accepted_count", 0),
            "ocsr_candidate_count": getattr(ocsr, "ocsr_candidate_count", 0),
        }
        return raw_result, summary

    def _run_generic_pipeline(
        self,
        job: AgentJob,
        session: AgentSession,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._run_stage(
            job,
            stage="generic_extraction",
            title="LLM plan-driven structured extraction",
            progress=88,
            detail="Applying the approved custom schema and evidence policy to MinerU blocks.",
            runner=lambda: self._extract_generic(job.paper_id, session.plan),
        )
        evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
        verified_count = sum(1 for item in evidence if isinstance(item, dict) and item.get("verified"))
        records = result.get("records") if isinstance(result.get("records"), list) else []
        return result, {
            "record_count": len(records),
            "evidence_count": len(evidence),
            "verified_evidence_count": verified_count,
        }

    def _run_stage(
        self,
        job: AgentJob,
        *,
        stage: str,
        title: str,
        progress: float,
        detail: str,
        runner: Callable[[], Any],
    ) -> Any:
        self.repository.update_job(
            job.job_id,
            status="running",
            current_step=stage,
            progress=max(0.0, min(progress - 3, 99.0)),
        )
        started = time.monotonic()
        self.repository.add_event(
            job_id=job.job_id,
            session_id=job.session_id,
            event_type="tool_call",
            stage=stage,
            status="running",
            title=title,
            detail=detail,
        )
        try:
            result = runner()
        except Exception as exc:
            self.repository.add_event(
                job_id=job.job_id,
                session_id=job.session_id,
                event_type="tool_result",
                stage=stage,
                status="failed",
                title=f"{title} failed",
                detail=str(exc),
                metadata={"duration_seconds": round(time.monotonic() - started, 3)},
            )
            raise
        duration = round(time.monotonic() - started, 3)
        self.repository.update_job(
            job.job_id,
            current_step=stage,
            progress=max(0.0, min(progress, 99.0)),
        )
        self.repository.add_event(
            job_id=job.job_id,
            session_id=job.session_id,
            event_type="tool_result",
            stage=stage,
            status="completed",
            title=f"{title} completed",
            detail=_result_detail(result),
            metadata={"duration_seconds": duration, **_result_metadata(result)},
        )
        return result

    def _record_skip(
        self,
        job: AgentJob,
        *,
        stage: str,
        title: str,
        progress: float,
        detail: str,
    ) -> None:
        self.repository.update_job(job.job_id, current_step=stage, progress=progress)
        self.repository.add_event(
            job_id=job.job_id,
            session_id=job.session_id,
            event_type="tool_result",
            stage=stage,
            status="skipped",
            title=title,
            detail=detail,
        )

    def _parse_paper(self, paper_id: str, *, force: bool) -> Any:
        paper = self.paper_service.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Paper not found: {paper_id}")
        if not force and paper.parse_status == "parsed" and self.blocks.list_by_paper(paper_id):
            return {"status": "reused", "block_count": len(self.blocks.list_by_paper(paper_id))}
        return self.mineru.parse_paper(paper_id)

    def _validate_registered_pdf(self, paper_id: str) -> dict[str, Any]:
        path = self.paper_service.get_pdf_path(paper_id)
        if not path:
            raise FileNotFoundError(f"Uploaded PDF is unavailable for {paper_id}.")
        reader = PdfReader(path)
        if not reader.pages:
            raise ValueError("Uploaded PDF has no readable pages.")
        return {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "page_count": len(reader.pages),
        }

    def _extract_generic(self, paper_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        blocks = self.blocks.list_by_paper(paper_id)
        if not blocks:
            raise ValueError("No MinerU document blocks are available for generic extraction.")
        provider_config = self.config.llm.providers[self.config.llm.default_provider]
        client = OpenAICompatibleLLMClient(provider_config)
        source_items: list[dict[str, Any]] = []
        used_chars = 0
        for block in blocks:
            if used_chars >= self.config.llm.max_source_chars:
                break
            remaining = self.config.llm.max_source_chars - used_chars
            text = block.text[:remaining]
            source_items.append(
                {
                    "block_id": block.block_id,
                    "page_id": block.page_id,
                    "block_type": block.block_type,
                    "text": text,
                }
            )
            used_chars += len(text)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are DataMining Agent. Execute the approved scientific extraction plan. "
                    "Return JSON only with top-level keys records and evidence. Every extracted "
                    "value must reference one or more evidence_id values. Evidence.quote must be "
                    "copied verbatim from one supplied block; never paraphrase or invent evidence. "
                    "Use null when the paper does not report a requested value."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt_version": GENERIC_MINING_PROMPT_VERSION,
                        "approved_plan": plan,
                        "required_output": {
                            "records": [
                                {
                                    "entity_type": "string",
                                    "entity_label": "string or null",
                                    "fields": [
                                        {
                                            "field_name": "string",
                                            "value": "JSON scalar/object/list or null",
                                            "confidence": "0..1",
                                            "evidence_refs": ["E001"],
                                        }
                                    ],
                                }
                            ],
                            "evidence": [
                                {
                                    "evidence_id": "E001",
                                    "block_id": "supplied block_id",
                                    "page_id": "integer",
                                    "quote": "verbatim source text",
                                }
                            ],
                        },
                        "document_blocks": source_items,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = client.generate_json(messages, max_tokens=12000)
        if response.parsed_json is None:
            raise ValueError(f"Generic extraction returned invalid JSON: {response.parse_error}")
        return self._align_generic_evidence(response.parsed_json, blocks)

    @staticmethod
    def _align_generic_evidence(payload: dict[str, Any], blocks: list[Any]) -> dict[str, Any]:
        block_by_id = {block.block_id: block for block in blocks}
        output = json.loads(json.dumps(payload, ensure_ascii=False))
        evidence_items = output.get("evidence")
        if not isinstance(evidence_items, list):
            output["evidence"] = []
            return output
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            selected = block_by_id.get(str(item.get("block_id") or ""))
            similarity = _text_similarity(quote, selected.text) if selected else 0.0
            if not selected or similarity < 0.8:
                selected, similarity = _best_evidence_block(quote, blocks)
            item["verified"] = bool(selected and quote and similarity >= 0.8)
            item["similarity"] = round(similarity, 4)
            if selected:
                item["block_id"] = selected.block_id
                item["page_id"] = selected.page_id
                item["bbox"] = selected.bbox
                item["source_text"] = selected.text
        return output

    def _draft_custom_plan_turn(
        self,
        session_id: str,
        *,
        force_plan: bool = False,
    ) -> AgentMessage:
        self._require_session(session_id)
        messages = self.repository.list_messages(session_id)
        user_context = [message.content for message in messages if message.role == "user"]
        if not user_context:
            return self.repository.add_message(
                session_id=session_id,
                role="assistant",
                content="Describe the scientific domain and requested output before I draft a plan.",
            )
        provider_config = self.config.llm.providers[self.config.llm.default_provider]
        client = OpenAICompatibleLLMClient(provider_config)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You design auditable scientific PDF data-mining plans. Decide whether the "
                    "requirements are sufficient. Return JSON only. If information is missing, "
                    "return status=needs_clarification, assistant_message, and questions. If "
                    "sufficient, return status=plan_ready, assistant_message, and a complete plan "
                    "with domain, objective, entities, fields, controlled vocabularies, evidence "
                    "policy, validation rules, tools, and ordered execution stages. Tool ids may "
                    "include mineru, llm_extraction, web_search, public_databases, vlm, decimer, "
                    "and cross_validation. Do not claim a tool is available unless requested or "
                    "generally applicable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt_version": CUSTOM_PLAN_PROMPT_VERSION,
                        "force_plan": force_plan,
                        "requirements_conversation": user_context,
                        "critic_extension": {
                            "status": "not_configured",
                            "future_flow": "agent_draft -> critic_review -> user_approval",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = client.generate_json(prompt, max_tokens=7000)
            payload = response.parsed_json
            if payload is None:
                raise ValueError(response.parse_error or "plan response is not JSON")
        except Exception as exc:
            return self.repository.add_message(
                session_id=session_id,
                role="assistant",
                content=(
                    "I could not call the planning model. Your requirements are preserved; "
                    "check the configured LLM service and try Generate plan again."
                ),
                message_type="error",
                metadata={"error": str(exc)},
            )
        status = str(payload.get("status") or "needs_clarification")
        assistant_text = str(
            payload.get("assistant_message")
            or "Please add the missing requirements before the plan is drafted."
        )
        if status == "plan_ready" and isinstance(payload.get("plan"), dict):
            plan = self._normalize_custom_plan(payload["plan"], fallback_domain="custom")
            self.repository.update_session(
                session_id,
                plan=plan,
                domain=str(plan.get("domain") or "custom"),
                status="awaiting_plan_approval",
                plan_status="awaiting_user_approval",
                requirements={"conversation": user_context},
            )
            return self.repository.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_text,
                message_type="plan",
                metadata={
                    "plan_status": "awaiting_user_approval",
                    "usage": response.usage,
                },
            )
        questions = payload.get("questions")
        self.repository.update_session(
            session_id,
            status="collecting_requirements",
            plan_status="collecting_requirements",
            requirements={"conversation": user_context, "questions": questions or []},
        )
        return self.repository.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_text,
            metadata={"questions": questions if isinstance(questions, list) else []},
        )

    @staticmethod
    def _normalize_custom_plan(
        plan: dict[str, Any],
        *,
        fallback_domain: str,
    ) -> dict[str, Any]:
        normalized = json.loads(json.dumps(plan, ensure_ascii=False))
        normalized.setdefault("plan_id", f"custom-{uuid4().hex[:12]}")
        normalized.setdefault("name", "Custom scientific data-mining plan")
        normalized.setdefault("domain", fallback_domain or "custom")
        normalized.setdefault("objective", "Extract requested structured facts from one PDF.")
        normalized.setdefault("entities", [])
        normalized.setdefault(
            "evidence_policy",
            {
                "required_for_every_value": True,
                "quote_must_be_verbatim": True,
                "store_page_block_and_bbox": True,
            },
        )
        normalized.setdefault("validation_rules", [])
        normalized.setdefault(
            "tools",
            [
                {"id": "mineru", "name": "MinerU", "enabled": True},
                {"id": "llm_extraction", "name": "LLM extraction", "enabled": True},
                {"id": "cross_validation", "name": "Evidence cross-check", "enabled": True},
            ],
        )
        normalized.setdefault(
            "execution_stages",
            [
                "pdf_validation",
                "document_parsing",
                "generic_extraction",
                "evidence_alignment",
                "human_review",
            ],
        )
        return normalized

    @staticmethod
    def _oled_plan() -> dict[str, Any]:
        return {
            "plan_id": "oled-device-material-v1",
            "name": "OLED Device and Material Mining v1",
            "domain": "oled",
            "objective": (
                "Build an evidence-backed record of OLED device architectures, fabrication, "
                "performance, and in-scope organic small-molecule structures."
            ),
            "template_id": "oled_device_v1",
            "scope": {
                "include": [
                    "OLEDs with organic small-molecule emitters",
                    "fluorescent, TADF, MR-TADF, hyperfluorescent, and eligible exciplex OLEDs",
                    "polymer hosts when the emitter remains an organic small molecule",
                ],
                "exclude_or_auto_classify": [
                    "perovskite, quantum-dot, and inorganic LEDs",
                    "polymer emitters",
                    "organometallic emitters from the current molecular ML subset",
                ],
            },
            "entities": [
                {
                    "name": "devices",
                    "cardinality": "many",
                    "children": ["layers", "performance"],
                    "description": "One record per reported OLED device.",
                },
                {
                    "name": "layers",
                    "cardinality": "variable per device",
                    "children": ["components"],
                    "description": "Ordered physical device stack with role and thickness.",
                },
                {
                    "name": "materials",
                    "cardinality": "many",
                    "description": (
                        "Paper-local names, full names, abbreviations, usages, and independently "
                        "resolved molecular structures."
                    ),
                },
                {
                    "name": "evidence",
                    "cardinality": "many",
                    "description": "Verbatim text, page, MinerU block, and PDF coordinates.",
                },
            ],
            "evidence_policy": {
                "required_for_every_non_null_value": True,
                "quote_must_be_verbatim": True,
                "store_page_block_and_bbox": True,
                "preserve_raw_llm_output": True,
                "unverified_values_require_human_review": True,
            },
            "validation_rules": [
                "Validate against oled_device_v1 field types and controlled vocabularies.",
                "Retain null instead of inventing an unreported value.",
                "Cross-check public material candidates with an LLM identity judge.",
                "Auto-decide material candidates only at configured high-confidence thresholds.",
                "Keep all user edits and confirmations in review-event history.",
            ],
            "tools": [
                {
                    "id": "mineru",
                    "name": "MinerU document parser",
                    "enabled": True,
                    "condition": "always",
                },
                {
                    "id": "llm_extraction",
                    "name": "DeepSeek structured extraction",
                    "enabled": True,
                    "condition": "always",
                },
                {
                    "id": "local_material_database",
                    "name": "Local confirmed material database",
                    "enabled": True,
                    "condition": "for every device-used material",
                },
                {
                    "id": "public_databases",
                    "name": "PubChem and OPSIN",
                    "enabled": True,
                    "condition": "when local resolution has no confirmed structure",
                },
                {
                    "id": "web_search",
                    "name": "Configured material web search",
                    "enabled": True,
                    "condition": "when names or public candidates require evidence",
                },
                {
                    "id": "identity_judge",
                    "name": "LLM material identity judge",
                    "enabled": True,
                    "condition": "for public structure candidates",
                },
                {
                    "id": "molecular_figure_ocsr",
                    "name": "Qwen VLM + DECIMER OCSR",
                    "enabled": True,
                    "condition": "only for unresolved in-scope small-molecule materials",
                },
            ],
            "execution_stages": [
                "pdf_validation",
                "document_parsing",
                "device_extraction",
                "local_material_resolution",
                "public_material_resolution",
                "material_identity_judge",
                "material_auto_decision",
                "conditional_molecular_figure_ocsr",
                "human_review",
            ],
            "critic_review": {
                "enabled": False,
                "status": "not_configured",
                "extension_point": "agent_draft -> critic_review -> user_approval",
            },
        }

    @staticmethod
    def _tool_enabled(plan: dict[str, Any], tool_id: str) -> bool:
        tools = plan.get("tools")
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if isinstance(tool, dict) and tool.get("id") == tool_id:
                return bool(tool.get("enabled", True))
        return False

    def _require_session(self, session_id: str) -> AgentSession:
        self.init_runtime()
        session = self.repository.get_session(session_id)
        if not session:
            raise KeyError(f"DataMining Agent session not found: {session_id}")
        return session

    def _drop_future(self, job_id: str) -> None:
        with self._future_lock:
            self._futures.pop(job_id, None)


def _safe_filename(value: str) -> str:
    name = Path(value or "uploaded.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._() -]+", "_", name).strip(" .")
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned or 'uploaded'}.pdf"
    return cleaned


def _result_detail(result: Any) -> str:
    if result is None:
        return "Tool completed without a structured return value."
    if isinstance(result, dict):
        return f"Tool returned {len(result)} top-level field(s)."
    model_fields = getattr(result, "model_fields", None)
    if model_fields:
        return f"Tool returned {type(result).__name__}."
    return f"Tool returned {type(result).__name__}."


def _result_metadata(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return {"result_keys": sorted(str(key) for key in result)[:20]}
    for field_name in (
        "candidate_run_id",
        "accepted_count",
        "rejected_count",
        "ocsr_candidate_count",
        "triage_count",
        "crop_count",
        "binding_count",
        "content_item_count",
    ):
        value = getattr(result, field_name, None)
        if value is not None:
            return {field_name: value}
    run = getattr(result, "run", None)
    if run is not None:
        return {
            "run_type": type(run).__name__,
            "run_id": (
                getattr(run, "llm_run_id", None)
                or getattr(run, "mineru_run_id", None)
                or getattr(run, "agent_run_id", None)
            ),
        }
    return {}


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _text_similarity(quote: str, source: str) -> float:
    normalised_quote = _normalise_text(quote)
    normalised_source = _normalise_text(source)
    if not normalised_quote or not normalised_source:
        return 0.0
    if normalised_quote in normalised_source:
        return 1.0
    return SequenceMatcher(None, normalised_quote, normalised_source).ratio()


def _best_evidence_block(quote: str, blocks: list[Any]) -> tuple[Any | None, float]:
    best = None
    best_score = 0.0
    for block in blocks:
        score = _text_similarity(quote, block.text)
        if score > best_score:
            best = block
            best_score = score
    return best, best_score
