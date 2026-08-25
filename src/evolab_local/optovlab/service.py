from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from evolab_local.mining_platform.data_mining_agent_service import DataMiningAgentService
from evolab_local.mining_platform.schemas.data_mining_agent import AgentSessionCreate
from evolab_local.optovlab.agent_runtime import MicrosoftAgentRuntime
from evolab_local.optovlab.analysis_skills import AnalysisSkillService
from evolab_local.optovlab.config import OptoVLabConfig
from evolab_local.optovlab.data_catalog import OLEDDeviceCatalog
from evolab_local.optovlab.demo_content import (
    DATA_MINING_ANALYSIS_DEMO_PROMPT,
    DATA_MINING_ANALYSIS_DEMO_SEED,
    DATA_MINING_WORKFLOW_DEMO_PROMPT,
    DATA_MINING_WORKFLOW_DEMO_SEED,
    seed_demo_content,
)
from evolab_local.optovlab.hpc_service import HPCService
from evolab_local.optovlab.rag_service import OLEDDeviceRAGService
from evolab_local.optovlab.repository import OptoVLabRepository
from evolab_local.optovlab.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AppSummary,
    ConversationTurn,
    RAGSearchResult,
    SessionCreate,
    SessionSummary,
    SessionWorkspace,
    TrainingJob,
    TrainingRequest,
)


MINING_TERMS = (
    "mine",
    "mining",
    "extract",
    "extraction",
    "oled data",
    "数据挖掘",
    "提取",
    "抽取",
    "开始运行",
)

ANALYSIS_TERMS = (
    "analy",
    "summary",
    "database",
    "dataset",
    "corpus",
    "distribution",
    "correlation",
    "visual",
    "统计",
    "分析",
    "总结",
    "数据库",
    "数据集",
    "概况",
    "分布",
    "相关",
    "可视化",
)


class OptoVLabService:
    def __init__(
        self,
        config: OptoVLabConfig,
        data_mining_service: DataMiningAgentService,
    ) -> None:
        self.config = config
        self.data_mining_service = data_mining_service
        self.repository = OptoVLabRepository(config.runtime.sqlite_path)
        self.catalog = OLEDDeviceCatalog(config.datasets.oled_devices)
        self.analysis = AnalysisSkillService(config.runtime.artifact_dir, self.repository)
        self.rag = OLEDDeviceRAGService(
            self.catalog,
            config.retrieval,
            cache_dir=config.runtime.root / "rag_index",
        )
        self.hpc = HPCService(config.modeling, config.runtime.root)
        self.agent_runtime = MicrosoftAgentRuntime(config.agents)

    def init_runtime(self) -> None:
        self.config.runtime.root.mkdir(parents=True, exist_ok=True)
        self.config.runtime.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.repository.init_runtime()
        self.repository.reconcile_interrupted_events()
        seed_demo_content(self.repository)
        self._seed_data_mining_workflow_demo()
        self._seed_data_mining_analysis_demo()

    async def close(self) -> None:
        await self.agent_runtime.close()

    def list_apps(self) -> list[AppSummary]:
        stats = self.catalog.stats()
        return [
            AppSummary(
                app_id="data-mining",
                name="Data Mining Agent",
                category="Research agents",
                description=(
                    "Mine evidence-backed OLED device and material data from PDFs, review "
                    "the result, and analyze one or many papers."
                ),
                route="/agents/data-mining",
                metrics={"template": "OLED device v1", "analysis_skills": len(self.analysis.catalog())},
            ),
            AppSummary(
                app_id="device-modeling",
                name="Device Modeling Agent",
                category="Research agents",
                description=(
                    "Inspect model-ready data, develop OLED-GAT experiments, and run controlled "
                    "training jobs on the local Slurm GPU cluster."
                ),
                route="/agents/device-modeling",
                metrics={"models": len(self.hpc.model_registry()), "framework": "PyTorch Geometric"},
            ),
            AppSummary(
                app_id="experimental-design",
                name="Experimental Design Agent",
                category="Research agents",
                description=(
                    "Retrieve closely related OLED devices and produce testable, provenance-rich "
                    "experimental recommendations."
                ),
                route="/agents/experimental-design",
                metrics={"retrieval_devices": stats["devices"], "papers": stats["papers"]},
            ),
            AppSummary(
                app_id="optoelectronics-database",
                name="Organic Optoelectronics Database",
                category="Explore and search",
                description="Explore the OLED, OFET, and OPV device collections.",
                route="/database",
                metrics={"oled_devices": stats["devices"], "oled_papers": stats["papers"]},
            ),
        ]

    def create_session(self, payload: SessionCreate) -> SessionSummary:
        session = self.repository.create_session(payload.agent_type, payload.title)
        self.repository.add_message(
            session.session_id,
            "assistant",
            self._welcome_message(payload.agent_type),
            metadata={"agent_type": payload.agent_type, "runtime": self.agent_runtime.describe()},
        )
        return session

    def get_workspace(self, session_id: str) -> SessionWorkspace:
        session = self._require_session(session_id)
        resources = self.repository.list_resources(session_id)
        linked_workspaces: list[dict[str, Any]] = []
        for resource in resources:
            if resource.resource_type != "data_mining_session":
                continue
            try:
                workspace = self.data_mining_service.get_workspace(resource.resource_id)
            except (KeyError, ValueError):
                continue
            linked_workspaces.append(workspace.model_dump(mode="json"))
        return SessionWorkspace(
            session=session,
            messages=self.repository.list_messages(session_id),
            tool_events=self.repository.list_tool_events(session_id),
            artifacts=self.repository.list_artifacts(session_id),
            resources=resources,
            linked_workspaces=linked_workspaces,
        )

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete the conversation shell while preserving scientific data assets."""
        session = self._require_session(session_id)
        if any(event.status == "running" for event in self.repository.list_tool_events(session_id)):
            raise ValueError("Wait for the active agent operation to finish before deleting this session")

        linked_resources = self.repository.list_resources(session_id)
        for resource in linked_resources:
            if resource.resource_type != "data_mining_session":
                continue
            try:
                workspace = self.data_mining_service.get_workspace(resource.resource_id)
            except (KeyError, ValueError):
                continue
            if any(job.status in {"queued", "running"} for job in workspace.jobs):
                raise ValueError("Wait for the active mining job to finish before deleting this session")

        artifact_root = self.config.runtime.artifact_dir.resolve()
        artifact_dir = (artifact_root / session_id).resolve()
        if not artifact_dir.is_relative_to(artifact_root):
            raise ValueError("Invalid session artifact path")
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        if not self.repository.delete_session(session_id):
            raise KeyError(f"Unknown OptoVLab session: {session_id}")
        return {
            "session_id": session_id,
            "agent_type": session.agent_type,
            "deleted": True,
            "preserved_linked_resources": len(linked_resources),
        }

    def upload_pdf(self, session_id: str, filename: str, content: bytes) -> dict[str, Any]:
        session = self._require_session(session_id)
        if session.agent_type != "data_mining":
            raise ValueError("PDF mining uploads are only available in Data Mining Agent sessions")
        mining_session = self.data_mining_service.create_session(
            AgentSessionCreate(
                mode="preset",
                template_id="oled_device_v1",
                title=Path(filename).stem,
                initial_message="Mine OLED device and material data from this paper.",
            )
        )
        if mining_session.plan_status != "approved":
            mining_session = self.data_mining_service.approve_plan(
                mining_session.session_id,
                actor="optovlab_user",
                message="OLED preset selected when the PDF was attached in OptoVLab.",
            )
        uploaded = self.data_mining_service.upload_pdf(
            mining_session.session_id,
            filename=filename,
            content=content,
        )
        link = self.repository.link_resource(
            session_id,
            "data_mining_session",
            mining_session.session_id,
            filename,
            {
                "paper_id": uploaded.paper_id,
                "size_bytes": uploaded.size_bytes,
                "sha256": uploaded.sha256,
                "page_count": uploaded.page_count,
            },
        )
        self.repository.add_message(
            session_id,
            "user",
            f"Uploaded {uploaded.filename}",
            message_type="file",
            metadata={
                "resource_id": mining_session.session_id,
                "paper_id": uploaded.paper_id,
                "filename": uploaded.filename,
                "mime_type": "application/pdf",
                "size_bytes": uploaded.size_bytes,
                "page_count": uploaded.page_count,
                "sha256": uploaded.sha256,
            },
        )
        self.repository.add_tool_event(
            session_id,
            "pdf_upload",
            "completed",
            f"Validated {filename}",
            detail=f"{uploaded.page_count} pages, {uploaded.size_bytes:,} bytes",
            payload={"mining_session_id": mining_session.session_id, "paper_id": uploaded.paper_id},
        )
        if len(self.repository.list_resources(session_id)) == 1:
            self.repository.update_session(session_id, title=Path(filename).stem)
        return {
            "resource": link.model_dump(mode="json"),
            "upload": uploaded.model_dump(mode="json"),
            "mining_session": uploaded.session.model_dump(mode="json"),
        }

    def start_data_mining(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        if session.agent_type != "data_mining":
            raise ValueError("This action requires a Data Mining Agent session")
        resources = [
            resource
            for resource in self.repository.list_resources(session_id)
            if resource.resource_type == "data_mining_session"
        ]
        if not resources:
            raise ValueError("Upload at least one PDF before starting OLED mining")
        started: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for resource in resources:
            workspace = self.data_mining_service.get_workspace(resource.resource_id)
            latest = workspace.jobs[-1] if workspace.jobs else None
            if latest and latest.status in {"queued", "running", "completed"}:
                skipped.append(
                    {
                        "mining_session_id": resource.resource_id,
                        "job_id": latest.job_id,
                        "status": latest.status,
                    }
                )
                continue
            if workspace.session.plan_status != "approved":
                self.data_mining_service.approve_plan(
                    resource.resource_id,
                    actor="optovlab_user",
                    message="OLED preset selected in OptoVLab.",
                )
            job = self.data_mining_service.start_job(resource.resource_id)
            started.append(job.model_dump(mode="json"))
            self.repository.add_tool_event(
                session_id,
                "oled_device_v1_pipeline",
                "running",
                f"Started OLED mining for {resource.label}",
                detail="MinerU -> DeepSeek -> material resolution -> conditional VLM/DECIMER",
                payload={"mining_session_id": resource.resource_id, "paper_id": job.paper_id},
                job_id=job.job_id,
            )
        return {"started": started, "skipped": skipped, "total_pdfs": len(resources)}

    async def handle_message(self, session_id: str, content: str) -> ConversationTurn:
        session = self._require_session(session_id)
        user_message = self.repository.add_message(session_id, "user", content)
        prior = [
            {"role": message.role, "content": message.content}
            for message in self.repository.list_messages(session_id)[-13:-1]
            if message.role in {"user", "assistant"}
        ]
        before_event_ids = {
            event.event_id for event in self.repository.list_tool_events(session_id)
        }
        artifacts = []
        assistant_message_type = "text"
        assistant_metadata: dict[str, Any] = {"agent_type": session.agent_type}
        if session.agent_type == "data_mining" and _contains(content, ANALYSIS_TERMS):
            request = self._analysis_request_from_message(content)
            analysis_result = self.run_analysis(session_id, request)
            artifacts = analysis_result.artifacts
            response_text = analysis_result.summary
            assistant_message_type = "analysis"
            assistant_metadata["analysis"] = self._analysis_message_metadata(
                request,
                analysis_result,
            )
        elif session.agent_type == "data_mining" and _contains(content, MINING_TERMS):
            result = self.start_data_mining(session_id)
            response_text = self._format_mining_start(result)
        elif session.agent_type == "experimental_design":
            search_result = self.search_devices(content)
            self.repository.add_tool_event(
                session_id,
                "oled_device_rag",
                "completed",
                f"Retrieved {len(search_result.hits)} device precedents",
                detail="Hybrid character TF-IDF and structured filtering over the frozen OLED corpus.",
                payload={
                    "query": content,
                    "hits": [hit.model_dump(mode="json") for hit in search_result.hits],
                },
            )
            response_text = await self._agent_response(
                session,
                content,
                prior,
                context={"retrieved_devices": [hit.model_dump(mode="json") for hit in search_result.hits]},
            )
        elif session.agent_type == "device_modeling" and _contains(
            content, ("status", "gpu", "slurm", "queue", "状态", "显卡", "队列")
        ):
            status = self.hpc.status()
            self.repository.add_tool_event(
                session_id,
                "hpc_status",
                "completed",
                "Inspected Slurm and GPU status",
                payload=status.model_dump(mode="json"),
            )
            response_text = self._format_hpc_status(status.model_dump(mode="json"))
        else:
            response_text = await self._agent_response(
                session,
                content,
                prior,
                context=self._context_for(session_id, session.agent_type),
            )
        assistant_message = self.repository.add_message(
            session_id,
            "assistant",
            response_text,
            message_type=assistant_message_type,
            metadata=assistant_metadata,
        )
        new_events = [
            event
            for event in self.repository.list_tool_events(session_id)
            if event.event_id not in before_event_ids
        ]
        return ConversationTurn(
            session=self._require_session(session_id),
            user_message=user_message,
            assistant_message=assistant_message,
            tool_events=list(reversed(new_events)),
            artifacts=artifacts,
        )

    def run_analysis(self, session_id: str, request: AnalysisRequest) -> AnalysisResult:
        self._require_session(session_id)
        records = [] if request.scope == "catalog" else self._linked_device_records(session_id)
        if request.scope == "catalog" or (
            request.scope == "auto" and not records and not request.session_ids
        ):
            records = self.catalog.records()
        for linked_session_id in request.session_ids:
            records.extend(self._linked_device_records(linked_session_id))
        records = _deduplicate_records(records)
        event = self.repository.add_tool_event(
            session_id,
            request.skill_id,
            "running",
            f"Running {request.skill_id.replace('_', ' ')}",
            payload={"record_count": len(records), "scope": request.scope},
        )
        try:
            result = self.analysis.run(session_id, request, records)
        except Exception as exc:
            self.repository.update_tool_event(
                event.event_id,
                status="failed",
                title=f"{request.skill_id.replace('_', ' ').title()} failed",
                detail=str(exc),
            )
            raise
        self.repository.update_tool_event(
            event.event_id,
            status="completed",
            title=f"Completed {request.skill_id.replace('_', ' ')}",
            detail=result.summary,
            payload=result.statistics,
        )
        return result

    def search_devices(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGSearchResult:
        return self.rag.search(query, top_k=top_k, filters=filters)

    def prepare_training(self, session_id: str, request: TrainingRequest) -> TrainingJob:
        session = self._require_session(session_id)
        if session.agent_type != "device_modeling":
            raise ValueError("Training jobs belong to Device Modeling Agent sessions")
        job = self.hpc.prepare_training(request)
        self.repository.add_tool_event(
            session_id,
            "oled_gat_training",
            "completed" if job.submitted else "prepared",
            "Submitted OLED-GAT training" if job.submitted else "Prepared OLED-GAT training",
            detail=(
                f"Slurm job {job.scheduler_job_id}" if job.scheduler_job_id else "Review the script before submission."
            ),
            payload=job.model_dump(mode="json"),
            job_id=job.scheduler_job_id or job.job_id,
        )
        return job

    def update_mining_result(
        self,
        session_id: str,
        mining_session_id: str,
        reviewed_result: dict[str, Any],
        message: str | None = None,
    ) -> dict[str, Any]:
        self._require_link(session_id, mining_session_id)
        result = self.data_mining_service.update_result(
            mining_session_id,
            reviewed_result=reviewed_result,
            actor="optovlab_user",
            message=message or "Edited in the OptoVLab review workspace.",
        )
        self.repository.add_tool_event(
            session_id,
            "result_review",
            "completed",
            "Saved reviewed mining result",
            payload={"mining_session_id": mining_session_id, "result_id": result.result_id},
        )
        return result.model_dump(mode="json")

    async def _agent_response(
        self,
        session: SessionSummary,
        content: str,
        history: list[dict[str, str]],
        *,
        context: dict[str, Any],
    ) -> str:
        event = self.repository.add_tool_event(
            session.session_id,
            "microsoft_agent_framework",
            "running",
            f"Consulting {session.agent_type.replace('_', ' ')} agent",
            payload=self.agent_runtime.describe(),
        )
        try:
            response = await self.agent_runtime.respond(
                session.agent_type,
                content,
                history=history,
                context=context,
                tools=self._tools_for(session),
            )
        except Exception as exc:
            self.repository.update_tool_event(
                event.event_id,
                status="failed",
                title="Agent model call failed",
                detail=str(exc),
            )
            return self._fallback_response(session.agent_type, context, str(exc))
        self.repository.update_tool_event(
            event.event_id,
            status="completed",
            title="Agent response completed",
            payload={"model": self.config.agents.model},
        )
        return response

    def _tools_for(self, session: SessionSummary) -> list[Any]:
        session_id = session.session_id

        def get_oled_dataset_summary() -> str:
            """Return authoritative counts and quality tiers for the OLED device corpus."""
            payload = self.catalog.stats()
            self.repository.add_tool_event(
                session_id,
                "oled_dataset_summary",
                "completed",
                "Read OLED dataset summary",
                payload=payload,
            )
            return json.dumps(payload, ensure_ascii=False)

        def search_oled_devices(query: str, top_k: int = 6) -> str:
            """Search OLED device precedents and return DOI-grounded structured records."""
            result = self.search_devices(query, top_k=min(max(top_k, 1), 12))
            self.repository.add_tool_event(
                session_id,
                "oled_device_rag",
                "completed",
                f"Retrieved {len(result.hits)} OLED devices",
                payload={"query": query},
            )
            return result.model_dump_json()

        def inspect_hpc_status() -> str:
            """Inspect current Slurm partitions, jobs, and local GPU availability."""
            result = self.hpc.status()
            self.repository.add_tool_event(
                session_id,
                "hpc_status",
                "completed",
                "Inspected HPC status",
                payload=result.model_dump(mode="json"),
            )
            return result.model_dump_json()

        def list_oled_gat_models() -> str:
            """List versioned OLED-GAT configurations and available evaluation metrics."""
            result = self.hpc.model_registry()
            self.repository.add_tool_event(
                session_id,
                "model_registry",
                "completed",
                f"Found {len(result)} OLED-GAT configurations",
            )
            return json.dumps(result, ensure_ascii=False, default=str)

        if session.agent_type == "data_mining":
            return [get_oled_dataset_summary]
        if session.agent_type == "device_modeling":
            return [get_oled_dataset_summary, search_oled_devices, inspect_hpc_status, list_oled_gat_models]
        return [get_oled_dataset_summary, search_oled_devices]

    def _context_for(self, session_id: str, agent_type: str) -> dict[str, Any]:
        if agent_type == "data_mining":
            resources = self.repository.list_resources(session_id)
            return {
                "uploaded_pdfs": len(resources),
                "linked_device_records": len(self._linked_device_records(session_id)),
                "available_analysis_skills": self.analysis.catalog(),
            }
        if agent_type == "device_modeling":
            return {
                "dataset": self.catalog.stats(),
                "models": self.hpc.model_registry(),
                "submission_policy": "Explicit confirm_submit=true is required.",
            }
        return {"dataset": self.catalog.stats(), "retrieval_policy": "Cite DOI and device ID."}

    def _linked_device_records(self, session_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for resource in self.repository.list_resources(session_id):
            if resource.resource_type != "data_mining_session":
                continue
            try:
                workspace = self.data_mining_service.get_workspace(resource.resource_id)
            except (KeyError, ValueError):
                continue
            if not workspace.result:
                continue
            payload = workspace.result.reviewed_result or workspace.result.raw_result
            paper = workspace.paper or {}
            for index, device in enumerate(payload.get("devices") or []):
                if not isinstance(device, dict):
                    continue
                record = dict(device)
                record.setdefault("id", f"{workspace.result.paper_id}::D{index + 1}")
                record.setdefault("paper_id", workspace.result.paper_id)
                for key in ("doi", "title", "journal", "publisher", "year"):
                    record.setdefault(key, paper.get(key))
                record.setdefault("architecture", record.get("architecture_text"))
                final_emitter = record.get("final_emitter")
                if isinstance(final_emitter, dict):
                    record["final_emitter"] = final_emitter.get("mention")
                    record["final_emitter_class"] = final_emitter.get("material_class")
                record.setdefault("material_count", len(payload.get("paper_materials") or []))
                record.setdefault("layer_count", len(record.get("layers") or []))
                for performance in record.get("performance") or []:
                    if not isinstance(performance, dict):
                        continue
                    metric = str(
                        performance.get("metric_name") or performance.get("metric_family") or ""
                    ).lower()
                    statistic = str(performance.get("statistic") or "").lower()
                    if metric == "eqe" and statistic in {"max", "maximum"}:
                        record.setdefault(
                            "eqe_max",
                            performance.get("normalized_value")
                            if performance.get("normalized_value") is not None
                            else performance.get("raw_value"),
                        )
                records.append(record)
        return records

    def _require_session(self, session_id: str) -> SessionSummary:
        session = self.repository.get_session(session_id)
        if not session:
            raise KeyError(f"Unknown OptoVLab session: {session_id}")
        return session

    def _require_link(self, session_id: str, mining_session_id: str) -> None:
        self._require_session(session_id)
        if not any(
            resource.resource_type == "data_mining_session"
            and resource.resource_id == mining_session_id
            for resource in self.repository.list_resources(session_id)
        ):
            raise KeyError("The mining session is not linked to this OptoVLab session")

    @staticmethod
    def _analysis_request_from_message(content: str) -> AnalysisRequest:
        lowered = content.lower()
        scope = (
            "catalog"
            if any(
                term in lowered
                for term in (
                    "database",
                    "full corpus",
                    "whole corpus",
                    "all devices",
                    "数据库",
                    "全库",
                    "全部器件",
                )
            )
            else "auto"
        )
        numeric_fields = [
            field
            for field in (
                "eqe_max",
                "ce_max",
                "pe_max",
                "luminance_max",
                "turn_on_voltage",
                "layer_count",
                "material_count",
                "el_peak",
                "fwhm",
                "lifetime",
            )
            if field.lower() in lowered
        ]
        if any(term in lowered for term in ("quality", "missing", "完整", "缺失")):
            return AnalysisRequest(skill_id="data_quality_profile", scope=scope)
        if any(term in lowered for term in ("matrix", "多变量", "相关矩阵")):
            return AnalysisRequest(skill_id="correlation_matrix", scope=scope)
        if any(term in lowered for term in ("compare", "group", "分组", "比较")):
            return AnalysisRequest(
                skill_id="group_comparison",
                scope=scope,
                metric=numeric_fields[0] if numeric_fields else "eqe_max",
                group_field="emission_color",
            )
        if any(term in lowered for term in ("relationship", "correlation", "关系", "相关")):
            return AnalysisRequest(
                skill_id="bivariate_relationship",
                scope=scope,
                x_field=numeric_fields[0] if numeric_fields else "layer_count",
                y_field=numeric_fields[1] if len(numeric_fields) > 1 else "eqe_max",
            )
        if any(term in lowered for term in ("distribution", "histogram", "分布", "直方图")):
            return AnalysisRequest(
                skill_id="univariate_distribution",
                scope=scope,
                metric=numeric_fields[0] if numeric_fields else "eqe_max",
            )
        return AnalysisRequest(skill_id="dataset_summary", scope=scope)

    @staticmethod
    def _analysis_message_metadata(
        request: AnalysisRequest,
        result: AnalysisResult,
    ) -> dict[str, Any]:
        return {
            "skill_id": result.skill_id,
            "scope": request.scope,
            "statistics": result.statistics,
            "artifacts": [artifact.model_dump(mode="json") for artifact in result.artifacts],
        }

    def _seed_data_mining_analysis_demo(self) -> None:
        session = next(
            (
                item
                for item in self.repository.list_sessions("data_mining", limit=500)
                if item.title == "Data Mining demo"
            ),
            None,
        )
        if session is None or any(
            message.metadata.get("demo_seed") == DATA_MINING_ANALYSIS_DEMO_SEED
            for message in self.repository.list_messages(session.session_id)
        ):
            return
        if not self.catalog.records():
            return
        request = AnalysisRequest(skill_id="dataset_summary", scope="catalog")
        result = self.run_analysis(session.session_id, request)
        self.repository.add_message(
            session.session_id,
            "user",
            DATA_MINING_ANALYSIS_DEMO_PROMPT,
            metadata={"demo_seed_prompt": DATA_MINING_ANALYSIS_DEMO_SEED},
            touch_session=False,
        )
        self.repository.add_message(
            session.session_id,
            "assistant",
            result.summary,
            message_type="analysis",
            metadata={
                "agent_type": "data_mining",
                "demo_seed": DATA_MINING_ANALYSIS_DEMO_SEED,
                "analysis": self._analysis_message_metadata(request, result),
            },
            touch_session=False,
        )

    def _seed_data_mining_workflow_demo(self) -> None:
        session = next(
            (
                item
                for item in self.repository.list_sessions("data_mining", limit=500)
                if item.title == "Data Mining demo"
            ),
            None,
        )
        if session is None:
            return
        messages = self.repository.list_messages(session.session_id)
        if any(
            message.metadata.get("demo_seed") == DATA_MINING_WORKFLOW_DEMO_SEED
            for message in messages
        ):
            return
        resource = next(
            (
                item
                for item in self.repository.list_resources(session.session_id)
                if item.resource_type == "data_mining_session"
            ),
            None,
        )
        if resource is None:
            return
        try:
            workspace = self.data_mining_service.get_workspace(resource.resource_id)
        except (KeyError, ValueError):
            return
        latest_job = workspace.jobs[-1] if workspace.jobs else None
        if not workspace.result or not latest_job or latest_job.status != "completed":
            return
        payload = workspace.result.reviewed_result or workspace.result.raw_result
        file_message = next(
            (message for message in workspace.messages if message.message_type == "file"),
            None,
        )
        file_metadata = file_message.metadata if file_message else {}
        filename = str(file_metadata.get("filename") or resource.label)
        page_count = int(file_metadata.get("page_count") or 0)
        device_count = len(payload.get("devices") or [])
        material_count = len(payload.get("paper_materials") or payload.get("materials") or [])
        evidence_count = len(payload.get("evidence") or [])
        mining_metadata = {
            "status": "completed",
            "filename": filename,
            "page_count": page_count,
            "device_count": device_count,
            "material_count": material_count,
            "evidence_count": evidence_count,
            "review_status": workspace.result.review_status,
            "pipeline": [
                "PDF validation",
                "MinerU document parsing",
                "DeepSeek device extraction",
                "Material resolution",
                "Evidence validation",
            ],
        }
        response = (
            f"## OLED mining completed\n\n"
            f"The validated pipeline processed **{filename}** and extracted {device_count} "
            f"devices, {material_count} materials, and {evidence_count} evidence records. "
            "The structured result is ready for review in Workbench."
        )
        base_time = datetime.fromisoformat(resource.created_at) + timedelta(seconds=1)
        user_message = next(
            (
                message
                for message in messages
                if message.role == "user"
                and "Mine OLED device and material data" in message.content
            ),
            None,
        )
        if user_message is None:
            user_message = self.repository.add_message(
                session.session_id,
                "user",
                DATA_MINING_WORKFLOW_DEMO_PROMPT,
                touch_session=False,
            )
        self.repository.update_message(
            user_message.message_id,
            role="user",
            content=DATA_MINING_WORKFLOW_DEMO_PROMPT,
            metadata={"demo_seed_prompt": DATA_MINING_WORKFLOW_DEMO_SEED},
            created_at=base_time.isoformat(timespec="seconds"),
        )
        assistant_message = next(
            (
                message
                for message in messages
                if message.role == "assistant"
                and message.content.startswith("No new jobs were started")
            ),
            None,
        )
        if assistant_message is None:
            assistant_message = self.repository.add_message(
                session.session_id,
                "assistant",
                response,
                touch_session=False,
            )
        self.repository.update_message(
            assistant_message.message_id,
            role="assistant",
            content=response,
            message_type="mining_result",
            metadata={
                "agent_type": "data_mining",
                "demo_seed": DATA_MINING_WORKFLOW_DEMO_SEED,
                "mining": mining_metadata,
            },
            created_at=(base_time + timedelta(seconds=1)).isoformat(timespec="seconds"),
        )
    @staticmethod
    def _format_mining_start(result: dict[str, Any]) -> str:
        started = len(result["started"])
        skipped = len(result["skipped"])
        if started:
            return (
                f"Started the OLED device v1 pipeline for {started} PDF(s). "
                "Tool events will update as MinerU, DeepSeek, material resolution, and conditional "
                f"VLM/DECIMER stages complete. {skipped} existing job(s) were left unchanged."
            )
        return f"No new jobs were started; {skipped} linked PDF job(s) are already active or complete."

    @staticmethod
    def _format_hpc_status(status: dict[str, Any]) -> str:
        return (
            f"Slurm reports {len(status['partitions'])} partition(s) and "
            f"{len(status['jobs'])} visible job(s). This host exposes {len(status['gpus'])} GPU(s). "
            "Open the HPC panel for memory and queue details."
        )

    @staticmethod
    def _fallback_response(agent_type: str, context: dict[str, Any], error: str) -> str:
        if agent_type == "experimental_design" and context.get("retrieved_devices"):
            lines = [
                "The model synthesis call failed, but the evidence retrieval completed. "
                "The closest precedents are:",
            ]
            for item in context["retrieved_devices"][:5]:
                lines.append(
                    f"- {item.get('device_id')} | DOI {item.get('doi') or 'unknown'} | "
                    f"EQEmax {item.get('eqe_max') if item.get('eqe_max') is not None else 'not reported'}"
                )
            lines.append(f"Agent error: {error}")
            return "\n".join(lines)
        return (
            "The deterministic application tools are available, but the conversational model "
            f"could not complete this request. Agent error: {error}"
        )

    @staticmethod
    def _welcome_message(agent_type: str) -> str:
        return {
            "data_mining": (
                "Upload one or more PDF papers, then ask me to mine OLED device data. I will show "
                "each tool stage, group results by device, and keep all edits auditable."
            ),
            "device_modeling": (
                "I can inspect the OLED corpus and model registry, help design OLED-GAT code, and "
                "prepare or explicitly submit training jobs to the configured Slurm cluster."
            ),
            "experimental_design": (
                "Describe an OLED target, material system, or device bottleneck. I will retrieve "
                "precedents from the device database and propose evidence-grounded experiments."
            ),
        }[agent_type]


def _contains(content: str, terms: tuple[str, ...]) -> bool:
    lowered = content.lower()
    return any(term in lowered for term in terms)


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        key = str(record.get("id") or f"{record.get('doi')}::{record.get('device_label')}::{index}")
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result
