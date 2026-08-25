from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_triage_service import (
    MaterialStructureTriageService,
)
from evolab_local.mining_platform.schemas.batch import (
    MaterialStage3Plan,
    MaterialStage3PlanItem,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialResolutionTask,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import MaterialResolutionTaskRepository


class MaterialStage3PlannerService:
    """Build and persist the cheap, material-level Stage 3 execution plan."""

    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        material_resolution_service: MaterialResolutionService | None = None,
        triage_service: MaterialStructureTriageService | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.material_resolution = material_resolution_service or MaterialResolutionService(config)
        self.triage = triage_service or MaterialStructureTriageService(config)
        self.tasks = MaterialResolutionTaskRepository(self.database)

    def init_runtime(self) -> None:
        initializer = getattr(self.material_resolution, "init_runtime", None)
        if callable(initializer):
            initializer()

    def plan_papers(
        self,
        paper_ids: list[str],
        *,
        refresh_local: bool = True,
        max_workers: int = 1,
    ) -> MaterialStage3Plan:
        self.init_runtime()
        unique_paper_ids = list(dict.fromkeys(paper_ids))
        worker_count = max(1, min(max_workers, len(unique_paper_ids)))
        if worker_count == 1:
            paper_plans = [
                self._plan_paper(paper_id, refresh_local=refresh_local)
                for paper_id in unique_paper_ids
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                paper_plans = list(
                    executor.map(
                        lambda paper_id: self._plan_paper(
                            paper_id,
                            refresh_local=refresh_local,
                        ),
                        unique_paper_ids,
                    )
                )
        plan = MaterialStage3Plan(paper_count=len(unique_paper_ids))
        for paper_plan in paper_plans:
            plan.skipped_paper_count += paper_plan.skipped_paper_count
            plan.material_count += paper_plan.material_count
            plan.core_material_count += paper_plan.core_material_count
            plan.terminal_scope_count += paper_plan.terminal_scope_count
            plan.local_resolved_count += paper_plan.local_resolved_count
            plan.public_pending_count += paper_plan.public_pending_count
            plan.items.extend(paper_plan.items)
        return plan

    def _plan_paper(
        self,
        paper_id: str,
        *,
        refresh_local: bool,
    ) -> MaterialStage3Plan:
        plan = MaterialStage3Plan(paper_count=1)
        bundle = (
            self.material_resolution.resolve_paper_materials(paper_id) if refresh_local else None
        )
        if isinstance(bundle, PaperMaterialStructureBundle) and hasattr(
            self.triage,
            "triage_bundle",
        ):
            triage = self.triage.triage_bundle(bundle)
        else:
            triage = self.triage.triage_paper(paper_id)
        if triage is None or not triage.candidate_run_id:
            plan.skipped_paper_count = 1
            return plan
        for item in triage.items:
            if item.category in {"identity_only", "out_of_scope_structure"}:
                route = "terminal_scope"
                plan.terminal_scope_count += 1
            elif item.has_accepted_or_matched_structure:
                route = "local_or_accepted"
                plan.local_resolved_count += 1
            else:
                route = "public_resolution"
                plan.public_pending_count += 1
            if item.category == "core_structure_required":
                plan.core_material_count += 1
            plan.material_count += 1
            plan_item = MaterialStage3PlanItem(
                paper_id=triage.paper_id,
                candidate_run_id=triage.candidate_run_id,
                paper_material_id=item.paper_material_id,
                material_label=item.material_label,
                category=item.category,
                route=route,
                link_status=item.link_status,
                reason=item.reason,
            )
            plan.items.append(plan_item)
            if route == "public_resolution":
                self.update_task_stage(
                    plan_item,
                    stage="planned",
                    next_action="resolve_public",
                )
            elif route == "terminal_scope":
                self.update_task_stage(
                    plan_item,
                    stage="completed",
                    next_action="none",
                    status="completed",
                )
            elif route == "local_or_accepted":
                self.update_task_stage(
                    plan_item,
                    stage="completed",
                    next_action="none",
                    status="completed",
                    assigned_strategy="accepted_or_local_structure",
                )
        return plan

    def update_task_stage(
        self,
        item: MaterialStage3PlanItem,
        *,
        stage: str,
        next_action: str,
        status: str | None = None,
        assigned_strategy: str | None = None,
        duration_seconds: float | None = None,
        error_message: str | None = None,
        increment_retry: bool = False,
    ) -> MaterialResolutionTask:
        current = self.tasks.get_by_paper_material(
            item.candidate_run_id,
            item.paper_material_id,
        )
        timestamp = now_iso()
        task = current or MaterialResolutionTask(
            task_id=uuid4().hex,
            paper_id=item.paper_id,
            candidate_run_id=item.candidate_run_id,
            paper_material_id=item.paper_material_id,
            material_mentions=[item.material_label] if item.material_label else [],
            material_context={"structure_scope_category": item.category},
            created_at=timestamp,
            updated_at=timestamp,
        )
        timings = dict(task.stage_timings)
        errors = dict(task.stage_errors)
        if duration_seconds is not None:
            timings[stage] = round(duration_seconds, 3)
        if error_message:
            errors[stage] = error_message
        else:
            errors.pop(stage, None)
        return self.tasks.upsert(
            task.model_copy(
                update={
                    "status": status or task.status,
                    "assigned_strategy": assigned_strategy or task.assigned_strategy,
                    "current_stage": stage,
                    "next_action": next_action,
                    "retry_count": task.retry_count + (1 if increment_retry else 0),
                    "stage_timings": timings,
                    "stage_errors": errors,
                    "error_message": error_message,
                    "completed_at": (
                        task.completed_at or timestamp
                        if (status or task.status) == "completed"
                        else None
                    ),
                    "updated_at": timestamp,
                }
            )
        )

    def run_timed_task_stage(
        self,
        item: MaterialStage3PlanItem,
        *,
        stage: str,
        success_next_action: str,
        runner,
    ):
        started = perf_counter()
        self.update_task_stage(item, stage=stage, next_action=f"running_{stage}", status="running")
        try:
            result = runner()
        except Exception as exc:
            self.update_task_stage(
                item,
                stage=stage,
                next_action=f"retry_{stage}",
                status="failed",
                duration_seconds=perf_counter() - started,
                error_message=str(exc),
                increment_retry=True,
            )
            raise
        current = self.tasks.get_by_paper_material(
            item.candidate_run_id,
            item.paper_material_id,
        )
        if current and current.status == "failed":
            self.update_task_stage(
                item,
                stage=stage,
                next_action=f"retry_{stage}",
                status="failed",
                duration_seconds=perf_counter() - started,
                error_message=current.error_message or f"{stage} failed",
                increment_retry=True,
            )
            return result
        self.update_task_stage(
            item,
            stage=stage,
            next_action=success_next_action,
            status="pending",
            duration_seconds=perf_counter() - started,
        )
        return result
