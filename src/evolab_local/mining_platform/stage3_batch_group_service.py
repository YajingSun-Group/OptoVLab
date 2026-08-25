from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from evolab_local.mining_platform.batch_worker_service import BatchWorkerService
from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.qwen_batch_vision_client import (
    QwenBatchVisionClient,
)
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
)


class Stage3BatchGroupService:
    """Run many review batches while sharing one durable Qwen Batch VLM client."""

    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        worker_service: BatchWorkerService | None = None,
        vision_client: QwenBatchVisionClient | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.progress = progress or (lambda message: print(message, flush=True))
        self._progress_lock = Lock()
        self._state_lock = Lock()
        self._owns_vision_client = vision_client is None
        if vision_client is None:
            batch_config = config.llm.vision_batch
            provider_config = config.llm.providers.get(batch_config.provider)
            if provider_config is None:
                raise ValueError(
                    f"Unknown vision Batch provider {batch_config.provider!r}."
                )
            vision_client = QwenBatchVisionClient(
                provider_config,
                batch_config,
                runtime_dir=config.paths.runtime_dir / "qwen_batch_vlm",
                progress=self._emit,
            )
        self.vision_client = vision_client
        self.worker = worker_service or BatchWorkerService(
            config,
            material_agent_service=MaterialStructureAgentService(
                config,
                vision_client=vision_client,
            ),
        )

    def close(self) -> None:
        if self._owns_vision_client:
            self.vision_client.close()

    def run(
        self,
        *,
        start_batch: int,
        end_batch: int | None = None,
        batch_size: int | None = None,
        batch_concurrency: int = 15,
        material_concurrency: int = 2,
        run_material_resolution: bool = True,
        public_concurrency: int | None = None,
        judge_concurrency: int | None = None,
        vlm_concurrency: int | None = None,
        ocsr_paper_concurrency: int | None = None,
        decimer_segmentation_concurrency: int | None = None,
        decimer_ocsr_concurrency: int | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        if start_batch < 1:
            raise ValueError("start_batch must be >= 1.")
        if batch_concurrency < 1 or material_concurrency < 1:
            raise ValueError("Batch and material concurrency must be positive.")
        self.worker.init_runtime()
        overview = self.worker.list_review_batches(batch_size=batch_size)
        final_batch = end_batch or overview.total_batches
        if final_batch < start_batch or final_batch > overview.total_batches:
            raise ValueError(
                f"Batch range {start_batch}-{final_batch} is outside 1-{overview.total_batches}."
            )
        batch_numbers = list(range(start_batch, final_batch + 1))
        run_dir = (
            self.config.paths.runtime_dir
            / "stage3_group_runs"
            / f"qwen-batch-batches-{start_batch:04d}-{final_batch:04d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "state.json"
        report_json_path = run_dir / "report.json"
        report_markdown_path = run_dir / "report.md"
        state = self._load_state(state_path) if resume else {}
        completed_before = {
            int(number)
            for number, item in (state.get("batches") or {}).items()
            if isinstance(item, dict) and item.get("status") == "completed"
        }
        pending_batch_numbers = [
            number for number in batch_numbers if number not in completed_before
        ]
        initial_vlm_stats = self.vision_client.statistics()
        started_at = state.get("started_at") or _iso_now()
        state.update(
            {
                "mode": "qwen_batch_vlm",
                "start_batch": start_batch,
                "end_batch": final_batch,
                "batch_size": overview.batch_size,
                "batch_concurrency": batch_concurrency,
                "material_concurrency": material_concurrency,
                "run_material_resolution": run_material_resolution,
                "started_at": started_at,
                "updated_at": _iso_now(),
                "batches": state.get("batches") or {},
            }
        )
        self._write_json_atomic(state_path, state)
        self._emit(
            f"[stage3-group] batches={start_batch}-{final_batch}, total={len(batch_numbers)}, "
            f"resume_skipped={len(completed_before)}, pending={len(pending_batch_numbers)}, "
            f"concurrency={batch_concurrency}"
        )

        group_started = time.perf_counter()
        completed_this_run = 0
        failed_this_run = 0
        with ThreadPoolExecutor(
            max_workers=min(batch_concurrency, max(1, len(pending_batch_numbers))),
            thread_name_prefix="stage3-review-batch",
        ) as executor:
            futures = {
                executor.submit(
                    self._run_one_batch,
                    batch_number,
                    batch_size=overview.batch_size,
                    material_concurrency=material_concurrency,
                    run_material_resolution=run_material_resolution,
                    public_concurrency=public_concurrency,
                    judge_concurrency=judge_concurrency,
                    vlm_concurrency=vlm_concurrency,
                    ocsr_paper_concurrency=ocsr_paper_concurrency,
                    decimer_segmentation_concurrency=decimer_segmentation_concurrency,
                    decimer_ocsr_concurrency=decimer_ocsr_concurrency,
                ): batch_number
                for batch_number in pending_batch_numbers
            }
            for future in as_completed(futures):
                batch_number = futures[future]
                try:
                    batch_result = future.result()
                    completed_this_run += 1
                except Exception as exc:
                    failed_this_run += 1
                    batch_result = {
                        "batch_number": batch_number,
                        "status": "failed",
                        "error": str(exc),
                        "finished_at": _iso_now(),
                    }
                with self._state_lock:
                    state["batches"][str(batch_number)] = batch_result
                    state["updated_at"] = _iso_now()
                    state["completed_count"] = sum(
                        item.get("status") == "completed"
                        for item in state["batches"].values()
                        if isinstance(item, dict)
                    )
                    state["failed_count"] = sum(
                        item.get("status") == "failed"
                        for item in state["batches"].values()
                        if isinstance(item, dict)
                    )
                    self._write_json_atomic(state_path, state)
                done = completed_this_run + failed_this_run
                self._emit(
                    f"[stage3-group] progress={done}/{len(pending_batch_numbers)}; "
                    f"batch={batch_number}; status={batch_result['status']}; "
                    f"completed={completed_this_run}; failed={failed_this_run}"
                )

        final_vlm_stats = self.vision_client.statistics()
        state["finished_at"] = _iso_now()
        state["wall_seconds_this_run"] = round(time.perf_counter() - group_started, 3)
        state["qwen_batch_vlm_before"] = initial_vlm_stats
        state["qwen_batch_vlm_after"] = final_vlm_stats
        state["qwen_batch_vlm_delta"] = _statistics_delta(
            initial_vlm_stats,
            final_vlm_stats,
        )
        state["updated_at"] = _iso_now()
        self._write_json_atomic(state_path, state)
        report = self._build_report(state)
        self._write_json_atomic(report_json_path, report)
        report_markdown_path.write_text(
            _render_markdown_report(report),
            encoding="utf-8",
        )
        report["state_path"] = state_path.as_posix()
        report["json_path"] = report_json_path.as_posix()
        report["markdown_path"] = report_markdown_path.as_posix()
        return report

    def _run_one_batch(
        self,
        batch_number: int,
        *,
        batch_size: int,
        material_concurrency: int,
        run_material_resolution: bool,
        public_concurrency: int | None,
        judge_concurrency: int | None,
        vlm_concurrency: int | None,
        ocsr_paper_concurrency: int | None,
        decimer_segmentation_concurrency: int | None,
        decimer_ocsr_concurrency: int | None,
    ) -> dict[str, Any]:
        self._emit(f"[stage3-group] batch={batch_number} started")
        started = time.perf_counter()
        result = self.worker.refresh_review_batch_materials(
            batch_index=batch_number - 1,
            batch_size=batch_size,
            material_concurrency=material_concurrency,
            run_metadata_enrichment=True,
            run_material_resolution=run_material_resolution,
            run_public_resolver=True,
            run_identity_judge=True,
            run_material_auto_decision=True,
            run_visual_prep=True,
            run_material_ocsr=True,
            public_concurrency=public_concurrency,
            judge_concurrency=judge_concurrency,
            vlm_concurrency=vlm_concurrency,
            ocsr_paper_concurrency=ocsr_paper_concurrency,
            decimer_segmentation_concurrency=decimer_segmentation_concurrency,
            decimer_ocsr_concurrency=decimer_ocsr_concurrency,
        )
        report = self.worker.write_batch_report(
            batch_index=batch_number - 1,
            batch_size=batch_size,
            run_result=result,
            run_options={
                "runner": "stage3_batch_group",
                "vision_inference": "qwen_batch_file",
                "material_concurrency": material_concurrency,
                "material_resolution": run_material_resolution,
                "public_concurrency": public_concurrency,
                "judge_concurrency": judge_concurrency,
                "vlm_concurrency": vlm_concurrency,
                "ocsr_paper_concurrency": ocsr_paper_concurrency,
                "decimer_segmentation_concurrency": decimer_segmentation_concurrency,
                "decimer_ocsr_concurrency": decimer_ocsr_concurrency,
            },
        )
        duration = round(time.perf_counter() - started, 3)
        return {
            "batch_number": batch_number,
            "status": "completed",
            "duration_seconds": duration,
            "processed_count": result.processed_count,
            "review_ready_count": result.review_ready_count,
            "failed_paper_count": result.failed_count,
            "skipped_count": result.skipped_count,
            "phase_timings": result.phase_timings,
            "material_metrics": result.material_metrics,
            "report_json_path": report.json_path,
            "report_markdown_path": report.markdown_path,
            "finished_at": _iso_now(),
        }

    def _build_report(self, state: dict[str, Any]) -> dict[str, Any]:
        batch_items = [
            item for item in state.get("batches", {}).values() if isinstance(item, dict)
        ]
        completed = [item for item in batch_items if item.get("status") == "completed"]
        failed = [item for item in batch_items if item.get("status") == "failed"]
        return {
            "mode": state.get("mode"),
            "batch_range": [state.get("start_batch"), state.get("end_batch")],
            "batch_size": state.get("batch_size"),
            "batch_concurrency": state.get("batch_concurrency"),
            "material_concurrency": state.get("material_concurrency"),
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
            "wall_seconds_this_run": state.get("wall_seconds_this_run"),
            "completed_batch_count": len(completed),
            "failed_batch_count": len(failed),
            "processed_paper_count": sum(int(item.get("processed_count") or 0) for item in completed),
            "review_ready_paper_count": sum(
                int(item.get("review_ready_count") or 0) for item in completed
            ),
            "failed_paper_count": sum(
                int(item.get("failed_paper_count") or 0) for item in completed
            ),
            "skipped_paper_count": sum(int(item.get("skipped_count") or 0) for item in completed),
            "failed_batches": [
                {
                    "batch_number": item.get("batch_number"),
                    "error": item.get("error"),
                }
                for item in failed
            ],
            "qwen_batch_vlm_delta": state.get("qwen_batch_vlm_delta") or {},
            "batches": sorted(batch_items, key=lambda item: int(item.get("batch_number") or 0)),
        }

    def _emit(self, message: str) -> None:
        with self._progress_lock:
            self.progress(message)

    @staticmethod
    def _load_state(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _statistics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def map_delta(key: str) -> dict[str, int]:
        previous = before.get(key) if isinstance(before.get(key), dict) else {}
        current = after.get(key) if isinstance(after.get(key), dict) else {}
        return {
            name: int(current.get(name, 0)) - int(previous.get(name, 0))
            for name in sorted(set(previous) | set(current))
        }

    before_usage = before.get("usage") if isinstance(before.get("usage"), dict) else {}
    after_usage = after.get("usage") if isinstance(after.get("usage"), dict) else {}
    return {
        "jobs_by_status": map_delta("jobs_by_status"),
        "requests_by_status": map_delta("requests_by_status"),
        "realtime_fallback_count": int(after.get("realtime_fallback_count", 0))
        - int(before.get("realtime_fallback_count", 0)),
        "usage": {
            key: int(after_usage.get(key, 0)) - int(before_usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    start_batch, end_batch = report["batch_range"]
    usage = report.get("qwen_batch_vlm_delta", {}).get("usage", {})
    requests = report.get("qwen_batch_vlm_delta", {}).get("requests_by_status", {})
    lines = [
        f"# Stage 3 Qwen Batch Report: Batch {start_batch}-{end_batch}",
        "",
        f"- Started: {report.get('started_at')}",
        f"- Finished: {report.get('finished_at')}",
        f"- Wall time: {report.get('wall_seconds_this_run')} seconds",
        f"- Completed batches: {report.get('completed_batch_count')}",
        f"- Failed batches: {report.get('failed_batch_count')}",
        f"- Processed papers: {report.get('processed_paper_count')}",
        f"- Review-ready papers: {report.get('review_ready_paper_count')}",
        f"- Skipped papers: {report.get('skipped_paper_count')}",
        "",
        "## Qwen Batch VLM",
        "",
        f"- Completed discounted requests: {requests.get('completed', 0)}",
        f"- Failed requests: {requests.get('failed', 0)}",
        f"- Realtime oversized fallbacks: {report.get('qwen_batch_vlm_delta', {}).get('realtime_fallback_count', 0)}",
        f"- Prompt tokens: {usage.get('prompt_tokens', 0)}",
        f"- Completion tokens: {usage.get('completion_tokens', 0)}",
        f"- Total tokens: {usage.get('total_tokens', 0)}",
        "",
        "Batch File requests are billed at 50% of the corresponding realtime model price.",
    ]
    failed = report.get("failed_batches") or []
    if failed:
        lines.extend(["", "## Failed Batches", ""])
        lines.extend(
            f"- Batch {item.get('batch_number')}: {item.get('error')}" for item in failed
        )
    return "\n".join(lines) + "\n"


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
