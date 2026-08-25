from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.schemas.batch import BatchWorkerRunResult
from evolab_local.mining_platform.stage3_batch_group_service import Stage3BatchGroupService


class FakeVisionClient:
    def statistics(self):
        return {
            "jobs_by_status": {"completed": 1},
            "requests_by_status": {"completed": 3},
            "realtime_fallback_count": 0,
            "usage": {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
        }

    def close(self) -> None:
        return None


class FakeWorker:
    def __init__(self) -> None:
        self.batch_numbers: list[int] = []
        self.material_resolution_flags: list[bool] = []

    def init_runtime(self) -> None:
        return None

    def list_review_batches(self, *, batch_size=None):
        return SimpleNamespace(total_batches=3, batch_size=batch_size or 10)

    def refresh_review_batch_materials(
        self,
        *,
        batch_index: int,
        run_material_resolution: bool,
        **_kwargs,
    ):
        self.batch_numbers.append(batch_index + 1)
        self.material_resolution_flags.append(run_material_resolution)
        return BatchWorkerRunResult(
            processed_count=10,
            review_ready_count=10,
            phase_timings={"stage3_material_refresh": 1.0},
            material_metrics={"material_count": 20},
        )

    def write_batch_report(self, *, batch_index: int, **_kwargs):
        return SimpleNamespace(
            json_path=f"batch-{batch_index + 1}.json",
            markdown_path=f"batch-{batch_index + 1}.md",
        )


def test_group_runner_persists_report_and_resume_state(tmp_path: Path) -> None:
    config = load_config(Path("config/mining_platform/mining_platform.yaml"))
    config.paths.runtime_dir = tmp_path
    worker = FakeWorker()
    service = Stage3BatchGroupService(
        config,
        worker_service=worker,
        vision_client=FakeVisionClient(),  # type: ignore[arg-type]
        progress=lambda _message: None,
    )

    report = service.run(
        start_batch=1,
        end_batch=2,
        batch_concurrency=2,
        material_concurrency=1,
        run_material_resolution=False,
    )
    repeated = service.run(
        start_batch=1,
        end_batch=2,
        batch_concurrency=2,
        material_concurrency=1,
    )

    assert sorted(worker.batch_numbers) == [1, 2]
    assert worker.material_resolution_flags == [False, False]
    assert report["completed_batch_count"] == 2
    assert report["processed_paper_count"] == 20
    assert Path(report["state_path"]).exists()
    assert Path(report["json_path"]).exists()
    assert Path(report["markdown_path"]).exists()
    assert repeated["completed_batch_count"] == 2
