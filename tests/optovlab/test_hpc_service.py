from __future__ import annotations

from pathlib import Path

from evolab_local.optovlab.config import load_optovlab_config
from evolab_local.optovlab.hpc_service import HPCService
from evolab_local.optovlab.schemas import TrainingRequest


def test_prepare_training_creates_reviewable_script_without_submission(tmp_path: Path) -> None:
    config = load_optovlab_config()
    service = HPCService(config.modeling, tmp_path)

    job = service.prepare_training(
        TrainingRequest(
            run_name="smoke-gat",
            partition="rtx5880",
            gpus=1,
            time_limit="00:30:00",
            confirm_submit=False,
        )
    )

    script = Path(job.script_path).read_text(encoding="utf-8")
    assert job.status == "prepared"
    assert job.submitted is False
    assert "#SBATCH --partition=rtx5880" in script
    assert "analysis/oled_gat/run_train.py" in script


def test_model_registry_discovers_versioned_configs() -> None:
    config = load_optovlab_config()
    models = HPCService(config.modeling, config.runtime.root).model_registry()

    assert {model["id"] for model in models} >= {"campaign_gat", "device_random"}
