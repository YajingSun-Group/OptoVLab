from __future__ import annotations

import json
import re
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from evolab_local.optovlab.config import ModelingConfig, REPOSITORY_ROOT
from evolab_local.optovlab.schemas import HPCStatus, TrainingJob, TrainingRequest


class HPCService:
    def __init__(self, config: ModelingConfig, runtime_root: Path) -> None:
        self.config = config
        self.runtime_root = runtime_root

    def status(self) -> HPCStatus:
        errors: list[str] = []
        partitions = self._rows(
            ["sinfo", "-h", "-o", "%P|%a|%l|%D|%G"],
            ("partition", "availability", "time_limit", "nodes", "gres"),
            errors,
        )
        jobs = self._rows(
            ["squeue", "-h", "-o", "%i|%P|%j|%u|%T|%M|%R"],
            ("job_id", "partition", "name", "user", "state", "elapsed", "reason_or_node"),
            errors,
        )
        gpus = self._rows(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            ("index", "name", "memory_total_mb", "memory_free_mb", "utilization_percent"),
            errors,
            delimiter=",",
        )
        for gpu in gpus:
            for key in ("index", "memory_total_mb", "memory_free_mb", "utilization_percent"):
                try:
                    gpu[key] = int(str(gpu[key]).strip())
                except (TypeError, ValueError):
                    pass
        return HPCStatus(
            scheduler_available=not any("sinfo" in error for error in errors),
            partitions=partitions,
            jobs=jobs,
            gpus=gpus,
            errors=errors,
        )

    def model_registry(self) -> list[dict[str, Any]]:
        project = self.config.project_dir
        models: list[dict[str, Any]] = []
        for config_path in sorted((project / "configs").glob("*.yaml")):
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            output_dir = REPOSITORY_ROOT / str(payload.get("output_dir", ""))
            metrics: list[dict[str, Any]] = []
            if output_dir.exists():
                for metric_path in sorted(output_dir.glob("**/*metrics.json")):
                    try:
                        metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    metrics.append(
                        {
                            "path": str(metric_path.relative_to(REPOSITORY_ROOT)),
                            "metrics": metric_payload,
                        }
                    )
            model_section = payload.get("model") or {}
            split_section = payload.get("split") or {}
            models.append(
                {
                    "id": config_path.stem,
                    "name": config_path.stem.replace("_", " ").title(),
                    "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
                    "architecture": model_section.get("architecture"),
                    "split_mode": split_section.get("mode"),
                    "quantiles": model_section.get("quantiles") or [],
                    "output_dir": str(output_dir.relative_to(REPOSITORY_ROOT))
                    if output_dir.is_relative_to(REPOSITORY_ROOT)
                    else str(output_dir),
                    "metrics": metrics[-5:],
                }
            )
        return models

    def prepare_training(self, request: TrainingRequest) -> TrainingJob:
        partition = request.partition or self.config.default_partition
        if partition not in self.config.allowed_partitions:
            raise ValueError(f"Unsupported Slurm partition: {partition}")
        config_path = self._resolve_config(request.config_path)
        time_limit = request.time_limit or self.config.default_time_limit
        if not re.fullmatch(r"(?:\d+-)?\d{1,2}:\d{2}:\d{2}", time_limit):
            raise ValueError("time_limit must use HH:MM:SS or D-HH:MM:SS")
        run_id = uuid.uuid4().hex
        run_dir = self.runtime_root / "training" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "uv",
            "run",
            "python",
            "analysis/oled_gat/run_train.py",
            "--config",
            str(config_path),
            "--run-name",
            request.run_name,
        ]
        if request.seed is not None:
            command.extend(["--seed", str(request.seed)])
        if request.evaluate_test:
            command.append("--evaluate-test")
        script_path = run_dir / "train.sbatch"
        script = "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH --job-name={request.run_name}",
                f"#SBATCH --partition={partition}",
                "#SBATCH --nodes=1",
                f"#SBATCH --gres=gpu:{request.gpus}",
                f"#SBATCH --time={time_limit}",
                f"#SBATCH --output={run_dir}/slurm-%j.out",
                "#SBATCH --error=" + str(run_dir / "slurm-%j.err"),
                "set -euo pipefail",
                f"cd {shlex.quote(str(REPOSITORY_ROOT))}",
                shlex.join(command),
                "",
            ]
        )
        script_path.write_text(script, encoding="utf-8")
        scheduler_job_id: str | None = None
        status = "prepared"
        if request.confirm_submit:
            completed = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            match = re.search(r"Submitted batch job\s+(\d+)", completed.stdout)
            if not match:
                raise RuntimeError(f"Unable to parse sbatch response: {completed.stdout.strip()}")
            scheduler_job_id = match.group(1)
            status = "submitted"
        return TrainingJob(
            job_id=run_id,
            status=status,
            submitted=request.confirm_submit,
            scheduler_job_id=scheduler_job_id,
            script_path=str(script_path),
            command=command,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    def _resolve_config(self, supplied: str | None) -> Path:
        path = Path(supplied) if supplied else self.config.default_config
        if not path.is_absolute():
            path = (REPOSITORY_ROOT / path).resolve()
        allowed_root = (self.config.project_dir / "configs").resolve()
        if not path.is_relative_to(allowed_root) or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Training config must be an OLED-GAT YAML file under analysis/oled_gat/configs")
        if not path.exists():
            raise FileNotFoundError(f"Training config not found: {path}")
        return path

    @staticmethod
    def _rows(
        command: list[str],
        fields: tuple[str, ...],
        errors: list[str],
        delimiter: str = "|",
    ) -> list[dict[str, Any]]:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{command[0]}: {exc}")
            return []
        rows: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(delimiter)]
            if len(values) == len(fields):
                rows.append(dict(zip(fields, values)))
        return rows
