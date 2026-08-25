from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class RuntimeConfig(BaseModel):
    root: Path = Path("runtime/optovlab")
    sqlite_path: Path = Path("runtime/optovlab/optovlab.sqlite")
    artifact_dir: Path = Path("runtime/optovlab/artifacts")


class DatasetConfig(BaseModel):
    oled_devices: Path = Path("apps/database-web/public/data/oled.json.gz")


class ModelingConfig(BaseModel):
    project_dir: Path = Path("analysis/oled_gat")
    default_config: Path = Path("analysis/oled_gat/configs/campaign_gat.yaml")
    allowed_partitions: list[str] = Field(default_factory=lambda: ["rtx5880", "rtx3090"])
    default_partition: str = "rtx5880"
    default_gpus: int = 1
    default_time_limit: str = "08:00:00"


class AgentProviderConfig(BaseModel):
    provider: str = "deepseek"
    model_env: str = "DEEPSEEK_DEFAULT_MODEL"
    base_url_env: str = "DEEPSEEK_BASE_URL"
    api_key_env: str = "DEEPSEEK_API_KEY"
    request_timeout_seconds: int = 180

    @property
    def model(self) -> str:
        return os.getenv(self.model_env, "deepseek-v4-flash")

    @property
    def base_url(self) -> str:
        return os.getenv(self.base_url_env, "https://api.deepseek.com")

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class RetrievalConfig(BaseModel):
    maximum_features: int = 24000
    default_top_k: int = 8
    maximum_top_k: int = 25


class OptoVLabConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    agents: AgentProviderConfig = Field(default_factory=AgentProviderConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    def resolve_paths(self, root: Path = REPOSITORY_ROOT) -> "OptoVLabConfig":
        payload = self.model_dump()
        for section, keys in {
            "runtime": ("root", "sqlite_path", "artifact_dir"),
            "datasets": ("oled_devices",),
            "modeling": ("project_dir", "default_config"),
        }.items():
            for key in keys:
                value = Path(payload[section][key])
                payload[section][key] = value if value.is_absolute() else (root / value).resolve()
        return OptoVLabConfig.model_validate(payload)


def load_optovlab_config(path: Path | str = Path("config/optovlab.yaml")) -> OptoVLabConfig:
    load_dotenv(REPOSITORY_ROOT / ".env")
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return OptoVLabConfig.model_validate(raw).resolve_paths()
