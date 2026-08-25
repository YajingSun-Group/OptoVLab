from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from evolab_local.mining_platform.core.paths import project_root_for_config, resolve_path


class PathConfig(BaseModel):
    runtime_dir: Path
    sqlite_path: Path
    paper_registry_path: Path
    parsed_dir: Path
    logs_dir: Path
    mineru_runs_dir: Path
    mining_runs_dir: Path
    inbox_pdfs_dir: Path
    batch_reports_dir: Path


class PdfDownloaderSourceConfig(BaseModel):
    manifest_path: Path
    sqlite_path: Path
    papers_dir: Path


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class LoggingConfig(BaseModel):
    level: str = "INFO"


class BatchWorkerConfig(BaseModel):
    scan_interval_seconds: float = 30.0
    stable_file_seconds: float = 5.0
    max_retries: int = 2
    default_domain: str = "oled"
    run_metadata_enrichment: bool = True
    run_public_resolver: bool = True
    run_identity_judge: bool = True
    run_material_auto_decision: bool = True
    material_auto_accept_min_confidence: float = 0.9
    material_auto_reject_min_confidence: float = 0.9
    material_auto_accept_decimer_ocsr: bool = False
    run_visual_prep: bool = False
    run_material_ocsr: bool = False
    material_ocsr_auto_match_min_confidence: float = 0.8
    material_public_concurrency: int = 16
    material_judge_concurrency: int = 20
    material_vlm_concurrency: int = 12
    material_ocsr_paper_concurrency: int = 2
    # Planning performs many short SQLite writes. One planner per review batch
    # avoids write-lock contention while the later network/VLM stages remain
    # independently concurrent.
    material_plan_concurrency: int = 1
    material_visual_prefetch_concurrency: int = 10
    material_decimer_segmentation_concurrency: int = 2
    material_decimer_ocsr_concurrency: int = 4
    review_batch_size: int = 10


class FeatureConfig(BaseModel):
    # Material property mining is intentionally paused in the current runtime version.
    material_properties: bool = False


class MinerUConfig(BaseModel):
    base_url: str = "http://127.0.0.1:18000"
    backend: str = "hybrid-auto-engine"
    parse_method: str = "auto"
    lang_list: str = "en"
    formula_enable: bool = True
    table_enable: bool = True
    return_md: bool = True
    return_content_list: bool = True
    return_middle_json: bool = False
    return_images: bool = False
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 600.0


class PubChemConfig(BaseModel):
    base_url: str = "https://pubchem.ncbi.nlm.nih.gov"
    timeout_seconds: float = 20.0
    max_results_per_query: int = 3
    max_synonyms: int = 20


class AnySearchConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.anysearch.com/v1/search"
    timeout_seconds: float = 30.0
    max_results_per_query: int = 5
    max_query_variants_per_name: int = 3


class OpsinConfig(BaseModel):
    base_url: str = "https://www.ebi.ac.uk/opsin/ws"
    timeout_seconds: float = 30.0


class PaperMetadataConfig(BaseModel):
    openalex_base_url: str = "https://api.openalex.org"
    crossref_base_url: str = "https://api.crossref.org"
    openalex_api_key: str = ""
    mailto: str = ""
    timeout_seconds: float = 20.0


class DecimerSegmentationConfig(BaseModel):
    base_url: str = "http://127.0.0.1:18100"
    timeout_seconds: float = 120.0
    max_segments: int = 20
    expand: bool = True


class DecimerSmilesConfig(BaseModel):
    base_url: str = "http://127.0.0.1:18101"
    timeout_seconds: float = 120.0
    confidence: bool = True
    hand_drawn: bool = False


class ExternalServicesConfig(BaseModel):
    mineru: MinerUConfig = MinerUConfig()
    paper_metadata: PaperMetadataConfig = PaperMetadataConfig()
    pubchem: PubChemConfig = PubChemConfig()
    anysearch: AnySearchConfig = AnySearchConfig()
    opsin: OpsinConfig = OpsinConfig()
    decimer_segmentation: DecimerSegmentationConfig = DecimerSegmentationConfig()
    decimer_smiles: DecimerSmilesConfig = DecimerSmilesConfig()


class LLMProviderConfig(BaseModel):
    api_key: str = ""
    base_url: str
    default_model: str
    vision_model: str | None = None
    vision_enable_thinking: bool | None = None
    thinking_mode: Literal["enabled", "disabled"] | None = None
    timeout_seconds: float = 120.0
    request_max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    temperature: float = 0.0
    response_format_json: bool = True


class VisionBatchConfig(BaseModel):
    provider: str = "qwen"
    completion_window: str = "24h"
    flush_seconds: float = 5.0
    poll_interval_seconds: float = 2.0
    max_requests_per_job: int = 2000
    max_file_bytes: int = 350_000_000
    max_line_bytes: int = 5_800_000
    max_active_jobs: int = 32
    image_max_dimension: int = 2048
    image_jpeg_quality: int = 90
    realtime_fallback: bool = True


class LLMConfig(BaseModel):
    default_provider: str = "deepseek"
    max_source_chars: int = 70000
    vision_batch: VisionBatchConfig = VisionBatchConfig()
    providers: dict[str, LLMProviderConfig]


class MiningPlatformConfig(BaseModel):
    project_root: Path
    paths: PathConfig
    pdf_downloader: PdfDownloaderSourceConfig
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()
    batch_worker: BatchWorkerConfig = BatchWorkerConfig()
    features: FeatureConfig = FeatureConfig()
    external_services: ExternalServicesConfig = ExternalServicesConfig()
    llm: LLMConfig = LLMConfig(
        providers={
            "deepseek": LLMProviderConfig(
                api_key="",
                base_url="https://api.deepseek.com",
                default_model="deepseek-v4-flash",
            ),
            "qwen": LLMProviderConfig(
                api_key="",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                default_model="qwen3.6-plus",
            ),
        }
    )

    def ensure_dirs(self) -> None:
        self.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.paths.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.paper_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.mineru_runs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.mining_runs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.inbox_pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.batch_reports_dir.mkdir(parents=True, exist_ok=True)


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), match.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _resolve_config_paths(data: dict[str, Any], project_root: Path) -> dict[str, Any]:
    resolved = dict(data)
    path_defaults = {
        "mineru_runs_dir": "./runtime/mining_platform/mineru_runs",
        "mining_runs_dir": "./runtime/mining_platform/mining_runs",
        "inbox_pdfs_dir": "./runtime/mining_platform/inbox_pdfs",
        "batch_reports_dir": "./runtime/mining_platform/batch_reports",
    }
    paths_data = dict(resolved.get("paths") or {})
    for key, value in path_defaults.items():
        paths_data.setdefault(key, value)
    resolved["paths"] = paths_data
    for section in ("paths", "pdf_downloader"):
        section_data = dict(resolved.get(section) or {})
        for key, value in section_data.items():
            if key.endswith("_dir") or key.endswith("_path"):
                section_data[key] = resolve_path(value, project_root)
        resolved[section] = section_data
    resolved["project_root"] = project_root
    return resolved


def load_config(
    path: Path = Path("config/mining_platform/mining_platform.yaml"),
) -> MiningPlatformConfig:
    load_dotenv()
    project_root = project_root_for_config(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    expanded = _expand_env(data)
    resolved = _resolve_config_paths(expanded, project_root)
    config = MiningPlatformConfig.model_validate(resolved)
    config.ensure_dirs()
    return config
