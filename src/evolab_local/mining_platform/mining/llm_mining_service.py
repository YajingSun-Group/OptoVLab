from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.domain_template_service import DomainTemplateService
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMClient,
    OpenAICompatibleLLMClient,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.mining.llm_prompt_builder import (
    build_oled_mining_messages,
    select_prompt_sources,
    sources_from_document_blocks,
)
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService
from evolab_local.mining_platform.schemas.external_runs import LLMMiningResult, LLMMiningRun
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    DocumentBlockRepository,
    LLMMiningRunRepository,
    MinerUParseRunRepository,
    PaperRepository,
)


class LLMMiningService:
    def __init__(self, config: MiningPlatformConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.blocks = DocumentBlockRepository(self.database)
        self.mineru_runs = MinerUParseRunRepository(self.database)
        self.llm_runs = LLMMiningRunRepository(self.database)
        self.templates = DomainTemplateService(config)
        self.ingestion = CandidateIngestionService(config)
        self.llm_client = llm_client

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def mine_paper(
        self,
        paper_id: str,
        template_id: str = "oled_device_v1",
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMMiningResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None

        selected_provider = provider or self.config.llm.default_provider
        provider_config = self._provider_config(selected_provider)
        selected_model = model or provider_config.default_model
        template = self.templates.get_template(template_id)
        blocks = self.blocks.list_by_paper(normalized_paper_id)
        sources = select_prompt_sources(
            sources_from_document_blocks(blocks),
            max_chars=self.config.llm.max_source_chars,
        )
        source_parser = self._source_parser(normalized_paper_id)
        run = LLMMiningRun(
            llm_run_id=uuid4().hex,
            paper_id=normalized_paper_id,
            template_id=template.template_id,
            provider=selected_provider,
            model=selected_model,
            status="running",
            source_parser=source_parser,
            input_item_count=len(sources),
            created_at=now_iso(),
        )
        self.llm_runs.create(run)
        run_dir = self.config.paths.mining_runs_dir / run.llm_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not sources:
                raise ValueError("No parsed document blocks found. Run parse-paper-mineru first.")
            messages = build_oled_mining_messages(template=template, sources=sources)
            prompt_path = _write_json(run_dir / "prompt.json", {"messages": messages})
            client = self.llm_client or OpenAICompatibleLLMClient(provider_config)
            response = client.generate_json(messages, model=selected_model)
            raw_response_path = _write_text(run_dir / "raw_response.txt", response.content)
            if response.parsed_json is None:
                raise ValueError("LLM response did not contain a JSON object.")
            mining_payload = sanitize_llm_mining_result(response.parsed_json)
            mining_result_path = _write_json(run_dir / "mining_result.json", mining_payload)
            ingestion_result = self.ingestion.ingest_mining_result(
                paper_id=normalized_paper_id,
                template_id=template.template_id,
                payload=mining_payload,
                source_name=f"llm:{selected_provider}",
                source_version=selected_model,
            )
            if not ingestion_result:
                raise ValueError(
                    f"Paper not found during candidate ingestion: {normalized_paper_id}"
                )
            validation_report = ingestion_result.validation_report.model_dump(mode="json")
            validation_report_path = _write_json(
                run_dir / "validation_report.json", validation_report
            )
            status = "completed" if ingestion_result.validation_report.valid else "failed"
            error_message = None
            if not ingestion_result.validation_report.valid:
                error_message = _validation_error_summary(validation_report)
            completed = run.model_copy(
                update={
                    "status": status,
                    "prompt_path": prompt_path,
                    "raw_response_path": raw_response_path,
                    "mining_result_path": mining_result_path,
                    "validation_report_path": validation_report_path,
                    "candidate_run_id": ingestion_result.run.candidate_run_id,
                    "error_message": error_message,
                    "token_usage": response.usage,
                    "completed_at": now_iso(),
                }
            )
            self.llm_runs.update(completed)
            return LLMMiningResult(
                run=completed,
                mining_result=mining_payload,
                validation_report=validation_report,
                candidate_run_id=ingestion_result.run.candidate_run_id,
            )
        except Exception as exc:
            failed = run.model_copy(
                update={
                    "status": "failed",
                    "error_message": str(exc),
                    "completed_at": now_iso(),
                }
            )
            self.llm_runs.update(failed)
            self.papers.set_mining_status(normalized_paper_id, "failed")
            raise

    def mine_paper_pipeline(
        self,
        paper_id: str,
        template_id: str = "oled_device_v1",
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMMiningResult | None:
        mineru = MinerUParseService(self.config)
        mineru.parse_paper(paper_id)
        return self.mine_paper(paper_id, template_id=template_id, provider=provider, model=model)

    def _provider_config(self, provider: str):
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown LLM provider {provider!r}. Available: {known}")
        return provider_config

    def _source_parser(self, paper_id: str) -> str:
        mineru_run = self.mineru_runs.latest_completed_by_paper(paper_id)
        if mineru_run:
            return "mineru"
        return "document_blocks"


def sanitize_llm_mining_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply narrow, loss-aware repairs for common LLM shape mistakes.

    The raw response is still preserved in raw_response.txt. These repairs only keep
    candidate ingestion from failing on optional scalar object fields when the model
    emits a list for composite electrodes or multilayer shorthand.
    """
    repaired = json.loads(json.dumps(payload, ensure_ascii=False))
    _repair_evidence_refs(repaired)
    material_mentions_by_id = _material_mentions_by_id(repaired)
    for device in repaired.get("devices", []) if isinstance(repaired.get("devices"), list) else []:
        if not isinstance(device, dict):
            continue
        for layer_index, layer in enumerate(
            device.get("layers", []) if isinstance(device.get("layers"), list) else []
        ):
            if not isinstance(layer, dict):
                continue
            if _is_empty_scalar(layer.get("layer_index")):
                layer["layer_index"] = layer_index + 1
            if _is_empty_scalar(layer.get("layer_role")):
                layer["layer_role"] = "unknown"
            _repair_optional_scalar_object(layer, "thickness")
            _repair_optional_scalar_object(layer, "deposition_rate")
            _repair_layer_components(layer, material_mentions_by_id)
        final_emitter = device.get("final_emitter")
        if isinstance(final_emitter, dict):
            _repair_optional_scalar_object(final_emitter, "concentration")
        fabrication = device.get("fabrication")
        if isinstance(fabrication, dict):
            _repair_optional_scalar_object(fabrication, "device_area")
    return repaired


def _repair_evidence_refs(payload: dict[str, Any]) -> None:
    evidence_items = payload.get("evidence")
    if not isinstance(evidence_items, list):
        return
    evidence_ids: set[str] = set()
    evidence_id_by_block_id: dict[str, str] = {}
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        evidence_ids.add(evidence_id)
        block_id = item.get("block_id")
        if isinstance(block_id, str) and block_id:
            evidence_id_by_block_id[block_id] = evidence_id

    def repair_node(node: Any) -> None:
        if isinstance(node, dict):
            refs = node.get("evidence_refs")
            if isinstance(refs, list):
                repaired_refs: list[str] = []
                for ref in refs:
                    if not isinstance(ref, str) or not ref:
                        continue
                    if ref in evidence_ids:
                        candidate = ref
                    else:
                        candidate = evidence_id_by_block_id.get(ref)
                    if candidate and candidate not in repaired_refs:
                        repaired_refs.append(candidate)
                node["evidence_refs"] = repaired_refs
            elif "evidence_refs" in node and refs is not None:
                node["evidence_refs"] = []
            for value in node.values():
                repair_node(value)
        elif isinstance(node, list):
            for item in node:
                repair_node(item)

    repair_node(payload)


def _material_mentions_by_id(payload: dict[str, Any]) -> dict[str, str]:
    mentions: dict[str, str] = {}
    materials = payload.get("materials")
    if not isinstance(materials, list):
        return mentions
    preferred_keys = (
        "abbreviation",
        "normalized_name",
        "canonical_name",
        "full_name_in_paper",
        "paper_specific_label",
    )
    for material in materials:
        if not isinstance(material, dict):
            continue
        material_id = material.get("paper_material_id")
        if not isinstance(material_id, str) or not material_id:
            continue
        mention = None
        mention_list = material.get("mention_list")
        if isinstance(mention_list, list):
            mention = next((item for item in mention_list if isinstance(item, str) and item), None)
        if mention is None:
            for key in preferred_keys:
                value = material.get(key)
                if isinstance(value, str) and value:
                    mention = value
                    break
        if mention:
            mentions[material_id] = mention
    return mentions


def _repair_layer_components(layer: dict[str, Any], material_mentions_by_id: dict[str, str]) -> None:
    components = layer.get("components")
    if not isinstance(components, list):
        components = []
        layer["components"] = components
    if not components:
        layer_name = layer.get("layer_name")
        if isinstance(layer_name, str) and layer_name:
            layer["components"] = [
                {
                    "paper_material_id": None,
                    "material_mention": layer_name,
                    "component_role": _component_role_from_layer_role(layer.get("layer_role")),
                    "ratio": None,
                    "evidence_refs": layer.get("evidence_refs")
                    if isinstance(layer.get("evidence_refs"), list)
                    else [],
                }
            ]
        return
    for component in components:
        if not isinstance(component, dict):
            continue
        if not _is_empty_scalar(component.get("material_mention")):
            continue
        material_id = component.get("paper_material_id")
        if isinstance(material_id, str) and material_id in material_mentions_by_id:
            component["material_mention"] = material_mentions_by_id[material_id]
            continue
        layer_name = layer.get("layer_name")
        if isinstance(layer_name, str) and layer_name:
            component["material_mention"] = layer_name


def _component_role_from_layer_role(layer_role: object) -> str:
    if not isinstance(layer_role, str):
        return "unknown"
    if layer_role in {"anode", "cathode"}:
        return "electrode_material"
    if layer_role in {"HIL", "EIL"}:
        return "injection_material"
    if layer_role in {"HTL", "ETL", "mixed_transport_layer"}:
        return "transport_material"
    if layer_role in {"EBL", "HBL"}:
        return "blocking_material"
    if layer_role in {"CPL", "outcoupling_layer"}:
        return "optical_capping_material"
    return "neat_material"


def _is_empty_scalar(value: object) -> bool:
    return value is None or value == "" or value == []


def _repair_optional_scalar_object(parent: dict[str, Any], key: str) -> None:
    value = parent.get(key)
    if not isinstance(value, list):
        return
    object_values = [item for item in value if isinstance(item, dict)]
    if len(object_values) == 1:
        parent[key] = object_values[0]
    else:
        parent[key] = None


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return path.as_posix()


def _validation_error_summary(report: dict[str, Any]) -> str:
    errors = report.get("errors")
    if not isinstance(errors, list) or not errors:
        return "Mining result validation failed."
    first = errors[0]
    if isinstance(first, dict):
        return str(first.get("message") or first.get("code") or "Mining result validation failed.")
    return "Mining result validation failed."
