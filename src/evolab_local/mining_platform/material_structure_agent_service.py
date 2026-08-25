from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw

from evolab_local.mining_platform.external.decimer_client import (
    DecimerSegment,
    DecimerSegmentationClient,
    DecimerSegmentationClientProtocol,
    DecimerSmilesClient,
    DecimerSmilesClientProtocol,
)
from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.openai_compatible_client import (
    OpenAICompatibleVisionClient,
    VisionClient,
    image_path_to_data_url,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_chemistry import standardize_smiles
from evolab_local.mining_platform.mining.mineru_parse_service import (
    MinerUParseService,
    mineru_item_text,
)
from evolab_local.mining_platform.schemas.candidate_ingestion import CandidateIngestionRun
from evolab_local.mining_platform.schemas.external_runs import MinerUParseRun
from evolab_local.mining_platform.schemas.material_agent import (
    DecimerSegmentationBatchResult,
    DocumentVisualBlock,
    FigureTriageBatchResult,
    FigureTriageResult,
    MaterialAgentFoundationResult,
    MaterialAgentRun,
    MaterialOCSRPipelineResult,
    MoleculeCrop,
    MoleculeCropValidation,
    MoleculeCropValidationBatchResult,
    MoleculeLabelBinding,
    MoleculeLabelBindingBatchResult,
    MoleculeLabelBindingReviewAction,
    MoleculeLabelBindingReviewEvent,
    VLMCallLog,
)
from evolab_local.mining_platform.schemas.material_structure import (
    DecimerOCSRBatchResult,
    MaterialStructureCandidate,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    CandidateIngestionRepository,
    DocumentVisualBlockRepository,
    FigureTriageResultRepository,
    MaterialAgentRunRepository,
    MinerUParseRunRepository,
    MoleculeCropRepository,
    MoleculeCropValidationRepository,
    MoleculeLabelBindingRepository,
    MoleculeLabelBindingReviewEventRepository,
    PaperRepository,
    MaterialStructureCandidateRepository,
    VLMCallLogRepository,
)


VISUAL_BLOCK_TYPES = {"image", "chart", "table"}
MINERU_OUTPUT_ROOT = Path(
    os.getenv("MINERU_OUTPUT_ROOT", "runtime/mining_platform/mineru_external")
)
FIGURE_TRIAGE_PROMPT_VERSION = "figure_triage_v2_2d_device_materials"
CROP_VALIDATION_PROMPT_VERSION = "crop_validation_v3_flat_2d_structure"
LABEL_BINDING_PROMPT_VERSION = "label_binding_v3_complete_named_material"
LABEL_BINDING_GROUP_MAX_CROPS = 6

_PARTIAL_STRUCTURE_REASON_PATTERN = re.compile(
    r"\b(?:r[\s-]?group|substituent(?:\s+(?:group|fragment))?|molecular\s+fragment|"
    r"partial\s+(?:structure|molecule|scaffold)|shared\s+scaffold|scaffold\s+only|"
    r"core\s+only|moiety\s+only)\b",
    flags=re.IGNORECASE,
)


class MaterialStructureAgentService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        vision_client: VisionClient | None = None,
        decimer_segmentation_client: DecimerSegmentationClientProtocol | None = None,
        decimer_smiles_client: DecimerSmilesClientProtocol | None = None,
    ) -> None:
        self.config = config
        self.vision_client = vision_client
        self.decimer_segmentation_client = decimer_segmentation_client
        self.decimer_smiles_client = decimer_smiles_client
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.candidate_runs = CandidateIngestionRepository(self.database)
        self.mineru_runs = MinerUParseRunRepository(self.database)
        self.agent_runs = MaterialAgentRunRepository(self.database)
        self.visual_blocks = DocumentVisualBlockRepository(self.database)
        self.figure_triage = FigureTriageResultRepository(self.database)
        self.molecule_crops = MoleculeCropRepository(self.database)
        self.crop_validations = MoleculeCropValidationRepository(self.database)
        self.label_bindings = MoleculeLabelBindingRepository(self.database)
        self.label_binding_events = MoleculeLabelBindingReviewEventRepository(self.database)
        self.structure_candidates = MaterialStructureCandidateRepository(self.database)
        self.vlm_calls = VLMCallLogRepository(self.database)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def run_foundation(self, paper_id: str) -> MaterialAgentFoundationResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        candidate_run = self._latest_completed_candidate_run(normalized_paper_id)
        mineru_run = self.mineru_runs.latest_completed_by_paper(normalized_paper_id)
        timestamp = now_iso()
        agent_run = MaterialAgentRun(
            agent_run_id=uuid4().hex,
            paper_id=normalized_paper_id,
            status="running",
            strategy="foundation",
            source_candidate_run_id=candidate_run.candidate_run_id if candidate_run else None,
            mineru_run_id=mineru_run.mineru_run_id if mineru_run else None,
            material_count=self._material_count(candidate_run),
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.agent_runs.create(agent_run)
        if not mineru_run or not mineru_run.content_list_path:
            failed = agent_run.model_copy(
                update={
                    "status": "failed",
                    "error_message": "No completed MinerU run with content_list_path.",
                    "updated_at": now_iso(),
                    "completed_at": now_iso(),
                }
            )
            self.agent_runs.update(failed)
            return MaterialAgentFoundationResult(run=failed, visual_blocks=[])

        try:
            content_list = read_mineru_content_list(Path(mineru_run.content_list_path))
            result_key = mineru_result_key(mineru_run)
            previous_blocks = self.visual_blocks.list_by_mineru_run(mineru_run.mineru_run_id)
            blocks = collect_document_visual_blocks(
                run=mineru_run,
                content_list=content_list,
                agent_run_id=agent_run.agent_run_id,
                result_key=result_key,
            )
            blocks = materialize_visual_block_images(
                blocks,
                output_dir=(
                    self.config.paths.runtime_dir
                    / "material_agent"
                    / "visual_blocks"
                    / agent_run.agent_run_id
                ),
                fallback_blocks=previous_blocks,
            )
            stored_blocks = self.visual_blocks.replace_for_mineru_run(
                mineru_run.mineru_run_id,
                blocks,
            )
            completed = agent_run.model_copy(
                update={
                    "status": "completed",
                    "visual_block_count": len(stored_blocks),
                    "tool_summary": {
                        "tools": [
                            "mineru_content_list_reader",
                            "document_visual_block_collector",
                            "document_visual_image_materializer",
                        ],
                        "content_item_count": len(content_list),
                        "result_key": result_key,
                        "visual_block_types": _count_by_type(stored_blocks),
                        "image_exists_count": sum(
                            1 for block in stored_blocks if block.image_exists
                        ),
                    },
                    "updated_at": now_iso(),
                    "completed_at": now_iso(),
                }
            )
            self.agent_runs.update(completed)
            return MaterialAgentFoundationResult(run=completed, visual_blocks=stored_blocks)
        except Exception as exc:
            failed = agent_run.model_copy(
                update={
                    "status": "failed",
                    "error_message": str(exc),
                    "updated_at": now_iso(),
                    "completed_at": now_iso(),
                }
            )
            self.agent_runs.update(failed)
            raise

    def list_runs(self, paper_id: str) -> list[MaterialAgentRun] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        return self.agent_runs.list_by_paper(normalized_paper_id)

    def list_visual_blocks(self, paper_id: str) -> list[DocumentVisualBlock] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            return []
        return [
            block
            for block in self.visual_blocks.list_by_paper(normalized_paper_id)
            if block.collected_by_agent_run_id == agent_run.agent_run_id
        ]

    def run_figure_triage(
        self,
        paper_id: str,
        *,
        provider: str = "qwen",
        model: str | None = None,
        limit: int | None = None,
        max_concurrency: int = 1,
        target_paper_material_ids: set[str] | None = None,
        force: bool = False,
    ) -> FigureTriageBatchResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            foundation = self.run_foundation(normalized_paper_id)
            if not foundation or foundation.run.status != "completed":
                raise ValueError(
                    f"Material agent foundation is not ready for {normalized_paper_id}."
                )
            agent_run = foundation.run
        agent_run = self._ensure_visual_images(normalized_paper_id, agent_run)
        blocks = [
            block
            for block in self.visual_blocks.list_by_paper(normalized_paper_id)
            if block.collected_by_agent_run_id == agent_run.agent_run_id
            or not block.collected_by_agent_run_id
        ]
        blocks = [block for block in blocks if block.image_exists and block.resolved_img_path]
        if not blocks:
            raise ValueError(
                "No readable MinerU visual images are available after image backfill."
            )
        if limit is not None:
            blocks = blocks[: max(0, limit)]
        material_context = self._paper_material_context(
            agent_run.source_candidate_run_id,
            device_used_only=True,
            target_paper_material_ids=target_paper_material_ids,
        )
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown VLM provider {provider!r}. Available: {known}")
        selected_model = model or provider_config.vision_model or provider_config.default_model
        client = self.vision_client or OpenAICompatibleVisionClient(provider_config)
        expected_material_ids = {
            str(material.get("paper_material_id"))
            for material in material_context
            if material.get("paper_material_id")
        }
        cached_entity_ids = (
            set()
            if force
            else self._completed_vlm_entity_ids(
                agent_run.agent_run_id,
                stage="figure_triage",
                provider=provider,
                model=selected_model,
                prompt_version=FIGURE_TRIAGE_PROMPT_VERSION,
                expected_material_ids=expected_material_ids,
            )
        )
        cached_results = {
            result.visual_block_id: result
            for result in self.figure_triage.list_by_run(agent_run.agent_run_id)
            if result.status == "completed"
            and result.provider == provider
            and result.model == selected_model
            and result.visual_block_id in cached_entity_ids
        }
        pending_blocks = [block for block in blocks if block.visual_block_id not in cached_results]
        triage_results = _map_with_concurrency(
            pending_blocks,
            max_concurrency=max_concurrency,
            runner=lambda block: self._triage_block(
                block,
                agent_run=agent_run,
                client=client,
                provider=provider,
                model=selected_model,
                enable_thinking=provider_config.vision_enable_thinking,
                material_context=material_context,
            ),
        )
        stored_results = {
            result.visual_block_id: self.figure_triage.upsert(result) for result in triage_results
        }
        stored_results.update(cached_results)
        results = [
            stored_results[block.visual_block_id]
            for block in blocks
            if block.visual_block_id in stored_results
        ]
        return FigureTriageBatchResult(run=agent_run, results=results)

    def list_figure_triage_results(self, paper_id: str) -> list[FigureTriageResult] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            return []
        return [
            result
            for result in self.figure_triage.list_by_paper(normalized_paper_id)
            if result.agent_run_id == agent_run.agent_run_id
        ]

    def run_decimer_segmentation(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
        max_segments: int | None = None,
        max_concurrency: int = 1,
        force: bool = False,
    ) -> DecimerSegmentationBatchResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            raise ValueError(f"Material agent foundation is not ready for {normalized_paper_id}.")
        triage_results = [
            result
            for result in self.figure_triage.list_by_paper(normalized_paper_id)
            if result.agent_run_id == agent_run.agent_run_id
            and result.status == "completed"
            and result.should_run_decimer_segmentation
        ]
        if limit is not None:
            triage_results = triage_results[: max(0, limit)]
        block_by_id = {
            block.visual_block_id: block
            for block in self.visual_blocks.list_by_paper(normalized_paper_id)
        }
        client = self.decimer_segmentation_client or DecimerSegmentationClient(
            self.config.external_services.decimer_segmentation
        )
        segment_config = self.config.external_services.decimer_segmentation
        output_root = self.config.paths.runtime_dir / "material_agent" / "molecule_crops"
        existing_by_triage: dict[str, list[MoleculeCrop]] = {}
        for crop in self.molecule_crops.list_by_run(agent_run.agent_run_id):
            if Path(crop.crop_path).is_file():
                existing_by_triage.setdefault(crop.triage_result_id, []).append(crop)

        def segment_one(triage) -> tuple[list[MoleculeCrop], list[dict[str, Any]]]:
            existing_crops = existing_by_triage.get(triage.triage_result_id, [])
            if existing_crops and not force:
                return existing_crops, []
            block = block_by_id.get(triage.visual_block_id)
            if not block or not block.resolved_img_path:
                return [], [
                    {
                        "triage_result_id": triage.triage_result_id,
                        "visual_block_id": triage.visual_block_id,
                        "error": "Visual block image path is missing.",
                    }
                ]
            image_path = Path(block.resolved_img_path)
            if not image_path.exists():
                return [], [
                    {
                        "triage_result_id": triage.triage_result_id,
                        "visual_block_id": triage.visual_block_id,
                        "error": f"Visual block image does not exist: {image_path}",
                    }
                ]
            try:
                output_dir = output_root / agent_run.agent_run_id / triage.visual_block_id
                if _should_use_full_visual_block_crop(triage):
                    triage_crops = build_molecule_crops_from_full_image(
                        paper_id=normalized_paper_id,
                        agent_run_id=agent_run.agent_run_id,
                        triage=triage,
                        image_path=image_path,
                        output_dir=output_dir,
                    )
                else:
                    triage_crops = build_molecule_crops_from_decimer_segmentation(
                        paper_id=normalized_paper_id,
                        agent_run_id=agent_run.agent_run_id,
                        triage=triage,
                        image_path=image_path,
                        output_dir=output_dir,
                        client=client,
                        expand=segment_config.expand,
                        max_segments=max_segments or segment_config.max_segments,
                    )
                stored = self.molecule_crops.replace_for_triage_result(
                    triage.triage_result_id,
                    triage_crops,
                )
                if triage.error_message:
                    self.figure_triage.upsert(triage.model_copy(update={"error_message": None}))
                return stored, []
            except Exception as exc:
                error_text = str(exc)
                if error_text.startswith("manual_crop_required:"):
                    self.molecule_crops.replace_for_triage_result(triage.triage_result_id, [])
                    self.figure_triage.upsert(
                        triage.model_copy(update={"error_message": error_text})
                    )
                    return [], [
                        {
                            "triage_result_id": triage.triage_result_id,
                            "visual_block_id": triage.visual_block_id,
                            "status": "manual_crop_required",
                            "suggested_action": "manual_crop_or_manual_structure_input",
                            "error": error_text,
                        }
                    ]
                return [], [
                    {
                        "triage_result_id": triage.triage_result_id,
                        "visual_block_id": triage.visual_block_id,
                        "error": error_text,
                    }
                ]

        segment_results = _map_with_concurrency(
            triage_results,
            max_concurrency=max_concurrency,
            runner=segment_one,
        )
        crops = [crop for stored, _ in segment_results for crop in stored]
        errors = [error for _, failures in segment_results for error in failures]
        return DecimerSegmentationBatchResult(run=agent_run, crops=crops, errors=errors)

    def list_molecule_crops(self, paper_id: str) -> list[MoleculeCrop] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            return []
        return self.molecule_crops.list_by_run(agent_run.agent_run_id)

    def run_crop_validation(
        self,
        paper_id: str,
        *,
        provider: str = "qwen",
        model: str | None = None,
        limit: int | None = None,
        max_concurrency: int = 1,
        force: bool = False,
    ) -> MoleculeCropValidationBatchResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            raise ValueError(f"Material agent foundation is not ready for {normalized_paper_id}.")
        crops = self.molecule_crops.list_by_run(agent_run.agent_run_id)
        if limit is not None:
            crops = crops[: max(0, limit)]
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown VLM provider {provider!r}. Available: {known}")
        selected_model = model or provider_config.vision_model or provider_config.default_model
        client = self.vision_client or OpenAICompatibleVisionClient(provider_config)
        cached_entity_ids = (
            set()
            if force
            else self._completed_vlm_entity_ids(
                agent_run.agent_run_id,
                stage="crop_validation",
                provider=provider,
                model=selected_model,
                prompt_version=CROP_VALIDATION_PROMPT_VERSION,
            )
        )
        cached_validations = {
            validation.crop_id: validation
            for validation in self.crop_validations.list_by_paper(normalized_paper_id)
            if validation.agent_run_id == agent_run.agent_run_id
            and validation.status == "completed"
            and validation.provider == provider
            and validation.model == selected_model
            and validation.crop_id in cached_entity_ids
        }
        pending_crops = [crop for crop in crops if crop.crop_id not in cached_validations]
        raw_validations = _map_with_concurrency(
            pending_crops,
            max_concurrency=max_concurrency,
            runner=lambda crop: self._validate_crop(
                crop,
                client=client,
                provider=provider,
                model=selected_model,
                enable_thinking=provider_config.vision_enable_thinking,
            ),
        )
        stored_by_crop = dict(cached_validations)
        for validation in raw_validations:
            stored = self.crop_validations.upsert(validation)
            self.molecule_crops.apply_validation(stored)
            stored_by_crop[stored.crop_id] = stored
        validations = [
            stored_by_crop[crop.crop_id] for crop in crops if crop.crop_id in stored_by_crop
        ]
        return MoleculeCropValidationBatchResult(run=agent_run, validations=validations)

    def list_crop_validations(self, paper_id: str) -> list[MoleculeCropValidation] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            return []
        return [
            validation
            for validation in self.crop_validations.list_by_paper(normalized_paper_id)
            if validation.agent_run_id == agent_run.agent_run_id
        ]

    def run_label_binding(
        self,
        paper_id: str,
        *,
        provider: str = "qwen",
        model: str | None = None,
        limit: int | None = None,
        max_concurrency: int = 1,
        target_paper_material_ids: set[str] | None = None,
        force: bool = False,
    ) -> MoleculeLabelBindingBatchResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            raise ValueError(f"Material agent foundation is not ready for {normalized_paper_id}.")
        if not agent_run.source_candidate_run_id:
            raise ValueError(f"No candidate mining result is linked to {normalized_paper_id}.")
        material_context = self._paper_material_context(
            agent_run.source_candidate_run_id,
            device_used_only=True,
            target_paper_material_ids=target_paper_material_ids,
        )
        if not material_context:
            raise ValueError(
                f"No paper-local material candidates are available for {normalized_paper_id}."
            )
        crops = [
            crop
            for crop in self.molecule_crops.list_by_run(agent_run.agent_run_id)
            if crop.status == "ready_for_ocsr"
        ]
        if limit is not None:
            crops = crops[: max(0, limit)]
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown VLM provider {provider!r}. Available: {known}")
        selected_model = model or provider_config.vision_model or provider_config.default_model
        client = self.vision_client or OpenAICompatibleVisionClient(provider_config)
        expected_material_ids = {
            str(material.get("paper_material_id"))
            for material in material_context
            if material.get("paper_material_id")
        }
        cached_entity_ids = (
            set()
            if force
            else self._completed_vlm_entity_ids(
                agent_run.agent_run_id,
                stage="label_binding",
                provider=provider,
                model=selected_model,
                prompt_version=LABEL_BINDING_PROMPT_VERSION,
                expected_material_ids=expected_material_ids,
            )
        )
        cached_bindings: dict[str, MoleculeLabelBinding] = {}
        if not force:
            for binding in self.label_bindings.list_by_run(agent_run.agent_run_id):
                binding_material_ids = {
                    str(material.get("paper_material_id"))
                    for material in binding.candidate_materials
                    if material.get("paper_material_id")
                }
                is_deterministic = bool(binding.raw_response.get("deterministic"))
                grouped_call_id = str(binding.raw_response.get("group_call_id") or "")
                if (
                    binding.status == "completed"
                    and binding.provider == provider
                    and binding.model == selected_model
                    and binding_material_ids == expected_material_ids
                    and (
                        is_deterministic
                        or binding.crop_id in cached_entity_ids
                        or grouped_call_id in cached_entity_ids
                    )
                ):
                    cached_bindings[binding.crop_id] = binding
        blocks = {
            block.visual_block_id: block
            for block in self.visual_blocks.list_by_paper(normalized_paper_id)
        }
        triage_by_block = {
            triage.visual_block_id: triage
            for triage in self.figure_triage.list_by_paper(normalized_paper_id)
            if triage.agent_run_id == agent_run.agent_run_id
        }
        pending_crops = [crop for crop in crops if crop.crop_id not in cached_bindings]
        crop_groups: dict[str, list[MoleculeCrop]] = {}
        for crop in pending_crops:
            crop_groups.setdefault(crop.visual_block_id, []).append(crop)
        binding_work: list[list[MoleculeCrop]] = []
        for group_crops in crop_groups.values():
            ordered = sorted(group_crops, key=lambda item: (item.segment_index, item.crop_id))
            for offset in range(0, len(ordered), LABEL_BINDING_GROUP_MAX_CROPS):
                binding_work.append(ordered[offset : offset + LABEL_BINDING_GROUP_MAX_CROPS])
        raw_binding_groups = _map_with_concurrency(
            binding_work,
            max_concurrency=max_concurrency,
            runner=lambda group: self._bind_crop_group(
                group,
                agent_run=agent_run,
                block=blocks.get(group[0].visual_block_id),
                triage=triage_by_block.get(group[0].visual_block_id),
                material_context=material_context,
                client=client,
                provider=provider,
                model=selected_model,
                enable_thinking=provider_config.vision_enable_thinking,
            ),
        )
        raw_bindings = [
            binding for binding_group in raw_binding_groups for binding in binding_group
        ]
        stored_by_crop = dict(cached_bindings)
        for binding in raw_bindings:
            stored = self.label_bindings.upsert_proposal(binding)
            stored_by_crop[stored.crop_id] = stored
        bindings = [
            stored_by_crop[crop.crop_id] for crop in crops if crop.crop_id in stored_by_crop
        ]
        return MoleculeLabelBindingBatchResult(run=agent_run, bindings=bindings)

    def list_label_bindings(self, paper_id: str) -> list[MoleculeLabelBinding] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            return []
        ready_crop_ids = {
            crop.crop_id
            for crop in self.molecule_crops.list_by_run(agent_run.agent_run_id)
            if crop.status == "ready_for_ocsr"
        }
        return [
            binding
            for binding in self.label_bindings.list_by_run(agent_run.agent_run_id)
            if binding.crop_id in ready_crop_ids
        ]

    def run_decimer_ocsr(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
        allow_unreviewed_matches: bool = False,
        min_model_confidence: float = 0.8,
        target_paper_material_ids: set[str] | None = None,
        max_concurrency: int = 1,
    ) -> DecimerOCSRBatchResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            raise ValueError(f"Material agent foundation is not ready for {normalized_paper_id}.")
        if not agent_run.source_candidate_run_id:
            raise ValueError(f"No candidate mining result is linked to {normalized_paper_id}.")
        material_context = {
            str(material["paper_material_id"]): material
            for material in self._paper_material_context(
                agent_run.source_candidate_run_id,
                device_used_only=True,
                target_paper_material_ids=target_paper_material_ids,
            )
        }
        bindings = [
            binding
            for binding in self.label_bindings.list_by_run(agent_run.agent_run_id)
            if _binding_eligible_for_ocsr(
                binding,
                allow_unreviewed_matches=allow_unreviewed_matches,
                min_model_confidence=min_model_confidence,
            )
            and (
                target_paper_material_ids is None
                or _binding_paper_material_id(binding) in target_paper_material_ids
            )
        ]
        if limit is not None:
            bindings = bindings[: max(0, limit)]
        client = self.decimer_smiles_client or DecimerSmilesClient(
            self.config.external_services.decimer_smiles
        )

        def predict_one(
            binding: MoleculeLabelBinding,
        ) -> tuple[
            MaterialStructureCandidate | None,
            dict[str, Any] | None,
            dict[str, Any] | None,
        ]:
            paper_material_id = _binding_paper_material_id(binding)
            if paper_material_id is None:
                return None, None, None
            crop = self.molecule_crops.get(binding.crop_id)
            if not crop or crop.status != "ready_for_ocsr":
                return (
                    None,
                    {
                        "binding_id": binding.binding_id,
                        "crop_id": binding.crop_id,
                        "reason": "crop_not_ready_for_ocsr",
                    },
                    None,
                )
            existing = self.structure_candidates.get_by_source(
                agent_run.source_candidate_run_id,
                paper_material_id,
                "decimer_ocsr",
                binding.crop_id,
            )
            if existing and crop.updated_at <= existing.updated_at:
                return existing, None, None
            try:
                response = client.predict_smiles(Path(crop.crop_path))
                chemistry_error: str | None = None
                try:
                    standardized = standardize_smiles(response.smiles or "")
                except ValueError as exc:
                    standardized = None
                    chemistry_error = str(exc)
                material = material_context.get(paper_material_id, {})
                quality_warning = _ocsr_structure_quality_warning(
                    standardized.canonical_smiles if standardized else None,
                    str(material.get("material_class") or "unknown"),
                )
                observed_label = _binding_observed_label(binding)
                timestamp = now_iso()
                candidate = MaterialStructureCandidate(
                    structure_candidate_id=(
                        existing.structure_candidate_id if existing else uuid4().hex
                    ),
                    paper_id=normalized_paper_id,
                    candidate_run_id=agent_run.source_candidate_run_id,
                    paper_material_id=paper_material_id,
                    provider="decimer_ocsr",
                    resolver_name="decimer_image_to_smiles",
                    query_text=observed_label or paper_material_id,
                    query_type="bound_crop",
                    source_identifier=binding.crop_id,
                    canonical_name=(
                        material.get("canonical_name")
                        or material.get("full_name_in_paper")
                        or observed_label
                    ),
                    material_class=str(material.get("material_class") or "unknown"),
                    representation_type=("small_molecule" if standardized else "invalid_smiles"),
                    raw_smiles=response.smiles,
                    canonical_smiles=(standardized.canonical_smiles if standardized else None),
                    isomeric_smiles=(standardized.isomeric_smiles if standardized else None),
                    inchi=standardized.inchi if standardized else None,
                    inchi_key=standardized.inchi_key if standardized else None,
                    formula=standardized.formula if standardized else None,
                    molecular_weight=(standardized.molecular_weight if standardized else None),
                    evidence={
                        "binding_id": binding.binding_id,
                        "crop_id": binding.crop_id,
                        "crop_path": crop.crop_path,
                        "highlighted_source_figure_path": binding.highlighted_source_figure_path,
                        "binding_review_status": binding.review_status,
                        "used_unreviewed_model_match": binding.review_status == "pending_review",
                        "model_decision": binding.model_decision,
                        "model_confidence": binding.model_confidence,
                        "reviewed_observed_label": binding.reviewed_observed_label,
                        "model_observed_label": binding.model_observed_label,
                        "decimer_response": response.raw_response,
                        "rdkit_valid": standardized is not None,
                        "rdkit_error": chemistry_error,
                        "structure_quality_warning": quality_warning,
                    },
                    status=(
                        "needs_correction"
                        if standardized is None or quality_warning
                        else "pending_review"
                    ),
                    created_at=existing.created_at if existing else timestamp,
                    updated_at=timestamp,
                )
                return self.structure_candidates.upsert(candidate), None, None
            except Exception as exc:
                return (
                    None,
                    None,
                    {
                        "binding_id": binding.binding_id,
                        "crop_id": binding.crop_id,
                        "error": str(exc),
                    },
                )

        prediction_results = _map_with_concurrency(
            bindings,
            max_concurrency=max_concurrency,
            runner=predict_one,
        )
        candidates = [candidate for candidate, _, _ in prediction_results if candidate]
        skipped = [item for _, item, _ in prediction_results if item]
        errors = [item for _, _, item in prediction_results if item]
        return DecimerOCSRBatchResult(
            paper_id=normalized_paper_id,
            agent_run_id=agent_run.agent_run_id,
            eligible_binding_count=len(bindings),
            candidates=candidates,
            skipped=skipped,
            errors=errors,
        )

    def prepare_visual_assets(self, paper_id: str) -> MaterialAgentRun | None:
        """Ensure the reusable MinerU images and visual foundation exist for one paper."""
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            foundation = self.run_foundation(normalized_paper_id)
            if not foundation or foundation.run.status != "completed":
                raise ValueError(f"Material agent foundation failed for {normalized_paper_id}.")
            agent_run = foundation.run
        return self._ensure_visual_images(normalized_paper_id, agent_run)

    def run_ocsr_pipeline(
        self,
        paper_id: str,
        *,
        vision_provider: str = "qwen",
        vision_model: str | None = None,
        allow_unreviewed_matches: bool = True,
        min_model_confidence: float = 0.8,
        limit_visual_blocks: int | None = None,
        limit_crops: int | None = None,
        limit_ocsr: int | None = None,
        vlm_concurrency: int = 1,
        decimer_segmentation_concurrency: int | None = None,
        decimer_ocsr_concurrency: int | None = None,
        target_paper_material_ids: set[str] | None = None,
    ) -> MaterialOCSRPipelineResult | None:
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        agent_run = self.prepare_visual_assets(normalized_paper_id)
        if not agent_run:
            return None

        triage = self.run_figure_triage(
            normalized_paper_id,
            provider=vision_provider,
            model=vision_model,
            limit=limit_visual_blocks,
            max_concurrency=vlm_concurrency,
            target_paper_material_ids=target_paper_material_ids,
        )
        segmentation = self.run_decimer_segmentation(
            normalized_paper_id,
            limit=limit_visual_blocks,
            max_concurrency=(
                decimer_segmentation_concurrency
                or self.config.batch_worker.material_decimer_segmentation_concurrency
            ),
        )
        validation = self.run_crop_validation(
            normalized_paper_id,
            provider=vision_provider,
            model=vision_model,
            limit=limit_crops,
            max_concurrency=vlm_concurrency,
        )
        binding = self.run_label_binding(
            normalized_paper_id,
            provider=vision_provider,
            model=vision_model,
            limit=limit_crops,
            max_concurrency=vlm_concurrency,
            target_paper_material_ids=target_paper_material_ids,
        )
        ocsr = self.run_decimer_ocsr(
            normalized_paper_id,
            limit=limit_ocsr,
            allow_unreviewed_matches=allow_unreviewed_matches,
            min_model_confidence=min_model_confidence,
            target_paper_material_ids=target_paper_material_ids,
            max_concurrency=(
                decimer_ocsr_concurrency
                or self.config.batch_worker.material_decimer_ocsr_concurrency
            ),
        )
        errors: list[dict[str, Any]] = []
        if segmentation:
            errors.extend(segmentation.errors)
        if ocsr:
            errors.extend(ocsr.errors)
        latest_run = self._latest_completed_agent_run(normalized_paper_id) or agent_run
        return MaterialOCSRPipelineResult(
            run=latest_run,
            triage_count=len(triage.results) if triage else 0,
            crop_count=len(segmentation.crops) if segmentation else 0,
            validation_count=len(validation.validations) if validation else 0,
            binding_count=len(binding.bindings) if binding else 0,
            eligible_binding_count=ocsr.eligible_binding_count if ocsr else 0,
            ocsr_candidate_count=len(ocsr.candidates) if ocsr else 0,
            skipped_count=len(ocsr.skipped) if ocsr else 0,
            errors=errors,
        )

    def _ensure_visual_images(
        self,
        paper_id: str,
        agent_run: MaterialAgentRun,
    ) -> MaterialAgentRun:
        latest_mineru = self.mineru_runs.latest_completed_by_paper(paper_id)
        if latest_mineru and agent_run.mineru_run_id != latest_mineru.mineru_run_id:
            refreshed = self.run_foundation(paper_id)
            if refreshed and refreshed.run.status == "completed":
                agent_run = refreshed.run
        blocks = (
            self.visual_blocks.list_by_mineru_run(agent_run.mineru_run_id)
            if agent_run.mineru_run_id
            else []
        )
        if any(block.image_exists for block in blocks):
            return agent_run
        latest_mineru = self.mineru_runs.latest_completed_by_paper(paper_id)
        if latest_mineru and _mineru_images_were_requested(latest_mineru):
            raise ValueError(
                "MinerU image backfill was requested, but no visual image files were returned."
            )
        parsed = MinerUParseService(self.config).parse_paper(paper_id, include_images=True)
        if not parsed:
            raise ValueError(f"MinerU visual-image backfill failed for {paper_id}.")
        refreshed = self.run_foundation(paper_id)
        if not refreshed or refreshed.run.status != "completed":
            raise ValueError(f"Material visual foundation refresh failed for {paper_id}.")
        return refreshed.run

    def list_label_binding_review_events(
        self, paper_id: str
    ) -> list[MoleculeLabelBindingReviewEvent] | None:
        bindings = self.list_label_bindings(paper_id)
        if bindings is None:
            return None
        binding_ids = {binding.binding_id for binding in bindings}
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return [
            event
            for event in self.label_binding_events.list_by_paper(normalized_paper_id)
            if event.binding_id in binding_ids
        ]

    def list_vlm_call_logs(self, paper_id: str, *, limit: int = 100) -> list[VLMCallLog] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        agent_run = self._latest_completed_agent_run(normalized_paper_id)
        if not agent_run:
            return []
        return self.vlm_calls.list_by_run(agent_run.agent_run_id)[: max(0, limit)]

    def review_label_binding(
        self,
        binding_id: str,
        action: MoleculeLabelBindingReviewAction,
    ) -> MoleculeLabelBinding | None:
        self.init_runtime()
        binding = self.label_bindings.get(binding_id)
        if not binding:
            return None
        already_reviewed = next(
            (
                candidate
                for candidate in self.label_bindings.list_by_crop(binding.crop_id)
                if candidate.binding_id != binding.binding_id
                and candidate.review_status
                in {
                    "confirmed",
                    "corrected",
                    "unresolved",
                    "material_missing",
                    "not_target_material",
                    "invalid_crop",
                }
            ),
            None,
        )
        if already_reviewed:
            raise ValueError(
                "This crop already has a reviewed decision from another model proposal. "
                "Undo that decision before reviewing this proposal."
            )
        binding_material_ids = {
            str(material["paper_material_id"])
            for material in binding.candidate_materials
            if material.get("paper_material_id")
        }
        latest_candidate_run = self._latest_completed_candidate_run(binding.paper_id)
        current_material_ids = {
            str(material["paper_material_id"])
            for material in self._paper_material_context(
                latest_candidate_run.candidate_run_id if latest_candidate_run else None
            )
            if material.get("paper_material_id")
        }
        # A binding keeps the material candidates from the model run that created it.
        # Human correction must also accept materials added by a newer extraction run.
        material_ids = current_material_ids or binding_material_ids
        timestamp = now_iso()
        if action.action == "confirm":
            paper_material_id = binding.model_proposed_paper_material_id
            if not paper_material_id or paper_material_id not in material_ids:
                raise ValueError(
                    "Binding does not have a valid model-proposed material to confirm."
                )
            review_status = "confirmed"
        elif action.action == "correct":
            paper_material_id = action.reviewed_paper_material_id
            if not paper_material_id or paper_material_id not in material_ids:
                raise ValueError(
                    "Corrected paper_material_id must exist in the paper's current material list."
                )
            review_status = "corrected"
        elif action.action == "mark_unresolved":
            paper_material_id = None
            review_status = "unresolved"
        elif action.action == "mark_material_missing":
            paper_material_id = None
            review_status = "material_missing"
        elif action.action == "mark_not_target_material":
            paper_material_id = None
            review_status = "not_target_material"
        elif action.action == "mark_invalid_crop":
            paper_material_id = None
            review_status = "invalid_crop"
        else:
            raise ValueError(f"Unsupported label binding review action: {action.action}")
        observed_label = (
            action.reviewed_observed_label
            if action.reviewed_observed_label is not None
            else binding.model_observed_label
        )
        updated = binding.model_copy(
            update={
                "reviewed_paper_material_id": paper_material_id,
                "reviewed_observed_label": observed_label,
                "review_status": review_status,
                "reviewed_by": action.actor,
                "reviewed_at": timestamp,
                "review_note": action.message,
                "updated_at": timestamp,
            }
        )
        self.label_bindings.update_review(updated)
        self.label_binding_events.add(
            MoleculeLabelBindingReviewEvent(
                event_id=uuid4().hex,
                binding_id=binding.binding_id,
                paper_id=binding.paper_id,
                crop_id=binding.crop_id,
                action=action.action,
                actor=action.actor,
                message=action.message,
                before_reviewed_paper_material_id=binding.reviewed_paper_material_id,
                after_reviewed_paper_material_id=updated.reviewed_paper_material_id,
                before_observed_label=binding.reviewed_observed_label,
                after_observed_label=updated.reviewed_observed_label,
                before_review_status=binding.review_status,
                after_review_status=updated.review_status,
                created_at=timestamp,
            )
        )
        return updated

    def undo_label_binding_review_event(
        self,
        event_id: str,
        action: MoleculeLabelBindingReviewAction,
    ) -> MoleculeLabelBinding | None:
        self.init_runtime()
        event = self.label_binding_events.get(event_id)
        if not event or event.action == "undo":
            return None
        binding = self.label_bindings.get(event.binding_id)
        if not binding:
            return None
        timestamp = now_iso()
        updated = binding.model_copy(
            update={
                "reviewed_paper_material_id": event.before_reviewed_paper_material_id,
                "reviewed_observed_label": event.before_observed_label,
                "review_status": event.before_review_status,
                "reviewed_by": action.actor,
                "reviewed_at": timestamp,
                "review_note": action.message or f"Undo label binding event {event.event_id}",
                "updated_at": timestamp,
            }
        )
        self.label_bindings.update_review(updated)
        self.label_binding_events.add(
            MoleculeLabelBindingReviewEvent(
                event_id=uuid4().hex,
                binding_id=binding.binding_id,
                paper_id=binding.paper_id,
                crop_id=binding.crop_id,
                action="undo",
                actor=action.actor,
                message=updated.review_note,
                before_reviewed_paper_material_id=binding.reviewed_paper_material_id,
                after_reviewed_paper_material_id=updated.reviewed_paper_material_id,
                before_observed_label=binding.reviewed_observed_label,
                after_observed_label=updated.reviewed_observed_label,
                before_review_status=binding.review_status,
                after_review_status=updated.review_status,
                created_at=timestamp,
            )
        )
        return updated

    def get_visual_block_image_path(self, visual_block_id: str) -> Path | None:
        self.init_runtime()
        block = self.visual_blocks.get(visual_block_id)
        if not block or not block.resolved_img_path:
            return None
        path = Path(block.resolved_img_path)
        return path if path.exists() and path.is_file() else None

    def get_molecule_crop_image_path(self, crop_id: str) -> Path | None:
        self.init_runtime()
        crop = self.molecule_crops.get(crop_id)
        if not crop:
            return None
        path = Path(crop.crop_path)
        return path if path.exists() and path.is_file() else None

    def get_label_binding_highlighted_image_path(self, binding_id: str) -> Path | None:
        self.init_runtime()
        binding = self.label_bindings.get(binding_id)
        if not binding:
            return None
        crop = self.molecule_crops.get(binding.crop_id)
        if crop and _is_full_visual_block_crop(crop):
            source_path = Path(crop.source_image_path)
            return source_path if source_path.exists() and source_path.is_file() else None
        path = Path(binding.highlighted_source_figure_path)
        return path if path.exists() and path.is_file() else None

    def _triage_block(
        self,
        block: DocumentVisualBlock,
        *,
        agent_run: MaterialAgentRun,
        client: VisionClient,
        provider: str,
        model: str,
        enable_thinking: bool | None,
        material_context: list[dict[str, Any]],
    ) -> FigureTriageResult:
        timestamp = now_iso()
        call: VLMCallLog | None = None
        started_perf: float | None = None
        response = None
        try:
            image_path = Path(block.resolved_img_path or "")
            messages = build_figure_triage_messages(
                block=block,
                image_data_url=image_path_to_data_url(image_path),
                material_context=material_context,
            )
            call, started_perf = self._start_vlm_call(
                agent_run=agent_run,
                stage="figure_triage",
                input_entity_type="visual_block",
                input_entity_id=block.visual_block_id,
                provider=provider,
                model=model,
                prompt_version=FIGURE_TRIAGE_PROMPT_VERSION,
                input_image_paths=[image_path.as_posix()],
                input_context={
                    "enable_thinking": enable_thinking,
                    "caption": block.caption,
                    "nearby_text": block.nearby_text,
                    "material_candidate_ids": [
                        material.get("paper_material_id") for material in material_context
                    ],
                    "material_candidate_scope": "device_used_materials_only",
                },
            )
            response = client.generate_json(messages, model=model, temperature=0)
            if response.parsed_json is None:
                raise ValueError("VLM response did not contain a JSON object.")
            payload = normalize_figure_triage_payload(response.parsed_json, material_context)
            self._finish_vlm_call(call, started_perf, response=response)
            return FigureTriageResult(
                triage_result_id=uuid4().hex,
                agent_run_id=agent_run.agent_run_id,
                visual_block_id=block.visual_block_id,
                paper_id=block.paper_id,
                provider=provider,
                model=model,
                raw_response={
                    "parsed": response.parsed_json,
                    "usage": response.usage,
                },
                created_at=timestamp,
                updated_at=timestamp,
                **payload,
            )
        except Exception as exc:
            if call and started_perf is not None:
                self._finish_vlm_call(call, started_perf, response=response, error=str(exc))
            return FigureTriageResult(
                triage_result_id=uuid4().hex,
                agent_run_id=agent_run.agent_run_id,
                visual_block_id=block.visual_block_id,
                paper_id=block.paper_id,
                provider=provider,
                model=model,
                status="failed",
                error_message=str(exc),
                created_at=timestamp,
                updated_at=timestamp,
            )

    def _validate_crop(
        self,
        crop: MoleculeCrop,
        *,
        client: VisionClient,
        provider: str,
        model: str,
        enable_thinking: bool | None,
    ) -> MoleculeCropValidation:
        timestamp = now_iso()
        call: VLMCallLog | None = None
        started_perf: float | None = None
        response = None
        try:
            crop_path = Path(crop.crop_path)
            messages = build_crop_validation_messages(
                crop=crop,
                image_data_url=image_path_to_data_url(crop_path),
            )
            call, started_perf = self._start_vlm_call(
                agent_run_id=crop.agent_run_id,
                paper_id=crop.paper_id,
                stage="crop_validation",
                input_entity_type="molecule_crop",
                input_entity_id=crop.crop_id,
                provider=provider,
                model=model,
                prompt_version=CROP_VALIDATION_PROMPT_VERSION,
                input_image_paths=[crop_path.as_posix()],
                input_context={
                    "enable_thinking": enable_thinking,
                    "bbox": crop.bbox,
                    "width": crop.width,
                    "height": crop.height,
                },
            )
            response = client.generate_json(messages, model=model, temperature=0)
            if response.parsed_json is None:
                raise ValueError("VLM response did not contain a JSON object.")
            payload = normalize_crop_validation_payload(response.parsed_json)
            self._finish_vlm_call(call, started_perf, response=response)
            return MoleculeCropValidation(
                validation_id=uuid4().hex,
                crop_id=crop.crop_id,
                paper_id=crop.paper_id,
                agent_run_id=crop.agent_run_id,
                visual_block_id=crop.visual_block_id,
                provider=provider,
                model=model,
                raw_response={
                    "parsed": response.parsed_json,
                    "usage": response.usage,
                },
                created_at=timestamp,
                updated_at=timestamp,
                **payload,
            )
        except Exception as exc:
            if call and started_perf is not None:
                self._finish_vlm_call(call, started_perf, response=response, error=str(exc))
            return MoleculeCropValidation(
                validation_id=uuid4().hex,
                crop_id=crop.crop_id,
                paper_id=crop.paper_id,
                agent_run_id=crop.agent_run_id,
                visual_block_id=crop.visual_block_id,
                provider=provider,
                model=model,
                status="failed",
                error_message=str(exc),
                created_at=timestamp,
                updated_at=timestamp,
            )

    def _bind_crop_group(
        self,
        crops: list[MoleculeCrop],
        *,
        agent_run: MaterialAgentRun,
        block: DocumentVisualBlock | None,
        triage: FigureTriageResult | None,
        material_context: list[dict[str, Any]],
        client: VisionClient,
        provider: str,
        model: str,
        enable_thinking: bool | None,
    ) -> list[MoleculeLabelBinding]:
        if not crops:
            return []
        deterministic: list[MoleculeCrop] = []
        grouped: list[MoleculeCrop] = []
        for crop in crops:
            if (
                deterministic_full_visual_block_label_binding_payload(
                    crop=crop,
                    triage=triage,
                    material_context=material_context,
                )
                is not None
            ):
                deterministic.append(crop)
            else:
                grouped.append(crop)
        bindings = [
            self._bind_crop_label(
                crop,
                agent_run=agent_run,
                block=block,
                triage=triage,
                material_context=material_context,
                client=client,
                provider=provider,
                model=model,
                enable_thinking=enable_thinking,
            )
            for crop in deterministic
        ]
        if len(grouped) == 1:
            bindings.append(
                self._bind_crop_label(
                    grouped[0],
                    agent_run=agent_run,
                    block=block,
                    triage=triage,
                    material_context=material_context,
                    client=client,
                    provider=provider,
                    model=model,
                    enable_thinking=enable_thinking,
                )
            )
        elif grouped:
            bindings.extend(
                self._bind_figure_labels(
                    grouped,
                    agent_run=agent_run,
                    block=block,
                    triage=triage,
                    material_context=material_context,
                    client=client,
                    provider=provider,
                    model=model,
                    enable_thinking=enable_thinking,
                )
            )
        return bindings

    def _bind_figure_labels(
        self,
        crops: list[MoleculeCrop],
        *,
        agent_run: MaterialAgentRun,
        block: DocumentVisualBlock | None,
        triage: FigureTriageResult | None,
        material_context: list[dict[str, Any]],
        client: VisionClient,
        provider: str,
        model: str,
        enable_thinking: bool | None,
    ) -> list[MoleculeLabelBinding]:
        timestamp = now_iso()
        source_path = Path(crops[0].source_image_path)
        highlighted_paths: dict[str, Path] = {}
        call: VLMCallLog | None = None
        started_perf: float | None = None
        response = None
        group_call_id = _label_binding_group_call_id(crops)
        try:
            if not block:
                raise ValueError(f"Visual block is missing for crop group {group_call_id}.")
            for crop in crops:
                highlighted_paths[crop.crop_id] = create_highlighted_source_figure(
                    crop,
                    output_dir=(
                        self.config.paths.runtime_dir
                        / "material_agent"
                        / "label_binding_context"
                        / agent_run.agent_run_id
                    ),
                )
            messages = build_grouped_label_binding_messages(
                crops=crops,
                block=block,
                triage=triage,
                material_context=material_context,
                original_image_data_url=image_path_to_data_url(source_path),
                highlighted_image_data_urls={
                    crop_id: image_path_to_data_url(path)
                    for crop_id, path in highlighted_paths.items()
                },
                crop_image_data_urls={
                    crop.crop_id: image_path_to_data_url(Path(crop.crop_path)) for crop in crops
                },
            )
            input_paths = [source_path.as_posix()]
            for crop in crops:
                input_paths.extend([highlighted_paths[crop.crop_id].as_posix(), crop.crop_path])
            call, started_perf = self._start_vlm_call(
                agent_run=agent_run,
                stage="label_binding",
                input_entity_type="visual_block_crop_group",
                input_entity_id=group_call_id,
                provider=provider,
                model=model,
                prompt_version=LABEL_BINDING_PROMPT_VERSION,
                input_image_paths=input_paths,
                input_context={
                    "enable_thinking": enable_thinking,
                    "grouped": True,
                    "crop_ids": [crop.crop_id for crop in crops],
                    "caption": block.caption,
                    "nearby_text": block.nearby_text,
                    "triage_label_candidates": triage.label_candidates if triage else [],
                    "material_candidate_ids": [
                        material.get("paper_material_id") for material in material_context
                    ],
                },
            )
            response = client.generate_json(messages, model=model, temperature=0)
            if response.parsed_json is None:
                raise ValueError("VLM response did not contain a JSON object.")
            raw_items = response.parsed_json.get("bindings")
            if not isinstance(raw_items, list):
                raise ValueError("Grouped label-binding response must contain a bindings array.")
            payload_by_crop = {
                str(item.get("crop_id")): item
                for item in raw_items
                if isinstance(item, dict) and item.get("crop_id")
            }
            self._finish_vlm_call(call, started_perf, response=response)
            bindings: list[MoleculeLabelBinding] = []
            for crop in crops:
                raw_payload = payload_by_crop.get(crop.crop_id) or {
                    "decision": "ambiguous",
                    "confidence": 0,
                    "reason": "The grouped VLM response omitted this crop_id.",
                }
                payload = normalize_label_binding_payload(raw_payload, material_context)
                bindings.append(
                    MoleculeLabelBinding(
                        binding_id=uuid4().hex,
                        paper_id=crop.paper_id,
                        candidate_run_id=agent_run.source_candidate_run_id or "",
                        agent_run_id=agent_run.agent_run_id,
                        crop_id=crop.crop_id,
                        visual_block_id=crop.visual_block_id,
                        provider=provider,
                        model=model,
                        source_figure_path=source_path.as_posix(),
                        highlighted_source_figure_path=highlighted_paths[crop.crop_id].as_posix(),
                        crop_path=crop.crop_path,
                        caption_text=block.caption,
                        nearby_text=block.nearby_text,
                        triage_label_candidates=triage.label_candidates if triage else [],
                        candidate_materials=material_context,
                        raw_response={
                            "grouped": True,
                            "group_call_id": group_call_id,
                            "group_crop_ids": [item.crop_id for item in crops],
                            "prompt_version": LABEL_BINDING_PROMPT_VERSION,
                            "parsed": raw_payload,
                            "usage": response.usage,
                        },
                        created_at=timestamp,
                        updated_at=timestamp,
                        **payload,
                    )
                )
            return bindings
        except Exception as exc:
            if call and started_perf is not None:
                self._finish_vlm_call(call, started_perf, response=response, error=str(exc))
            return [
                MoleculeLabelBinding(
                    binding_id=uuid4().hex,
                    paper_id=crop.paper_id,
                    candidate_run_id=agent_run.source_candidate_run_id or "",
                    agent_run_id=agent_run.agent_run_id,
                    crop_id=crop.crop_id,
                    visual_block_id=crop.visual_block_id,
                    provider=provider,
                    model=model,
                    source_figure_path=source_path.as_posix(),
                    highlighted_source_figure_path=highlighted_paths.get(
                        crop.crop_id, source_path
                    ).as_posix(),
                    crop_path=crop.crop_path,
                    caption_text=block.caption if block else None,
                    nearby_text=block.nearby_text if block else None,
                    triage_label_candidates=triage.label_candidates if triage else [],
                    candidate_materials=material_context,
                    raw_response={
                        "grouped": True,
                        "group_call_id": group_call_id,
                        "group_crop_ids": [item.crop_id for item in crops],
                        "prompt_version": LABEL_BINDING_PROMPT_VERSION,
                    },
                    status="failed",
                    model_decision="failed",
                    error_message=str(exc),
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                for crop in crops
            ]

    def _bind_crop_label(
        self,
        crop: MoleculeCrop,
        *,
        agent_run: MaterialAgentRun,
        block: DocumentVisualBlock | None,
        triage: FigureTriageResult | None,
        material_context: list[dict[str, Any]],
        client: VisionClient,
        provider: str,
        model: str,
        enable_thinking: bool | None,
    ) -> MoleculeLabelBinding:
        timestamp = now_iso()
        source_path = Path(crop.source_image_path)
        highlighted_path = source_path
        call: VLMCallLog | None = None
        started_perf: float | None = None
        response = None
        try:
            if not block:
                raise ValueError(f"Visual block is missing for crop {crop.crop_id}.")
            highlighted_path = create_highlighted_source_figure(
                crop,
                output_dir=(
                    self.config.paths.runtime_dir
                    / "material_agent"
                    / "label_binding_context"
                    / agent_run.agent_run_id
                ),
            )
            deterministic_payload = deterministic_full_visual_block_label_binding_payload(
                crop=crop,
                triage=triage,
                material_context=material_context,
            )
            if deterministic_payload is not None:
                return MoleculeLabelBinding(
                    binding_id=uuid4().hex,
                    paper_id=crop.paper_id,
                    candidate_run_id=agent_run.source_candidate_run_id or "",
                    agent_run_id=agent_run.agent_run_id,
                    crop_id=crop.crop_id,
                    visual_block_id=crop.visual_block_id,
                    provider=provider,
                    model=model,
                    source_figure_path=source_path.as_posix(),
                    highlighted_source_figure_path=highlighted_path.as_posix(),
                    crop_path=crop.crop_path,
                    caption_text=block.caption,
                    nearby_text=block.nearby_text,
                    triage_label_candidates=triage.label_candidates if triage else [],
                    candidate_materials=material_context,
                    raw_response={
                        "deterministic": True,
                        "rule": "single_full_visual_block_related_material",
                        "triage_result_id": triage.triage_result_id if triage else None,
                        "crop_raw_segment": crop.raw_segment,
                    },
                    created_at=timestamp,
                    updated_at=timestamp,
                    **deterministic_payload,
                )
            messages = build_label_binding_messages(
                crop=crop,
                block=block,
                triage=triage,
                material_context=material_context,
                original_image_data_url=image_path_to_data_url(source_path),
                highlighted_image_data_url=image_path_to_data_url(highlighted_path),
                crop_image_data_url=image_path_to_data_url(Path(crop.crop_path)),
            )
            call, started_perf = self._start_vlm_call(
                agent_run=agent_run,
                stage="label_binding",
                input_entity_type="molecule_crop",
                input_entity_id=crop.crop_id,
                provider=provider,
                model=model,
                prompt_version=LABEL_BINDING_PROMPT_VERSION,
                input_image_paths=[
                    source_path.as_posix(),
                    highlighted_path.as_posix(),
                    crop.crop_path,
                ],
                input_context={
                    "enable_thinking": enable_thinking,
                    "caption": block.caption,
                    "nearby_text": block.nearby_text,
                    "triage_label_candidates": triage.label_candidates if triage else [],
                    "material_candidate_ids": [
                        material.get("paper_material_id") for material in material_context
                    ],
                },
            )
            response = client.generate_json(messages, model=model, temperature=0)
            if response.parsed_json is None:
                raise ValueError("VLM response did not contain a JSON object.")
            payload = normalize_label_binding_payload(response.parsed_json, material_context)
            self._finish_vlm_call(call, started_perf, response=response)
            return MoleculeLabelBinding(
                binding_id=uuid4().hex,
                paper_id=crop.paper_id,
                candidate_run_id=agent_run.source_candidate_run_id or "",
                agent_run_id=agent_run.agent_run_id,
                crop_id=crop.crop_id,
                visual_block_id=crop.visual_block_id,
                provider=provider,
                model=model,
                source_figure_path=source_path.as_posix(),
                highlighted_source_figure_path=highlighted_path.as_posix(),
                crop_path=crop.crop_path,
                caption_text=block.caption,
                nearby_text=block.nearby_text,
                triage_label_candidates=triage.label_candidates if triage else [],
                candidate_materials=material_context,
                raw_response={"parsed": response.parsed_json, "usage": response.usage},
                created_at=timestamp,
                updated_at=timestamp,
                **payload,
            )
        except Exception as exc:
            if call and started_perf is not None:
                self._finish_vlm_call(call, started_perf, response=response, error=str(exc))
            return MoleculeLabelBinding(
                binding_id=uuid4().hex,
                paper_id=crop.paper_id,
                candidate_run_id=agent_run.source_candidate_run_id or "",
                agent_run_id=agent_run.agent_run_id,
                crop_id=crop.crop_id,
                visual_block_id=crop.visual_block_id,
                provider=provider,
                model=model,
                source_figure_path=source_path.as_posix(),
                highlighted_source_figure_path=highlighted_path.as_posix(),
                crop_path=crop.crop_path,
                caption_text=block.caption if block else None,
                nearby_text=block.nearby_text if block else None,
                triage_label_candidates=triage.label_candidates if triage else [],
                candidate_materials=material_context,
                status="failed",
                model_decision="failed",
                error_message=str(exc),
                created_at=timestamp,
                updated_at=timestamp,
            )

    def _start_vlm_call(
        self,
        *,
        stage: str,
        input_entity_type: str,
        input_entity_id: str,
        provider: str,
        model: str,
        prompt_version: str,
        input_image_paths: list[str],
        input_context: dict[str, Any],
        agent_run: MaterialAgentRun | None = None,
        agent_run_id: str | None = None,
        paper_id: str | None = None,
    ) -> tuple[VLMCallLog, float]:
        started_perf = perf_counter()
        call = VLMCallLog(
            vlm_call_id=uuid4().hex,
            paper_id=paper_id or (agent_run.paper_id if agent_run else ""),
            agent_run_id=agent_run_id or (agent_run.agent_run_id if agent_run else ""),
            stage=stage,
            input_entity_type=input_entity_type,
            input_entity_id=input_entity_id,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            input_image_paths=input_image_paths,
            input_context=input_context,
            status="running",
            started_at=now_iso(),
        )
        self.vlm_calls.create(call)
        return call, started_perf

    def _completed_vlm_entity_ids(
        self,
        agent_run_id: str,
        *,
        stage: str,
        provider: str,
        model: str,
        prompt_version: str,
        expected_material_ids: set[str] | None = None,
    ) -> set[str]:
        completed: set[str] = set()
        for call in self.vlm_calls.list_by_run(agent_run_id):
            if (
                call.status != "completed"
                or call.stage != stage
                or call.provider != provider
                or call.model != model
                or call.prompt_version != prompt_version
            ):
                continue
            if expected_material_ids is not None:
                call_material_ids = {
                    str(value)
                    for value in call.input_context.get("material_candidate_ids", [])
                    if value
                }
                if call_material_ids != expected_material_ids:
                    continue
            completed.add(call.input_entity_id)
        return completed

    def _finish_vlm_call(
        self,
        call: VLMCallLog,
        started_perf: float,
        *,
        response: Any | None = None,
        error: str | None = None,
    ) -> VLMCallLog:
        completed = call.model_copy(
            update={
                "parsed_response": response.parsed_json
                if response and response.parsed_json
                else {},
                "usage": response.usage if response else {},
                "status": "failed" if error else "completed",
                "error_message": error,
                "finished_at": now_iso(),
                "duration_ms": round((perf_counter() - started_perf) * 1000),
            }
        )
        return self.vlm_calls.update(completed)

    def _latest_completed_candidate_run(self, paper_id: str) -> CandidateIngestionRun | None:
        runs = self.candidate_runs.list_runs_by_paper(paper_id)
        return next((run for run in runs if run.status == "completed"), None)

    def _latest_completed_agent_run(self, paper_id: str) -> MaterialAgentRun | None:
        runs = self.agent_runs.list_by_paper(paper_id)
        return next((run for run in runs if run.status == "completed"), None)

    def _material_count(self, candidate_run: CandidateIngestionRun | None) -> int:
        if not candidate_run:
            return 0
        entities = self.candidate_runs.list_entities_by_run(candidate_run.candidate_run_id)
        return sum(1 for entity in entities if entity.entity_type == "materials")

    def _paper_material_context(
        self,
        candidate_run_id: str | None,
        *,
        device_used_only: bool = False,
        target_paper_material_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidate_run_id:
            return []
        entities = self.candidate_runs.list_entities_by_run(candidate_run_id)
        device_used_ids = _device_used_material_ids_from_entities(entities)
        materials: list[dict[str, Any]] = []
        for entity in entities:
            if entity.entity_type != "materials":
                continue
            source = entity.source_json
            if not isinstance(source, dict):
                source = {}
            paper_material_id = (
                _str_or_none(source.get("paper_material_id"))
                or entity.entity_label
                or entity.entity_path
            )
            is_device_used = paper_material_id in device_used_ids if device_used_ids else True
            if device_used_only and device_used_ids and not is_device_used:
                continue
            if (
                target_paper_material_ids is not None
                and paper_material_id not in target_paper_material_ids
            ):
                continue
            materials.append(
                {
                    "paper_material_id": paper_material_id,
                    "entity_label": entity.entity_label,
                    "mention_list": _string_or_list(source.get("mention_list")),
                    "full_name_in_paper": _str_or_none(source.get("full_name_in_paper")),
                    "canonical_name": _str_or_none(source.get("canonical_name")),
                    "abbreviation": _str_or_none(source.get("abbreviation")),
                    "paper_specific_label": _str_or_none(source.get("paper_specific_label")),
                    "material_class": _str_or_none(source.get("material_class")),
                    "is_device_used": is_device_used,
                    "selection_scope": (
                        "device_used_material"
                        if device_used_ids
                        else "fallback_all_materials_no_device_refs"
                    ),
                }
            )
        return materials


def _device_used_material_ids_from_entities(entities: list[Any]) -> set[str]:
    used: set[str] = set()
    for entity in entities:
        if entity.entity_type == "materials":
            continue
        source = entity.source_json
        if not isinstance(source, dict | list):
            continue
        used.update(_collect_paper_material_ids(source))
    return used


def _collect_paper_material_ids(value: object) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        material_id = _str_or_none(value.get("paper_material_id"))
        if material_id:
            ids.add(material_id)
        for child in value.values():
            ids.update(_collect_paper_material_ids(child))
    elif isinstance(value, list):
        for child in value:
            ids.update(_collect_paper_material_ids(child))
    return ids


def collect_document_visual_blocks(
    *,
    run: MinerUParseRun,
    content_list: list[dict[str, Any]],
    agent_run_id: str | None,
    result_key: str | None = None,
) -> list[DocumentVisualBlock]:
    timestamp = now_iso()
    blocks: list[DocumentVisualBlock] = []
    for content_index, item in enumerate(content_list):
        content_type = str(item.get("type") or "")
        if content_type not in VISUAL_BLOCK_TYPES:
            continue
        img_path = _image_path(item)
        resolved_img_path, image_candidates = _resolve_image_path(run, result_key, img_path)
        page_idx = _int_or_none(item.get("page_idx"))
        visual_block_id = f"{run.mineru_run_id}_{content_index:05d}"
        blocks.append(
            DocumentVisualBlock(
                visual_block_id=visual_block_id,
                paper_id=run.paper_id,
                mineru_run_id=run.mineru_run_id,
                collected_by_agent_run_id=agent_run_id,
                content_index=content_index,
                content_type=content_type,
                sub_type=_str_or_none(item.get("sub_type")),
                page_idx=page_idx,
                page_id=page_idx + 1 if page_idx is not None else None,
                bbox=_bbox(item),
                img_path=img_path,
                resolved_img_path=resolved_img_path.as_posix() if resolved_img_path else None,
                image_exists=bool(resolved_img_path and resolved_img_path.exists()),
                caption=_caption_text(item) or None,
                nearby_text=_nearby_text(content_list, content_index) or None,
                source_json={
                    "content_item": item,
                    "image_path_candidates": [path.as_posix() for path in image_candidates],
                    "result_key": result_key,
                },
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
    return blocks


def materialize_visual_block_images(
    blocks: list[DocumentVisualBlock],
    *,
    output_dir: Path,
    fallback_blocks: list[DocumentVisualBlock] | None = None,
) -> list[DocumentVisualBlock]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_by_id = {
        block.visual_block_id: block
        for block in (fallback_blocks or [])
        if block.resolved_img_path and Path(block.resolved_img_path).is_file()
    }
    materialized: list[DocumentVisualBlock] = []
    for block in blocks:
        source_path = _recoverable_visual_block_source(
            block,
            output_root=output_dir.parent,
            fallback_block=fallback_by_id.get(block.visual_block_id),
        )
        if source_path is None:
            materialized.append(block)
            continue
        suffix = source_path.suffix or ".png"
        target_path = output_dir / f"{block.visual_block_id}{suffix}"
        shutil.copy2(source_path, target_path)
        source_json = dict(block.source_json)
        if not block.resolved_img_path or Path(block.resolved_img_path) != source_path:
            source_json["recovered_from_materialized_path"] = source_path.as_posix()
        materialized.append(
            block.model_copy(
                update={
                    "resolved_img_path": target_path.as_posix(),
                    "image_exists": True,
                    "source_json": source_json,
                    "updated_at": now_iso(),
                }
            )
        )
    return materialized


def _recoverable_visual_block_source(
    block: DocumentVisualBlock,
    *,
    output_root: Path,
    fallback_block: DocumentVisualBlock | None,
) -> Path | None:
    if block.resolved_img_path:
        current = Path(block.resolved_img_path)
        if current.is_file():
            return current
    if fallback_block and fallback_block.resolved_img_path:
        previous = Path(fallback_block.resolved_img_path)
        if previous.is_file():
            return previous
    matches = [path for path in output_root.glob(f"*/{block.visual_block_id}.*") if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime, default=None)


def build_figure_triage_messages(
    *,
    block: DocumentVisualBlock,
    image_data_url: str,
    material_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schema = {
        "contains_molecular_structures": "boolean",
        "image_role": (
            "one of flat_2d_molecular_structures, reaction_scheme_with_2d_structures, "
            "3d_ball_and_stick_or_crystal_structure, molecule_plus_orbital, "
            "energy_level_diagram, device_structure, spectrum_or_plot, table, mixed_or_unclear, unknown"
        ),
        "has_clean_structure_depictions": "boolean",
        "has_flat_2d_structure_diagrams": "boolean; true only for normal 2D chemical line drawings",
        "has_3d_ball_and_stick_model": "boolean; true for X-ray/crystal/3D ball-stick renderings",
        "has_crystal_structure_rendering": "boolean",
        "has_surface_or_photo_rendering": "boolean",
        "has_orbital_overlay": "boolean",
        "has_energy_level_diagram": "boolean",
        "has_device_stack": "boolean",
        "should_run_decimer_segmentation": "boolean",
        "label_candidates": "array of visible or caption-implied molecule/material labels",
        "related_paper_material_ids": "array of paper_material_id values from the provided material list",
        "confidence": "number from 0 to 1",
        "reason": "short evidence-based explanation",
    }
    context = {
        "visual_block": {
            "visual_block_id": block.visual_block_id,
            "content_type": block.content_type,
            "sub_type": block.sub_type,
            "page_id": block.page_id,
            "caption": block.caption,
            "nearby_text": block.nearby_text,
        },
        "paper_materials": material_context,
        "required_json_schema": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a chemistry figure triage assistant for OLED papers. "
                "Send visual blocks to DECIMER when they contain at least one separable, clean, "
                "normal, flat 2D chemical structure line drawing. Do not reject the whole image "
                "only because separate HOMO/LUMO orbital panels, 3D conformers, plots, or annotations "
                "appear elsewhere in the same visual block; DECIMER segmentation and crop validation "
                "will filter those non-structure regions later. Reject pure 3D ball-and-stick models, "
                "X-ray/crystal structure renderings, pure molecular orbital/electron-density plots, "
                "device schematics, spectra, tables and photos when no separable 2D line drawing is present. "
                "Use only the provided device-used paper material candidates; do not bind or "
                "prioritize reference molecules that are not used in the extracted device structures. "
                "Do not invent labels. Return only one valid JSON object."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Classify this visual block for a material-structure extraction pipeline. "
                        "The paper_materials list contains only materials used in the extracted OLED "
                        "device structures. Use that list only for related_paper_material_ids. "
                        "Set should_run_decimer_segmentation=true when the image contains at least one "
                        "clean, complete, spatially separable flat 2D chemical line drawing suitable "
                        "for DECIMER segmentation. A mixed energy-level or device figure must still be "
                        "segmented when such a separable 2D structure is present. Set it to false for "
                        "pure energy-level/device diagrams and for ball-and-stick/crystal renderings, "
                        "3D models, orbital-only plots, spectra, tables, photos, or surfaces without a "
                        "separable flat 2D structure.\n\n"
                        f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def normalize_figure_triage_payload(
    payload: dict[str, Any],
    material_context: list[dict[str, Any]],
) -> dict[str, Any]:
    known_material_ids = {
        str(item.get("paper_material_id"))
        for item in material_context
        if item.get("paper_material_id")
    }
    related_ids = [
        item
        for item in _string_or_list(payload.get("related_paper_material_ids"))
        if item in known_material_ids
    ]
    image_role = _str_or_none(payload.get("image_role")) or "unknown"
    confidence = _float_or_none(payload.get("confidence"))
    has_flat_2d = bool(payload.get("has_flat_2d_structure_diagrams")) or image_role in {
        "flat_2d_molecular_structures",
        "reaction_scheme_with_2d_structures",
        "molecule_structures",
        "reaction_scheme",
    }
    has_3d_model = bool(payload.get("has_3d_ball_and_stick_model"))
    has_crystal = bool(payload.get("has_crystal_structure_rendering")) or image_role in {
        "3d_ball_and_stick_or_crystal_structure",
        "crystal_structure",
    }
    has_surface_or_photo = bool(payload.get("has_surface_or_photo_rendering"))
    has_orbital_overlay = bool(payload.get("has_orbital_overlay"))
    has_energy_level_diagram = bool(payload.get("has_energy_level_diagram"))
    has_device_stack = bool(payload.get("has_device_stack"))
    model_segmentation = bool(payload.get("should_run_decimer_segmentation"))
    has_clean_structure_depictions = bool(payload.get("has_clean_structure_depictions"))
    is_molecule_plus_orbital = image_role == "molecule_plus_orbital"
    if is_molecule_plus_orbital and has_clean_structure_depictions:
        has_flat_2d = True
    has_separable_flat_structure = (
        bool(payload.get("contains_molecular_structures"))
        and has_flat_2d
        and has_clean_structure_depictions
    )
    model_or_mixed_figure_request = model_segmentation or (
        has_separable_flat_structure and bool(related_ids)
    )
    blocks_segmentation_by_role = image_role in {
        "3d_ball_and_stick_or_crystal_structure",
        "spectrum_or_plot",
        "table",
    }
    has_nonseparable_diagram = (
        has_energy_level_diagram or has_device_stack
    ) and not has_separable_flat_structure
    should_segment = (
        model_or_mixed_figure_request
        and has_separable_flat_structure
        and not blocks_segmentation_by_role
        and not has_surface_or_photo
        and not has_nonseparable_diagram
    )
    return {
        "contains_molecular_structures": bool(payload.get("contains_molecular_structures")),
        "image_role": image_role,
        "has_clean_structure_depictions": has_clean_structure_depictions,
        "has_flat_2d_structure_diagrams": has_flat_2d,
        "has_3d_ball_and_stick_model": has_3d_model,
        "has_crystal_structure_rendering": has_crystal,
        "has_surface_or_photo_rendering": has_surface_or_photo,
        "has_orbital_overlay": has_orbital_overlay,
        "has_energy_level_diagram": has_energy_level_diagram,
        "has_device_stack": has_device_stack,
        "should_run_decimer_segmentation": should_segment,
        "label_candidates": _dedupe_terms(_string_or_list(payload.get("label_candidates"))),
        "related_paper_material_ids": _dedupe_terms(related_ids),
        "confidence": None if confidence is None else max(0.0, min(1.0, confidence)),
        "reason": _str_or_none(payload.get("reason")),
        "status": "completed",
    }


def build_crop_validation_messages(
    *,
    crop: MoleculeCrop,
    image_data_url: str,
) -> list[dict[str, Any]]:
    schema = {
        "is_molecular_depiction": "boolean",
        "is_single_molecule": "boolean",
        "is_complete_structure": "boolean",
        "has_benign_highlight": "boolean; colored atoms/rings or emphasis that does not hide bond topology",
        "is_flat_2d_structure_diagram": "boolean; true only for a normal 2D chemical line drawing",
        "has_3d_ball_and_stick_model": "boolean",
        "has_crystal_structure_rendering": "boolean",
        "has_photo_or_surface_rendering": "boolean",
        "is_ocsr_readable": "boolean; all atoms and bonds of one target molecule remain readable for image-to-SMILES",
        "has_blocking_interference": "boolean; overlay or graphic actually obscures or merges with target atoms/bonds",
        "has_orbital_overlay": "boolean",
        "has_excess_annotation": "boolean",
        "has_multiple_structures": "boolean",
        "has_reaction_arrow": "boolean",
        "has_non_structural_graphics": "boolean",
        "should_run_ocsr": "boolean",
        "confidence": "number from 0 to 1",
        "reason": "short evidence-based explanation",
    }
    return [
        {
            "role": "system",
            "content": (
                "You decide whether a DECIMER crop contains a molecular graph readable by optical "
                "chemical structure recognition (OCSR). Approve only clean, normal, flat 2D "
                "chemical line drawings with readable atom labels and bond topology. Reject 3D "
                "ball-and-stick models, X-ray/crystal structure renderings, molecular orbital "
                "or electron-density images, photos/surface renderings, and any depiction where "
                "stereoscopic 3D geometry rather than a 2D line drawing carries the structure. "
                "Benign atom/ring highlights on a 2D drawing are allowed if topology remains clear. "
                "Return only one valid JSON object."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Decide whether this DECIMER segmentation crop should be sent to an "
                        "image-to-SMILES model. Set should_run_ocsr=true only for one complete, "
                        "readable, flat 2D chemical structure diagram. Set should_run_ocsr=false "
                        "for ball-and-stick/crystal structures, 3D models, orbital/electron-density "
                        "renderings, photos, surface renderings, plots, tables, device diagrams, "
                        "or any non-2D depiction, even if the molecule identity is visually clear. "
                        "Benign atom/ring highlighting on an otherwise clean 2D drawing is acceptable.\n\n"
                        f"Crop metadata: {json.dumps({'crop_id': crop.crop_id, 'bbox': crop.bbox, 'width': crop.width, 'height': crop.height}, ensure_ascii=False)}\n"
                        f"Required JSON schema: {json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def normalize_crop_validation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    is_molecular_depiction = bool(payload.get("is_molecular_depiction"))
    is_single_molecule = bool(payload.get("is_single_molecule"))
    is_complete_structure = bool(payload.get("is_complete_structure"))
    has_benign_highlight = bool(payload.get("has_benign_highlight"))
    is_flat_2d = bool(payload.get("is_flat_2d_structure_diagram", True))
    has_3d_model = bool(payload.get("has_3d_ball_and_stick_model"))
    has_crystal = bool(payload.get("has_crystal_structure_rendering"))
    has_photo_or_surface = bool(payload.get("has_photo_or_surface_rendering"))
    has_orbital_overlay = bool(payload.get("has_orbital_overlay"))
    has_excess_annotation = bool(payload.get("has_excess_annotation"))
    has_multiple_structures = bool(payload.get("has_multiple_structures"))
    has_reaction_arrow = bool(payload.get("has_reaction_arrow"))
    has_non_structural_graphics = bool(payload.get("has_non_structural_graphics"))
    has_blocking_interference = bool(payload.get("has_blocking_interference"))
    legacy_readability_default = (
        is_molecular_depiction
        and is_single_molecule
        and is_complete_structure
        and not has_orbital_overlay
        and not has_excess_annotation
        and not has_multiple_structures
        and not has_reaction_arrow
        and not has_non_structural_graphics
    )
    is_ocsr_readable = bool(payload.get("is_ocsr_readable", legacy_readability_default))
    model_recommendation = bool(payload.get("should_run_ocsr"))
    should_run_ocsr = (
        model_recommendation
        and is_molecular_depiction
        and is_single_molecule
        and is_complete_structure
        and is_ocsr_readable
        and is_flat_2d
        and not has_3d_model
        and not has_crystal
        and not has_photo_or_surface
        and not has_blocking_interference
        and not has_multiple_structures
    )
    confidence = _float_or_none(payload.get("confidence"))
    return {
        "is_molecular_depiction": is_molecular_depiction,
        "is_single_molecule": is_single_molecule,
        "is_complete_structure": is_complete_structure,
        "has_benign_highlight": has_benign_highlight,
        "is_flat_2d_structure_diagram": is_flat_2d,
        "has_3d_ball_and_stick_model": has_3d_model,
        "has_crystal_structure_rendering": has_crystal,
        "has_photo_or_surface_rendering": has_photo_or_surface,
        "is_ocsr_readable": is_ocsr_readable,
        "has_blocking_interference": has_blocking_interference,
        "has_orbital_overlay": has_orbital_overlay,
        "has_excess_annotation": has_excess_annotation,
        "has_multiple_structures": has_multiple_structures,
        "has_reaction_arrow": has_reaction_arrow,
        "has_non_structural_graphics": has_non_structural_graphics,
        "should_run_ocsr": should_run_ocsr,
        "confidence": None if confidence is None else max(0.0, min(1.0, confidence)),
        "reason": _str_or_none(payload.get("reason")),
        "status": "completed",
    }


def _is_full_visual_block_crop(crop: MoleculeCrop) -> bool:
    return (
        crop.raw_segment.get("source") == "full_visual_block"
        or crop.validation_json.get("crop_mode") == "full_visual_block"
    )


def deterministic_full_visual_block_label_binding_payload(
    *,
    crop: MoleculeCrop,
    triage: FigureTriageResult | None,
    material_context: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_full_visual_block_crop(crop) or triage is None:
        return None
    known_material_ids = {
        str(material.get("paper_material_id"))
        for material in material_context
        if material.get("paper_material_id")
    }
    related_ids = _dedupe_terms(
        [
            *(str(item) for item in crop.raw_segment.get("related_paper_material_ids", []) if item),
            *triage.related_paper_material_ids,
        ]
    )
    related_ids = [material_id for material_id in related_ids if material_id in known_material_ids]
    label_candidates = _dedupe_terms(
        [
            *(str(item) for item in crop.raw_segment.get("label_candidates", []) if item),
            *triage.label_candidates,
        ]
    )
    if len(related_ids) != 1 or len(label_candidates) != 1:
        return None
    confidence = (
        triage.confidence if triage.confidence is not None else crop.segmentation_confidence
    )
    return {
        "model_observed_label": label_candidates[0],
        "model_label_source": "triage_single_full_visual_block",
        "model_proposed_paper_material_id": related_ids[0],
        "model_alternative_paper_material_ids": [],
        "model_decision": "matched",
        "model_confidence": None if confidence is None else max(0.0, min(1.0, confidence)),
        "model_reason": (
            "Deterministic binding from a full visual-block crop with exactly one triage label "
            "and exactly one related device-used paper material."
        ),
        "review_status": "pending_review",
        "status": "completed",
    }


def create_highlighted_source_figure(crop: MoleculeCrop, *, output_dir: Path) -> Path:
    source_path = Path(crop.source_image_path)
    if not source_path.exists():
        raise ValueError(f"Crop source image does not exist: {source_path}")
    bbox = crop.bbox
    if len(bbox) != 4:
        raise ValueError(f"Crop bbox is invalid: {crop.crop_id}")
    if _is_full_visual_block_crop(crop):
        return source_path
    output_dir.mkdir(parents=True, exist_ok=True)
    highlighted_path = output_dir / f"{crop.crop_id}_highlighted.png"
    with Image.open(source_path) as image:
        rendered = image.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        stroke_width = max(3, min(rendered.size) // 120)
        draw.rectangle(
            tuple(int(round(value)) for value in bbox), outline="red", width=stroke_width
        )
        rendered.save(highlighted_path)
    return highlighted_path


def build_label_binding_messages(
    *,
    crop: MoleculeCrop,
    block: DocumentVisualBlock,
    triage: FigureTriageResult | None,
    material_context: list[dict[str, Any]],
    original_image_data_url: str,
    highlighted_image_data_url: str,
    crop_image_data_url: str,
) -> list[dict[str, Any]]:
    schema = {
        "observed_label": "visible label for the red-boxed molecular structure, or null",
        "label_source": "one of visible_near_crop, figure_caption, nearby_text, inferred_context, unknown",
        "depiction_scope": "one of full_named_material, fragment_or_r_group, unclear",
        "proposed_paper_material_id": "one paper_material_id from paper_material_candidates, or null",
        "alternative_paper_material_ids": "array of other possible IDs from paper_material_candidates",
        "decision": "one of matched, ambiguous, no_visible_label, material_not_in_candidate_list",
        "confidence": "number from 0 to 1",
        "reason": "short evidence-based explanation",
    }
    context = {
        "current_crop": {
            "crop_id": crop.crop_id,
            "bbox_in_source_figure": crop.bbox,
            "segment_index": crop.segment_index,
        },
        "visual_context": {
            "caption": block.caption,
            "nearby_text": block.nearby_text,
            "triage_label_candidates": triage.label_candidates if triage else [],
        },
        "paper_material_candidates": material_context,
        "required_json_schema": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "You bind one flat 2D molecular-structure crop in an OLED paper figure to a "
                "paper-local material entity used in the extracted device structures. Use visible "
                "labels, caption text and provided material candidates only. Never invent a "
                "paper_material_id and never bind a reference molecule or synthesis intermediate "
                "that is absent from the candidate list. A crop containing only an R group, "
                "substituent, shared scaffold, or other partial graph is not the complete named "
                "material even when the surrounding figure identifies which material uses it. "
                "Classify it as fragment_or_r_group and do not return matched. If the structure "
                "cannot be uniquely assigned, explicitly return an unresolved decision. Return "
                "only one valid JSON object."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The images are ordered as: (1) original source figure, "
                        "(2) context image with the current crop boxed in red when the crop is smaller "
                        "than the source figure; for full-figure crops this image may be identical to "
                        "the original, (3) the extracted crop. Identify only the current crop structure. "
                        "A matched decision is permitted only when proposed_paper_material_id "
                        "is one of the supplied device-used candidates and the crop itself contains "
                        "the complete molecular graph of that named material. A detached R-group, "
                        "substituent, or common core must use depiction_scope=fragment_or_r_group "
                        "and an unresolved decision; do not treat it as the complete material. "
                        "If a visible label describes "
                        "a material absent from the candidate list, including a reference emitter or "
                        "synthetic intermediate, use material_not_in_candidate_list.\n\n"
                        f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": original_image_data_url}},
                {"type": "image_url", "image_url": {"url": highlighted_image_data_url}},
                {"type": "image_url", "image_url": {"url": crop_image_data_url}},
            ],
        },
    ]


def build_grouped_label_binding_messages(
    *,
    crops: list[MoleculeCrop],
    block: DocumentVisualBlock,
    triage: FigureTriageResult | None,
    material_context: list[dict[str, Any]],
    original_image_data_url: str,
    highlighted_image_data_urls: dict[str, str],
    crop_image_data_urls: dict[str, str],
) -> list[dict[str, Any]]:
    binding_schema = {
        "crop_id": "exact crop_id supplied in crop_contexts",
        "observed_label": "visible label for this molecular structure, or null",
        "label_source": (
            "one of visible_near_crop, figure_caption, nearby_text, inferred_context, unknown"
        ),
        "depiction_scope": "one of full_named_material, fragment_or_r_group, unclear",
        "proposed_paper_material_id": (
            "one paper_material_id from paper_material_candidates, or null"
        ),
        "alternative_paper_material_ids": (
            "array of other possible IDs from paper_material_candidates"
        ),
        "decision": ("one of matched, ambiguous, no_visible_label, material_not_in_candidate_list"),
        "confidence": "number from 0 to 1",
        "reason": "short evidence-based explanation",
    }
    context = {
        "crop_contexts": [
            {
                "crop_id": crop.crop_id,
                "bbox_in_source_figure": crop.bbox,
                "segment_index": crop.segment_index,
            }
            for crop in crops
        ],
        "visual_context": {
            "caption": block.caption,
            "nearby_text": block.nearby_text,
            "triage_label_candidates": triage.label_candidates if triage else [],
        },
        "paper_material_candidates": material_context,
        "required_json_schema": {"bindings": [binding_schema]},
    }
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "First image: the original source figure. It is followed by two images for each "
                "crop in crop_contexts order: a context image with that crop boxed in red, then "
                "the extracted crop. Return exactly one bindings entry for every supplied crop_id. "
                "Associate each crop independently. A matched decision is allowed only when the "
                "paper_material_id is one of the supplied device-used candidates. Reference "
                "molecules and synthesis intermediates absent from the candidate list must use "
                "material_not_in_candidate_list. A crop that is only an R group, substituent, "
                "shared scaffold, or partial molecular graph must use "
                "depiction_scope=fragment_or_r_group and cannot be matched to the complete named "
                "material. Do not infer an assignment merely because a candidate exists.\n\n"
                f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            ),
        },
        {"type": "image_url", "image_url": {"url": original_image_data_url}},
    ]
    for crop in crops:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Context and extracted image for crop_id={crop.crop_id}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": highlighted_image_data_urls[crop.crop_id]},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": crop_image_data_urls[crop.crop_id]},
                },
            ]
        )
    return [
        {
            "role": "system",
            "content": (
                "You bind multiple flat 2D molecular-structure crops from one OLED paper figure "
                "to paper-local material entities used in extracted device structures. Use visible "
                "labels, relative positions, caption text and supplied candidates only. Never "
                "invent a paper_material_id. Distinguish a complete named material from an R group, "
                "substituent, shared scaffold, or partial graph; partial depictions cannot receive "
                "a matched decision. Preserve crop_id exactly and return only one valid JSON object "
                "matching the required schema."
            ),
        },
        {"role": "user", "content": content},
    ]


def normalize_label_binding_payload(
    payload: dict[str, Any],
    material_context: list[dict[str, Any]],
) -> dict[str, Any]:
    known_material_ids = {
        str(material["paper_material_id"])
        for material in material_context
        if material.get("paper_material_id")
    }
    decision = _str_or_none(payload.get("decision")) or "ambiguous"
    allowed_decisions = {
        "matched",
        "ambiguous",
        "no_visible_label",
        "material_not_in_candidate_list",
    }
    if decision not in allowed_decisions:
        decision = "ambiguous"
    reason = _str_or_none(payload.get("reason"))
    depiction_scope = (_str_or_none(payload.get("depiction_scope")) or "unclear").lower()
    is_partial_structure = depiction_scope == "fragment_or_r_group" or (
        depiction_scope not in {"full_named_material", "fragment_or_r_group"}
        and _label_binding_reason_indicates_partial_structure(reason)
    )
    proposed_material_id = _str_or_none(payload.get("proposed_paper_material_id"))
    if proposed_material_id not in known_material_ids:
        if proposed_material_id:
            decision = "material_not_in_candidate_list"
        proposed_material_id = None
    if decision == "matched" and not proposed_material_id:
        decision = "ambiguous"
    if is_partial_structure:
        decision = "ambiguous"
        proposed_material_id = None
    alternatives = [
        material_id
        for material_id in _dedupe_terms(
            _string_or_list(payload.get("alternative_paper_material_ids"))
        )
        if material_id in known_material_ids and material_id != proposed_material_id
    ]
    confidence = _float_or_none(payload.get("confidence"))
    return {
        "model_observed_label": _str_or_none(payload.get("observed_label")),
        "model_label_source": _str_or_none(payload.get("label_source")) or "unknown",
        "model_proposed_paper_material_id": proposed_material_id,
        "model_alternative_paper_material_ids": alternatives,
        "model_decision": decision,
        "model_confidence": None if confidence is None else max(0.0, min(1.0, confidence)),
        "model_reason": reason,
        "review_status": "pending_review",
        "status": "completed",
    }


def build_molecule_crops_from_segments(
    *,
    paper_id: str,
    agent_run_id: str,
    triage: FigureTriageResult,
    image_path: Path,
    output_dir: Path,
    segments: list[DecimerSegment],
) -> list[MoleculeCrop]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso()
    crops: list[MoleculeCrop] = []
    with Image.open(image_path) as image:
        width, height = image.size
        for segment in segments:
            output_index = segment.index
            bbox = _clamp_bbox(segment.bbox, image_width=width, image_height=height)
            if not bbox:
                continue
            crop_image = image.crop(tuple(int(round(value)) for value in bbox))
            crop_path = output_dir / f"segment_{output_index:03d}.png"
            crop_image.save(crop_path)
            crop_width, crop_height = crop_image.size
            validation_json = {
                "source_image_width": width,
                "source_image_height": height,
                "triage_confidence": triage.confidence,
                "triage_image_role": triage.image_role,
            }
            raw_segment = dict(segment.raw_segment)
            crops.append(
                MoleculeCrop(
                    crop_id=f"{triage.triage_result_id}_{output_index:03d}",
                    paper_id=paper_id,
                    agent_run_id=agent_run_id,
                    triage_result_id=triage.triage_result_id,
                    visual_block_id=triage.visual_block_id,
                    segment_index=output_index,
                    bbox=bbox,
                    source_image_path=image_path.as_posix(),
                    crop_path=crop_path.as_posix(),
                    width=crop_width,
                    height=crop_height,
                    validation_json=validation_json,
                    raw_segment=raw_segment,
                    status="pending_validation",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
    return crops


def build_molecule_crops_from_decimer_segmentation(
    *,
    paper_id: str,
    agent_run_id: str,
    triage: FigureTriageResult,
    image_path: Path,
    output_dir: Path,
    client: DecimerSegmentationClientProtocol,
    expand: bool,
    max_segments: int,
) -> list[MoleculeCrop]:
    try:
        response = client.segment_image(
            image_path,
            expand=expand,
            return_images=False,
            max_segments=max_segments,
        )
    except Exception as exc:
        raise RuntimeError(
            "manual_crop_required: DECIMER segmentation failed on the original visual block; "
            "automatic region guessing is disabled. Use manual crop selection or manual "
            f"structure input. original_error={exc}"
        ) from exc
    return build_molecule_crops_from_segments(
        paper_id=paper_id,
        agent_run_id=agent_run_id,
        triage=triage,
        image_path=image_path,
        output_dir=output_dir,
        segments=response.segments,
    )


def _should_use_full_visual_block_crop(triage: FigureTriageResult) -> bool:
    return (
        triage.should_run_decimer_segmentation
        and triage.contains_molecular_structures
        and triage.has_clean_structure_depictions
        and triage.image_role == "flat_2d_molecular_structures"
        and len(triage.label_candidates) == 1
        and len(triage.related_paper_material_ids) == 1
        and not triage.has_orbital_overlay
        and not triage.has_energy_level_diagram
        and not triage.has_device_stack
    )


def build_molecule_crops_from_full_image(
    *,
    paper_id: str,
    agent_run_id: str,
    triage: FigureTriageResult,
    image_path: Path,
    output_dir: Path,
) -> list[MoleculeCrop]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso()
    with Image.open(image_path) as image:
        width, height = image.size
        crop_path = output_dir / "segment_000.png"
        image.save(crop_path)
    return [
        MoleculeCrop(
            crop_id=f"{triage.triage_result_id}_000",
            paper_id=paper_id,
            agent_run_id=agent_run_id,
            triage_result_id=triage.triage_result_id,
            visual_block_id=triage.visual_block_id,
            segment_index=0,
            bbox=[0.0, 0.0, float(width), float(height)],
            source_image_path=image_path.as_posix(),
            crop_path=crop_path.as_posix(),
            width=width,
            height=height,
            segmentation_confidence=triage.confidence,
            validation_json={
                "source_image_width": width,
                "source_image_height": height,
                "triage_confidence": triage.confidence,
                "triage_image_role": triage.image_role,
                "crop_mode": "full_visual_block",
                "reason": "single-material flat 2D visual block; DECIMER segmentation skipped",
            },
            raw_segment={
                "source": "full_visual_block",
                "label_candidates": triage.label_candidates,
                "related_paper_material_ids": triage.related_paper_material_ids,
            },
            status="pending_validation",
            created_at=timestamp,
            updated_at=timestamp,
        )
    ]


def read_mineru_content_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"MinerU content_list must be a JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def mineru_result_key(run: MinerUParseRun) -> str | None:
    if not run.result_path:
        return None
    result_path = Path(run.result_path)
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or not results:
        return None
    return str(next(iter(results.keys())))


def _mineru_images_were_requested(run: MinerUParseRun) -> bool:
    if not run.result_path:
        return False
    result_path = Path(run.result_path)
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = payload.get("_evolab_local") if isinstance(payload, dict) else None
    return bool(isinstance(metadata, dict) and metadata.get("images_requested"))


def _caption_text(item: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "image_caption",
        "image_footnote",
        "chart_caption",
        "chart_footnote",
        "table_caption",
        "table_footnote",
        "text",
    ):
        parts.extend(_string_or_list(item.get(key)))
    if isinstance(item.get("table_body"), str):
        parts.append(re.sub(r"<[^>]+>", " ", str(item["table_body"])))
    if isinstance(item.get("content"), str):
        parts.append(str(item["content"]))
    return _compact_text(" ".join(parts))


def _nearby_text(content_list: list[dict[str, Any]], content_index: int, window: int = 2) -> str:
    item = content_list[content_index]
    page_idx = item.get("page_idx")
    nearby: list[str] = []
    lower = max(0, content_index - window)
    upper = min(len(content_list), content_index + window + 1)
    for index in range(lower, upper):
        if index == content_index:
            continue
        candidate = content_list[index]
        if candidate.get("page_idx") != page_idx:
            continue
        text = mineru_item_text(candidate) or _caption_text(candidate)
        if text:
            nearby.append(text)
    return _compact_text(" ".join(nearby))[:4000]


def _resolve_image_path(
    run: MinerUParseRun,
    result_key: str | None,
    img_path: str | None,
) -> tuple[Path | None, list[Path]]:
    if not img_path:
        return None, []
    raw_path = Path(img_path)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    if run.content_list_path:
        run_dir = Path(run.content_list_path).parent
        candidates.extend([run_dir / raw_path, run_dir / "hybrid_auto" / raw_path])
    if run.task_id:
        parser_version = run.parser_version or "3.1.0"
        for paper_name in _dedupe_terms([result_key, run.paper_id]):
            candidates.append(
                MINERU_OUTPUT_ROOT
                / parser_version
                / "output"
                / run.task_id
                / paper_name
                / "hybrid_auto"
                / raw_path
            )
    resolved = next((path for path in candidates if path.exists()), None)
    return resolved, candidates


def _image_path(item: Mapping[str, Any]) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _bbox(item: Mapping[str, Any]) -> list[float]:
    value = item.get("bbox")
    if not isinstance(value, list):
        return []
    return [float(raw) for raw in value if isinstance(raw, int | float)]


def _string_or_list(value: object) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _str_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _clamp_bbox(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
) -> list[float]:
    if len(bbox) != 4:
        return []
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(float(image_width), x0))
    x1 = max(0.0, min(float(image_width), x1))
    y0 = max(0.0, min(float(image_height), y0))
    y1 = max(0.0, min(float(image_height), y1))
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _dedupe_terms(values: list[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        compact = _compact_text(str(value))
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(compact)
    return output


def _count_by_type(blocks: list[DocumentVisualBlock]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.content_type] = counts.get(block.content_type, 0) + 1
    return counts


def _map_with_concurrency(
    items: list[Any],
    *,
    max_concurrency: int,
    runner,
) -> list[Any]:
    if not items:
        return []
    worker_count = max(1, min(max_concurrency, len(items)))
    if worker_count == 1:
        return [runner(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(runner, items))


def _label_binding_group_call_id(crops: list[MoleculeCrop]) -> str:
    visual_block_id = crops[0].visual_block_id if crops else "missing"
    crop_key = "\n".join(sorted(crop.crop_id for crop in crops))
    digest = sha256(crop_key.encode("utf-8")).hexdigest()[:16]
    return f"{visual_block_id}:crop-group:{digest}"


def _binding_paper_material_id(binding: MoleculeLabelBinding) -> str | None:
    if binding.review_status in {"confirmed", "corrected"} and binding.reviewed_paper_material_id:
        return binding.reviewed_paper_material_id
    if binding.model_decision == "matched" and binding.model_proposed_paper_material_id:
        return binding.model_proposed_paper_material_id
    return None


def _binding_observed_label(binding: MoleculeLabelBinding) -> str | None:
    return binding.reviewed_observed_label or binding.model_observed_label


def _label_binding_reason_indicates_partial_structure(reason: str | None) -> bool:
    return bool(reason and _PARTIAL_STRUCTURE_REASON_PATTERN.search(reason))


def _binding_is_partial_structure(binding: MoleculeLabelBinding) -> bool:
    parsed = binding.raw_response.get("parsed")
    depiction_scope = ""
    if isinstance(parsed, Mapping):
        depiction_scope = str(parsed.get("depiction_scope") or "").strip().lower()
    return depiction_scope == "fragment_or_r_group" or (
        binding.model_label_source == "inferred_context"
        and not binding.model_observed_label
        and _label_binding_reason_indicates_partial_structure(binding.model_reason)
    )


def _ocsr_structure_quality_warning(
    canonical_smiles: str | None,
    material_class: str,
) -> str | None:
    if not canonical_smiles:
        return None
    normalized_class = material_class.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_class in {"small_molecule", "small_molecule_organic"} and "." in canonical_smiles:
        return (
            "OCSR produced multiple disconnected components for a material classified as "
            "a single organic small molecule; compare with the source crop and remove any "
            "hallucinated reagent or fragment before acceptance."
        )
    return None


def _binding_eligible_for_ocsr(
    binding: MoleculeLabelBinding,
    *,
    allow_unreviewed_matches: bool,
    min_model_confidence: float,
) -> bool:
    if binding.status != "completed":
        return False
    if _binding_is_partial_structure(binding):
        return False
    if binding.review_status in {"confirmed", "corrected"} and binding.reviewed_paper_material_id:
        return True
    if not allow_unreviewed_matches:
        return False
    if binding.review_status != "pending_review":
        return False
    if binding.model_decision != "matched" or not binding.model_proposed_paper_material_id:
        return False
    confidence = binding.model_confidence or 0.0
    return confidence >= min_model_confidence
