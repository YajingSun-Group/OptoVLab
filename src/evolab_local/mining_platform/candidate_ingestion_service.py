from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.domain_template_service import DomainTemplateService
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.mining_result_validator import (
    MiningResultValidator,
    is_empty_value,
    resolve_field_values,
)
from evolab_local.mining_platform.paper_review_policy import no_device_review_reason
from evolab_local.mining_platform.schemas.candidate_ingestion import (
    CandidateEntity,
    CandidateFinalConfirmResult,
    CandidateFinalRecord,
    CandidateIngestionResult,
    CandidateReviewV2Bundle,
    CandidateValue,
    CandidateValueReviewEvent,
    CandidateValueUpdate,
)
from evolab_local.mining_platform.schemas.domain_template import DomainTemplate, TemplateField
from evolab_local.mining_platform.schemas.evidence import EvidenceAnchor
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    CandidateFinalRecordRepository,
    CandidateIngestionRepository,
    CandidateValueReviewEventRepository,
    EvidenceAnchorRepository,
    MaterialGlobalRepository,
    MaterialPropertyCandidateRepository,
    MaterialStructureCandidateRepository,
    PaperMaterialLinkRepository,
    PaperRepository,
)


ENTITY_ALIASES = {
    "material": "materials",
    "materials": "materials",
    "device": "devices",
    "devices": "devices",
    "layer": "layers",
    "layers": "layers",
    "component": "components",
    "components": "components",
    "performance": "performance",
}

ENTITY_PATH_PATTERNS = {
    "materials": re.compile(r"^materials\[\d+\]"),
    "devices": re.compile(r"^devices\[\d+\]"),
    "layers": re.compile(r"^devices\[\d+\]\.layers\[\d+\]"),
    "components": re.compile(r"^devices\[\d+\]\.layers\[\d+\]\.components\[\d+\]"),
    "performance": re.compile(r"^devices\[\d+\]\.performance\[\d+\]"),
}

PATH_PART_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")


class CandidateIngestionService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.anchors = EvidenceAnchorRepository(self.database)
        self.candidates = CandidateIngestionRepository(self.database)
        self.value_review_events = CandidateValueReviewEventRepository(self.database)
        self.final_records = CandidateFinalRecordRepository(self.database)
        self.material_links = PaperMaterialLinkRepository(self.database)
        self.material_structure_candidates = MaterialStructureCandidateRepository(self.database)
        self.material_property_candidates = MaterialPropertyCandidateRepository(self.database)
        self.global_materials = MaterialGlobalRepository(self.database)
        self.templates = DomainTemplateService(config)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def ingest_mining_result(
        self,
        paper_id: str,
        template_id: str,
        payload: dict[str, Any],
        source_name: str = "mock",
        source_version: str | None = None,
    ) -> CandidateIngestionResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None

        template = self.templates.get_template(template_id)
        report = MiningResultValidator(template).validate(payload)
        review_reason = no_device_review_reason(payload)
        status = "no_device" if review_reason else ("completed" if report.valid else "failed")
        error_message = _validation_error_summary(report)
        run = self.candidates.create_run(
            paper_id=normalized_paper_id,
            template_id=template.template_id,
            template_version=template.version,
            source_name=source_name,
            source_version=source_version,
            status=status,
            validation_report=report.model_dump(mode="json"),
            mining_result=payload,
            error_message=error_message,
        )

        if status == "failed":
            self.papers.set_mining_status(normalized_paper_id, "failed")
            self.papers.set_review_status(
                normalized_paper_id,
                "confirmed",
                reason="device_data_validation_failed",
            )
            return CandidateIngestionResult(run=run, validation_report=report)

        if status == "no_device":
            self.papers.set_mining_status(normalized_paper_id, "completed")
            self.papers.set_review_status(
                normalized_paper_id,
                "confirmed",
                reason=review_reason,
            )
            return CandidateIngestionResult(run=run, validation_report=report)

        timestamp = now_iso()
        anchor_map, anchors = self._build_evidence_anchors(
            normalized_paper_id,
            payload,
            timestamp,
        )
        entities, entity_ids_by_path = self._build_entities(
            normalized_paper_id,
            run.candidate_run_id,
            template,
            payload,
            timestamp,
        )
        values = self._build_values(
            normalized_paper_id,
            run.candidate_run_id,
            template,
            payload,
            entity_ids_by_path,
            anchor_map,
            timestamp,
        )
        self.anchors.add_many(anchors)
        self.candidates.add_entities(entities)
        self.candidates.add_values(values)
        self.papers.set_mining_status(normalized_paper_id, "completed")
        if review_reason:
            self.papers.set_review_status(
                normalized_paper_id,
                "confirmed",
                reason=review_reason,
            )
        else:
            self.papers.set_review_status(normalized_paper_id, "needs_review")
        return CandidateIngestionResult(
            run=run,
            validation_report=report,
            entity_count=len(entities),
            value_count=len(values),
            evidence_anchor_count=len(anchors),
        )

    def get_review_bundle(
        self,
        paper_id: str,
        *,
        compact: bool = False,
    ) -> CandidateReviewV2Bundle | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None
        runs = self.candidates.list_runs_by_paper(
            normalized_paper_id,
            include_payload=not compact,
        )
        run = next(
            (item for item in runs if item.status in {"completed", "no_device"}),
            runs[0] if runs else None,
        )
        if not run:
            return CandidateReviewV2Bundle()
        template = None if compact else self.templates.get_template(run.template_id)
        entities = self.candidates.list_entities_by_run(run.candidate_run_id)
        if compact:
            entities = [entity.model_copy(update={"source_json": {}}) for entity in entities]
        return CandidateReviewV2Bundle(
            run=run,
            entities=entities,
            values=self.candidates.list_values_by_run(run.candidate_run_id),
            evidence_anchors=self.anchors.list_by_paper(normalized_paper_id),
            template=template.model_dump(mode="json") if template else None,
        )

    def update_candidate_value(
        self,
        candidate_value_id: str,
        payload: CandidateValueUpdate,
    ) -> CandidateValue | None:
        self.init_runtime()
        before = self.candidates.get_value(candidate_value_id)
        if not before:
            return None
        reviewed_value = (
            payload.reviewed_value_json
            if "reviewed_value_json" in payload.model_fields_set
            else before.reviewed_value_json
        )
        requested_status = _normalize_review_status(payload.status)
        updated = self.candidates.update_value(
            candidate_value_id,
            reviewed_value,
            requested_status,
        )
        if not updated:
            return None
        self.value_review_events.add(
            before=before,
            after=updated,
            action=_review_action(before, updated, requested_status),
            actor=payload.actor,
            message=payload.message,
        )
        return updated

    def set_candidate_value_status(
        self,
        candidate_value_id: str,
        status: str,
        actor: str = "local_user",
        message: str | None = None,
    ) -> CandidateValue | None:
        self.init_runtime()
        current = self.candidates.get_value(candidate_value_id)
        if not current:
            return None
        normalized_status = _normalize_review_status(status) or status
        reviewed = current.reviewed_value_json
        if reviewed is None:
            reviewed = current.value_json
        updated = self.candidates.update_value(candidate_value_id, reviewed, normalized_status)
        if not updated:
            return None
        self.value_review_events.add(
            before=current,
            after=updated,
            action=_action_for_status(normalized_status),
            actor=actor,
            message=message,
        )
        return updated

    def list_value_review_events_by_run(
        self,
        candidate_run_id: str,
    ) -> list[CandidateValueReviewEvent]:
        self.init_runtime()
        return self.value_review_events.list_by_run(candidate_run_id)

    def list_value_review_events_by_value(
        self,
        candidate_value_id: str,
    ) -> list[CandidateValueReviewEvent]:
        self.init_runtime()
        return self.value_review_events.list_by_value(candidate_value_id)

    def undo_value_review_event(
        self,
        event_id: str,
        actor: str = "local_user",
        message: str | None = None,
    ) -> CandidateValue | None:
        self.init_runtime()
        event = self.value_review_events.get(event_id)
        if not event:
            return None
        current = self.candidates.get_value(event.candidate_value_id)
        if not current:
            return None
        updated = self.candidates.update_value(
            event.candidate_value_id,
            event.before_reviewed_value_json,
            event.before_status,
        )
        if not updated:
            return None
        self.value_review_events.add(
            before=current,
            after=updated,
            action="undo",
            actor=actor,
            message=message or f"Undo review event {event.event_id}",
        )
        return updated

    def list_final_records(self, paper_id: str) -> list[CandidateFinalRecord] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None
        return self.final_records.list_by_paper(normalized_paper_id)

    def confirm_review_v2(
        self,
        paper_id: str,
        actor: str = "local_user",
    ) -> CandidateFinalConfirmResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None
        runs = self.candidates.list_runs_by_paper(normalized_paper_id)
        run = next((item for item in runs if item.status == "completed"), None)
        if not run:
            return None

        values = self.candidates.list_values_by_run(run.candidate_run_id)
        final_json = _final_json_from_candidate_values(run.mining_result, values)
        final_json = self._enrich_final_json_with_material_structures(
            run.candidate_run_id,
            final_json,
        )
        if self.config.features.material_properties:
            final_json = self._enrich_final_json_with_material_properties(
                run.candidate_run_id,
                final_json,
            )
        else:
            _apply_material_property_payloads(final_json, {})
        source_value_ids = [
            value.candidate_value_id for value in values if value.status != "rejected"
        ]
        timestamp = now_iso()
        record = CandidateFinalRecord(
            final_record_id=uuid4().hex,
            paper_id=normalized_paper_id,
            candidate_run_id=run.candidate_run_id,
            template_id=run.template_id,
            template_version=run.template_version,
            final_json=final_json,
            source_candidate_value_ids=source_value_ids,
            confirmed_by=actor,
            status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
            confirmed_at=timestamp,
        )
        stored = self.final_records.replace_for_paper_template(
            normalized_paper_id,
            run.template_id,
            record,
        )
        self.papers.set_review_status(normalized_paper_id, "confirmed")
        return CandidateFinalConfirmResult(
            paper_id=normalized_paper_id,
            final_record=stored,
            final_value_count=len(source_value_ids),
        )

    def _enrich_final_json_with_material_structures(
        self,
        candidate_run_id: str,
        final_json: dict[str, Any],
    ) -> dict[str, Any]:
        material_statuses = self._material_structure_statuses_by_paper_material(
            candidate_run_id
        )
        if not material_statuses:
            return final_json
        _apply_material_structure_statuses(final_json, material_statuses)
        return final_json

    def _enrich_final_json_with_material_properties(
        self,
        candidate_run_id: str,
        final_json: dict[str, Any],
    ) -> dict[str, Any]:
        property_payloads = _accepted_material_property_payloads_by_material(
            self.material_property_candidates.list_by_run(candidate_run_id)
        )
        if not property_payloads:
            return final_json
        _apply_material_property_payloads(final_json, property_payloads)
        return final_json

    def _material_structure_statuses_by_paper_material(
        self,
        candidate_run_id: str,
    ) -> dict[str, dict[str, Any]]:
        links = self.material_links.list_by_run(candidate_run_id)
        if not links:
            return {}
        candidates = self.material_structure_candidates.list_by_run(candidate_run_id)
        accepted_candidates_by_id = {
            candidate.structure_candidate_id: candidate
            for candidate in candidates
            if candidate.status == "accepted"
        }
        accepted_candidates_by_material: dict[str, list[Any]] = {}
        for candidate in accepted_candidates_by_id.values():
            accepted_candidates_by_material.setdefault(
                candidate.paper_material_id,
                [],
            ).append(candidate)
        global_ids = sorted(
            {link.global_material_id for link in links if link.global_material_id}
        )
        global_by_id = {
            material.global_material_id: material
            for material in self.global_materials.get_many(global_ids)
        }
        statuses: dict[str, dict[str, Any]] = {}
        for link in links:
            candidate = None
            candidate_id = link.evidence.get("structure_candidate_id")
            if isinstance(candidate_id, str):
                candidate = accepted_candidates_by_id.get(candidate_id)
            if candidate is None:
                candidates_for_material = accepted_candidates_by_material.get(
                    link.paper_material_id,
                    [],
                )
                candidate = candidates_for_material[0] if candidates_for_material else None
            global_material = (
                global_by_id.get(link.global_material_id)
                if link.global_material_id
                else None
            )
            status = _material_structure_status_from_link(
                link,
                candidate,
                global_material,
            )
            if status:
                statuses[link.paper_material_id] = status
        return statuses

    def _build_evidence_anchors(
        self,
        paper_id: str,
        payload: Mapping[str, Any],
        timestamp: str,
    ) -> tuple[dict[str, str], list[EvidenceAnchor]]:
        anchor_ids_by_evidence_id: dict[str, str] = {}
        anchors: list[EvidenceAnchor] = []
        evidence_items = payload.get("evidence")
        if not isinstance(evidence_items, list):
            return anchor_ids_by_evidence_id, anchors
        for item in evidence_items:
            if not isinstance(item, Mapping):
                continue
            evidence_id = item.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                continue
            anchor_id = uuid4().hex
            anchor_ids_by_evidence_id[evidence_id] = anchor_id
            anchors.append(
                EvidenceAnchor(
                    evidence_anchor_id=anchor_id,
                    paper_id=paper_id,
                    page_id=item.get("page_id") if isinstance(item.get("page_id"), int) else None,
                    block_id=item.get("block_id")
                    if isinstance(item.get("block_id"), str)
                    else None,
                    bbox=item.get("bbox") if isinstance(item.get("bbox"), list) else [],
                    source_text=(
                        item.get("source_text")
                        if isinstance(item.get("source_text"), str)
                        else None
                    ),
                    source_type=(
                        item.get("source_type")
                        if isinstance(item.get("source_type"), str)
                        else "text"
                    ),
                    created_at=timestamp,
                )
            )
        return anchor_ids_by_evidence_id, anchors

    def _build_entities(
        self,
        paper_id: str,
        candidate_run_id: str,
        template: DomainTemplate,
        payload: Mapping[str, Any],
        timestamp: str,
    ) -> tuple[list[CandidateEntity], dict[str, str]]:
        entities: list[CandidateEntity] = []
        entity_ids_by_path: dict[str, str] = {}

        def add_entity(
            entity_type: str,
            entity_path: str,
            source: Mapping[str, Any],
            label: str | None,
            sort_order: int,
            parent_path: str | None = None,
        ) -> None:
            entity_id = uuid4().hex
            entity_ids_by_path[entity_path] = entity_id
            entities.append(
                CandidateEntity(
                    candidate_entity_id=entity_id,
                    candidate_run_id=candidate_run_id,
                    paper_id=paper_id,
                    template_id=template.template_id,
                    entity_type=entity_type,
                    entity_path=entity_path,
                    entity_label=label,
                    parent_entity_id=entity_ids_by_path.get(parent_path) if parent_path else None,
                    sort_order=sort_order,
                    source_json=dict(source),
                    review_status="pending",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )

        for index, material in enumerate(_iter_mapping_list(payload.get("materials"))):
            path = f"materials[{index}]"
            add_entity("materials", path, material, _material_label(material, index), index)

        for device_index, device in enumerate(_iter_mapping_list(payload.get("devices"))):
            device_path = f"devices[{device_index}]"
            add_entity(
                "devices", device_path, device, _device_label(device, device_index), device_index
            )
            for layer_index, layer in enumerate(_iter_mapping_list(device.get("layers"))):
                layer_path = f"{device_path}.layers[{layer_index}]"
                add_entity(
                    "layers",
                    layer_path,
                    layer,
                    _layer_label(layer, layer_index),
                    layer_index,
                    parent_path=device_path,
                )
                for component_index, component in enumerate(
                    _iter_mapping_list(layer.get("components"))
                ):
                    component_path = f"{layer_path}.components[{component_index}]"
                    add_entity(
                        "components",
                        component_path,
                        component,
                        _component_label(component, component_index),
                        component_index,
                        parent_path=layer_path,
                    )
            for metric_index, metric in enumerate(_iter_mapping_list(device.get("performance"))):
                metric_path = f"{device_path}.performance[{metric_index}]"
                add_entity(
                    "performance",
                    metric_path,
                    metric,
                    _performance_label(metric, metric_index),
                    metric_index,
                    parent_path=device_path,
                )

        return entities, entity_ids_by_path

    def _build_values(
        self,
        paper_id: str,
        candidate_run_id: str,
        template: DomainTemplate,
        payload: Mapping[str, Any],
        entity_ids_by_path: Mapping[str, str],
        anchor_ids_by_evidence_id: Mapping[str, str],
        timestamp: str,
    ) -> list[CandidateValue]:
        values: list[CandidateValue] = []
        for field in template.fields:
            if _skip_candidate_value_field(field):
                continue
            canonical_entity = _canonical_entity(field.entity)
            if canonical_entity is None:
                continue
            path_values, _ = resolve_field_values(payload, field.field_path)
            for path_value in path_values:
                if not path_value.exists or is_empty_value(path_value.value):
                    continue
                concrete_path = _strip_root(path_value.path)
                entity_path = _entity_path_for_value(canonical_entity, concrete_path)
                if entity_path is None:
                    continue
                entity_id = entity_ids_by_path.get(entity_path)
                if not entity_id:
                    continue
                values.append(
                    CandidateValue(
                        candidate_value_id=uuid4().hex,
                        candidate_run_id=candidate_run_id,
                        candidate_entity_id=entity_id,
                        paper_id=paper_id,
                        template_id=template.template_id,
                        template_field_path=field.field_path,
                        concrete_path=concrete_path,
                        field_label=field.label,
                        data_type=field.data_type,
                        value_json=path_value.value,
                        reviewed_value_json=None,
                        display_value=_display_value(path_value.value),
                        evidence_anchor_ids=_nearest_evidence_anchor_ids(
                            payload,
                            concrete_path,
                            anchor_ids_by_evidence_id,
                        ),
                        status="pending",
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )
        return values


def _final_json_from_candidate_values(
    mining_result: Mapping[str, Any],
    values: list[CandidateValue],
) -> dict[str, Any]:
    final_json = copy.deepcopy(dict(mining_result))
    for value in values:
        if value.status == "rejected":
            reviewed_value = None
        elif value.reviewed_value_json is not None:
            reviewed_value = value.reviewed_value_json
        else:
            reviewed_value = value.value_json
        _set_by_path(final_json, value.concrete_path, reviewed_value)
    _filter_materials_to_device_used(final_json)
    return final_json


ACCEPTED_MATERIAL_PROPERTY_STATUSES = {"accepted", "edited_accepted", "manual_added"}


def _accepted_material_property_payloads_by_material(
    candidates: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    payloads: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        status = getattr(candidate, "status", None)
        if status not in ACCEPTED_MATERIAL_PROPERTY_STATUSES:
            continue
        paper_material_id = getattr(candidate, "paper_material_id", None)
        if not isinstance(paper_material_id, str) or not paper_material_id.strip():
            continue
        payloads.setdefault(paper_material_id, []).append(
            _material_property_final_payload(candidate)
        )
    return payloads


def _material_property_final_payload(candidate: Any) -> dict[str, Any]:
    condition = getattr(candidate, "condition", {})
    evidence_anchor = getattr(candidate, "evidence_anchor", {})
    return _drop_none_values(
        {
            "source": "paper_reported",
            "review_status": getattr(candidate, "status", None),
            "property_candidate_id": getattr(candidate, "property_candidate_id", None),
            "candidate_run_id": getattr(candidate, "candidate_run_id", None),
            "paper_material_id": getattr(candidate, "paper_material_id", None),
            "global_material_id": getattr(candidate, "global_material_id", None),
            "property_name": getattr(candidate, "property_name", None),
            "property_category": getattr(candidate, "property_category", None),
            "value_numeric": getattr(candidate, "value_numeric", None),
            "value_text": getattr(candidate, "value_text", None),
            "value_raw": getattr(candidate, "value_raw", None),
            "unit": getattr(candidate, "unit", None),
            "normalized_value_numeric": getattr(
                candidate, "normalized_value_numeric", None
            ),
            "normalized_unit": getattr(candidate, "normalized_unit", None),
            "condition": dict(condition) if isinstance(condition, Mapping) else {},
            "method": getattr(candidate, "method", None),
            "source_type": getattr(candidate, "source_type", None),
            "evidence_text": getattr(candidate, "evidence_text", None),
            "llm_evidence_text": getattr(candidate, "llm_evidence_text", None),
            "source_block_text": getattr(candidate, "source_block_text", None),
            "evidence_anchor": dict(evidence_anchor)
            if isinstance(evidence_anchor, Mapping)
            else {},
            "provider": getattr(candidate, "provider", None),
            "model": getattr(candidate, "model", None),
            "prompt_version": getattr(candidate, "prompt_version", None),
            "confidence": getattr(candidate, "confidence", None),
            "updated_at": getattr(candidate, "updated_at", None),
        }
    )


def _apply_material_property_payloads(
    final_json: dict[str, Any],
    payloads_by_material_id: dict[str, list[dict[str, Any]]],
) -> None:
    materials = final_json.get("materials")
    if not isinstance(materials, list):
        return
    for material in materials:
        if not isinstance(material, dict):
            continue
        paper_material_id = material.get("paper_material_id")
        if not isinstance(paper_material_id, str):
            continue
        material.pop("reported_properties", None)
        payloads = payloads_by_material_id.get(paper_material_id)
        if payloads:
            material["reported_properties"] = payloads


STRUCTURE_IDENTIFIER_KEYS = (
    "smiles",
    "canonical_smiles",
    "isomeric_smiles",
    "raw_smiles",
    "inchi",
    "inchi_key",
)


def _material_structure_status_from_link(
    link: Any,
    candidate: Any | None,
    global_material: Any | None,
) -> dict[str, Any] | None:
    if link.match_status in {"matched_candidate", "matched_local"} and global_material:
        return _accepted_or_local_material_structure_status(
            link,
            candidate,
            global_material,
        )
    if link.match_status in {"identity_only", "out_of_scope_structure"}:
        scope = link.evidence.get("structure_scope")
        if not isinstance(scope, Mapping):
            scope = {
                "category": link.match_status,
                "requires_structure": False,
                "requires_public_resolution": False,
            }
        return {
            "material_payload": {
                "global_material_id": link.global_material_id,
                "material_structure_status": link.match_status,
                "structure_review_status": link.match_status,
                "structure_source": "structure_scope_rule",
                "material_structure_scope": dict(scope),
            },
            "reference_payload": {
                "global_material_id": link.global_material_id,
                "material_structure_status": link.match_status,
                "material_structure_scope": dict(scope),
            },
            "clear_structure_identifiers": True,
        }
    return None


def _accepted_or_local_material_structure_status(
    link: Any,
    candidate: Any | None,
    global_material: Any,
) -> dict[str, Any]:
    canonical_smiles = _first_text(
        getattr(candidate, "canonical_smiles", None),
        global_material.canonical_smiles,
    )
    isomeric_smiles = _first_text(
        getattr(candidate, "isomeric_smiles", None),
        global_material.isomeric_smiles,
    )
    raw_smiles = _first_text(getattr(candidate, "raw_smiles", None), global_material.raw_smiles)
    inchi = _first_text(getattr(candidate, "inchi", None), global_material.inchi)
    inchi_key = _first_text(getattr(candidate, "inchi_key", None), global_material.inchi_key)
    material_class = _first_text(
        getattr(candidate, "material_class", None),
        global_material.material_class,
    )
    representation_type = _first_text(
        getattr(candidate, "representation_type", None),
        global_material.representation_type,
    )
    structure_status = "accepted" if link.match_status == "matched_candidate" else "matched_local"
    structure_source = _first_text(getattr(candidate, "provider", None), global_material.source)
    structure_candidate_id = getattr(candidate, "structure_candidate_id", None)
    material_payload: dict[str, Any] = {
        "global_material_id": global_material.global_material_id,
        "material_structure_status": structure_status,
        "structure_review_status": "accepted"
        if link.match_status == "matched_candidate"
        else global_material.review_status,
        "structure_source": structure_source,
        "material_class": material_class,
        "representation_type": representation_type,
        "canonical_smiles": canonical_smiles,
        "isomeric_smiles": isomeric_smiles,
        "raw_smiles": raw_smiles,
        "smiles": canonical_smiles or isomeric_smiles or raw_smiles,
        "inchi": inchi,
        "inchi_key": inchi_key,
        "formula": _first_text(getattr(candidate, "formula", None), global_material.formula),
        "molecular_weight": _first_number(
            getattr(candidate, "molecular_weight", None),
            global_material.molecular_weight,
        ),
        "structure_candidate_id": structure_candidate_id,
        "structure_evidence": {
            "match_status": link.match_status,
            "match_method": link.match_method,
            "match_confidence": link.match_confidence,
            "provider": getattr(candidate, "provider", None),
            "resolver_name": getattr(candidate, "resolver_name", None),
            "source_identifier": getattr(candidate, "source_identifier", None),
            "source_url": getattr(candidate, "source_url", None),
        },
    }
    material_payload = _drop_none_values(material_payload)
    reference_payload = _drop_none_values(
        {
            "global_material_id": global_material.global_material_id,
            "material_structure_status": structure_status,
            "structure_candidate_id": structure_candidate_id,
            "structure_source": structure_source,
        }
    )
    return {
        "material_payload": material_payload,
        "reference_payload": reference_payload,
        "clear_structure_identifiers": False,
    }


def _apply_material_structure_statuses(
    final_json: dict[str, Any],
    statuses_by_material_id: dict[str, dict[str, Any]],
) -> None:
    materials = final_json.get("materials")
    if isinstance(materials, list):
        for material in materials:
            if not isinstance(material, dict):
                continue
            paper_material_id = material.get("paper_material_id")
            if not isinstance(paper_material_id, str):
                continue
            status = statuses_by_material_id.get(paper_material_id)
            if not status:
                continue
            if status.get("clear_structure_identifiers"):
                _clear_structure_identifiers(material)
            material.update(status["material_payload"])
    devices = final_json.get("devices")
    if isinstance(devices, list):
        for device in devices:
            if isinstance(device, dict):
                _apply_material_structure_to_device_references(
                    device,
                    statuses_by_material_id,
                )


def _apply_material_structure_to_device_references(
    device: dict[str, Any],
    statuses_by_material_id: dict[str, dict[str, Any]],
) -> None:
    final_emitter = device.get("final_emitter")
    if isinstance(final_emitter, dict):
        _apply_material_structure_reference(final_emitter, statuses_by_material_id)
    layers = device.get("layers")
    if not isinstance(layers, list):
        return
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        components = layer.get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if isinstance(component, dict):
                _apply_material_structure_reference(component, statuses_by_material_id)


def _apply_material_structure_reference(
    payload: dict[str, Any],
    statuses_by_material_id: dict[str, dict[str, Any]],
) -> None:
    paper_material_id = payload.get("paper_material_id")
    if not isinstance(paper_material_id, str):
        return
    status = statuses_by_material_id.get(paper_material_id)
    if not status:
        return
    payload.update(status["reference_payload"])


def _clear_structure_identifiers(payload: dict[str, Any]) -> None:
    for key in STRUCTURE_IDENTIFIER_KEYS:
        payload[key] = None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_number(*values: Any) -> int | float | None:
    for value in values:
        if isinstance(value, int | float):
            return value
    return None


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _filter_materials_to_device_used(final_json: dict[str, Any]) -> None:
    used_material_ids = _device_used_material_ids(final_json)
    if not used_material_ids:
        return
    materials = final_json.get("materials")
    if not isinstance(materials, list):
        return
    final_json["materials"] = [
        material
        for material in materials
        if isinstance(material, dict)
        and isinstance(material.get("paper_material_id"), str)
        and material["paper_material_id"] in used_material_ids
    ]


def _device_used_material_ids(payload: Mapping[str, Any]) -> set[str]:
    devices = payload.get("devices")
    if not isinstance(devices, list):
        return set()
    used: set[str] = set()
    for device in devices:
        _collect_paper_material_ids(device, used)
    return used


def _collect_paper_material_ids(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        material_id = value.get("paper_material_id")
        if isinstance(material_id, str) and material_id.strip():
            output.add(material_id.strip())
        for child in value.values():
            _collect_paper_material_ids(child, output)
        return
    if isinstance(value, list):
        for child in value:
            _collect_paper_material_ids(child, output)


def _set_by_path(payload: dict[str, Any], concrete_path: str, value: Any) -> None:
    current: Any = payload
    parts = concrete_path.split(".")
    for part in parts[:-1]:
        match = PATH_PART_RE.match(part)
        if not match:
            return
        key, index_text = match.groups()
        if not isinstance(current, Mapping) or key not in current:
            return
        current = current[key]
        if index_text is not None:
            if not isinstance(current, list):
                return
            index = int(index_text)
            if index >= len(current):
                return
            current = current[index]
    if not parts:
        return
    match = PATH_PART_RE.match(parts[-1])
    if not match:
        return
    key, index_text = match.groups()
    if index_text is None:
        if isinstance(current, dict):
            current[key] = value
        return
    if not isinstance(current, Mapping) or key not in current or not isinstance(current[key], list):
        return
    index = int(index_text)
    if index < len(current[key]):
        current[key][index] = value


def _validation_error_summary(result: object) -> str | None:
    errors = getattr(result, "errors", [])
    if not errors:
        return None
    return "; ".join(issue.message for issue in errors[:3])


def _normalize_review_status(status: str | None) -> str | None:
    if status is None:
        return None
    aliases = {
        "edited": "modified",
        "confirmed": "accepted",
    }
    normalized = aliases.get(status, status)
    allowed = {"pending", "modified", "accepted", "rejected"}
    return normalized if normalized in allowed else status


def _review_action(
    before: CandidateValue,
    after: CandidateValue,
    requested_status: str | None,
) -> str:
    if after.status == "rejected" or requested_status == "rejected":
        return "rejected"
    if after.status == "accepted" or requested_status == "accepted":
        return "accepted"
    if after.reviewed_value_json != before.reviewed_value_json:
        return "modified"
    return "saved"


def _action_for_status(status: str) -> str:
    if status == "accepted":
        return "accepted"
    if status == "rejected":
        return "rejected"
    if status == "modified":
        return "modified"
    return "saved"


def _skip_candidate_value_field(field: TemplateField) -> bool:
    return field.entity == "evidence" or field.field_path.endswith(".evidence_refs")


def _canonical_entity(entity: str) -> str | None:
    return ENTITY_ALIASES.get(entity)


def _entity_path_for_value(entity: str, concrete_path: str) -> str | None:
    pattern = ENTITY_PATH_PATTERNS.get(entity)
    if not pattern:
        return None
    match = pattern.match(concrete_path)
    return match.group(0) if match else None


def _nearest_evidence_anchor_ids(
    payload: Mapping[str, Any],
    concrete_path: str,
    anchor_ids_by_evidence_id: Mapping[str, str],
) -> list[str]:
    parts = concrete_path.split(".")
    for end_index in range(len(parts), 0, -1):
        candidate = _get_by_path(payload, ".".join(parts[:end_index]))
        if not isinstance(candidate, Mapping):
            continue
        refs = candidate.get("evidence_refs")
        if isinstance(refs, list):
            return [
                anchor_ids_by_evidence_id[ref] for ref in refs if ref in anchor_ids_by_evidence_id
            ]
    return []


def _get_by_path(payload: Any, concrete_path: str) -> Any:
    current = payload
    if concrete_path == "":
        return current
    for part in concrete_path.split("."):
        match = PATH_PART_RE.match(part)
        if not match:
            return None
        key, index_text = match.groups()
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
        if index_text is not None:
            if not isinstance(current, list):
                return None
            index = int(index_text)
            if index >= len(current):
                return None
            current = current[index]
    return current


def _iter_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _strip_root(path: str) -> str:
    return path[2:] if path.startswith("$.") else path


def _display_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _material_label(material: Mapping[str, Any], index: int) -> str:
    for key in ("normalized_name", "abbreviation", "paper_material_id", "canonical_name"):
        value = material.get(key)
        if isinstance(value, str) and value:
            return value
    return f"Material {index + 1}"


def _device_label(device: Mapping[str, Any], index: int) -> str:
    value = device.get("device_label")
    return value if isinstance(value, str) and value else f"Device {index + 1}"


def _layer_label(layer: Mapping[str, Any], index: int) -> str:
    layer_role = layer.get("layer_role")
    layer_name = layer.get("layer_name")
    label_parts = [f"Layer {index + 1}"]
    if isinstance(layer_role, str) and layer_role:
        label_parts.append(layer_role)
    if isinstance(layer_name, str) and layer_name:
        label_parts.append(layer_name)
    return ": ".join(label_parts[:2]) if len(label_parts) == 2 else " - ".join(label_parts)


def _component_label(component: Mapping[str, Any], index: int) -> str:
    material = component.get("material_mention") or component.get("paper_material_id")
    role = component.get("component_role")
    if isinstance(material, str) and material and isinstance(role, str) and role:
        return f"{material} ({role})"
    if isinstance(material, str) and material:
        return material
    return f"Component {index + 1}"


def _performance_label(metric: Mapping[str, Any], index: int) -> str:
    metric_name = metric.get("metric_name")
    if isinstance(metric_name, str) and metric_name:
        return metric_name
    metric_family = metric.get("metric_family")
    statistic = metric.get("statistic")
    if (
        isinstance(metric_family, str)
        and metric_family
        and isinstance(statistic, str)
        and statistic
    ):
        return f"{metric_family} {statistic}"
    return f"Performance {index + 1}"
