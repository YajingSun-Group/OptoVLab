from __future__ import annotations

from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_property_mining_service import PROPERTY_CATEGORY_BY_NAME
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialPropertyCandidate,
    MaterialPropertyManualAddAction,
    MaterialPropertyReview,
    MaterialPropertyReviewAction,
    MaterialPropertyReviewEvent,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialPropertyCandidateRepository,
    MaterialPropertyReviewEventRepository,
    MaterialPropertyReviewRepository,
)


class MaterialPropertyReviewService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.material_resolution = MaterialResolutionService(config)
        self.property_candidates = MaterialPropertyCandidateRepository(self.database)
        self.property_reviews = MaterialPropertyReviewRepository(self.database)
        self.property_review_events = MaterialPropertyReviewEventRepository(self.database)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def accept_candidate(
        self,
        property_candidate_id: str,
        action: MaterialPropertyReviewAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        candidate = self.property_candidates.get(property_candidate_id)
        if not candidate:
            return None
        updated = self.property_candidates.update_status(property_candidate_id, "accepted")
        after = updated or candidate.model_copy(update={"status": "accepted"})
        self._record_review_and_event(
            before=candidate,
            after=after,
            decision="accept",
            event_type="accept",
            action=action,
        )
        return self.material_resolution.get_material_structure_bundle(candidate.paper_id)

    def edit_candidate(
        self,
        property_candidate_id: str,
        action: MaterialPropertyReviewAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        candidate = self.property_candidates.get(property_candidate_id)
        if not candidate:
            return None
        timestamp = now_iso()
        property_name = _reviewed_property_name(candidate, action)
        value_numeric = (
            action.reviewed_value_numeric
            if action.reviewed_value_numeric is not None
            else candidate.value_numeric
        )
        unit = action.reviewed_unit if action.reviewed_unit is not None else candidate.unit
        normalized_value_numeric, normalized_unit = _normalized_value(property_name, value_numeric, unit)
        updated = self.property_candidates.upsert(
            candidate.model_copy(
                update={
                    "property_name": property_name,
                    "property_category": PROPERTY_CATEGORY_BY_NAME.get(
                        property_name, candidate.property_category
                    ),
                    "value_numeric": value_numeric,
                    "value_text": (
                        action.reviewed_value_text
                        if action.reviewed_value_text is not None
                        else candidate.value_text
                    ),
                    "value_raw": (
                        action.reviewed_value_raw
                        if action.reviewed_value_raw is not None
                        else candidate.value_raw
                    ),
                    "unit": unit,
                    "normalized_value_numeric": normalized_value_numeric,
                    "normalized_unit": normalized_unit,
                    "condition": action.reviewed_condition or candidate.condition,
                    "evidence_anchor": action.reviewed_evidence_anchor or candidate.evidence_anchor,
                    "status": "edited_accepted",
                    "updated_at": timestamp,
                }
            )
        )
        self._record_review_and_event(
            before=candidate,
            after=updated,
            decision="edit_accept",
            event_type="edit_accept",
            action=action,
        )
        return self.material_resolution.get_material_structure_bundle(candidate.paper_id)

    def reject_candidate(
        self,
        property_candidate_id: str,
        action: MaterialPropertyReviewAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        candidate = self.property_candidates.get(property_candidate_id)
        if not candidate:
            return None
        updated = self.property_candidates.update_status(property_candidate_id, "rejected")
        after = updated or candidate.model_copy(update={"status": "rejected"})
        self._record_review_and_event(
            before=candidate,
            after=after,
            decision="reject",
            event_type="reject",
            action=action,
        )
        return self.material_resolution.get_material_structure_bundle(candidate.paper_id)

    def add_manual_property(
        self,
        paper_id: str,
        paper_material_id: str,
        action: MaterialPropertyManualAddAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if not bundle:
            return None
        if not bundle.candidate_run_id:
            return bundle
        if not any(material.paper_material_id == paper_material_id for material in bundle.materials):
            raise ValueError(f"Paper material not found: {paper_material_id}")
        property_name = _normalize_property_name(action.property_name)
        timestamp = now_iso()
        normalized_value_numeric, normalized_unit = _normalized_value(
            property_name, action.value_numeric, action.unit
        )
        candidate = self.property_candidates.upsert(
            MaterialPropertyCandidate(
                property_candidate_id=f"prop_manual_{uuid4().hex}",
                paper_id=bundle.paper_id,
                candidate_run_id=bundle.candidate_run_id,
                paper_material_id=paper_material_id,
                property_name=property_name,
                property_category=action.property_category
                or PROPERTY_CATEGORY_BY_NAME.get(property_name, "unknown"),
                value_numeric=action.value_numeric,
                value_text=action.value_text,
                value_raw=action.value_raw,
                unit=action.unit,
                normalized_value_numeric=(
                    action.normalized_value_numeric
                    if action.normalized_value_numeric is not None
                    else normalized_value_numeric
                ),
                normalized_unit=action.normalized_unit or normalized_unit,
                condition=action.condition,
                method=action.method,
                source_type=action.source_type,
                evidence_text=action.evidence_text,
                evidence_anchor=action.evidence_anchor,
                provider="manual_input",
                model=None,
                prompt_version="manual_property_input_v1",
                confidence=action.confidence,
                status="manual_added",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        self._record_review_and_event(
            before=None,
            after=candidate,
            decision="manual_add",
            event_type="manual_add",
            action=MaterialPropertyReviewAction(
                actor=action.actor,
                message=action.message,
                reviewed_property_name=candidate.property_name,
                reviewed_value_numeric=candidate.value_numeric,
                reviewed_value_text=candidate.value_text,
                reviewed_value_raw=candidate.value_raw,
                reviewed_unit=candidate.unit,
                reviewed_condition=candidate.condition,
                reviewed_evidence_anchor=candidate.evidence_anchor,
            ),
        )
        return self.material_resolution.get_material_structure_bundle(bundle.paper_id)

    def _record_review_and_event(
        self,
        *,
        before: MaterialPropertyCandidate | None,
        after: MaterialPropertyCandidate,
        decision: str,
        event_type: str,
        action: MaterialPropertyReviewAction,
    ) -> None:
        timestamp = now_iso()
        self.property_reviews.add(
            MaterialPropertyReview(
                review_id=uuid4().hex,
                property_candidate_id=after.property_candidate_id,
                paper_id=after.paper_id,
                candidate_run_id=after.candidate_run_id,
                paper_material_id=after.paper_material_id,
                decision=decision,
                reviewed_property_name=after.property_name,
                reviewed_value_numeric=after.value_numeric,
                reviewed_value_text=after.value_text,
                reviewed_unit=after.unit,
                reviewed_condition=after.condition,
                reviewed_evidence_anchor=after.evidence_anchor,
                actor=action.actor,
                message=action.message,
                created_at=timestamp,
            )
        )
        self.property_review_events.add(
            MaterialPropertyReviewEvent(
                event_id=uuid4().hex,
                paper_id=after.paper_id,
                candidate_run_id=after.candidate_run_id,
                paper_material_id=after.paper_material_id,
                property_candidate_id=after.property_candidate_id,
                event_type=event_type,
                before=before.model_dump(mode="json") if before else None,
                after=after.model_dump(mode="json"),
                actor=action.actor,
                message=action.message,
                created_at=timestamp,
            )
        )


def _reviewed_property_name(
    candidate: MaterialPropertyCandidate,
    action: MaterialPropertyReviewAction,
) -> str:
    if not action.reviewed_property_name:
        return candidate.property_name
    return _normalize_property_name(action.reviewed_property_name)


def _normalize_property_name(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Property name is required.")
    if stripped in PROPERTY_CATEGORY_BY_NAME:
        return stripped
    lowered = stripped.lower()
    for candidate in PROPERTY_CATEGORY_BY_NAME:
        if candidate.lower() == lowered:
            return candidate
    raise ValueError(f"Unsupported material property name: {value}")


def _normalized_value(
    property_name: str,
    value_numeric: float | None,
    unit: str | None,
) -> tuple[float | None, str | None]:
    if value_numeric is None or unit is None:
        return None, None
    normalized_unit = unit.strip()
    if property_name == "PLQY" and normalized_unit == "%":
        return value_numeric / 100.0, "fraction"
    return value_numeric, normalized_unit
