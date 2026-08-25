from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import re
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_chemistry import StandardizedSmiles, standardize_smiles
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialManualStructureAction,
    PaperMaterialStructureBundle,
)


COORDINATION_COMPLEX_CLASSES = {
    "organometallic_complex",
    "coordination_complex",
    "metal_complex",
    "transition_metal_complex",
}
MULTI_COMPONENT_CLASSES = {"salt", "mixture", "composite"}
COORDINATION_METAL_ELEMENTS = {
    "Ag",
    "Al",
    "Au",
    "Be",
    "Ca",
    "Cd",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Ga",
    "Hf",
    "Hg",
    "In",
    "Ir",
    "K",
    "Li",
    "Mg",
    "Mn",
    "Na",
    "Ni",
    "Os",
    "Pb",
    "Pd",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "Sn",
    "Ti",
    "Zn",
    "Zr",
}
_FORMULA_ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


class LiteratureVerifiedStructureRecord(BaseModel):
    paper_id: str
    doi: str
    doi_url: str | None = None
    paper_material_id: str
    primary_material_name: str
    full_name_in_paper: str | None = None
    resolved_full_name: str | None = None
    material_class: str = "unknown"
    smiles: str
    evidence_level: str
    evidence_status: str
    source_references: list[str] = Field(default_factory=list)
    reported_formula: str | None = None
    reported_molecular_weight: float | str | None = None
    rdkit_validation: str | None = None
    workbook_path: str
    workbook_sha256: str
    main_sheet_row: int
    evidence_sheet_row: int


class LiteratureVerifiedImportDecision(BaseModel):
    paper_id: str
    doi: str
    paper_material_id: str
    primary_material_name: str
    evidence_level: str
    action: str
    reason: str
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    calculated_formula: str | None = None
    existing_candidate_ids: list[str] = Field(default_factory=list)
    existing_structure_keys: list[str] = Field(default_factory=list)
    imported_candidate_id: str | None = None
    global_material_id: str | None = None
    error: str | None = None


class LiteratureVerifiedImportReport(BaseModel):
    mode: str
    generated_at: str
    source_workbook: str
    source_workbook_sha256: str
    record_count: int
    counts: dict[str, int]
    affected_paper_count: int = 0
    decisions: list[LiteratureVerifiedImportDecision] = Field(default_factory=list)


class LiteratureVerifiedImportService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.resolution = MaterialResolutionService(config)
        self.review = MaterialStructureReviewService(config)

    def run(
        self,
        records: list[LiteratureVerifiedStructureRecord],
        *,
        apply: bool = False,
        actor: str = "literature_verified_excel_import",
    ) -> LiteratureVerifiedImportReport:
        decisions = self.assess(records)
        affected_papers: set[str] = set()
        if apply:
            record_by_key = {
                (record.paper_id, record.paper_material_id): record for record in records
            }
            for index, decision in enumerate(decisions):
                if decision.action != "import":
                    continue
                record = record_by_key[(decision.paper_id, decision.paper_material_id)]
                current = self._assess_record(record)
                if current.action != "import":
                    decisions[index] = current
                    continue
                try:
                    standardized = standardize_smiles(record.smiles)
                    self.review.save_manual_structure(
                        record.paper_id,
                        record.paper_material_id,
                        MaterialManualStructureAction(
                            actor=actor,
                            message=(
                                "Accepted literature-verified structure from supplemental "
                                f"workbook; evidence level {record.evidence_level}."
                            ),
                            reviewed_name=(
                                record.resolved_full_name or record.primary_material_name
                            ),
                            full_name_in_paper=record.full_name_in_paper,
                            material_class=record.material_class or "small_molecule_organic",
                            representation_type=material_representation_type(
                                record,
                                standardized,
                            ),
                            smiles=record.smiles,
                            source_url=_primary_source_url(record),
                            source_note=_source_note(record),
                        ),
                        defer_completion=True,
                        return_bundle=False,
                    )
                    affected_papers.add(record.paper_id)
                    decisions[index] = current.model_copy(
                        update={
                            "action": "imported",
                            "reason": (
                                "Literature-verified structure was accepted through the normal "
                                "material review service."
                            ),
                        }
                    )
                except Exception as exc:
                    decisions[index] = current.model_copy(
                        update={"action": "failed", "reason": "Import failed.", "error": str(exc)}
                    )

            for paper_id in sorted(affected_papers):
                self.review.material_completion.confirm_paper_if_materials_complete(paper_id)
            self._attach_imported_ids(decisions)

        counts = Counter(decision.action for decision in decisions)
        first_record = records[0] if records else None
        return LiteratureVerifiedImportReport(
            mode="apply" if apply else "dry_run",
            generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            source_workbook=first_record.workbook_path if first_record else "",
            source_workbook_sha256=first_record.workbook_sha256 if first_record else "",
            record_count=len(records),
            counts=dict(sorted(counts.items())),
            affected_paper_count=len(affected_papers),
            decisions=decisions,
        )

    def assess(
        self,
        records: list[LiteratureVerifiedStructureRecord],
    ) -> list[LiteratureVerifiedImportDecision]:
        bundles: dict[str, PaperMaterialStructureBundle | None] = {}
        decisions: list[LiteratureVerifiedImportDecision] = []
        for record in records:
            if record.paper_id not in bundles:
                bundles[record.paper_id] = self.resolution.get_material_structure_bundle(
                    record.paper_id
                )
            bundle = bundles[record.paper_id]
            decisions.append(self._assess_record(record, bundle=bundle))
        return decisions

    def _assess_record(
        self,
        record: LiteratureVerifiedStructureRecord,
        *,
        bundle: PaperMaterialStructureBundle | None = None,
    ) -> LiteratureVerifiedImportDecision:
        base = {
            "paper_id": record.paper_id,
            "doi": record.doi,
            "paper_material_id": record.paper_material_id,
            "primary_material_name": record.primary_material_name,
            "evidence_level": record.evidence_level,
        }
        try:
            standardized = standardize_smiles(record.smiles)
        except ValueError as exc:
            return LiteratureVerifiedImportDecision(
                **base,
                action="invalid_smiles",
                reason="SMILES failed the platform RDKit standardization step.",
                error=str(exc),
            )
        structure_fields = _standardized_fields(standardized)
        current_bundle = bundle or self.resolution.get_material_structure_bundle(record.paper_id)
        if current_bundle is None or not current_bundle.candidate_run_id:
            return LiteratureVerifiedImportDecision(
                **base,
                **structure_fields,
                action="missing_paper_bundle",
                reason="No completed material bundle exists for this paper.",
            )
        material = next(
            (
                item
                for item in current_bundle.materials
                if item.paper_material_id == record.paper_material_id
            ),
            None,
        )
        if material is None:
            return LiteratureVerifiedImportDecision(
                **base,
                **structure_fields,
                action="missing_paper_material",
                reason="The current candidate run does not contain this paper_material_id.",
            )

        candidates = [
            candidate
            for candidate in current_bundle.structure_candidates
            if candidate.paper_material_id == record.paper_material_id
        ]
        accepted_candidates = [
            candidate
            for candidate in candidates
            if candidate.status == "accepted"
            and (candidate.canonical_smiles or candidate.inchi_key)
        ]
        globals_by_id = {
            material.global_material_id: material for material in current_bundle.global_materials
        }
        linked_globals = [
            globals_by_id[link.global_material_id]
            for link in current_bundle.links
            if link.paper_material_id == record.paper_material_id
            and link.global_material_id in globals_by_id
            and (
                globals_by_id[link.global_material_id].canonical_smiles
                or globals_by_id[link.global_material_id].inchi_key
            )
        ]
        resolved_structures: list[Any] = [*accepted_candidates, *linked_globals]
        existing_ids = [candidate.structure_candidate_id for candidate in accepted_candidates]
        existing_keys = sorted(
            {
                key
                for item in resolved_structures
                for key in [getattr(item, "inchi_key", None)]
                if key
            }
        )
        if any(_structure_matches(item, standardized) for item in resolved_structures):
            return LiteratureVerifiedImportDecision(
                **base,
                **structure_fields,
                action="skip_already_resolved_same",
                reason="The platform already has the same accepted or globally linked structure.",
                existing_candidate_ids=existing_ids,
                existing_structure_keys=existing_keys,
            )
        if resolved_structures:
            return LiteratureVerifiedImportDecision(
                **base,
                **structure_fields,
                action="skip_resolved_conflict",
                reason=(
                    "The literature-verified structure differs from an existing accepted or "
                    "globally linked structure; automatic replacement is disabled."
                ),
                existing_candidate_ids=existing_ids,
                existing_structure_keys=existing_keys,
            )
        same_pending = [
            candidate.structure_candidate_id
            for candidate in candidates
            if candidate.status != "rejected" and _structure_matches(candidate, standardized)
        ]
        return LiteratureVerifiedImportDecision(
            **base,
            **structure_fields,
            action="import",
            reason=(
                "A matching unaccepted candidate exists; the verified workbook will create an "
                "independent accepted provenance record."
                if same_pending
                else "No accepted structure exists and this record is safe to import."
            ),
            existing_candidate_ids=same_pending,
        )

    def _attach_imported_ids(
        self,
        decisions: list[LiteratureVerifiedImportDecision],
    ) -> None:
        imported_by_paper: dict[str, list[int]] = {}
        for index, decision in enumerate(decisions):
            if decision.action == "imported":
                imported_by_paper.setdefault(decision.paper_id, []).append(index)
        for paper_id, indexes in imported_by_paper.items():
            bundle = self.resolution.get_material_structure_bundle(paper_id)
            if bundle is None:
                continue
            links = {link.paper_material_id: link for link in bundle.links}
            for index in indexes:
                decision = decisions[index]
                candidates = [
                    candidate
                    for candidate in bundle.structure_candidates
                    if candidate.paper_material_id == decision.paper_material_id
                    and candidate.provider == "manual_input"
                    and candidate.status == "accepted"
                    and candidate.inchi_key == decision.inchi_key
                ]
                if candidates:
                    latest = max(candidates, key=lambda item: (item.updated_at, item.created_at))
                    decision.imported_candidate_id = latest.structure_candidate_id
                link = links.get(decision.paper_material_id)
                if link:
                    decision.global_material_id = link.global_material_id


def material_representation_type(
    record: LiteratureVerifiedStructureRecord,
    standardized: StandardizedSmiles,
) -> str:
    normalized_class = record.material_class.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_class in MULTI_COMPONENT_CLASSES or "." in standardized.canonical_smiles:
        return "multi_component"
    if "polymer" in normalized_class:
        return "polymer"
    formula_elements = set(_FORMULA_ELEMENT_PATTERN.findall(standardized.formula or ""))
    if normalized_class in COORDINATION_COMPLEX_CLASSES or (
        formula_elements & COORDINATION_METAL_ELEMENTS
    ):
        return "coordination_complex"
    return "small_molecule"


def _standardized_fields(standardized: StandardizedSmiles) -> dict[str, str | None]:
    return {
        "canonical_smiles": standardized.canonical_smiles,
        "inchi_key": standardized.inchi_key,
        "calculated_formula": standardized.formula,
    }


def _structure_matches(item: Any, standardized: StandardizedSmiles) -> bool:
    item_key = getattr(item, "inchi_key", None)
    if item_key and standardized.inchi_key:
        return item_key == standardized.inchi_key
    item_smiles = getattr(item, "canonical_smiles", None)
    return bool(item_smiles and item_smiles == standardized.canonical_smiles)


def _primary_source_url(record: LiteratureVerifiedStructureRecord) -> str | None:
    for reference in record.source_references:
        if reference.startswith(("http://", "https://")):
            return reference
    return record.doi_url


def _source_note(record: LiteratureVerifiedStructureRecord) -> str:
    return json.dumps(
        {
            "import_type": "literature_verified_excel",
            "evidence_level": record.evidence_level,
            "evidence_status": record.evidence_status,
            "resolved_full_name": record.resolved_full_name,
            "source_references": record.source_references,
            "reported_formula": record.reported_formula,
            "reported_molecular_weight": record.reported_molecular_weight,
            "rdkit_validation": record.rdkit_validation,
            "workbook_path": record.workbook_path,
            "workbook_sha256": record.workbook_sha256,
            "main_sheet_row": record.main_sheet_row,
            "evidence_sheet_row": record.evidence_sheet_row,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
