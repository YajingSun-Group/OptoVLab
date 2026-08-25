from __future__ import annotations

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    classify_material_structure_scope,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialStructureTriageItem,
    MaterialStructureTriageResult,
    PaperMaterialStructureBundle,
    PaperMaterialLink,
)


STRUCTURE_READY_LINK_STATUSES = {"matched_local", "matched_candidate"}
STRUCTURE_SKIP_LINK_STATUSES = {"identity_only", "out_of_scope_structure"}


class MaterialStructureTriageService:
    """Decide whether paper materials still need visual OCSR.

    This service is intentionally cheap. It reuses the resolved paper-material bundle,
    existing scope rules, local/public accepted links, and accepted structure candidates.
    It does not call LLM/VLM.
    """

    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.material_resolution = MaterialResolutionService(config)

    def init_runtime(self) -> None:
        self.material_resolution.init_runtime()

    def triage_paper(self, paper_id: str) -> MaterialStructureTriageResult | None:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None:
            return None
        return self.triage_bundle(bundle)

    def triage_bundle(
        self,
        bundle: PaperMaterialStructureBundle,
    ) -> MaterialStructureTriageResult:
        """Triage an already-loaded bundle without repeating the repository fan-out."""
        links_by_material = {link.paper_material_id: link for link in bundle.links}
        accepted_material_ids = {
            candidate.paper_material_id
            for candidate in bundle.structure_candidates
            if candidate.status == "accepted"
            and bool(candidate.canonical_smiles or candidate.inchi_key)
        }
        items: list[MaterialStructureTriageItem] = []
        for material in bundle.materials:
            link = links_by_material.get(material.paper_material_id)
            scope = classify_material_structure_scope(material)
            has_ready_link = _link_has_ready_structure(link)
            has_accepted = material.paper_material_id in accepted_material_ids
            link_status = link.match_status if link else None
            has_accepted_or_matched_structure = has_ready_link or has_accepted
            should_run_ocsr = (
                bool(scope.get("requires_structure"))
                and not has_accepted_or_matched_structure
                and link_status not in STRUCTURE_SKIP_LINK_STATUSES
            )
            items.append(
                MaterialStructureTriageItem(
                    paper_material_id=material.paper_material_id,
                    material_label=material.entity_label,
                    category=str(scope.get("category") or "unknown"),
                    requires_structure=bool(scope.get("requires_structure", True)),
                    requires_public_resolution=bool(scope.get("requires_public_resolution", True)),
                    has_accepted_or_matched_structure=has_accepted_or_matched_structure,
                    should_run_ocsr=should_run_ocsr,
                    confidence=_float_or_none(scope.get("confidence")),
                    reason=str(scope.get("reason") or "") or None,
                    rule=str(scope.get("rule") or "") or None,
                    link_status=link_status,
                    matched_alias=(
                        str(scope.get("matched_alias"))
                        if scope.get("matched_alias") is not None
                        else None
                    ),
                )
            )
        needs_ocsr_count = sum(1 for item in items if item.should_run_ocsr)
        return MaterialStructureTriageResult(
            paper_id=bundle.paper_id,
            candidate_run_id=bundle.candidate_run_id,
            material_count=len(items),
            needs_ocsr_count=needs_ocsr_count,
            skipped_count=len(items) - needs_ocsr_count,
            items=items,
        )

    def paper_needs_ocsr(self, paper_id: str) -> bool:
        result = self.triage_paper(paper_id)
        return bool(result and result.should_run_ocsr)


def _link_has_ready_structure(link: PaperMaterialLink | None) -> bool:
    if link is None:
        return False
    if link.match_status not in STRUCTURE_READY_LINK_STATUSES:
        return False
    evidence = link.evidence or {}
    structure_scope = evidence.get("structure_scope")
    if isinstance(structure_scope, dict) and not structure_scope.get("requires_structure", True):
        return False
    return bool(link.global_material_id)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
