from __future__ import annotations

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialGlobal,
    PaperMaterialStructureBundle,
    MaterialResolutionTask,
    MaterialStructureCandidate,
    PaperMaterialLink,
)
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    MaterialResolutionTaskRepository,
    PaperRepository,
)


NON_STRUCTURE_COMPLETE_STATUSES = {"identity_only", "out_of_scope_structure"}
STRUCTURE_COMPLETE_STATUSES = {"matched_local", "matched_candidate"}
NON_STRUCTURE_COMPLETE_STRATEGIES = {"structure_unavailable_proprietary"}


class PaperMaterialCompletionService:
    """Confirm a paper when all device-used materials are structurally resolved.

    This is intentionally a review status shortcut only. It marks the paper as
    confirmed for batch navigation when material structure review no longer needs
    human intervention; it does not create final device records.
    """

    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.papers = PaperRepository(self.database)
        self.tasks = MaterialResolutionTaskRepository(self.database)
        self.material_resolution = MaterialResolutionService(config)

    def init_runtime(self) -> None:
        self.material_resolution.init_runtime()

    def confirm_paper_if_materials_complete(self, paper_id: str) -> bool:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None or not bundle.materials:
            return False
        if not self._bundle_materials_complete(bundle):
            return False
        paper = self.papers.get(bundle.paper_id)
        if paper and paper.review_status != "confirmed":
            self.papers.set_review_status(bundle.paper_id, "confirmed")
        return True

    def confirm_papers_if_materials_complete(
        self,
        paper_ids: list[str],
    ) -> dict[str, bool]:
        """Evaluate each paper once and commit all status changes in one write batch."""
        self.init_runtime()
        results: dict[str, bool] = {}
        to_confirm: list[str] = []
        for paper_id in dict.fromkeys(paper_ids):
            bundle = self.material_resolution.get_material_structure_bundle(paper_id)
            complete = bool(bundle and bundle.materials and self._bundle_materials_complete(bundle))
            results[paper_id] = complete
            if not complete or bundle is None:
                continue
            paper = self.papers.get(bundle.paper_id)
            if paper and paper.review_status != "confirmed":
                to_confirm.append(bundle.paper_id)
        self.papers.set_review_status_many(to_confirm, "confirmed")
        return results

    def paper_materials_complete(self, paper_id: str) -> bool:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None or not bundle.materials:
            return False
        return self._bundle_materials_complete(bundle)

    def _bundle_materials_complete(self, bundle: PaperMaterialStructureBundle) -> bool:
        links_by_material = {link.paper_material_id: link for link in bundle.links}
        globals_by_id = {
            material.global_material_id: material for material in bundle.global_materials
        }
        accepted_by_material = {
            candidate.paper_material_id: candidate
            for candidate in bundle.structure_candidates
            if _accepted_candidate_has_structure(candidate)
        }
        tasks_by_material = {task.paper_material_id: task for task in bundle.tasks}
        return all(
            _material_is_complete(
                link=links_by_material.get(material.paper_material_id),
                global_materials=globals_by_id,
                accepted_candidate=accepted_by_material.get(material.paper_material_id),
                task=tasks_by_material.get(material.paper_material_id),
            )
            for material in bundle.materials
        )


def _material_is_complete(
    *,
    link: PaperMaterialLink | None,
    global_materials: dict[str, MaterialGlobal],
    accepted_candidate: MaterialStructureCandidate | None,
    task: MaterialResolutionTask | None,
) -> bool:
    if link and link.match_status in NON_STRUCTURE_COMPLETE_STATUSES:
        return True
    if accepted_candidate:
        return True
    if (
        task
        and task.status == "completed"
        and task.assigned_strategy in NON_STRUCTURE_COMPLETE_STRATEGIES
    ):
        return True
    if not link or link.match_status not in STRUCTURE_COMPLETE_STATUSES:
        return False
    if not link.global_material_id:
        return False
    global_material = global_materials.get(link.global_material_id)
    return bool(global_material and _global_material_has_structure(global_material))


def _accepted_candidate_has_structure(candidate: MaterialStructureCandidate) -> bool:
    return candidate.status == "accepted" and bool(
        candidate.canonical_smiles or candidate.inchi_key
    )


def _global_material_has_structure(material: MaterialGlobal) -> bool:
    return bool(material.canonical_smiles or material.inchi_key)
