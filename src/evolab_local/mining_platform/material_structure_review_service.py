from __future__ import annotations

from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_completion_service import PaperMaterialCompletionService
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    has_confirmed_alias_conflict,
    is_reusable_material_alias_text,
    normalize_material_alias,
)
from evolab_local.mining_platform.material_chemistry import smiles_depiction_svg, standardize_smiles
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialAlias,
    MaterialGlobal,
    MaterialManualStructureAction,
    MaterialReviewAction,
    MaterialReviewEvent,
    MaterialResolutionTask,
    MaterialStructureCandidate,
    MaterialStructureEditAction,
    PaperLocalMaterial,
    PaperMaterialLink,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialAliasRepository,
    MaterialGlobalRepository,
    MaterialIdentityJudgmentRepository,
    MaterialIdentityEvidenceItemRepository,
    MaterialResolutionTaskRepository,
    MaterialReviewEventRepository,
    MaterialStructureCandidateRepository,
    PaperMaterialLinkRepository,
)


class MaterialStructureReviewService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.material_resolution = MaterialResolutionService(config)
        self.global_materials = MaterialGlobalRepository(self.database)
        self.aliases = MaterialAliasRepository(self.database)
        self.links = PaperMaterialLinkRepository(self.database)
        self.tasks = MaterialResolutionTaskRepository(self.database)
        self.structure_candidates = MaterialStructureCandidateRepository(self.database)
        self.identity_judgments = MaterialIdentityJudgmentRepository(self.database)
        self.identity_evidence_items = MaterialIdentityEvidenceItemRepository(self.database)
        self.material_review_events = MaterialReviewEventRepository(self.database)
        self.material_completion = PaperMaterialCompletionService(config)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def accept_structure_candidate(
        self,
        structure_candidate_id: str,
        action: MaterialReviewAction,
        *,
        paper_material: PaperLocalMaterial | None = None,
        defer_completion: bool = False,
        return_bundle: bool = True,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        candidate = self.structure_candidates.get(structure_candidate_id)
        if not candidate:
            return None
        if not candidate.canonical_smiles:
            raise ValueError("Candidate SMILES must be corrected and validated before acceptance.")
        identity_judgment = self.identity_judgments.latest_by_candidate(structure_candidate_id)
        if candidate.resolver_name == "anysearch_to_pubchem" and not identity_judgment:
            raise ValueError(
                "Candidates discovered through web search require Material Identity Judge review "
                "before acceptance."
            )
        if identity_judgment and identity_judgment.verdict in {"conflict", "rejected"}:
            raise ValueError(
                "Candidate identity is blocked by the latest Material Identity Judge result: "
                f"{identity_judgment.verdict}. Resolve or rerun identity judgment before acceptance."
            )
        if identity_judgment and identity_judgment.recommended_action == "search_more_evidence":
            raise ValueError(
                "Candidate identity requires recorded supporting evidence before acceptance. "
                "Run evidence enrichment and review the resulting source."
            )
        if candidate.resolver_name.startswith("identity_evidence_"):
            evidence_item_ids = candidate.evidence.get("identity_evidence_item_ids", [])
            confirmed_item = any(
                item and item.review_status == "confirmed"
                for evidence_item_id in evidence_item_ids
                if isinstance(evidence_item_id, str)
                for item in [self.identity_evidence_items.get(evidence_item_id)]
            )
            if not confirmed_item:
                raise ValueError(
                    "Candidates generated from enrichment evidence require a confirmed "
                    "evidence source before acceptance."
                )
        if paper_material is None:
            bundle = self.material_resolution.get_material_structure_bundle(candidate.paper_id)
            paper_material = (
                next(
                    (
                        material
                        for material in bundle.materials
                        if material.paper_material_id == candidate.paper_material_id
                    ),
                    None,
                )
                if bundle
                else None
            )
        before_link = self.links.get_by_paper_material(
            candidate.candidate_run_id,
            candidate.paper_material_id,
        )
        before_task = self.tasks.get_by_paper_material(
            candidate.candidate_run_id,
            candidate.paper_material_id,
        )
        material, created_global_id = self._global_material_for_candidate(candidate, action)
        self._add_aliases_for_candidate(material, candidate, paper_material)
        updated_candidate = self.structure_candidates.set_status(
            structure_candidate_id,
            "accepted",
        )
        timestamp = now_iso()
        after_link = self.links.upsert(
            PaperMaterialLink(
                paper_material_link_id=uuid4().hex,
                paper_id=candidate.paper_id,
                candidate_run_id=candidate.candidate_run_id,
                paper_material_id=candidate.paper_material_id,
                global_material_id=material.global_material_id,
                match_method=f"{candidate.provider}_candidate",
                match_confidence=candidate.confidence,
                match_status="matched_candidate",
                evidence={
                    "structure_candidate_id": candidate.structure_candidate_id,
                    "provider": candidate.provider,
                    "source_identifier": candidate.source_identifier,
                    "source_url": candidate.source_url,
                    "query_text": candidate.query_text,
                },
                created_at=timestamp,
                updated_at=timestamp,
                confirmed_at=timestamp,
            )
        )
        after_task = self._completed_task_for_candidate(candidate, before_task)
        self.material_review_events.add(
            MaterialReviewEvent(
                event_id=uuid4().hex,
                paper_id=candidate.paper_id,
                candidate_run_id=candidate.candidate_run_id,
                paper_material_id=candidate.paper_material_id,
                structure_candidate_id=candidate.structure_candidate_id,
                global_material_id=material.global_material_id,
                action="accept",
                actor=action.actor,
                message=action.message,
                before_candidate_status=candidate.status,
                after_candidate_status=updated_candidate.status
                if updated_candidate
                else "accepted",
                before_link=_dump_model(before_link),
                after_link=after_link.model_dump(mode="json"),
                before_task=_dump_model(before_task),
                after_task=after_task.model_dump(mode="json"),
                created_global_material_id=created_global_id,
                created_at=timestamp,
            )
        )
        if not defer_completion:
            self.material_completion.confirm_paper_if_materials_complete(candidate.paper_id)
        if not return_bundle:
            return None
        return self.material_resolution.get_material_structure_bundle(candidate.paper_id)

    def correct_structure_candidate(
        self,
        structure_candidate_id: str,
        action: MaterialStructureEditAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        candidate = self.structure_candidates.get(structure_candidate_id)
        if not candidate:
            return None
        if candidate.status == "accepted":
            raise ValueError("Undo the accepted candidate before editing its SMILES.")
        standardized = standardize_smiles(action.smiles)
        timestamp = now_iso()
        evidence = {
            **candidate.evidence,
            "manual_correction": {
                "actor": action.actor,
                "previous_raw_smiles": candidate.raw_smiles,
                "updated_at": timestamp,
            },
        }
        updated = self.structure_candidates.update(
            candidate.model_copy(
                update={
                    "raw_smiles": standardized.raw_smiles,
                    "canonical_smiles": standardized.canonical_smiles,
                    "isomeric_smiles": standardized.isomeric_smiles,
                    "inchi": standardized.inchi,
                    "inchi_key": standardized.inchi_key,
                    "formula": standardized.formula,
                    "molecular_weight": standardized.molecular_weight,
                    "representation_type": "small_molecule",
                    "evidence": evidence,
                    "status": "pending_review",
                    "updated_at": timestamp,
                }
            )
        )
        self.material_review_events.add(
            MaterialReviewEvent(
                event_id=uuid4().hex,
                paper_id=candidate.paper_id,
                candidate_run_id=candidate.candidate_run_id,
                paper_material_id=candidate.paper_material_id,
                structure_candidate_id=candidate.structure_candidate_id,
                action="edit_smiles",
                actor=action.actor,
                message=action.message,
                before_candidate_status=candidate.status,
                after_candidate_status=updated.status,
                before_candidate=candidate.model_dump(mode="json"),
                after_candidate=updated.model_dump(mode="json"),
                created_at=timestamp,
            )
        )
        return self.material_resolution.get_material_structure_bundle(candidate.paper_id)

    def save_manual_structure(
        self,
        paper_id: str,
        paper_material_id: str,
        action: MaterialManualStructureAction,
        *,
        defer_completion: bool = False,
        return_bundle: bool = True,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        bundle = self.material_resolution.resolve_paper_materials(normalized_paper_id)
        if not bundle or not bundle.candidate_run_id:
            return bundle
        paper_material = next(
            (
                material
                for material in bundle.materials
                if material.paper_material_id == paper_material_id
            ),
            None,
        )
        if not paper_material:
            raise ValueError(f"Paper material not found: {paper_material_id}")
        standardized = standardize_smiles(action.smiles)
        timestamp = now_iso()
        query_text = _manual_candidate_name(action, paper_material)
        candidate = self.structure_candidates.upsert(
            MaterialStructureCandidate(
                structure_candidate_id=uuid4().hex,
                paper_id=bundle.paper_id,
                candidate_run_id=bundle.candidate_run_id,
                paper_material_id=paper_material_id,
                provider="manual_input",
                resolver_name="human_manual_structure",
                query_text=query_text,
                query_type="manual_structure",
                source_identifier=f"manual:{uuid4().hex}",
                source_url=action.source_url,
                canonical_name=action.reviewed_name or query_text,
                material_class=action.material_class or paper_material.material_class or "unknown",
                representation_type=action.representation_type or "small_molecule",
                raw_smiles=standardized.raw_smiles,
                canonical_smiles=standardized.canonical_smiles,
                isomeric_smiles=standardized.isomeric_smiles,
                inchi=standardized.inchi,
                inchi_key=standardized.inchi_key,
                formula=standardized.formula,
                molecular_weight=standardized.molecular_weight,
                synonyms=[
                    item
                    for item in [
                        action.reviewed_name,
                        action.full_name_in_paper,
                        paper_material.abbreviation,
                        *paper_material.mention_list,
                    ]
                    if item
                ],
                evidence={
                    "resolver": "human_manual_structure",
                    "source_note": action.source_note,
                    "source_url": action.source_url,
                    "full_name_in_paper": action.full_name_in_paper,
                    "paper_material": paper_material.model_dump(mode="json"),
                    "standardized_by": "rdkit",
                },
                confidence=1.0,
                status="pending_review",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        message = action.message or "Manual structure input saved and accepted."
        return self.accept_structure_candidate(
            candidate.structure_candidate_id,
            MaterialReviewAction(
                actor=action.actor, message=message, global_material_id=action.global_material_id
            ),
            defer_completion=defer_completion,
            return_bundle=return_bundle,
        )

    def link_existing_global_material(
        self,
        paper_id: str,
        paper_material_id: str,
        action: MaterialReviewAction,
        *,
        defer_completion: bool = False,
        return_bundle: bool = True,
    ) -> PaperMaterialStructureBundle | None:
        """Bind a paper-local identity to an independently verified global structure."""
        self.init_runtime()
        if not action.global_material_id:
            raise ValueError("global_material_id is required for an existing-global link.")
        global_material = self.global_materials.get(action.global_material_id)
        if global_material is None:
            raise ValueError(f"Global material not found: {action.global_material_id}")
        if global_material.review_status != "confirmed":
            raise ValueError("Only a confirmed global material can be linked automatically.")
        if not (
            global_material.canonical_smiles
            or global_material.isomeric_smiles
            or global_material.raw_smiles
        ):
            raise ValueError("The selected global material does not have a SMILES structure.")

        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        bundle = self.material_resolution.resolve_paper_materials(normalized_paper_id)
        if bundle is None or not bundle.candidate_run_id:
            return bundle
        paper_material = next(
            (
                item
                for item in bundle.materials
                if item.paper_material_id == paper_material_id
            ),
            None,
        )
        if paper_material is None:
            raise ValueError(f"Paper material not found: {paper_material_id}")

        before_link = self.links.get_by_paper_material(
            bundle.candidate_run_id,
            paper_material_id,
        )
        before_task = self.tasks.get_by_paper_material(
            bundle.candidate_run_id,
            paper_material_id,
        )
        if (
            before_link
            and before_link.global_material_id == global_material.global_material_id
            and before_link.match_status in {"matched_local", "matched_candidate"}
        ):
            return (
                self.material_resolution.get_material_structure_bundle(bundle.paper_id)
                if return_bundle
                else None
            )
        if (
            before_link
            and before_link.global_material_id
            and before_link.global_material_id != global_material.global_material_id
            and before_link.match_status in {"matched_local", "matched_candidate"}
        ):
            raise ValueError(
                "The paper material is already linked to a different confirmed structure; "
                "undo or replace that review explicitly before relinking."
            )

        timestamp = now_iso()
        confidence = action.evidence.get("confidence", 0.99)
        if not isinstance(confidence, (int, float)):
            confidence = 0.99
        after_link = self.links.upsert(
            PaperMaterialLink(
                paper_material_link_id=uuid4().hex,
                paper_id=bundle.paper_id,
                candidate_run_id=bundle.candidate_run_id,
                paper_material_id=paper_material_id,
                global_material_id=global_material.global_material_id,
                match_method="chemical_agent_verified_global",
                match_confidence=max(0.0, min(float(confidence), 1.0)),
                match_status="matched_candidate",
                evidence={
                    "verification": action.evidence,
                    "selected_global": {
                        "global_material_id": global_material.global_material_id,
                        "canonical_name": global_material.canonical_name,
                        "inchi_key": global_material.inchi_key,
                        "source": global_material.source,
                    },
                    "paper_material": {
                        "entity_label": paper_material.entity_label,
                        "abbreviation": paper_material.abbreviation,
                        "full_name_in_paper": paper_material.full_name_in_paper,
                        "mention_list": paper_material.mention_list,
                    },
                },
                created_at=timestamp,
                updated_at=timestamp,
                confirmed_at=timestamp,
            )
        )
        base_task = before_task or MaterialResolutionTask(
            task_id=uuid4().hex,
            paper_id=bundle.paper_id,
            candidate_run_id=bundle.candidate_run_id,
            paper_material_id=paper_material_id,
            material_mentions=paper_material.mention_list,
            material_context={},
            created_at=timestamp,
            updated_at=timestamp,
        )
        after_task = self.tasks.upsert(
            base_task.model_copy(
                update={
                    "status": "completed",
                    "assigned_strategy": "chemical_agent_verified_global",
                    "current_stage": "completed",
                    "next_action": "none",
                    "error_message": None,
                    "completed_at": timestamp,
                    "updated_at": timestamp,
                }
            )
        )
        self.material_review_events.add(
            MaterialReviewEvent(
                event_id=uuid4().hex,
                paper_id=bundle.paper_id,
                candidate_run_id=bundle.candidate_run_id,
                paper_material_id=paper_material_id,
                global_material_id=global_material.global_material_id,
                action="link_existing_global",
                actor=action.actor,
                message=action.message,
                before_link=_dump_model(before_link),
                after_link=after_link.model_dump(mode="json"),
                before_task=_dump_model(before_task),
                after_task=after_task.model_dump(mode="json"),
                created_at=timestamp,
            )
        )
        if not defer_completion:
            self.material_completion.confirm_paper_if_materials_complete(bundle.paper_id)
        if not return_bundle:
            return None
        return self.material_resolution.get_material_structure_bundle(bundle.paper_id)

    def get_structure_candidate_depiction_svg(
        self,
        structure_candidate_id: str,
    ) -> str | None:
        self.init_runtime()
        candidate = self.structure_candidates.get(structure_candidate_id)
        if not candidate:
            return None
        smiles = candidate.canonical_smiles or candidate.isomeric_smiles or candidate.raw_smiles
        if not smiles:
            raise ValueError("Candidate does not have a SMILES representation.")
        return smiles_depiction_svg(smiles)

    def get_global_material_depiction_svg(
        self,
        global_material_id: str,
    ) -> str | None:
        self.init_runtime()
        material = self.global_materials.get(global_material_id)
        if not material:
            return None
        smiles = material.canonical_smiles or material.isomeric_smiles or material.raw_smiles
        if not smiles:
            raise ValueError("Global material does not have a SMILES representation.")
        return smiles_depiction_svg(smiles)

    def reject_structure_candidate(
        self,
        structure_candidate_id: str,
        action: MaterialReviewAction,
        *,
        return_bundle: bool = True,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        candidate = self.structure_candidates.get(structure_candidate_id)
        if not candidate:
            return None
        before_task = self.tasks.get_by_paper_material(
            candidate.candidate_run_id,
            candidate.paper_material_id,
        )
        updated_candidate = self.structure_candidates.set_status(
            structure_candidate_id,
            "rejected",
        )
        after_task = self._task_after_reject(candidate, before_task)
        timestamp = now_iso()
        self.material_review_events.add(
            MaterialReviewEvent(
                event_id=uuid4().hex,
                paper_id=candidate.paper_id,
                candidate_run_id=candidate.candidate_run_id,
                paper_material_id=candidate.paper_material_id,
                structure_candidate_id=candidate.structure_candidate_id,
                action="reject",
                actor=action.actor,
                message=action.message,
                before_candidate_status=candidate.status,
                after_candidate_status=updated_candidate.status
                if updated_candidate
                else "rejected",
                before_link=None,
                after_link=None,
                before_task=_dump_model(before_task),
                after_task=after_task.model_dump(mode="json") if after_task else None,
                created_at=timestamp,
            )
        )
        if not return_bundle:
            return None
        return self.material_resolution.get_material_structure_bundle(candidate.paper_id)

    def undo_material_review_event(
        self,
        event_id: str,
        action: MaterialReviewAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        event = self.material_review_events.get(event_id)
        if not event or event.action == "undo":
            return None
        if event.action == "edit_smiles" and event.before_candidate:
            current_candidate = (
                self.structure_candidates.get(event.structure_candidate_id)
                if event.structure_candidate_id
                else None
            )
            restored = MaterialStructureCandidate.model_validate(event.before_candidate).model_copy(
                update={"updated_at": now_iso()}
            )
            self.structure_candidates.update(restored)
            timestamp = now_iso()
            self.material_review_events.add(
                MaterialReviewEvent(
                    event_id=uuid4().hex,
                    paper_id=event.paper_id,
                    candidate_run_id=event.candidate_run_id,
                    paper_material_id=event.paper_material_id,
                    structure_candidate_id=event.structure_candidate_id,
                    action="undo",
                    actor=action.actor,
                    message=action.message or f"Undo material review event {event.event_id}",
                    before_candidate_status=(
                        current_candidate.status if current_candidate else None
                    ),
                    after_candidate_status=restored.status,
                    before_candidate=(
                        current_candidate.model_dump(mode="json") if current_candidate else None
                    ),
                    after_candidate=restored.model_dump(mode="json"),
                    created_at=timestamp,
                )
            )
            return self.material_resolution.get_material_structure_bundle(event.paper_id)
        before_status = None
        if event.structure_candidate_id:
            current_candidate = self.structure_candidates.get(event.structure_candidate_id)
            before_status = current_candidate.status if current_candidate else None
            if event.before_candidate_status:
                self.structure_candidates.set_status(
                    event.structure_candidate_id,
                    event.before_candidate_status,
                )
        if event.before_link:
            self.links.upsert(PaperMaterialLink.model_validate(event.before_link))
        else:
            self.links.delete_by_paper_material(event.candidate_run_id, event.paper_material_id)
        if event.before_task:
            # Undo must restore the exact snapshot. The normal task upsert protects
            # completed orchestration from accidental regression to unresolved,
            # which is correct for workers but would otherwise defeat review undo.
            self.tasks.delete_by_paper_material(
                event.candidate_run_id,
                event.paper_material_id,
            )
            self.tasks.upsert(MaterialResolutionTask.model_validate(event.before_task))
        else:
            self.tasks.delete_by_paper_material(event.candidate_run_id, event.paper_material_id)
        if event.created_global_material_id:
            self._retract_created_global_material(event.created_global_material_id)
        timestamp = now_iso()
        self.material_review_events.add(
            MaterialReviewEvent(
                event_id=uuid4().hex,
                paper_id=event.paper_id,
                candidate_run_id=event.candidate_run_id,
                paper_material_id=event.paper_material_id,
                structure_candidate_id=event.structure_candidate_id,
                global_material_id=event.global_material_id,
                action="undo",
                actor=action.actor,
                message=action.message or f"Undo material review event {event.event_id}",
                before_candidate_status=before_status,
                after_candidate_status=event.before_candidate_status,
                before_link=event.after_link,
                after_link=event.before_link,
                before_task=event.after_task,
                after_task=event.before_task,
                created_global_material_id=event.created_global_material_id,
                created_at=timestamp,
            )
        )
        return self.material_resolution.get_material_structure_bundle(event.paper_id)

    def _global_material_for_candidate(
        self,
        candidate: MaterialStructureCandidate,
        action: MaterialReviewAction,
    ) -> tuple[MaterialGlobal, str | None]:
        if action.global_material_id:
            existing = self.global_materials.get(action.global_material_id)
            if existing:
                return self._merge_candidate_into_global_material(existing, candidate), None
        if candidate.inchi_key:
            existing = self.global_materials.get_by_inchi_key(candidate.inchi_key)
            if existing and existing.review_status != "retracted":
                return self._merge_candidate_into_global_material(existing, candidate), None
        timestamp = now_iso()
        material = MaterialGlobal(
            global_material_id=uuid4().hex,
            canonical_name=candidate.canonical_name or candidate.query_text,
            material_class=candidate.material_class,
            representation_type=candidate.representation_type,
            raw_smiles=candidate.raw_smiles,
            canonical_smiles=candidate.canonical_smiles,
            isomeric_smiles=candidate.isomeric_smiles,
            inchi=candidate.inchi,
            inchi_key=candidate.inchi_key,
            formula=candidate.formula,
            molecular_weight=candidate.molecular_weight,
            source=candidate.provider,
            source_detail={
                "structure_candidate_id": candidate.structure_candidate_id,
                "source_identifier": candidate.source_identifier,
                "source_url": candidate.source_url,
                "resolver_name": candidate.resolver_name,
            },
            confidence=candidate.confidence,
            review_status="confirmed",
            created_at=timestamp,
            updated_at=timestamp,
            confirmed_at=timestamp,
        )
        return self.global_materials.upsert(material), material.global_material_id

    def _merge_candidate_into_global_material(
        self,
        material: MaterialGlobal,
        candidate: MaterialStructureCandidate,
    ) -> MaterialGlobal:
        timestamp = now_iso()
        source_detail = {
            **material.source_detail,
            "last_accepted_structure_candidate": {
                "structure_candidate_id": candidate.structure_candidate_id,
                "paper_id": candidate.paper_id,
                "paper_material_id": candidate.paper_material_id,
                "provider": candidate.provider,
                "resolver_name": candidate.resolver_name,
                "source_identifier": candidate.source_identifier,
                "source_url": candidate.source_url,
                "accepted_at": timestamp,
            },
        }
        updated = material.model_copy(
            update={
                # A paper-local label may be wrong even when its submitted structure is
                # identical to an existing material. Keep the confirmed global identity
                # stable and record the paper-local name as an alias below.
                "canonical_name": material.canonical_name or candidate.canonical_name,
                "material_class": (
                    candidate.material_class
                    if candidate.material_class != "unknown"
                    else material.material_class
                ),
                "representation_type": (
                    candidate.representation_type
                    if candidate.representation_type != "unknown"
                    else material.representation_type
                ),
                "raw_smiles": candidate.raw_smiles or material.raw_smiles,
                "canonical_smiles": candidate.canonical_smiles or material.canonical_smiles,
                "isomeric_smiles": candidate.isomeric_smiles or material.isomeric_smiles,
                "inchi": candidate.inchi or material.inchi,
                "inchi_key": candidate.inchi_key or material.inchi_key,
                "formula": candidate.formula or material.formula,
                "molecular_weight": candidate.molecular_weight or material.molecular_weight,
                "source": candidate.provider or material.source,
                "source_detail": source_detail,
                "confidence": max(
                    value
                    for value in [material.confidence, candidate.confidence]
                    if value is not None
                )
                if material.confidence is not None or candidate.confidence is not None
                else None,
                "review_status": "confirmed",
                "updated_at": timestamp,
                "confirmed_at": material.confirmed_at or timestamp,
            }
        )
        return self.global_materials.upsert(updated)

    def _add_aliases_for_candidate(
        self,
        material: MaterialGlobal,
        candidate: MaterialStructureCandidate,
        paper_material: PaperLocalMaterial | None,
    ) -> None:
        timestamp = now_iso()
        for alias_text, alias_type in _dedupe_alias_items(
            [
                *(_paper_local_alias_items(paper_material) if paper_material is not None else []),
                (material.canonical_name, "structure_candidate"),
                (candidate.canonical_name, "structure_candidate"),
                (candidate.query_text, "structure_candidate"),
                *((synonym, "structure_candidate") for synonym in candidate.synonyms),
            ]
        ):
            normalized_alias = normalize_material_alias(alias_text)
            if has_confirmed_alias_conflict(
                self.aliases.find_by_normalized(normalized_alias),
                material.global_material_id,
            ):
                continue
            self.aliases.add_if_missing(
                MaterialAlias(
                    alias_id=uuid4().hex,
                    global_material_id=material.global_material_id,
                    alias_text=alias_text,
                    normalized_alias=normalized_alias,
                    alias_type=alias_type,
                    source_paper_id=candidate.paper_id,
                    source=candidate.provider,
                    confidence=candidate.confidence,
                    review_status="confirmed",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )

    def _completed_task_for_candidate(
        self,
        candidate: MaterialStructureCandidate,
        before_task: MaterialResolutionTask | None,
    ) -> MaterialResolutionTask:
        timestamp = now_iso()
        base = before_task or MaterialResolutionTask(
            task_id=uuid4().hex,
            paper_id=candidate.paper_id,
            candidate_run_id=candidate.candidate_run_id,
            paper_material_id=candidate.paper_material_id,
            material_mentions=[candidate.query_text],
            material_context={},
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self.tasks.upsert(
            base.model_copy(
                update={
                    "status": "completed",
                    "assigned_strategy": (
                        "manual_structure_input"
                        if candidate.provider == "manual_input"
                        else "human_accepted_public_candidate"
                    ),
                    "current_stage": "completed",
                    "next_action": "none",
                    "error_message": None,
                    "completed_at": timestamp,
                    "updated_at": timestamp,
                }
            )
        )

    def _task_after_reject(
        self,
        candidate: MaterialStructureCandidate,
        before_task: MaterialResolutionTask | None,
    ) -> MaterialResolutionTask | None:
        timestamp = now_iso()
        existing_link = self.links.get_by_paper_material(
            candidate.candidate_run_id,
            candidate.paper_material_id,
        )
        if (
            before_task
            and candidate.status != "accepted"
            and existing_link
            and existing_link.global_material_id
            and existing_link.match_status in {"matched_local", "matched_candidate"}
        ):
            return before_task
        remaining = [
            item
            for item in self.structure_candidates.list_by_paper_material(
                candidate.candidate_run_id,
                candidate.paper_material_id,
            )
            if item.structure_candidate_id != candidate.structure_candidate_id
            and item.status != "rejected"
        ]
        if not before_task:
            return None
        if remaining:
            next_status = "needs_review"
            next_strategy = "public_database_review"
            next_stage = "human_review"
            next_action = "review_public_candidate"
        else:
            next_status = "pending"
            next_strategy = "public_candidate_rejected_continue_resolution"
            next_stage = "visual_ocsr_pending"
            next_action = "run_visual_ocsr"
        return self.tasks.upsert(
            before_task.model_copy(
                update={
                    "status": next_status,
                    "assigned_strategy": next_strategy,
                    "current_stage": next_stage,
                    "next_action": next_action,
                    "updated_at": timestamp,
                    "completed_at": None,
                }
            )
        )

    def _retract_created_global_material(self, global_material_id: str) -> None:
        material = self.global_materials.get(global_material_id)
        if not material:
            return
        timestamp = now_iso()
        self.global_materials.upsert(
            material.model_copy(
                update={
                    "review_status": "retracted",
                    "updated_at": timestamp,
                    "confirmed_at": None,
                }
            )
        )
        self.aliases.set_review_status_by_global_material(global_material_id, "retracted")


def _manual_candidate_name(
    action: MaterialManualStructureAction,
    material: PaperLocalMaterial,
) -> str:
    for value in (
        action.reviewed_name,
        action.full_name_in_paper,
        material.canonical_name,
        material.normalized_name,
        material.full_name_in_paper,
        material.abbreviation,
        material.entity_label,
        material.mention_list[0] if material.mention_list else None,
        material.paper_material_id,
    ):
        if value and value.strip():
            return value.strip()
    return material.paper_material_id


def _dump_model(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return]
    return None


def _paper_local_alias_items(
    material: PaperLocalMaterial,
) -> list[tuple[str | None, str]]:
    return [
        (material.abbreviation, "paper_local_abbreviation"),
        *((mention, "paper_local_mention") for mention in material.mention_list),
        (material.entity_label, "paper_local_entity_label"),
        (material.normalized_name, "paper_local_normalized_name"),
        (material.full_name_in_paper, "paper_local_full_name"),
        (material.canonical_name, "paper_local_canonical_name"),
    ]


def _dedupe_alias_items(
    values: list[tuple[str | None, str]],
) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value, alias_type in values:
        normalized = normalize_material_alias(value)
        if (
            not value
            or not normalized
            or normalized in seen
            or (
                alias_type.startswith("paper_local_")
                and not is_reusable_material_alias_text(value)
            )
        ):
            continue
        seen.add(normalized)
        aliases.append((value, alias_type))
    return aliases
