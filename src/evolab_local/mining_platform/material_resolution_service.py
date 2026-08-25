from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.external.anysearch_client import (
    AnySearchClient,
    AnySearchResult,
    MaterialWebSearchClient,
)
from evolab_local.mining_platform.external.pubchem_client import (
    PubChemClient,
    PubChemCompound,
    PublicCompoundResolverClient,
)

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.candidate_ingestion import (
    CandidateEntity,
    CandidateIngestionRun,
    CandidateValue,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialAlias,
    MaterialGlobal,
    MaterialReviewEvent,
    MaterialStructureCandidate,
    PaperMaterialNameReview,
    PaperMaterialNameReviewAction,
    PaperMaterialNameSuggestion,
    MaterialResolutionTask,
    MaterialUsage,
    PaperLocalMaterial,
    PaperMaterialLink,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    CandidateIngestionRepository,
    MaterialAliasRepository,
    MaterialGlobalRepository,
    MaterialIdentityEvidenceItemRepository,
    MaterialIdentityEvidenceRunRepository,
    MaterialIdentityJudgmentRepository,
    MaterialPropertyCandidateRepository,
    MaterialPropertyReviewEventRepository,
    MaterialPropertyReviewRepository,
    MaterialResolutionTaskRepository,
    MaterialReviewEventRepository,
    PaperMaterialNameReviewRepository,
    PaperMaterialNameSuggestionRepository,
    MaterialStructureCandidateRepository,
    PaperMaterialLinkRepository,
    PaperRepository,
    CandidateValueReviewEventRepository,
)


PAPER_SPECIFIC_PREFIX_RE = re.compile(
    r"^(compound|molecule|emitter|host|material|sample|device)\s*[-_ ]?[a-z0-9]+$",
    re.IGNORECASE,
)
SHORT_LABEL_RE = re.compile(r"^[a-z]{0,3}\d+[a-z]?$", re.IGNORECASE)
EMPIRICAL_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")

# These strings describe a device role or a paper-local placeholder, not a
# reusable chemical identity.  Keeping them in provenance is useful, but using
# them as global aliases creates cross-paper false matches (for example, two
# unrelated commercial hosts both abbreviated as "BH").
NON_REUSABLE_GLOBAL_ALIAS_NORMALIZED = {
    "acceptor",
    "assistantdopant",
    "bh",
    "bluedopant",
    "blueemitter",
    "bluehost",
    "dopant",
    "donor",
    "emitter",
    "finalemitter",
    "guest",
    "host",
    "hostmaterial",
    "material",
    "sensitizer",
    "transportmaterial",
}

OUT_OF_SCOPE_STRUCTURE_ALIASES = {
    "ito": "transparent electrode / anode",
    "indiumtinoxide": "transparent electrode / anode",
    "al": "metal electrode",
    "aluminum": "metal electrode",
    "aluminium": "metal electrode",
    "ag": "metal electrode",
    "au": "metal electrode",
    "ca": "metal electrode",
    "mg": "metal electrode",
    "glass": "substrate",
    "quartz": "substrate",
    "sio2": "inorganic substrate / dielectric",
    "silicondioxide": "inorganic substrate / dielectric",
    "tio2": "inorganic optical coating / dielectric",
    "titaniumdioxide": "inorganic optical coating / dielectric",
    "grapheneoxide": "non-molecular graphene oxide material",
    "perovskite": "perovskite material outside the organic small-molecule OLED scope",
    "aupsnps": "gold-polystyrene nanoparticles / composite",
    "hbl": "device-layer role placeholder (hole-blocking layer), not a material identity",
    "holeblockinglayer": (
        "device-layer role placeholder (hole-blocking layer), not a material identity"
    ),
}

IDENTITY_ONLY_ALIASES = {
    "liq": "electron injection material identity record",
    "lif": "electron injection inorganic salt identity record",
    "lithiumfluoride": "electron injection inorganic salt identity record",
    "pedotpss": "conductive polymer blend identity record",
    "pedot": "conductive polymer identity record",
    "pss": "polymer electrolyte identity record",
    "pvk": "polymer host identity record",
    "pmma": "polymer matrix identity record",
    "polyvinylcarbazole": "polymer host identity record",
    "poly9vinylcarbazole": "polymer host identity record",
    "px2cz": "photocrosslinkable hole-transporting polymer identity record",
    "cfx": "fluorocarbon film identity record",
    "fluorocarbon": "fluorocarbon film identity record",
}

KNOWN_MATERIAL_CLASS_OVERRIDES = {
    "px2cz": "polymer",
}

ORGANOMETALLIC_COMPLEX_RE = re.compile(
    r"(?:^|[^a-z0-9])(ir|pt|pd|ru|os|rh|re|cu|zn|al)\s*\(",
    re.IGNORECASE,
)
ORGANOMETALLIC_WORD_RE = re.compile(
    r"\b(iridium|platinum|palladium|ruthenium|osmium|rhodium|rhenium|copper|zinc|"
    r"aluminum|aluminium|gallium)\b",
    re.IGNORECASE,
)
ORGANOMETALLIC_FORMULA_PREFIX_RE = re.compile(
    r"^(?:(?:Ir|Pt|Pd|Ru|Os|Rh|Re|Cu|Zn)(?=[A-Z0-9(\-])|Alq\d+$)"
)
POLYMER_HINT_RE = re.compile(
    r"\b(polymer|copolymer|oligomer|poly\s*\(|polystyrene|polyvinyl|poly\()",
    re.IGNORECASE,
)
DENDRIMER_HINT_RE = re.compile(r"\b(dendrimer|dendritic molecule)\b", re.IGNORECASE)
COMPOSITE_HINT_RE = re.compile(r"[:/]", re.IGNORECASE)

OUT_OF_SCOPE_COMPONENT_ROLES = {"electrode_material"}
OUT_OF_SCOPE_LAYER_ROLES = {"substrate", "anode", "cathode", "encapsulation"}
IDENTITY_ONLY_COMPONENT_ROLES = {"injection_material"}
IDENTITY_ONLY_LAYER_ROLES = {"eil"}
IDENTITY_ONLY_MATERIAL_CLASSES = {
    "organometallic_complex",
    "coordination_complex",
    "metal_complex",
    "transition_metal_complex",
}
IDENTITY_ONLY_POLYMER_MATERIAL_CLASSES = {
    "polymer",
    "polymer_blend",
    "polymer_composite",
    "polymer_emitter",
    "polymer_host",
    "polymer_matrix",
    "copolymer",
    "composite",
}
IDENTITY_ONLY_NON_SINGLE_STRUCTURE_CLASSES = {
    "mixture",
    "proprietary",
}
NON_MATERIAL_REFERENCE_CLASSES = {
    "non_material_reference",
}


class MaterialResolutionService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        pubchem_client: PublicCompoundResolverClient | None = None,
        anysearch_client: MaterialWebSearchClient | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.candidates = CandidateIngestionRepository(self.database)
        self.global_materials = MaterialGlobalRepository(self.database)
        self.aliases = MaterialAliasRepository(self.database)
        self.links = PaperMaterialLinkRepository(self.database)
        self.tasks = MaterialResolutionTaskRepository(self.database)
        self.structure_candidates = MaterialStructureCandidateRepository(self.database)
        self.identity_judgments = MaterialIdentityJudgmentRepository(self.database)
        self.identity_evidence_runs = MaterialIdentityEvidenceRunRepository(self.database)
        self.identity_evidence_items = MaterialIdentityEvidenceItemRepository(self.database)
        self.material_review_events = MaterialReviewEventRepository(self.database)
        self.property_candidates = MaterialPropertyCandidateRepository(self.database)
        self.property_reviews = MaterialPropertyReviewRepository(self.database)
        self.property_review_events = MaterialPropertyReviewEventRepository(self.database)
        self.material_name_reviews = PaperMaterialNameReviewRepository(self.database)
        self.material_name_suggestions = PaperMaterialNameSuggestionRepository(self.database)
        self.value_review_events = CandidateValueReviewEventRepository(self.database)
        self.pubchem_client = pubchem_client or PubChemClient(config.external_services.pubchem)
        self.anysearch_client = anysearch_client or (
            AnySearchClient(config.external_services.anysearch)
            if config.external_services.anysearch.api_key.strip()
            else None
        )

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def create_global_material(
        self,
        *,
        canonical_name: str,
        aliases: list[str] | None = None,
        material_class: str = "unknown",
        representation_type: str = "unknown",
        raw_smiles: str | None = None,
        canonical_smiles: str | None = None,
        isomeric_smiles: str | None = None,
        inchi: str | None = None,
        inchi_key: str | None = None,
        source: str = "manual",
        confidence: float | None = None,
        review_status: str = "candidate",
    ) -> MaterialGlobal:
        self.init_runtime()
        timestamp = now_iso()
        material = MaterialGlobal(
            global_material_id=uuid4().hex,
            canonical_name=canonical_name,
            material_class=material_class,
            representation_type=representation_type,
            raw_smiles=raw_smiles,
            canonical_smiles=canonical_smiles,
            isomeric_smiles=isomeric_smiles,
            inchi=inchi,
            inchi_key=inchi_key,
            source=source,
            confidence=confidence,
            review_status=review_status,
            created_at=timestamp,
            updated_at=timestamp,
            confirmed_at=timestamp if review_status == "confirmed" else None,
        )
        stored = self.global_materials.upsert(material)
        for alias_text in _dedupe_text([canonical_name, *(aliases or [])]):
            self.aliases.add(
                MaterialAlias(
                    alias_id=uuid4().hex,
                    global_material_id=stored.global_material_id,
                    alias_text=alias_text,
                    normalized_alias=normalize_material_alias(alias_text),
                    alias_type="canonical_name"
                    if alias_text == canonical_name
                    else "database_synonym",
                    source=source,
                    confidence=confidence,
                    review_status=review_status,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
        return stored

    def seed_common_oled_materials(self) -> list[MaterialGlobal]:
        seeded: list[MaterialGlobal] = []
        for canonical_name, aliases, material_class, representation_type in (
            ("mCBP", ["3,3'-di(9H-carbazol-9-yl)biphenyl"], "small_molecule_organic", "unknown"),
            ("TAPC", [], "small_molecule_organic", "unknown"),
            ("TPBi", [], "small_molecule_organic", "unknown"),
            ("HAT-CN", ["HATCN", "HAT CN"], "small_molecule_organic", "unknown"),
            ("LiF", ["lithium fluoride"], "inorganic", "inorganic"),
            ("ITO", ["indium tin oxide"], "inorganic", "inorganic"),
            ("Al", ["aluminum", "aluminium"], "inorganic", "inorganic"),
        ):
            existing_aliases = self.aliases.find_by_normalized(
                normalize_material_alias(canonical_name)
            )
            if existing_aliases:
                existing = self.global_materials.get(existing_aliases[0].global_material_id)
                if existing:
                    seeded.append(existing)
                    continue
            seeded.append(
                self.create_global_material(
                    canonical_name=canonical_name,
                    aliases=aliases,
                    material_class=material_class,
                    representation_type=representation_type,
                    source="common_oled_seed",
                    confidence=0.8,
                    review_status="candidate",
                )
            )
        return seeded

    def get_material_structure_bundle(self, paper_id: str) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        run = self._latest_completed_candidate_run(normalized_paper_id)
        if not run:
            return PaperMaterialStructureBundle(paper_id=normalized_paper_id)
        materials = self._paper_local_materials(run)
        material_name_reviews = self.material_name_reviews.list_by_run(run.candidate_run_id)
        material_name_suggestions = self.material_name_suggestions.list_by_run(run.candidate_run_id)
        links = self.links.list_by_run(run.candidate_run_id)
        tasks = self.tasks.list_by_run(run.candidate_run_id)
        structure_candidates = self.structure_candidates.list_by_run(run.candidate_run_id)
        identity_judgments = self.identity_judgments.list_by_run(run.candidate_run_id)
        identity_evidence_runs = self.identity_evidence_runs.list_by_run(run.candidate_run_id)
        identity_evidence_items = self.identity_evidence_items.list_by_run(run.candidate_run_id)
        material_review_events = self.material_review_events.list_by_run(run.candidate_run_id)
        if self.config.features.material_properties:
            property_candidates = self.property_candidates.list_by_run(run.candidate_run_id)
            property_reviews = self.property_reviews.list_by_run(run.candidate_run_id)
            property_review_events = self.property_review_events.list_by_run(run.candidate_run_id)
        else:
            property_candidates = []
            property_reviews = []
            property_review_events = []
        (
            materials,
            material_name_reviews,
            material_name_suggestions,
            links,
            tasks,
            structure_candidates,
            identity_judgments,
            identity_evidence_runs,
            identity_evidence_items,
            material_review_events,
            property_candidates,
            property_reviews,
            property_review_events,
        ) = _filter_review_bundle_to_device_used_materials(
            materials=materials,
            material_name_reviews=material_name_reviews,
            material_name_suggestions=material_name_suggestions,
            links=links,
            tasks=tasks,
            structure_candidates=structure_candidates,
            identity_judgments=identity_judgments,
            identity_evidence_runs=identity_evidence_runs,
            identity_evidence_items=identity_evidence_items,
            material_review_events=material_review_events,
            property_candidates=property_candidates,
            property_reviews=property_reviews,
            property_review_events=property_review_events,
        )
        global_ids = sorted({link.global_material_id for link in links if link.global_material_id})
        global_materials = self.global_materials.get_many(global_ids)
        return PaperMaterialStructureBundle(
            paper_id=normalized_paper_id,
            candidate_run_id=run.candidate_run_id,
            materials=materials,
            material_name_reviews=material_name_reviews,
            material_name_suggestions=material_name_suggestions,
            links=links,
            tasks=tasks,
            structure_candidates=structure_candidates,
            identity_judgments=identity_judgments,
            identity_evidence_runs=identity_evidence_runs,
            identity_evidence_items=identity_evidence_items,
            material_review_events=material_review_events,
            property_candidates=property_candidates,
            property_reviews=property_reviews,
            property_review_events=property_review_events,
            global_materials=global_materials,
        )

    def resolve_paper_materials(self, paper_id: str) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        run = self._latest_completed_candidate_run(normalized_paper_id)
        if not run:
            return PaperMaterialStructureBundle(paper_id=normalized_paper_id)
        for material in _device_used_or_all_materials(self._paper_local_materials(run)):
            self._resolve_material(run, material)
        return self.get_material_structure_bundle(normalized_paper_id)

    def list_resolution_tasks(self) -> list[MaterialResolutionTask]:
        self.init_runtime()
        return self.tasks.list()

    def review_paper_material_name(
        self,
        paper_id: str,
        paper_material_id: str,
        action: PaperMaterialNameReviewAction,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        run = self._latest_completed_candidate_run(normalized_paper_id)
        if not run:
            return PaperMaterialStructureBundle(paper_id=normalized_paper_id)
        before_materials = self._paper_local_materials(run)
        before_material = next(
            (
                material
                for material in before_materials
                if material.paper_material_id == paper_material_id
            ),
            None,
        )
        if not before_material:
            raise ValueError(f"Paper material not found: {paper_material_id}")
        timestamp = now_iso()
        review = self.material_name_reviews.upsert(
            PaperMaterialNameReview(
                review_id=uuid4().hex,
                paper_id=normalized_paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=paper_material_id,
                reviewed_name=_clean_text(action.reviewed_name),
                reviewed_full_name_in_paper=_clean_text(action.reviewed_full_name_in_paper),
                reviewed_abbreviation=_clean_text(action.reviewed_abbreviation),
                reviewed_normalized_name=_reviewed_normalized_name(action, before_material),
                reviewed_canonical_name=_clean_text(action.reviewed_canonical_name),
                actor=action.actor,
                message=action.message,
                source="manual_review",
                source_detail={"before_material": before_material.model_dump(mode="json")},
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        self._sync_material_name_review_to_candidate_values(run, before_material, review)
        after_bundle = self.get_material_structure_bundle(normalized_paper_id)
        after_material = None
        if after_bundle:
            after_material = next(
                (
                    material
                    for material in after_bundle.materials
                    if material.paper_material_id == paper_material_id
                ),
                None,
            )
        self.material_review_events.add(
            event=MaterialReviewEvent(
                event_id=uuid4().hex,
                paper_id=normalized_paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=paper_material_id,
                action="correct_material_name",
                actor=action.actor,
                message=action.message,
                before_candidate=before_material.model_dump(mode="json"),
                after_candidate=after_material.model_dump(mode="json") if after_material else None,
                created_at=now_iso(),
            )
        )
        return self.get_material_structure_bundle(normalized_paper_id)

    def validate_material_names(
        self,
        paper_id: str,
        paper_material_id: str | None = None,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        run = self._latest_completed_candidate_run(normalized_paper_id)
        if not run:
            return PaperMaterialStructureBundle(paper_id=normalized_paper_id)
        materials = _device_used_or_all_materials(self._paper_local_materials(run))
        for material in materials:
            if paper_material_id and material.paper_material_id != paper_material_id:
                continue
            suggestion = _agent_material_name_suggestion(
                run,
                material,
                pubchem_client=self.pubchem_client,
                anysearch_client=self.anysearch_client,
            )
            if suggestion:
                self.material_name_suggestions.upsert(suggestion)
        return self.get_material_structure_bundle(normalized_paper_id)

    def _sync_material_name_review_to_candidate_values(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
        review: PaperMaterialNameReview,
    ) -> None:
        values = self.candidates.list_values_by_run(run.candidate_run_id)
        material_entity_id = _material_entity_id_for_paper_material(
            self.candidates.list_entities_by_run(run.candidate_run_id),
            values,
            material.paper_material_id,
        )
        if not material_entity_id:
            return
        entity_values = [
            value for value in values if value.candidate_entity_id == material_entity_id
        ]
        updates = _material_name_candidate_value_updates(material, review)
        for value in entity_values:
            if value.template_field_path not in updates:
                continue
            before = value
            updated = self.candidates.update_value(
                value.candidate_value_id,
                updates[value.template_field_path],
                "modified",
            )
            if not updated:
                continue
            self.value_review_events.add(
                before=before,
                after=updated,
                action="material_name_corrected",
                actor=review.actor,
                message=review.message,
            )

    def _latest_completed_candidate_run(self, paper_id: str) -> CandidateIngestionRun | None:
        runs = self.candidates.list_runs_by_paper(paper_id)
        return next((run for run in runs if run.status == "completed"), None)

    def _paper_local_materials(self, run: CandidateIngestionRun) -> list[PaperLocalMaterial]:
        entities = self.candidates.list_entities_by_run(run.candidate_run_id)
        values = self.candidates.list_values_by_run(run.candidate_run_id)
        values_by_entity_id = _candidate_values_by_entity_id(values)
        material_entities = [entity for entity in entities if entity.entity_type == "materials"]
        usage_by_material_id = _material_usage_by_id(run.mining_result)
        reviews_by_material_id = {
            review.paper_material_id: review
            for review in self.material_name_reviews.list_by_run(run.candidate_run_id)
        }
        materials: list[PaperLocalMaterial] = []
        for entity in material_entities:
            material = _paper_local_material_from_entity(
                entity,
                values_by_entity_id.get(entity.candidate_entity_id, []),
            )
            material = _apply_material_name_review(
                material,
                reviews_by_material_id.get(material.paper_material_id),
            )
            materials.append(
                material.model_copy(
                    update={"used_in": usage_by_material_id.get(material.paper_material_id, [])}
                )
            )
        return sorted(materials, key=lambda item: item.paper_material_id)

    def _resolve_material(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
    ) -> None:
        timestamp = now_iso()
        current_link = self.links.get_by_paper_material(
            run.candidate_run_id,
            material.paper_material_id,
        )
        structure_scope = classify_material_structure_scope(material)
        if (
            current_link
            and current_link.match_method == "structure_scope_rule"
            and current_link.match_status in {"identity_only", "out_of_scope_structure"}
            and structure_scope["category"] == "core_structure_required"
        ):
            # A classifier fix can turn an old terminal scope decision back into
            # actionable work. Remove its terminal task before rebuilding it so
            # repository orchestration preservation does not keep it completed.
            self.tasks.delete_by_paper_material(
                run.candidate_run_id,
                material.paper_material_id,
            )
        if _is_protected_material_link(current_link):
            if current_link and current_link.global_material_id:
                global_material = self.global_materials.get(current_link.global_material_id)
                if global_material and global_material.review_status != "retracted":
                    self._add_paper_local_aliases_for_global_material(
                        run,
                        material,
                        global_material,
                        source="matched_candidate_link",
                        confidence=current_link.match_confidence,
                    )
            return
        if self._restore_accepted_candidate_link(run, material):
            return
        if self._apply_structure_scope_resolution(
            run,
            material,
            timestamp,
            scope=structure_scope,
        ):
            return
        mention_candidates = _mention_candidates(material)
        safe_candidates = [
            candidate for candidate in mention_candidates if not candidate["is_paper_specific"]
        ]
        match = self._lookup_local_material(safe_candidates)
        if match["status"] == "matched":
            link = PaperMaterialLink(
                paper_material_link_id=uuid4().hex,
                paper_id=run.paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=material.paper_material_id,
                global_material_id=match["global_material_id"],
                match_method=match["method"],
                match_confidence=match["confidence"],
                match_status=match["match_status"],
                evidence={
                    "matched_alias": match["matched_alias"],
                    "normalized_alias": match["normalized_alias"],
                    "source_candidates": mention_candidates,
                },
                created_at=timestamp,
                updated_at=timestamp,
                confirmed_at=timestamp if match["match_status"] == "matched_local" else None,
            )
            self.links.upsert(link)
            if match["match_status"] == "matched_local":
                self.tasks.delete_by_paper_material(
                    run.candidate_run_id,
                    material.paper_material_id,
                )
            else:
                self._upsert_task(run, material, "needs_review", "local_registry_review")
            return
        if match["status"] == "ambiguous":
            self.links.upsert(
                PaperMaterialLink(
                    paper_material_link_id=uuid4().hex,
                    paper_id=run.paper_id,
                    candidate_run_id=run.candidate_run_id,
                    paper_material_id=material.paper_material_id,
                    match_method="ambiguous_alias",
                    match_confidence=0.5,
                    match_status="ambiguous",
                    evidence={
                        "matched_alias": match["matched_alias"],
                        "normalized_alias": match["normalized_alias"],
                        "candidate_global_material_ids": match["candidate_global_material_ids"],
                        "source_candidates": mention_candidates,
                    },
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            # Reopen stale orchestration state. A material may previously have
            # matched one local record and later become ambiguous after another
            # confirmed structure is added for the same alias. At this point
            # accepted-candidate restoration has already failed, so preserving
            # an old completed task would contradict the ambiguous link.
            self.tasks.delete_by_paper_material(
                run.candidate_run_id,
                material.paper_material_id,
            )
            self._upsert_task(run, material, "needs_review", "ambiguous_local_alias")
            return
        self.links.upsert(
            PaperMaterialLink(
                paper_material_link_id=uuid4().hex,
                paper_id=run.paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=material.paper_material_id,
                match_method="none",
                match_status="unresolved",
                evidence={"source_candidates": mention_candidates},
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        self._upsert_task(
            run,
            material,
            "pending",
            "manual_structure_required" if not safe_candidates else "unresolved",
        )

    def _apply_structure_scope_resolution(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
        timestamp: str,
        *,
        scope: Mapping[str, Any] | None = None,
    ) -> bool:
        scope = scope or classify_material_structure_scope(material)
        if scope["category"] == "core_structure_required":
            return False
        mention_candidates = _mention_candidates(material)
        self.links.upsert(
            PaperMaterialLink(
                paper_material_link_id=uuid4().hex,
                paper_id=run.paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=material.paper_material_id,
                match_method="structure_scope_rule",
                match_confidence=_float(scope.get("confidence")),
                match_status=str(scope["category"]),
                evidence={
                    "structure_scope": scope,
                    "source_candidates": mention_candidates,
                },
                created_at=timestamp,
                updated_at=timestamp,
                confirmed_at=timestamp,
            )
        )
        self._upsert_task(
            run,
            material,
            "completed",
            str(scope["category"]),
            extra_context={"structure_scope": scope},
        )
        return True

    def _restore_accepted_candidate_link(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
    ) -> bool:
        candidates = self.structure_candidates.list_by_paper_material(
            run.candidate_run_id,
            material.paper_material_id,
        )
        accepted_candidates = [
            candidate for candidate in candidates if candidate.status == "accepted"
        ]
        review_events = self.material_review_events.list_by_run(run.candidate_run_id)
        active_accept_events = _active_accept_events(
            review_events,
            material.paper_material_id,
        )
        candidate_by_id = {candidate.structure_candidate_id: candidate for candidate in candidates}
        for event in reversed(active_accept_events):
            candidate = (
                candidate_by_id.get(event.structure_candidate_id)
                if event.structure_candidate_id
                else None
            )
            if not candidate and event.structure_candidate_id:
                candidate = self.structure_candidates.get(event.structure_candidate_id)
            global_material = self._global_material_from_accept(candidate, event)
            if global_material:
                self._upsert_restored_accepted_link(
                    run,
                    material,
                    candidate,
                    global_material,
                    event,
                )
                return True
        for candidate in accepted_candidates:
            global_material = self._global_material_from_accepted_candidate(candidate)
            if global_material:
                self._upsert_restored_accepted_link(
                    run,
                    material,
                    candidate,
                    global_material,
                    None,
                )
                return True
        return False

    def _global_material_from_accept(
        self,
        candidate: MaterialStructureCandidate | None,
        event: MaterialReviewEvent,
    ) -> MaterialGlobal | None:
        global_material_id = None
        if event.after_link:
            raw_global_id = event.after_link.get("global_material_id")
            if isinstance(raw_global_id, str) and raw_global_id:
                global_material_id = raw_global_id
        global_material_id = (
            global_material_id or event.global_material_id or event.created_global_material_id
        )
        if global_material_id:
            material = self.global_materials.get(global_material_id)
            if material and material.review_status != "retracted":
                return material
        return self._global_material_from_accepted_candidate(candidate)

    def _global_material_from_accepted_candidate(
        self,
        candidate: MaterialStructureCandidate | None,
    ) -> MaterialGlobal | None:
        if not candidate or not candidate.inchi_key:
            return None
        material = self.global_materials.get_by_inchi_key(candidate.inchi_key)
        if material and material.review_status != "retracted":
            return material
        return None

    def _upsert_restored_accepted_link(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
        candidate: MaterialStructureCandidate | None,
        global_material: MaterialGlobal,
        event: MaterialReviewEvent | None,
    ) -> None:
        timestamp = now_iso()
        event_link = (
            PaperMaterialLink.model_validate(event.after_link)
            if event and event.after_link
            else None
        )
        evidence = dict(event_link.evidence) if event_link else {}
        if candidate:
            evidence.update(
                {
                    "structure_candidate_id": candidate.structure_candidate_id,
                    "provider": candidate.provider,
                    "source_identifier": candidate.source_identifier,
                    "source_url": candidate.source_url,
                    "query_text": candidate.query_text,
                }
            )
        evidence.update(
            {
                "restored_from": "accepted_structure_candidate",
                "source_candidates": _mention_candidates(material),
            }
        )
        self._add_paper_local_aliases_for_global_material(
            run,
            material,
            global_material,
            source=candidate.provider if candidate else "accepted_candidate_restore",
            confidence=(
                event_link.match_confidence
                if event_link
                else candidate.confidence
                if candidate
                else global_material.confidence
            ),
        )
        self.links.upsert(
            PaperMaterialLink(
                paper_material_link_id=uuid4().hex,
                paper_id=run.paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=material.paper_material_id,
                global_material_id=global_material.global_material_id,
                match_method=(
                    event_link.match_method
                    if event_link
                    else f"{candidate.provider}_candidate"
                    if candidate
                    else "accepted_review_event"
                ),
                match_confidence=(
                    event_link.match_confidence
                    if event_link
                    else candidate.confidence
                    if candidate
                    else global_material.confidence
                ),
                match_status="matched_candidate",
                evidence=evidence,
                created_at=timestamp,
                updated_at=timestamp,
                confirmed_at=(
                    event_link.confirmed_at
                    if event_link and event_link.confirmed_at
                    else event.created_at
                    if event
                    else timestamp
                ),
            )
        )
        self._upsert_completed_accepted_task(run, material)

    def _add_paper_local_aliases_for_global_material(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
        global_material: MaterialGlobal,
        *,
        source: str,
        confidence: float | None,
    ) -> None:
        timestamp = now_iso()
        for alias_text, alias_type in _paper_local_alias_items(material):
            normalized_alias = normalize_material_alias(alias_text)
            if has_confirmed_alias_conflict(
                self.aliases.find_by_normalized(normalized_alias),
                global_material.global_material_id,
            ):
                continue
            self.aliases.add_if_missing(
                MaterialAlias(
                    alias_id=uuid4().hex,
                    global_material_id=global_material.global_material_id,
                    alias_text=alias_text,
                    normalized_alias=normalized_alias,
                    alias_type=alias_type,
                    source_paper_id=run.paper_id,
                    source=source,
                    confidence=confidence,
                    review_status="confirmed",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )

    def _upsert_completed_accepted_task(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
    ) -> MaterialResolutionTask:
        timestamp = now_iso()
        base = self.tasks.get_by_paper_material(
            run.candidate_run_id,
            material.paper_material_id,
        ) or MaterialResolutionTask(
            task_id=uuid4().hex,
            paper_id=run.paper_id,
            candidate_run_id=run.candidate_run_id,
            paper_material_id=material.paper_material_id,
            material_mentions=[candidate["text"] for candidate in _mention_candidates(material)],
            material_context={
                "full_name_in_paper": material.full_name_in_paper,
                "normalized_name": material.normalized_name,
                "abbreviation": material.abbreviation,
                "paper_specific_label": material.paper_specific_label,
                "material_class": material.material_class,
                "used_in": [usage.model_dump(mode="json") for usage in material.used_in],
            },
            priority=_material_priority(material),
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self.tasks.upsert(
            base.model_copy(
                update={
                    "status": "completed",
                    "assigned_strategy": "human_accepted_public_candidate",
                    "error_message": None,
                    "completed_at": base.completed_at or timestamp,
                    "updated_at": timestamp,
                }
            )
        )

    def _lookup_local_material(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        candidate_matches: list[tuple[dict[str, Any], list[MaterialGlobal]]] = []
        for candidate in candidates:
            normalized = candidate["normalized_alias"]
            if not normalized:
                continue
            aliases = [
                alias
                for alias in self.aliases.find_by_normalized(normalized)
                if _is_trusted_local_alias(alias)
            ]
            if not aliases:
                continue
            matched_materials = []
            seen_global_ids: set[str] = set()
            for alias in aliases:
                if alias.global_material_id in seen_global_ids:
                    continue
                material = self.global_materials.get(alias.global_material_id)
                if not material or material.review_status == "retracted":
                    continue
                seen_global_ids.add(alias.global_material_id)
                matched_materials.append(material)
            if not matched_materials:
                continue
            candidate_matches.append((candidate, matched_materials))
        if not candidate_matches:
            return {"status": "not_found"}

        matched_by_id = {
            material.global_material_id: material
            for _, matched_materials in candidate_matches
            for material in matched_materials
        }
        matched_materials = list(matched_by_id.values())
        selected: MaterialGlobal | None = None
        method = "normalized_alias"
        if len(matched_materials) == 1:
            selected = matched_materials[0]
        else:
            selected = _preferred_confirmed_structured_material(matched_materials)
            if selected:
                method = "normalized_alias_preferred_confirmed"
            else:
                selected = _preferred_equivalent_structured_material(matched_materials)
                if selected:
                    method = "normalized_alias_equivalent_structure"
        if selected is None:
            matched_aliases = [candidate["text"] for candidate, _ in candidate_matches]
            return {
                "status": "ambiguous",
                "matched_alias": matched_aliases[0],
                "matched_aliases": matched_aliases,
                "normalized_alias": candidate_matches[0][0]["normalized_alias"],
                "candidate_global_material_ids": sorted(matched_by_id),
            }

        matching_candidates = [
            candidate
            for candidate, materials in candidate_matches
            if any(
                material.global_material_id == selected.global_material_id
                for material in materials
            )
        ]
        matched_alias = matching_candidates[0]
        is_confirmed = selected.review_status == "confirmed"
        return {
            "status": "matched",
            "global_material_id": selected.global_material_id,
            "matched_alias": matched_alias["text"],
            "matched_aliases": [candidate["text"] for candidate in matching_candidates],
            "normalized_alias": matched_alias["normalized_alias"],
            "method": method,
            "confidence": 0.97 if len(candidate_matches) > 1 else 0.95 if is_confirmed else 0.85,
            "match_status": "matched_local" if is_confirmed else "candidate_match",
            "candidate_global_material_ids": sorted(matched_by_id),
        }

    def _upsert_task(
        self,
        run: CandidateIngestionRun,
        material: PaperLocalMaterial,
        status: str,
        assigned_strategy: str,
        *,
        extra_context: dict[str, Any] | None = None,
    ) -> MaterialResolutionTask:
        timestamp = now_iso()
        material_context = {
            "full_name_in_paper": material.full_name_in_paper,
            "normalized_name": material.normalized_name,
            "abbreviation": material.abbreviation,
            "paper_specific_label": material.paper_specific_label,
            "material_class": material.material_class,
            "used_in": [usage.model_dump(mode="json") for usage in material.used_in],
        }
        if extra_context:
            material_context.update(extra_context)
        return self.tasks.upsert(
            MaterialResolutionTask(
                task_id=uuid4().hex,
                paper_id=run.paper_id,
                candidate_run_id=run.candidate_run_id,
                paper_material_id=material.paper_material_id,
                material_mentions=[
                    candidate["text"] for candidate in _mention_candidates(material)
                ],
                material_context=material_context,
                priority=_material_priority(material),
                status=status,
                assigned_strategy=assigned_strategy,
                current_stage="completed" if status == "completed" else "unresolved",
                next_action="none" if status == "completed" else "resolve",
                created_at=timestamp,
                updated_at=timestamp,
                completed_at=timestamp if status == "completed" else None,
            )
        )


def normalize_material_alias(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ν": "nu",
        "μ": "mu",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def is_reusable_material_alias_text(value: str | None) -> bool:
    if not value:
        return False
    normalized = normalize_material_alias(value)
    if not normalized or normalized in NON_REUSABLE_GLOBAL_ALIAS_NORMALIZED:
        return False
    stripped = value.strip()
    return not (
        PAPER_SPECIFIC_PREFIX_RE.fullmatch(stripped)
        or SHORT_LABEL_RE.fullmatch(stripped)
        or (
            any(character.isdigit() for character in stripped)
            and EMPIRICAL_FORMULA_RE.fullmatch(stripped)
        )
    )


def has_confirmed_alias_conflict(
    aliases: list[MaterialAlias],
    global_material_id: str,
) -> bool:
    return any(
        alias.review_status == "confirmed"
        and alias.global_material_id != global_material_id
        for alias in aliases
    )


def classify_material_structure_scope(material: PaperLocalMaterial) -> dict[str, Any]:
    mention_candidates = _mention_candidates(material)
    normalized_aliases = [
        (candidate["text"], candidate["normalized_alias"])
        for candidate in mention_candidates
        if candidate["normalized_alias"]
    ]
    usage_payload = [usage.model_dump(mode="json") for usage in material.used_in]
    component_roles = {
        str(usage.component_role).strip().lower()
        for usage in material.used_in
        if usage.component_role
    }
    layer_roles = {
        str(usage.layer_role).strip().lower() for usage in material.used_in if usage.layer_role
    }
    material_class = (material.material_class or "").strip().lower()

    if material_class in NON_MATERIAL_REFERENCE_CLASSES:
        return _structure_scope_payload(
            category="out_of_scope_structure",
            confidence=0.99,
            reason=(
                "Generic layer, role, or comparison label extracted as a paper material; "
                "it is not a uniquely named chemical identity and must not be assigned a SMILES."
            ),
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="material_class_non_material_reference",
            usage=usage_payload,
        )

    for alias_text, normalized in normalized_aliases:
        if normalized in OUT_OF_SCOPE_STRUCTURE_ALIASES:
            return _structure_scope_payload(
                category="out_of_scope_structure",
                confidence=0.98,
                reason=OUT_OF_SCOPE_STRUCTURE_ALIASES[normalized],
                matched_alias=alias_text,
                rule="known_non_molecular_oled_material",
                usage=usage_payload,
            )
    if component_roles & OUT_OF_SCOPE_COMPONENT_ROLES or layer_roles & OUT_OF_SCOPE_LAYER_ROLES:
        return _structure_scope_payload(
            category="out_of_scope_structure",
            confidence=0.94,
            reason="OLED electrode, substrate, cathode/anode, or encapsulation material; molecular SMILES is not meaningful for the first-phase organic small-molecule database.",
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="device_role_non_molecular",
            usage=usage_payload,
        )

    for alias_text, normalized in normalized_aliases:
        if normalized in IDENTITY_ONLY_ALIASES:
            return _structure_scope_payload(
                category="identity_only",
                confidence=0.96,
                reason=IDENTITY_ONLY_ALIASES[normalized],
                matched_alias=alias_text,
                rule="known_identity_only_oled_auxiliary_material",
                usage=usage_payload,
            )
    if _material_class_is_polymer_or_composite(material_class):
        return _structure_scope_payload(
            category="identity_only",
            confidence=0.9,
            reason="LLM classified this material as a polymer, blend, or composite; record identity for OLED device context, but skip first-phase small-molecule SMILES/OCSR resolution.",
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="material_class_polymer_identity_only",
            usage=usage_payload,
        )
    if material_class in IDENTITY_ONLY_NON_SINGLE_STRUCTURE_CLASSES:
        return _structure_scope_payload(
            category="identity_only",
            confidence=0.94,
            reason=(
                "Material is a mixture or proprietary formulation without one verified "
                "small-molecule graph; preserve its OLED device identity without inventing "
                "a single SMILES representation."
            ),
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="material_class_non_single_structure_identity_only",
            usage=usage_payload,
        )
    for alias_text, _normalized in normalized_aliases:
        if DENDRIMER_HINT_RE.search(alias_text):
            return _structure_scope_payload(
                category="identity_only",
                confidence=0.92,
                reason=(
                    "Dendrimer identity is relevant to the OLED device, but symbolic or "
                    "repeat-unit depictions do not provide one reliably extractable "
                    "small-molecule SMILES representation for the first-phase database."
                ),
                matched_alias=alias_text,
                rule="dendrimer_identity_only",
                usage=usage_payload,
            )
    for alias_text, normalized in normalized_aliases:
        if _looks_like_polymer_or_composite(alias_text, normalized):
            return _structure_scope_payload(
                category="identity_only",
                confidence=0.88,
                reason="Polymer, blend, or composite material; record identity for OLED device context, but skip first-phase small-molecule SMILES/OCSR resolution.",
                matched_alias=alias_text,
                rule="polymer_or_composite_identity_only",
                usage=usage_payload,
            )
    if _material_class_is_coordination_complex(material_class):
        return _structure_scope_payload(
            category="identity_only",
            confidence=0.9,
            reason="LLM classified this material as an organometallic or coordination complex; first-phase database targets organic small-molecule emitters, so record identity and defer specialized representation.",
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="material_class_coordination_complex_identity_only",
            usage=usage_payload,
        )
    for alias_text, _normalized in normalized_aliases:
        if _looks_like_coordination_complex(alias_text):
            return _structure_scope_payload(
                category="identity_only",
                confidence=0.86,
                reason="Likely organometallic or coordination complex; first-phase database targets organic small-molecule emitters, so record identity and defer specialized representation.",
                matched_alias=alias_text,
                rule="coordination_complex_identity_only",
                usage=usage_payload,
            )
    if material_class in {"inorganic", "salt"} and (
        component_roles & IDENTITY_ONLY_COMPONENT_ROLES or layer_roles & IDENTITY_ONLY_LAYER_ROLES
    ):
        return _structure_scope_payload(
            category="identity_only",
            confidence=0.88,
            reason="Auxiliary inorganic/salt injection material; record identity, but do not block review on small-molecule SMILES.",
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="auxiliary_injection_material",
            usage=usage_payload,
        )
    if material_class == "inorganic":
        return _structure_scope_payload(
            category="out_of_scope_structure",
            confidence=0.92,
            reason="Inorganic material is outside the first-phase organic small-molecule structure database; preserve its device-layer identity without requiring molecular SMILES.",
            matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
            rule="material_class_inorganic_non_molecular",
            usage=usage_payload,
        )

    return _structure_scope_payload(
        category="core_structure_required",
        confidence=0.5,
        reason="Potentially relevant organic OLED material; continue public database, OCSR, or manual structure resolution.",
        matched_alias=normalized_aliases[0][0] if normalized_aliases else material.entity_label,
        rule="default_core_resolution",
        usage=usage_payload,
    )


def _looks_like_polymer_or_composite(alias_text: str, normalized: str) -> bool:
    compact = normalized.lower()
    if compact.startswith("poly") and len(compact) > 6:
        return True
    if POLYMER_HINT_RE.search(alias_text):
        return True
    return bool(
        COMPOSITE_HINT_RE.search(alias_text)
        and any(token in compact for token in ("pedot", "pss", "pmma", "pvk"))
    )


def _looks_like_coordination_complex(alias_text: str) -> bool:
    if ORGANOMETALLIC_COMPLEX_RE.search(alias_text):
        return True
    if ORGANOMETALLIC_WORD_RE.search(alias_text):
        return True
    return bool(ORGANOMETALLIC_FORMULA_PREFIX_RE.search(alias_text.strip()))


def _material_class_is_polymer_or_composite(material_class: str) -> bool:
    compact = material_class.strip().lower().replace("-", "_").replace(" ", "_")
    return compact in IDENTITY_ONLY_POLYMER_MATERIAL_CLASSES or any(
        (
            compact.startswith("polymer_"),
            compact.startswith("copolymer_"),
            compact.endswith("_polymer"),
            compact.endswith("_copolymer"),
        )
    )


def _material_class_is_coordination_complex(material_class: str) -> bool:
    compact = material_class.strip().lower().replace("-", "_").replace(" ", "_")
    return compact in IDENTITY_ONLY_MATERIAL_CLASSES or any(
        token in compact
        for token in (
            "organometallic",
            "coordination_complex",
            "metal_complex",
            "transition_metal_complex",
        )
    )


def _structure_scope_payload(
    *,
    category: str,
    confidence: float,
    reason: str,
    matched_alias: str | None,
    rule: str,
    usage: list[dict[str, Any]],
) -> dict[str, Any]:
    requires_structure = category == "core_structure_required"
    return {
        "category": category,
        "requires_structure": requires_structure,
        "requires_public_resolution": requires_structure,
        "matched_alias": matched_alias,
        "rule": rule,
        "reason": reason,
        "confidence": confidence,
        "first_phase_scope": "organic_small_molecule_oled",
        "usage": usage,
    }


def _candidate_values_by_entity_id(values: list[CandidateValue]) -> dict[str, list[CandidateValue]]:
    values_by_entity_id: dict[str, list[CandidateValue]] = {}
    for value in values:
        values_by_entity_id.setdefault(value.candidate_entity_id, []).append(value)
    return values_by_entity_id


def _paper_local_alias_items(
    material: PaperLocalMaterial,
) -> list[tuple[str, str]]:
    alias_items = [
        (material.abbreviation, "paper_local_abbreviation"),
        *((mention, "paper_local_mention") for mention in material.mention_list),
        (material.entity_label, "paper_local_entity_label"),
        (material.normalized_name, "paper_local_normalized_name"),
        (material.full_name_in_paper, "paper_local_full_name"),
        (material.canonical_name, "paper_local_canonical_name"),
    ]
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias_text, alias_type in alias_items:
        normalized = normalize_material_alias(alias_text)
        if (
            not alias_text
            or not normalized
            or normalized in seen
            or not is_reusable_material_alias_text(alias_text)
        ):
            continue
        seen.add(normalized)
        deduped.append((alias_text, alias_type))
    return deduped


def _active_accept_events(
    events: list[MaterialReviewEvent],
    paper_material_id: str,
) -> list[MaterialReviewEvent]:
    accept_events = [
        event
        for event in events
        if event.action == "accept" and event.paper_material_id == paper_material_id
    ]
    undo_events = [
        event
        for event in events
        if event.action == "undo" and event.paper_material_id == paper_material_id
    ]
    return [
        event
        for event in accept_events
        if not any(
            undo_event.structure_candidate_id == event.structure_candidate_id
            and undo_event.created_at > event.created_at
            for undo_event in undo_events
        )
    ]


def _is_protected_material_link(link: PaperMaterialLink | None) -> bool:
    if not link or not link.global_material_id:
        return False
    return link.match_status == "matched_candidate"


def _preferred_confirmed_structured_material(
    materials: list[MaterialGlobal],
) -> MaterialGlobal | None:
    confirmed_structured = [
        material
        for material in materials
        if material.review_status == "confirmed" and _has_structural_identifier(material)
    ]
    if len(confirmed_structured) != 1:
        return None
    preferred = confirmed_structured[0]
    others = [
        material
        for material in materials
        if material.global_material_id != preferred.global_material_id
    ]
    if all(_is_unconfirmed_placeholder_material(material) for material in others):
        return preferred
    return None


def _preferred_equivalent_structured_material(
    materials: list[MaterialGlobal],
) -> MaterialGlobal | None:
    confirmed_structured = [
        material
        for material in materials
        if material.review_status == "confirmed" and _has_structural_identifier(material)
    ]
    if len(confirmed_structured) < 2:
        return None
    structure_keys = {_material_structure_key(material) for material in confirmed_structured}
    if None in structure_keys or len(structure_keys) != 1:
        return None
    remaining = [material for material in materials if material not in confirmed_structured]
    if not all(_is_unconfirmed_placeholder_material(material) for material in remaining):
        return None
    return max(
        confirmed_structured,
        key=lambda material: (material.confidence or 0.0, material.updated_at),
    )


def _material_structure_key(material: MaterialGlobal) -> tuple[str, str] | None:
    for kind, value in (
        ("inchi_key", material.inchi_key),
        ("inchi", material.inchi),
        ("canonical_smiles", material.canonical_smiles),
        ("isomeric_smiles", material.isomeric_smiles),
        ("raw_smiles", material.raw_smiles),
    ):
        if value:
            return kind, value.strip()
    return None


def _is_trusted_local_alias(alias: MaterialAlias) -> bool:
    # Public-database synonym lists are useful provenance but can contain noisy
    # vendor names or unrelated cross-references. They require identity review
    # and must not silently create a paper-to-global structure link.
    if alias.review_status == "rejected" or not is_reusable_material_alias_text(
        alias.alias_text
    ):
        return False
    if alias.alias_type != "structure_candidate":
        return True
    # A confirmed alias bound while reviewing a specific paper is a curated
    # paper-local identity, not an unreviewed synonym imported from a public
    # candidate. Reuse it so common OLED materials are not reviewed repeatedly.
    return alias.review_status == "confirmed" and bool(alias.source_paper_id)


def _has_structural_identifier(material: MaterialGlobal) -> bool:
    return any(
        (
            material.inchi_key,
            material.inchi,
            material.canonical_smiles,
            material.isomeric_smiles,
            material.raw_smiles,
        )
    )


def _is_unconfirmed_placeholder_material(material: MaterialGlobal) -> bool:
    return (
        material.review_status != "confirmed"
        and not _has_structural_identifier(material)
        and material.source in {"common_oled_seed", "manual_seed", "seed"}
    )


def _material_source_from_candidate_values(
    source_json: Mapping[str, Any],
    values: list[CandidateValue],
) -> dict[str, Any]:
    source = dict(source_json)
    for value in values:
        if not value.template_field_path.startswith("materials[]."):
            continue
        field_name = value.template_field_path.rsplit(".", 1)[-1]
        if field_name == "paper_material_id":
            continue
        source[field_name] = (
            value.reviewed_value_json if value.reviewed_value_json is not None else value.value_json
        )
    return source


def _material_entity_label(source: Mapping[str, Any]) -> str | None:
    for key in (
        "abbreviation",
        "normalized_name",
        "canonical_name",
        "full_name_in_paper",
        "paper_material_id",
    ):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    mentions = source.get("mention_list")
    if isinstance(mentions, list):
        for mention in mentions:
            if isinstance(mention, str) and mention.strip():
                return mention.strip()
    return None


def _apply_material_name_review(
    material: PaperLocalMaterial,
    review: PaperMaterialNameReview | None,
) -> PaperLocalMaterial:
    if not review:
        return material
    reviewed_name = _clean_text(review.reviewed_name)
    mention_list = material.mention_list
    if reviewed_name:
        mention_list = _dedupe_text([reviewed_name, *material.mention_list])
    return material.model_copy(
        update={
            "entity_label": reviewed_name or material.entity_label,
            "mention_list": mention_list,
            "full_name_in_paper": review.reviewed_full_name_in_paper or material.full_name_in_paper,
            "normalized_name": review.reviewed_normalized_name or material.normalized_name,
            "canonical_name": review.reviewed_canonical_name or material.canonical_name,
            "abbreviation": review.reviewed_abbreviation or reviewed_name or material.abbreviation,
        }
    )


def _material_entity_id_for_paper_material(
    entities: list[CandidateEntity],
    values: list[CandidateValue],
    paper_material_id: str,
) -> str | None:
    for entity in entities:
        if entity.entity_type != "materials":
            continue
        source_id = _string(entity.source_json.get("paper_material_id"))
        if source_id == paper_material_id:
            return entity.candidate_entity_id
    for value in values:
        if value.template_field_path != "materials[].paper_material_id":
            continue
        candidate_id = (
            value.reviewed_value_json if value.reviewed_value_json is not None else value.value_json
        )
        if candidate_id == paper_material_id:
            return value.candidate_entity_id
    return None


def _material_name_candidate_value_updates(
    material: PaperLocalMaterial,
    review: PaperMaterialNameReview,
) -> dict[str, object]:
    updates: dict[str, object] = {}
    reviewed_name = _clean_text(review.reviewed_name)
    if reviewed_name:
        updates["materials[].mention_list"] = _dedupe_text([reviewed_name, *material.mention_list])
        updates["materials[].abbreviation"] = review.reviewed_abbreviation or reviewed_name
    if review.reviewed_abbreviation:
        updates["materials[].abbreviation"] = review.reviewed_abbreviation
    if review.reviewed_full_name_in_paper:
        updates["materials[].full_name_in_paper"] = review.reviewed_full_name_in_paper
    if review.reviewed_normalized_name:
        updates["materials[].normalized_name"] = review.reviewed_normalized_name
    if review.reviewed_canonical_name:
        updates["materials[].canonical_name"] = review.reviewed_canonical_name
    return updates


def _agent_material_name_suggestion(
    run: CandidateIngestionRun,
    material: PaperLocalMaterial,
    *,
    pubchem_client: PublicCompoundResolverClient | None = None,
    anysearch_client: MaterialWebSearchClient | None = None,
) -> PaperMaterialNameSuggestion | None:
    names = _dedupe_text(
        [
            material.abbreviation,
            material.normalized_name,
            material.canonical_name,
            material.full_name_in_paper,
            material.entity_label,
            *material.mention_list,
        ]
    )
    normalized_names = {normalize_material_alias(name) for name in names}
    if "irppy" not in normalized_names:
        return None

    reference_evidence: list[dict[str, Any]] = [
        {
            "source_type": "deterministic_rule",
            "title": "Known OLED alias sanity check",
            "status": "supports_suggestion",
            "excerpt": (
                "Ir(ppy) is a suspicious truncated alias for tris(2-phenylpyridine)"
                "iridium(III), commonly written as Ir(ppy)3 in OLED literature."
            ),
            "matched_terms": ["Ir(ppy)", "Ir(ppy)3"],
        }
    ]
    reference_evidence.extend(_pubchem_name_references(pubchem_client, "Ir(ppy)3"))
    reference_evidence.extend(
        _web_name_references(
            anysearch_client,
            [
                '"Ir(ppy)3" OLED material',
                '"Ir(ppy)3" "Ir(ppy)" OLED',
                '"tris(2-phenylpyridine)iridium" "Ir(ppy)3"',
            ],
        )
    )
    timestamp = now_iso()
    already_reviewed = "irppy3" in normalized_names
    return PaperMaterialNameSuggestion(
        suggestion_id=uuid4().hex,
        paper_id=run.paper_id,
        candidate_run_id=run.candidate_run_id,
        paper_material_id=material.paper_material_id,
        agent_name="material_name_agent_v2",
        original_name=_preferred_original_name(names, "irppy") or "Ir(ppy)",
        suggested_name="Ir(ppy)3",
        suggested_abbreviation="Ir(ppy)3",
        suggested_normalized_name="Ir(ppy)3",
        confidence=0.9 if not already_reviewed else 0.78,
        reason=(
            "Material name verification found a likely truncated OLED sensitizer alias. "
            "Keep the paper-local ID unchanged, but review whether the display name should be "
            "Ir(ppy)3. Public/reference evidence is attached for human review."
        ),
        evidence={
            "agent_version": "material_name_agent_v2",
            "verification_status": "already_reviewed" if already_reviewed else "needs_human_review",
            "source_names": names,
            "normalized_names": sorted(normalized_names),
            "suggested_name": "Ir(ppy)3",
            "checks": [
                {
                    "check_type": "alias_truncation",
                    "status": "triggered",
                    "detail": "Found normalized alias irppy, which is commonly a dropped-stoichiometry form of Ir(ppy)3.",
                },
                {
                    "check_type": "public_reference_lookup",
                    "status": "completed_with_evidence"
                    if any(item.get("status") == "found" for item in reference_evidence)
                    else "no_external_hit_or_unconfigured",
                },
            ],
            "reference_evidence": reference_evidence,
        },
        created_at=timestamp,
        updated_at=timestamp,
    )


def _preferred_original_name(names: list[str], normalized_target: str) -> str | None:
    for name in names:
        if normalize_material_alias(name) == normalized_target:
            return name
    return None


def _reviewed_normalized_name(
    action: PaperMaterialNameReviewAction,
    before_material: PaperLocalMaterial,
) -> str | None:
    explicit = _clean_text(action.reviewed_normalized_name)
    reviewed_name = _clean_text(action.reviewed_name)
    reviewed_abbreviation = _clean_text(action.reviewed_abbreviation)
    old_display_names = set(
        _dedupe_text(
            [
                before_material.normalized_name,
                before_material.abbreviation,
                before_material.entity_label,
                *before_material.mention_list,
            ]
        )
    )
    if explicit and reviewed_name and explicit in old_display_names and reviewed_name != explicit:
        return reviewed_name
    if explicit:
        return explicit
    return reviewed_name or reviewed_abbreviation


def _pubchem_name_references(
    pubchem_client: PublicCompoundResolverClient | None,
    query_text: str,
) -> list[dict[str, Any]]:
    if pubchem_client is None:
        return [
            {
                "source_type": "pubchem",
                "status": "skipped",
                "query_text": query_text,
                "reason": "PubChem client is not configured.",
            }
        ]
    try:
        compounds = pubchem_client.resolve_name(query_text, max_results=3)
    except Exception as exc:
        return [
            {
                "source_type": "pubchem",
                "status": "failed",
                "query_text": query_text,
                "error": str(exc),
            }
        ]
    if not compounds:
        return [
            {
                "source_type": "pubchem",
                "status": "not_found",
                "query_text": query_text,
            }
        ]
    return [_pubchem_reference_payload(compound) for compound in compounds]


def _pubchem_reference_payload(compound: PubChemCompound) -> dict[str, Any]:
    synonyms = [item for item in compound.synonyms if item][:8]
    title = compound.iupac_name or (synonyms[0] if synonyms else f"PubChem CID {compound.cid}")
    matched_terms = [
        item
        for item in [compound.query_text, compound.iupac_name, *synonyms]
        if isinstance(item, str) and item
    ]
    return {
        "source_type": "pubchem",
        "status": "found",
        "title": title,
        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}",
        "query_text": compound.query_text,
        "pubchem_cid": compound.cid,
        "formula": compound.formula,
        "inchi_key": compound.inchi_key,
        "canonical_smiles": compound.canonical_smiles,
        "matched_terms": matched_terms[:10],
        "excerpt": "; ".join(matched_terms[:4]),
    }


def _web_name_references(
    anysearch_client: MaterialWebSearchClient | None,
    query_plan: list[str],
) -> list[dict[str, Any]]:
    if anysearch_client is None:
        return [
            {
                "source_type": "web_search",
                "status": "skipped",
                "reason": "AnySearch API key is not configured.",
                "query_plan": query_plan,
            }
        ]
    evidence: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for query in query_plan:
        try:
            results = anysearch_client.search(query, max_results=3)
        except Exception as exc:
            evidence.append(
                {
                    "source_type": "web_search",
                    "status": "failed",
                    "query_text": query,
                    "error": str(exc),
                }
            )
            continue
        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            evidence.append(_web_reference_payload(result, query_text=query))
    if not evidence:
        return [
            {
                "source_type": "web_search",
                "status": "not_found",
                "query_plan": query_plan,
            }
        ]
    return evidence[:6]


def _web_reference_payload(result: AnySearchResult, *, query_text: str) -> dict[str, Any]:
    excerpt = result.content[:700] or result.description[:700]
    return {
        "source_type": "web_search",
        "status": "found",
        "title": result.title,
        "url": result.url,
        "query_text": query_text,
        "excerpt": excerpt,
        "cas_numbers": result.cas_numbers(),
        "score": result.score,
    }


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    return stripped or None


def _paper_local_material_from_entity(
    entity: CandidateEntity,
    values: list[CandidateValue] | None = None,
) -> PaperLocalMaterial:
    source = _material_source_from_candidate_values(entity.source_json, values or [])
    material_class = _known_material_class_override(source) or _string(
        source.get("material_class")
    )
    return PaperLocalMaterial(
        paper_material_id=_string(source.get("paper_material_id"))
        or entity.entity_label
        or entity.entity_path,
        entity_path=entity.entity_path,
        entity_label=_material_entity_label(source) or entity.entity_label,
        mention_list=_string_list(source.get("mention_list")),
        full_name_in_paper=_string(source.get("full_name_in_paper")),
        normalized_name=_string(source.get("normalized_name")),
        canonical_name=_string(source.get("canonical_name")),
        abbreviation=_string(source.get("abbreviation")),
        paper_specific_label=_string(source.get("paper_specific_label")),
        material_class=material_class,
        smiles=_string(source.get("smiles")),
        inchi=_string(source.get("inchi")),
        inchi_key=_string(source.get("inchi_key")),
        structure_source=_string(source.get("structure_source")),
        structure_confidence=_float(source.get("structure_confidence")),
        evidence_refs=_string_list(source.get("evidence_refs")),
    )


def _known_material_class_override(source: Mapping[str, Any]) -> str | None:
    names = [
        source.get("abbreviation"),
        source.get("normalized_name"),
        source.get("canonical_name"),
        source.get("paper_specific_label"),
        *(_string_list(source.get("mention_list"))),
    ]
    for name in names:
        material_class = KNOWN_MATERIAL_CLASS_OVERRIDES.get(
            normalize_material_alias(_string(name))
        )
        if material_class:
            return material_class
    return None


def _material_usage_by_id(payload: Mapping[str, Any]) -> dict[str, list[MaterialUsage]]:
    usage: dict[str, list[MaterialUsage]] = {}
    devices = payload.get("devices")
    if not isinstance(devices, list):
        return usage
    for device in devices:
        if not isinstance(device, Mapping):
            continue
        device_label = _string(device.get("device_label"))
        final_emitter = device.get("final_emitter")
        if isinstance(final_emitter, Mapping):
            material_id = _string(final_emitter.get("paper_material_id"))
            if material_id:
                usage.setdefault(material_id, []).append(
                    MaterialUsage(
                        device_label=device_label,
                        component_role="final_emitter",
                        material_mention=_string(final_emitter.get("mention")),
                        usage_type="final_emitter",
                    )
                )
        layers = device.get("layers")
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, Mapping):
                continue
            components = layer.get("components")
            if not isinstance(components, list):
                continue
            for component in components:
                if not isinstance(component, Mapping):
                    continue
                material_id = _string(component.get("paper_material_id"))
                if not material_id:
                    continue
                usage.setdefault(material_id, []).append(
                    MaterialUsage(
                        device_label=device_label,
                        layer_index=_int(layer.get("layer_index")),
                        layer_role=_string(layer.get("layer_role")),
                        layer_name=_string(layer.get("layer_name")),
                        component_role=_string(component.get("component_role")),
                        material_mention=_string(component.get("material_mention")),
                        usage_type="component",
                    )
                )
    return usage


def _device_used_or_all_materials(materials: list[PaperLocalMaterial]) -> list[PaperLocalMaterial]:
    device_used = [material for material in materials if material.used_in]
    return device_used or materials


def _filter_review_bundle_to_device_used_materials(
    *,
    materials: list[PaperLocalMaterial],
    material_name_reviews: list[PaperMaterialNameReview],
    material_name_suggestions: list[PaperMaterialNameSuggestion],
    links: list[PaperMaterialLink],
    tasks: list[MaterialResolutionTask],
    structure_candidates: list[MaterialStructureCandidate],
    identity_judgments: list[Any],
    identity_evidence_runs: list[Any],
    identity_evidence_items: list[Any],
    material_review_events: list[MaterialReviewEvent],
    property_candidates: list[Any],
    property_reviews: list[Any],
    property_review_events: list[Any],
):
    active_material_ids = {material.paper_material_id for material in materials if material.used_in}
    if not active_material_ids:
        return (
            materials,
            material_name_reviews,
            material_name_suggestions,
            links,
            tasks,
            structure_candidates,
            identity_judgments,
            identity_evidence_runs,
            identity_evidence_items,
            material_review_events,
            property_candidates,
            property_reviews,
            property_review_events,
        )

    materials = [
        material for material in materials if material.paper_material_id in active_material_ids
    ]
    material_name_reviews = [
        review
        for review in material_name_reviews
        if review.paper_material_id in active_material_ids
    ]
    material_name_suggestions = [
        suggestion
        for suggestion in material_name_suggestions
        if suggestion.paper_material_id in active_material_ids
    ]
    links = [link for link in links if link.paper_material_id in active_material_ids]
    tasks = [task for task in tasks if task.paper_material_id in active_material_ids]
    structure_candidates = [
        candidate
        for candidate in structure_candidates
        if candidate.paper_material_id in active_material_ids
    ]
    visible_candidate_ids = {candidate.structure_candidate_id for candidate in structure_candidates}
    identity_judgments = [
        judgment
        for judgment in identity_judgments
        if judgment.paper_material_id in active_material_ids
        and judgment.structure_candidate_id in visible_candidate_ids
    ]
    identity_evidence_runs = [
        run for run in identity_evidence_runs if run.paper_material_id in active_material_ids
    ]
    visible_evidence_run_ids = {run.evidence_run_id for run in identity_evidence_runs}
    identity_evidence_items = [
        item
        for item in identity_evidence_items
        if item.paper_material_id in active_material_ids
        and item.evidence_run_id in visible_evidence_run_ids
    ]
    material_review_events = [
        event for event in material_review_events if event.paper_material_id in active_material_ids
    ]
    property_candidates = [
        candidate
        for candidate in property_candidates
        if candidate.paper_material_id in active_material_ids
    ]
    visible_property_candidate_ids = {
        candidate.property_candidate_id for candidate in property_candidates
    }
    property_reviews = [
        review
        for review in property_reviews
        if review.paper_material_id in active_material_ids
        and review.property_candidate_id in visible_property_candidate_ids
    ]
    property_review_events = [
        event
        for event in property_review_events
        if event.paper_material_id in active_material_ids
        and (
            event.property_candidate_id is None
            or event.property_candidate_id in visible_property_candidate_ids
        )
    ]
    return (
        materials,
        material_name_reviews,
        material_name_suggestions,
        links,
        tasks,
        structure_candidates,
        identity_judgments,
        identity_evidence_runs,
        identity_evidence_items,
        material_review_events,
        property_candidates,
        property_reviews,
        property_review_events,
    )


def _mention_candidates(material: PaperLocalMaterial) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for text, alias_type in (
        (material.full_name_in_paper, "full_name"),
        (material.canonical_name, "canonical_name"),
        (material.normalized_name, "normalized_name"),
        (material.abbreviation, "abbreviation"),
        (material.paper_specific_label, "paper_specific_label"),
    ):
        _append_candidate(candidates, text, alias_type)
    for mention in material.mention_list:
        _append_candidate(candidates, mention, "mention")
    return candidates


def _append_candidate(candidates: list[dict[str, Any]], text: str | None, alias_type: str) -> None:
    if not text:
        return
    normalized = normalize_material_alias(text)
    if not normalized:
        return
    if any(item["normalized_alias"] == normalized for item in candidates):
        return
    candidates.append(
        {
            "text": text,
            "alias_type": alias_type,
            "normalized_alias": normalized,
            "is_paper_specific": _is_paper_specific_alias(text, alias_type),
        }
    )


def _is_paper_specific_alias(text: str, alias_type: str) -> bool:
    if alias_type == "paper_specific_label":
        return True
    stripped = text.strip()
    normalized = normalize_material_alias(stripped)
    if PAPER_SPECIFIC_PREFIX_RE.match(stripped):
        return True
    letter_count = sum(1 for char in normalized if char.isalpha())
    return bool(SHORT_LABEL_RE.match(normalized) and letter_count <= 3)


def _material_priority(material: PaperLocalMaterial) -> str:
    roles = {usage.component_role for usage in material.used_in if usage.component_role}
    if {"final_emitter", "emitter", "emitter_dopant", "sensitizer", "sensitizer_dopant"} & roles:
        return "high"
    if material.paper_specific_label:
        return "high"
    return "normal"


def _dedupe_text(values: list[str | None]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_material_alias(value)
        if not value or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _float(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) else None
