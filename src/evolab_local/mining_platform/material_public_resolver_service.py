from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import quote
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.anysearch_client import (
    AnySearchClient,
    AnySearchResult,
    MaterialWebSearchClient,
)
from evolab_local.mining_platform.external.opsin_client import (
    OpsinClient,
    OpsinCompound,
    SystematicNameResolverClient,
)
from evolab_local.mining_platform.external.pubchem_client import (
    PubChemClient,
    PubChemCompound,
    PublicCompoundResolverClient,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_chemistry import standardize_smiles
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    _mention_candidates,
    classify_material_structure_scope,
    normalize_material_alias,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialResolutionTask,
    MaterialStructureCandidate,
    PaperLocalMaterial,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialGlobalRepository,
    MaterialResolutionTaskRepository,
    MaterialStructureCandidateRepository,
    PaperMaterialLinkRepository,
    PaperRepository,
)


DIRECT_PUBCHEM_QUERY_MISMATCH_CONFIDENCE = 0.62


class MaterialPublicResolverService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        pubchem_client: PublicCompoundResolverClient | None = None,
        anysearch_client: MaterialWebSearchClient | None = None,
        opsin_client: SystematicNameResolverClient | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.global_materials = MaterialGlobalRepository(self.database)
        self.links = PaperMaterialLinkRepository(self.database)
        self.tasks = MaterialResolutionTaskRepository(self.database)
        self.structure_candidates = MaterialStructureCandidateRepository(self.database)
        self.material_resolution = MaterialResolutionService(config)
        self.pubchem_client = pubchem_client or PubChemClient(config.external_services.pubchem)
        self.anysearch_client = anysearch_client or (
            AnySearchClient(config.external_services.anysearch)
            if config.external_services.anysearch.api_key.strip()
            else None
        )
        self.opsin_client = opsin_client or OpsinClient(config.external_services.opsin)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def resolve_paper_public(
        self,
        paper_id: str,
        *,
        paper_material_id: str | None = None,
        max_queries_per_material: int = 2,
        max_results_per_query: int | None = None,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        bundle = self.material_resolution.resolve_paper_materials(paper_id)
        if not bundle or not bundle.candidate_run_id:
            return bundle
        materials = [
            material
            for material in bundle.materials
            if paper_material_id is None or material.paper_material_id == paper_material_id
        ]
        links_by_material = {link.paper_material_id: link for link in bundle.links}
        global_by_id = {
            material.global_material_id: material for material in bundle.global_materials
        }
        for material in materials:
            scope = classify_material_structure_scope(material)
            if not scope["requires_public_resolution"]:
                continue
            link = links_by_material.get(material.paper_material_id)
            linked_global = (
                global_by_id.get(link.global_material_id)
                if link and link.global_material_id
                else None
            )
            if (
                link
                and link.match_status in {"matched_local", "matched_candidate"}
                and linked_global
                and (linked_global.canonical_smiles or linked_global.inchi_key)
            ):
                continue
            self._resolve_material_with_pubchem(
                bundle,
                material,
                max_queries_per_material=max_queries_per_material,
                max_results_per_query=max_results_per_query,
            )
        return self.material_resolution.get_material_structure_bundle(bundle.paper_id)

    def resolve_material_public(
        self,
        paper_id: str,
        paper_material_id: str,
        *,
        max_queries_per_material: int = 2,
        max_results_per_query: int | None = None,
        force: bool = False,
    ) -> PaperMaterialStructureBundle | None:
        """Resolve one material without re-running local resolution for its whole paper."""
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if not bundle or not bundle.candidate_run_id:
            return bundle
        material = next(
            (item for item in bundle.materials if item.paper_material_id == paper_material_id),
            None,
        )
        if material is None:
            raise ValueError(f"Unknown paper material {paper_material_id!r} for {bundle.paper_id}.")
        if not classify_material_structure_scope(material)["requires_public_resolution"]:
            return bundle
        existing_task = self.tasks.get_by_paper_material(
            bundle.candidate_run_id,
            paper_material_id,
        )
        existing_public_candidates = [
            candidate
            for candidate in bundle.structure_candidates
            if candidate.paper_material_id == paper_material_id
            and candidate.provider in {"pubchem", "opsin"}
        ]
        if not force and (
            existing_public_candidates
            or (
                existing_task
                and existing_task.assigned_strategy
                in {
                    "public_database_not_found",
                    "public_database_review",
                    "opsin_name_review",
                    "web_search_to_pubchem_review",
                    "web_search_evidence_review",
                    "public_candidate_rejected_continue_resolution",
                }
            )
        ):
            return bundle
        self._resolve_material_with_pubchem(
            bundle,
            material,
            max_queries_per_material=max_queries_per_material,
            max_results_per_query=max_results_per_query,
        )
        return self.material_resolution.get_material_structure_bundle(bundle.paper_id)

    def list_structure_candidates(
        self,
        paper_id: str,
    ) -> list[MaterialStructureCandidate]:
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        return bundle.structure_candidates if bundle else []

    def _resolve_material_with_pubchem(
        self,
        bundle: PaperMaterialStructureBundle,
        material: PaperLocalMaterial,
        *,
        max_queries_per_material: int,
        max_results_per_query: int | None,
    ) -> None:
        if not bundle.candidate_run_id:
            return
        stored_candidates: list[MaterialStructureCandidate] = []
        web_search_evidence: list[dict[str, object]] = []
        resolver_warnings: list[str] = []
        query_texts = list(_public_query_texts(material))[:max_queries_per_material]
        if not query_texts:
            return
        for query_text, query_type in query_texts:
            try:
                compounds = self.pubchem_client.resolve_name(
                    query_text,
                    max_results=max_results_per_query,
                )
            except Exception as exc:
                self._upsert_task(
                    bundle,
                    material,
                    status="failed",
                    assigned_strategy="public_database_failed",
                    error_message=str(exc),
                    candidate_ids=[],
                )
                return
            for compound in compounds:
                candidate = _candidate_from_pubchem_compound(
                    bundle=bundle,
                    material=material,
                    compound=compound,
                    query_text=query_text,
                    query_type=query_type,
                )
                if not is_plausible_public_structure_candidate(candidate):
                    resolver_warnings.append(
                        "Ignored direct PubChem candidate with no lexical identity match: "
                        f"query={query_text!r}, cid={compound.cid}"
                    )
                    continue
                stored_candidates.append(self._store_candidate_preserving_review(candidate))
        # A short OLED abbreviation can be a valid PubChem synonym for a completely
        # different material. Always resolve explicit paper full names independently;
        # an abbreviation hit must never suppress the stronger name-derived candidate.
        for query_text, query_type in query_texts:
            if query_type not in {"full_name", "canonical_name"}:
                continue
            try:
                compound = self.opsin_client.resolve_name(query_text)
            except Exception as exc:
                resolver_warnings.append(f"OPSIN query failed for {query_text}: {exc}")
                continue
            if compound:
                opsin_candidate = _candidate_from_opsin_compound(
                    bundle=bundle,
                    material=material,
                    compound=compound,
                    query_text=query_text,
                    query_type=query_type,
                    source_url=(
                        f"{self.config.external_services.opsin.base_url.rstrip('/')}/"
                        f"{quote(query_text, safe='')}.json"
                    ),
                )
                if not _contains_equivalent_structure(stored_candidates, opsin_candidate):
                    stored_candidates.append(
                        self._store_candidate_preserving_review(opsin_candidate)
                    )

        if self.anysearch_client and _needs_web_identity_support(stored_candidates):
            try:
                web_candidates, web_search_evidence = self._resolve_via_anysearch_identifiers(
                    bundle,
                    material,
                    query_texts=query_texts,
                    max_results_per_query=max_results_per_query,
                )
                known_candidate_ids = {
                    candidate.structure_candidate_id for candidate in stored_candidates
                }
                stored_candidates.extend(
                    candidate
                    for candidate in web_candidates
                    if candidate.structure_candidate_id not in known_candidate_ids
                    and not _contains_equivalent_structure(stored_candidates, candidate)
                )
            except Exception as exc:
                self._upsert_task(
                    bundle,
                    material,
                    status="failed",
                    assigned_strategy="web_search_failed",
                    error_message=str(exc),
                    candidate_ids=[],
                    extra_context={"resolver_warnings": resolver_warnings}
                    if resolver_warnings
                    else None,
                )
                return
        if any(candidate.status == "accepted" for candidate in stored_candidates):
            return
        pending_candidates = [
            candidate
            for candidate in stored_candidates
            if candidate.status not in {"accepted", "rejected"}
        ]
        if stored_candidates and not pending_candidates:
            self._upsert_task(
                bundle,
                material,
                status="pending",
                assigned_strategy="public_candidate_rejected_continue_resolution",
                error_message=None,
                candidate_ids=[],
                extra_context=_resolution_extra_context(web_search_evidence, resolver_warnings),
            )
            return
        if pending_candidates:
            self._upsert_task(
                bundle,
                material,
                status="needs_review",
                assigned_strategy=(
                    "web_search_to_pubchem_review"
                    if web_search_evidence
                    else "opsin_name_review"
                    if any(candidate.provider == "opsin" for candidate in pending_candidates)
                    else "public_database_review"
                ),
                error_message=None,
                candidate_ids=[
                    candidate.structure_candidate_id for candidate in pending_candidates
                ],
                extra_context=_resolution_extra_context(web_search_evidence, resolver_warnings),
            )
        elif web_search_evidence:
            self._upsert_task(
                bundle,
                material,
                status="needs_review",
                assigned_strategy="web_search_evidence_review",
                error_message=None,
                candidate_ids=[],
                extra_context=_resolution_extra_context(web_search_evidence, resolver_warnings),
            )
        else:
            self._upsert_task(
                bundle,
                material,
                status="pending",
                assigned_strategy="public_database_not_found",
                error_message=None,
                candidate_ids=[],
                extra_context=_resolution_extra_context(web_search_evidence, resolver_warnings),
            )

    def _resolve_via_anysearch_identifiers(
        self,
        bundle: PaperMaterialStructureBundle,
        material: PaperLocalMaterial,
        *,
        query_texts: list[tuple[str, str]],
        max_results_per_query: int | None,
    ) -> tuple[list[MaterialStructureCandidate], list[dict[str, object]]]:
        if not self.anysearch_client:
            return [], []
        evidence: list[dict[str, object]] = []
        seen_identifiers: set[str] = set()
        for query_text, query_type in query_texts:
            for search_query in list(_web_search_queries(query_text))[
                : self.config.external_services.anysearch.max_query_variants_per_name
            ]:
                results = self.anysearch_client.search(search_query)
                evidence.extend(
                    _web_search_evidence(
                        results,
                        query_text=query_text,
                        search_query=search_query,
                    )
                )
                for result in results:
                    for identifier in result.cas_numbers():
                        if identifier in seen_identifiers:
                            continue
                        seen_identifiers.add(identifier)
                        compounds = self.pubchem_client.resolve_name(
                            identifier,
                            max_results=max_results_per_query,
                        )
                        candidates = [
                            self._store_candidate_preserving_review(
                                _candidate_from_pubchem_compound(
                                    bundle=bundle,
                                    material=material,
                                    compound=compound,
                                    query_text=query_text,
                                    query_type=query_type,
                                    discovery_evidence={
                                        "resolver": "anysearch_to_pubchem",
                                        "discovered_identifier": identifier,
                                        "web_search_results": evidence,
                                    },
                                )
                            )
                            for compound in compounds
                        ]
                        if candidates:
                            return candidates, evidence
        return [], evidence

    def _store_candidate_preserving_review(
        self,
        candidate: MaterialStructureCandidate,
    ) -> MaterialStructureCandidate:
        existing = self.structure_candidates.get_by_source(
            candidate.candidate_run_id,
            candidate.paper_material_id,
            candidate.provider,
            candidate.source_identifier,
        )
        if existing and existing.status in {"accepted", "rejected"}:
            return existing
        return self.structure_candidates.upsert(candidate)

    def _upsert_task(
        self,
        bundle: PaperMaterialStructureBundle,
        material: PaperLocalMaterial,
        *,
        status: str,
        assigned_strategy: str,
        error_message: str | None,
        candidate_ids: list[str],
        extra_context: dict[str, object] | None = None,
    ) -> MaterialResolutionTask:
        timestamp = now_iso()
        next_action = (
            "retry_public_resolution"
            if status == "failed"
            else "judge_candidates"
            if candidate_ids
            else "run_visual_ocsr"
        )
        return self.tasks.upsert(
            MaterialResolutionTask(
                task_id=uuid4().hex,
                paper_id=bundle.paper_id,
                candidate_run_id=bundle.candidate_run_id or "",
                paper_material_id=material.paper_material_id,
                material_mentions=[
                    candidate["text"] for candidate in _mention_candidates(material)
                ],
                material_context={
                    "full_name_in_paper": material.full_name_in_paper,
                    "normalized_name": material.normalized_name,
                    "abbreviation": material.abbreviation,
                    "paper_specific_label": material.paper_specific_label,
                    "material_class": material.material_class,
                    "used_in": [usage.model_dump(mode="json") for usage in material.used_in],
                    "structure_candidate_ids": candidate_ids,
                    **(extra_context or {}),
                },
                priority=_public_resolution_priority(material),
                status=status,
                assigned_strategy=assigned_strategy,
                current_stage=(
                    "visual_ocsr_pending"
                    if next_action == "run_visual_ocsr"
                    else "public_resolution"
                ),
                next_action=next_action,
                error_message=error_message,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )


def _public_query_texts(material: PaperLocalMaterial) -> Iterable[tuple[str, str]]:
    seen: set[str] = set()
    for candidate in _mention_candidates(material):
        if candidate["is_paper_specific"]:
            continue
        text = candidate["text"].strip()
        normalized = candidate["normalized_alias"]
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        yield text, candidate["alias_type"]


def _candidate_from_pubchem_compound(
    *,
    bundle: PaperMaterialStructureBundle,
    material: PaperLocalMaterial,
    compound: PubChemCompound,
    query_text: str,
    query_type: str,
    discovery_evidence: dict[str, object] | None = None,
) -> MaterialStructureCandidate:
    timestamp = now_iso()
    source_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}"
    canonical_name = _best_compound_name(compound, fallback=query_text)
    return MaterialStructureCandidate(
        structure_candidate_id=uuid4().hex,
        paper_id=bundle.paper_id,
        candidate_run_id=bundle.candidate_run_id or "",
        paper_material_id=material.paper_material_id,
        provider="pubchem",
        resolver_name="anysearch_to_pubchem" if discovery_evidence else "pubchem_name",
        query_text=query_text,
        query_type=query_type,
        source_identifier=compound.cid,
        source_url=source_url,
        canonical_name=canonical_name,
        material_class=_infer_material_class(compound, material),
        representation_type="small_molecule" if compound.canonical_smiles else "unknown",
        raw_smiles=compound.canonical_smiles,
        canonical_smiles=compound.canonical_smiles,
        isomeric_smiles=compound.isomeric_smiles,
        inchi=compound.inchi,
        inchi_key=compound.inchi_key,
        formula=compound.formula,
        molecular_weight=compound.molecular_weight,
        synonyms=compound.synonyms,
        evidence={
            "pubchem_cid": compound.cid,
            "query_text": query_text,
            "query_type": query_type,
            "iupac_name": compound.iupac_name,
            "source_url": source_url,
            "paper_material": material.model_dump(mode="json"),
            **(discovery_evidence or {}),
        },
        confidence=_pubchem_candidate_confidence(query_text, query_type, compound),
        status="pending_review",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _candidate_from_opsin_compound(
    *,
    bundle: PaperMaterialStructureBundle,
    material: PaperLocalMaterial,
    compound: OpsinCompound,
    query_text: str,
    query_type: str,
    source_url: str,
) -> MaterialStructureCandidate:
    timestamp = now_iso()
    standardized = standardize_smiles(compound.smiles)
    return MaterialStructureCandidate(
        structure_candidate_id=uuid4().hex,
        paper_id=bundle.paper_id,
        candidate_run_id=bundle.candidate_run_id or "",
        paper_material_id=material.paper_material_id,
        provider="opsin",
        resolver_name="opsin_systematic_name",
        query_text=query_text,
        query_type=query_type,
        source_identifier=compound.inchi_key
        or standardized.inchi_key
        or standardized.canonical_smiles,
        source_url=source_url,
        canonical_name=query_text,
        material_class=material.material_class,
        representation_type="small_molecule",
        raw_smiles=compound.smiles,
        canonical_smiles=standardized.canonical_smiles,
        isomeric_smiles=standardized.isomeric_smiles,
        inchi=compound.inchi or standardized.inchi,
        inchi_key=compound.inchi_key or standardized.inchi_key,
        formula=standardized.formula,
        molecular_weight=standardized.molecular_weight,
        synonyms=[query_text],
        evidence={
            "resolver": "opsin_systematic_name",
            "query_text": query_text,
            "query_type": query_type,
            "source_url": source_url,
            "opsin_response": compound.raw_result,
            "paper_material": material.model_dump(mode="json"),
        },
        confidence=0.84,
        status="pending_review",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _web_search_evidence(
    results: list[AnySearchResult],
    *,
    query_text: str,
    search_query: str,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for result in results:
        evidence.append(
            {
                "query_text": query_text,
                "search_query": search_query,
                "title": result.title,
                "url": result.url,
                "description": result.description[:500],
                "content_excerpt": result.content[:1000],
                "cas_numbers": result.cas_numbers(),
                "score": result.score,
            }
        )
    return evidence


def _web_search_queries(query_text: str) -> Iterable[str]:
    quoted = f'"{query_text}"'
    yield f"{quoted} OLED material CAS chemical structure"
    yield f"{quoted} CAS SMILES"
    yield (
        f"{quoted} (site:ossila.com OR site:chemicalbook.com "
        "OR site:bldpharm.com OR site:bocsci.com)"
    )


def _resolution_extra_context(
    web_search_evidence: list[dict[str, object]],
    resolver_warnings: list[str],
) -> dict[str, object] | None:
    context: dict[str, object] = {}
    if web_search_evidence:
        context["web_search_results"] = web_search_evidence
    if resolver_warnings:
        context["resolver_warnings"] = resolver_warnings
    return context or None


def _best_compound_name(compound: PubChemCompound, *, fallback: str) -> str:
    for synonym in compound.synonyms:
        if synonym and len(synonym) <= 80:
            return synonym
    return compound.iupac_name or fallback


def _pubchem_candidate_confidence(
    query_text: str,
    query_type: str,
    compound: PubChemCompound,
) -> float:
    normalized_query = normalize_material_alias(query_text)
    if not normalized_query:
        return 0.5
    normalized_names = {
        normalize_material_alias(value)
        for value in [compound.iupac_name, *compound.synonyms]
        if value
    }
    if normalized_query in normalized_names:
        confidence = 0.9
    elif any(
        normalized_query in name or name in normalized_query for name in normalized_names if name
    ):
        confidence = 0.72
    else:
        confidence = 0.62
    if query_type in {"abbreviation", "mention"}:
        return min(confidence, 0.72)
    if query_type in {"normalized_name", "paper_specific_label"}:
        return min(confidence, 0.8)
    return confidence


def is_plausible_public_structure_candidate(
    candidate: MaterialStructureCandidate,
) -> bool:
    """Return whether a public candidate is strong enough to enter identity review.

    PubChem's name endpoint can return a compound for a short OLED label even when
    that label is absent from every returned name or synonym. Such direct hits are
    not evidence of identity. Identifier-backed AnySearch candidates remain eligible
    for the identity judge because their CAS/identifier provenance is independent.
    """
    if candidate.provider != "pubchem" or candidate.resolver_name != "pubchem_name":
        return True
    return (candidate.confidence or 0.0) > DIRECT_PUBCHEM_QUERY_MISMATCH_CONFIDENCE


def _contains_equivalent_structure(
    candidates: list[MaterialStructureCandidate],
    candidate: MaterialStructureCandidate,
) -> bool:
    for existing in candidates:
        if candidate.inchi_key and existing.inchi_key == candidate.inchi_key:
            return True
        if (
            not candidate.inchi_key
            and candidate.canonical_smiles
            and existing.canonical_smiles == candidate.canonical_smiles
        ):
            return True
    return False


def _needs_web_identity_support(candidates: list[MaterialStructureCandidate]) -> bool:
    return not any(
        candidate.status != "rejected" and candidate.query_type in {"full_name", "canonical_name"}
        for candidate in candidates
    )


def _infer_material_class(compound: PubChemCompound, material: PaperLocalMaterial) -> str:
    if material.material_class and material.material_class != "unknown":
        return material.material_class
    formula = compound.formula or ""
    if "C" in formula:
        return "small_molecule_organic"
    return "unknown"


def _public_resolution_priority(material: PaperLocalMaterial) -> str:
    roles = {usage.component_role for usage in material.used_in if usage.component_role}
    if {"final_emitter", "emitter", "emitter_dopant", "sensitizer", "sensitizer_dopant"} & roles:
        return "high"
    return "normal"
