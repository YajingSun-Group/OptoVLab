from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_chemistry import standardize_smiles
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    normalize_material_alias,
)
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialIdentityJudgment,
    MaterialReviewAction,
    MaterialStructureCandidate,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialIdentityJudgmentRepository,
    MaterialStructureCandidateRepository,
)


DecisionAction = Literal["accept", "defer", "unresolved"]
EvidenceRole = Literal["paper_identity", "independent_identity", "structure", "image"]

STRONG_STRUCTURE_SOURCE_TYPES = {
    "article",
    "cas_common_chemistry",
    "chebi",
    "chemspider",
    "comptox",
    "jglobal",
    "opsin",
    "pubchem",
    "supporting_information",
}


class MaterialWebRescueSource(BaseModel):
    title: str
    url: str
    source_type: str
    roles: list[EvidenceRole] = Field(min_length=1)
    identifier: str | None = None
    evidence: str

    @model_validator(mode="after")
    def validate_url(self) -> MaterialWebRescueSource:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Evidence source must have an HTTP(S) URL: {self.url}")
        return self


class MaterialWebRescueDecision(BaseModel):
    paper_id: str
    candidate_run_id: str
    paper_material_id: str
    expected_mentions: list[str] = Field(min_length=1)
    action: DecisionAction
    reviewed_name: str | None = None
    canonical_name: str | None = None
    full_name_in_paper: str | None = None
    material_class: str = "small_molecule_organic"
    representation_type: str = "small_molecule"
    smiles: str | None = None
    identity_verdict: Literal["exact_match", "likely_match", "insufficient_evidence"] = (
        "insufficient_evidence"
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    structure_method: Literal[
        "database_record",
        "published_smiles",
        "name_to_structure",
        "decimer_ocsr",
        "manual_transcription",
        "none",
    ] = "none"
    sources: list[MaterialWebRescueSource] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    notes: str | None = None


class MaterialWebRescueDecisionFile(BaseModel):
    schema_version: str = "material_web_rescue_v1"
    run_id: str
    researcher: str = "codex_web_rescue_agent"
    created_at: str
    decisions: list[MaterialWebRescueDecision]


class MaterialWebRescueApplyItem(BaseModel):
    paper_id: str
    paper_material_id: str
    requested_action: DecisionAction
    status: Literal["accepted", "candidate_stored", "unresolved", "skipped", "failed"]
    structure_candidate_id: str | None = None
    global_material_id: str | None = None
    inchi_key: str | None = None
    message: str


class MaterialWebRescueApplyResult(BaseModel):
    run_id: str
    dry_run: bool
    accepted_count: int = 0
    candidate_stored_count: int = 0
    unresolved_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    items: list[MaterialWebRescueApplyItem] = Field(default_factory=list)


class MaterialWebRescueService:
    """Apply externally researched material structures through the normal review audit path."""

    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.material_resolution = MaterialResolutionService(config)
        self.review_service = MaterialStructureReviewService(config)
        self.structure_candidates = MaterialStructureCandidateRepository(self.database)
        self.identity_judgments = MaterialIdentityJudgmentRepository(self.database)

    def apply_decision_file(
        self,
        decision_path: Path,
        *,
        dry_run: bool = False,
    ) -> MaterialWebRescueApplyResult:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        decision_file = MaterialWebRescueDecisionFile.model_validate(payload)
        result = MaterialWebRescueApplyResult(run_id=decision_file.run_id, dry_run=dry_run)
        for decision in decision_file.decisions:
            try:
                item = self.apply_decision(
                    decision,
                    actor=decision_file.researcher,
                    research_run_id=decision_file.run_id,
                    dry_run=dry_run,
                )
            except Exception as exc:  # Keep a large research batch resumable.
                item = MaterialWebRescueApplyItem(
                    paper_id=decision.paper_id,
                    paper_material_id=decision.paper_material_id,
                    requested_action=decision.action,
                    status="failed",
                    message=str(exc),
                )
            result.items.append(item)
        result.accepted_count = sum(item.status == "accepted" for item in result.items)
        result.candidate_stored_count = sum(
            item.status == "candidate_stored" for item in result.items
        )
        result.unresolved_count = sum(item.status == "unresolved" for item in result.items)
        result.skipped_count = sum(item.status == "skipped" for item in result.items)
        result.failed_count = sum(item.status == "failed" for item in result.items)
        return result

    def apply_decision(
        self,
        decision: MaterialWebRescueDecision,
        *,
        actor: str,
        research_run_id: str,
        dry_run: bool = False,
    ) -> MaterialWebRescueApplyItem:
        paper_id = self.review_service.paper_service.normalize_paper_id(decision.paper_id)
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None or bundle.candidate_run_id is None:
            raise ValueError(f"Candidate material bundle not found: {paper_id}")
        if bundle.candidate_run_id != decision.candidate_run_id:
            raise ValueError(
                "Stale rescue decision: candidate run changed from "
                f"{decision.candidate_run_id} to {bundle.candidate_run_id}."
            )
        material = next(
            (
                item
                for item in bundle.materials
                if item.paper_material_id == decision.paper_material_id
            ),
            None,
        )
        if material is None:
            raise ValueError(f"Paper material not found: {decision.paper_material_id}")
        self._validate_target_mentions(decision, material.mention_list)

        link = next(
            (item for item in bundle.links if item.paper_material_id == decision.paper_material_id),
            None,
        )
        if link and link.global_material_id and link.match_status in {
            "matched_candidate",
            "matched_local",
            "confirmed",
        }:
            return MaterialWebRescueApplyItem(
                paper_id=paper_id,
                paper_material_id=decision.paper_material_id,
                requested_action=decision.action,
                status="skipped",
                global_material_id=link.global_material_id,
                message="Material already has a protected resolved link.",
            )
        if decision.action == "unresolved":
            return MaterialWebRescueApplyItem(
                paper_id=paper_id,
                paper_material_id=decision.paper_material_id,
                requested_action=decision.action,
                status="unresolved",
                message=decision.notes or "Research did not establish a unique structure.",
            )

        standardized = standardize_smiles(decision.smiles or "")
        acceptance_errors = self._acceptance_errors(decision)
        if decision.action == "accept" and acceptance_errors:
            raise ValueError("Unsafe automatic acceptance: " + "; ".join(acceptance_errors))
        source_identifier = self._source_identifier(decision, standardized.inchi_key)
        existing_candidate = self.structure_candidates.get_by_source(
            decision.candidate_run_id,
            decision.paper_material_id,
            "web_rescue_agent",
            source_identifier,
        )
        if (
            decision.action == "accept"
            and existing_candidate is not None
            and existing_candidate.status == "rejected"
        ):
            raise ValueError(
                "Identical web-rescue structure and source set was previously rejected; "
                "provide a corrected structure or materially different evidence."
            )
        if dry_run:
            status = "accepted" if decision.action == "accept" else "candidate_stored"
            return MaterialWebRescueApplyItem(
                paper_id=paper_id,
                paper_material_id=decision.paper_material_id,
                requested_action=decision.action,
                status=status,
                inchi_key=standardized.inchi_key,
                message="Decision passed validation; dry-run made no database changes.",
            )

        timestamp = now_iso()
        primary_source = next(
            (source for source in decision.sources if "structure" in source.roles),
            decision.sources[0],
        )
        candidate = self.structure_candidates.upsert(
            MaterialStructureCandidate(
                structure_candidate_id=uuid4().hex,
                paper_id=paper_id,
                candidate_run_id=decision.candidate_run_id,
                paper_material_id=decision.paper_material_id,
                provider="web_rescue_agent",
                resolver_name=f"verified_{decision.structure_method}",
                query_text=decision.reviewed_name or decision.expected_mentions[0],
                query_type="web_research",
                source_identifier=source_identifier,
                source_url=primary_source.url,
                canonical_name=decision.canonical_name or decision.reviewed_name,
                material_class=decision.material_class,
                representation_type=decision.representation_type,
                raw_smiles=standardized.raw_smiles,
                canonical_smiles=standardized.canonical_smiles,
                isomeric_smiles=standardized.isomeric_smiles,
                inchi=standardized.inchi,
                inchi_key=standardized.inchi_key,
                formula=standardized.formula,
                molecular_weight=standardized.molecular_weight,
                synonyms=_unique_text(
                    [
                        decision.reviewed_name,
                        decision.full_name_in_paper,
                        *decision.expected_mentions,
                        *material.mention_list,
                    ]
                ),
                evidence={
                    "research_run_id": research_run_id,
                    "structure_method": decision.structure_method,
                    "sources": [source.model_dump(mode="json") for source in decision.sources],
                    "conflicts": decision.conflicts,
                    "notes": decision.notes,
                    "expected_mentions": decision.expected_mentions,
                    "paper_material": material.model_dump(mode="json"),
                    "standardized_by": "rdkit",
                },
                confidence=decision.confidence,
                status="pending_review",
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        judgment = MaterialIdentityJudgment(
            judgment_id=uuid4().hex,
            paper_id=paper_id,
            candidate_run_id=decision.candidate_run_id,
            paper_material_id=decision.paper_material_id,
            structure_candidate_id=candidate.structure_candidate_id,
            provider="codex_web_research",
            model="codex_web_research",
            prompt_version="material_web_rescue_v1",
            verdict=decision.identity_verdict,
            confidence=decision.confidence,
            supporting_evidence=[source.evidence for source in decision.sources],
            conflicts=decision.conflicts,
            recommended_action=(
                "ready_for_human_accept"
                if decision.identity_verdict in {"exact_match", "likely_match"}
                else "manual_review"
            ),
            deterministic_checks={
                "rdkit_valid": True,
                "inchi_key": standardized.inchi_key,
                "unique_source_domains": sorted(_source_domains(decision.sources)),
                "acceptance_errors": acceptance_errors,
            },
            input_context=decision.model_dump(mode="json"),
            raw_response={
                "research_run_id": research_run_id,
                "decision_source": "human_supervised_codex_web_research",
            },
            status="completed",
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.identity_judgments.add(judgment)
        if decision.action == "defer":
            return MaterialWebRescueApplyItem(
                paper_id=paper_id,
                paper_material_id=decision.paper_material_id,
                requested_action=decision.action,
                status="candidate_stored",
                structure_candidate_id=candidate.structure_candidate_id,
                inchi_key=standardized.inchi_key,
                message="Candidate and evidence stored for human review.",
            )

        accepted_bundle = self.review_service.accept_structure_candidate(
            candidate.structure_candidate_id,
            MaterialReviewAction(
                actor=actor,
                message=(
                    f"Accepted by verified web rescue run {research_run_id}: "
                    f"{decision.notes or 'identity and structure sources verified.'}"
                ),
            ),
            paper_material=material,
        )
        accepted_link = (
            next(
                (
                    item
                    for item in accepted_bundle.links
                    if item.paper_material_id == decision.paper_material_id
                ),
                None,
            )
            if accepted_bundle
            else None
        )
        return MaterialWebRescueApplyItem(
            paper_id=paper_id,
            paper_material_id=decision.paper_material_id,
            requested_action=decision.action,
            status="accepted",
            structure_candidate_id=candidate.structure_candidate_id,
            global_material_id=accepted_link.global_material_id if accepted_link else None,
            inchi_key=standardized.inchi_key,
            message="Verified web rescue candidate accepted through the review service.",
        )

    @staticmethod
    def _validate_target_mentions(
        decision: MaterialWebRescueDecision,
        current_mentions: list[str],
    ) -> None:
        expected = {normalize_material_alias(item) for item in decision.expected_mentions if item}
        current = {normalize_material_alias(item) for item in current_mentions if item}
        if not expected.intersection(current):
            raise ValueError(
                "Stale or mis-targeted rescue decision: expected mentions do not match the "
                f"current material ({sorted(current)})."
            )

    @staticmethod
    def _acceptance_errors(decision: MaterialWebRescueDecision) -> list[str]:
        errors: list[str] = []
        if decision.identity_verdict != "exact_match":
            errors.append("identity verdict must be exact_match")
        if decision.confidence is None or decision.confidence < 0.95:
            errors.append("confidence must be at least 0.95")
        if decision.conflicts:
            errors.append("conflicts must be empty")
        domains = _source_domains(decision.sources)
        if len(domains) < 2:
            errors.append("at least two independent source domains are required")
        roles = {role for source in decision.sources for role in source.roles}
        if not roles.intersection({"paper_identity", "independent_identity"}):
            errors.append("an identity-linking source is required")
        if "structure" not in roles:
            errors.append("a structure-bearing source is required")
        strong_structure_source = any(
            "structure" in source.roles
            and source.source_type.lower() in STRONG_STRUCTURE_SOURCE_TYPES
            for source in decision.sources
        )
        if not strong_structure_source:
            errors.append("a strong structure source is required")
        if decision.structure_method == "decimer_ocsr":
            corroborated = any(
                "structure" in source.roles
                and source.source_type.lower() in STRONG_STRUCTURE_SOURCE_TYPES
                and source.source_type.lower() not in {"article", "supporting_information"}
                for source in decision.sources
            )
            if not corroborated:
                errors.append("DECIMER output requires an independent structured-source match")
        return errors

    @staticmethod
    def _source_identifier(
        decision: MaterialWebRescueDecision,
        inchi_key: str | None,
    ) -> str:
        payload = "|".join(
            [
                inchi_key or decision.smiles or "",
                *sorted(source.url for source in decision.sources),
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"web-rescue:{digest}"


def _source_domains(sources: list[MaterialWebRescueSource]) -> set[str]:
    return {
        (urlparse(source.url).hostname or "").lower().removeprefix("www.")
        for source in sources
        if urlparse(source.url).hostname
    }


def _unique_text(values: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        normalized = normalize_material_alias(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result
