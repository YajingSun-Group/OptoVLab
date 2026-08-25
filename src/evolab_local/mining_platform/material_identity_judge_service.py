from __future__ import annotations

from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMClient,
    OpenAICompatibleLLMClient,
)
from evolab_local.mining_platform.external.opsin_client import (
    OpsinClient,
    SystematicNameResolverClient,
)
from evolab_local.mining_platform.material_chemistry import standardize_smiles
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    normalize_material_alias,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialIdentityEvidenceItem,
    MaterialIdentityJudgment,
    MaterialStructureCandidate,
    PaperLocalMaterial,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialIdentityEvidenceItemRepository,
    MaterialIdentityJudgmentRepository,
)


VERDICTS = {
    "exact_match",
    "likely_match",
    "ambiguous",
    "conflict",
    "rejected",
    "insufficient_evidence",
}
RECOMMENDED_ACTIONS = {
    "ready_for_human_accept",
    "manual_review",
    "reject_candidate",
    "search_more_evidence",
    "run_figure_pipeline",
}
PROMPT_VERSION = "material_identity_judge_v3_verdict_confidence"


class MaterialIdentityJudgeExecutionError(RuntimeError):
    """Operational judge failure; callers should retry instead of requesting review."""


class MaterialIdentityJudgeService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        llm_client: LLMClient | None = None,
        opsin_client: SystematicNameResolverClient | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.material_resolution = MaterialResolutionService(config)
        self.judgments = MaterialIdentityJudgmentRepository(self.database)
        self.evidence_items = MaterialIdentityEvidenceItemRepository(self.database)
        self.opsin_client = opsin_client or OpsinClient(config.external_services.opsin)
        self.llm_client = llm_client

    def init_runtime(self) -> None:
        self.material_resolution.init_runtime()

    def judge_paper_candidates(
        self,
        paper_id: str,
        *,
        paper_material_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None or not bundle.candidate_run_id:
            return bundle
        material_ids = sorted(
            {
                candidate.paper_material_id
                for candidate in bundle.structure_candidates
                if (
                    paper_material_id is None
                    or candidate.paper_material_id == paper_material_id
                )
                and candidate.status not in {"accepted", "rejected"}
                and bool(candidate.canonical_smiles or candidate.inchi_key)
            }
        )
        for selected_material_id in material_ids:
            self.judge_material_candidates(
                bundle.paper_id,
                selected_material_id,
                provider=provider,
                model=model,
            )
        return self.material_resolution.get_material_structure_bundle(bundle.paper_id)

    def judge_material_candidates(
        self,
        paper_id: str,
        paper_material_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        force: bool = False,
    ) -> list[MaterialIdentityJudgment]:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None or not bundle.candidate_run_id:
            return []
        material = next(
            (
                item
                for item in bundle.materials
                if item.paper_material_id == paper_material_id
            ),
            None,
        )
        if material is None:
            raise ValueError(
                f"Unknown paper material {paper_material_id!r} for {bundle.paper_id}."
            )
        candidates = sorted(
            [
                candidate
                for candidate in bundle.structure_candidates
                if candidate.paper_material_id == paper_material_id
                and candidate.status not in {"accepted", "rejected"}
                and bool(candidate.canonical_smiles or candidate.inchi_key)
            ],
            key=lambda item: (
                -(item.confidence or 0.0),
                item.provider,
                item.structure_candidate_id,
            ),
        )
        if not candidates:
            return []
        selected_provider = provider or self.config.llm.default_provider
        provider_config = self.config.llm.providers.get(selected_provider)
        if provider_config is None:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown LLM provider {selected_provider!r}. Available: {known}")
        selected_model = model or provider_config.default_model
        timestamp = now_iso()
        evidence_items = self.evidence_items.list_by_material(
            bundle.candidate_run_id,
            paper_material_id,
        )
        contexts: dict[str, dict[str, Any]] = {}
        checks_by_candidate: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            checks = self._deterministic_checks(material, candidate, evidence_items)
            checks_by_candidate[candidate.structure_candidate_id] = checks
            contexts[candidate.structure_candidate_id] = _input_context(
                material,
                candidate,
                checks,
                evidence_items,
            )
        group_context = {
            "paper_material": contexts[candidates[0].structure_candidate_id]["paper_material"],
            "candidates": [
                {
                    "structure_candidate_id": candidate.structure_candidate_id,
                    **contexts[candidate.structure_candidate_id]["candidate"],
                    "web_discovery_evidence": contexts[candidate.structure_candidate_id][
                        "web_discovery_evidence"
                    ],
                    "deterministic_checks": checks_by_candidate[
                        candidate.structure_candidate_id
                    ],
                }
                for candidate in candidates
            ],
            "identity_enrichment_evidence": contexts[
                candidates[0].structure_candidate_id
            ]["identity_enrichment_evidence"],
        }
        group_fingerprint = sha256(
            json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "provider": selected_provider,
                    "model": selected_model,
                    "group_context": group_context,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if not force:
            cached = [
                self.judgments.latest_by_candidate(candidate.structure_candidate_id)
                for candidate in candidates
            ]
            if all(
                judgment is not None
                and judgment.status == "completed"
                and judgment.prompt_version == PROMPT_VERSION
                and judgment.provider == selected_provider
                and judgment.model == selected_model
                and judgment.input_context.get("group_fingerprint") == group_fingerprint
                for judgment in cached
            ):
                return [judgment for judgment in cached if judgment is not None]
        client = self.llm_client or OpenAICompatibleLLMClient(provider_config)
        try:
            response = client.generate_json(
                _judge_group_messages(group_context),
                model=selected_model,
                temperature=0,
            )
            parsed = response.parsed_json or {}
            payloads = _group_judgment_payloads(parsed, candidates)
            stored: list[MaterialIdentityJudgment] = []
            for candidate in candidates:
                input_context = {
                    **contexts[candidate.structure_candidate_id],
                    "group_fingerprint": group_fingerprint,
                    "group_candidate_ids": [
                        item.structure_candidate_id for item in candidates
                    ],
                }
                judgment = self._judgment_from_payload(
                    candidate,
                    payloads.get(candidate.structure_candidate_id, {}),
                    checks=checks_by_candidate[candidate.structure_candidate_id],
                    input_context=input_context,
                    provider=selected_provider,
                    model=selected_model,
                    raw_response=response.raw_response,
                    timestamp=timestamp,
                )
                stored.append(self.judgments.add(judgment))
            return stored
        except Exception as exc:
            failed: list[MaterialIdentityJudgment] = []
            for candidate in candidates:
                input_context = {
                    **contexts[candidate.structure_candidate_id],
                    "group_fingerprint": group_fingerprint,
                    "group_candidate_ids": [
                        item.structure_candidate_id for item in candidates
                    ],
                }
                failed.append(
                    self.judgments.add(
                        MaterialIdentityJudgment(
                            judgment_id=uuid4().hex,
                            paper_id=candidate.paper_id,
                            candidate_run_id=candidate.candidate_run_id,
                            paper_material_id=candidate.paper_material_id,
                            structure_candidate_id=candidate.structure_candidate_id,
                            provider=selected_provider,
                            model=selected_model,
                            prompt_version=PROMPT_VERSION,
                            verdict="insufficient_evidence",
                            recommended_action="manual_review",
                            deterministic_checks=checks_by_candidate[
                                candidate.structure_candidate_id
                            ],
                            input_context=input_context,
                            status="failed",
                            error_message=str(exc),
                            created_at=timestamp,
                            updated_at=timestamp,
                        )
                    )
                )
            raise MaterialIdentityJudgeExecutionError(
                f"Material Identity Judge failed for {bundle.paper_id} "
                f"{paper_material_id}: {exc}"
            ) from exc

    def _judgment_from_payload(
        self,
        candidate: MaterialStructureCandidate,
        payload: dict[str, Any],
        *,
        checks: dict[str, Any],
        input_context: dict[str, Any],
        provider: str,
        model: str,
        raw_response: dict[str, Any],
        timestamp: str,
    ) -> MaterialIdentityJudgment:
        verdict = _choice(payload.get("verdict"), VERDICTS, "insufficient_evidence")
        supporting_evidence = _string_list(payload.get("supporting_evidence"))
        conflicts = _string_list(payload.get("conflicts"))
        recommended_action = _choice(
            payload.get("recommended_action"),
            RECOMMENDED_ACTIONS,
            "manual_review",
        )
        confidence = _confidence(payload.get("confidence"))
        confirmed_evidence_matches_candidate = (
            checks.get("confirmed_evidence_inchi_key_comparison") == "same"
        )
        if checks.get("confirmed_evidence_inchi_key_comparison") == "conflict":
            verdict = "conflict"
            recommended_action = "reject_candidate"
            conflicts = [
                "Candidate InChIKey conflicts with OPSIN parsing of a human-confirmed "
                "identity evidence full name.",
                *conflicts,
            ]
        elif checks.get("opsin_inchi_key_comparison") == "conflict":
            if confirmed_evidence_matches_candidate:
                if verdict in {"conflict", "rejected", "insufficient_evidence"}:
                    verdict = "likely_match"
                recommended_action = "ready_for_human_accept"
                supporting_evidence = [
                    "Human-confirmed identity evidence resolves to the same InChIKey as the candidate; "
                    "the OPSIN conflict from the paper full name is treated as a likely malformed "
                    "or lossy paper-local name.",
                    *supporting_evidence,
                ]
                conflicts = [
                    "Paper full-name OPSIN parsing conflicts with the candidate, but a confirmed "
                    "A/B-tier identity evidence full name resolves to the candidate InChIKey.",
                    *conflicts,
                ]
            else:
                verdict = "conflict"
                recommended_action = "reject_candidate"
                conflicts = [
                    "Candidate InChIKey conflicts with OPSIN parsing of the full name extracted from the paper.",
                    *conflicts,
                ]
        elif (
            candidate.resolver_name == "anysearch_to_pubchem"
            and checks.get("identifier_source_title_matches_paper_alias") is False
        ):
            if verdict in {"exact_match", "likely_match"}:
                verdict = "ambiguous"
            if verdict not in {"conflict", "rejected"}:
                recommended_action = "search_more_evidence"
            conflicts = [
                "The CAS-discovery source names a material that does not match any paper alias; "
                "the retrieved structure needs independent identity evidence.",
                *conflicts,
            ]
        return MaterialIdentityJudgment(
            judgment_id=uuid4().hex,
            paper_id=candidate.paper_id,
            candidate_run_id=candidate.candidate_run_id,
            paper_material_id=candidate.paper_material_id,
            structure_candidate_id=candidate.structure_candidate_id,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence=_dedupe(supporting_evidence),
            conflicts=_dedupe(conflicts),
            recommended_action=recommended_action,
            deterministic_checks=checks,
            input_context=input_context,
            raw_response=raw_response,
            status="completed",
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _deterministic_checks(
        self,
        material: PaperLocalMaterial,
        candidate: MaterialStructureCandidate,
        evidence_items: list[MaterialIdentityEvidenceItem],
    ) -> dict[str, Any]:
        aliases = _material_aliases(material)
        normalized_aliases = {normalize_material_alias(value) for value in aliases if value}
        source_results = _candidate_source_results(candidate)
        identified_results = [
            result
            for result in source_results
            if candidate.evidence.get("discovered_identifier")
            in (result.get("cas_numbers") if isinstance(result.get("cas_numbers"), list) else [])
        ]
        source_titles = [
            str(result.get("title")) for result in identified_results if result.get("title")
        ]
        source_title_alias_match = (
            any(
                alias and alias in normalize_material_alias(title)
                for alias in normalized_aliases
                for title in source_titles
            )
            if source_titles
            else None
        )
        checks: dict[str, Any] = {
            "paper_aliases": aliases,
            "candidate_source_identifier": candidate.source_identifier,
            "candidate_inchi_key": candidate.inchi_key,
            "candidate_formula": candidate.formula,
            "discovered_identifier": candidate.evidence.get("discovered_identifier"),
            "identifier_source_titles": source_titles,
            "identifier_source_title_matches_paper_alias": source_title_alias_match,
            "identity_evidence_items": [
                {
                    "evidence_item_id": item.evidence_item_id,
                    "source_tier": item.source_tier,
                    "full_name": item.full_name,
                    "cas_number": item.cas_number,
                    "explicitly_linked": item.explicitly_linked,
                    "review_status": item.review_status,
                }
                for item in evidence_items
            ],
        }
        if material.full_name_in_paper:
            try:
                reference = self.opsin_client.resolve_name(material.full_name_in_paper)
                if reference:
                    standardized = standardize_smiles(reference.smiles)
                    checks.update(
                        {
                            "opsin_query": material.full_name_in_paper,
                            "opsin_inchi_key": standardized.inchi_key,
                            "opsin_formula": standardized.formula,
                            "opsin_inchi_key_comparison": (
                                "same"
                                if candidate.inchi_key
                                and candidate.inchi_key == standardized.inchi_key
                                else "conflict"
                                if candidate.inchi_key and standardized.inchi_key
                                else "unavailable"
                            ),
                        }
                    )
            except Exception as exc:
                checks["opsin_error"] = str(exc)
        confirmed_reference = next(
            (
                item
                for item in evidence_items
                if item.review_status == "confirmed"
                and item.explicitly_linked
                and item.source_tier in {"A", "B"}
                and item.full_name
            ),
            None,
        )
        if confirmed_reference and confirmed_reference.full_name:
            try:
                reference = self.opsin_client.resolve_name(confirmed_reference.full_name)
                if reference:
                    standardized = standardize_smiles(reference.smiles)
                    checks.update(
                        {
                            "confirmed_evidence_item_id": confirmed_reference.evidence_item_id,
                            "confirmed_evidence_full_name": confirmed_reference.full_name,
                            "confirmed_evidence_inchi_key": standardized.inchi_key,
                            "confirmed_evidence_formula": standardized.formula,
                            "confirmed_evidence_inchi_key_comparison": (
                                "same"
                                if candidate.inchi_key
                                and candidate.inchi_key == standardized.inchi_key
                                else "conflict"
                                if candidate.inchi_key and standardized.inchi_key
                                else "unavailable"
                            ),
                        }
                    )
            except Exception as exc:
                checks["confirmed_evidence_opsin_error"] = str(exc)
        return checks


def _judge_group_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        "You are Material Identity Judge v2 for an OLED materials database. "
        "Compare every retrieved molecular-structure candidate for one paper-local material "
        "in a single decision. Be conservative and evaluate candidates against each other. "
        "A web result containing a CAS number is "
        "discovery evidence only: if its displayed product/material name differs from the paper "
        "mention, do not treat it as identity proof. If an OPSIN structure derived from a full "
        "name in the paper conflicts by InChIKey with the candidate, verdict must be conflict "
        "unless a human-confirmed A/B-tier identity evidence full name resolves to the same "
        "InChIKey as the candidate; in that case treat the paper full-name conflict as likely "
        "malformed or lossy text extraction and recommend human acceptance. "
        "Do not invent full names, CAS numbers, structures, or supporting facts. "
        "Return JSON only with a judgments array. Each item must contain the exact "
        "structure_candidate_id from the input plus: verdict (exact_match, likely_match, "
        "ambiguous, conflict, rejected, insufficient_evidence), confidence (0 to 1), "
        "where confidence means confidence that the stated verdict and recommended action "
        "are correct, not the probability that the candidate matches. A clear identity "
        "mismatch should therefore be conflict with high confidence, not low confidence. "
        "supporting_evidence (array of strings), conflicts (array of strings), and "
        "recommended_action (ready_for_human_accept, manual_review, reject_candidate, "
        "search_more_evidence, run_figure_pipeline). Return one item for every candidate."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Assess and compare these material identity candidates:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True),
        },
    ]


def _group_judgment_payloads(
    payload: dict[str, Any],
    candidates: list[MaterialStructureCandidate],
) -> dict[str, dict[str, Any]]:
    raw_items = payload.get("judgments")
    if isinstance(raw_items, list):
        candidate_ids = {candidate.structure_candidate_id for candidate in candidates}
        return {
            str(item.get("structure_candidate_id")): item
            for item in raw_items
            if isinstance(item, dict)
            and str(item.get("structure_candidate_id")) in candidate_ids
        }
    if len(candidates) == 1:
        return {candidates[0].structure_candidate_id: payload}
    return {}


def _input_context(
    material: PaperLocalMaterial,
    candidate: MaterialStructureCandidate,
    checks: dict[str, Any],
    evidence_items: list[MaterialIdentityEvidenceItem],
) -> dict[str, Any]:
    return {
        "paper_material": {
            "paper_material_id": material.paper_material_id,
            "mentions": material.mention_list,
            "entity_label": material.entity_label,
            "full_name_in_paper": material.full_name_in_paper,
            "abbreviation": material.abbreviation,
            "material_class": material.material_class,
            "used_in": [usage.model_dump(mode="json") for usage in material.used_in],
        },
        "candidate": {
            "provider": candidate.provider,
            "resolver_name": candidate.resolver_name,
            "query_text": candidate.query_text,
            "source_identifier": candidate.source_identifier,
            "source_url": candidate.source_url,
            "canonical_name": candidate.canonical_name,
            "synonyms": candidate.synonyms[:20],
            "formula": candidate.formula,
            "inchi_key": candidate.inchi_key,
            "canonical_smiles": candidate.canonical_smiles,
        },
        "web_discovery_evidence": _candidate_source_results(candidate)[:5],
        "identity_enrichment_evidence": [
            {
                "source_tier": item.source_tier,
                "source_title": item.source_title,
                "source_url": item.source_url,
                "excerpt": item.excerpt,
                "alias": item.alias,
                "full_name": item.full_name,
                "cas_number": item.cas_number,
                "explicitly_linked": item.explicitly_linked,
                "review_status": item.review_status,
            }
            for item in evidence_items[:10]
        ],
        "deterministic_checks": checks,
    }


def _material_aliases(material: PaperLocalMaterial) -> list[str]:
    return _dedupe(
        [
            material.entity_label,
            *material.mention_list,
            material.full_name_in_paper,
            material.normalized_name,
            material.canonical_name,
            material.abbreviation,
        ]
    )


def _candidate_source_results(candidate: MaterialStructureCandidate) -> list[dict[str, Any]]:
    values = candidate.evidence.get("web_search_results")
    if not isinstance(values, list):
        return []
    return [
        {
            "title": value.get("title"),
            "url": value.get("url"),
            "cas_numbers": value.get("cas_numbers", []),
            "content_excerpt": str(value.get("description") or value.get("content_excerpt") or "")[
                :500
            ],
        }
        for value in values
        if isinstance(value, dict)
    ]


def _choice(value: Any, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _confidence(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return max(0.0, min(float(value), 1.0))
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def _dedupe(values: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
