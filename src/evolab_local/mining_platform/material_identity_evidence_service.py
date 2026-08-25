from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from typing import Any, Protocol
from urllib.parse import quote, urlparse
from uuid import uuid4

import httpx

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.anysearch_client import (
    AnySearchClient,
    AnySearchResult,
    MaterialWebSearchClient,
)
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMClient,
    OpenAICompatibleLLMClient,
)
from evolab_local.mining_platform.external.opsin_client import (
    OpsinClient,
    SystematicNameResolverClient,
)
from evolab_local.mining_platform.external.pubchem_client import (
    PubChemClient,
    PublicCompoundResolverClient,
)
from evolab_local.mining_platform.material_identity_judge_service import (
    MaterialIdentityJudgeService,
)
from evolab_local.mining_platform.material_public_resolver_service import (
    _candidate_from_opsin_compound,
    _candidate_from_pubchem_compound,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialIdentityEvidenceItem,
    MaterialIdentityEvidenceReviewAction,
    MaterialIdentityEvidenceRun,
    MaterialIdentityJudgment,
    MaterialStructureCandidate,
    PaperLocalMaterial,
    PaperMaterialStructureBundle,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialIdentityEvidenceItemRepository,
    MaterialIdentityEvidenceRunRepository,
    MaterialStructureCandidateRepository,
    PaperRepository,
)


PROMPT_VERSION = "material_identity_evidence_v1"
SOURCE_TIERS = {"A", "B", "C"}
REVIEW_DECISIONS = {"confirm": "confirmed", "reject": "rejected"}
TRUSTED_PUBLISHER_DOMAINS = {
    "pubs.rsc.org",
    "pubs.acs.org",
    "onlinelibrary.wiley.com",
    "advanced.onlinelibrary.wiley.com",
    "www.nature.com",
    "nature.com",
}


class SourcePageFetcher(Protocol):
    def fetch_text(self, url: str) -> str | None: ...


class PublisherPageFetcher:
    def __init__(self, timeout_seconds: float = 25.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_text(self, url: str) -> str | None:
        with httpx.Client(
            timeout=self.timeout_seconds,
            trust_env=False,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            response = client.get(url)
        if response.status_code >= 400:
            return None
        return response.text


class MaterialIdentityEvidenceService:
    def __init__(
        self,
        config: MiningPlatformConfig,
        *,
        anysearch_client: MaterialWebSearchClient | None = None,
        llm_client: LLMClient | None = None,
        opsin_client: SystematicNameResolverClient | None = None,
        pubchem_client: PublicCompoundResolverClient | None = None,
        judge_service: MaterialIdentityJudgeService | None = None,
        page_fetcher: SourcePageFetcher | None = None,
    ) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.material_resolution = MaterialResolutionService(config)
        self.papers = PaperRepository(self.database)
        self.runs = MaterialIdentityEvidenceRunRepository(self.database)
        self.items = MaterialIdentityEvidenceItemRepository(self.database)
        self.candidates = MaterialStructureCandidateRepository(self.database)
        self.anysearch_client = anysearch_client or (
            AnySearchClient(config.external_services.anysearch)
            if config.external_services.anysearch.api_key.strip()
            else None
        )
        self.llm_client = llm_client
        self.opsin_client = opsin_client or OpsinClient(config.external_services.opsin)
        self.pubchem_client = pubchem_client or PubChemClient(config.external_services.pubchem)
        self.judge_service = judge_service or MaterialIdentityJudgeService(config)
        self.page_fetcher = page_fetcher or PublisherPageFetcher()

    def init_runtime(self) -> None:
        self.material_resolution.init_runtime()

    def enrich_material_identity(
        self,
        paper_id: str,
        *,
        paper_material_id: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> PaperMaterialStructureBundle | None:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None or not bundle.candidate_run_id:
            return bundle
        material = next(
            (item for item in bundle.materials if item.paper_material_id == paper_material_id),
            None,
        )
        if material is None:
            raise ValueError(f"Material not found in paper: {paper_material_id}")
        if not self.anysearch_client:
            raise ValueError("AnySearch API key is not configured for evidence enrichment.")
        selected_provider, selected_model, client = self._llm(provider, model)
        trigger = _latest_trigger_judgment(bundle.identity_judgments, paper_material_id)
        paper = self.papers.get(bundle.paper_id)
        query_plan = _evidence_queries(material, paper.title if paper else None)
        timestamp = now_iso()
        run = MaterialIdentityEvidenceRun(
            evidence_run_id=uuid4().hex,
            paper_id=bundle.paper_id,
            candidate_run_id=bundle.candidate_run_id,
            paper_material_id=paper_material_id,
            trigger_judgment_id=trigger.judgment_id if trigger else None,
            provider=selected_provider,
            model=selected_model,
            query_plan=query_plan,
            status="running",
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.runs.upsert(run)
        try:
            search_results = self._search(query_plan)
            search_results = self._append_publisher_excerpts(search_results, material)
            response = client.generate_json(
                _evidence_messages(material, trigger, query_plan, search_results),
                model=selected_model,
                temperature=0,
            )
            evidence_items = self._store_extracted_items(
                run,
                response.parsed_json or {},
                search_results,
            )
            generated_candidates = self._generate_candidates(bundle, material, run, evidence_items)
            next_action = (
                "review_generated_candidates"
                if generated_candidates
                else "run_figure_pipeline"
                if not any(item.explicitly_linked for item in evidence_items)
                else "manual_review"
            )
            completed = run.model_copy(
                update={
                    "status": "completed",
                    "generated_candidate_ids": [
                        candidate.structure_candidate_id for candidate in generated_candidates
                    ],
                    "recommended_next_action": next_action,
                    "raw_response": {
                        "search_results": [
                            _search_result_payload(result) for result in search_results
                        ],
                        "llm_response": response.raw_response,
                        "parsed_response": response.parsed_json or {},
                    },
                    "updated_at": now_iso(),
                    "completed_at": now_iso(),
                }
            )
            self.runs.upsert(completed)
            if generated_candidates:
                return self.judge_service.judge_paper_candidates(
                    bundle.paper_id,
                    paper_material_id=paper_material_id,
                    provider=provider,
                    model=model,
                )
            return self.material_resolution.get_material_structure_bundle(bundle.paper_id)
        except Exception as exc:
            self.runs.upsert(
                run.model_copy(
                    update={
                        "status": "failed",
                        "error_message": str(exc),
                        "updated_at": now_iso(),
                        "completed_at": now_iso(),
                    }
                )
            )
            raise

    def review_evidence_item(
        self,
        evidence_item_id: str,
        action: MaterialIdentityEvidenceReviewAction,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> PaperMaterialStructureBundle | None:
        decision = REVIEW_DECISIONS.get(action.decision)
        if decision is None:
            raise ValueError("Evidence decision must be confirm or reject.")
        item = self.items.update_review(
            evidence_item_id,
            review_status=decision,
            reviewed_by=action.actor,
            review_note=action.message,
        )
        if item is None:
            return None
        if decision == "confirmed":
            return self.judge_service.judge_paper_candidates(
                item.paper_id,
                paper_material_id=item.paper_material_id,
                provider=provider,
                model=model,
            )
        return self.material_resolution.get_material_structure_bundle(item.paper_id)

    def _llm(self, provider: str | None, model: str | None) -> tuple[str, str, LLMClient]:
        selected_provider = provider or self.config.llm.default_provider
        provider_config = self.config.llm.providers.get(selected_provider)
        if provider_config is None:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown LLM provider {selected_provider!r}. Available: {known}")
        return (
            selected_provider,
            model or provider_config.default_model,
            self.llm_client or OpenAICompatibleLLMClient(provider_config),
        )

    def _search(self, query_plan: list[str]) -> list[AnySearchResult]:
        if not self.anysearch_client:
            return []
        results: list[AnySearchResult] = []
        seen_urls: set[str] = set()
        for query in query_plan:
            for result in self.anysearch_client.search(query):
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                results.append(result)
        return results

    def _append_publisher_excerpts(
        self,
        results: list[AnySearchResult],
        material: PaperLocalMaterial,
    ) -> list[AnySearchResult]:
        aliases = _material_aliases(material)
        fetched_count = 0
        enriched: list[AnySearchResult] = []
        for result in results:
            domain = urlparse(result.url).netloc.lower()
            contains_alias = any(
                normalize_token(alias)
                in normalize_token(" ".join([result.title, result.description, result.content]))
                for alias in aliases
                if alias
            )
            if fetched_count >= 3 or domain not in TRUSTED_PUBLISHER_DOMAINS or not contains_alias:
                enriched.append(result)
                continue
            try:
                page_text = self.page_fetcher.fetch_text(result.url)
            except Exception:
                page_text = None
            excerpt = _context_around_alias(page_text or "", aliases)
            if not excerpt:
                enriched.append(result)
                continue
            fetched_count += 1
            enriched.append(
                replace(
                    result,
                    content=(
                        f"{result.content}\nPublisher full-text excerpt near material alias: {excerpt}"
                    ).strip(),
                )
            )
        return enriched

    def _store_extracted_items(
        self,
        run: MaterialIdentityEvidenceRun,
        payload: dict[str, Any],
        search_results: list[AnySearchResult],
    ) -> list[MaterialIdentityEvidenceItem]:
        values = payload.get("evidence_items")
        if not isinstance(values, list):
            return []
        stored: list[MaterialIdentityEvidenceItem] = []
        for extracted in values:
            if not isinstance(extracted, dict):
                continue
            source_index = extracted.get("source_index")
            if (
                not isinstance(source_index, int)
                or source_index < 0
                or source_index >= len(search_results)
            ):
                continue
            source = search_results[source_index]
            timestamp = now_iso()
            tier = extracted.get("source_tier")
            stored.append(
                self.items.add(
                    MaterialIdentityEvidenceItem(
                        evidence_item_id=uuid4().hex,
                        evidence_run_id=run.evidence_run_id,
                        paper_id=run.paper_id,
                        candidate_run_id=run.candidate_run_id,
                        paper_material_id=run.paper_material_id,
                        source_tier=tier if tier in SOURCE_TIERS else "C",
                        source_title=source.title,
                        source_url=source.url,
                        query_text=_text(extracted.get("query_text")),
                        excerpt=_text(extracted.get("excerpt")) or source.description[:1000],
                        alias=_text(extracted.get("alias")),
                        full_name=_text(extracted.get("full_name")),
                        cas_number=_text(extracted.get("cas_number")),
                        pubchem_cid=_text(extracted.get("pubchem_cid")),
                        explicitly_linked=bool(extracted.get("explicitly_linked")),
                        confidence=_confidence(extracted.get("confidence")),
                        extraction=dict(extracted),
                        raw_source=_search_result_payload(source),
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )
            )
        return stored

    def _generate_candidates(
        self,
        bundle: PaperMaterialStructureBundle,
        material: PaperLocalMaterial,
        run: MaterialIdentityEvidenceRun,
        evidence_items: list[MaterialIdentityEvidenceItem],
    ) -> list[MaterialStructureCandidate]:
        candidates: list[MaterialStructureCandidate] = []
        seen_inchi_keys: set[str] = set()
        for item in evidence_items:
            if item.source_tier not in {"A", "B"} or not item.explicitly_linked:
                continue
            if item.full_name:
                compound = self.opsin_client.resolve_name(item.full_name)
                if compound:
                    candidate = _candidate_from_opsin_compound(
                        bundle=bundle,
                        material=material,
                        compound=compound,
                        query_text=item.full_name,
                        query_type="identity_evidence_full_name",
                        source_url=(
                            f"{self.config.external_services.opsin.base_url.rstrip('/')}/"
                            f"{quote(item.full_name, safe='')}.json"
                        ),
                    ).model_copy(
                        update={
                            "provider": "opsin_evidence",
                            "resolver_name": "identity_evidence_opsin",
                            "confidence": 0.9 if item.source_tier == "A" else 0.82,
                            "evidence": _candidate_evidence(run, item),
                        }
                    )
                    stored = self._store_candidate_preserving_review(candidate)
                    candidates.append(stored)
                    if stored.inchi_key:
                        seen_inchi_keys.add(stored.inchi_key)
            identifier = item.cas_number or item.pubchem_cid
            if not identifier:
                continue
            for compound in self.pubchem_client.resolve_name(identifier):
                if compound.inchi_key and compound.inchi_key in seen_inchi_keys:
                    continue
                candidate = _candidate_from_pubchem_compound(
                    bundle=bundle,
                    material=material,
                    compound=compound,
                    query_text=identifier,
                    query_type="identity_evidence_identifier",
                    discovery_evidence=_candidate_evidence(run, item),
                ).model_copy(
                    update={
                        "provider": "pubchem_evidence",
                        "resolver_name": "identity_evidence_pubchem",
                        "confidence": 0.9 if item.source_tier == "A" else 0.82,
                    }
                )
                stored = self._store_candidate_preserving_review(candidate)
                candidates.append(stored)
                if stored.inchi_key:
                    seen_inchi_keys.add(stored.inchi_key)
        return candidates

    def _store_candidate_preserving_review(
        self,
        candidate: MaterialStructureCandidate,
    ) -> MaterialStructureCandidate:
        existing = self.candidates.get_by_source(
            candidate.candidate_run_id,
            candidate.paper_material_id,
            candidate.provider,
            candidate.source_identifier,
        )
        if existing and existing.status in {"accepted", "rejected"}:
            return existing
        return self.candidates.upsert(candidate)


def _latest_trigger_judgment(
    judgments: list[MaterialIdentityJudgment],
    paper_material_id: str,
) -> MaterialIdentityJudgment | None:
    return next(
        (
            judgment
            for judgment in judgments
            if judgment.paper_material_id == paper_material_id
            and (
                judgment.verdict in {"ambiguous", "insufficient_evidence", "conflict"}
                or judgment.recommended_action == "search_more_evidence"
            )
        ),
        None,
    )


def _evidence_queries(material: PaperLocalMaterial, title: str | None) -> list[str]:
    alias = (
        material.abbreviation
        or material.entity_label
        or next(iter(material.mention_list), None)
        or material.paper_material_id
    )
    queries = [
        f'"{alias}" OLED full chemical name',
        f'"{alias}" OLED structure CAS',
        f'"{alias}" benzimidazole anthracene OLED',
    ]
    if title:
        queries.append(f'"{alias}" "{title}"')
    return queries


def _material_aliases(material: PaperLocalMaterial) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for value in [
                material.abbreviation,
                material.entity_label,
                *material.mention_list,
                material.normalized_name,
            ]
            if value
        )
    )


def _evidence_messages(
    material: PaperLocalMaterial,
    trigger: MaterialIdentityJudgment | None,
    query_plan: list[str],
    search_results: list[AnySearchResult],
) -> list[dict[str, str]]:
    context = {
        "paper_material": material.model_dump(mode="json"),
        "trigger_judgment": trigger.model_dump(mode="json") if trigger else None,
        "query_plan": query_plan,
        "sources": [
            {"source_index": index, **_search_result_payload(result)}
            for index, result in enumerate(search_results)
        ],
    }
    system_prompt = (
        "You extract evidence for resolving an ambiguous OLED material identity. "
        "Return only explicitly stated links between the paper alias and a full chemical name, "
        "CAS number, PubChem CID, or structure identifier. Do not infer a link from similarity. "
        "Source tier A means the target paper or its supporting information explicitly states the "
        "link; tier B means a scholarly article, authoritative database, or supplier page explicitly "
        "states it; tier C means a discovery lead or ambiguous mention. "
        "Return JSON with evidence_items, each containing source_index, source_tier (A/B/C), "
        "explicitly_linked (boolean), alias, full_name, cas_number, pubchem_cid, excerpt, "
        "confidence, and query_text. Keep null for absent identifiers. Exclude irrelevant sources."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Extract auditable identity evidence from these retrieved sources:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True),
        },
    ]


def _candidate_evidence(
    run: MaterialIdentityEvidenceRun,
    item: MaterialIdentityEvidenceItem,
) -> dict[str, Any]:
    return {
        "resolver": "material_identity_evidence",
        "identity_evidence_run_id": run.evidence_run_id,
        "identity_evidence_item_ids": [item.evidence_item_id],
        "identity_evidence_full_name": item.full_name,
        "identity_evidence_tier": item.source_tier,
        "identity_evidence_explicitly_linked": item.explicitly_linked,
        "web_search_results": [item.raw_source],
    }


def _search_result_payload(result: AnySearchResult) -> dict[str, Any]:
    return {
        "title": result.title,
        "url": result.url,
        "description": result.description[:1000],
        "content_excerpt": result.content[:1500],
        "cas_numbers": result.cas_numbers(),
        "score": result.score,
    }


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _confidence(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return max(0.0, min(float(value), 1.0))
    return None


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _context_around_alias(page_text: str, aliases: list[str]) -> str | None:
    if not page_text:
        return None
    plain_text = html.unescape(re.sub(r"<[^>]+>", " ", page_text))
    plain_text = re.sub(r"\s+", " ", plain_text).strip()
    lower_text = plain_text.casefold()
    for alias in aliases:
        index = lower_text.find(alias.casefold())
        if index >= 0:
            start = max(0, index - 500)
            end = min(len(plain_text), index + len(alias) + 500)
            return plain_text[start:end]
    return None
