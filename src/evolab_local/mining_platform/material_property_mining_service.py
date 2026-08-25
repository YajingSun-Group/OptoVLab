from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.openai_compatible_client import (
    LLMClient,
    OpenAICompatibleLLMClient,
)
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.mining.llm_prompt_builder import PromptSource, sources_from_document_blocks
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialPropertyCandidate,
    PaperLocalMaterial,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    DocumentBlockRepository,
    MaterialPropertyCandidateRepository,
)


PROPERTY_CATEGORY_BY_NAME = {
    "HOMO": "electronic",
    "LUMO": "electronic",
    "S1": "electronic",
    "T1": "electronic",
    "delta_EST": "electronic",
    "optical_bandgap": "electronic",
    "electrochemical_gap": "electronic",
    "PLQY": "photophysical",
    "emission_peak": "photophysical",
    "absorption_peak": "photophysical",
    "FWHM_material": "photophysical",
    "prompt_lifetime": "photophysical",
    "delayed_lifetime": "photophysical",
    "phosphorescence_lifetime": "photophysical",
    "kRISC": "photophysical",
    "kISC": "photophysical",
    "radiative_rate": "photophysical",
    "nonradiative_rate": "photophysical",
    "Td": "thermal",
    "Tg": "thermal",
    "Tm": "thermal",
    "crystallization_temperature": "thermal",
    "oxidation_potential": "electrochemical",
    "reduction_potential": "electrochemical",
    "onset_oxidation_potential": "electrochemical",
    "onset_reduction_potential": "electrochemical",
    "dipole_moment": "computed",
    "oscillator_strength": "computed",
    "spin_orbit_coupling": "computed",
    "reorganization_energy": "computed",
}

PROPERTY_NAME_ALIASES = {
    "delta est": "delta_EST",
    "delta_est": "delta_EST",
    "dest": "delta_EST",
    "Δest": "delta_EST",
    "ΔEST": "delta_EST",
    "bandgap": "optical_bandgap",
    "optical gap": "optical_bandgap",
    "eg": "optical_bandgap",
    "photoluminescence quantum yield": "PLQY",
    "plqy": "PLQY",
    "pl quantum yield": "PLQY",
    "emission wavelength": "emission_peak",
    "emission maximum": "emission_peak",
    "pl peak": "emission_peak",
    "absorption maximum": "absorption_peak",
    "absorption wavelength": "absorption_peak",
    "fwhm": "FWHM_material",
    "td": "Td",
    "t d": "Td",
    "tg": "Tg",
    "tm": "Tm",
    "oxidation potential": "oxidation_potential",
    "reduction potential": "reduction_potential",
}

EVIDENCE_FUZZY_MATCH_THRESHOLD = 0.82
MAX_EVIDENCE_FUZZY_FRAGMENTS = 48
MAX_EVIDENCE_FRAGMENT_CHARS = 700
MAX_LLM_EVIDENCE_CHARS = 1000
MAX_JSON_REPAIR_CONTENT_CHARS = 30000


PROPERTY_KEYWORDS = (
    "HOMO",
    "LUMO",
    "S1",
    "T1",
    "ΔEST",
    "delta EST",
    "bandgap",
    "band gap",
    "PLQY",
    "quantum yield",
    "emission peak",
    "emission maximum",
    "absorption peak",
    "FWHM",
    "lifetime",
    "delayed fluorescence",
    "prompt fluorescence",
    "kRISC",
    "kISC",
    "Td",
    "Tg",
    "Tm",
    "decomposition temperature",
    "glass transition",
    "oxidation potential",
    "reduction potential",
    "cyclic voltammetry",
    "CV",
)


@dataclass(frozen=True)
class MaterialPropertyMiningResult:
    paper_id: str
    candidate_run_id: str | None = None
    candidates: list[MaterialPropertyCandidate] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    source_count: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)


class MaterialPropertyMiningService:
    def __init__(self, config: MiningPlatformConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.blocks = DocumentBlockRepository(self.database)
        self.material_resolution = MaterialResolutionService(config)
        self.property_candidates = MaterialPropertyCandidateRepository(self.database)
        self.llm_client = llm_client

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def mine_paper_properties(
        self,
        paper_id: str,
        *,
        paper_material_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> MaterialPropertyMiningResult | None:
        if not self.config.features.material_properties:
            raise ValueError(
                "Material property extraction is disabled in the current runtime version. "
                "Set features.material_properties=true to re-enable it for development."
            )
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.paper_service.get_paper(normalized_paper_id):
            return None
        bundle = self.material_resolution.resolve_paper_materials(normalized_paper_id)
        if not bundle or not bundle.candidate_run_id:
            return MaterialPropertyMiningResult(paper_id=normalized_paper_id)
        materials = bundle.materials
        if paper_material_id:
            materials = [
                material for material in materials if material.paper_material_id == paper_material_id
            ]
        if not materials:
            return MaterialPropertyMiningResult(
                paper_id=normalized_paper_id,
                candidate_run_id=bundle.candidate_run_id,
                skipped=[{"reason": "paper_material_id_not_found", "paper_material_id": paper_material_id}],
            )

        blocks = self.blocks.list_by_paper(normalized_paper_id)
        sources = _select_property_sources(
            sources_from_document_blocks(blocks),
            materials=materials,
            max_chars=self.config.llm.max_source_chars,
        )
        if not sources:
            return MaterialPropertyMiningResult(
                paper_id=normalized_paper_id,
                candidate_run_id=bundle.candidate_run_id,
                skipped=[{"reason": "no_property_source_blocks"}],
            )
        selected_provider = provider or self.config.llm.default_provider
        provider_config = self._provider_config(selected_provider)
        selected_model = model or provider_config.default_model
        messages = build_material_property_mining_messages(materials=materials, sources=sources)
        client = self.llm_client or OpenAICompatibleLLMClient(provider_config)
        response = client.generate_json(messages, model=selected_model, max_tokens=8192)
        if response.parsed_json is None:
            repair_messages = build_material_property_json_repair_messages(
                response.content,
                parse_error=response.parse_error,
            )
            response = client.generate_json(
                repair_messages,
                model=selected_model,
                temperature=0,
                max_tokens=8192,
            )
        if response.parsed_json is None:
            detail = f": {response.parse_error}" if response.parse_error else ""
            raise ValueError(f"LLM response did not contain a JSON object{detail}.")
        candidates, skipped = material_property_candidates_from_response(
            response.parsed_json,
            paper_id=normalized_paper_id,
            candidate_run_id=bundle.candidate_run_id,
            materials=materials,
            sources=sources,
            provider=selected_provider,
            model=selected_model,
        )
        stored = [self.property_candidates.upsert(candidate) for candidate in candidates]
        return MaterialPropertyMiningResult(
            paper_id=normalized_paper_id,
            candidate_run_id=bundle.candidate_run_id,
            candidates=stored,
            skipped=skipped,
            source_count=len(sources),
            raw_response=response.parsed_json,
        )

    def _provider_config(self, provider: str):
        provider_config = self.config.llm.providers.get(provider)
        if not provider_config:
            known = ", ".join(sorted(self.config.llm.providers))
            raise ValueError(f"Unknown LLM provider {provider!r}. Available: {known}")
        return provider_config


def build_material_property_mining_messages(
    *,
    materials: list[PaperLocalMaterial],
    sources: list[PromptSource],
) -> list[dict[str, str]]:
    material_payload = [
        {
            "paper_material_id": material.paper_material_id,
            "mentions": material.mention_list,
            "full_name_in_paper": material.full_name_in_paper,
            "normalized_name": material.normalized_name,
            "canonical_name": material.canonical_name,
            "abbreviation": material.abbreviation,
            "paper_specific_label": material.paper_specific_label,
            "material_class": material.material_class,
            "used_in": [usage.model_dump(mode="json") for usage in material.used_in],
        }
        for material in materials
    ]
    source_payload = [
        {
            "block_id": source.block_id,
            "page_id": source.page_id,
            "source_type": source.source_type,
            "bbox": source.bbox,
            "text": source.text,
        }
        for source in sources
    ]
    system = (
        "You are a precise OLED material-property mining engine. Output only one JSON object. "
        "Extract paper-reported material properties only from the supplied source blocks. "
        "Do not use outside knowledge and do not extract OLED device performance metrics."
    )
    user = (
        "Extract material property candidates for the provided paper-local materials.\n\n"
        "Allowed property_name values:\n"
        f"{json.dumps(sorted(PROPERTY_CATEGORY_BY_NAME), ensure_ascii=False)}\n\n"
        "Return JSON with this shape:\n"
        "{\n"
        '  "properties": [\n'
        "    {\n"
        '      "paper_material_id": "one ID from PAPER_MATERIALS_JSON, or null if unclear",\n'
        '      "property_name": "one allowed property_name",\n'
        '      "property_category": "electronic|photophysical|thermal|electrochemical|computed",\n'
        '      "value_numeric": 83.0,\n'
        '      "value_text": null,\n'
        '      "value_raw": "83%",\n'
        '      "unit": "%",\n'
        '      "condition": {"sample_form": "doped_film", "host": "mCBP"},\n'
        '      "method": "integrating_sphere",\n'
        '      "source_type": "text|table|caption|figure|scheme|unknown",\n'
        '      "evidence_text": "short exact copied substring from the selected source block",\n'
        '      "evidence_block_id": "block_id copied from SOURCE_BLOCKS_JSON",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. paper_material_id must be copied exactly from PAPER_MATERIALS_JSON. Use null if unclear.\n"
        "2. evidence_block_id must be copied exactly from SOURCE_BLOCKS_JSON.\n"
        "3. Return at most 25 total property records; choose the highest-confidence material properties.\n"
        "4. evidence_text must be copied verbatim from the selected source block. Do not paraphrase, summarize, normalize symbols, change punctuation, or rewrite formulas.\n"
        "5. evidence_text must be concise, preferably one sentence or a short substring, and must not exceed 250 characters.\n"
        "6. Prefer a fragment that contains both the material mention and the reported value. If one sentence is too long, copy the shortest exact substring that still supports the value.\n"
        "7. The response must be valid JSON. Use standard JSON string escaping for backslashes and double quotes in evidence_text. Do not put raw newlines inside JSON strings.\n"
        "8. If an exact evidence fragment contains LaTeX/math backslashes or many quotes and you cannot safely escape them, choose a nearby exact plain-text fragment from the same block, or omit the property.\n"
        "9. If the evidence contains special characters, Greek letters, subscripts, superscripts, or math artifacts, copy them exactly only when the resulting JSON string remains valid.\n"
        "10. If you cannot find exact source text for a property, do not output that property.\n"
        "11. If the same property appears under multiple conditions, output multiple records.\n"
        "12. Keep value_raw exactly as written; use value_numeric only when the number is explicit.\n"
        "13. Do not extract EQE, CE, PE, LT50/LT80/LT95, turn-on voltage, or other device performance metrics.\n\n"
        f"PAPER_MATERIALS_JSON:\n{json.dumps(material_payload, ensure_ascii=False, indent=2)}\n\n"
        f"SOURCE_BLOCKS_JSON:\n{json.dumps(source_payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_material_property_json_repair_messages(
    content: str,
    *,
    parse_error: str | None = None,
) -> list[dict[str, str]]:
    clipped_content = content[:MAX_JSON_REPAIR_CONTENT_CHARS]
    system = (
        "You repair invalid JSON produced by an OLED material-property mining step. "
        "Output only one valid JSON object. Do not add markdown."
    )
    user = (
        "Repair the following invalid JSON-like response into valid JSON.\n"
        "Keep the same schema: {\"properties\": [...]}.\n"
        "Preserve all property values, material IDs, block IDs, and evidence_text strings exactly when possible.\n"
        "Do not paraphrase evidence_text. Only escape characters required by JSON syntax.\n"
        "If a property object is too broken to repair safely, omit that one object.\n"
        "Return valid JSON only.\n\n"
        f"Parse error: {parse_error or 'unknown'}\n\n"
        f"INVALID_RESPONSE:\n{clipped_content}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def material_property_candidates_from_response(
    payload: dict[str, Any],
    *,
    paper_id: str,
    candidate_run_id: str,
    materials: list[PaperLocalMaterial],
    sources: list[PromptSource],
    provider: str,
    model: str,
) -> tuple[list[MaterialPropertyCandidate], list[dict[str, Any]]]:
    raw_items = payload.get("properties")
    if raw_items is None:
        raw_items = payload.get("material_properties")
    if not isinstance(raw_items, list):
        raise ValueError("Material property response must contain a properties[] array.")
    materials_by_id = {material.paper_material_id: material for material in materials}
    sources_by_id = {source.block_id: source for source in sources}
    now = now_iso()
    candidates: list[MaterialPropertyCandidate] = []
    skipped: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            skipped.append({"index": index, "reason": "not_object"})
            continue
        material_id = _str_or_none(raw_item.get("paper_material_id"))
        if material_id not in materials_by_id:
            skipped.append({"index": index, "reason": "invalid_paper_material_id", "value": material_id})
            continue
        property_name = _normalize_property_name(raw_item.get("property_name"))
        if property_name is None:
            skipped.append({"index": index, "reason": "invalid_property_name", "value": raw_item.get("property_name")})
            continue
        source = _source_for_property(raw_item, sources_by_id)
        if source is None:
            skipped.append({"index": index, "reason": "evidence_block_not_found"})
            continue
        evidence_match = _evidence_match(raw_item, source)
        if evidence_match is None:
            skipped.append(
                {
                    "index": index,
                    "reason": "evidence_text_below_similarity_threshold",
                    "threshold": EVIDENCE_FUZZY_MATCH_THRESHOLD,
                }
            )
            continue
        evidence_text = evidence_match.evidence_text
        value_numeric = _float_or_none(raw_item.get("value_numeric"))
        value_raw = _str_or_none(raw_item.get("value_raw"))
        if value_numeric is None:
            value_numeric = _parse_first_number(value_raw or _str_or_none(raw_item.get("value_text")))
        unit = _str_or_none(raw_item.get("unit"))
        normalized_value_numeric, normalized_unit = _normalized_value(property_name, value_numeric, unit)
        property_category = _str_or_none(raw_item.get("property_category")) or PROPERTY_CATEGORY_BY_NAME[property_name]
        candidate = MaterialPropertyCandidate(
            property_candidate_id=_stable_property_candidate_id(
                candidate_run_id,
                material_id,
                property_name,
                source.block_id,
                value_raw,
                value_numeric,
                unit,
                evidence_text,
            ),
            paper_id=paper_id,
            candidate_run_id=candidate_run_id,
            paper_material_id=material_id,
            property_name=property_name,
            property_category=property_category,
            value_numeric=value_numeric,
            value_text=_str_or_none(raw_item.get("value_text")),
            value_raw=value_raw,
            unit=unit,
            normalized_value_numeric=normalized_value_numeric,
            normalized_unit=normalized_unit,
            condition=_dict_or_empty(raw_item.get("condition")),
            method=_str_or_none(raw_item.get("method")),
            source_type=_str_or_none(raw_item.get("source_type")) or source.source_type,
            evidence_text=evidence_text,
            llm_evidence_text=evidence_match.llm_evidence_text,
            source_block_text=evidence_match.source_block_text,
            evidence_anchor={
                "block_id": source.block_id,
                "page_id": source.page_id,
                "bbox": source.bbox,
                "source_type": source.source_type,
                "evidence_match_score": evidence_match.score,
                "evidence_match_method": evidence_match.method,
                "evidence_similarity_threshold": EVIDENCE_FUZZY_MATCH_THRESHOLD,
            },
            provider=provider,
            model=model,
            prompt_version="material_property_miner_v1",
            confidence=_clamp_confidence(raw_item.get("confidence")),
            status="pending_review",
            created_at=now,
            updated_at=now,
        )
        candidates.append(candidate)
    return candidates, skipped


def _select_property_sources(
    sources: list[PromptSource],
    *,
    materials: list[PaperLocalMaterial],
    max_chars: int,
) -> list[PromptSource]:
    material_terms = _material_terms(materials)
    scored = [
        (index, source, _property_source_score(source, material_terms))
        for index, source in enumerate(sources)
        if source.text.strip()
    ]
    scored = [item for item in scored if item[2] > 0]
    scored.sort(key=lambda item: (-item[2], item[0]))
    selected: list[PromptSource] = []
    used_chars = 0
    for _, source, _score in scored:
        source_chars = len(source.text)
        if selected and used_chars + source_chars > max_chars:
            continue
        selected.append(source)
        used_chars += source_chars
        if used_chars >= max_chars:
            break
    return sorted(selected, key=lambda source: (source.page_id or 0, source.block_id))


def _property_source_score(source: PromptSource, material_terms: set[str]) -> int:
    text = source.text.lower()
    score = 0
    for keyword in PROPERTY_KEYWORDS:
        if keyword.lower() in text:
            score += 4
    for term in material_terms:
        if term and term.lower() in text:
            score += 2
    if source.source_type in {"table", "caption"}:
        score += 1
    return score


def _material_terms(materials: list[PaperLocalMaterial]) -> set[str]:
    terms: set[str] = set()
    for material in materials:
        terms.update(term.strip() for term in material.mention_list if term.strip())
        for value in (
            material.full_name_in_paper,
            material.normalized_name,
            material.canonical_name,
            material.abbreviation,
            material.paper_specific_label,
        ):
            if value and value.strip():
                terms.add(value.strip())
    return terms


def _source_for_property(
    item: dict[str, Any],
    sources_by_id: dict[str, PromptSource],
) -> PromptSource | None:
    block_id = _str_or_none(item.get("evidence_block_id") or item.get("block_id"))
    if block_id and block_id in sources_by_id:
        return sources_by_id[block_id]
    evidence_text = _str_or_none(item.get("evidence_text"))
    if evidence_text:
        lowered = evidence_text.lower()
        for source in sources_by_id.values():
            if lowered in source.text.lower():
                return source
    return None


@dataclass(frozen=True)
class EvidenceTextMatch:
    evidence_text: str
    llm_evidence_text: str | None
    source_block_text: str
    score: float
    method: str


def _evidence_match(item: dict[str, Any], source: PromptSource) -> EvidenceTextMatch | None:
    source_block_text = source.text.strip()
    if not source_block_text:
        return None
    llm_evidence_text = _str_or_none(item.get("evidence_text"))
    if not llm_evidence_text:
        fallback = source_block_text[:500]
        return EvidenceTextMatch(
            evidence_text=fallback,
            llm_evidence_text=None,
            source_block_text=source_block_text,
            score=1.0,
            method="source_block_fallback",
        )
    llm_evidence_for_match = llm_evidence_text[:MAX_LLM_EVIDENCE_CHARS]
    if llm_evidence_for_match.lower() in source_block_text.lower():
        return EvidenceTextMatch(
            evidence_text=_shorten_source_fragment(llm_evidence_for_match),
            llm_evidence_text=llm_evidence_text,
            source_block_text=source_block_text,
            score=1.0,
            method="exact_substring",
        )
    fragment, score = _best_source_fragment_for_evidence(llm_evidence_for_match, source_block_text)
    if score < EVIDENCE_FUZZY_MATCH_THRESHOLD:
        return None
    return EvidenceTextMatch(
        evidence_text=fragment,
        llm_evidence_text=llm_evidence_text,
        source_block_text=source_block_text,
        score=round(score, 4),
        method="fuzzy_source_fragment",
    )


def _best_source_fragment_for_evidence(evidence_text: str, source_text: str) -> tuple[str, float]:
    evidence_norm = _normalize_for_similarity(evidence_text)
    if not evidence_norm:
        return source_text[:500], 0.0
    best_fragment = source_text[:500]
    best_score = _similarity(evidence_norm, _normalize_for_similarity(best_fragment))
    for fragment in _candidate_source_fragments(source_text):
        score = _similarity(evidence_norm, _normalize_for_similarity(fragment))
        if score > best_score:
            best_fragment = fragment
            best_score = score
    return best_fragment, best_score


@lru_cache(maxsize=512)
def _candidate_source_fragments(source_text: str) -> tuple[str, ...]:
    fragments: list[str] = []
    seen: set[str] = set()

    def add_fragment(fragment: str) -> None:
        if len(fragments) >= MAX_EVIDENCE_FUZZY_FRAGMENTS:
            return
        shortened = _shorten_source_fragment(fragment)
        if not shortened:
            return
        key = _normalize_for_similarity(shortened)
        if key and key not in seen:
            seen.add(key)
            fragments.append(shortened)

    for part in re.split(r"(?<=[.!?。；;])\s+", source_text):
        add_fragment(part)
    if len(fragments) < MAX_EVIDENCE_FUZZY_FRAGMENTS:
        step = max(200, MAX_EVIDENCE_FRAGMENT_CHARS // 2)
        for start in range(0, len(source_text), step):
            add_fragment(source_text[start : start + MAX_EVIDENCE_FRAGMENT_CHARS])
            if len(fragments) >= MAX_EVIDENCE_FUZZY_FRAGMENTS:
                break
    if not fragments:
        add_fragment(source_text)
    return tuple(fragments)


def _shorten_source_fragment(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= MAX_EVIDENCE_FRAGMENT_CHARS:
        return text
    return text[:MAX_EVIDENCE_FRAGMENT_CHARS].rsplit(" ", 1)[0].strip()


def _normalize_for_similarity(value: str) -> str:
    normalized = value.lower().replace("−", "-").replace("–", "-").replace("—", "-")
    normalized = normalized.replace("φ", "phi").replace("Φ", "phi")
    normalized = re.sub(r"[^a-z0-9.%+\-/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    return overlap / len(left_tokens)


def _normalize_property_name(value: object) -> str | None:
    raw = _str_or_none(value)
    if not raw:
        return None
    if raw in PROPERTY_CATEGORY_BY_NAME:
        return raw
    alias_key = re.sub(r"\s+", " ", raw.strip()).lower()
    if alias_key in PROPERTY_NAME_ALIASES:
        return PROPERTY_NAME_ALIASES[alias_key]
    for candidate in PROPERTY_CATEGORY_BY_NAME:
        if candidate.lower() == alias_key:
            return candidate
    return None


def _stable_property_candidate_id(
    candidate_run_id: str,
    paper_material_id: str,
    property_name: str,
    block_id: str,
    value_raw: str | None,
    value_numeric: float | None,
    unit: str | None,
    evidence_text: str,
) -> str:
    payload = "|".join(
        [
            candidate_run_id,
            paper_material_id,
            property_name,
            block_id,
            value_raw or "",
            "" if value_numeric is None else f"{value_numeric:.12g}",
            unit or "",
            evidence_text,
        ]
    )
    return "prop_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


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


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    text = _str_or_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_first_number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def _dict_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _clamp_confidence(value: object) -> float | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return max(0.0, min(1.0, numeric))
