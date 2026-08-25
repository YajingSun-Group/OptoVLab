from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.mining.mineru_parse_service import mineru_item_text
from evolab_local.mining_platform.schemas.external_runs import MinerUParseRun
from evolab_local.mining_platform.schemas.material_structure import ChemicalFigureBlock
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    CandidateIngestionRepository,
    ChemicalFigureBlockRepository,
    MinerUParseRunRepository,
    PaperRepository,
)


VISUAL_BLOCK_TYPES = {"image", "chart", "table"}

KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "keyword_molecular_structure",
        re.compile(
            r"\b(molecular|chemical|clar)\s+structures?\b|\bstructures?\s+of\b",
            re.IGNORECASE,
        ),
    ),
    (
        "keyword_scheme",
        re.compile(
            r"\bscheme\s*\d*\b|\bsynthetic\s+route\b|\bsynthesis\b|\breaction\s+scheme\b",
            re.IGNORECASE,
        ),
    ),
    (
        "keyword_molecular_design",
        re.compile(r"\bmolecular\s+design\b|\bdesign\s+concept\b", re.IGNORECASE),
    ),
    (
        "keyword_electronic_structure",
        re.compile(r"\bHOMO\b|\bLUMO\b|\borbital\b|\benergy\s+level\b", re.IGNORECASE),
    ),
    (
        "keyword_compound_label",
        re.compile(r"\bcompound\s+\d+\b|\bmolecule\s+[A-Za-z0-9-]+\b", re.IGNORECASE),
    ),
    (
        "keyword_oled_material",
        re.compile(r"\bemitter[s]?\b|\bdopant[s]?\b|\bhost[s]?\b", re.IGNORECASE),
    ),
)

MINERU_OUTPUT_ROOT = Path(
    os.getenv("MINERU_OUTPUT_ROOT", "runtime/mining_platform/mineru_external")
)


class ChemicalFigureCollectorService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.mineru_runs = MinerUParseRunRepository(self.database)
        self.candidates = CandidateIngestionRepository(self.database)
        self.figure_blocks = ChemicalFigureBlockRepository(self.database)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def collect_for_paper(self, paper_id: str) -> list[ChemicalFigureBlock] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        run = self.mineru_runs.latest_completed_by_paper(normalized_paper_id)
        if not run or not run.content_list_path:
            return []
        content_list = _read_content_list(Path(run.content_list_path))
        result_key = _result_key(run)
        material_terms = self._paper_material_terms(normalized_paper_id)
        blocks = collect_chemical_figure_blocks(
            run=run,
            content_list=content_list,
            result_key=result_key,
            material_terms=material_terms,
        )
        return self.figure_blocks.replace_for_run(run.mineru_run_id, blocks)

    def list_for_paper(self, paper_id: str) -> list[ChemicalFigureBlock] | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        if not self.papers.get(normalized_paper_id):
            return None
        return self.figure_blocks.list_by_paper(normalized_paper_id)

    def get_image_path(self, figure_block_id: str) -> Path | None:
        self.init_runtime()
        block = self.figure_blocks.get(figure_block_id)
        if not block or not block.resolved_img_path:
            return None
        path = Path(block.resolved_img_path)
        return path if path.exists() and path.is_file() else None

    def _paper_material_terms(self, paper_id: str) -> list[str]:
        runs = self.candidates.list_runs_by_paper(paper_id)
        run = next((item for item in runs if item.status == "completed"), None)
        if not run:
            return []
        terms: list[str] = []
        for material in _list_mapping(run.mining_result.get("materials")):
            for key in (
                "paper_material_id",
                "entity_label",
                "canonical_name",
                "full_name_in_paper",
                "normalized_name",
                "abbreviation",
                "paper_specific_label",
            ):
                terms.extend(_string_or_list(material.get(key)))
            terms.extend(_string_or_list(material.get("mention_list")))
        return _dedupe_terms(terms)


def collect_chemical_figure_blocks(
    *,
    run: MinerUParseRun,
    content_list: list[dict[str, Any]],
    result_key: str | None = None,
    material_terms: list[str] | None = None,
) -> list[ChemicalFigureBlock]:
    timestamp = now_iso()
    blocks: list[ChemicalFigureBlock] = []
    for content_index, item in enumerate(content_list):
        content_type = str(item.get("type") or "")
        if content_type not in VISUAL_BLOCK_TYPES:
            continue
        caption = _caption_text(item)
        nearby_text = _nearby_text(content_list, content_index)
        search_text = " ".join(
            value for value in (mineru_item_text(item), caption, nearby_text) if value
        )
        score = _score_candidate(item, search_text, material_terms or [])
        if not score["tags"]:
            continue
        img_path = _image_path(item)
        resolved_img_path, image_candidates = _resolve_image_path(run, result_key, img_path)
        page_idx = _int_or_none(item.get("page_idx"))
        source_json = {
            "content_item": item,
            "image_path_candidates": [path.as_posix() for path in image_candidates],
            "result_key": result_key,
        }
        blocks.append(
            ChemicalFigureBlock(
                figure_block_id=uuid4().hex,
                paper_id=run.paper_id,
                mineru_run_id=run.mineru_run_id,
                content_index=content_index,
                content_type=content_type,
                sub_type=_str_or_none(item.get("sub_type")),
                page_idx=page_idx,
                page_id=page_idx + 1 if page_idx is not None else None,
                bbox=_bbox(item),
                img_path=img_path,
                resolved_img_path=resolved_img_path.as_posix() if resolved_img_path else None,
                image_exists=bool(resolved_img_path and resolved_img_path.exists()),
                caption=caption or None,
                nearby_text=nearby_text or None,
                heuristic_tags=score["tags"],
                confidence=score["confidence"],
                status="pending_classification",
                source_json=source_json,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
    return blocks


def _score_candidate(
    item: Mapping[str, Any],
    search_text: str,
    material_terms: list[str],
) -> dict[str, Any]:
    content_type = str(item.get("type") or "")
    sub_type = _str_or_none(item.get("sub_type"))
    tags: list[str] = []
    confidence = 0.0
    if sub_type == "chemical":
        tags.append("mineru_subtype_chemical")
        confidence = max(confidence, 0.85)
    for tag, pattern in KEYWORD_PATTERNS:
        if pattern.search(search_text):
            tags.append(tag)
            confidence = max(confidence, 0.72 if content_type == "image" else 0.52)
    normalized_text = _normalize_for_search(search_text)
    if _matches_material_term(normalized_text, material_terms):
        tags.append("paper_material_mention")
        confidence = max(confidence, 0.62 if content_type == "image" else 0.42)

    tags = sorted(set(tags))
    strong_visual_tags = (
        "keyword_molecular_structure",
        "keyword_scheme",
        "keyword_compound_label",
        "keyword_electronic_structure",
        "mineru_subtype_chemical",
    )
    if content_type in {"chart", "table"} and not any(tag in tags for tag in strong_visual_tags):
        return {"tags": [], "confidence": None}
    if "mineru_subtype_chemical" in tags and len(tags) > 1:
        confidence = max(confidence, 0.9)
    return {"tags": tags, "confidence": round(min(confidence, 0.95), 2) if tags else None}


def _read_content_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"MinerU content_list must be a JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _result_key(run: MinerUParseRun) -> str | None:
    if not run.result_path:
        return None
    result_path = Path(run.result_path)
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or not results:
        return None
    return str(next(iter(results.keys())))


def _caption_text(item: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "image_caption",
        "image_footnote",
        "chart_caption",
        "chart_footnote",
        "table_caption",
        "table_footnote",
        "text",
    ):
        parts.extend(_string_or_list(item.get(key)))
    if isinstance(item.get("table_body"), str):
        parts.append(re.sub(r"<[^>]+>", " ", str(item["table_body"])))
    if isinstance(item.get("content"), str):
        parts.append(str(item["content"]))
    return _compact_text(" ".join(parts))


def _nearby_text(content_list: list[dict[str, Any]], content_index: int, window: int = 2) -> str:
    item = content_list[content_index]
    page_idx = item.get("page_idx")
    nearby: list[str] = []
    lower = max(0, content_index - window)
    upper = min(len(content_list), content_index + window + 1)
    for index in range(lower, upper):
        if index == content_index:
            continue
        candidate = content_list[index]
        if candidate.get("page_idx") != page_idx:
            continue
        text = mineru_item_text(candidate) or _caption_text(candidate)
        if text:
            nearby.append(text)
    return _compact_text(" ".join(nearby))[:4000]


def _resolve_image_path(
    run: MinerUParseRun,
    result_key: str | None,
    img_path: str | None,
) -> tuple[Path | None, list[Path]]:
    if not img_path:
        return None, []
    raw_path = Path(img_path)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    if run.content_list_path:
        run_dir = Path(run.content_list_path).parent
        candidates.extend([run_dir / raw_path, run_dir / "hybrid_auto" / raw_path])
    if run.task_id:
        parser_version = run.parser_version or "3.1.0"
        for paper_name in _dedupe_terms([result_key, run.paper_id]):
            candidates.append(
                MINERU_OUTPUT_ROOT
                / parser_version
                / "output"
                / run.task_id
                / paper_name
                / "hybrid_auto"
                / raw_path
            )
    resolved = next((path for path in candidates if path.exists()), None)
    return resolved, candidates


def _image_path(item: Mapping[str, Any]) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _bbox(item: Mapping[str, Any]) -> list[float]:
    value = item.get("bbox")
    if not isinstance(value, list):
        return []
    return [float(raw) for raw in value if isinstance(raw, int | float)]


def _string_or_list(value: object) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _list_mapping(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _str_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _normalize_for_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _matches_material_term(normalized_text: str, material_terms: list[str]) -> bool:
    for term in material_terms:
        normalized_term = _normalize_for_search(term)
        if len(normalized_term) >= 3 and normalized_term in normalized_text:
            return True
    return False


def _dedupe_terms(values: list[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        compact = _compact_text(str(value))
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(compact)
    return output
