from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.external.mineru_client import MinerUClient, MinerUParsedDocument
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.document import DocumentBlock
from evolab_local.mining_platform.schemas.external_runs import MinerUParseRun
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    DocumentBlockRepository,
    MinerUParseRunRepository,
    PaperRepository,
)


class MinerUParseService:
    def __init__(self, config: MiningPlatformConfig, client: MinerUClient | None = None) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.blocks = DocumentBlockRepository(self.database)
        self.runs = MinerUParseRunRepository(self.database)
        self.client = client or MinerUClient(config.external_services.mineru)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def parse_paper(
        self,
        paper_id: str,
        *,
        include_images: bool = False,
    ) -> MinerUParseRun | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        paper = self.paper_service.get_paper(normalized_paper_id)
        if not paper:
            return None
        pdf_path = self.paper_service.get_pdf_path(normalized_paper_id)
        if not pdf_path:
            self.papers.set_parse_status(normalized_paper_id, "failed")
            raise FileNotFoundError(f"PDF not found for paper_id={normalized_paper_id}")

        run = MinerUParseRun(
            mineru_run_id=uuid4().hex,
            paper_id=normalized_paper_id,
            status="running",
            service_base_url=self.config.external_services.mineru.base_url,
            created_at=now_iso(),
        )
        self.runs.create(run)
        self.papers.set_parse_status(normalized_paper_id, "parsing")
        try:
            parsed = (
                self.client.parse_pdf(pdf_path, return_images=True)
                if include_images
                else self.client.parse_pdf(pdf_path)
            )
            result_paths = write_mineru_outputs(
                self.config.paths.mineru_runs_dir,
                run.mineru_run_id,
                parsed,
                images_requested=include_images,
            )
            blocks = mineru_content_list_to_blocks(normalized_paper_id, parsed.content_list)
            self.blocks.replace_for_paper(normalized_paper_id, blocks)
            completed = run.model_copy(
                update={
                    "task_id": parsed.task_id,
                    "status": "completed",
                    "parser_version": parsed.version,
                    "content_item_count": len(parsed.content_list),
                    "result_path": result_paths["result_path"],
                    "content_list_path": result_paths["content_list_path"],
                    "markdown_path": result_paths["markdown_path"],
                    "completed_at": now_iso(),
                }
            )
            self.runs.update(completed)
            self.papers.set_parse_status(normalized_paper_id, "parsed")
            return completed
        except Exception as exc:
            failed = run.model_copy(
                update={
                    "status": "failed",
                    "error_message": str(exc),
                    "completed_at": now_iso(),
                }
            )
            self.runs.update(failed)
            self.papers.set_parse_status(normalized_paper_id, "failed")
            raise


def write_mineru_outputs(
    root_dir: Path,
    run_id: str,
    parsed: MinerUParsedDocument,
    *,
    images_requested: bool = False,
) -> dict[str, str]:
    run_dir = root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    content_list_path = run_dir / "content_list.json"
    markdown_path = run_dir / "markdown.md"
    stored_images = _write_mineru_images(run_dir, parsed.images)
    result_payload = _mineru_result_without_embedded_images(parsed.raw_result)
    result_payload["_evolab_local"] = {
        "images_requested": images_requested,
        "stored_image_count": len(stored_images),
        "stored_image_paths": stored_images,
    }
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    content_list_path.write_text(
        json.dumps(parsed.content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(parsed.md_content, encoding="utf-8")
    return {
        "result_path": result_path.as_posix(),
        "content_list_path": content_list_path.as_posix(),
        "markdown_path": markdown_path.as_posix(),
    }


def _write_mineru_images(run_dir: Path, images: dict[str, str]) -> list[str]:
    if not images:
        return []
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    stored: list[str] = []
    for raw_name, data_url in images.items():
        filename = Path(raw_name).name
        if not filename:
            continue
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError:
            continue
        output_path = image_dir / filename
        output_path.write_bytes(payload)
        stored.append(f"images/{filename}")
    return stored


def _mineru_result_without_embedded_images(raw_result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw_result)
    results = raw_result.get("results")
    if not isinstance(results, dict):
        return payload
    sanitized_results: dict[str, Any] = {}
    for key, value in results.items():
        if not isinstance(value, dict):
            sanitized_results[str(key)] = value
            continue
        sanitized = dict(value)
        images = sanitized.pop("images", None)
        if isinstance(images, dict):
            sanitized["image_names"] = sorted(str(name) for name in images)
        sanitized_results[str(key)] = sanitized
    payload["results"] = sanitized_results
    return payload


def mineru_content_list_to_blocks(
    paper_id: str,
    content_list: list[dict[str, Any]],
) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    per_page_index: dict[int, int] = {}
    for item_index, item in enumerate(content_list):
        text = mineru_item_text(item)
        if not text:
            continue
        page_id = _page_id(item)
        block_index = per_page_index.get(page_id, 0)
        per_page_index[page_id] = block_index + 1
        block_type = str(item.get("type") or "text")
        blocks.append(
            DocumentBlock(
                paper_id=paper_id,
                block_id=f"mineru_p{page_id}_b{block_index}",
                page_id=page_id,
                block_index=item_index,
                block_type=block_type,
                text=text,
                bbox=_bbox(item),
                source="mineru",
            )
        )
    return blocks


def mineru_item_text(item: dict[str, Any]) -> str:
    block_type = str(item.get("type") or "")
    parts: list[str] = []
    if isinstance(item.get("text"), str):
        parts.append(item["text"])
    if block_type == "table":
        parts.extend(_string_list(item.get("table_caption")))
        if isinstance(item.get("table_body"), str):
            parts.append(_strip_html(item["table_body"]))
        parts.extend(_string_list(item.get("table_footnote")))
    elif block_type in {"image", "chart"}:
        parts.extend(_string_list(item.get("image_caption")))
        parts.extend(_string_list(item.get("image_footnote")))
        if isinstance(item.get("sub_type"), str):
            parts.append(f"image_sub_type: {item['sub_type']}")
    return " ".join(" ".join(parts).split())


def _page_id(item: dict[str, Any]) -> int:
    value = item.get("page_idx")
    if isinstance(value, int):
        return value + 1
    return 1


def _bbox(item: dict[str, Any]) -> list[float]:
    value = item.get("bbox")
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for raw in value:
        if isinstance(raw, int | float):
            output.append(float(raw))
    return output


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)
