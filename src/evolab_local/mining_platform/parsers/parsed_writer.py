from __future__ import annotations

import json
from pathlib import Path

from evolab_local.mining_platform.schemas.document import DocumentBlock, ParseResult


def write_parsed_document(
    parsed_root: Path,
    paper_id: str,
    parser_name: str,
    page_count: int,
    blocks: list[DocumentBlock],
) -> ParseResult:
    output_dir = parsed_root / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    document_path = output_dir / "document.json"
    blocks_path = output_dir / "blocks.jsonl"

    result = ParseResult(
        paper_id=paper_id,
        parser=parser_name,
        page_count=page_count,
        block_count=len(blocks),
        document_path=document_path.as_posix(),
        blocks_path=blocks_path.as_posix(),
        status="parsed",
    )
    document_payload = {
        **result.model_dump(mode="json"),
        "blocks": [block.model_dump(mode="json") for block in blocks],
    }
    document_path.write_text(
        json.dumps(document_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with blocks_path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(json.dumps(block.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return result
