from __future__ import annotations

from pathlib import Path

import pymupdf

from evolab_local.mining_platform.schemas.document import DocumentBlock


def parse_pdf_with_pymupdf(paper_id: str, pdf_path: Path) -> tuple[int, list[DocumentBlock]]:
    blocks: list[DocumentBlock] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            block_index = 0
            for raw_block in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, text, _block_no, block_type = raw_block[:7]
                if int(block_type) != 0:
                    continue
                normalized_text = " ".join(str(text).split())
                if not normalized_text:
                    continue
                blocks.append(
                    DocumentBlock(
                        paper_id=paper_id,
                        block_id=f"p{page_number}_b{block_index}",
                        page_id=page_number,
                        block_index=block_index,
                        block_type="text",
                        text=normalized_text,
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        source="pymupdf",
                    )
                )
                block_index += 1
        return document.page_count, blocks
