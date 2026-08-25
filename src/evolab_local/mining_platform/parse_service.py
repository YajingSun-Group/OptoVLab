from __future__ import annotations

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.parsers.parsed_writer import write_parsed_document
from evolab_local.mining_platform.parsers.pymupdf_parser import parse_pdf_with_pymupdf
from evolab_local.mining_platform.schemas.document import DocumentBlock, ParseResult
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    DocumentBlockRepository,
    PaperRepository,
)


class ParseService:
    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.paper_service = PaperService(config)
        self.papers = PaperRepository(self.database)
        self.blocks = DocumentBlockRepository(self.database)

    def init_runtime(self) -> None:
        self.paper_service.init_runtime()

    def parse_paper(self, paper_id: str) -> ParseResult | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        paper = self.paper_service.get_paper(normalized_paper_id)
        if not paper:
            return None

        pdf_path = self.paper_service.get_pdf_path(normalized_paper_id)
        if not pdf_path:
            self.papers.set_parse_status(normalized_paper_id, "failed")
            raise FileNotFoundError(f"PDF not found for paper_id={normalized_paper_id}")

        self.papers.set_parse_status(normalized_paper_id, "parsing")
        try:
            page_count, blocks = parse_pdf_with_pymupdf(normalized_paper_id, pdf_path)
            stored_blocks = self.blocks.replace_for_paper(normalized_paper_id, blocks)
            result = write_parsed_document(
                self.config.paths.parsed_dir,
                normalized_paper_id,
                "pymupdf",
                page_count,
                stored_blocks,
            )
            self.papers.set_parse_status(normalized_paper_id, "parsed")
            return result
        except Exception:
            self.papers.set_parse_status(normalized_paper_id, "failed")
            raise

    def parse_all_papers(self) -> list[ParseResult]:
        self.init_runtime()
        results: list[ParseResult] = []
        for paper in self.paper_service.list_papers():
            try:
                result = self.parse_paper(paper.paper_id)
            except Exception as exc:
                results.append(
                    ParseResult(
                        paper_id=paper.paper_id,
                        parser="pymupdf",
                        page_count=0,
                        block_count=0,
                        document_path=(
                            self.config.paths.parsed_dir / paper.paper_id / "document.json"
                        ).as_posix(),
                        blocks_path=(
                            self.config.paths.parsed_dir / paper.paper_id / "blocks.jsonl"
                        ).as_posix(),
                        status="failed",
                        error_message=str(exc),
                    )
                )
                continue
            if result:
                results.append(result)
        return results

    def list_blocks(self, paper_id: str) -> list[DocumentBlock]:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.blocks.list_by_paper(normalized_paper_id)

    def get_block(self, paper_id: str, block_id: str) -> DocumentBlock | None:
        self.init_runtime()
        normalized_paper_id = self.paper_service.normalize_paper_id(paper_id)
        return self.blocks.get(normalized_paper_id, block_id)
