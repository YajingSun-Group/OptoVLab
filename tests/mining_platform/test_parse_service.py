from __future__ import annotations

import json

from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.parse_service import ParseService


def test_parse_service_writes_blocks_and_updates_parse_status(
    mining_config_path,
    text_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = text_pdf_factory(
        mining_config_path,
        text="OLED device D1 showed a maximum EQE of 18.2%. The EML used emitter A.",
    )
    manifest_writer(
        mining_config_path,
        [
            {
                "doi": "10.1000/example",
                "status": "completed",
                "journal": "Journal",
                "publisher": "Publisher",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    PaperService(config).ingest_from_pdf_downloader(domain="oled")

    service = ParseService(config)
    result = service.parse_paper("10.1000/example")
    paper = PaperService(config).get_paper("10.1000%2Fexample")
    blocks = service.list_blocks("10.1000%2Fexample")

    assert result is not None
    assert result.paper_id == "10.1000%2Fexample"
    assert result.page_count == 1
    assert result.block_count >= 1
    assert paper is not None
    assert paper.parse_status == "parsed"
    assert blocks
    assert "maximum EQE" in blocks[0].text
    assert blocks[0].page_id == 1
    assert len(blocks[0].bbox) == 4

    document_payload = json.loads(
        config.paths.parsed_dir.joinpath(result.paper_id, "document.json").read_text()
    )
    block_lines = (
        config.paths.parsed_dir.joinpath(result.paper_id, "blocks.jsonl").read_text().splitlines()
    )
    assert document_payload["block_count"] == result.block_count
    assert len(block_lines) == result.block_count


def test_parse_all_papers_returns_failed_result_for_invalid_pdf(
    mining_config_path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = fake_pdf_factory(mining_config_path)
    manifest_writer(
        mining_config_path,
        [
            {
                "doi": "10.1000/example",
                "status": "completed",
                "journal": "Journal",
                "publisher": "Publisher",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    PaperService(config).ingest_from_pdf_downloader(domain="oled")

    results = ParseService(config).parse_all_papers()
    paper = PaperService(config).get_paper("10.1000%2Fexample")

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_message
    assert paper is not None
    assert paper.parse_status == "failed"
