from __future__ import annotations

import json

from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.library.paper_service import PaperService


def test_ingest_pdfs_upserts_sqlite_and_registry(
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
    service = PaperService(config)

    first = service.ingest_from_pdf_downloader(domain="oled")
    second = service.ingest_from_pdf_downloader(domain="oled")
    papers = service.list_papers()
    registry_lines = config.paths.paper_registry_path.read_text(encoding="utf-8").splitlines()

    assert first.imported_count == 1
    assert second.imported_count == 1
    assert len(papers) == 1
    assert papers[0].paper_id == "10.1000%2Fexample"
    assert papers[0].domain == "oled"
    assert len(registry_lines) == 1
    assert json.loads(registry_lines[0])["doi"] == "10.1000/example"
