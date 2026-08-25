from __future__ import annotations

from evolab_local.mining_platform.ingest.pdf_downloader_adapter import paper_id_from_doi
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.paper import Paper


def test_paper_id_uses_url_encoded_doi() -> None:
    assert paper_id_from_doi("10.1002/adma.202405163") == "10.1002%2Fadma.202405163"


def test_paper_service_normalizes_doi_and_encoded_paper_ids() -> None:
    expected = "10.1002%2Fadma.202405163"
    assert PaperService.normalize_paper_id("10.1002/adma.202405163") == expected
    assert PaperService.normalize_paper_id("10.1002%2Fadma.202405163") == expected
    assert PaperService.normalize_paper_id("10.1002%2fadma.202405163") == expected


def test_paper_schema_defaults() -> None:
    paper = Paper(
        paper_id="10.1002%2Fadma.202405163",
        doi="10.1002/adma.202405163",
        pdf_path="runtime/pdf_downloader/papers/10.1002%2Fadma.202405163.pdf",
        pdf_sha256="abc",
        pdf_size_bytes=123,
    )

    assert paper.source == "pdf_downloader"
    assert paper.download_status == "completed"
    assert paper.parse_status == "pending"
    assert paper.mining_status == "pending"
    assert paper.review_status == "not_started"
    assert paper.domain == "unknown"
