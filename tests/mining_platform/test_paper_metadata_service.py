from __future__ import annotations

from pathlib import Path

from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.paper_metadata_service import (
    PaperMetadata,
    PaperMetadataEnrichmentService,
)
from evolab_local.mining_platform.schemas.paper import Paper
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import PaperRepository


class FakeMetadataClient:
    def __init__(self, metadata: PaperMetadata | None) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    def fetch(self, doi: str) -> PaperMetadata | None:
        self.calls.append(doi)
        return self.metadata


def test_metadata_enrichment_updates_missing_fields(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    database = Database(config.paths.sqlite_path)
    database.init_db()
    PaperRepository(database).upsert(
        Paper(
            paper_id="10.1000%2Fmeta",
            doi="10.1000/meta",
            pdf_path="runtime/mining_platform/inbox_pdfs/10.1000%2Fmeta.pdf",
            pdf_sha256="abc",
            pdf_size_bytes=12,
            domain="oled",
        )
    )
    client = FakeMetadataClient(
        PaperMetadata(
            source="fake",
            title="A metadata paper",
            journal="Journal of Tests",
            publisher="Test Publisher",
            year=2026,
        )
    )

    result = PaperMetadataEnrichmentService(config, metadata_client=client).enrich_paper(
        "10.1000/meta"
    )

    assert result.status == "updated"
    assert set(result.updated_fields) == {"journal", "publisher", "title", "year"}
    assert client.calls == ["10.1000/meta"]
    paper = PaperRepository(database).get("10.1000%2Fmeta")
    assert paper is not None
    assert paper.journal == "Journal of Tests"
    assert paper.year == 2026


def test_metadata_enrichment_skips_complete_paper_unless_forced(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    database = Database(config.paths.sqlite_path)
    database.init_db()
    PaperRepository(database).upsert(
        Paper(
            paper_id="10.1000%2Fcomplete",
            doi="10.1000/complete",
            title="Existing",
            journal="Existing Journal",
            publisher="Existing Publisher",
            year=2024,
            pdf_path="runtime/mining_platform/inbox_pdfs/10.1000%2Fcomplete.pdf",
            pdf_sha256="abc",
            pdf_size_bytes=12,
            domain="oled",
        )
    )
    client = FakeMetadataClient(
        PaperMetadata(
            source="fake",
            title="Fresh",
            journal="Fresh Journal",
            publisher="Fresh Publisher",
            year=2026,
        )
    )
    service = PaperMetadataEnrichmentService(config, metadata_client=client)

    skipped = service.enrich_paper("10.1000/complete")
    forced = service.enrich_paper("10.1000/complete", force=True)

    assert skipped.status == "skipped_complete"
    assert client.calls == ["10.1000/complete"]
    assert forced.status == "updated"
    paper = PaperRepository(database).get("10.1000%2Fcomplete")
    assert paper is not None
    assert paper.title == "Fresh"
