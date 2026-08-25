from __future__ import annotations

import sqlite3

from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.ingest.pdf_downloader_adapter import (
    papers_from_pdf_downloader_manifest,
)


def test_adapter_imports_only_completed_existing_pdfs(
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
            },
            {
                "doi": "10.1000/pending",
                "status": "pending",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            },
            {
                "doi": "10.1000/missing",
                "status": "completed",
                "pdf_path": "runtime/pdf_downloader/papers/missing.pdf",
            },
        ],
    )

    papers, skipped_count = papers_from_pdf_downloader_manifest(
        config.pdf_downloader.manifest_path,
        config.project_root,
        sqlite_path=config.pdf_downloader.sqlite_path,
        domain="oled",
    )

    assert len(papers) == 1
    assert skipped_count == 2
    assert papers[0].paper_id == "10.1000%2Fexample"
    assert papers[0].domain == "oled"
    assert papers[0].journal == "Journal"
    assert papers[0].pdf_sha256
    assert papers[0].pdf_size_bytes == pdf_path.stat().st_size


def test_adapter_uses_pdf_downloader_sqlite_metadata_fallback(
    mining_config_path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = fake_pdf_factory(mining_config_path)
    config.pdf_downloader.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.pdf_downloader.sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE download_tasks (
              doi TEXT PRIMARY KEY,
              status TEXT,
              publisher TEXT,
              journal TEXT,
              pdf_path TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO download_tasks (doi, status, publisher, journal, pdf_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "10.1000/sqlite",
                "completed",
                "SQLite Publisher",
                "SQLite Journal",
                pdf_path.relative_to(config.project_root).as_posix(),
            ),
        )
    manifest_writer(mining_config_path, [{"doi": "10.1000/sqlite"}])

    papers, skipped_count = papers_from_pdf_downloader_manifest(
        config.pdf_downloader.manifest_path,
        config.project_root,
        sqlite_path=config.pdf_downloader.sqlite_path,
    )

    assert skipped_count == 0
    assert len(papers) == 1
    assert papers[0].journal == "SQLite Journal"
    assert papers[0].publisher == "SQLite Publisher"
