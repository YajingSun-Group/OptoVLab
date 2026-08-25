from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import PaperRepository


def test_paper_api_ingests_lists_details_and_serves_pdf(
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
    client = TestClient(create_app(config=config))

    ingest_response = client.post("/api/papers/ingest-from-pdf-downloader?domain=oled")
    list_response = client.get("/api/papers")
    detail_response = client.get("/api/papers/10.1000%2Fexample")
    pdf_response = client.get("/api/papers/10.1000%2Fexample/pdf")

    assert ingest_response.status_code == 200
    assert ingest_response.json()["imported_count"] == 1
    assert list_response.status_code == 200
    assert list_response.json()[0]["paper_id"] == "10.1000%2Fexample"
    assert detail_response.status_code == 200
    assert detail_response.json()["domain"] == "oled"
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"
    assert "inline" in pdf_response.headers["content-disposition"]
    assert pdf_response.headers["content-encoding"] == "identity"
    assert pdf_response.headers["cache-control"] == "private, max-age=3600"
    assert pdf_response.content.startswith(b"%PDF")

    range_response = client.get(
        "/api/papers/10.1000%2Fexample/pdf",
        headers={"Range": "bytes=0-15"},
    )
    assert range_response.status_code == 206
    assert range_response.headers["accept-ranges"] == "bytes"
    assert range_response.headers["content-encoding"] == "identity"
    assert len(range_response.content) == 16


def test_reingest_preserves_existing_pipeline_status_for_same_pdf(
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
    client = TestClient(create_app(config=config))
    assert client.post("/api/papers/ingest-from-pdf-downloader?domain=oled").status_code == 200

    repository = PaperRepository(Database(config.paths.sqlite_path))
    repository.set_parse_status("10.1000%2Fexample", "completed")
    repository.set_mining_status("10.1000%2Fexample", "completed")
    repository.set_review_status("10.1000%2Fexample", "needs_review")

    assert client.post("/api/papers/ingest-from-pdf-downloader?domain=oled").status_code == 200
    detail_response = client.get("/api/papers/10.1000%2Fexample")

    assert detail_response.status_code == 200
    assert detail_response.json()["parse_status"] == "completed"
    assert detail_response.json()["mining_status"] == "completed"
    assert detail_response.json()["review_status"] == "needs_review"


def test_paper_api_excludes_and_restores_review_article_with_audit_events(
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
                "doi": "10.1000/review",
                "status": "completed",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    client = TestClient(create_app(config=config))
    assert client.post("/api/papers/ingest-from-pdf-downloader?domain=oled").status_code == 200

    excluded_response = client.post(
        "/api/papers/10.1000%2Freview/exclude-review",
        json={"actor": "researcher", "message": "This is a review article"},
    )

    assert excluded_response.status_code == 200
    assert excluded_response.json()["review_status"] == "excluded"
    assert excluded_response.json()["review_reason"] == "review_article"
    events = client.get("/api/papers/10.1000%2Freview/review-events").json()
    assert events[0]["event_type"] == "paper_excluded"
    assert events[0]["actor"] == "researcher"
    assert events[0]["before_json"]
    assert events[0]["after_json"]

    restored_response = client.post(
        "/api/papers/10.1000%2Freview/restore-review",
        json={"actor": "researcher", "message": "Restore for extraction"},
    )

    assert restored_response.status_code == 200
    assert restored_response.json()["review_status"] == "needs_review"
    assert restored_response.json()["review_reason"] == "restored_from_review_exclusion"
    events = client.get("/api/papers/10.1000%2Freview/review-events").json()
    assert events[0]["event_type"] == "paper_exclusion_restored"


def test_paper_api_does_not_exclude_confirmed_paper(
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
                "doi": "10.1000/confirmed",
                "status": "completed",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    client = TestClient(create_app(config=config))
    assert client.post("/api/papers/ingest-from-pdf-downloader?domain=oled").status_code == 200
    PaperRepository(Database(config.paths.sqlite_path)).set_review_status(
        "10.1000%2Fconfirmed",
        "confirmed",
    )

    response = client.post("/api/papers/10.1000%2Fconfirmed/exclude-review")

    assert response.status_code == 409
