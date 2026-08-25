from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.core.config import load_config


def test_device_record_review_flow(
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
    client.post("/api/papers/ingest-from-pdf-downloader?domain=oled")

    create_response = client.post(
        "/api/papers/10.1000%2Fexample/device-records",
        json={
            "device_label": "D1",
            "architecture": "ITO / HTL / EML / ETL / Al",
            "eml_emitter": "Emitter-A",
            "eqe_max": "18.2%",
            "evidence_text": "The maximum EQE of device D1 was 18.2%.",
            "evidence_page": 5,
            "message": "Initial manual entry",
        },
    )
    assert create_response.status_code == 200
    record = create_response.json()
    record_id = record["record_id"]
    assert record["review_status"] == "in_progress"
    assert record["eml_emitter"] == "Emitter-A"

    update_response = client.put(
        f"/api/device-records/{record_id}",
        json={
            "eqe_max": "18.3%",
            "message": "Corrected EQE after checking the table",
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["eqe_max"] == "18.3%"

    confirm_response = client.post(
        f"/api/device-records/{record_id}/confirm",
        json={"message": "Checked against PDF"},
    )
    assert confirm_response.status_code == 200
    assert confirm_response.json()["review_status"] == "confirmed"
    assert confirm_response.json()["confirmed_at"]

    list_response = client.get("/api/papers/10.1000%2Fexample/device-records")
    events_response = client.get("/api/papers/10.1000%2Fexample/review-events")

    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
    assert events_response.status_code == 200
    event_types = [event["event_type"] for event in events_response.json()]
    assert event_types == ["confirmed", "updated", "created"]
