from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.candidate_review_service import CandidateReviewService
from evolab_local.mining_platform.core.config import load_config


def test_candidate_review_api_updates_fields_and_confirms_paper(
    mining_config_path,
    text_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = text_pdf_factory(
        mining_config_path,
        text=(
            "Device D1 used ITO / HATCN / NPB / EML / TPBi / LiF / Al. "
            "The maximum EQE of device D1 was 18.2%, current efficiency was "
            "45 cd A-1, and turn-on voltage was 3.1 V."
        ),
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
    client = TestClient(create_app(config=config))
    client.post("/api/papers/ingest-from-pdf-downloader?domain=oled")
    client.post("/api/papers/10.1000%2Fexample/extract-oled")

    seed_result = CandidateReviewService(config).seed_candidate_fields_from_raw("10.1000%2Fexample")
    assert seed_result is not None
    assert seed_result.field_count >= 3
    assert seed_result.evidence_count == seed_result.field_count

    fields_response = client.get("/api/papers/10.1000%2Fexample/candidate-fields")
    assert fields_response.status_code == 200
    fields = fields_response.json()
    eqe_field = next(field for field in fields if field["field_name"] == "eqe_max")
    assert eqe_field["mined_value"] == "18.2%"
    assert eqe_field["evidence_anchor_id"]

    anchors_response = client.get("/api/papers/10.1000%2Fexample/evidence-anchors")
    assert anchors_response.status_code == 200
    assert len(anchors_response.json()) == seed_result.evidence_count

    update_response = client.put(
        f"/api/candidate-fields/{eqe_field['candidate_field_id']}",
        json={
            "reviewed_value": "19.0%",
            "field_status": "edited",
            "message": "Corrected EQE in test",
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["reviewed_value"] == "19.0%"
    assert update_response.json()["field_status"] == "edited"

    confirm_response = client.post(
        "/api/papers/10.1000%2Fexample/confirm-review",
        json={"message": "Confirmed test paper"},
    )
    assert confirm_response.status_code == 200
    confirmed = confirm_response.json()
    assert confirmed["final_count"] == 1
    assert confirmed["final_devices"][0]["eqe_max"] == "19.0%"

    final_response = client.get("/api/papers/10.1000%2Fexample/final-devices")
    assert final_response.status_code == 200
    assert len(final_response.json()) == 1


def test_candidate_review_api_rejects_candidate_field(
    mining_config_path,
    text_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = text_pdf_factory(
        mining_config_path,
        text="Device D2 showed a maximum EQE of 12.5% and turn-on voltage of 3.8 V.",
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
    client = TestClient(create_app(config=config))
    client.post("/api/papers/ingest-from-pdf-downloader?domain=oled")
    client.post("/api/papers/10.1000%2Fexample/extract-oled")
    CandidateReviewService(config).seed_candidate_fields_from_raw("10.1000%2Fexample")

    fields = client.get("/api/papers/10.1000%2Fexample/candidate-fields").json()
    eqe_field = next(field for field in fields if field["field_name"] == "eqe_max")

    reject_response = client.post(
        f"/api/candidate-fields/{eqe_field['candidate_field_id']}/reject",
        json={"message": "Rejected field in test"},
    )

    assert reject_response.status_code == 200
    assert reject_response.json()["field_status"] == "rejected"
