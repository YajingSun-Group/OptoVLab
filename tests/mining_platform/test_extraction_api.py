from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.core.config import load_config


def test_extract_oled_api_generates_raw_record_and_accepts_it(
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

    extract_response = client.post("/api/papers/10.1000%2Fexample/extract-oled")
    assert extract_response.status_code == 200
    result = extract_response.json()
    assert result["run"]["status"] == "completed"
    assert result["run"]["raw_record_count"] == 1
    raw_record = result["raw_records"][0]
    assert raw_record["eqe_max"] == "18.2%"
    assert raw_record["review_status"] == "pending"
    assert raw_record["field_evidence"]["eqe_max"]["block_ids"]

    list_response = client.get("/api/papers/10.1000%2Fexample/raw-device-records")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    accept_response = client.post(
        f"/api/raw-device-records/{raw_record['raw_record_id']}/accept",
        json={"message": "Accepted test candidate"},
    )
    assert accept_response.status_code == 200
    reviewed = accept_response.json()
    assert reviewed["eqe_max"] == "18.2%"
    assert reviewed["device_label"] == "D1"

    records_response = client.get("/api/papers/10.1000%2Fexample/device-records")
    assert records_response.status_code == 200
    assert len(records_response.json()) == 1


def test_reject_raw_device_record_api(
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
    raw_record = client.post("/api/papers/10.1000%2Fexample/extract-oled").json()["raw_records"][0]

    reject_response = client.post(
        f"/api/raw-device-records/{raw_record['raw_record_id']}/reject",
        json={"message": "Rejected test candidate"},
    )

    assert reject_response.status_code == 200
    assert reject_response.json()["review_status"] == "rejected"
