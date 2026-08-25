from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.core.config import load_config


def test_parse_api_parses_and_serves_blocks(
    mining_config_path,
    text_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = text_pdf_factory(
        mining_config_path,
        text="Device D1 used ITO / HTL / EML / ETL / Al and reached an EQE of 20%.",
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

    parse_response = client.post("/api/papers/10.1000%2Fexample/parse")
    blocks_response = client.get("/api/papers/10.1000%2Fexample/blocks")

    assert parse_response.status_code == 200
    assert parse_response.json()["block_count"] >= 1
    assert blocks_response.status_code == 200
    blocks = blocks_response.json()
    assert blocks[0]["paper_id"] == "10.1000%2Fexample"
    assert blocks[0]["page_id"] == 1
    assert "reached an EQE" in blocks[0]["text"]

    block_response = client.get(f"/api/papers/10.1000%2Fexample/blocks/{blocks[0]['block_id']}")
    assert block_response.status_code == 200
    assert block_response.json()["block_id"] == blocks[0]["block_id"]
