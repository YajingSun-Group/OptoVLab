from __future__ import annotations

from fastapi.testclient import TestClient

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.core.config import load_config


def test_oled_session_plan_approval_and_pdf_upload(
    mining_config_path,
    text_pdf_factory,
) -> None:
    config = load_config(mining_config_path)
    source_pdf = text_pdf_factory(
        mining_config_path,
        name="agent-upload.pdf",
        text="Device D1 used ITO / TAPC / EML / TmPyPB / Liq / Al and reached 18.2% EQE.",
    )
    with TestClient(create_app(config=config)) as client:
        templates = client.get("/api/data-mining-agent/templates")
        assert templates.status_code == 200
        assert templates.json()[0]["template_id"] == "oled_device_v1"

        created = client.post(
            "/api/data-mining-agent/sessions",
            json={
                "mode": "preset",
                "template_id": "oled_device_v1",
                "title": "OLED test",
            },
        )
        assert created.status_code == 201
        session = created.json()
        assert session["plan_status"] == "awaiting_user_approval"
        assert session["plan"]["tools"]

        session_id = session["session_id"]
        approved = client.post(
            f"/api/data-mining-agent/sessions/{session_id}/plan/approve",
            json={"actor": "test_user"},
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "awaiting_pdf"

        with source_pdf.open("rb") as stream:
            uploaded = client.post(
                f"/api/data-mining-agent/sessions/{session_id}/pdf",
                files={"pdf": ("paper.pdf", stream, "application/pdf")},
            )
        assert uploaded.status_code == 200
        upload_payload = uploaded.json()
        assert upload_payload["page_count"] == 1
        assert upload_payload["paper_id"].startswith("upload-")
        assert upload_payload["session"]["status"] == "ready_to_run"

        workspace = client.get(
            f"/api/data-mining-agent/sessions/{session_id}/workspace"
        )
        assert workspace.status_code == 200
        workspace_payload = workspace.json()
        assert workspace_payload["paper"]["source"] == "data_mining_agent_upload"
        assert len(workspace_payload["messages"]) == 3
        assert workspace_payload["candidate_bundle"] is None


def test_custom_session_can_collect_requirements_without_calling_llm(
    mining_config_path,
) -> None:
    config = load_config(mining_config_path)
    with TestClient(create_app(config=config)) as client:
        created = client.post(
            "/api/data-mining-agent/sessions",
            json={"mode": "custom", "template_id": None, "title": "Battery extraction"},
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        turn = client.post(
            f"/api/data-mining-agent/sessions/{session_id}/messages",
            json={
                "content": "Extract cathode composition and cycling capacity with page evidence.",
                "auto_respond": False,
            },
        )
        assert turn.status_code == 200
        assert turn.json()["assistant_message"] is None

        workspace = client.get(
            f"/api/data-mining-agent/sessions/{session_id}/workspace"
        ).json()
        assert [message["role"] for message in workspace["messages"]] == [
            "assistant",
            "user",
        ]


def test_upload_requires_approved_plan(mining_config_path) -> None:
    config = load_config(mining_config_path)
    with TestClient(create_app(config=config)) as client:
        created = client.post(
            "/api/data-mining-agent/sessions",
            json={"mode": "preset", "template_id": "oled_device_v1"},
        )
        session_id = created.json()["session_id"]
        response = client.post(
            f"/api/data-mining-agent/sessions/{session_id}/pdf",
            files={"pdf": ("paper.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
        )
        assert response.status_code == 422
        assert "Approve the mining plan" in response.json()["detail"]
