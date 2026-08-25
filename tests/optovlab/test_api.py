from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from evolab_local.optovlab.api import router
from evolab_local.optovlab.config import DatasetConfig, OptoVLabConfig, RuntimeConfig
from evolab_local.optovlab.service import OptoVLabService


def _app(tmp_path: Path) -> FastAPI:
    dataset_path = tmp_path / "oled.json.gz"
    records = [
        {
            "id": "D1",
            "paper_id": "10.1000%2Fone",
            "doi": "10.1000/one",
            "title": "Green TADF device",
            "architecture": "ITO/mCP:4CzIPN/PPF/Al",
            "final_emitter": "4CzIPN",
            "emission_color": "green",
            "eqe_max": 26.5,
        },
        {
            "id": "D2",
            "paper_id": "10.1000%2Ftwo",
            "doi": "10.1000/two",
            "title": "Blue OLED",
            "architecture": "ITO/HTL/DMAC-DPS/ETL/Al",
            "final_emitter": "DMAC-DPS",
            "emission_color": "blue",
            "eqe_max": 22.5,
        },
    ]
    with gzip.open(dataset_path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)
    runtime = tmp_path / "runtime"
    config = OptoVLabConfig(
        runtime=RuntimeConfig(
            root=runtime,
            sqlite_path=runtime / "optovlab.sqlite",
            artifact_dir=runtime / "artifacts",
        ),
        datasets=DatasetConfig(oled_devices=dataset_path),
    )
    service = OptoVLabService(config, Mock())
    service.init_runtime()
    app = FastAPI()
    app.state.optovlab_service = service
    app.include_router(router)
    return app


def test_status_session_workspace_and_rag_endpoints(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        status = client.get("/api/optovlab/status")
        assert status.status_code == 200
        assert status.json()["dataset"]["devices"] == 2

        created = client.post(
            "/api/optovlab/sessions",
            json={"agent_type": "experimental_design", "title": "TADF study"},
        )
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        workspace = client.get(f"/api/optovlab/sessions/{session_id}/workspace")
        assert workspace.status_code == 200
        assert workspace.json()["messages"][0]["role"] == "assistant"

        retrieval = client.post(
            "/api/optovlab/rag/search",
            json={"query": "4CzIPN PPF green", "top_k": 1, "filters": {}},
        )
        assert retrieval.status_code == 200
        assert retrieval.json()["hits"][0]["device_id"] == "D1"


def test_analysis_endpoint_writes_auditable_artifacts(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        created = client.post(
            "/api/optovlab/sessions",
            json={"agent_type": "data_mining", "title": "Analysis"},
        ).json()
        response = client.post(
            f"/api/optovlab/sessions/{created['session_id']}/analysis",
            json={"skill_id": "univariate_distribution", "metric": "eqe_max"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["statistics"]["count"] == 2
        assert {item["mime_type"] for item in payload["artifacts"]} == {
            "image/png",
            "text/csv",
        }


def test_delete_session_removes_conversation_and_analysis_files(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        keep = client.post(
            "/api/optovlab/sessions",
            json={"agent_type": "data_mining", "title": "Keep this session"},
        ).json()
        deleted = client.post(
            "/api/optovlab/sessions",
            json={"agent_type": "data_mining", "title": "Delete this session"},
        ).json()
        analysis = client.post(
            f"/api/optovlab/sessions/{deleted['session_id']}/analysis",
            json={"skill_id": "univariate_distribution", "metric": "eqe_max"},
        )
        assert analysis.status_code == 200
        artifact_dir = app.state.optovlab_service.config.runtime.artifact_dir / deleted["session_id"]
        assert artifact_dir.is_dir()

        response = client.delete(f"/api/optovlab/sessions/{deleted['session_id']}")

        assert response.status_code == 200
        assert response.json() == {
            "session_id": deleted["session_id"],
            "agent_type": "data_mining",
            "deleted": True,
            "preserved_linked_resources": 0,
        }
        assert not artifact_dir.exists()
        assert client.get(f"/api/optovlab/sessions/{deleted['session_id']}/workspace").status_code == 404
        assert client.get(f"/api/optovlab/sessions/{keep['session_id']}/workspace").status_code == 200
        assert client.delete(f"/api/optovlab/sessions/{deleted['session_id']}").status_code == 404


def test_chat_can_invoke_full_catalog_analysis_skill(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        session = client.post(
            "/api/optovlab/sessions",
            json={"agent_type": "data_mining", "title": "Catalog analysis"},
        ).json()

        response = client.post(
            f"/api/optovlab/sessions/{session['session_id']}/messages",
            json={"content": "分析一下整个OLED数据库并总结器件组成"},
        )

        assert response.status_code == 200
        payload = response.json()
        assistant = payload["assistant_message"]
        assert assistant["message_type"] == "analysis"
        analysis = assistant["metadata"]["analysis"]
        assert analysis["skill_id"] == "dataset_summary"
        assert analysis["scope"] == "catalog"
        assert analysis["statistics"]["devices"] == 2
        assert len(analysis["artifacts"]) == 2
        assert {artifact["mime_type"] for artifact in payload["artifacts"]} == {
            "image/png",
            "text/csv",
        }
        workspace = client.get(
            f"/api/optovlab/sessions/{session['session_id']}/workspace"
        ).json()
        assert workspace["tool_events"][0]["tool_name"] == "dataset_summary"
