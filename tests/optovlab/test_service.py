from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolab_local.optovlab.config import DatasetConfig, OptoVLabConfig, RuntimeConfig
from evolab_local.optovlab.schemas import SessionCreate
from evolab_local.optovlab.service import OptoVLabService


class PlanGatedMiningService:
    def __init__(self) -> None:
        self.approved = False

    def create_session(self, _payload):
        return self._session("awaiting_user_approval")

    def approve_plan(self, session_id: str, **_kwargs):
        assert session_id == "mining-1"
        self.approved = True
        return self._session("approved")

    def upload_pdf(self, session_id: str, *, filename: str, content: bytes):
        assert self.approved
        assert session_id == "mining-1"
        assert filename == "paper.pdf"
        assert content.startswith(b"%PDF")
        return SimpleNamespace(
            paper_id="paper-1",
            filename=filename,
            size_bytes=len(content),
            sha256="abc",
            page_count=1,
            session=self._session("approved"),
            model_dump=lambda **_kwargs: {"paper_id": "paper-1"},
        )

    @staticmethod
    def _session(plan_status: str):
        return SimpleNamespace(
            session_id="mining-1",
            plan_status=plan_status,
            model_dump=lambda **_kwargs: {
                "session_id": "mining-1",
                "plan_status": plan_status,
            },
        )


class WorkspaceMiningService:
    def __init__(self, job_status: str) -> None:
        self.job_status = job_status

    def get_workspace(self, _session_id: str):
        return SimpleNamespace(jobs=[SimpleNamespace(status=self.job_status)])


class CompletedMiningDemoService:
    def get_workspace(self, _session_id: str):
        return SimpleNamespace(
            jobs=[SimpleNamespace(status="completed")],
            messages=[
                SimpleNamespace(
                    message_type="file",
                    metadata={"filename": "demo.pdf", "page_count": 13},
                )
            ],
            result=SimpleNamespace(
                reviewed_result={
                    "devices": [{"device_id": f"D{index}"} for index in range(4)],
                    "materials": [{"material_id": f"M{index}"} for index in range(14)],
                    "evidence": [{"evidence_id": f"E{index}"} for index in range(18)],
                },
                raw_result={},
                review_status="in_review",
            ),
        )


def _service(
    tmp_path: Path,
    mining_service,
    records: list[dict] | None = None,
) -> OptoVLabService:
    dataset = tmp_path / "oled.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as handle:
        json.dump(records or [], handle)
    runtime = tmp_path / "runtime"
    service = OptoVLabService(
        OptoVLabConfig(
            runtime=RuntimeConfig(
                root=runtime,
                sqlite_path=runtime / "optovlab.sqlite",
                artifact_dir=runtime / "artifacts",
            ),
            datasets=DatasetConfig(oled_devices=dataset),
        ),
        mining_service,
    )
    service.init_runtime()
    return service

def test_pdf_upload_approves_preset_plan_before_calling_mining_service(tmp_path: Path) -> None:
    dataset = tmp_path / "oled.json.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as handle:
        json.dump([], handle)
    runtime = tmp_path / "runtime"
    mining = PlanGatedMiningService()
    service = OptoVLabService(
        OptoVLabConfig(
            runtime=RuntimeConfig(
                root=runtime,
                sqlite_path=runtime / "optovlab.sqlite",
                artifact_dir=runtime / "artifacts",
            ),
            datasets=DatasetConfig(oled_devices=dataset),
        ),
        mining,
    )
    service.init_runtime()
    session = service.create_session(SessionCreate(agent_type="data_mining"))

    result = service.upload_pdf(session.session_id, "paper.pdf", b"%PDF-1.4 test")

    assert mining.approved is True
    assert result["upload"]["paper_id"] == "paper-1"
    assert result["mining_session"]["plan_status"] == "approved"
    assert service.repository.list_resources(session.session_id)[0].resource_id == "mining-1"
    file_messages = [
        message
        for message in service.repository.list_messages(session.session_id)
        if message.message_type == "file"
    ]
    assert len(file_messages) == 1
    assert file_messages[0].role == "user"
    assert file_messages[0].metadata == {
        "resource_id": "mining-1",
        "paper_id": "paper-1",
        "filename": "paper.pdf",
        "mime_type": "application/pdf",
        "size_bytes": len(b"%PDF-1.4 test"),
        "page_count": 1,
        "sha256": "abc",
    }


def test_delete_session_cleans_artifacts_but_preserves_linked_mining_data(tmp_path: Path) -> None:
    service = _service(tmp_path, WorkspaceMiningService("completed"))
    session = service.create_session(SessionCreate(agent_type="data_mining"))
    service.repository.link_resource(
        session.session_id,
        "data_mining_session",
        "mining-complete",
        "paper.pdf",
    )
    artifact_dir = service.config.runtime.artifact_dir / session.session_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "chart.png").write_bytes(b"chart")

    result = service.delete_session(session.session_id)

    assert result["deleted"] is True
    assert result["preserved_linked_resources"] == 1
    assert service.repository.get_session(session.session_id) is None
    assert not artifact_dir.exists()


def test_delete_session_rejects_active_linked_mining_job(tmp_path: Path) -> None:
    service = _service(tmp_path, WorkspaceMiningService("running"))
    session = service.create_session(SessionCreate(agent_type="data_mining"))
    service.repository.link_resource(
        session.session_id,
        "data_mining_session",
        "mining-running",
        "paper.pdf",
    )

    with pytest.raises(ValueError, match="active mining job"):
        service.delete_session(session.session_id)

    assert service.repository.get_session(session.session_id) is not None


def test_data_mining_database_analysis_demo_is_real_idempotent_and_keeps_timestamp(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        WorkspaceMiningService("completed"),
        records=[
            {
                "id": "D1",
                "doi": "10.1000/demo",
                "final_emitter": "4CzIPN",
                "emission_color": "green",
                "eqe_max": 20.0,
            }
        ],
    )
    session = service.create_session(
        SessionCreate(agent_type="data_mining", title="Data Mining demo")
    )
    original_updated_at = service.repository.get_session(session.session_id).updated_at

    service._seed_data_mining_analysis_demo()
    service._seed_data_mining_analysis_demo()

    messages = service.repository.list_messages(session.session_id)
    analysis_messages = [message for message in messages if message.message_type == "analysis"]
    assert len(analysis_messages) == 1
    metadata = analysis_messages[0].metadata["analysis"]
    assert metadata["skill_id"] == "dataset_summary"
    assert metadata["scope"] == "catalog"
    assert len(metadata["artifacts"]) == 2
    assert service.repository.get_session(session.session_id).updated_at == original_updated_at


def test_data_mining_workflow_demo_upgrades_existing_messages_in_order(tmp_path: Path) -> None:
    service = _service(tmp_path, CompletedMiningDemoService())
    session = service.create_session(
        SessionCreate(agent_type="data_mining", title="Data Mining demo")
    )
    service.repository.link_resource(
        session.session_id,
        "data_mining_session",
        "mining-demo",
        "demo.pdf",
    )
    service.repository.add_message(
        session.session_id,
        "assistant",
        "No new jobs were started; 1 linked PDF job(s) are already active or complete.",
    )
    service.repository.add_message(
        session.session_id,
        "user",
        "Mine OLED device and material data from the uploaded paper.",
    )
    original_updated_at = service.repository.get_session(session.session_id).updated_at

    service._seed_data_mining_workflow_demo()
    service._seed_data_mining_workflow_demo()

    messages = service.repository.list_messages(session.session_id)
    mining_messages = [message for message in messages if message.message_type == "mining_result"]
    assert len(mining_messages) == 1
    mining = mining_messages[0].metadata["mining"]
    assert mining["device_count"] == 4
    assert mining["material_count"] == 14
    assert mining["evidence_count"] == 18
    prompt_index = next(
        index
        for index, message in enumerate(messages)
        if message.metadata.get("demo_seed_prompt") == "data_mining_workflow_v1"
    )
    result_index = messages.index(mining_messages[0])
    assert prompt_index < result_index
    assert service.repository.get_session(session.session_id).updated_at == original_updated_at
