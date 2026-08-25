from __future__ import annotations

from pathlib import Path

from evolab_local.optovlab.demo_content import (
    DEVICE_MODELING_DEMO_SEED,
    seed_demo_content,
)
from evolab_local.optovlab.repository import OptoVLabRepository


def test_repository_persists_sessions_messages_events_and_resources(tmp_path: Path) -> None:
    repository = OptoVLabRepository(tmp_path / "optovlab.sqlite")
    repository.init_runtime()

    session = repository.create_session("data_mining", "OLED paper set")
    message = repository.add_message(session.session_id, "user", "Mine these papers")
    event = repository.add_tool_event(
        session.session_id,
        "mineru",
        "running",
        "Parsing PDF",
        payload={"paper_id": "paper-1"},
    )
    resource = repository.link_resource(
        session.session_id,
        "data_mining_session",
        "mining-1",
        "paper.pdf",
    )

    assert repository.get_session(session.session_id) == session
    assert repository.list_messages(session.session_id)[0] == message
    assert repository.list_tool_events(session.session_id)[0] == event
    assert repository.list_resources(session.session_id)[0].resource_id == resource.resource_id

    updated_event = repository.update_tool_event(
        event.event_id,
        status="completed",
        title="PDF parsed",
    )
    assert updated_event.status == "completed"
    assert repository.list_tool_events(session.session_id)[0].title == "PDF parsed"


def test_repository_updates_title_without_changing_identity(tmp_path: Path) -> None:
    repository = OptoVLabRepository(tmp_path / "optovlab.sqlite")
    repository.init_runtime()
    session = repository.create_session("device_modeling")

    updated = repository.update_session(session.session_id, title="Campaign GAT")

    assert updated.session_id == session.session_id
    assert updated.title == "Campaign GAT"


def test_repository_reconciles_tool_calls_interrupted_by_restart(tmp_path: Path) -> None:
    repository = OptoVLabRepository(tmp_path / "optovlab.sqlite")
    repository.init_runtime()
    session = repository.create_session("experimental_design")
    repository.add_tool_event(session.session_id, "agent", "running", "Thinking")

    assert repository.reconcile_interrupted_events() == 1

    event = repository.list_tool_events(session.session_id)[0]
    assert event.status == "failed"
    assert "restart" in (event.detail or "")


def test_repository_deletes_session_children_with_foreign_key_cascades(tmp_path: Path) -> None:
    repository = OptoVLabRepository(tmp_path / "optovlab.sqlite")
    repository.init_runtime()
    session = repository.create_session("data_mining", "Disposable workspace")
    repository.add_message(session.session_id, "user", "Temporary message")
    repository.add_tool_event(session.session_id, "analysis", "completed", "Temporary event")
    repository.add_artifact(
        session.session_id,
        "chart",
        "Temporary chart",
        "chart.png",
        "image/png",
        "/chart.png",
    )
    repository.link_resource(
        session.session_id,
        "data_mining_session",
        "mining-temporary",
        "paper.pdf",
    )

    assert repository.delete_session(session.session_id) is True
    assert repository.get_session(session.session_id) is None
    assert repository.list_messages(session.session_id) == []
    assert repository.list_tool_events(session.session_id) == []
    assert repository.list_artifacts(session.session_id) == []
    assert repository.list_resources(session.session_id) == []
    assert repository.delete_session(session.session_id) is False


def test_device_modeling_demo_content_is_idempotent_without_touching_session_time(
    tmp_path: Path,
) -> None:
    repository = OptoVLabRepository(tmp_path / "optovlab.sqlite")
    repository.init_runtime()
    session = repository.create_session("device_modeling", "Device Modeling demo")
    original_updated_at = session.updated_at

    assert seed_demo_content(repository) == 4
    assert seed_demo_content(repository) == 0

    messages = repository.list_messages(session.session_id)
    assert len(messages) == 4
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]
    assert all(message.metadata["demo_seed"] == DEVICE_MODELING_DEMO_SEED for message in messages)
    assert repository.get_session(session.session_id).updated_at == original_updated_at
    events = repository.list_tool_events(session.session_id)
    assert len([event for event in events if event.payload.get("demo_seed")]) == 3
