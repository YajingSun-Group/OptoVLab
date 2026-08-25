from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from evolab_local.optovlab.schemas import (
    Artifact,
    Message,
    ResourceLink,
    SessionSummary,
    ToolEvent,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _decode(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


class OptoVLabRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_runtime(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS optovlab_sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS optovlab_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES optovlab_sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_optovlab_messages_session
                    ON optovlab_messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS optovlab_tool_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    job_id TEXT,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    detail TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES optovlab_sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_optovlab_events_session
                    ON optovlab_tool_events(session_id, created_at);

                CREATE TABLE IF NOT EXISTS optovlab_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    relative_url TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES optovlab_sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_optovlab_artifacts_session
                    ON optovlab_artifacts(session_id, created_at);

                CREATE TABLE IF NOT EXISTS optovlab_resource_links (
                    link_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, resource_type, resource_id),
                    FOREIGN KEY(session_id) REFERENCES optovlab_sessions(session_id) ON DELETE CASCADE
                );
                """
            )

    def reconcile_interrupted_events(self) -> int:
        """Close local tool calls that could not survive an API process restart."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE optovlab_tool_events
                SET status = 'failed',
                    detail = CASE
                        WHEN detail IS NULL OR detail = ''
                            THEN 'Interrupted by an OptoVLab API restart.'
                        ELSE detail || '\nInterrupted by an OptoVLab API restart.'
                    END
                WHERE status = 'running'
                """
            )
        return max(cursor.rowcount, 0)

    def create_session(self, agent_type: str, title: str | None = None) -> SessionSummary:
        session_id = uuid.uuid4().hex
        now = _now()
        resolved_title = title or {
            "data_mining": "New mining workspace",
            "device_modeling": "New modeling workspace",
            "experimental_design": "New experiment workspace",
        }.get(agent_type, "New workspace")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO optovlab_sessions
                    (session_id, agent_type, title, status, created_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (session_id, agent_type, resolved_title, now, now),
            )
        return SessionSummary(
            session_id=session_id,
            agent_type=agent_type,
            title=resolved_title,
            status="active",
            created_at=now,
            updated_at=now,
        )

    def get_session(self, session_id: str) -> SessionSummary | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM optovlab_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._session(row) if row else None

    def list_sessions(self, agent_type: str | None = None, limit: int = 100) -> list[SessionSummary]:
        query = "SELECT * FROM optovlab_sessions"
        parameters: list[Any] = []
        if agent_type:
            query += " WHERE agent_type = ?"
            parameters.append(agent_type)
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._session(row) for row in rows]

    def update_session(self, session_id: str, **changes: str) -> SessionSummary:
        allowed = {"title", "status"}
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
        if assignments:
            assignments.append("updated_at = ?")
            values.append(_now())
            values.append(session_id)
            with self.connect() as connection:
                connection.execute(
                    f"UPDATE optovlab_sessions SET {', '.join(assignments)} WHERE session_id = ?",
                    values,
                )
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Unknown OptoVLab session: {session_id}")
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete one session and its local child records through SQLite cascades."""
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM optovlab_sessions WHERE session_id = ?",
                (session_id,),
            )
        return cursor.rowcount == 1

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        message_type: str = "text",
        metadata: dict[str, Any] | None = None,
        touch_session: bool = True,
    ) -> Message:
        message_id = uuid.uuid4().hex
        now = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO optovlab_messages
                    (message_id, session_id, role, message_type, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    message_type,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            if touch_session:
                connection.execute(
                    "UPDATE optovlab_sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
        return Message(
            message_id=message_id,
            session_id=session_id,
            role=role,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            created_at=now,
        )

    def list_messages(self, session_id: str, limit: int = 500) -> list[Message]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM optovlab_messages
                WHERE session_id = ? ORDER BY created_at ASC, rowid ASC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            Message(
                message_id=row["message_id"],
                session_id=row["session_id"],
                role=row["role"],
                message_type=row["message_type"],
                content=row["content"],
                metadata=_decode(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def update_message(
        self,
        message_id: str,
        *,
        role: str | None = None,
        content: str | None = None,
        message_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> Message:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM optovlab_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown OptoVLab message: {message_id}")
            resolved_role = role or row["role"]
            resolved_content = content if content is not None else row["content"]
            resolved_type = message_type or row["message_type"]
            resolved_metadata = metadata if metadata is not None else _decode(row["metadata_json"])
            resolved_created_at = created_at or row["created_at"]
            connection.execute(
                """
                UPDATE optovlab_messages
                SET role = ?, content = ?, message_type = ?, metadata_json = ?, created_at = ?
                WHERE message_id = ?
                """,
                (
                    resolved_role,
                    resolved_content,
                    resolved_type,
                    json.dumps(resolved_metadata, ensure_ascii=False),
                    resolved_created_at,
                    message_id,
                ),
            )
        return Message(
            message_id=message_id,
            session_id=row["session_id"],
            role=resolved_role,
            message_type=resolved_type,
            content=resolved_content,
            metadata=resolved_metadata,
            created_at=resolved_created_at,
        )

    def add_tool_event(
        self,
        session_id: str,
        tool_name: str,
        status: str,
        title: str,
        *,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> ToolEvent:
        event = ToolEvent(
            event_id=uuid.uuid4().hex,
            session_id=session_id,
            job_id=job_id,
            tool_name=tool_name,
            status=status,
            title=title,
            detail=detail,
            payload=payload or {},
            created_at=_now(),
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO optovlab_tool_events
                    (event_id, session_id, job_id, tool_name, status, title, detail,
                     payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.session_id,
                    event.job_id,
                    event.tool_name,
                    event.status,
                    event.title,
                    event.detail,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.created_at,
                ),
            )
        return event

    def list_tool_events(self, session_id: str, limit: int = 500) -> list[ToolEvent]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM optovlab_tool_events
                WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            ToolEvent(
                event_id=row["event_id"],
                session_id=row["session_id"],
                job_id=row["job_id"],
                tool_name=row["tool_name"],
                status=row["status"],
                title=row["title"],
                detail=row["detail"],
                payload=_decode(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def update_tool_event(
        self,
        event_id: str,
        *,
        status: str,
        title: str | None = None,
        detail: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ToolEvent:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM optovlab_tool_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown OptoVLab tool event: {event_id}")
            resolved_title = title or row["title"]
            resolved_detail = detail if detail is not None else row["detail"]
            resolved_payload = payload if payload is not None else _decode(row["payload_json"])
            connection.execute(
                """
                UPDATE optovlab_tool_events
                SET status = ?, title = ?, detail = ?, payload_json = ?
                WHERE event_id = ?
                """,
                (
                    status,
                    resolved_title,
                    resolved_detail,
                    json.dumps(resolved_payload, ensure_ascii=False),
                    event_id,
                ),
            )
        return ToolEvent(
            event_id=event_id,
            session_id=row["session_id"],
            job_id=row["job_id"],
            tool_name=row["tool_name"],
            status=status,
            title=resolved_title,
            detail=resolved_detail,
            payload=resolved_payload,
            created_at=row["created_at"],
        )

    def add_artifact(
        self,
        session_id: str,
        artifact_type: str,
        title: str,
        filename: str,
        mime_type: str,
        relative_url: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        artifact = Artifact(
            artifact_id=uuid.uuid4().hex,
            session_id=session_id,
            artifact_type=artifact_type,
            title=title,
            filename=filename,
            mime_type=mime_type,
            url=relative_url,
            metadata=metadata or {},
            created_at=_now(),
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO optovlab_artifacts
                    (artifact_id, session_id, artifact_type, title, filename, mime_type,
                     relative_url, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.session_id,
                    artifact.artifact_type,
                    artifact.title,
                    artifact.filename,
                    artifact.mime_type,
                    artifact.url,
                    json.dumps(artifact.metadata, ensure_ascii=False),
                    artifact.created_at,
                ),
            )
        return artifact

    def list_artifacts(self, session_id: str) -> list[Artifact]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM optovlab_artifacts
                WHERE session_id = ? ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [
            Artifact(
                artifact_id=row["artifact_id"],
                session_id=row["session_id"],
                artifact_type=row["artifact_type"],
                title=row["title"],
                filename=row["filename"],
                mime_type=row["mime_type"],
                url=row["relative_url"],
                metadata=_decode(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def link_resource(
        self,
        session_id: str,
        resource_type: str,
        resource_id: str,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResourceLink:
        link = ResourceLink(
            link_id=uuid.uuid4().hex,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            label=label,
            metadata=metadata or {},
            created_at=_now(),
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO optovlab_resource_links
                    (link_id, session_id, resource_type, resource_id, label,
                     metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, resource_type, resource_id) DO UPDATE SET
                    label = excluded.label,
                    metadata_json = excluded.metadata_json
                """,
                (
                    link.link_id,
                    link.session_id,
                    link.resource_type,
                    link.resource_id,
                    link.label,
                    json.dumps(link.metadata, ensure_ascii=False),
                    link.created_at,
                ),
            )
        return link

    def list_resources(self, session_id: str) -> list[ResourceLink]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM optovlab_resource_links
                WHERE session_id = ? ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ResourceLink(
                link_id=row["link_id"],
                session_id=row["session_id"],
                resource_type=row["resource_type"],
                resource_id=row["resource_id"],
                label=row["label"],
                metadata=_decode(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _session(row: sqlite3.Row) -> SessionSummary:
        return SessionSummary(
            session_id=row["session_id"],
            agent_type=row["agent_type"],
            title=row["title"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
