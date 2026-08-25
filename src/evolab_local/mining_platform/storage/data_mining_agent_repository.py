from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import uuid4

from evolab_local.mining_platform.schemas.data_mining_agent import (
    AgentEvent,
    AgentJob,
    AgentMessage,
    AgentResult,
    AgentSession,
)
from evolab_local.mining_platform.storage.database import Database, now_iso


class DataMiningAgentRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def init_schema(self) -> None:
        self.database.init_db()
        with self.database.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS data_mining_agent_sessions (
                  session_id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  status TEXT NOT NULL,
                  domain TEXT NOT NULL,
                  template_id TEXT,
                  paper_id TEXT,
                  plan_status TEXT NOT NULL DEFAULT 'not_started',
                  critic_status TEXT NOT NULL DEFAULT 'not_configured',
                  requirements_json TEXT NOT NULL DEFAULT '{}',
                  plan_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated
                  ON data_mining_agent_sessions(updated_at DESC);

                CREATE TABLE IF NOT EXISTS data_mining_agent_messages (
                  message_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  message_type TEXT NOT NULL DEFAULT 'text',
                  content TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES data_mining_agent_sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_messages_session
                  ON data_mining_agent_messages(session_id, created_at, message_id);

                CREATE TABLE IF NOT EXISTS data_mining_agent_jobs (
                  job_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  paper_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  current_step TEXT NOT NULL,
                  progress REAL NOT NULL DEFAULT 0,
                  error_message TEXT,
                  result_summary_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  started_at TEXT,
                  completed_at TEXT,
                  FOREIGN KEY(session_id) REFERENCES data_mining_agent_sessions(session_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_jobs_session
                  ON data_mining_agent_jobs(session_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS data_mining_agent_events (
                  event_id TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  status TEXT NOT NULL,
                  title TEXT NOT NULL,
                  detail TEXT,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(job_id) REFERENCES data_mining_agent_jobs(job_id),
                  FOREIGN KEY(session_id) REFERENCES data_mining_agent_sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_events_job
                  ON data_mining_agent_events(job_id, created_at, event_id);
                CREATE INDEX IF NOT EXISTS idx_agent_events_session
                  ON data_mining_agent_events(session_id, created_at, event_id);

                CREATE TABLE IF NOT EXISTS data_mining_agent_results (
                  result_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  job_id TEXT NOT NULL UNIQUE,
                  paper_id TEXT NOT NULL,
                  result_type TEXT NOT NULL,
                  raw_result_json TEXT NOT NULL DEFAULT '{}',
                  reviewed_result_json TEXT NOT NULL DEFAULT '{}',
                  review_status TEXT NOT NULL DEFAULT 'pending_review',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES data_mining_agent_sessions(session_id),
                  FOREIGN KEY(job_id) REFERENCES data_mining_agent_jobs(job_id),
                  FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_results_session
                  ON data_mining_agent_results(session_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS data_mining_agent_result_events (
                  event_id TEXT PRIMARY KEY,
                  result_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  action TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  message TEXT,
                  before_json TEXT NOT NULL DEFAULT '{}',
                  after_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(result_id) REFERENCES data_mining_agent_results(result_id),
                  FOREIGN KEY(session_id) REFERENCES data_mining_agent_sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_result_events_result
                  ON data_mining_agent_result_events(result_id, created_at, event_id);
                """
            )

    def create_session(self, session: AgentSession) -> AgentSession:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_mining_agent_sessions (
                  session_id, title, mode, status, domain, template_id, paper_id,
                  plan_status, critic_status, requirements_json, plan_json,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.title,
                    session.mode,
                    session.status,
                    session.domain,
                    session.template_id,
                    session.paper_id,
                    session.plan_status,
                    session.critic_status,
                    _dump(session.requirements),
                    _dump(session.plan),
                    session.created_at,
                    session.updated_at,
                ),
            )
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_mining_agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._session(row) if row else None

    def list_sessions(self, limit: int = 50) -> list[AgentSession]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM data_mining_agent_sessions
                ORDER BY updated_at DESC, session_id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [self._session(row) for row in rows]

    def update_session(self, session_id: str, **updates: Any) -> AgentSession | None:
        allowed = {
            "title",
            "status",
            "domain",
            "template_id",
            "paper_id",
            "plan_status",
            "critic_status",
            "requirements",
            "plan",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return self.get_session(session_id)
        columns: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            column = f"{key}_json" if key in {"requirements", "plan"} else key
            columns.append(f"{column} = ?")
            params.append(_dump(value) if key in {"requirements", "plan"} else value)
        columns.append("updated_at = ?")
        params.append(now_iso())
        params.append(session_id)
        with self.database.connect() as conn:
            conn.execute(
                f"UPDATE data_mining_agent_sessions SET {', '.join(columns)} WHERE session_id = ?",
                tuple(params),
            )
        return self.get_session(session_id)

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        message = AgentMessage(
            message_id=uuid4().hex,
            session_id=session_id,
            role=role,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            created_at=now_iso(),
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_mining_agent_messages (
                  message_id, session_id, role, message_type, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.role,
                    message.message_type,
                    message.content,
                    _dump(message.metadata),
                    message.created_at,
                ),
            )
            conn.execute(
                "UPDATE data_mining_agent_sessions SET updated_at = ? WHERE session_id = ?",
                (message.created_at, session_id),
            )
        return message

    def list_messages(self, session_id: str) -> list[AgentMessage]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM data_mining_agent_messages
                WHERE session_id = ?
                ORDER BY created_at, rowid
                """,
                (session_id,),
            ).fetchall()
        return [self._message(row) for row in rows]

    def create_job(self, job: AgentJob) -> AgentJob:
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_mining_agent_jobs (
                  job_id, session_id, paper_id, status, current_step, progress,
                  error_message, result_summary_json, created_at, updated_at,
                  started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.session_id,
                    job.paper_id,
                    job.status,
                    job.current_step,
                    job.progress,
                    job.error_message,
                    _dump(job.result_summary),
                    job.created_at,
                    job.updated_at,
                    job.started_at,
                    job.completed_at,
                ),
            )
        return job

    def get_job(self, job_id: str) -> AgentJob | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_mining_agent_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._job(row) if row else None

    def list_jobs(self, session_id: str) -> list[AgentJob]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM data_mining_agent_jobs
                WHERE session_id = ?
                ORDER BY created_at DESC, job_id DESC
                """,
                (session_id,),
            ).fetchall()
        return [self._job(row) for row in rows]

    def update_job(self, job_id: str, **updates: Any) -> AgentJob | None:
        allowed = {
            "status",
            "current_step",
            "progress",
            "error_message",
            "result_summary",
            "started_at",
            "completed_at",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return self.get_job(job_id)
        columns: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            column = "result_summary_json" if key == "result_summary" else key
            columns.append(f"{column} = ?")
            params.append(_dump(value) if key == "result_summary" else value)
        columns.append("updated_at = ?")
        params.append(now_iso())
        params.append(job_id)
        with self.database.connect() as conn:
            conn.execute(
                f"UPDATE data_mining_agent_jobs SET {', '.join(columns)} WHERE job_id = ?",
                tuple(params),
            )
        return self.get_job(job_id)

    def add_event(
        self,
        *,
        job_id: str,
        session_id: str,
        event_type: str,
        stage: str,
        status: str,
        title: str,
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            event_id=uuid4().hex,
            job_id=job_id,
            session_id=session_id,
            event_type=event_type,
            stage=stage,
            status=status,
            title=title,
            detail=detail,
            metadata=metadata or {},
            created_at=now_iso(),
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_mining_agent_events (
                  event_id, job_id, session_id, event_type, stage, status,
                  title, detail, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.job_id,
                    event.session_id,
                    event.event_type,
                    event.stage,
                    event.status,
                    event.title,
                    event.detail,
                    _dump(event.metadata),
                    event.created_at,
                ),
            )
        return event

    def list_events(
        self,
        *,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> list[AgentEvent]:
        if not session_id and not job_id:
            return []
        where = "job_id = ?" if job_id else "session_id = ?"
        value = job_id or session_id
        with self.database.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM data_mining_agent_events
                WHERE {where}
                ORDER BY created_at, rowid
                """,
                (value,),
            ).fetchall()
        return [self._event(row) for row in rows]

    def upsert_result(
        self,
        *,
        session_id: str,
        job_id: str,
        paper_id: str,
        result_type: str,
        raw_result: dict[str, Any],
        reviewed_result: dict[str, Any] | None = None,
        review_status: str = "pending_review",
    ) -> AgentResult:
        existing = self.get_result_by_job(job_id)
        timestamp = now_iso()
        result = AgentResult(
            result_id=existing.result_id if existing else uuid4().hex,
            session_id=session_id,
            job_id=job_id,
            paper_id=paper_id,
            result_type=result_type,
            raw_result=raw_result,
            reviewed_result=reviewed_result if reviewed_result is not None else raw_result,
            review_status=review_status,
            created_at=existing.created_at if existing else timestamp,
            updated_at=timestamp,
        )
        with self.database.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_mining_agent_results (
                  result_id, session_id, job_id, paper_id, result_type,
                  raw_result_json, reviewed_result_json, review_status,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                  result_type=excluded.result_type,
                  raw_result_json=excluded.raw_result_json,
                  reviewed_result_json=excluded.reviewed_result_json,
                  review_status=excluded.review_status,
                  updated_at=excluded.updated_at
                """,
                (
                    result.result_id,
                    result.session_id,
                    result.job_id,
                    result.paper_id,
                    result.result_type,
                    _dump(result.raw_result),
                    _dump(result.reviewed_result),
                    result.review_status,
                    result.created_at,
                    result.updated_at,
                ),
            )
        return result

    def get_result_by_job(self, job_id: str) -> AgentResult | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_mining_agent_results WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._result(row) if row else None

    def latest_result(self, session_id: str) -> AgentResult | None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM data_mining_agent_results
                WHERE session_id = ?
                ORDER BY updated_at DESC, result_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._result(row) if row else None

    def update_result(
        self,
        result_id: str,
        reviewed_result: dict[str, Any],
        *,
        actor: str,
        message: str | None,
    ) -> AgentResult | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_mining_agent_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row:
                return None
            before = self._result(row)
            timestamp = now_iso()
            conn.execute(
                """
                UPDATE data_mining_agent_results
                SET reviewed_result_json = ?, review_status = 'in_review', updated_at = ?
                WHERE result_id = ?
                """,
                (_dump(reviewed_result), timestamp, result_id),
            )
            conn.execute(
                """
                INSERT INTO data_mining_agent_result_events (
                  event_id, result_id, session_id, action, actor, message,
                  before_json, after_json, created_at
                ) VALUES (?, ?, ?, 'update_result', ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    result_id,
                    before.session_id,
                    actor,
                    message,
                    _dump(before.reviewed_result),
                    _dump(reviewed_result),
                    timestamp,
                ),
            )
        return self.get_result(result_id)

    def get_result(self, result_id: str) -> AgentResult | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT * FROM data_mining_agent_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        return self._result(row) if row else None

    def interrupt_stale_jobs(self) -> int:
        timestamp = now_iso()
        with self.database.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE data_mining_agent_jobs
                SET status = 'interrupted',
                    current_step = 'interrupted',
                    error_message = COALESCE(
                      error_message,
                      'API process restarted while the job was running.'
                    ),
                    updated_at = ?,
                    completed_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (timestamp, timestamp),
            )
        return cursor.rowcount

    @staticmethod
    def _session(row: sqlite3.Row) -> AgentSession:
        payload = dict(row)
        payload["requirements"] = _load(payload.pop("requirements_json"))
        payload["plan"] = _load(payload.pop("plan_json"))
        return AgentSession.model_validate(payload)

    @staticmethod
    def _message(row: sqlite3.Row) -> AgentMessage:
        payload = dict(row)
        payload["metadata"] = _load(payload.pop("metadata_json"))
        return AgentMessage.model_validate(payload)

    @staticmethod
    def _job(row: sqlite3.Row) -> AgentJob:
        payload = dict(row)
        payload["result_summary"] = _load(payload.pop("result_summary_json"))
        return AgentJob.model_validate(payload)

    @staticmethod
    def _event(row: sqlite3.Row) -> AgentEvent:
        payload = dict(row)
        payload["metadata"] = _load(payload.pop("metadata_json"))
        return AgentEvent.model_validate(payload)

    @staticmethod
    def _result(row: sqlite3.Row) -> AgentResult:
        payload = dict(row)
        payload["raw_result"] = _load(payload.pop("raw_result_json"))
        payload["reviewed_result"] = _load(payload.pop("reviewed_result_json"))
        return AgentResult.model_validate(payload)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    payload = json.loads(value)
    return payload if isinstance(payload, dict) else {"value": payload}
