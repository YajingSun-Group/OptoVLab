from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentType = Literal["data_mining", "device_modeling", "experimental_design"]


class AppSummary(BaseModel):
    app_id: str
    name: str
    category: str
    description: str
    route: str
    status: str = "ready"
    metrics: dict[str, Any] = Field(default_factory=dict)


class SessionCreate(BaseModel):
    agent_type: AgentType
    title: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    agent_type: AgentType
    title: str
    status: str
    created_at: str
    updated_at: str


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


class Message(BaseModel):
    message_id: str
    session_id: str
    role: Literal["user", "assistant", "system", "tool"]
    message_type: str = "text"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ToolEvent(BaseModel):
    event_id: str
    session_id: str
    job_id: str | None = None
    tool_name: str
    status: str
    title: str
    detail: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class Artifact(BaseModel):
    artifact_id: str
    session_id: str
    artifact_type: str
    title: str
    filename: str
    mime_type: str
    url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ResourceLink(BaseModel):
    link_id: str
    session_id: str
    resource_type: str
    resource_id: str
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ConversationTurn(BaseModel):
    session: SessionSummary
    user_message: Message
    assistant_message: Message
    tool_events: list[ToolEvent] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class SessionWorkspace(BaseModel):
    session: SessionSummary
    messages: list[Message] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    resources: list[ResourceLink] = Field(default_factory=list)
    linked_workspaces: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    skill_id: Literal[
        "dataset_summary",
        "data_quality_profile",
        "univariate_distribution",
        "bivariate_relationship",
        "group_comparison",
        "correlation_matrix",
    ]
    scope: Literal["auto", "session", "catalog"] = "auto"
    session_ids: list[str] = Field(default_factory=list)
    x_field: str | None = None
    y_field: str | None = None
    group_field: str | None = None
    metric: str | None = None


class AnalysisResult(BaseModel):
    skill_id: str
    summary: str
    statistics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)


class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    top_k: int | None = Field(default=None, ge=1, le=25)
    filters: dict[str, Any] = Field(default_factory=dict)


class RAGHit(BaseModel):
    rank: int
    score: float
    device_id: str
    doi: str | None = None
    title: str | None = None
    journal: str | None = None
    device_label: str | None = None
    architecture: str | None = None
    final_emitter: str | None = None
    eqe_max: float | None = None
    record: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResult(BaseModel):
    query: str
    total_devices: int
    hits: list[RAGHit]


class TrainingRequest(BaseModel):
    run_name: str = Field(default="optovlab-run", pattern=r"^[A-Za-z0-9_.-]+$")
    config_path: str | None = None
    partition: str | None = None
    gpus: int = Field(default=1, ge=1, le=4)
    time_limit: str | None = None
    seed: int | None = None
    evaluate_test: bool = False
    confirm_submit: bool = False


class TrainingJob(BaseModel):
    job_id: str
    status: str
    submitted: bool
    scheduler_job_id: str | None = None
    script_path: str
    command: list[str]
    created_at: str


class MiningResultUpdate(BaseModel):
    mining_session_id: str
    reviewed_result: dict[str, Any]
    message: str | None = None


class HPCStatus(BaseModel):
    scheduler_available: bool
    partitions: list[dict[str, Any]] = Field(default_factory=list)
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
