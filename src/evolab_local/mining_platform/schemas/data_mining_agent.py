from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentSessionMode = Literal["preset", "custom"]


class AgentTemplateSummary(BaseModel):
    template_id: str
    name: str
    version: str
    domain: str
    description: str
    status: str = "ready"
    capabilities: list[str] = Field(default_factory=list)
    plan: dict[str, Any] = Field(default_factory=dict)


class AgentSessionCreate(BaseModel):
    mode: AgentSessionMode = "preset"
    template_id: str | None = "oled_device_v1"
    title: str | None = None
    initial_message: str | None = None


class AgentSession(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    title: str
    mode: str
    status: str
    domain: str
    template_id: str | None = None
    paper_id: str | None = None
    plan_status: str = "not_started"
    critic_status: str = "not_configured"
    requirements: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class AgentMessageCreate(BaseModel):
    content: str
    auto_respond: bool = True


class AgentMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: str
    session_id: str
    role: str
    message_type: str = "text"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AgentConversationTurn(BaseModel):
    session: AgentSession
    user_message: AgentMessage
    assistant_message: AgentMessage | None = None


class AgentPlanUpdate(BaseModel):
    plan: dict[str, Any]
    actor: str = "local_user"
    message: str | None = None


class AgentPlanAction(BaseModel):
    actor: str = "local_user"
    message: str | None = None


class AgentJobCreate(BaseModel):
    force: bool = False


class AgentJob(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    session_id: str
    paper_id: str
    status: str
    current_step: str
    progress: float = 0.0
    error_message: str | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class AgentEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    job_id: str
    session_id: str
    event_type: str
    stage: str
    status: str
    title: str
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AgentUploadedPaper(BaseModel):
    session: AgentSession
    paper_id: str
    filename: str
    size_bytes: int
    sha256: str
    page_count: int


class AgentResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    result_id: str
    session_id: str
    job_id: str
    paper_id: str
    result_type: str
    raw_result: dict[str, Any] = Field(default_factory=dict)
    reviewed_result: dict[str, Any] = Field(default_factory=dict)
    review_status: str = "pending_review"
    created_at: str
    updated_at: str


class AgentResultUpdate(BaseModel):
    reviewed_result: dict[str, Any]
    actor: str = "local_user"
    message: str | None = None


class AgentWorkspace(BaseModel):
    session: AgentSession
    messages: list[AgentMessage] = Field(default_factory=list)
    jobs: list[AgentJob] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    paper: dict[str, Any] | None = None
    result: AgentResult | None = None
    candidate_bundle: dict[str, Any] | None = None
    material_bundle: dict[str, Any] | None = None
