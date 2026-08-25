from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from evolab_local.mining_platform.data_mining_agent_service import DataMiningAgentService
from evolab_local.mining_platform.schemas.data_mining_agent import (
    AgentConversationTurn,
    AgentEvent,
    AgentJob,
    AgentJobCreate,
    AgentMessageCreate,
    AgentPlanAction,
    AgentPlanUpdate,
    AgentResult,
    AgentResultUpdate,
    AgentSession,
    AgentSessionCreate,
    AgentTemplateSummary,
    AgentUploadedPaper,
    AgentWorkspace,
)

router = APIRouter(prefix="/api/data-mining-agent", tags=["data-mining-agent"])


def _service(request: Request) -> DataMiningAgentService:
    return request.app.state.data_mining_agent_service


def _not_found_or_unprocessable(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    return HTTPException(status_code=422, detail=str(exc))


@router.get("/templates", response_model=list[AgentTemplateSummary])
def list_templates(request: Request) -> list[AgentTemplateSummary]:
    return _service(request).list_templates()


@router.post("/sessions", response_model=AgentSession, status_code=201)
def create_session(request: Request, payload: AgentSessionCreate) -> AgentSession:
    try:
        return _service(request).create_session(payload)
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.get("/sessions", response_model=list[AgentSession])
def list_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[AgentSession]:
    return _service(request).list_sessions(limit=limit)


@router.get("/sessions/{session_id}", response_model=AgentSession)
def get_session(request: Request, session_id: str) -> AgentSession:
    session = _service(request).get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="DataMining Agent session not found")
    return session


@router.get("/sessions/{session_id}/workspace", response_model=AgentWorkspace)
def get_workspace(request: Request, session_id: str) -> AgentWorkspace:
    try:
        return _service(request).get_workspace(session_id)
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.post(
    "/sessions/{session_id}/messages",
    response_model=AgentConversationTurn,
)
def add_message(
    request: Request,
    session_id: str,
    payload: AgentMessageCreate,
) -> AgentConversationTurn:
    try:
        return _service(request).add_message(
            session_id,
            content=payload.content,
            auto_respond=payload.auto_respond,
        )
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.post("/sessions/{session_id}/plan/generate", response_model=AgentSession)
def generate_plan(request: Request, session_id: str) -> AgentSession:
    try:
        return _service(request).generate_plan(session_id)
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.put("/sessions/{session_id}/plan", response_model=AgentSession)
def update_plan(
    request: Request,
    session_id: str,
    payload: AgentPlanUpdate,
) -> AgentSession:
    try:
        return _service(request).update_plan(
            session_id,
            plan=payload.plan,
            actor=payload.actor,
            message=payload.message,
        )
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.post("/sessions/{session_id}/plan/approve", response_model=AgentSession)
def approve_plan(
    request: Request,
    session_id: str,
    payload: AgentPlanAction = AgentPlanAction(),
) -> AgentSession:
    try:
        return _service(request).approve_plan(
            session_id,
            actor=payload.actor,
            message=payload.message,
        )
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.post("/sessions/{session_id}/pdf", response_model=AgentUploadedPaper)
async def upload_pdf(
    request: Request,
    session_id: str,
    pdf: UploadFile = File(...),
) -> AgentUploadedPaper:
    try:
        content = await pdf.read()
        return _service(request).upload_pdf(
            session_id,
            filename=pdf.filename or "uploaded.pdf",
            content=content,
        )
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc
    finally:
        await pdf.close()


@router.post("/sessions/{session_id}/run", response_model=AgentJob, status_code=202)
def start_job(
    request: Request,
    session_id: str,
    payload: AgentJobCreate = AgentJobCreate(),
) -> AgentJob:
    try:
        return _service(request).start_job(session_id, force=payload.force)
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc


@router.get("/jobs/{job_id}", response_model=AgentJob)
def get_job(request: Request, job_id: str) -> AgentJob:
    job = _service(request).get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="DataMining Agent job not found")
    return job


@router.get("/jobs/{job_id}/events", response_model=list[AgentEvent])
def list_job_events(request: Request, job_id: str) -> list[AgentEvent]:
    if not _service(request).get_job(job_id):
        raise HTTPException(status_code=404, detail="DataMining Agent job not found")
    return _service(request).list_job_events(job_id)


@router.put("/sessions/{session_id}/result", response_model=AgentResult)
def update_result(
    request: Request,
    session_id: str,
    payload: AgentResultUpdate,
) -> AgentResult:
    try:
        return _service(request).update_result(
            session_id,
            reviewed_result=payload.reviewed_result,
            actor=payload.actor,
            message=payload.message,
        )
    except (KeyError, ValueError) as exc:
        raise _not_found_or_unprocessable(exc) from exc
