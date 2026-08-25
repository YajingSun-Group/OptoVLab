from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from evolab_local.optovlab.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AppSummary,
    ConversationTurn,
    HPCStatus,
    MessageCreate,
    MiningResultUpdate,
    RAGSearchRequest,
    RAGSearchResult,
    SessionCreate,
    SessionSummary,
    SessionWorkspace,
    TrainingJob,
    TrainingRequest,
)
from evolab_local.optovlab.service import OptoVLabService


router = APIRouter(prefix="/api/optovlab", tags=["optovlab"])


def _service(request: Request) -> OptoVLabService:
    return request.app.state.optovlab_service


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


@router.get("/status")
def status(request: Request) -> dict:
    service = _service(request)
    return {
        "status": "ok",
        "agent_runtime": service.agent_runtime.describe(),
        "dataset": service.catalog.stats(),
    }


@router.get("/apps", response_model=list[AppSummary])
def list_apps(request: Request) -> list[AppSummary]:
    return _service(request).list_apps()


@router.get("/skills")
def list_skills(request: Request) -> list[dict]:
    return _service(request).analysis.catalog()


@router.post("/sessions", response_model=SessionSummary, status_code=201)
def create_session(request: Request, payload: SessionCreate) -> SessionSummary:
    return _service(request).create_session(payload)


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(
    request: Request,
    agent_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[SessionSummary]:
    return _service(request).repository.list_sessions(agent_type, limit)


@router.get("/sessions/{session_id}/workspace", response_model=SessionWorkspace)
def get_workspace(request: Request, session_id: str) -> SessionWorkspace:
    try:
        return _service(request).get_workspace(session_id)
    except (KeyError, ValueError) as exc:
        raise _error(exc) from exc


@router.delete("/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    try:
        return _service(request).delete_session(session_id)
    except (KeyError, ValueError, OSError) as exc:
        raise _error(exc) from exc


@router.post("/sessions/{session_id}/messages", response_model=ConversationTurn)
async def add_message(
    request: Request,
    session_id: str,
    payload: MessageCreate,
) -> ConversationTurn:
    try:
        return await _service(request).handle_message(session_id, payload.content)
    except (KeyError, ValueError) as exc:
        raise _error(exc) from exc


@router.post("/sessions/{session_id}/pdf", status_code=201)
async def upload_pdf(
    request: Request,
    session_id: str,
    pdf: UploadFile = File(...),
) -> dict:
    try:
        content = await pdf.read()
        return _service(request).upload_pdf(
            session_id,
            filename=pdf.filename or "uploaded.pdf",
            content=content,
        )
    except (KeyError, ValueError) as exc:
        raise _error(exc) from exc
    finally:
        await pdf.close()


@router.post("/sessions/{session_id}/mining/start", status_code=202)
def start_mining(request: Request, session_id: str) -> dict:
    try:
        return _service(request).start_data_mining(session_id)
    except (KeyError, ValueError) as exc:
        raise _error(exc) from exc


@router.put("/sessions/{session_id}/mining/result")
def update_mining_result(
    request: Request,
    session_id: str,
    payload: MiningResultUpdate,
) -> dict:
    try:
        return _service(request).update_mining_result(
            session_id,
            payload.mining_session_id,
            payload.reviewed_result,
            payload.message,
        )
    except (KeyError, ValueError) as exc:
        raise _error(exc) from exc


@router.post("/sessions/{session_id}/analysis", response_model=AnalysisResult)
def run_analysis(
    request: Request,
    session_id: str,
    payload: AnalysisRequest,
) -> AnalysisResult:
    try:
        return _service(request).run_analysis(session_id, payload)
    except (KeyError, ValueError) as exc:
        raise _error(exc) from exc


@router.post("/rag/search", response_model=RAGSearchResult)
def search_rag(request: Request, payload: RAGSearchRequest) -> RAGSearchResult:
    try:
        return _service(request).search_devices(
            payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except ValueError as exc:
        raise _error(exc) from exc


@router.get("/models")
def list_models(request: Request) -> list[dict]:
    return _service(request).hpc.model_registry()


@router.get("/hpc/status", response_model=HPCStatus)
def hpc_status(request: Request) -> HPCStatus:
    return _service(request).hpc.status()


@router.post("/sessions/{session_id}/training", response_model=TrainingJob)
def prepare_training(
    request: Request,
    session_id: str,
    payload: TrainingRequest,
) -> TrainingJob:
    try:
        return _service(request).prepare_training(session_id, payload)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise _error(exc) from exc


@router.get("/artifacts/{session_id}/{filename}")
def get_artifact(request: Request, session_id: str, filename: str) -> FileResponse:
    root = (_service(request).config.runtime.artifact_dir / session_id).resolve()
    target = (root / filename).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)
