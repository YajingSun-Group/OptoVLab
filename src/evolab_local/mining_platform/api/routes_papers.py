from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.schemas.paper import Paper, PaperIngestResult
from evolab_local.mining_platform.schemas.review import ReviewAction

router = APIRouter(prefix="/api/papers", tags=["papers"])


def _service(request: Request) -> PaperService:
    return request.app.state.paper_service


@router.get("", response_model=list[Paper])
def list_papers(request: Request) -> list[Paper]:
    return _service(request).list_papers()


@router.post("/ingest-from-pdf-downloader", response_model=PaperIngestResult)
def ingest_from_pdf_downloader(
    request: Request,
    domain: str = Query(default="unknown"),
) -> PaperIngestResult:
    return _service(request).ingest_from_pdf_downloader(domain=domain)


@router.post("/{paper_id:path}/exclude-review", response_model=Paper)
def exclude_review_article(
    request: Request,
    paper_id: str,
    action: ReviewAction = ReviewAction(),
) -> Paper:
    try:
        paper = _service(request).exclude_review_article(
            paper_id,
            actor=action.actor,
            message=action.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.post("/{paper_id:path}/restore-review", response_model=Paper)
def restore_excluded_paper(
    request: Request,
    paper_id: str,
    action: ReviewAction = ReviewAction(),
) -> Paper:
    try:
        paper = _service(request).restore_excluded_paper(
            paper_id,
            actor=action.actor,
            message=action.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.get("/{paper_id:path}/pdf")
def get_paper_pdf(request: Request, paper_id: str) -> FileResponse:
    pdf_path = _service(request).get_pdf_path(paper_id)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Encoding": "identity",
        },
    )


@router.get("/{paper_id:path}", response_model=Paper)
def get_paper(request: Request, paper_id: str) -> Paper:
    paper = _service(request).get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper
