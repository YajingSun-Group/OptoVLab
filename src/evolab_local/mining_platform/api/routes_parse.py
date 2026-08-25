from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from evolab_local.mining_platform.parse_service import ParseService
from evolab_local.mining_platform.schemas.document import DocumentBlock, ParseResult

router = APIRouter(prefix="/api/papers", tags=["parse"])


def _service(request: Request) -> ParseService:
    return request.app.state.parse_service


@router.post("/{paper_id:path}/parse", response_model=ParseResult)
def parse_paper(request: Request, paper_id: str) -> ParseResult:
    try:
        result = _service(request).parse_paper(paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="Paper not found")
    return result


@router.get("/{paper_id:path}/blocks", response_model=list[DocumentBlock])
def list_blocks(request: Request, paper_id: str) -> list[DocumentBlock]:
    return _service(request).list_blocks(paper_id)


@router.get("/{paper_id:path}/blocks/{block_id}", response_model=DocumentBlock)
def get_block(request: Request, paper_id: str, block_id: str) -> DocumentBlock:
    block = _service(request).get_block(paper_id, block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Document block not found")
    return block
