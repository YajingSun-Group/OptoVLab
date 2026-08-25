from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from evolab_local.mining_platform.batch_worker_service import BatchWorkerService
from evolab_local.mining_platform.schemas.batch import (
    BatchReviewDetail,
    BatchReviewOverview,
    BatchWorkerRunResult,
)

router = APIRouter(prefix="/api/batches", tags=["batch-review"])


def _service(request: Request) -> BatchWorkerService:
    return request.app.state.batch_worker_service


@router.get("/review", response_model=BatchReviewOverview)
def list_review_batches(
    request: Request,
    batch_size: int | None = Query(default=None, ge=1, le=100),
) -> BatchReviewOverview:
    return _service(request).list_review_batches(batch_size=batch_size)


@router.get("/review/{batch_index}", response_model=BatchReviewDetail)
def get_review_batch(
    request: Request,
    batch_index: int,
    batch_size: int | None = Query(default=None, ge=1, le=100),
) -> BatchReviewDetail:
    try:
        return _service(request).get_review_batch(batch_index, batch_size=batch_size)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/review/run-next", response_model=BatchWorkerRunResult)
def run_next_review_batch(
    request: Request,
    batch_size: int | None = Query(default=None, ge=1, le=100),
    retry_failed: bool = Query(default=False),
    public_resolve: bool | None = Query(default=None),
    identity_judge: bool | None = Query(default=None),
    material_ocsr: bool | None = Query(default=None),
) -> BatchWorkerRunResult:
    return _service(request).run_next_review_batch(
        batch_size=batch_size,
        include_failed_retries=retry_failed,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_ocsr=material_ocsr,
    )
