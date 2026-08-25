from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from evolab_local.mining_platform.api.routes_batch import router as batch_router
from evolab_local.mining_platform.api.routes_candidate_review import (
    router as candidate_review_router,
)
from evolab_local.mining_platform.api.routes_data_mining_agent import (
    router as data_mining_agent_router,
)
from evolab_local.mining_platform.api.routes_extraction import router as extraction_router
from evolab_local.mining_platform.api.routes_health import router as health_router
from evolab_local.mining_platform.api.routes_materials import router as materials_router
from evolab_local.mining_platform.api.routes_papers import router as papers_router
from evolab_local.mining_platform.api.routes_parse import router as parse_router
from evolab_local.mining_platform.api.routes_review import router as review_router
from evolab_local.mining_platform.batch_worker_service import BatchWorkerService
from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.candidate_review_service import CandidateReviewService
from evolab_local.mining_platform.chemical_figure_collector_service import (
    ChemicalFigureCollectorService,
)
from evolab_local.mining_platform.core.config import MiningPlatformConfig, load_config
from evolab_local.mining_platform.domain_template_service import DomainTemplateService
from evolab_local.mining_platform.data_mining_agent_service import DataMiningAgentService
from evolab_local.mining_platform.extraction_service import ExtractionService
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.material_public_resolver_service import (
    MaterialPublicResolverService,
)
from evolab_local.mining_platform.material_identity_judge_service import (
    MaterialIdentityJudgeService,
)
from evolab_local.mining_platform.material_identity_evidence_service import (
    MaterialIdentityEvidenceService,
)
from evolab_local.mining_platform.material_auto_decision_service import (
    MaterialAutoDecisionService,
)
from evolab_local.mining_platform.material_property_review_service import (
    MaterialPropertyReviewService,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
)
from evolab_local.mining_platform.parse_service import ParseService
from evolab_local.mining_platform.review_service import ReviewService
from evolab_local.optovlab.api import router as optovlab_router
from evolab_local.optovlab.config import load_optovlab_config
from evolab_local.optovlab.service import OptoVLabService


def create_app(
    config_path: Path = Path("config/mining_platform/mining_platform.yaml"),
    config: MiningPlatformConfig | None = None,
) -> FastAPI:
    app_config = config or load_config(config_path)
    app = FastAPI(
        title="OptoVLab Research Platform",
        version="0.1.0",
        description="Evidence-backed organic optoelectronics mining and agent APIs.",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = app_config
    app.state.paper_service = PaperService(app_config)
    app.state.batch_worker_service = BatchWorkerService(app_config)
    app.state.parse_service = ParseService(app_config)
    app.state.extraction_service = ExtractionService(app_config)
    app.state.review_service = ReviewService(app_config)
    app.state.candidate_review_service = CandidateReviewService(app_config)
    app.state.candidate_ingestion_service = CandidateIngestionService(app_config)
    app.state.domain_template_service = DomainTemplateService(app_config)
    app.state.material_resolution_service = MaterialResolutionService(app_config)
    app.state.material_public_resolver_service = MaterialPublicResolverService(app_config)
    app.state.material_identity_judge_service = MaterialIdentityJudgeService(app_config)
    app.state.material_identity_evidence_service = MaterialIdentityEvidenceService(app_config)
    app.state.material_auto_decision_service = MaterialAutoDecisionService(app_config)
    app.state.material_property_review_service = MaterialPropertyReviewService(app_config)
    app.state.material_structure_review_service = MaterialStructureReviewService(app_config)
    app.state.chemical_figure_collector_service = ChemicalFigureCollectorService(app_config)
    app.state.material_structure_agent_service = MaterialStructureAgentService(app_config)
    app.state.data_mining_agent_service = DataMiningAgentService(
        app_config,
        paper_service=app.state.paper_service,
        candidate_ingestion_service=app.state.candidate_ingestion_service,
        material_resolution_service=app.state.material_resolution_service,
        material_public_resolver_service=app.state.material_public_resolver_service,
        material_identity_judge_service=app.state.material_identity_judge_service,
        material_auto_decision_service=app.state.material_auto_decision_service,
        material_structure_agent_service=app.state.material_structure_agent_service,
    )
    # Finish schema checks before the server reports ready so the first review request is fast.
    app.state.paper_service.init_runtime()
    app.state.data_mining_agent_service.init_runtime()
    app.state.data_mining_agent_service.recover_interrupted_jobs()
    app.state.optovlab_service = OptoVLabService(
        load_optovlab_config(),
        data_mining_service=app.state.data_mining_agent_service,
    )
    app.state.optovlab_service.init_runtime()
    app.router.add_event_handler("shutdown", app.state.data_mining_agent_service.close)
    app.router.add_event_handler("shutdown", app.state.optovlab_service.close)
    app.include_router(health_router)
    app.include_router(data_mining_agent_router)
    app.include_router(batch_router)
    app.include_router(materials_router)
    app.include_router(candidate_review_router)
    app.include_router(review_router)
    app.include_router(parse_router)
    app.include_router(extraction_router)
    app.include_router(papers_router)
    app.include_router(optovlab_router)
    return app


def _cors_origins() -> list[str]:
    configured = os.getenv("OPTOVLAB_CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:5175",
        "http://localhost:5175",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]
