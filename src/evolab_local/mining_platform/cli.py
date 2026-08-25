from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.batch_worker_service import BatchWorkerService
from evolab_local.mining_platform.candidate_ingestion_service import CandidateIngestionService
from evolab_local.mining_platform.candidate_review_service import CandidateReviewService
from evolab_local.mining_platform.chemical_figure_collector_service import (
    ChemicalFigureCollectorService,
)
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.core.logging import configure_logging
from evolab_local.mining_platform.domain_template_service import DomainTemplateService
from evolab_local.mining_platform.external.openai_compatible_client import StaticJSONLLMClient
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
from evolab_local.mining_platform.material_property_mining_service import (
    MaterialPropertyMiningService,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.material_web_rescue_service import (
    MaterialWebRescueService,
)
from evolab_local.mining_platform.material_web_rescue_report_service import (
    MaterialWebRescueReportService,
)
from evolab_local.mining_platform.material_structure_agent_service import (
    MaterialStructureAgentService,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialReviewAction,
    PaperMaterialNameReviewAction,
    MaterialStructureEditAction,
)
from evolab_local.mining_platform.mining.llm_mining_service import LLMMiningService
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService
from evolab_local.mining_platform.paper_metadata_service import PaperMetadataEnrichmentService
from evolab_local.mining_platform.parse_service import ParseService
from evolab_local.mining_platform.stage3_batch_group_service import Stage3BatchGroupService

app = typer.Typer(help="Human-in-the-loop literature mining platform.")
console = Console()


def _service(config_path: Path) -> PaperService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return PaperService(config)


def _batch_worker_service(config_path: Path) -> BatchWorkerService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return BatchWorkerService(config)


def _parse_service(config_path: Path) -> ParseService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return ParseService(config)


def _extraction_service(config_path: Path) -> ExtractionService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return ExtractionService(config)


def _candidate_review_service(config_path: Path) -> CandidateReviewService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return CandidateReviewService(config)


def _template_service(config_path: Path) -> DomainTemplateService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return DomainTemplateService(config)


def _candidate_ingestion_service(config_path: Path) -> CandidateIngestionService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return CandidateIngestionService(config)


def _mineru_parse_service(config_path: Path) -> MinerUParseService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MinerUParseService(config)


def _llm_mining_service(
    config_path: Path,
    mock_input: Path | None = None,
) -> LLMMiningService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    client = None
    if mock_input is not None:
        payload = json.loads(mock_input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise typer.BadParameter("Mock mining result JSON must be an object.")
        client = StaticJSONLLMClient(payload)
    return LLMMiningService(config, llm_client=client)


def _material_property_mining_service(
    config_path: Path,
    mock_input: Path | None = None,
) -> MaterialPropertyMiningService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    client = None
    if mock_input is not None:
        payload = json.loads(mock_input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise typer.BadParameter("Mock material property result JSON must be an object.")
        client = StaticJSONLLMClient(payload)
    return MaterialPropertyMiningService(config, llm_client=client)


def _material_resolution_service(config_path: Path) -> MaterialResolutionService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialResolutionService(config)


def _material_public_resolver_service(config_path: Path) -> MaterialPublicResolverService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialPublicResolverService(config)


def _material_identity_judge_service(config_path: Path) -> MaterialIdentityJudgeService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialIdentityJudgeService(config)


def _material_identity_evidence_service(config_path: Path) -> MaterialIdentityEvidenceService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialIdentityEvidenceService(config)


def _material_auto_decision_service(config_path: Path) -> MaterialAutoDecisionService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialAutoDecisionService(config)


def _material_structure_review_service(config_path: Path) -> MaterialStructureReviewService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialStructureReviewService(config)


def _material_web_rescue_service(config_path: Path) -> MaterialWebRescueService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialWebRescueService(config)


def _chemical_figure_collector_service(config_path: Path) -> ChemicalFigureCollectorService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return ChemicalFigureCollectorService(config)


def _material_structure_agent_service(config_path: Path) -> MaterialStructureAgentService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return MaterialStructureAgentService(config)


def _paper_metadata_service(config_path: Path) -> PaperMetadataEnrichmentService:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    return PaperMetadataEnrichmentService(config)


@app.command("init-runtime")
def init_runtime(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _service(config)
    service.init_runtime()
    console.print(f"Runtime: {service.config.paths.runtime_dir}")
    console.print(f"SQLite: {service.config.paths.sqlite_path}")
    console.print(f"Registry: {service.config.paths.paper_registry_path}")


@app.command("ingest-pdfs")
def ingest_pdfs(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
    domain: str = typer.Option("unknown", "--domain", help="Domain label for imported papers."),
) -> None:
    service = _service(config)
    result = service.ingest_from_pdf_downloader(domain=domain)
    console.print(
        f"Imported {result.imported_count} paper(s); skipped {result.skipped_count} manifest item(s)."
    )
    console.print(f"SQLite: {service.config.paths.sqlite_path}")
    console.print(f"Registry: {service.config.paths.paper_registry_path}")


@app.command("import-inbox-pdfs-from-csv")
def import_inbox_pdfs_from_csv(
    csv_path: Path = typer.Option(
        ...,
        "--csv",
        help="CSV file containing DOI-encoded PDF names.",
    ),
    source_pdf_dir: Path = typer.Option(
        ...,
        "--source-pdf-dir",
        help="Directory containing DOI-encoded PDF files.",
    ),
    doi_column: str = typer.Option("doi_encode", "--doi-column"),
    limit: int | None = typer.Option(None, "--limit"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    domain: str | None = typer.Option(None, "--domain"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _batch_worker_service(config)
    result = service.import_pdfs_from_csv(
        csv_path=csv_path,
        source_pdf_dir=source_pdf_dir,
        doi_column=doi_column,
        limit=limit,
        overwrite=overwrite,
        domain=domain,
    )
    console.print(
        f"requested={result.requested_count}; copied={result.copied_count}; "
        f"existing={result.existing_count}; missing={result.missing_count}; "
        f"registered={result.registered_count}"
    )
    if result.missing:
        for item in result.missing[:20]:
            console.print(f"[yellow]missing[/yellow] {item}")
        if len(result.missing) > 20:
            console.print(f"[yellow]... {len(result.missing) - 20} more missing files[/yellow]")
    console.print(f"Inbox: {service.config.paths.inbox_pdfs_dir}")


@app.command("scan-inbox-pdfs")
def scan_inbox_pdfs(
    limit: int | None = typer.Option(None, "--limit"),
    domain: str | None = typer.Option(None, "--domain"),
    stable_file_seconds: float | None = typer.Option(None, "--stable-file-seconds"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _batch_worker_service(config)
    result = service.scan_inbox_pdfs(
        limit=limit,
        domain=domain,
        stable_file_seconds=stable_file_seconds,
    )
    console.print(
        f"scanned={result.scanned_count}; registered={result.registered_count}; "
        f"existing={result.existing_count}; unstable={result.skipped_unstable_count}"
    )
    console.print(f"Inbox: {service.config.paths.inbox_pdfs_dir}")


@app.command("list-batch-jobs")
def list_batch_jobs(
    status: str | None = typer.Option(None, "--status"),
    limit: int | None = typer.Option(50, "--limit"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    jobs = _batch_worker_service(config).list_jobs(status=status, limit=limit)
    _print_batch_jobs(jobs, title="Batch Mining Jobs")


@app.command("list-review-batches")
def list_review_batches(
    batch_size: int | None = typer.Option(None, "--batch-size"),
    limit: int | None = typer.Option(30, "--limit"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    overview = _batch_worker_service(config).list_review_batches(batch_size=batch_size)
    table = Table(title="Review Batches")
    for column in ("batch", "status", "confirmed", "ready", "failed", "registered", "doi range"):
        table.add_column(column)
    for item in overview.batches[: limit or len(overview.batches)]:
        table.add_row(
            str(item.batch_number),
            item.status,
            f"{item.confirmed_count}/{item.total_count}",
            str(item.review_ready_count),
            str(item.failed_count),
            str(item.registered_count),
            f"{item.first_doi or '-'} -> {item.last_doi or '-'}",
        )
    console.print(
        f"total_batches={overview.total_batches}; current={overview.current_batch_index}; "
        f"confirmed_batches={overview.confirmed_batch_count}"
    )
    console.print(table)


@app.command("run-next-review-batch")
def run_next_review_batch(
    batch_size: int | None = typer.Option(None, "--batch-size"),
    include_failed_retries: bool = typer.Option(False, "--retry-failed"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    public_resolve: bool | None = typer.Option(None, "--public-resolve/--no-public-resolve"),
    identity_judge: bool | None = typer.Option(None, "--identity-judge/--no-identity-judge"),
    auto_decision: bool | None = typer.Option(None, "--auto-decision/--no-auto-decision"),
    material_ocsr: bool | None = typer.Option(None, "--material-ocsr/--no-material-ocsr"),
    write_report: bool = typer.Option(True, "--report/--no-report"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _batch_worker_service(config)
    overview = service.list_review_batches(batch_size=batch_size)
    target_batch_index = overview.current_batch_index
    result = service.run_next_review_batch(
        batch_size=batch_size,
        include_failed_retries=include_failed_retries,
        provider=provider,
        model=model,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_auto_decision=auto_decision,
        run_material_ocsr=material_ocsr,
    )
    console.print(
        f"processed={result.processed_count}; review_ready={result.review_ready_count}; "
        f"failed={result.failed_count}; skipped={result.skipped_count}"
    )
    _print_batch_jobs(result.jobs, title="Next Review Batch Results")
    if write_report and target_batch_index is not None:
        report = service.write_batch_report(
            batch_index=target_batch_index,
            batch_size=batch_size,
            run_result=result,
            run_options=_batch_run_options(
                service,
                provider=provider,
                model=model,
                retry_failed=include_failed_retries,
                public_resolve=public_resolve,
                identity_judge=identity_judge,
                auto_decision=auto_decision,
                material_ocsr=material_ocsr,
            ),
        )
        _print_batch_report_paths(report)


@app.command("run-review-batch")
def run_review_batch(
    batch_number: int = typer.Option(..., "--batch-number", help="1-based review batch number."),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    include_failed_retries: bool = typer.Option(False, "--retry-failed"),
    template_id: str = typer.Option("oled_device_v1", "--template-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    parse: bool = typer.Option(True, "--parse/--no-parse"),
    device_mining: bool = typer.Option(True, "--device-mining/--no-device-mining"),
    material_resolution: bool = typer.Option(
        True, "--material-resolution/--no-material-resolution"
    ),
    metadata_enrichment: bool | None = typer.Option(
        None,
        "--metadata-enrichment/--no-metadata-enrichment",
        help="Override batch_worker.run_metadata_enrichment.",
    ),
    public_resolve: bool | None = typer.Option(None, "--public-resolve/--no-public-resolve"),
    identity_judge: bool | None = typer.Option(None, "--identity-judge/--no-identity-judge"),
    auto_decision: bool | None = typer.Option(None, "--auto-decision/--no-auto-decision"),
    visual_prep: bool | None = typer.Option(None, "--visual-prep/--no-visual-prep"),
    material_ocsr: bool | None = typer.Option(None, "--material-ocsr/--no-material-ocsr"),
    write_report: bool = typer.Option(True, "--report/--no-report"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    if batch_number < 1:
        raise typer.BadParameter("--batch-number must be >= 1")
    batch_index = batch_number - 1
    service = _batch_worker_service(config)
    result = service.run_review_batch(
        batch_index=batch_index,
        batch_size=batch_size,
        include_failed_retries=include_failed_retries,
        template_id=template_id,
        provider=provider,
        model=model,
        run_parse=parse,
        run_device_mining=device_mining,
        run_material_resolution=material_resolution,
        run_metadata_enrichment=metadata_enrichment,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_auto_decision=auto_decision,
        run_visual_prep=visual_prep,
        run_material_ocsr=material_ocsr,
    )
    console.print(
        f"processed={result.processed_count}; review_ready={result.review_ready_count}; "
        f"failed={result.failed_count}; skipped={result.skipped_count}"
    )
    _print_batch_jobs(result.jobs, title=f"Review Batch {batch_number} Results")
    if write_report:
        report = service.write_batch_report(
            batch_index=batch_index,
            batch_size=batch_size,
            run_result=result,
            run_options=_batch_run_options(
                service,
                provider=provider,
                model=model,
                retry_failed=include_failed_retries,
                template_id=template_id,
                parse=parse,
                device_mining=device_mining,
                material_resolution=material_resolution,
                metadata_enrichment=metadata_enrichment,
                public_resolve=public_resolve,
                identity_judge=identity_judge,
                auto_decision=auto_decision,
                visual_prep=visual_prep,
                material_ocsr=material_ocsr,
            ),
        )
        _print_batch_report_paths(report)


@app.command("run-review-batch-staged")
def run_review_batch_staged(
    batch_number: int = typer.Option(..., "--batch-number", help="1-based review batch number."),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    include_failed_retries: bool = typer.Option(False, "--retry-failed"),
    template_id: str = typer.Option("oled_device_v1", "--template-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    parse_concurrency: int = typer.Option(1, "--parse-concurrency"),
    llm_concurrency: int = typer.Option(10, "--llm-concurrency"),
    material_concurrency: int = typer.Option(5, "--material-concurrency"),
    parse: bool = typer.Option(True, "--parse/--no-parse"),
    device_mining: bool = typer.Option(True, "--device-mining/--no-device-mining"),
    material_resolution: bool = typer.Option(
        True, "--material-resolution/--no-material-resolution"
    ),
    metadata_enrichment: bool | None = typer.Option(
        None,
        "--metadata-enrichment/--no-metadata-enrichment",
        help="Override batch_worker.run_metadata_enrichment.",
    ),
    public_resolve: bool | None = typer.Option(None, "--public-resolve/--no-public-resolve"),
    identity_judge: bool | None = typer.Option(None, "--identity-judge/--no-identity-judge"),
    auto_decision: bool | None = typer.Option(None, "--auto-decision/--no-auto-decision"),
    visual_prep: bool | None = typer.Option(None, "--visual-prep/--no-visual-prep"),
    material_ocsr: bool | None = typer.Option(None, "--material-ocsr/--no-material-ocsr"),
    write_report: bool = typer.Option(True, "--report/--no-report"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    if batch_number < 1:
        raise typer.BadParameter("--batch-number must be >= 1")
    for option_name, value in (
        ("--parse-concurrency", parse_concurrency),
        ("--llm-concurrency", llm_concurrency),
        ("--material-concurrency", material_concurrency),
    ):
        if value < 1:
            raise typer.BadParameter(f"{option_name} must be >= 1")
    batch_index = batch_number - 1
    service = _batch_worker_service(config)
    result = service.run_review_batch_staged(
        batch_index=batch_index,
        batch_size=batch_size,
        include_failed_retries=include_failed_retries,
        template_id=template_id,
        provider=provider,
        model=model,
        parse_concurrency=parse_concurrency,
        llm_concurrency=llm_concurrency,
        material_concurrency=material_concurrency,
        run_parse=parse,
        run_device_mining=device_mining,
        run_material_resolution=material_resolution,
        run_metadata_enrichment=metadata_enrichment,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_auto_decision=auto_decision,
        run_visual_prep=visual_prep,
        run_material_ocsr=material_ocsr,
    )
    console.print(
        f"processed={result.processed_count}; review_ready={result.review_ready_count}; "
        f"failed={result.failed_count}; skipped={result.skipped_count}"
    )
    _print_batch_jobs(result.jobs, title=f"Review Batch {batch_number} Staged Results")
    if write_report:
        report = service.write_batch_report(
            batch_index=batch_index,
            batch_size=batch_size,
            run_result=result,
            run_options=_batch_run_options(
                service,
                runner="staged",
                provider=provider,
                model=model,
                retry_failed=include_failed_retries,
                template_id=template_id,
                parse_concurrency=parse_concurrency,
                llm_concurrency=llm_concurrency,
                material_concurrency=material_concurrency,
                parse=parse,
                device_mining=device_mining,
                material_resolution=material_resolution,
                metadata_enrichment=metadata_enrichment,
                public_resolve=public_resolve,
                identity_judge=identity_judge,
                auto_decision=auto_decision,
                visual_prep=visual_prep,
                material_ocsr=material_ocsr,
            ),
        )
        _print_batch_report_paths(report)


@app.command("refresh-review-batch-materials")
def refresh_review_batch_materials(
    batch_number: int = typer.Option(..., "--batch-number", help="1-based review batch number."),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    material_concurrency: int = typer.Option(5, "--material-concurrency"),
    metadata_enrichment: bool | None = typer.Option(
        None,
        "--metadata-enrichment/--no-metadata-enrichment",
        help="Fill missing title/journal/publisher/year before Stage 3.",
    ),
    public_concurrency: int | None = typer.Option(None, "--public-concurrency"),
    judge_concurrency: int | None = typer.Option(None, "--judge-concurrency"),
    vlm_concurrency: int | None = typer.Option(None, "--vlm-concurrency"),
    ocsr_paper_concurrency: int | None = typer.Option(
        None,
        "--ocsr-paper-concurrency",
    ),
    decimer_segmentation_concurrency: int | None = typer.Option(
        None,
        "--decimer-segmentation-concurrency",
    ),
    decimer_ocsr_concurrency: int | None = typer.Option(
        None,
        "--decimer-ocsr-concurrency",
    ),
    material_resolution: bool = typer.Option(
        True,
        "--material-resolution/--no-material-resolution",
    ),
    public_resolve: bool | None = typer.Option(None, "--public-resolve/--no-public-resolve"),
    identity_judge: bool | None = typer.Option(None, "--identity-judge/--no-identity-judge"),
    auto_decision: bool | None = typer.Option(None, "--auto-decision/--no-auto-decision"),
    visual_prep: bool | None = typer.Option(None, "--visual-prep/--no-visual-prep"),
    material_ocsr: bool | None = typer.Option(None, "--material-ocsr/--no-material-ocsr"),
    write_report: bool = typer.Option(True, "--report/--no-report"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    if batch_number < 1:
        raise typer.BadParameter("--batch-number must be >= 1")
    if material_concurrency < 1:
        raise typer.BadParameter("--material-concurrency must be >= 1")
    for option_name, option_value in (
        ("--public-concurrency", public_concurrency),
        ("--judge-concurrency", judge_concurrency),
        ("--vlm-concurrency", vlm_concurrency),
        ("--ocsr-paper-concurrency", ocsr_paper_concurrency),
        ("--decimer-segmentation-concurrency", decimer_segmentation_concurrency),
        ("--decimer-ocsr-concurrency", decimer_ocsr_concurrency),
    ):
        if option_value is not None and option_value < 1:
            raise typer.BadParameter(f"{option_name} must be >= 1")
    batch_index = batch_number - 1
    service = _batch_worker_service(config)
    result = service.refresh_review_batch_materials(
        batch_index=batch_index,
        batch_size=batch_size,
        provider=provider,
        model=model,
        material_concurrency=material_concurrency,
        run_metadata_enrichment=metadata_enrichment,
        run_material_resolution=material_resolution,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_auto_decision=auto_decision,
        run_visual_prep=visual_prep,
        run_material_ocsr=material_ocsr,
        public_concurrency=public_concurrency,
        judge_concurrency=judge_concurrency,
        vlm_concurrency=vlm_concurrency,
        ocsr_paper_concurrency=ocsr_paper_concurrency,
        decimer_segmentation_concurrency=decimer_segmentation_concurrency,
        decimer_ocsr_concurrency=decimer_ocsr_concurrency,
    )
    if result.material_metrics:
        console.print("material_metrics=" + json.dumps(result.material_metrics, ensure_ascii=False))
    console.print(
        f"processed={result.processed_count}; review_ready={result.review_ready_count}; "
        f"failed={result.failed_count}; skipped={result.skipped_count}"
    )
    _print_batch_jobs(result.jobs, title=f"Review Batch {batch_number} Material Refresh")
    if write_report:
        report = service.write_batch_report(
            batch_index=batch_index,
            batch_size=batch_size,
            run_result=result,
            run_options=_batch_run_options(
                service,
                runner="material_refresh",
                provider=provider,
                model=model,
                material_concurrency=material_concurrency,
                public_concurrency=(
                    public_concurrency or service.config.batch_worker.material_public_concurrency
                ),
                judge_concurrency=(
                    judge_concurrency or service.config.batch_worker.material_judge_concurrency
                ),
                vlm_concurrency=(
                    vlm_concurrency or service.config.batch_worker.material_vlm_concurrency
                ),
                ocsr_paper_concurrency=(
                    ocsr_paper_concurrency
                    or service.config.batch_worker.material_ocsr_paper_concurrency
                ),
                decimer_segmentation_concurrency=(
                    decimer_segmentation_concurrency
                    or service.config.batch_worker.material_decimer_segmentation_concurrency
                ),
                decimer_ocsr_concurrency=(
                    decimer_ocsr_concurrency
                    or service.config.batch_worker.material_decimer_ocsr_concurrency
                ),
                metadata_enrichment=metadata_enrichment,
                material_resolution=material_resolution,
                public_resolve=public_resolve,
                identity_judge=identity_judge,
                auto_decision=auto_decision,
                visual_prep=visual_prep,
                material_ocsr=material_ocsr,
            ),
        )
        _print_batch_report_paths(report)


@app.command("run-stage3-batch-group")
def run_stage3_batch_group(
    start_batch: int = typer.Option(..., "--start-batch", help="First 1-based batch number."),
    end_batch: int | None = typer.Option(
        None,
        "--end-batch",
        help="Last 1-based batch number; defaults to the final review batch.",
    ),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    batch_concurrency: int = typer.Option(15, "--batch-concurrency"),
    material_concurrency: int = typer.Option(2, "--material-concurrency"),
    material_resolution: bool = typer.Option(
        True,
        "--material-resolution/--no-material-resolution",
        help="Refresh local material links before public resolution.",
    ),
    public_concurrency: int | None = typer.Option(None, "--public-concurrency"),
    judge_concurrency: int | None = typer.Option(None, "--judge-concurrency"),
    vlm_concurrency: int | None = typer.Option(None, "--vlm-concurrency"),
    ocsr_paper_concurrency: int | None = typer.Option(None, "--ocsr-paper-concurrency"),
    decimer_segmentation_concurrency: int | None = typer.Option(
        None,
        "--decimer-segmentation-concurrency",
    ),
    decimer_ocsr_concurrency: int | None = typer.Option(
        None,
        "--decimer-ocsr-concurrency",
    ),
    resume: bool = typer.Option(True, "--resume/--fresh"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    for option_name, option_value in (
        ("--start-batch", start_batch),
        ("--batch-concurrency", batch_concurrency),
        ("--material-concurrency", material_concurrency),
    ):
        if option_value < 1:
            raise typer.BadParameter(f"{option_name} must be >= 1")
    if end_batch is not None and end_batch < start_batch:
        raise typer.BadParameter("--end-batch must be >= --start-batch")
    for option_name, option_value in (
        ("--public-concurrency", public_concurrency),
        ("--judge-concurrency", judge_concurrency),
        ("--vlm-concurrency", vlm_concurrency),
        ("--ocsr-paper-concurrency", ocsr_paper_concurrency),
        ("--decimer-segmentation-concurrency", decimer_segmentation_concurrency),
        ("--decimer-ocsr-concurrency", decimer_ocsr_concurrency),
    ):
        if option_value is not None and option_value < 1:
            raise typer.BadParameter(f"{option_name} must be >= 1")

    loaded_config = load_config(config)
    configure_logging(loaded_config.logging.level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    service = Stage3BatchGroupService(
        loaded_config,
        progress=lambda message: console.print(message, markup=False),
    )
    try:
        report = service.run(
            start_batch=start_batch,
            end_batch=end_batch,
            batch_size=batch_size,
            batch_concurrency=batch_concurrency,
            material_concurrency=material_concurrency,
            run_material_resolution=material_resolution,
            public_concurrency=public_concurrency,
            judge_concurrency=judge_concurrency,
            vlm_concurrency=vlm_concurrency,
            ocsr_paper_concurrency=ocsr_paper_concurrency,
            decimer_segmentation_concurrency=decimer_segmentation_concurrency,
            decimer_ocsr_concurrency=decimer_ocsr_concurrency,
            resume=resume,
        )
    finally:
        service.close()
    console.print(
        f"completed_batches={report['completed_batch_count']}; "
        f"failed_batches={report['failed_batch_count']}; "
        f"processed_papers={report['processed_paper_count']}; "
        f"review_ready={report['review_ready_paper_count']}"
    )
    console.print(f"State: {report['state_path']}")
    console.print(f"JSON report: {report['json_path']}")
    console.print(f"Markdown report: {report['markdown_path']}")


@app.command("write-batch-report")
def write_batch_report(
    batch_number: int = typer.Option(..., "--batch-number", help="1-based review batch number."),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    if batch_number < 1:
        raise typer.BadParameter("--batch-number must be >= 1")
    service = _batch_worker_service(config)
    report = service.write_batch_report(batch_index=batch_number - 1, batch_size=batch_size)
    _print_batch_report_paths(report)


@app.command("run-batch-worker-once")
def run_batch_worker_once(
    max_jobs: int = typer.Option(1, "--max-jobs"),
    include_failed_retries: bool = typer.Option(False, "--retry-failed"),
    template_id: str = typer.Option("oled_device_v1", "--template-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    parse: bool = typer.Option(True, "--parse/--no-parse"),
    device_mining: bool = typer.Option(True, "--device-mining/--no-device-mining"),
    material_resolution: bool = typer.Option(
        True, "--material-resolution/--no-material-resolution"
    ),
    metadata_enrichment: bool | None = typer.Option(
        None,
        "--metadata-enrichment/--no-metadata-enrichment",
        help="Override batch_worker.run_metadata_enrichment.",
    ),
    public_resolve: bool | None = typer.Option(
        None,
        "--public-resolve/--no-public-resolve",
        help="Override batch_worker.run_public_resolver.",
    ),
    identity_judge: bool | None = typer.Option(
        None,
        "--identity-judge/--no-identity-judge",
        help="Override batch_worker.run_identity_judge.",
    ),
    auto_decision: bool | None = typer.Option(
        None,
        "--auto-decision/--no-auto-decision",
        help="Override batch_worker.run_material_auto_decision.",
    ),
    visual_prep: bool | None = typer.Option(
        None,
        "--visual-prep/--no-visual-prep",
        help="Override batch_worker.run_visual_prep.",
    ),
    material_ocsr: bool | None = typer.Option(
        None,
        "--material-ocsr/--no-material-ocsr",
        help="Override batch_worker.run_material_ocsr.",
    ),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _batch_worker_service(config)
    result = service.run_once(
        max_jobs=max_jobs,
        include_failed_retries=include_failed_retries,
        template_id=template_id,
        provider=provider,
        model=model,
        run_parse=parse,
        run_device_mining=device_mining,
        run_material_resolution=material_resolution,
        run_metadata_enrichment=metadata_enrichment,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_auto_decision=auto_decision,
        run_visual_prep=visual_prep,
        run_material_ocsr=material_ocsr,
    )
    console.print(
        f"processed={result.processed_count}; review_ready={result.review_ready_count}; "
        f"failed={result.failed_count}; skipped={result.skipped_count}"
    )
    _print_batch_jobs(result.jobs, title="Batch Worker Results")


@app.command("watch-inbox-pdfs")
def watch_inbox_pdfs(
    interval_seconds: float | None = typer.Option(None, "--interval-seconds"),
    max_cycles: int | None = typer.Option(None, "--max-cycles"),
    max_jobs_per_cycle: int = typer.Option(1, "--max-jobs-per-cycle"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    metadata_enrichment: bool | None = typer.Option(
        None, "--metadata-enrichment/--no-metadata-enrichment"
    ),
    public_resolve: bool | None = typer.Option(None, "--public-resolve/--no-public-resolve"),
    identity_judge: bool | None = typer.Option(None, "--identity-judge/--no-identity-judge"),
    auto_decision: bool | None = typer.Option(None, "--auto-decision/--no-auto-decision"),
    visual_prep: bool | None = typer.Option(None, "--visual-prep/--no-visual-prep"),
    material_ocsr: bool | None = typer.Option(None, "--material-ocsr/--no-material-ocsr"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _batch_worker_service(config)
    results = service.watch(
        interval_seconds=interval_seconds,
        max_cycles=max_cycles,
        max_jobs_per_cycle=max_jobs_per_cycle,
        provider=provider,
        model=model,
        run_metadata_enrichment=metadata_enrichment,
        run_public_resolver=public_resolve,
        run_identity_judge=identity_judge,
        run_material_auto_decision=auto_decision,
        run_visual_prep=visual_prep,
        run_material_ocsr=material_ocsr,
    )
    processed = sum(item.processed_count for item in results)
    ready = sum(item.review_ready_count for item in results)
    failed = sum(item.failed_count for item in results)
    console.print(
        f"cycles={len(results)}; processed={processed}; review_ready={ready}; failed={failed}"
    )


def _batch_run_options(service: BatchWorkerService, **options: object) -> dict[str, object]:
    resolved = dict(options)
    default_flags = {
        "metadata_enrichment": "run_metadata_enrichment",
        "public_resolve": "run_public_resolver",
        "identity_judge": "run_identity_judge",
        "auto_decision": "run_material_auto_decision",
        "visual_prep": "run_visual_prep",
        "material_ocsr": "run_material_ocsr",
    }
    for option_key, config_key in default_flags.items():
        if option_key in resolved and resolved[option_key] is None:
            resolved[option_key] = getattr(service.config.batch_worker, config_key)
    return {key: value for key, value in resolved.items() if value is not None}


def _print_batch_report_paths(report) -> None:
    console.print(f"Batch report JSON: {report.json_path}")
    console.print(f"Batch report Markdown: {report.markdown_path}")


def _print_batch_jobs(jobs, *, title: str) -> None:
    table = Table(title=title)
    for column in ("paper_id", "status", "stage", "retry", "parse/mining", "error"):
        table.add_column(column)
    for job in jobs:
        timings = ", ".join(
            f"{key}={value}s" for key, value in sorted(job.stage_timings.items())[-2:]
        )
        table.add_row(
            job.paper_id,
            job.status,
            job.current_stage,
            f"{job.retry_count}/{job.max_retries}",
            timings,
            (job.error_message or "")[:80],
        )
    console.print(table)


@app.command("list-papers")
def list_papers(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    papers = _service(config).list_papers()
    table = Table(title="Mining Platform Papers")
    for column in ("paper_id", "doi", "journal", "domain", "review_status", "pdf_size_bytes"):
        table.add_column(column)
    for paper in papers:
        table.add_row(
            paper.paper_id,
            paper.doi,
            paper.journal or "",
            paper.domain,
            paper.review_status,
            str(paper.pdf_size_bytes),
        )
    console.print(table)


@app.command("enrich-paper-metadata")
def enrich_paper_metadata(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing metadata fields."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _paper_metadata_service(config).enrich_paper(paper_id, force=force)
    if result.status == "not_found" and result.error_message:
        raise typer.BadParameter(result.error_message)
    table = Table(title="Paper Metadata Enrichment")
    for column in ("paper_id", "status", "source", "journal", "year", "updated"):
        table.add_column(column)
    table.add_row(
        result.paper_id,
        result.status,
        result.source or "",
        result.journal or "",
        str(result.year or ""),
        ", ".join(result.updated_fields),
    )
    console.print(table)
    if result.error_message:
        console.print(f"[yellow]{result.error_message}[/yellow]")


@app.command("enrich-missing-paper-metadata")
def enrich_missing_paper_metadata(
    limit: int | None = typer.Option(None, "--limit"),
    force: bool = typer.Option(
        False, "--force", help="Refresh every paper, not only missing fields."
    ),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    results = _paper_metadata_service(config).enrich_missing(limit=limit, force=force)
    table = Table(title="Missing Paper Metadata Enrichment")
    for column in ("paper_id", "status", "source", "journal", "year", "updated"):
        table.add_column(column)
    for result in results:
        table.add_row(
            result.paper_id,
            result.status,
            result.source or "",
            result.journal or "",
            str(result.year or ""),
            ", ".join(result.updated_fields),
        )
    console.print(table)
    console.print(f"processed={len(results)}")


@app.command("parse-paper")
def parse_paper(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _parse_service(config)
    result = service.parse_paper(paper_id)
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Parsed {result.paper_id}: {result.page_count} page(s), {result.block_count} block(s)."
    )
    console.print(f"Document: {result.document_path}")
    console.print(f"Blocks: {result.blocks_path}")


@app.command("parse-all-papers")
def parse_all_papers(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    results = _parse_service(config).parse_all_papers()
    table = Table(title="PDF Parse Results")
    for column in ("paper_id", "status", "pages", "blocks", "error"):
        table.add_column(column)
    for result in results:
        table.add_row(
            result.paper_id,
            result.status,
            str(result.page_count),
            str(result.block_count),
            result.error_message or "",
        )
    console.print(table)


@app.command("extract-oled")
def extract_oled(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _extraction_service(config)
    result = service.extract_oled(paper_id)
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Extraction run {result.run.run_id}: {result.run.status}; "
        f"{len(result.raw_records)} raw candidate(s)."
    )
    for record in result.raw_records:
        console.print(
            f"- {record.raw_record_id}: {record.device_label or 'unlabeled'} "
            f"EQE={record.eqe_max or ''} page={record.evidence_page or ''}"
        )


@app.command("extract-oled-all")
def extract_oled_all(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _extraction_service(config)
    papers = service.paper_service.list_papers()
    table = Table(title="OLED Raw Extraction Results")
    for column in ("paper_id", "status", "raw_records", "error"):
        table.add_column(column)
    for paper in papers:
        try:
            result = service.extract_oled(paper.paper_id)
        except Exception as exc:
            table.add_row(paper.paper_id, "failed", "0", str(exc))
            continue
        table.add_row(
            paper.paper_id,
            result.run.status if result else "missing",
            str(len(result.raw_records) if result else 0),
            "",
        )
    console.print(table)


@app.command("seed-candidate-fields")
def seed_candidate_fields(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _candidate_review_service(config)
    result = service.seed_candidate_fields_from_raw(paper_id)
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Seeded {result.field_count} candidate field(s) and "
        f"{result.evidence_count} evidence anchor(s) from "
        f"{result.raw_record_count} raw record(s) for {result.paper_id}."
    )


@app.command("validate-template")
def validate_template(
    template_id: str = typer.Option("oled_device_v1", "--template-id", help="Domain template ID."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    service = _template_service(config)
    template = service.get_template(template_id)
    console.print(
        f"Template {template.template_id} ({template.domain}, {template.version}) is valid."
    )
    console.print(f"Source: {template.source_path}")
    console.print(f"Required output keys: {', '.join(template.required_output_keys)}")
    console.print(f"Fields: {len(template.fields)}")
    console.print(f"Vocabularies: {len(template.vocabularies)}")


@app.command("validate-mining-result")
def validate_mining_result(
    input_path: Path = typer.Option(..., "--input", "-i", help="Mining result JSON file."),
    template_id: str = typer.Option("oled_device_v1", "--template-id", help="Domain template ID."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    report = _template_service(config).validate_mining_result(template_id, payload)
    status = "valid" if report.valid else "invalid"
    console.print(
        f"Mining result is {status}: {report.fatal_count} fatal error(s), "
        f"{report.repairable_count} repairable issue(s), "
        f"{report.warning_count} warning(s)."
    )
    if report.errors:
        table = Table(title="Mining Result Fatal Validation Errors")
        for column in ("code", "path", "field_path", "message"):
            table.add_column(column)
        for issue in report.errors:
            table.add_row(issue.code, issue.path, issue.field_path or "", issue.message)
        console.print(table)
        raise typer.Exit(code=1)


@app.command("ingest-mining-result")
def ingest_mining_result(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    input_path: Path = typer.Option(..., "--input", "-i", help="Mining result JSON file."),
    template_id: str = typer.Option("oled_device_v1", "--template-id", help="Domain template ID."),
    source_name: str = typer.Option("mock", "--source-name", help="Mining result source name."),
    source_version: str | None = typer.Option(None, "--source-version"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("Mining result JSON must be an object.")
    result = _candidate_ingestion_service(config).ingest_mining_result(
        paper_id=paper_id,
        template_id=template_id,
        payload=payload,
        source_name=source_name,
        source_version=source_version,
    )
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Ingestion run {result.run.candidate_run_id}: {result.run.status}; "
        f"{result.entity_count} entit(ies), {result.value_count} value(s), "
        f"{result.evidence_anchor_count} evidence anchor(s)."
    )
    if not result.validation_report.valid:
        for issue in result.validation_report.errors:
            console.print(f"- {issue.code}: {issue.path} {issue.message}")
        raise typer.Exit(code=1)


@app.command("parse-paper-mineru")
def parse_paper_mineru(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _mineru_parse_service(config).parse_paper(paper_id)
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"MinerU run {result.mineru_run_id}: {result.status}; "
        f"{result.content_item_count} content item(s)."
    )
    console.print(f"Content list: {result.content_list_path or ''}")


@app.command("mine-paper-llm")
def mine_paper_llm(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    template_id: str = typer.Option("oled_device_v1", "--template-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    mock_input: Path | None = typer.Option(
        None,
        "--mock-input",
        help="Use a local mining_result JSON instead of calling a commercial LLM.",
    ),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _llm_mining_service(config, mock_input=mock_input).mine_paper(
        paper_id=paper_id,
        template_id=template_id,
        provider=provider,
        model=model,
    )
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"LLM mining run {result.run.llm_run_id}: {result.run.status}; "
        f"candidate_run={result.candidate_run_id or ''}"
    )
    console.print(f"Mining result: {result.run.mining_result_path or ''}")
    console.print(f"Validation report: {result.run.validation_report_path or ''}")
    if result.run.status != "completed":
        raise typer.Exit(code=1)


@app.command("mine-paper-pipeline")
def mine_paper_pipeline(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    template_id: str = typer.Option("oled_device_v1", "--template-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _llm_mining_service(config).mine_paper_pipeline(
        paper_id=paper_id,
        template_id=template_id,
        provider=provider,
        model=model,
    )
    if not result:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Pipeline LLM mining run {result.run.llm_run_id}: {result.run.status}; "
        f"candidate_run={result.candidate_run_id or ''}"
    )
    if result.run.status != "completed":
        raise typer.Exit(code=1)


@app.command("confirm-review-v2")
def confirm_review_v2(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    actor: str = typer.Option("local_user", "--actor"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _candidate_ingestion_service(config).confirm_review_v2(paper_id, actor=actor)
    if not result:
        raise typer.BadParameter(f"Completed candidate v2 run not found: {paper_id}")
    console.print(
        f"Confirmed {result.paper_id}: final record "
        f"{result.final_record.final_record_id}; {result.final_value_count} value(s)."
    )


@app.command("seed-common-materials")
def seed_common_materials(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    materials = _material_resolution_service(config).seed_common_oled_materials()
    table = Table(title="Seeded Common OLED Materials")
    for column in ("global_material_id", "canonical_name", "class", "status"):
        table.add_column(column)
    for material in materials:
        table.add_row(
            material.global_material_id,
            material.canonical_name or "",
            material.material_class,
            material.review_status,
        )
    console.print(table)


@app.command("resolve-paper-materials")
def resolve_paper_materials(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_resolution_service(config).resolve_paper_materials(paper_id)
    if not bundle:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Resolution: {bundle.paper_id}")
    for column in ("paper_material_id", "mentions", "status", "method", "global_material_id"):
        table.add_column(column)
    links_by_material = {link.paper_material_id: link for link in bundle.links}
    for material in bundle.materials:
        link = links_by_material.get(material.paper_material_id)
        table.add_row(
            material.paper_material_id,
            ", ".join(material.mention_list[:3]),
            link.match_status if link else "not_resolved",
            link.match_method if link else "",
            link.global_material_id if link and link.global_material_id else "",
        )
    console.print(table)
    console.print(f"Tasks: {len(bundle.tasks)}")


@app.command("mine-material-properties")
def mine_material_properties(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    paper_material_id: str | None = typer.Option(None, "--paper-material-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    mock_input: Path | None = typer.Option(
        None,
        "--mock-input",
        help="Use a local material_properties JSON instead of calling a commercial LLM.",
    ),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    try:
        result = _material_property_mining_service(
            config, mock_input=mock_input
        ).mine_paper_properties(
            paper_id,
            paper_material_id=paper_material_id,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Property Candidates: {result.paper_id}")
    for column in (
        "paper_material_id",
        "property",
        "value",
        "unit",
        "source",
        "confidence",
        "status",
    ):
        table.add_column(column)
    for candidate in result.candidates:
        value = (
            str(candidate.value_numeric)
            if candidate.value_numeric is not None
            else candidate.value_text or candidate.value_raw or ""
        )
        table.add_row(
            candidate.paper_material_id,
            candidate.property_name,
            value,
            candidate.unit or "",
            candidate.source_type,
            "" if candidate.confidence is None else f"{candidate.confidence:.2f}",
            candidate.status,
        )
    console.print(table)
    candidate_run_id = result.candidate_run_id or ""
    console.print(
        f"candidate_run={candidate_run_id}; "
        f"sources={result.source_count}; stored={len(result.candidates)}; skipped={len(result.skipped)}"
    )
    for skipped in result.skipped:
        console.print(f"[yellow]skipped[/yellow] {json.dumps(skipped, ensure_ascii=False)}")


@app.command("validate-material-names")
def validate_material_names(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    paper_material_id: str | None = typer.Option(None, "--paper-material-id"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_resolution_service(config).validate_material_names(
        paper_id,
        paper_material_id=paper_material_id,
    )
    if not bundle:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Name Agent Suggestions: {bundle.paper_id}")
    for column in ("paper_material_id", "original", "suggested", "confidence", "reason"):
        table.add_column(column)
    for suggestion in bundle.material_name_suggestions:
        if paper_material_id and suggestion.paper_material_id != paper_material_id:
            continue
        table.add_row(
            suggestion.paper_material_id,
            suggestion.original_name or "",
            suggestion.suggested_name,
            "" if suggestion.confidence is None else f"{suggestion.confidence:.2f}",
            suggestion.reason or "",
        )
    console.print(table)


@app.command("review-material-name")
def review_material_name(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    paper_material_id: str = typer.Option(..., "--paper-material-id"),
    name: str = typer.Option(..., "--name", help="Corrected primary material name."),
    abbreviation: str | None = typer.Option(None, "--abbreviation"),
    full_name: str | None = typer.Option(None, "--full-name"),
    normalized_name: str | None = typer.Option(None, "--normalized-name"),
    canonical_name: str | None = typer.Option(None, "--canonical-name"),
    actor: str = typer.Option("local_user", "--actor"),
    message: str | None = typer.Option(None, "--message"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_resolution_service(config).review_paper_material_name(
        paper_id,
        paper_material_id,
        PaperMaterialNameReviewAction(
            actor=actor,
            message=message,
            reviewed_name=name,
            reviewed_abbreviation=abbreviation or name,
            reviewed_full_name_in_paper=full_name,
            reviewed_normalized_name=normalized_name,
            reviewed_canonical_name=canonical_name,
        ),
    )
    if not bundle:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    material = next(
        item for item in bundle.materials if item.paper_material_id == paper_material_id
    )
    console.print(
        f"Reviewed {paper_material_id}: {material.paper_material_id} -> {material.abbreviation or material.entity_label}"
    )


@app.command("list-material-tasks")
def list_material_tasks(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    tasks = _material_resolution_service(config).list_resolution_tasks()
    table = Table(title="Material Resolution Tasks")
    for column in ("paper_id", "paper_material_id", "priority", "status", "strategy"):
        table.add_column(column)
    for task in tasks:
        table.add_row(
            task.paper_id,
            task.paper_material_id,
            task.priority,
            task.status,
            task.assigned_strategy,
        )
    console.print(table)


@app.command("resolve-public-materials")
def resolve_public_materials(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    paper_material_id: str | None = typer.Option(None, "--paper-material-id"),
    max_queries_per_material: int = typer.Option(2, "--max-queries-per-material"),
    max_results_per_query: int | None = typer.Option(None, "--max-results-per-query"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_public_resolver_service(config).resolve_paper_public(
        paper_id,
        paper_material_id=paper_material_id,
        max_queries_per_material=max_queries_per_material,
        max_results_per_query=max_results_per_query,
    )
    if not bundle:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Public Material Candidates: {bundle.paper_id}")
    for column in ("paper_material_id", "provider", "query", "candidate", "confidence", "status"):
        table.add_column(column)
    for candidate in bundle.structure_candidates:
        if paper_material_id and candidate.paper_material_id != paper_material_id:
            continue
        table.add_row(
            candidate.paper_material_id,
            candidate.provider,
            candidate.query_text,
            candidate.canonical_name or candidate.source_identifier or "",
            f"{candidate.confidence:.2f}" if candidate.confidence is not None else "",
            candidate.status,
        )
    console.print(table)
    console.print(f"Candidates: {len(bundle.structure_candidates)}; Tasks: {len(bundle.tasks)}")


@app.command("auto-decide-materials")
def auto_decide_materials(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    accept_min_confidence: float | None = typer.Option(None, "--accept-min-confidence"),
    reject_min_confidence: float | None = typer.Option(None, "--reject-min-confidence"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_auto_decision_service(config).apply_paper_auto_decisions(
        paper_id,
        dry_run=dry_run,
        accept_min_confidence=accept_min_confidence,
        reject_min_confidence=reject_min_confidence,
    )
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Auto Decisions: {result.paper_id}")
    for column in ("material", "candidate", "action", "applied", "confidence", "reason", "error"):
        table.add_column(column)
    for decision in result.decisions:
        table.add_row(
            decision.paper_material_id,
            (decision.structure_candidate_id or "")[:12],
            decision.action,
            "yes" if decision.applied else "no",
            f"{decision.confidence:.2f}" if decision.confidence is not None else "",
            decision.reason,
            decision.error_message or "",
        )
    console.print(
        f"accepted={result.accepted_count}; rejected={result.rejected_count}; "
        f"skipped={result.skipped_count}; dry_run={result.dry_run}"
    )
    console.print(table)


@app.command("accept-material-candidate")
def accept_material_candidate(
    structure_candidate_id: str = typer.Option(..., "--structure-candidate-id"),
    actor: str = typer.Option("local_user", "--actor"),
    message: str | None = typer.Option(None, "--message"),
    global_material_id: str | None = typer.Option(None, "--global-material-id"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_structure_review_service(config).accept_structure_candidate(
        structure_candidate_id,
        MaterialReviewAction(
            actor=actor,
            message=message,
            global_material_id=global_material_id,
        ),
    )
    if not bundle:
        raise typer.BadParameter(
            f"Material structure candidate not found: {structure_candidate_id}"
        )
    console.print(
        f"Accepted {structure_candidate_id}; "
        f"{len(bundle.global_materials)} linked global material(s)."
    )


@app.command("apply-material-web-rescue")
def apply_material_web_rescue(
    decisions: Path = typer.Option(..., "--decisions", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    """Validate and apply externally researched material structures with audit history."""
    result = _material_web_rescue_service(config).apply_decision_file(
        decisions,
        dry_run=dry_run,
    )
    table = Table(title=f"Material Web Rescue: {result.run_id}")
    for column in ("paper", "material", "requested", "status", "InChIKey", "message"):
        table.add_column(column)
    for item in result.items:
        table.add_row(
            item.paper_id,
            item.paper_material_id,
            item.requested_action,
            item.status,
            item.inchi_key or "",
            item.message,
        )
    console.print(table)
    console.print(
        f"accepted={result.accepted_count}; stored={result.candidate_stored_count}; "
        f"unresolved={result.unresolved_count}; skipped={result.skipped_count}; "
        f"failed={result.failed_count}; dry_run={result.dry_run}"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"Result: {output}")
    if result.failed_count:
        raise typer.Exit(code=1)


@app.command("report-material-web-rescue")
def report_material_web_rescue(
    inventory: Path = typer.Option(..., "--inventory", exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    apply_result: Path | None = typer.Option(
        None,
        "--apply-result",
        exists=True,
        dir_okay=False,
    ),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    """Generate current JSON and Markdown reports for a frozen rescue inventory."""
    service = MaterialWebRescueReportService(load_config(config))
    report = service.generate(
        inventory,
        output_dir,
        apply_result_path=apply_result,
    )
    table = Table(title=f"Material Web Rescue Report: {report['run_id']}")
    table.add_column("status")
    table.add_column("count", justify="right")
    for status, count in report["summary"].items():
        table.add_row(status, str(count))
    console.print(table)
    console.print(f"audit_errors={report['audit_error_count']}")
    console.print(f"JSON: {output_dir / 'material-web-rescue-report.json'}")
    console.print(f"Markdown: {output_dir / 'material-web-rescue-report.md'}")


@app.command("judge-material-identities")
def judge_material_identities(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    paper_material_id: str | None = typer.Option(None, "--paper-material-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_identity_judge_service(config).judge_paper_candidates(
        paper_id,
        paper_material_id=paper_material_id,
        provider=provider,
        model=model,
    )
    if not bundle:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Identity Judgments: {bundle.paper_id}")
    for column in ("paper_material_id", "candidate", "verdict", "confidence", "action", "status"):
        table.add_column(column)
    for judgment in bundle.identity_judgments:
        if paper_material_id and judgment.paper_material_id != paper_material_id:
            continue
        table.add_row(
            judgment.paper_material_id,
            judgment.structure_candidate_id[:12],
            judgment.verdict,
            f"{judgment.confidence:.2f}" if judgment.confidence is not None else "",
            judgment.recommended_action,
            judgment.status,
        )
    console.print(table)


@app.command("enrich-material-identity")
def enrich_material_identity(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    paper_material_id: str = typer.Option(..., "--paper-material-id"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_identity_evidence_service(config).enrich_material_identity(
        paper_id,
        paper_material_id=paper_material_id,
        provider=provider,
        model=model,
    )
    if not bundle:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Identity Evidence: {bundle.paper_id} / {paper_material_id}")
    for column in ("tier", "status", "alias", "full_name", "CAS", "source"):
        table.add_column(column)
    for item in bundle.identity_evidence_items:
        if item.paper_material_id != paper_material_id:
            continue
        table.add_row(
            item.source_tier,
            item.review_status,
            item.alias or "",
            item.full_name or "",
            item.cas_number or "",
            item.source_title or item.source_url or "",
        )
    console.print(table)


@app.command("reject-material-candidate")
def reject_material_candidate(
    structure_candidate_id: str = typer.Option(..., "--structure-candidate-id"),
    actor: str = typer.Option("local_user", "--actor"),
    message: str | None = typer.Option(None, "--message"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_structure_review_service(config).reject_structure_candidate(
        structure_candidate_id,
        MaterialReviewAction(actor=actor, message=message),
    )
    if not bundle:
        raise typer.BadParameter(
            f"Material structure candidate not found: {structure_candidate_id}"
        )
    console.print(f"Rejected {structure_candidate_id}; events={len(bundle.material_review_events)}")


@app.command("undo-material-review-event")
def undo_material_review_event(
    event_id: str = typer.Option(..., "--event-id"),
    actor: str = typer.Option("local_user", "--actor"),
    message: str | None = typer.Option(None, "--message"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_structure_review_service(config).undo_material_review_event(
        event_id,
        MaterialReviewAction(actor=actor, message=message),
    )
    if not bundle:
        raise typer.BadParameter(f"Material review event not found: {event_id}")
    console.print(f"Undone {event_id}; events={len(bundle.material_review_events)}")


@app.command("collect-chemical-figures")
def collect_chemical_figures(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    blocks = _chemical_figure_collector_service(config).collect_for_paper(paper_id)
    if blocks is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_chemical_figure_table(blocks, title=f"Chemical Figure Blocks: {paper_id}")


@app.command("list-chemical-figures")
def list_chemical_figures(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    blocks = _chemical_figure_collector_service(config).list_for_paper(paper_id)
    if blocks is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_chemical_figure_table(blocks, title=f"Chemical Figure Blocks: {paper_id}")


def _print_chemical_figure_table(blocks, *, title: str) -> None:
    table = Table(title=title)
    for column in ("figure_block_id", "page", "type", "sub_type", "tags", "conf", "image"):
        table.add_column(column)
    for block in blocks:
        table.add_row(
            block.figure_block_id,
            str(block.page_id or ""),
            block.content_type,
            block.sub_type or "",
            ", ".join(block.heuristic_tags[:3]),
            f"{block.confidence:.2f}" if block.confidence is not None else "",
            "yes" if block.image_exists else "missing",
        )
    console.print(table)
    console.print(f"Blocks: {len(blocks)}")


@app.command("run-material-agent-foundation")
def run_material_agent_foundation(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_foundation(paper_id)
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Material agent run {result.run.agent_run_id}: {result.run.status}; "
        f"materials={result.run.material_count}; visual_blocks={len(result.visual_blocks)}"
    )
    _print_document_visual_block_table(
        result.visual_blocks, title=f"Document Visual Blocks: {paper_id}"
    )


@app.command("list-material-agent-runs")
def list_material_agent_runs(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    runs = _material_structure_agent_service(config).list_runs(paper_id)
    if runs is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material Agent Runs: {paper_id}")
    for column in ("agent_run_id", "status", "strategy", "materials", "visual_blocks", "created"):
        table.add_column(column)
    for run in runs:
        table.add_row(
            run.agent_run_id,
            run.status,
            run.strategy,
            str(run.material_count),
            str(run.visual_block_count),
            run.created_at,
        )
    console.print(table)


@app.command("list-document-visual-blocks")
def list_document_visual_blocks(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    blocks = _material_structure_agent_service(config).list_visual_blocks(paper_id)
    if blocks is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_document_visual_block_table(blocks, title=f"Document Visual Blocks: {paper_id}")


@app.command("triage-figures")
def triage_figures(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    provider: str = typer.Option("qwen", "--provider", help="VLM provider key from config."),
    model: str | None = typer.Option(None, "--model", help="Override configured vision model."),
    limit: int | None = typer.Option(None, "--limit", help="Only triage the first N image blocks."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_figure_triage(
        paper_id,
        provider=provider,
        model=model,
        limit=limit,
    )
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Figure triage for agent run {result.run.agent_run_id}: {len(result.results)} result(s)"
    )
    _print_figure_triage_table(result.results, title=f"Figure Triage: {paper_id}")


@app.command("list-figure-triage")
def list_figure_triage(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    results = _material_structure_agent_service(config).list_figure_triage_results(paper_id)
    if results is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_figure_triage_table(results, title=f"Figure Triage: {paper_id}")


@app.command("segment-figures")
def segment_figures(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    limit: int | None = typer.Option(
        None, "--limit", help="Only segment the first N triage results."
    ),
    max_segments: int | None = typer.Option(None, "--max-segments"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_decimer_segmentation(
        paper_id,
        limit=limit,
        max_segments=max_segments,
    )
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"DECIMER segmentation for agent run {result.run.agent_run_id}: "
        f"{len(result.crops)} crop(s), {len(result.errors)} error(s)"
    )
    _print_molecule_crop_table(result.crops, title=f"Molecule Crops: {paper_id}")
    for error in result.errors:
        console.print(f"[yellow]warning[/yellow] {error}")


@app.command("list-molecule-crops")
def list_molecule_crops(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    crops = _material_structure_agent_service(config).list_molecule_crops(paper_id)
    if crops is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_molecule_crop_table(crops, title=f"Molecule Crops: {paper_id}")


@app.command("validate-molecule-crops")
def validate_molecule_crops(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    provider: str = typer.Option("qwen", "--provider", help="VLM provider key from config."),
    model: str | None = typer.Option(None, "--model", help="Override configured vision model."),
    limit: int | None = typer.Option(None, "--limit", help="Only validate the first N crops."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_crop_validation(
        paper_id,
        provider=provider,
        model=model,
        limit=limit,
    )
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Crop validation for agent run {result.run.agent_run_id}: "
        f"{len(result.validations)} result(s)"
    )
    _print_crop_validation_table(result.validations, title=f"Molecule Crop Validation: {paper_id}")


@app.command("list-molecule-crop-validations")
def list_molecule_crop_validations(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    validations = _material_structure_agent_service(config).list_crop_validations(paper_id)
    if validations is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_crop_validation_table(validations, title=f"Molecule Crop Validation: {paper_id}")


@app.command("bind-molecule-labels")
def bind_molecule_labels(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    provider: str = typer.Option("qwen", "--provider", help="VLM provider key from config."),
    model: str | None = typer.Option(None, "--model", help="Override configured vision model."),
    limit: int | None = typer.Option(None, "--limit", help="Only bind the first N eligible crops."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_label_binding(
        paper_id,
        provider=provider,
        model=model,
        limit=limit,
    )
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    console.print(
        f"Label binding for agent run {result.run.agent_run_id}: "
        f"{len(result.bindings)} eligible crop(s)"
    )
    _print_label_binding_table(result.bindings, title=f"Molecule Label Bindings: {paper_id}")


@app.command("list-molecule-label-bindings")
def list_molecule_label_bindings(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bindings = _material_structure_agent_service(config).list_label_bindings(paper_id)
    if bindings is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_label_binding_table(bindings, title=f"Molecule Label Bindings: {paper_id}")


@app.command("run-decimer-ocsr")
def run_decimer_ocsr(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    limit: int | None = typer.Option(
        None, "--limit", help="Only process the first N reviewed bindings."
    ),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_decimer_ocsr(paper_id, limit=limit)
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"DECIMER OCSR Candidates: {result.paper_id}")
    for column in ("paper_material_id", "label", "SMILES", "status"):
        table.add_column(column)
    for candidate in result.candidates:
        table.add_row(
            candidate.paper_material_id,
            candidate.query_text,
            candidate.canonical_smiles or candidate.raw_smiles or "",
            candidate.status,
        )
    console.print(table)
    console.print(
        f"Eligible bindings: {result.eligible_binding_count}; "
        f"candidates: {len(result.candidates)}; errors: {len(result.errors)}"
    )


@app.command("run-material-ocsr-pipeline")
def run_material_ocsr_pipeline(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    provider: str = typer.Option("qwen", "--provider", help="VLM provider key from config."),
    model: str | None = typer.Option(None, "--model", help="Override configured vision model."),
    allow_unreviewed_matches: bool = typer.Option(
        True,
        "--allow-unreviewed-matches/--no-allow-unreviewed-matches",
        help="Allow high-confidence pending VLM matched bindings to produce review candidates.",
    ),
    min_model_confidence: float = typer.Option(0.8, "--min-model-confidence"),
    limit_visual_blocks: int | None = typer.Option(None, "--limit-visual-blocks"),
    limit_crops: int | None = typer.Option(None, "--limit-crops"),
    limit_ocsr: int | None = typer.Option(None, "--limit-ocsr"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    result = _material_structure_agent_service(config).run_ocsr_pipeline(
        paper_id,
        vision_provider=provider,
        vision_model=model,
        allow_unreviewed_matches=allow_unreviewed_matches,
        min_model_confidence=min_model_confidence,
        limit_visual_blocks=limit_visual_blocks,
        limit_crops=limit_crops,
        limit_ocsr=limit_ocsr,
    )
    if result is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    table = Table(title=f"Material OCSR Pipeline: {paper_id}")
    for column in (
        "triage",
        "crops",
        "validations",
        "bindings",
        "eligible",
        "candidates",
        "skipped",
        "errors",
    ):
        table.add_column(column)
    table.add_row(
        str(result.triage_count),
        str(result.crop_count),
        str(result.validation_count),
        str(result.binding_count),
        str(result.eligible_binding_count),
        str(result.ocsr_candidate_count),
        str(result.skipped_count),
        str(len(result.errors)),
    )
    console.print(table)
    for error in result.errors[:10]:
        console.print(f"[yellow]warning[/yellow] {error}")


@app.command("edit-material-candidate-smiles")
def edit_material_candidate_smiles(
    structure_candidate_id: str = typer.Option(..., "--structure-candidate-id"),
    smiles: str = typer.Option(..., "--smiles"),
    actor: str = typer.Option("local_user", "--actor"),
    message: str | None = typer.Option(None, "--message"),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    bundle = _material_structure_review_service(config).correct_structure_candidate(
        structure_candidate_id,
        MaterialStructureEditAction(actor=actor, message=message, smiles=smiles),
    )
    if not bundle:
        raise typer.BadParameter(
            f"Material structure candidate not found: {structure_candidate_id}"
        )
    console.print(
        f"Corrected SMILES for {structure_candidate_id}; events={len(bundle.material_review_events)}"
    )


@app.command("list-vlm-call-logs")
def list_vlm_call_logs(
    paper_id: str = typer.Option(..., "--paper-id", help="URL-encoded paper_id or raw DOI."),
    limit: int = typer.Option(100, "--limit", help="Maximum latest call records to print."),
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
) -> None:
    calls = _material_structure_agent_service(config).list_vlm_call_logs(paper_id, limit=limit)
    if calls is None:
        raise typer.BadParameter(f"Paper not found: {paper_id}")
    _print_vlm_call_table(calls, title=f"VLM Call Log: {paper_id}")


def _print_document_visual_block_table(blocks, *, title: str) -> None:
    table = Table(title=title)
    for column in ("visual_block_id", "page", "type", "sub_type", "caption", "image"):
        table.add_column(column)
    for block in blocks:
        table.add_row(
            block.visual_block_id,
            str(block.page_id or ""),
            block.content_type,
            block.sub_type or "",
            (block.caption or "")[:60],
            "yes" if block.image_exists else "missing",
        )
    console.print(table)
    console.print(f"Blocks: {len(blocks)}")


def _print_figure_triage_table(results, *, title: str) -> None:
    table = Table(title=title)
    for column in (
        "visual_block_id",
        "status",
        "role",
        "decimer",
        "conf",
        "labels",
        "reason",
    ):
        table.add_column(column)
    for result in results:
        table.add_row(
            result.visual_block_id,
            result.status,
            result.image_role,
            "yes" if result.should_run_decimer_segmentation else "no",
            f"{result.confidence:.2f}" if result.confidence is not None else "",
            ", ".join(result.label_candidates[:4]),
            (result.error_message or result.reason or "")[:80],
        )
    console.print(table)
    console.print(f"Results: {len(results)}")


def _print_molecule_crop_table(crops, *, title: str) -> None:
    table = Table(title=title)
    for column in ("crop_id", "visual_block_id", "segment", "bbox", "size", "status", "path"):
        table.add_column(column)
    for crop in crops:
        table.add_row(
            crop.crop_id,
            crop.visual_block_id,
            str(crop.segment_index),
            ", ".join(f"{value:.0f}" for value in crop.bbox),
            f"{crop.width or '-'}x{crop.height or '-'}",
            crop.status,
            crop.crop_path,
        )
    console.print(table)
    console.print(f"Crops: {len(crops)}")


def _print_crop_validation_table(validations, *, title: str) -> None:
    table = Table(title=title)
    for column in ("crop_id", "status", "molecule", "single", "complete", "ocsr", "conf", "reason"):
        table.add_column(column)
    for validation in validations:
        table.add_row(
            validation.crop_id,
            validation.status,
            "yes" if validation.is_molecular_depiction else "no",
            "yes" if validation.is_single_molecule else "no",
            "yes" if validation.is_complete_structure else "no",
            "yes" if validation.should_run_ocsr else "no",
            f"{validation.confidence:.2f}" if validation.confidence is not None else "",
            (validation.error_message or validation.reason or "")[:80],
        )
    console.print(table)
    console.print(f"Validations: {len(validations)}")


def _print_label_binding_table(bindings, *, title: str) -> None:
    table = Table(title=title)
    for column in ("crop_id", "decision", "observed_label", "proposal", "conf", "review", "reason"):
        table.add_column(column)
    for binding in bindings:
        table.add_row(
            binding.crop_id,
            binding.model_decision,
            binding.model_observed_label or "",
            binding.model_proposed_paper_material_id or "",
            f"{binding.model_confidence:.2f}" if binding.model_confidence is not None else "",
            binding.review_status,
            (binding.error_message or binding.model_reason or "")[:80],
        )
    console.print(table)
    console.print(f"Bindings: {len(bindings)}")


def _print_vlm_call_table(calls, *, title: str) -> None:
    table = Table(title=title)
    for column in (
        "stage",
        "entity",
        "model",
        "thinking",
        "status",
        "duration",
        "tokens",
        "started_at",
        "error",
    ):
        table.add_column(column)
    for call in calls:
        total_tokens = call.usage.get("total_tokens") if isinstance(call.usage, dict) else None
        enable_thinking = call.input_context.get("enable_thinking")
        table.add_row(
            call.stage,
            call.input_entity_id,
            call.model,
            ("on" if enable_thinking is True else "off" if enable_thinking is False else "unknown"),
            call.status,
            f"{call.duration_ms} ms" if call.duration_ms is not None else "",
            str(total_tokens) if total_tokens is not None else "",
            call.started_at,
            (call.error_message or "")[:80],
        )
    console.print(table)
    console.print(f"VLM calls: {len(calls)}")


@app.command("serve-api")
def serve_api(
    config: Path = typer.Option(
        Path("config/mining_platform/mining_platform.yaml"),
        "--config",
        "-c",
    ),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    app_config = load_config(config)
    configure_logging(app_config.logging.level)
    api_app = create_app(config=app_config)
    uvicorn.run(api_app, host=host or app_config.server.host, port=port or app_config.server.port)
