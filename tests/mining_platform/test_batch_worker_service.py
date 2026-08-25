from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evolab_local.mining_platform.batch_worker_service import (
    BatchWorkerService,
    _public_candidate_should_yield_to_visual_fallback,
)
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.schemas.batch import MaterialStage3PlanItem
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialIdentityJudgment,
    MaterialStructureCandidate,
)
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    BatchJobRepository,
    CandidateIngestionRepository,
    PaperRepository,
)


class FakeMinerUService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse_paper(self, paper_id: str):
        self.calls.append(paper_id)
        return object()


class FakeLLMService:
    def __init__(self, config_path: Path) -> None:
        self.config = load_config(config_path)
        self.database = Database(self.config.paths.sqlite_path)
        self.candidates = CandidateIngestionRepository(self.database)
        self.calls: list[str] = []

    def mine_paper(self, paper_id: str, template_id: str, provider=None, model=None):
        self.calls.append(paper_id)
        self.candidates.create_run(
            paper_id=paper_id,
            template_id=template_id,
            template_version="test",
            source_name="fake_llm",
            source_version=model,
            status="completed",
            validation_report={"valid": True, "errors": []},
            mining_result={
                "evidence": [],
                "materials": [],
                "devices": [
                    {
                        "device_label": "D1",
                        "architecture_text": "ITO/organic layer/Al",
                        "layers": [],
                        "performance": [],
                    }
                ],
            },
        )
        return object()


class FakeNoDeviceLLMService(FakeLLMService):
    def mine_paper(self, paper_id: str, template_id: str, provider=None, model=None):
        self.calls.append(paper_id)
        self.candidates.create_run(
            paper_id=paper_id,
            template_id=template_id,
            template_version="test",
            source_name="fake_llm",
            source_version=model,
            status="completed",
            validation_report={"valid": True, "errors": []},
            mining_result={"evidence": [], "materials": [], "devices": []},
        )
        return object()


class FakeMaterialResolutionService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve_paper_materials(self, paper_id: str):
        self.calls.append(paper_id)
        return object()


class FakeAcceptedOCSRMaterialResolutionService(FakeMaterialResolutionService):
    def get_material_structure_bundle(self, paper_id: str):
        return SimpleNamespace(
            structure_candidates=[
                SimpleNamespace(
                    paper_material_id="M001",
                    provider="decimer_ocsr",
                    status="accepted",
                    canonical_smiles="c1ccccc1",
                    inchi_key=None,
                )
            ]
        )


class FakePublicResolverService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve_paper_public(self, paper_id: str):
        self.calls.append(paper_id)
        return object()


class FakeIdentityJudgeService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge_paper_candidates(self, paper_id: str, provider=None, model=None):
        self.calls.append(paper_id)
        return object()


class FakeMetadataService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enrich_paper(self, paper_id: str):
        self.calls.append(paper_id)
        return SimpleNamespace(status="updated")


class PersistingFakeMetadataService(FakeMetadataService):
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        config = load_config(config_path)
        self.papers = PaperRepository(Database(config.paths.sqlite_path))

    def enrich_paper(self, paper_id: str):
        self.calls.append(paper_id)
        paper = self.papers.get(paper_id)
        assert paper is not None
        self.papers.upsert(
            paper.model_copy(
                update={
                    "title": f"Title {paper_id}",
                    "journal": "Test Journal",
                    "publisher": "Test Publisher",
                    "year": 2024,
                }
            )
        )
        return SimpleNamespace(status="updated")


class FakeMaterialAgentService:
    def __init__(self) -> None:
        self.foundation_calls: list[str] = []
        self.ocsr_calls: list[tuple[str, bool, float]] = []

    def run_foundation(self, paper_id: str):
        self.foundation_calls.append(paper_id)
        return object()

    def run_ocsr_pipeline(
        self,
        paper_id: str,
        *,
        allow_unreviewed_matches: bool,
        min_model_confidence: float,
    ):
        self.ocsr_calls.append((paper_id, allow_unreviewed_matches, min_model_confidence))
        return object()


class FakeMaterialStructureTriageService:
    def __init__(self, should_run_ocsr: bool) -> None:
        self.should_run_ocsr = should_run_ocsr
        self.calls: list[str] = []

    def triage_paper(self, paper_id: str):
        self.calls.append(paper_id)
        return SimpleNamespace(should_run_ocsr=self.should_run_ocsr)


class FakeMaterialCompletionService:
    def __init__(self, config_path: Path, should_confirm: bool) -> None:
        self.config = load_config(config_path)
        self.should_confirm = should_confirm
        self.calls: list[str] = []

    def confirm_paper_if_materials_complete(self, paper_id: str) -> bool:
        self.calls.append(paper_id)
        if self.should_confirm:
            PaperRepository(Database(self.config.paths.sqlite_path)).set_review_status(
                paper_id,
                "confirmed",
            )
        return self.should_confirm


def _public_candidate() -> MaterialStructureCandidate:
    return MaterialStructureCandidate(
        structure_candidate_id="candidate-1",
        paper_id="10.1000%2Fidentity",
        candidate_run_id="run-1",
        paper_material_id="M001",
        provider="pubchem",
        resolver_name="anysearch_to_pubchem",
        query_text="PaperLabel",
        canonical_smiles="c1ccccc1",
        created_at="2026-07-16T00:00:00+08:00",
        updated_at="2026-07-16T00:00:00+08:00",
    )


def _identity_judgment(
    *,
    verdict: str,
    recommended_action: str,
) -> MaterialIdentityJudgment:
    return MaterialIdentityJudgment(
        judgment_id=f"judgment-{verdict}",
        paper_id="10.1000%2Fidentity",
        candidate_run_id="run-1",
        paper_material_id="M001",
        structure_candidate_id="candidate-1",
        provider="deepseek",
        model="test-model",
        verdict=verdict,
        recommended_action=recommended_action,
        created_at="2026-07-16T00:00:00+08:00",
        updated_at="2026-07-16T00:00:00+08:00",
    )


def test_ambiguous_public_candidate_does_not_block_visual_ocsr() -> None:
    candidate = _public_candidate()
    judgment = _identity_judgment(
        verdict="ambiguous",
        recommended_action="search_more_evidence",
    )

    assert _public_candidate_should_yield_to_visual_fallback(candidate, [judgment]) is True


def test_identity_ready_public_candidate_still_blocks_visual_ocsr() -> None:
    candidate = _public_candidate()
    judgment = _identity_judgment(
        verdict="likely_match",
        recommended_action="ready_for_human_accept",
    )

    assert _public_candidate_should_yield_to_visual_fallback(candidate, [judgment]) is False


def test_unjudged_public_candidate_still_waits_for_identity_judge() -> None:
    assert _public_candidate_should_yield_to_visual_fallback(_public_candidate(), []) is False


def test_import_pdfs_from_csv_copies_and_registers_jobs(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    source_dir = mining_config_path.parent.parent.parent / "source_pdfs"
    source_dir.mkdir()
    pdf_path = source_dir / "10.1000%2Fexample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nexample\n%%EOF\n")
    csv_path = source_dir / "devices.csv"
    csv_path.write_text("doi_encode\n10.1000%2Fexample\n10.1000%2Fexample\n", encoding="utf-8")

    service = BatchWorkerService(config)
    result = service.import_pdfs_from_csv(csv_path=csv_path, source_pdf_dir=source_dir)

    assert result.requested_count == 1
    assert result.copied_count == 1
    assert result.registered_count == 1
    assert (config.paths.inbox_pdfs_dir / "10.1000%2Fexample.pdf").exists()

    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fexample")
    assert paper is not None
    assert paper.doi == "10.1000/example"
    assert paper.source == "inbox_watcher"

    job = BatchJobRepository(Database(config.paths.sqlite_path)).get_by_paper("10.1000%2Fexample")
    assert job is not None
    assert job.status == "registered"
    assert job.source_pdf_path == pdf_path.resolve().as_posix()


def test_batch_worker_runs_registered_job_with_fake_services(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Fworker.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nworker\n%%EOF\n")

    fake_mineru = FakeMinerUService()
    fake_llm = FakeLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    fake_public = FakePublicResolverService()
    fake_identity = FakeIdentityJudgeService()
    fake_completion = FakeMaterialCompletionService(mining_config_path, should_confirm=True)
    service = BatchWorkerService(
        config,
        mineru_service=fake_mineru,
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
        material_public_resolver_service=fake_public,
        material_identity_judge_service=fake_identity,
        material_completion_service=fake_completion,
    )
    scan = service.scan_inbox_pdfs(stable_file_seconds=0)
    assert scan.registered_count == 1

    result = service.run_once(max_jobs=1, provider="deepseek", model="fake-model")

    assert result.processed_count == 1
    assert result.review_ready_count == 1
    job = result.jobs[0]
    assert job.status == "review_ready"
    assert job.last_completed_stage == "review_ready"
    assert fake_mineru.calls == ["10.1000%2Fworker"]
    assert fake_llm.calls == ["10.1000%2Fworker"]
    assert fake_materials.calls == ["10.1000%2Fworker"]
    assert fake_public.calls == ["10.1000%2Fworker"]
    assert fake_identity.calls == ["10.1000%2Fworker"]
    assert fake_completion.calls == ["10.1000%2Fworker"]
    assert "device_mining_llm" in job.stage_timings
    assert "material_completion_check" in job.stage_timings
    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fworker")
    assert paper is not None
    assert paper.review_status == "confirmed"


def test_batch_worker_auto_confirms_no_device_paper_and_skips_material_pipeline(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Fno-device.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nno device\n%%EOF\n")

    fake_llm = FakeNoDeviceLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    fake_public = FakePublicResolverService()
    fake_identity = FakeIdentityJudgeService()
    service = BatchWorkerService(
        config,
        mineru_service=FakeMinerUService(),
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
        material_public_resolver_service=fake_public,
        material_identity_judge_service=fake_identity,
    )
    service.scan_inbox_pdfs(stable_file_seconds=0)

    result = service.run_once(max_jobs=1)

    assert result.review_ready_count == 1
    assert fake_materials.calls == []
    assert fake_public.calls == []
    assert fake_identity.calls == []
    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Fno-device")
    assert paper is not None
    assert paper.review_status == "confirmed"
    assert paper.review_reason == "no_device_data"
    detail = service.get_review_batch(0, batch_size=10)
    assert detail.papers[0].is_confirmed is True
    assert detail.papers[0].paper.review_reason == "no_device_data"


def test_excluded_review_article_is_terminal_but_not_counted_as_confirmed(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Freview-article.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nreview article\n%%EOF\n")
    service = BatchWorkerService(config)
    service.scan_inbox_pdfs(stable_file_seconds=0)
    papers = PaperRepository(Database(config.paths.sqlite_path))
    papers.set_review_status(
        "10.1000%2Freview-article",
        "excluded",
        reason="review_article",
    )

    assert BatchJobRepository(Database(config.paths.sqlite_path)).next_runnable(limit=10) == []
    detail = service.get_review_batch(0, batch_size=10)
    overview = service.list_review_batches(batch_size=10)

    assert detail.papers[0].is_confirmed is False
    assert detail.papers[0].is_excluded is True
    assert detail.summary.confirmed_count == 0
    assert detail.summary.excluded_count == 1
    assert detail.summary.all_confirmed is False
    assert detail.summary.all_resolved is True
    assert detail.summary.status == "completed"
    assert overview.current_batch_index is None
    assert overview.confirmed_batch_count == 0
    assert overview.resolved_batch_count == 1


def test_batch_worker_runs_optional_metadata_and_material_ocsr_stages(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Foptional.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\noptional\n%%EOF\n")

    fake_mineru = FakeMinerUService()
    fake_llm = FakeLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    fake_public = FakePublicResolverService()
    fake_identity = FakeIdentityJudgeService()
    fake_metadata = FakeMetadataService()
    fake_agent = FakeMaterialAgentService()
    fake_triage = FakeMaterialStructureTriageService(should_run_ocsr=True)
    service = BatchWorkerService(
        config,
        mineru_service=fake_mineru,
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
        material_public_resolver_service=fake_public,
        material_identity_judge_service=fake_identity,
        material_agent_service=fake_agent,
        material_structure_triage_service=fake_triage,
        metadata_service=fake_metadata,
    )
    service.scan_inbox_pdfs(stable_file_seconds=0)

    result = service.run_once(
        max_jobs=1,
        run_metadata_enrichment=True,
        run_visual_prep=True,
        run_material_ocsr=True,
    )

    assert result.review_ready_count == 1
    job = result.jobs[0]
    assert fake_metadata.calls == ["10.1000%2Foptional"]
    assert fake_agent.foundation_calls == ["10.1000%2Foptional"]
    assert fake_agent.ocsr_calls == [("10.1000%2Foptional", True, 0.8)]
    assert fake_triage.calls == ["10.1000%2Foptional", "10.1000%2Foptional"]
    assert "metadata_enrich" in job.stage_timings
    assert "material_structure_triage" in job.stage_timings
    assert "material_ocsr" in job.stage_timings


def test_batch_worker_skips_material_ocsr_when_triage_says_no_core_materials(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Fskip-ocsr.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nskip ocsr\n%%EOF\n")

    fake_mineru = FakeMinerUService()
    fake_llm = FakeLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    fake_agent = FakeMaterialAgentService()
    fake_triage = FakeMaterialStructureTriageService(should_run_ocsr=False)
    service = BatchWorkerService(
        config,
        mineru_service=fake_mineru,
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
        material_agent_service=fake_agent,
        material_structure_triage_service=fake_triage,
    )
    service.scan_inbox_pdfs(stable_file_seconds=0)

    result = service.run_once(
        max_jobs=1,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=True,
    )

    assert result.review_ready_count == 1
    job = result.jobs[0]
    assert fake_triage.calls == ["10.1000%2Fskip-ocsr", "10.1000%2Fskip-ocsr"]
    assert fake_agent.ocsr_calls == []
    assert "material_structure_triage" in job.stage_timings
    assert job.stage_timings["material_ocsr"] == 0.0


def test_run_review_batch_targets_requested_batch(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    config.paths.inbox_pdfs_dir.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        pdf_path = config.paths.inbox_pdfs_dir / f"10.1000%2Fbatch{index:02d}.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nbatch\n%%EOF\n")

    fake_mineru = FakeMinerUService()
    fake_llm = FakeLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    service = BatchWorkerService(
        config,
        mineru_service=fake_mineru,
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
    )
    scan = service.scan_inbox_pdfs(stable_file_seconds=0)
    assert scan.registered_count == 12

    result = service.run_review_batch(
        batch_index=1,
        batch_size=10,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
    )
    assert result.processed_count == 2
    assert result.review_ready_count == 2
    assert set(fake_mineru.calls) == {"10.1000%2Fbatch10", "10.1000%2Fbatch11"}
    assert set(fake_llm.calls) == {"10.1000%2Fbatch10", "10.1000%2Fbatch11"}
    first_batch_jobs = service.get_review_batch(0, batch_size=10).papers
    assert {item.job.status for item in first_batch_jobs} == {"registered"}


def test_run_review_batch_staged_runs_stage_barriers(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    config.paths.inbox_pdfs_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = {f"10.1000%2Fstaged{index}" for index in range(3)}
    for index in range(3):
        pdf_path = config.paths.inbox_pdfs_dir / f"10.1000%2Fstaged{index}.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nstaged\n%%EOF\n")

    fake_mineru = FakeMinerUService()
    fake_llm = FakeLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    service = BatchWorkerService(
        config,
        mineru_service=fake_mineru,
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
    )
    scan = service.scan_inbox_pdfs(stable_file_seconds=0)
    assert scan.registered_count == 3

    result = service.run_review_batch_staged(
        batch_index=0,
        batch_size=10,
        provider="deepseek",
        model="fake-model",
        parse_concurrency=1,
        llm_concurrency=2,
        material_concurrency=2,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=False,
    )

    assert result.processed_count == 3
    assert result.review_ready_count == 3
    assert result.failed_count == 0
    assert {
        "stage1_metadata_parse",
        "stage2_device_mining",
        "stage3_material_pipeline",
    }.issubset(result.phase_timings)
    assert "stage3_plan_local" in result.phase_timings
    assert {job.paper_id for job in result.jobs} == expected_ids
    assert {job.status for job in result.jobs} == {"review_ready"}
    assert set(fake_mineru.calls) == expected_ids
    assert set(fake_llm.calls) == expected_ids
    assert set(fake_materials.calls) == expected_ids
    for job in result.jobs:
        assert "parse_mineru" in job.stage_timings
        assert "device_mining_llm" in job.stage_timings


def test_run_review_batch_staged_enriches_skipped_review_ready_paper(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    config.paths.inbox_pdfs_dir.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        pdf_path = config.paths.inbox_pdfs_dir / f"10.1000%2Fmixed{index}.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nmixed staged\n%%EOF\n")

    metadata = PersistingFakeMetadataService(mining_config_path)
    service = BatchWorkerService(
        config,
        mineru_service=FakeMinerUService(),
        llm_service=FakeLLMService(mining_config_path),
        material_resolution_service=FakeMaterialResolutionService(),
        metadata_service=metadata,
    )
    service.scan_inbox_pdfs(stable_file_seconds=0)
    detail = service.get_review_batch(0, batch_size=10)
    skipped_id = detail.papers[0].paper.paper_id
    service._mark_review_ready(detail.papers[0].job)

    result = service.run_review_batch_staged(
        batch_index=0,
        batch_size=10,
        run_metadata_enrichment=True,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=False,
    )

    expected_ids = {item.paper.paper_id for item in detail.papers}
    assert set(metadata.calls) == expected_ids
    assert len(metadata.calls) == 2
    assert result.material_metrics["metadata_target_count"] == 1
    assert result.material_metrics["metadata_attempted_count"] == 1
    assert result.material_metrics["metadata_updated_count"] == 1
    assert result.material_metrics["scheduler_mode"] == "paper_dag"
    assert result.material_metrics["phase_timings_overlap"] is True
    assert "stage3_visual_prefetch" in result.phase_timings
    assert (
        PaperRepository(Database(config.paths.sqlite_path)).get(skipped_id).journal
        == "Test Journal"
    )


def test_run_review_batch_staged_enriches_metadata_when_no_jobs_are_runnable(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    pdf_path = config.paths.inbox_pdfs_dir / "10.1000%2Fmetadata-only.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.7\nmetadata only\n%%EOF\n")
    metadata = PersistingFakeMetadataService(mining_config_path)
    service = BatchWorkerService(config, metadata_service=metadata)
    service.scan_inbox_pdfs(stable_file_seconds=0)
    detail = service.get_review_batch(0, batch_size=10)
    service._mark_review_ready(detail.papers[0].job)

    result = service.run_review_batch_staged(
        batch_index=0,
        batch_size=10,
        run_metadata_enrichment=True,
    )

    assert result.processed_count == 0
    assert result.material_metrics["metadata_target_count"] == 1
    assert result.material_metrics["metadata_attempted_count"] == 1
    assert result.material_metrics["metadata_updated_count"] == 1
    assert metadata.calls == ["10.1000%2Fmetadata-only"]


def test_refresh_review_batch_materials_processes_review_ready_jobs(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Frefresh.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nrefresh\n%%EOF\n")

    initial_service = BatchWorkerService(
        config,
        mineru_service=FakeMinerUService(),
        llm_service=FakeLLMService(mining_config_path),
        material_resolution_service=FakeMaterialResolutionService(),
    )
    initial_service.scan_inbox_pdfs(stable_file_seconds=0)
    initial = initial_service.run_once(
        max_jobs=1,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=False,
    )
    assert initial.jobs[0].status == "review_ready"

    fake_materials = FakeMaterialResolutionService()
    fake_completion = FakeMaterialCompletionService(mining_config_path, should_confirm=True)
    fake_metadata = FakeMetadataService()
    refresh_service = BatchWorkerService(
        config,
        material_resolution_service=fake_materials,
        material_completion_service=fake_completion,
        metadata_service=fake_metadata,
    )
    result = refresh_service.refresh_review_batch_materials(
        batch_index=0,
        batch_size=10,
        run_metadata_enrichment=True,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=False,
    )

    assert result.processed_count == 1
    assert result.review_ready_count == 1
    assert result.phase_timings["stage3_material_refresh"] >= 0
    assert result.phase_timings["metadata_enrichment"] >= 0
    assert result.material_metrics["metadata_attempted_count"] == 1
    assert result.material_metrics["metadata_updated_count"] == 1
    assert fake_metadata.calls == ["10.1000%2Frefresh"]
    assert fake_materials.calls == ["10.1000%2Frefresh"]
    assert fake_completion.calls == ["10.1000%2Frefresh"]
    paper = PaperRepository(Database(config.paths.sqlite_path)).get("10.1000%2Frefresh")
    assert paper is not None
    assert paper.review_status == "confirmed"


def test_refresh_review_batch_materials_enriches_confirmed_papers(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Fconfirmed-metadata.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nconfirmed metadata\n%%EOF\n")
    fake_metadata = FakeMetadataService()
    service = BatchWorkerService(config, metadata_service=fake_metadata)
    service.scan_inbox_pdfs(stable_file_seconds=0)
    PaperRepository(Database(config.paths.sqlite_path)).set_review_status(
        "10.1000%2Fconfirmed-metadata",
        "confirmed",
    )

    result = service.refresh_review_batch_materials(
        batch_index=0,
        batch_size=10,
        run_metadata_enrichment=True,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=False,
    )

    assert result.processed_count == 0
    assert result.review_ready_count == 1
    assert result.material_metrics["metadata_attempted_count"] == 1
    assert result.material_metrics["metadata_updated_count"] == 1
    assert fake_metadata.calls == ["10.1000%2Fconfirmed-metadata"]
    job = BatchJobRepository(Database(config.paths.sqlite_path)).get_by_paper(
        "10.1000%2Fconfirmed-metadata"
    )
    assert job is not None
    assert job.status == "review_ready"


def test_refresh_review_batch_materials_skips_stage2_incomplete_papers(
    mining_config_path: Path,
) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Fstage2-incomplete.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nstage2 incomplete\n%%EOF\n")
    fake_materials = FakeMaterialResolutionService()
    service = BatchWorkerService(
        config,
        material_resolution_service=fake_materials,
    )
    service.scan_inbox_pdfs(stable_file_seconds=0)

    result = service.refresh_review_batch_materials(
        batch_index=0,
        batch_size=10,
        run_metadata_enrichment=False,
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
        run_visual_prep=False,
        run_material_ocsr=False,
    )

    assert result.processed_count == 0
    assert result.skipped_count == 1
    assert result.review_ready_count == 0
    assert fake_materials.calls == []
    job = BatchJobRepository(Database(config.paths.sqlite_path)).get_by_paper(
        "10.1000%2Fstage2-incomplete"
    )
    assert job is not None
    assert job.status == "registered"


def test_finalize_ocsr_marks_accepted_candidate_complete(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    service = BatchWorkerService(
        config,
        material_resolution_service=FakeAcceptedOCSRMaterialResolutionService(),
    )
    service.init_runtime()
    item = MaterialStage3PlanItem(
        paper_id="10.1000%2Faccepted-ocsr",
        candidate_run_id="candidate-run-1",
        paper_material_id="M001",
        material_label="Accepted OCSR material",
        category="core_structure_required",
        route="visual_ocsr",
    )

    human_review_count = service._finalize_stage3_ocsr_routes(
        [item],
        ocsr_was_run=True,
    )

    task = service.material_stage3_planner_service.tasks.get_by_paper_material(
        item.candidate_run_id,
        item.paper_material_id,
    )
    assert human_review_count == 0
    assert task is not None
    assert task.status == "completed"
    assert task.current_stage == "completed"
    assert task.next_action == "none"
    assert task.assigned_strategy == "accepted_ocsr_structure"


def test_write_batch_report_outputs_json_and_markdown(mining_config_path: Path) -> None:
    config = load_config(mining_config_path)
    inbox_pdf = config.paths.inbox_pdfs_dir / "10.1000%2Freport.pdf"
    inbox_pdf.parent.mkdir(parents=True, exist_ok=True)
    inbox_pdf.write_bytes(b"%PDF-1.7\nreport\n%%EOF\n")

    fake_mineru = FakeMinerUService()
    fake_llm = FakeLLMService(mining_config_path)
    fake_materials = FakeMaterialResolutionService()
    service = BatchWorkerService(
        config,
        mineru_service=fake_mineru,
        llm_service=fake_llm,
        material_resolution_service=fake_materials,
    )
    service.scan_inbox_pdfs(stable_file_seconds=0)
    result = service.run_once(
        max_jobs=1,
        provider="deepseek",
        model="fake-model",
        run_public_resolver=False,
        run_identity_judge=False,
        run_material_auto_decision=False,
    )
    result.phase_timings["stage3_material_refresh"] = 1.25
    result.material_metrics.update(
        {
            "scheduler_mode": "paper_dag",
            "phase_timings_overlap": True,
            "stage3_phase_task_seconds": {"public_resolution": 2.5},
        }
    )

    report = service.write_batch_report(
        batch_index=0,
        batch_size=10,
        run_result=result,
        run_options={"provider": "deepseek", "model": "fake-model"},
    )

    json_path = Path(report.json_path or "")
    markdown_path = Path(report.markdown_path or "")
    if not json_path.is_absolute():
        json_path = config.project_root / json_path
    if not markdown_path.is_absolute():
        markdown_path = config.project_root / markdown_path
    assert json_path.exists()
    assert markdown_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["batch_number"] == 1
    assert payload["summary"]["processed_count"] == 1
    assert payload["summary"]["review_ready_count"] == 1
    assert payload["jobs"][0]["paper_id"] == "10.1000%2Freport"
    assert payload["jobs"][0]["candidate_run_count"] == 1
    assert payload["jobs"][0]["total_seconds"] >= 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Batch Report: batch-0001" in markdown
    assert "10.1000/report" in markdown
    assert "device_mining_llm" in markdown
    assert "These phase windows overlap" in markdown
    assert "Stage 3 Cumulative Task Time" in markdown
