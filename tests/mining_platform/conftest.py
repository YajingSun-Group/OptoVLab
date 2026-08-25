from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture()
def mining_config_path(tmp_path: Path) -> Path:
    root = tmp_path
    config_dir = root / "config" / "mining_platform"
    config_dir.mkdir(parents=True)
    (root / "runtime" / "pdf_downloader" / "papers").mkdir(parents=True)
    manifest_path = root / "runtime" / "pdf_downloader" / "download_manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")
    config_path = config_dir / "mining_platform.yaml"
    config_path.write_text(
        """
paths:
  runtime_dir: ./runtime/mining_platform
  sqlite_path: ./runtime/mining_platform/platform.sqlite
  paper_registry_path: ./runtime/mining_platform/paper_registry.jsonl
  parsed_dir: ./runtime/mining_platform/parsed
  logs_dir: ./runtime/mining_platform/logs
  inbox_pdfs_dir: ./runtime/mining_platform/inbox_pdfs
  batch_reports_dir: ./runtime/mining_platform/batch_reports

pdf_downloader:
  manifest_path: ./runtime/pdf_downloader/download_manifest.json
  sqlite_path: ./runtime/pdf_downloader/results.sqlite
  papers_dir: ./runtime/pdf_downloader/papers

server:
  host: 127.0.0.1
  port: 8000

logging:
  level: INFO

batch_worker:
  scan_interval_seconds: 1
  stable_file_seconds: 0
  max_retries: 2
  default_domain: oled
  run_metadata_enrichment: false
  run_public_resolver: true
  run_identity_judge: true
  run_material_auto_decision: true
  material_auto_accept_min_confidence: 0.9
  material_auto_reject_min_confidence: 0.9
  material_auto_accept_decimer_ocsr: false
  run_visual_prep: false
  run_material_ocsr: false
  material_ocsr_auto_match_min_confidence: 0.8

features:
  material_properties: true
""".strip(),
        encoding="utf-8",
    )
    return config_path


def write_pdf_downloader_manifest(config_path: Path, items: list[dict[str, object]]) -> None:
    root = config_path.parent.parent.parent
    manifest_path = root / "runtime" / "pdf_downloader" / "download_manifest.json"
    manifest_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def create_fake_pdf(config_path: Path, name: str = "10.1000%2Fexample.pdf") -> Path:
    root = config_path.parent.parent.parent
    pdf_path = root / "runtime" / "pdf_downloader" / "papers" / name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.7\nfake module 1 paper\n%%EOF\n")
    return pdf_path


def create_text_pdf(
    config_path: Path,
    name: str = "10.1000%2Fexample.pdf",
    text: str = "OLED device D1 showed a maximum EQE of 18.2%.",
) -> Path:
    root = config_path.parent.parent.parent
    pdf_path = root / "runtime" / "pdf_downloader" / "papers" / name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture()
def manifest_writer():
    return write_pdf_downloader_manifest


@pytest.fixture()
def fake_pdf_factory():
    return create_fake_pdf


@pytest.fixture()
def text_pdf_factory():
    return create_text_pdf
