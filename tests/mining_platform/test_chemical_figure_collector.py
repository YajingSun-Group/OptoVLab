from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from evolab_local.mining_platform.api.app import create_app
from evolab_local.mining_platform.chemical_figure_collector_service import (
    ChemicalFigureCollectorService,
)
from evolab_local.mining_platform.cli import app
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.external.mineru_client import MinerUParsedDocument
from evolab_local.mining_platform.library.paper_service import PaperService
from evolab_local.mining_platform.mining.mineru_parse_service import MinerUParseService


def test_chemical_figure_collector_collects_mineru_visual_blocks(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "scheme.jpg").write_bytes(b"fake image")
    (image_dir / "homo.jpg").write_bytes(b"fake image")

    service = ChemicalFigureCollectorService(config)
    blocks = service.collect_for_paper("10.1000/example")
    assert blocks is not None
    assert len(blocks) == 2
    assert blocks[0].content_type == "image"
    assert "mineru_subtype_chemical" in blocks[0].heuristic_tags
    assert blocks[0].image_exists is True
    assert blocks[1].content_type == "chart"
    assert "keyword_electronic_structure" in blocks[1].heuristic_tags

    rerun_blocks = service.collect_for_paper("10.1000/example")
    assert rerun_blocks is not None
    assert len(rerun_blocks) == 2
    listed = service.list_for_paper("10.1000/example")
    assert listed is not None
    assert len(listed) == 2


def test_chemical_figure_collector_api_and_cli(
    mining_config_path: Path,
    fake_pdf_factory,
    manifest_writer,
) -> None:
    config = load_config(mining_config_path)
    _seed_paper(config, mining_config_path, fake_pdf_factory, manifest_writer)
    run = MinerUParseService(config, client=_FakeMinerUClient()).parse_paper("10.1000/example")
    assert run is not None
    image_dir = Path(run.content_list_path).parent / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "scheme.jpg").write_bytes(b"fake image")

    client = TestClient(create_app(config=config))
    collect_response = client.post("/api/papers/10.1000%2Fexample/collect-chemical-figures")
    assert collect_response.status_code == 200
    payload = collect_response.json()
    assert len(payload) == 2
    assert payload[0]["paper_id"] == "10.1000%2Fexample"

    list_response = client.get("/api/papers/10.1000%2Fexample/chemical-figures")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 2

    cli_result = CliRunner().invoke(
        app,
        [
            "list-chemical-figures",
            "--config",
            str(mining_config_path),
            "--paper-id",
            "10.1000/example",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Chemical Figure Blocks" in cli_result.output
    assert "scheme" in cli_result.output


class _FakeMinerUClient:
    def parse_pdf(self, pdf_path: Path) -> MinerUParsedDocument:
        content_list = [
            {
                "type": "text",
                "text": "The molecular design strategy is shown in Scheme 1.",
                "page_idx": 0,
                "bbox": [1, 2, 3, 4],
            },
            {
                "type": "image",
                "sub_type": "chemical",
                "img_path": "images/scheme.jpg",
                "image_caption": ["SCHEME 1 Molecular design concept."],
                "page_idx": 0,
                "bbox": [10, 20, 200, 220],
            },
            {
                "type": "chart",
                "sub_type": "line",
                "img_path": "images/homo.jpg",
                "chart_caption": ["Figure 1 HOMO-LUMO distributions of the emitters."],
                "page_idx": 1,
                "bbox": [30, 40, 250, 260],
            },
            {
                "type": "table",
                "img_path": "images/table.jpg",
                "table_caption": ["Table 1 Device performance summary."],
                "table_body": "<table><tr><td>EQE</td></tr></table>",
                "page_idx": 2,
                "bbox": [50, 60, 300, 330],
            },
        ]
        return MinerUParsedDocument(
            task_id="task-chemical-figures",
            backend="hybrid-auto-engine",
            version="3.1.0",
            file_name=pdf_path.stem,
            md_content="# mock",
            content_list=content_list,
            raw_result={"results": {pdf_path.stem: {"md_content": "# mock", "content_list": content_list}}},
        )


def _seed_paper(config, mining_config_path: Path, fake_pdf_factory, manifest_writer) -> None:
    pdf_path = fake_pdf_factory(mining_config_path)
    manifest_writer(
        mining_config_path,
        [
            {
                "doi": "10.1000/example",
                "status": "completed",
                "journal": "Journal",
                "publisher": "Publisher",
                "pdf_path": pdf_path.relative_to(config.project_root).as_posix(),
            }
        ],
    )
    PaperService(config).ingest_from_pdf_downloader(domain="oled")
