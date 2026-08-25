from __future__ import annotations

from pathlib import Path

from evolab_local.optovlab.analysis_skills import AnalysisSkillService
from evolab_local.optovlab.repository import OptoVLabRepository
from evolab_local.optovlab.schemas import AnalysisRequest


def _records() -> list[dict]:
    return [
        {
            "id": f"D{index}",
            "doi": f"10.1/{index // 2}",
            "eqe_max": float(index * 2 + 4),
            "layer_count": index + 4,
            "material_count": index + 2,
            "emission_color": "green" if index % 2 else "blue",
            "fabrication_method": "vacuum_deposition",
            "final_emitter": f"Emitter-{index % 3}",
        }
        for index in range(12)
    ]


def _service(tmp_path: Path) -> tuple[AnalysisSkillService, str]:
    repository = OptoVLabRepository(tmp_path / "optovlab.sqlite")
    repository.init_runtime()
    session = repository.create_session("data_mining")
    return AnalysisSkillService(tmp_path / "artifacts", repository), session.session_id


def test_univariate_skill_computes_statistics_and_writes_artifacts(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)

    result = service.run(
        session_id,
        AnalysisRequest(skill_id="univariate_distribution", metric="eqe_max"),
        _records(),
    )

    assert result.statistics["count"] == 12
    assert result.statistics["median"] == 15.0
    assert len(result.artifacts) == 2
    assert (tmp_path / "artifacts" / session_id / result.artifacts[0].filename).is_file()


def test_bivariate_skill_reports_known_positive_relationship(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)

    result = service.run(
        session_id,
        AnalysisRequest(
            skill_id="bivariate_relationship",
            x_field="layer_count",
            y_field="eqe_max",
        ),
        _records(),
    )

    assert result.statistics["pearson_r"] == 1.0
    assert result.statistics["spearman_rho"] == 1.0
    assert "not causal" in result.summary


def test_data_quality_skill_measures_missing_values(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    records = _records()
    records[0]["eqe_max"] = None

    result = service.run(
        session_id,
        AnalysisRequest(skill_id="data_quality_profile"),
        records,
    )

    assert result.statistics["eqe_max"] == 91.67
    assert result.statistics["emission_color"] == 100.0


def test_group_comparison_uses_current_matplotlib_boxplot_api(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)

    result = service.run(
        session_id,
        AnalysisRequest(
            skill_id="group_comparison",
            group_field="emission_color",
            metric="eqe_max",
        ),
        _records(),
    )

    assert result.statistics["counts"] == {"blue": 6, "green": 6}
    assert result.artifacts[0].title == "Grouped performance comparison"


def test_correlation_matrix_supports_small_demo_dataset(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)

    result = service.run(
        session_id,
        AnalysisRequest(skill_id="correlation_matrix"),
        _records()[:4],
    )

    assert result.statistics["minimum_observations"] == 4
    assert {"eqe_max", "layer_count", "material_count"}.issubset(
        result.statistics["fields"]
    )
    assert result.artifacts[0].title == "Correlation matrix"


def test_correlation_matrix_reports_single_varying_demo_field(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    records = _records()[:4]
    for record in records:
        record["layer_count"] = 5
        record["material_count"] = 3

    result = service.run(
        session_id,
        AnalysisRequest(skill_id="correlation_matrix"),
        records,
    )

    assert result.statistics["fields"] == ["eqe_max"]
    assert "do not have enough overlapping variation" in result.summary
