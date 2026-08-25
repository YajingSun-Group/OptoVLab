from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evolab_local.optovlab.repository import OptoVLabRepository
from evolab_local.optovlab.schemas import AnalysisRequest, AnalysisResult, Artifact


NUMERIC_FIELDS = (
    "eqe_max",
    "ce_max",
    "pe_max",
    "luminance_max",
    "turn_on_voltage",
    "el_peak",
    "fwhm",
    "lifetime",
    "layer_count",
    "material_count",
    "year",
)

CATEGORICAL_FIELDS = (
    "emission_color",
    "fabrication_method",
    "device_type",
    "final_emitter",
    "final_emitter_class",
    "journal",
    "quality_tier",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, dict):
        for key in ("value", "median", "max"):
            if key in value:
                return _number(value[key])
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: record.get(key) for key in CATEGORICAL_FIELDS}
        for key in NUMERIC_FIELDS:
            row[key] = _number(record.get(key))
        row["device_id"] = record.get("id") or record.get("device_label")
        row["doi"] = record.get("doi")
        rows.append(row)
    return pd.DataFrame(rows)


class AnalysisSkillService:
    def __init__(self, artifact_dir: Path, repository: OptoVLabRepository) -> None:
        self.artifact_dir = artifact_dir
        self.repository = repository

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "skill_id": "dataset_summary",
                "name": "Dataset summary",
                "description": "Count papers, devices, emitters, and major device categories.",
            },
            {
                "skill_id": "data_quality_profile",
                "name": "Data quality profile",
                "description": "Measure field completeness before interpreting extracted data.",
            },
            {
                "skill_id": "univariate_distribution",
                "name": "Univariate distribution",
                "description": "Summarize and visualize EQE or another numeric device field.",
            },
            {
                "skill_id": "bivariate_relationship",
                "name": "Bivariate relationship",
                "description": "Measure Pearson and Spearman association between two fields.",
            },
            {
                "skill_id": "group_comparison",
                "name": "Group comparison",
                "description": "Compare a performance metric across colors, mechanisms, or methods.",
            },
            {
                "skill_id": "correlation_matrix",
                "name": "Correlation matrix",
                "description": "Inspect relationships among available numeric device fields.",
            },
        ]

    def run(
        self,
        session_id: str,
        request: AnalysisRequest,
        records: list[dict[str, Any]],
    ) -> AnalysisResult:
        if not records:
            raise ValueError("No device records are available for analysis")
        frame = _frame(records)
        handler = getattr(self, f"_{request.skill_id}")
        summary, statistics, figures = handler(frame, request)
        artifacts = [
            self._save_figure(session_id, title, figure, metadata)
            for title, figure, metadata in figures
        ]
        artifacts.append(self._save_records_csv(session_id, records))
        return AnalysisResult(
            skill_id=request.skill_id,
            summary=summary,
            statistics=statistics,
            artifacts=artifacts,
        )

    def _dataset_summary(
        self, frame: pd.DataFrame, request: AnalysisRequest
    ) -> tuple[str, dict[str, Any], list[tuple[str, Any, dict[str, Any]]]]:
        del request
        colors = frame["emission_color"].fillna("unknown").astype(str).value_counts().head(10)
        stats = {
            "devices": int(len(frame)),
            "papers": int(frame["doi"].dropna().nunique()),
            "emitters": int(frame["final_emitter"].dropna().nunique()),
            "eqe_reported": int(frame["eqe_max"].notna().sum()),
            "median_eqe": _finite(frame["eqe_max"].median()),
            "colors": {str(key): int(value) for key, value in colors.items()},
        }
        figure, axis = plt.subplots(figsize=(8.4, 4.8))
        colors.sort_values().plot.barh(ax=axis, color="#0aa89e")
        axis.set_title("Device records by emission color")
        axis.set_xlabel("Devices")
        axis.set_ylabel("")
        _finish(figure)
        summary = (
            f"The selected scope contains {stats['devices']:,} devices from "
            f"{stats['papers']:,} papers and {stats['emitters']:,} reported final emitters."
        )
        return summary, stats, [("Dataset composition", figure, {"chart": "bar"})]

    def _data_quality_profile(
        self, frame: pd.DataFrame, request: AnalysisRequest
    ) -> tuple[str, dict[str, Any], list[tuple[str, Any, dict[str, Any]]]]:
        del request
        fields = list(NUMERIC_FIELDS) + list(CATEGORICAL_FIELDS)
        completeness = (frame[fields].notna().mean() * 100).sort_values()
        stats = {str(key): round(float(value), 2) for key, value in completeness.items()}
        figure, axis = plt.subplots(figsize=(9.2, 6.2))
        completeness.plot.barh(ax=axis, color="#2878b5")
        axis.set_xlim(0, 100)
        axis.set_xlabel("Completeness (%)")
        axis.set_title("Field completeness")
        _finish(figure)
        weakest = ", ".join(completeness.head(3).index.tolist())
        return (
            f"Completeness was measured before analysis. The least complete fields are {weakest}.",
            stats,
            [("Data quality profile", figure, {"chart": "bar"})],
        )

    def _univariate_distribution(
        self, frame: pd.DataFrame, request: AnalysisRequest
    ) -> tuple[str, dict[str, Any], list[tuple[str, Any, dict[str, Any]]]]:
        field = request.metric or request.x_field or "eqe_max"
        series = self._numeric_series(frame, field)
        stats = {
            "field": field,
            "count": int(series.size),
            "mean": _finite(series.mean()),
            "median": _finite(series.median()),
            "std": _finite(series.std()),
            "minimum": _finite(series.min()),
            "q25": _finite(series.quantile(0.25)),
            "q75": _finite(series.quantile(0.75)),
            "maximum": _finite(series.max()),
        }
        figure, axis = plt.subplots(figsize=(8.6, 5.2))
        bins = min(36, max(8, int(math.sqrt(series.size))))
        axis.hist(series, bins=bins, color="#0aa89e", alpha=0.86, edgecolor="white")
        axis.axvline(series.median(), color="#e45756", linewidth=2, label="Median")
        axis.set_title(f"Distribution of {field}")
        axis.set_xlabel(field)
        axis.set_ylabel("Devices")
        axis.legend(frameon=False)
        _finish(figure)
        summary = (
            f"{field} is available for {series.size:,} devices; the median is "
            f"{stats['median']:.3g} and the interquartile range is "
            f"{stats['q25']:.3g} to {stats['q75']:.3g}."
        )
        return summary, stats, [(f"{field} distribution", figure, {"field": field})]

    def _bivariate_relationship(
        self, frame: pd.DataFrame, request: AnalysisRequest
    ) -> tuple[str, dict[str, Any], list[tuple[str, Any, dict[str, Any]]]]:
        x_field = request.x_field or "layer_count"
        y_field = request.y_field or request.metric or "eqe_max"
        pair = frame[[x_field, y_field]].dropna()
        if len(pair) < 3:
            raise ValueError(f"At least three paired values are required for {x_field} and {y_field}")
        pearson = float(pair[x_field].corr(pair[y_field], method="pearson"))
        spearman = float(pair[x_field].corr(pair[y_field], method="spearman"))
        slope, intercept = np.polyfit(pair[x_field], pair[y_field], deg=1)
        ordered_x = np.linspace(pair[x_field].min(), pair[x_field].max(), 100)
        figure, axis = plt.subplots(figsize=(8.2, 5.4))
        axis.scatter(pair[x_field], pair[y_field], s=25, alpha=0.55, color="#2878b5")
        axis.plot(ordered_x, slope * ordered_x + intercept, color="#e45756", linewidth=2)
        axis.set_xlabel(x_field)
        axis.set_ylabel(y_field)
        axis.set_title(f"{y_field} versus {x_field}")
        _finish(figure)
        stats = {
            "x_field": x_field,
            "y_field": y_field,
            "paired_count": int(len(pair)),
            "pearson_r": round(pearson, 5),
            "spearman_rho": round(spearman, 5),
            "linear_slope": round(float(slope), 6),
        }
        summary = (
            f"Across {len(pair):,} paired records, Pearson r={pearson:.3f} and "
            f"Spearman rho={spearman:.3f}. This is association, not causal evidence."
        )
        return summary, stats, [("Bivariate relationship", figure, stats)]

    def _group_comparison(
        self, frame: pd.DataFrame, request: AnalysisRequest
    ) -> tuple[str, dict[str, Any], list[tuple[str, Any, dict[str, Any]]]]:
        group_field = request.group_field or "emission_color"
        metric = request.metric or "eqe_max"
        data = frame[[group_field, metric]].dropna()
        if data.empty:
            raise ValueError(f"No paired values are available for {group_field} and {metric}")
        top_groups = data[group_field].astype(str).value_counts().head(8).index
        data = data[data[group_field].astype(str).isin(top_groups)].copy()
        data[group_field] = data[group_field].astype(str)
        ordered = list(top_groups)
        arrays = [data.loc[data[group_field] == group, metric].to_numpy() for group in ordered]
        medians = {group: _finite(np.median(values)) for group, values in zip(ordered, arrays)}
        counts = {group: int(values.size) for group, values in zip(ordered, arrays)}
        figure, axis = plt.subplots(figsize=(9.4, 5.6))
        boxes = axis.boxplot(arrays, tick_labels=ordered, patch_artist=True, showfliers=False)
        for patch in boxes["boxes"]:
            patch.set_facecolor("#77c8bf")
        axis.set_ylabel(metric)
        axis.set_xlabel(group_field)
        axis.set_title(f"{metric} by {group_field}")
        axis.tick_params(axis="x", rotation=25)
        _finish(figure)
        stats = {"group_field": group_field, "metric": metric, "counts": counts, "medians": medians}
        best = max(medians, key=lambda key: medians[key] if medians[key] is not None else -math.inf)
        summary = (
            f"The comparison includes {len(data):,} records across {len(ordered)} groups. "
            f"{best} has the highest observed median {metric}; extraction coverage and group size "
            "must be considered before interpretation."
        )
        return summary, stats, [("Grouped performance comparison", figure, stats)]

    def _correlation_matrix(
        self, frame: pd.DataFrame, request: AnalysisRequest
    ) -> tuple[str, dict[str, Any], list[tuple[str, Any, dict[str, Any]]]]:
        del request
        minimum_observations = min(10, max(3, len(frame)))
        available = [
            field
            for field in NUMERIC_FIELDS
            if frame[field].notna().sum() >= minimum_observations
            and frame[field].dropna().nunique() >= 2
        ]
        if not available:
            raise ValueError(
                "At least one varying numeric field with "
                f"{minimum_observations} observations are required"
            )
        correlation = frame[available].corr(method="spearman")
        figure, axis = plt.subplots(figsize=(9.2, 7.4))
        image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_xticks(range(len(available)), labels=available, rotation=45, ha="right")
        axis.set_yticks(range(len(available)), labels=available)
        axis.set_title("Spearman correlation matrix")
        figure.colorbar(image, ax=axis, fraction=0.04, pad=0.04)
        _finish(figure)
        matrix = {
            row: {column: _finite(correlation.loc[row, column]) for column in available}
            for row in available
        }
        pairs: list[tuple[float, str, str]] = []
        for index, left in enumerate(available):
            for right in available[index + 1 :]:
                value = float(correlation.loc[left, right])
                if math.isfinite(value):
                    pairs.append((abs(value), left, right))
        pairs.sort(reverse=True)
        if pairs:
            strongest = pairs[0]
            summary = (
                f"The strongest absolute Spearman association is between {strongest[1]} and "
                f"{strongest[2]} (|rho|={strongest[0]:.3f}). Correlation does not establish causality."
            )
        else:
            summary = (
                "The available numeric fields do not have enough overlapping variation for a "
                "finite pairwise Spearman coefficient."
            )
        statistics = {
            "fields": available,
            "minimum_observations": minimum_observations,
            "matrix": matrix,
        }
        return summary, statistics, [("Correlation matrix", figure, {})]

    @staticmethod
    def _numeric_series(frame: pd.DataFrame, field: str) -> pd.Series:
        if field not in frame.columns or field not in NUMERIC_FIELDS:
            raise ValueError(f"Unsupported numeric field: {field}")
        series = pd.to_numeric(frame[field], errors="coerce").dropna()
        if series.size < 2:
            raise ValueError(f"At least two values are required for {field}")
        return series

    def _save_figure(
        self,
        session_id: str,
        title: str,
        figure: Any,
        metadata: dict[str, Any],
    ) -> Artifact:
        target_dir = self.artifact_dir / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.png"
        target = target_dir / filename
        figure.savefig(target, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return self.repository.add_artifact(
            session_id,
            "chart",
            title,
            filename,
            "image/png",
            f"/api/optovlab/artifacts/{session_id}/{filename}",
            metadata,
        )

    def _save_records_csv(self, session_id: str, records: list[dict[str, Any]]) -> Artifact:
        target_dir = self.artifact_dir / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"analysis-records-{uuid.uuid4().hex}.csv"
        target = target_dir / filename
        fields = ["id", "doi", "device_label", *NUMERIC_FIELDS, *CATEGORICAL_FIELDS]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        return self.repository.add_artifact(
            session_id,
            "data",
            "Analysis records",
            filename,
            "text/csv",
            f"/api/optovlab/artifacts/{session_id}/{filename}",
            {"record_count": len(records)},
        )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _finish(figure: Any) -> None:
    figure.tight_layout()
