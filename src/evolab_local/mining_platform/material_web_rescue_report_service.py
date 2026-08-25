from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.storage.database import now_iso


RESOLVED_LINK_STATUSES = {"confirmed", "matched_candidate", "matched_local"}


class MaterialWebRescueReportService:
    """Build a current, auditable report from a frozen rescue inventory."""

    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.material_resolution = MaterialResolutionService(config)

    def generate(
        self,
        inventory_path: Path,
        output_dir: Path,
        *,
        apply_result_path: Path | None = None,
    ) -> dict[str, Any]:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        source_items = inventory.get("items")
        if not isinstance(source_items, list):
            raise ValueError("Rescue inventory must contain an items list.")

        bundles: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for source in source_items:
            paper_id = str(source["paper_id"])
            bundle = bundles.get(paper_id)
            if bundle is None:
                bundle = self.material_resolution.get_material_structure_bundle(paper_id)
                if bundle is None:
                    raise ValueError(f"Paper bundle no longer exists: {paper_id}")
                bundles[paper_id] = bundle
            rows.append(self._current_row(source, bundle))

        summary = dict(sorted(Counter(row["resolution"] for row in rows).items()))
        unresolved_summary = dict(
            sorted(
                Counter(
                    row["unresolved_reason"]["code"]
                    for row in rows
                    if row["resolution"] == "unresolved" and row["unresolved_reason"]
                ).items()
            )
        )
        audit_errors = [
            {"target_id": row["target_id"], "errors": row["audit_errors"]}
            for row in rows
            if row["audit_errors"]
        ]
        apply_result = (
            json.loads(apply_result_path.read_text(encoding="utf-8"))
            if apply_result_path is not None
            else None
        )
        report = {
            "schema_version": "material_web_rescue_report_v1",
            "run_id": inventory.get("run_id") or inventory_path.parent.name,
            "created_at": now_iso(),
            "database": str(self.config.paths.sqlite_path),
            "inventory_path": str(inventory_path.resolve()),
            "apply_result_path": (
                str(apply_result_path.resolve()) if apply_result_path is not None else None
            ),
            "target_count": len(rows),
            "summary": summary,
            "unresolved_summary": unresolved_summary,
            "acceptance_policy": {
                "identity_verdict": "exact_match",
                "minimum_confidence": 0.95,
                "minimum_source_domains": 2,
                "requires_identity_source": True,
                "requires_strong_structure_source": True,
                "requires_empty_conflicts": True,
                "decimer_requires_independent_structured_corroboration": True,
            },
            "audit_error_count": len(audit_errors),
            "audit_errors": audit_errors,
            "apply_result": apply_result,
            "items": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "material-web-rescue-report.json"
        markdown_path = output_dir / "material-web-rescue-report.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(self._markdown(report), encoding="utf-8")
        return report

    @staticmethod
    def _current_row(source: dict[str, Any], bundle: Any) -> dict[str, Any]:
        material_id = str(source["paper_material_id"])
        material = next(
            (item for item in bundle.materials if item.paper_material_id == material_id),
            None,
        )
        link = next(
            (item for item in bundle.links if item.paper_material_id == material_id),
            None,
        )
        task = next(
            (item for item in bundle.tasks if item.paper_material_id == material_id),
            None,
        )
        candidates = [
            item for item in bundle.structure_candidates if item.paper_material_id == material_id
        ]
        judgments = [
            item for item in bundle.identity_judgments if item.paper_material_id == material_id
        ]
        review_events = [
            item for item in bundle.material_review_events if item.paper_material_id == material_id
        ]
        web_candidates = [item for item in candidates if item.provider == "web_rescue_agent"]
        web_candidate = max(web_candidates, key=lambda item: item.updated_at, default=None)
        web_judgments = [item for item in judgments if item.provider == "codex_web_research"]
        web_judgment = max(web_judgments, key=lambda item: item.created_at, default=None)
        web_events = [
            item
            for item in review_events
            if web_candidate is not None
            and item.structure_candidate_id == web_candidate.structure_candidate_id
        ]
        web_event = max(web_events, key=lambda item: item.created_at, default=None)
        global_material = next(
            (
                item
                for item in bundle.global_materials
                if link is not None and item.global_material_id == link.global_material_id
            ),
            None,
        )
        resolution = _resolution(link, task, web_candidate)
        audit_errors = _audit_errors(
            resolution=resolution,
            link=link,
            global_material=global_material,
            candidate=web_candidate,
            judgment=web_judgment,
            event=web_event,
        )
        material_context = source.get("material_context") or {}
        rejected_judgments = source.get("identity_judgments") or []
        conflicts = _unique(
            str(conflict)
            for judgment in rejected_judgments
            for conflict in (judgment.get("conflicts") or [])
            if conflict
        )
        return {
            "target_id": source["target_id"],
            "paper_id": source["paper_id"],
            "doi": source.get("doi"),
            "title": source.get("title"),
            "journal": source.get("journal"),
            "candidate_run_id": source["candidate_run_id"],
            "paper_material_id": material_id,
            "material_mentions": (
                material.mention_list
                if material is not None
                else source.get("material_mentions", [])
            ),
            "full_name_in_paper": (
                material.full_name_in_paper
                if material is not None and material.full_name_in_paper
                else material_context.get("full_name_in_paper")
            ),
            "resolution": resolution,
            "unresolved_reason": _unresolved_reason(
                resolution,
                material_context,
                source.get("visual_counts") or {},
                conflicts,
            ),
            "current_link": link.model_dump(mode="json") if link else None,
            "current_task": task.model_dump(mode="json") if task else None,
            "web_rescue_candidate": (
                web_candidate.model_dump(mode="json") if web_candidate else None
            ),
            "web_rescue_judgment": (web_judgment.model_dump(mode="json") if web_judgment else None),
            "web_rescue_review_event": (web_event.model_dump(mode="json") if web_event else None),
            "global_material": (
                global_material.model_dump(mode="json") if global_material else None
            ),
            "conflicts_from_rejected_candidates": conflicts,
            "visual_counts": source.get("visual_counts") or {},
            "audit_errors": audit_errors,
        }

    @staticmethod
    def _markdown(report: dict[str, Any]) -> str:
        lines = [
            "# Material Web Rescue Report",
            "",
            f"- Run: `{report['run_id']}`",
            f"- Generated: `{report['created_at']}`",
            f"- Database: `{report['database']}`",
            f"- Frozen targets: **{report['target_count']}**",
            f"- Audit errors: **{report['audit_error_count']}**",
            "",
            "## Current Status",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
        for status, count in report["summary"].items():
            lines.append(f"| `{status}` | {count} |")

        lines.extend(
            [
                "",
                "## Method And Safety Gates",
                "",
                "The frozen inventory contains materials whose previous public candidate was rejected. Research used the paper/SI, public chemical databases, independent literature or patents, name-to-structure conversion, RDKit validation, and structure images when useful.",
                "",
                "Automatic acceptance required all of the following:",
                "",
                "1. Exact paper-local identity match with confidence >= 0.95.",
                "2. No unresolved identity or connectivity conflict.",
                "3. Evidence from at least two source domains.",
                "4. Both identity evidence and a strong structure-bearing source.",
                "5. RDKit-valid structure written through the normal review service, including an identity judgment, global-material link, and undoable review event.",
                "",
                "A DECIMER result alone is never enough for automatic acceptance. It must agree with an independent structured source; generic scaffolds, orbital panels, partial structures, and uncertain label bindings remain for review.",
                "",
                "## Remaining Work By Reason",
                "",
                "| Reason code | Count |",
                "| --- | ---: |",
            ]
        )
        for reason, count in report["unresolved_summary"].items():
            lines.append(f"| `{reason}` | {count} |")

        accepted = [item for item in report["items"] if item["resolution"] == "accepted_web_rescue"]
        lines.extend(
            [
                "",
                "## Verified And Accepted",
                "",
                "Only exact identity matches with RDKit-valid structures, at least two independent source domains, no unresolved conflicts, and a complete review event were automatically accepted.",
                "",
                "| DOI | Material | Canonical name | InChIKey | Method | Provenance |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in accepted:
            candidate = item["web_rescue_candidate"] or {}
            evidence = candidate.get("evidence") or {}
            sources = evidence.get("sources") or []
            links = "<br>".join(
                f"[{_escape(source.get('title') or source.get('url'))}]({source.get('url')})"
                for source in sources
                if source.get("url")
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(item.get("doi") or item["paper_id"]),
                        _escape(" / ".join(item["material_mentions"])),
                        _escape(candidate.get("canonical_name") or ""),
                        _escape(candidate.get("inchi_key") or ""),
                        _escape(str(evidence.get("structure_method") or "")),
                        links,
                    ]
                )
                + " |"
            )

        corrected = [
            item
            for item in report["items"]
            if (item.get("web_rescue_candidate") or {}).get("status") == "rejected"
            and (item.get("web_rescue_review_event") or {}).get("action") == "reject"
        ]
        lines.extend(
            [
                "",
                "## Rejected During Semantic Audit",
                "",
                "These candidates passed syntactic structure checks but failed a later chemistry-level identity or scaffold check. Their accept event was undone before the candidate was rejected.",
                "",
                "| DOI | Material | Rejected InChIKey | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in corrected:
            candidate = item["web_rescue_candidate"] or {}
            event = item["web_rescue_review_event"] or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(item.get("doi") or item["paper_id"]),
                        _escape(" / ".join(item["material_mentions"])),
                        _escape(candidate.get("inchi_key") or ""),
                        _escape(event.get("message") or "Semantic audit rejection"),
                    ]
                )
                + " |"
            )

        unresolved = [item for item in report["items"] if item["resolution"] == "unresolved"]
        lines.extend(
            [
                "",
                "## Still Unresolved",
                "",
                "These records were deliberately not accepted. A rejected search hit is not evidence for a replacement structure.",
                "",
                "| DOI | Material | Reason | Next action |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in unresolved:
            reason = item["unresolved_reason"] or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(item.get("doi") or item["paper_id"]),
                        _escape(" / ".join(item["material_mentions"])),
                        _escape(reason.get("reason") or "Insufficient evidence"),
                        _escape(reason.get("next_action") or "Manual review"),
                    ]
                )
                + " |"
            )

        terminal = [
            item
            for item in report["items"]
            if item["resolution"] in {"identity_only", "out_of_scope_structure"}
        ]
        lines.extend(
            [
                "",
                "## Non-SMILES Terminal Records",
                "",
                "These materials keep their device identity but do not require a single small-molecule SMILES in the first-phase database.",
                "",
                "| DOI | Material | Status | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in terminal:
            scope = ((item.get("current_link") or {}).get("evidence") or {}).get(
                "structure_scope"
            ) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(item.get("doi") or item["paper_id"]),
                        _escape(" / ".join(item["material_mentions"])),
                        _escape(item["resolution"]),
                        _escape(scope.get("reason") or "Non-single molecular structure"),
                    ]
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)


def _resolution(link: Any, task: Any, web_candidate: Any) -> str:
    if (
        web_candidate is not None
        and web_candidate.status == "accepted"
        and link is not None
        and link.global_material_id
    ):
        return "accepted_web_rescue"
    if link is not None and link.global_material_id and link.match_status in RESOLVED_LINK_STATUSES:
        return "resolved_other"
    if link is not None and link.match_status in {
        "identity_only",
        "out_of_scope_structure",
        "ambiguous",
    }:
        return str(link.match_status)
    if task is not None and task.assigned_strategy in {
        "identity_only",
        "out_of_scope_structure",
    }:
        return str(task.assigned_strategy)
    return "unresolved"


def _audit_errors(
    *,
    resolution: str,
    link: Any,
    global_material: Any,
    candidate: Any,
    judgment: Any,
    event: Any,
) -> list[str]:
    if resolution != "accepted_web_rescue":
        return []
    checks = {
        "candidate is not accepted": candidate is None or candidate.status != "accepted",
        "identity judgment is not exact_match": (
            judgment is None or judgment.verdict != "exact_match"
        ),
        "accept review event is missing": event is None or event.action != "accept",
        "resolved global link is missing": (
            link is None
            or not link.global_material_id
            or link.match_status not in RESOLVED_LINK_STATUSES
        ),
        "global material is missing": global_material is None,
        "candidate/global InChIKey mismatch": (
            candidate is not None
            and global_material is not None
            and candidate.inchi_key != global_material.inchi_key
        ),
    }
    return [message for message, failed in checks.items() if failed]


def _unresolved_reason(
    resolution: str,
    material_context: dict[str, Any],
    visual_counts: dict[str, Any],
    conflicts: list[str],
) -> dict[str, str] | None:
    if resolution != "unresolved":
        return None
    full_name = material_context.get("full_name_in_paper")
    crops = int(visual_counts.get("crops") or 0)
    bindings = int(visual_counts.get("bindings") or 0)
    if conflicts and any(
        "mptbc" in item.lower() or "tricarbonitrile" in item.lower() for item in conflicts
    ):
        return {
            "code": "paper_identity_conflict",
            "reason": "Paper-local name and public isomer identity conflict; automatic acceptance is unsafe.",
            "next_action": "Correct the paper-local name from primary evidence, then rerun identity resolution.",
        }
    if full_name:
        return {
            "code": "full_name_not_structurally_corroborated",
            "reason": "A paper full name exists, but no unique independently corroborated structure was established.",
            "next_action": "Verify the systematic name against the article/SI and a structured source or labeled OCSR result.",
        }
    if bindings:
        return {
            "code": "visual_candidate_requires_review",
            "reason": "A labeled visual candidate exists but has not passed identity and structure validation.",
            "next_action": "Review the crop/label binding and edit or accept the OCSR structure.",
        }
    if crops:
        return {
            "code": "article_specific_visual_binding_required",
            "reason": "Only a paper-specific label is available; figure crops exist but none is reliably bound to this material.",
            "next_action": "Bind a clean labeled 2D structure crop, run DECIMER, then verify against article context.",
        }
    return {
        "code": "insufficient_identity_evidence",
        "reason": "The record has only an ambiguous abbreviation or paper-local label and no usable structure crop.",
        "next_action": "Locate primary article/SI identity evidence or enter the structure manually.",
    }


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
