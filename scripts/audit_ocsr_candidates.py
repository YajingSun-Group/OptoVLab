#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
from threading import Lock
from typing import Any

from rdkit import Chem
from rdkit.Chem import Draw

from evolab_local.mining_platform.batch_worker_service import BatchWorkerService
from evolab_local.mining_platform.core.config import load_config
from evolab_local.mining_platform.external.openai_compatible_client import (
    OpenAICompatibleVisionClient,
    image_path_to_data_url,
)
from evolab_local.mining_platform.material_resolution_service import (
    MaterialResolutionService,
    normalize_material_alias,
)
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.schemas.material_structure import MaterialReviewAction


@dataclass
class AuditRecord:
    paper_id: str
    paper_material_id: str
    structure_candidate_id: str
    material_names: list[str]
    candidate_smiles: str | None
    decision: str
    applied: bool
    reason: str
    flash: dict[str, Any] | None = None
    plus: dict[str, Any] | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively audit DECIMER OCSR candidates with dual VLM consensus."
    )
    parser.add_argument("--start-batch", type=int)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--targets",
        type=Path,
        help=(
            "JSONL file containing paper_id and paper_material_id (or "
            "paper_material_ids). This is mutually exclusive with a batch range."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/mining_platform/mining_platform.yaml")
    )
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--flash-model", default="qwen3.6-flash")
    parser.add_argument("--review-model", default="qwen3.6-plus")
    parser.add_argument("--min-confidence", type=float, default=0.99)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/mining_platform/reports/ocsr_dual_vlm_audit.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.targets:
        if args.start_batch is not None or args.end_batch is not None:
            raise SystemExit("Use either --targets or a batch range, not both.")
        target_pairs = _load_target_pairs(args.targets)
        if not target_pairs:
            raise SystemExit("The target file did not contain any material targets.")
        papers = {paper_id for paper_id, _ in target_pairs}
    else:
        if (
            args.start_batch is None
            or args.end_batch is None
            or args.start_batch < 1
            or args.end_batch < args.start_batch
        ):
            raise SystemExit("Provide a valid --start-batch/--end-batch range or --targets.")
        target_pairs = None
        papers = _paper_ids(config, args.start_batch, args.end_batch, args.batch_size)
    work = _candidate_work(config, papers, target_pairs)
    if args.limit is not None:
        work = work[: max(0, args.limit)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    depiction_dir = args.output.parent / "ocsr_audit_depictions"
    depiction_dir.mkdir(parents=True, exist_ok=True)
    output_lock = Lock()

    grouped_work: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in work:
        grouped_work[(item["paper_id"], item["paper_material_id"])].append(item)

    def run(items: list[dict[str, Any]]) -> list[AuditRecord]:
        records: list[AuditRecord] = []
        for item in items:
            record = _audit_candidate(config, item, args, depiction_dir)
            records.append(record)
            if record.decision == "accepted":
                break
        return records

    counts: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(run, items): items for items in grouped_work.values()}
            completed = 0
            for future in as_completed(futures):
                items = futures[future]
                try:
                    records = future.result()
                except Exception as exc:  # keep the batch auditable and resumable.
                    first = items[0]
                    records = [
                        AuditRecord(
                            paper_id=first["paper_id"],
                            paper_material_id=first["paper_material_id"],
                            structure_candidate_id=first["structure_candidate_id"],
                            material_names=first["material_names"],
                            candidate_smiles=first.get("candidate_smiles"),
                            decision="error",
                            applied=False,
                            reason="audit_exception",
                            error=repr(exc),
                        )
                    ]
                for record in records:
                    completed += 1
                    counts[record.decision] = counts.get(record.decision, 0) + 1
                    with output_lock:
                        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                        handle.flush()
                    print(
                        f"{completed}/{len(work)} {record.paper_id} "
                        f"{record.paper_material_id} {record.decision} "
                        f"applied={record.applied}",
                        flush=True,
                    )
    print(json.dumps({"total": len(work), "counts": counts, "output": str(args.output)}))


def _paper_ids(config, start_batch: int, end_batch: int, batch_size: int) -> set[str]:
    service = BatchWorkerService(config)
    paper_ids: set[str] = set()
    for batch_number in range(start_batch, end_batch + 1):
        detail = service.get_review_batch(batch_number - 1, batch_size=batch_size)
        for item in detail.papers:
            if not item.is_confirmed and not item.is_excluded:
                paper_ids.add(item.paper.paper_id)
    return paper_ids


def _candidate_work(
    config,
    paper_ids: set[str],
    target_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    service = MaterialResolutionService(config)
    work: list[dict[str, Any]] = []
    for paper_id in sorted(paper_ids):
        bundle = service.get_material_structure_bundle(paper_id)
        if not bundle:
            continue
        materials = {item.paper_material_id: item for item in bundle.materials}
        accepted_material_ids = {
            item.paper_material_id
            for item in bundle.structure_candidates
            if item.status == "accepted"
        }
        for candidate in bundle.structure_candidates:
            if candidate.provider != "decimer_ocsr" or candidate.status != "pending_review":
                continue
            if target_pairs is not None and (
                paper_id,
                candidate.paper_material_id,
            ) not in target_pairs:
                continue
            if candidate.paper_material_id in accepted_material_ids:
                continue
            material = materials.get(candidate.paper_material_id)
            if not material:
                continue
            names = _material_names(material)
            work.append(
                {
                    "paper_id": paper_id,
                    "paper_material_id": candidate.paper_material_id,
                    "structure_candidate_id": candidate.structure_candidate_id,
                    "material_names": names,
                    "query_text": candidate.query_text,
                    "candidate_smiles": candidate.canonical_smiles or candidate.raw_smiles,
                    "evidence": candidate.evidence,
                    "created_at": candidate.created_at,
                }
            )
    return sorted(
        work,
        key=lambda item: (
            item["paper_id"],
            item["paper_material_id"],
            item["created_at"],
        ),
        reverse=True,
    )


def _load_target_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict) or not payload.get("paper_id"):
                raise ValueError(f"Missing paper_id on {path}:{line_number}.")
            material_ids = payload.get("paper_material_ids")
            if material_ids is None:
                material_ids = [payload.get("paper_material_id")]
            if not isinstance(material_ids, list) or not all(material_ids):
                raise ValueError(f"Missing paper material IDs on {path}:{line_number}.")
            pairs.update(
                (str(payload["paper_id"]), str(material_id)) for material_id in material_ids
            )
    return pairs


def _material_names(material) -> list[str]:
    values = [
        material.entity_label,
        material.abbreviation,
        material.full_name_in_paper,
        material.normalized_name,
        material.canonical_name,
        *material.mention_list,
    ]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_material_alias(value)
        if not value or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _audit_candidate(config, item: dict[str, Any], args, depiction_dir: Path) -> AuditRecord:
    evidence = item["evidence"]
    smiles = item.get("candidate_smiles")
    preflight_reason = _preflight_reason(item, evidence)
    if preflight_reason:
        return _record(item, "skipped", False, preflight_reason)
    depiction_path = depiction_dir / f"{item['structure_candidate_id']}.png"
    _write_depiction(smiles, depiction_path)
    source_path = _first_existing_path(
        evidence.get("highlighted_source_figure_path"),
        evidence.get("source_figure_path"),
    )
    crop_path = _first_existing_path(evidence.get("crop_path"))
    if not source_path or not crop_path:
        return _record(item, "skipped", False, "missing_source_or_crop_image")
    messages = _audit_messages(item, source_path, crop_path, depiction_path)
    provider = config.llm.providers["qwen"]
    client = OpenAICompatibleVisionClient(provider)
    flash = client.generate_json(
        messages, model=args.flash_model, temperature=0, max_tokens=900
    ).parsed_json
    if not _valid_decision(flash):
        return _record(item, "uncertain", False, "flash_invalid_or_uncertain", flash=flash)
    if flash["decision"] == "uncertain":
        return _record(item, "uncertain", False, "flash_uncertain", flash=flash)
    plus = client.generate_json(
        messages, model=args.review_model, temperature=0, max_tokens=900
    ).parsed_json
    if not _valid_decision(plus):
        return _record(item, "uncertain", False, "review_model_invalid_or_uncertain", flash, plus)
    decision = str(flash["decision"])
    consensus = decision == str(plus["decision"])
    confident = min(float(flash["confidence"]), float(plus["confidence"])) >= args.min_confidence
    if not consensus or not confident:
        return _record(item, "uncertain", False, "dual_model_consensus_not_reached", flash, plus)
    if decision == "exact_match" and not _acceptance_gate(item, evidence, flash, plus):
        return _record(
            item, "uncertain", False, "exact_match_failed_deterministic_gate", flash, plus
        )
    if decision not in {"exact_match", "reject"}:
        return _record(item, "uncertain", False, "unsupported_consensus", flash, plus)
    applied = False
    if not args.dry_run:
        review = MaterialStructureReviewService(config)
        message = (
            "Dual-VLM assisted OCSR audit: "
            f"flash={flash['decision']}({flash['confidence']}); "
            f"review={plus['decision']}({plus['confidence']}); "
            f"flash_reason={flash.get('reason')}; review_reason={plus.get('reason')}"
        )
        action = MaterialReviewAction(actor="ocsr_dual_vlm_auditor", message=message)
        if decision == "exact_match":
            review.accept_structure_candidate(item["structure_candidate_id"], action)
        else:
            review.reject_structure_candidate(item["structure_candidate_id"], action)
        applied = True
    return _record(
        item,
        "accepted" if decision == "exact_match" else "rejected",
        applied,
        "dual_model_consensus",
        flash,
        plus,
    )


def _preflight_reason(item: dict[str, Any], evidence: dict[str, Any]) -> str | None:
    smiles = item.get("candidate_smiles")
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        return "candidate_is_not_rdkit_valid"
    response = evidence.get("decimer_response")
    if not isinstance(response, dict):
        response = {}
    rdkit_valid = evidence.get("rdkit_valid", response.get("rdkit_valid"))
    if rdkit_valid is not True:
        return "decimer_response_not_rdkit_valid"
    warning = evidence.get("structure_quality_warning", response.get("structure_quality_warning"))
    if warning:
        return "candidate_has_structure_quality_warning"
    return None


def _acceptance_gate(
    item: dict[str, Any], evidence: dict[str, Any], flash: dict[str, Any], plus: dict[str, Any]
) -> bool:
    response = evidence.get("decimer_response") or {}
    model_decision = evidence.get("model_decision", response.get("model_decision"))
    if model_decision != "matched":
        return False
    model_confidence = evidence.get("model_confidence", response.get("model_confidence"))
    if float(model_confidence or 0) < 0.98:
        return False
    if not flash.get("crop_is_complete_named_material") or not plus.get(
        "crop_is_complete_named_material"
    ):
        return False
    paper_names = {normalize_material_alias(value) for value in item["material_names"]}
    observed = {
        normalize_material_alias(value)
        for value in (
            response.get("model_observed_label"),
            response.get("reviewed_observed_label"),
            evidence.get("model_observed_label"),
            evidence.get("reviewed_observed_label"),
            item.get("query_text"),
        )
        if value
    }
    return bool(observed & paper_names)


def _audit_messages(
    item: dict[str, Any], source_path: Path, crop_path: Path, depiction_path: Path
) -> list[dict[str, Any]]:
    prompt = f"""
You are a conservative chemical-structure auditor. Compare three images:
1. the source-paper figure or highlighted context;
2. the DECIMER crop assigned to the material;
3. an RDKit depiction generated from the candidate SMILES.

Paper material ID: {item["paper_material_id"]}
Paper material names: {json.dumps(item["material_names"], ensure_ascii=False)}
Candidate SMILES: {item["candidate_smiles"]}

Return one JSON object with exactly these fields:
{{"decision":"exact_match|reject|uncertain","confidence":0.0,
"crop_is_complete_named_material":true,"observed_label":null,
"differences":[],"reason":""}}

Use exact_match only when atom types, ring topology, bond orders, formal charges,
substituent identities, counts, and attachment positions all match. A generic
scaffold, R group, reaction intermediate, fragment, orbital diagram, partial
structure, unreadable crop, or material-label mismatch is reject. Use uncertain
whenever a small positional or bond difference cannot be verified. Do not infer
missing atoms from the material name.
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in (source_path, crop_path, depiction_path):
        content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(path)}})
    return [{"role": "user", "content": content}]


def _valid_decision(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("decision") not in {"exact_match", "reject", "uncertain"}:
        return False
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return False
    return 0 <= confidence <= 1


def _write_depiction(smiles: str, path: Path) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse candidate SMILES.")
    Draw.MolToImage(molecule, size=(900, 600)).save(path)


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            return path
    return None


def _record(
    item: dict[str, Any],
    decision: str,
    applied: bool,
    reason: str,
    flash: dict[str, Any] | None = None,
    plus: dict[str, Any] | None = None,
) -> AuditRecord:
    return AuditRecord(
        paper_id=item["paper_id"],
        paper_material_id=item["paper_material_id"],
        structure_candidate_id=item["structure_candidate_id"],
        material_names=item["material_names"],
        candidate_smiles=item.get("candidate_smiles"),
        decision=decision,
        applied=applied,
        reason=reason,
        flash=flash,
        plus=plus,
    )


if __name__ == "__main__":
    main()
