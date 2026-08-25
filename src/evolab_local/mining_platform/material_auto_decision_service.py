from __future__ import annotations

from evolab_local.mining_platform.core.config import MiningPlatformConfig
from evolab_local.mining_platform.material_public_resolver_service import (
    is_plausible_public_structure_candidate,
)
from evolab_local.mining_platform.material_resolution_service import MaterialResolutionService
from evolab_local.mining_platform.material_structure_review_service import (
    MaterialStructureReviewService,
)
from evolab_local.mining_platform.schemas.material_structure import (
    MaterialAutoDecision,
    MaterialAutoDecisionResult,
    MaterialIdentityJudgment,
    MaterialReviewAction,
    MaterialStructureCandidate,
    PaperLocalMaterial,
)
from evolab_local.mining_platform.storage.database import Database
from evolab_local.mining_platform.storage.repositories import (
    MaterialIdentityJudgmentRepository,
)


AUTO_DECISION_ACTOR = "automation_policy"
AUTO_ACCEPT_VERDICTS = {"exact_match", "likely_match"}
AUTO_REJECT_VERDICTS = {"conflict", "rejected"}
AUTO_ACCEPT_RECOMMENDED_ACTIONS = {"ready_for_human_accept"}
AUTO_REJECT_RECOMMENDED_ACTIONS = {"reject_candidate"}


class MaterialAutoDecisionService:
    """Apply conservative automated material-structure review decisions.

    The service deliberately reuses MaterialStructureReviewService so automation
    writes the same review events and remains undoable.
    """

    def __init__(self, config: MiningPlatformConfig) -> None:
        self.config = config
        self.database = Database(config.paths.sqlite_path)
        self.material_resolution = MaterialResolutionService(config)
        self.review_service = MaterialStructureReviewService(config)
        self.identity_judgments = MaterialIdentityJudgmentRepository(self.database)

    def init_runtime(self) -> None:
        self.review_service.init_runtime()

    def apply_paper_auto_decisions(
        self,
        paper_id: str,
        *,
        dry_run: bool = False,
        actor: str = AUTO_DECISION_ACTOR,
        accept_min_confidence: float | None = None,
        reject_min_confidence: float | None = None,
        allow_decimer_ocsr_auto_accept: bool | None = None,
    ) -> MaterialAutoDecisionResult | None:
        self.init_runtime()
        bundle = self.material_resolution.get_material_structure_bundle(paper_id)
        if bundle is None:
            return None
        accept_threshold = (
            self.config.batch_worker.material_auto_accept_min_confidence
            if accept_min_confidence is None
            else accept_min_confidence
        )
        reject_threshold = (
            self.config.batch_worker.material_auto_reject_min_confidence
            if reject_min_confidence is None
            else reject_min_confidence
        )
        allow_ocsr_accept = (
            self.config.batch_worker.material_auto_accept_decimer_ocsr
            if allow_decimer_ocsr_auto_accept is None
            else allow_decimer_ocsr_auto_accept
        )
        result = MaterialAutoDecisionResult(
            paper_id=bundle.paper_id,
            candidate_run_id=bundle.candidate_run_id,
            dry_run=dry_run,
            actor=actor,
        )
        accepted_material_ids = {
            candidate.paper_material_id
            for candidate in bundle.structure_candidates
            if candidate.status == "accepted"
        }
        linked_material_ids = {
            link.paper_material_id
            for link in bundle.links
            if link.match_status in {"matched_candidate", "matched_local"}
        }
        paper_materials = {material.paper_material_id: material for material in bundle.materials}
        for candidate in sorted(
            bundle.structure_candidates,
            key=lambda item: (
                item.paper_material_id,
                -(item.confidence or 0),
                item.provider,
                item.structure_candidate_id,
            ),
        ):
            plausible_candidate = is_plausible_public_structure_candidate(candidate)
            if plausible_candidate and (
                candidate.paper_material_id in accepted_material_ids
                or (
                    candidate.paper_material_id in linked_material_ids
                    and candidate.status != "accepted"
                )
            ):
                decision = self._skip_decision(
                    candidate,
                    "material_already_linked_or_accepted",
                    self.identity_judgments.latest_by_candidate(candidate.structure_candidate_id),
                )
                result.decisions.append(decision)
                result.skipped_count += 1
                continue
            judgment = self.identity_judgments.latest_by_candidate(candidate.structure_candidate_id)
            decision = self._evaluate_candidate(
                candidate,
                judgment,
                accept_min_confidence=accept_threshold,
                reject_min_confidence=reject_threshold,
                allow_decimer_ocsr_auto_accept=allow_ocsr_accept,
            )
            if decision.action == "auto_accept":
                if dry_run:
                    result.decisions.append(decision)
                    continue
                applied = self._apply_accept(
                    candidate,
                    decision,
                    actor,
                    paper_material=paper_materials.get(candidate.paper_material_id),
                )
                result.decisions.append(applied)
                if applied.applied:
                    result.accepted_count += 1
                    accepted_material_ids.add(candidate.paper_material_id)
                    linked_material_ids.add(candidate.paper_material_id)
                else:
                    result.skipped_count += 1
            elif decision.action == "auto_reject":
                if dry_run:
                    result.decisions.append(decision)
                    continue
                applied = self._apply_reject(candidate, decision, actor)
                result.decisions.append(applied)
                if applied.applied:
                    result.rejected_count += 1
                else:
                    result.skipped_count += 1
            else:
                result.decisions.append(decision)
                result.skipped_count += 1
        if not dry_run and result.accepted_count:
            self.review_service.material_completion.confirm_paper_if_materials_complete(
                bundle.paper_id
            )
        return result

    def _evaluate_candidate(
        self,
        candidate: MaterialStructureCandidate,
        judgment: MaterialIdentityJudgment | None,
        *,
        accept_min_confidence: float,
        reject_min_confidence: float,
        allow_decimer_ocsr_auto_accept: bool,
    ) -> MaterialAutoDecision:
        if candidate.status in {"accepted", "rejected"}:
            return self._skip_decision(candidate, f"candidate_already_{candidate.status}", judgment)
        if not is_plausible_public_structure_candidate(candidate):
            return self._decision(
                candidate,
                judgment,
                action="auto_reject",
                reason="direct_pubchem_query_mismatch",
            )
        if judgment is None:
            return self._skip_decision(candidate, "missing_identity_judgment", judgment)
        if judgment.status != "completed":
            return self._skip_decision(candidate, "identity_judgment_not_completed", judgment)
        confidence = judgment.confidence or 0.0
        if (
            judgment.verdict in AUTO_REJECT_VERDICTS
            and judgment.recommended_action in AUTO_REJECT_RECOMMENDED_ACTIONS
        ):
            if self._has_deterministic_reject_signal(candidate, judgment):
                return self._decision(
                    candidate,
                    judgment,
                    action="auto_reject",
                    reason="deterministic_identity_conflict",
                )
            if confidence >= reject_min_confidence:
                return self._decision(
                    candidate,
                    judgment,
                    action="auto_reject",
                    reason="high_confidence_identity_reject",
                )
            return self._skip_decision(candidate, "reject_confidence_below_threshold", judgment)
        if (
            judgment.verdict in AUTO_ACCEPT_VERDICTS
            and judgment.recommended_action in AUTO_ACCEPT_RECOMMENDED_ACTIONS
        ):
            if confidence < accept_min_confidence:
                return self._skip_decision(candidate, "accept_confidence_below_threshold", judgment)
            if not candidate.canonical_smiles:
                return self._skip_decision(candidate, "candidate_missing_valid_smiles", judgment)
            if candidate.provider == "decimer_ocsr" and not allow_decimer_ocsr_auto_accept:
                return self._skip_decision(candidate, "decimer_ocsr_auto_accept_disabled", judgment)
            return self._decision(
                candidate,
                judgment,
                action="auto_accept",
                reason="high_confidence_identity_accept",
            )
        return self._skip_decision(candidate, "identity_judgment_requires_human_review", judgment)

    @staticmethod
    def _has_deterministic_reject_signal(
        candidate: MaterialStructureCandidate,
        judgment: MaterialIdentityJudgment,
    ) -> bool:
        checks = judgment.deterministic_checks
        if checks.get("confirmed_evidence_inchi_key_comparison") == "conflict":
            return True
        return (
            candidate.resolver_name == "anysearch_to_pubchem"
            and checks.get("identifier_source_title_matches_paper_alias") is False
        )

    def _apply_accept(
        self,
        candidate: MaterialStructureCandidate,
        decision: MaterialAutoDecision,
        actor: str,
        *,
        paper_material: PaperLocalMaterial | None = None,
    ) -> MaterialAutoDecision:
        try:
            self.review_service.accept_structure_candidate(
                candidate.structure_candidate_id,
                MaterialReviewAction(
                    actor=actor,
                    message=(
                        "Auto accepted by material automation policy: "
                        f"{decision.reason}; confidence={decision.confidence}"
                    ),
                ),
                paper_material=paper_material,
                defer_completion=True,
                return_bundle=False,
            )
            return decision.model_copy(update={"applied": True})
        except Exception as exc:  # keep batch jobs alive; surface the blocked reason.
            return decision.model_copy(update={"applied": False, "error_message": str(exc)})

    def _apply_reject(
        self,
        candidate: MaterialStructureCandidate,
        decision: MaterialAutoDecision,
        actor: str,
    ) -> MaterialAutoDecision:
        try:
            self.review_service.reject_structure_candidate(
                candidate.structure_candidate_id,
                MaterialReviewAction(
                    actor=actor,
                    message=(
                        "Auto rejected by material automation policy: "
                        f"{decision.reason}; confidence={decision.confidence}"
                    ),
                ),
                return_bundle=False,
            )
            return decision.model_copy(update={"applied": True})
        except Exception as exc:
            return decision.model_copy(update={"applied": False, "error_message": str(exc)})

    def _skip_decision(
        self,
        candidate: MaterialStructureCandidate,
        reason: str,
        judgment: MaterialIdentityJudgment | None,
    ) -> MaterialAutoDecision:
        return self._decision(candidate, judgment, action="skip", reason=reason)

    @staticmethod
    def _decision(
        candidate: MaterialStructureCandidate,
        judgment: MaterialIdentityJudgment | None,
        *,
        action: str,
        reason: str,
    ) -> MaterialAutoDecision:
        return MaterialAutoDecision(
            paper_material_id=candidate.paper_material_id,
            structure_candidate_id=candidate.structure_candidate_id,
            action=action,
            reason=reason,
            verdict=judgment.verdict if judgment else None,
            recommended_action=judgment.recommended_action if judgment else None,
            confidence=judgment.confidence if judgment else candidate.confidence,
            applied=False,
        )
