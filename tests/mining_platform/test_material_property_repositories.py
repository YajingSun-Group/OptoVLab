from __future__ import annotations

from pathlib import Path

from evolab_local.mining_platform.schemas.material_structure import (
    MaterialPropertyCandidate,
    MaterialPropertyReview,
    MaterialPropertyReviewEvent,
)
from evolab_local.mining_platform.storage.database import Database, now_iso
from evolab_local.mining_platform.storage.repositories import (
    MaterialPropertyCandidateRepository,
    MaterialPropertyReviewEventRepository,
    MaterialPropertyReviewRepository,
)


def test_material_property_candidate_review_event_roundtrip(tmp_path: Path) -> None:
    database = Database(tmp_path / "platform.sqlite")
    database.init_db()
    candidates = MaterialPropertyCandidateRepository(database)
    reviews = MaterialPropertyReviewRepository(database)
    events = MaterialPropertyReviewEventRepository(database)

    timestamp = now_iso()
    candidate = candidates.upsert(
        MaterialPropertyCandidate(
            property_candidate_id="prop-1",
            paper_id="10.1000/example",
            candidate_run_id="run-1",
            paper_material_id="M001",
            global_material_id="GMAT-1",
            property_name="PLQY",
            property_category="photophysical",
            value_numeric=83.0,
            value_raw="83%",
            unit="%",
            normalized_value_numeric=0.83,
            normalized_unit="fraction",
            condition={
                "sample_form": "doped_film",
                "host": "mCBP",
                "dopant_concentration": "2 wt%",
            },
            method="integrating_sphere",
            source_type="table",
            evidence_text="The PLQY of the doped film was 83%.",
            llm_evidence_text="The PLQY of the doped film was 83%.",
            source_block_text="Table 2. The PLQY of the doped film was 83%.",
            evidence_anchor={"page": 5, "block_id": "table-2", "bbox": [0.1, 0.2, 0.3, 0.4]},
            provider="deepseek",
            model="deepseek-v4-flash",
            prompt_version="material_property_miner_v1",
            confidence=0.86,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )

    assert candidate.condition["host"] == "mCBP"
    assert candidate.llm_evidence_text == "The PLQY of the doped film was 83%."
    assert candidate.source_block_text == "Table 2. The PLQY of the doped film was 83%."
    assert candidate.evidence_anchor["block_id"] == "table-2"
    assert candidates.get("prop-1") == candidate
    assert [item.property_candidate_id for item in candidates.list_by_material("run-1", "M001")] == [
        "prop-1"
    ]

    accepted = candidates.update_status("prop-1", "accepted")
    assert accepted is not None
    assert accepted.status == "accepted"

    review = reviews.add(
        MaterialPropertyReview(
            review_id="review-1",
            property_candidate_id="prop-1",
            paper_id="10.1000/example",
            candidate_run_id="run-1",
            paper_material_id="M001",
            decision="accept",
            reviewed_property_name="PLQY",
            reviewed_value_numeric=83.0,
            reviewed_unit="%",
            reviewed_condition={"sample_form": "doped_film"},
            reviewed_evidence_anchor={"page": 5, "block_id": "table-2"},
            actor="tester",
            message="Looks correct.",
            created_at=now_iso(),
        )
    )
    assert reviews.list_by_candidate("prop-1")[0] == review

    event = events.add(
        MaterialPropertyReviewEvent(
            event_id="event-1",
            paper_id="10.1000/example",
            candidate_run_id="run-1",
            paper_material_id="M001",
            property_candidate_id="prop-1",
            event_type="accept",
            before={"status": "pending_review"},
            after={"status": "accepted"},
            actor="tester",
            message="Accepted PLQY.",
            created_at=now_iso(),
        )
    )
    assert events.get("event-1") == event
    assert events.list_by_run("run-1")[0].after == {"status": "accepted"}
