from __future__ import annotations

from evolab_local.mining_platform.extractors.oled_rules import OledRuleBasedExtractor
from evolab_local.mining_platform.schemas.document import DocumentBlock


def test_oled_rule_extractor_builds_candidate_with_field_evidence() -> None:
    blocks = [
        DocumentBlock(
            paper_id="10.1000%2Fexample",
            block_id="p1_b0",
            page_id=1,
            block_index=0,
            text=(
                "Device D1 used ITO / HATCN / NPB / EML / TPBi / LiF / Al. "
                "The maximum EQE of device D1 was 18.2%, current efficiency was "
                "45 cd A-1, and turn-on voltage was 3.1 V."
            ),
            bbox=[0, 0, 100, 100],
        )
    ]

    candidates = OledRuleBasedExtractor().extract("10.1000%2Fexample", blocks)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.device_label == "D1"
    assert candidate.eqe_max == "18.2%"
    assert candidate.ce_max == "45 cd A-1"
    assert candidate.turn_on_voltage == "3.1 V"
    assert candidate.architecture == "ITO / HATCN / NPB / EML / TPBi / LiF / Al"
    assert candidate.field_evidence["eqe_max"].block_ids == ["p1_b0"]
    assert candidate.confidence["components"]["matched_field_count"] >= 4
