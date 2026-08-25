from __future__ import annotations

import re
from collections.abc import Iterable

from evolab_local.mining_platform.schemas.document import DocumentBlock
from evolab_local.mining_platform.schemas.extraction import FieldEvidence, RawDeviceCandidate


class OledRuleBasedExtractor:
    name = "rule_based_oled"
    version = "v1"

    KEYWORDS = (
        "oled",
        "device",
        "eqe",
        "external quantum efficiency",
        "current efficiency",
        "power efficiency",
        "luminance",
        "turn-on",
        "turn on",
        "cie",
        "el peak",
        "emission peak",
        "fwhm",
        "lifetime",
        "ito",
        "eml",
        "htl",
        "etl",
    )
    FIELD_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
        "eqe_max": (
            re.compile(
                r"(?:maximum|max\.?|peak)?\s*(?:EQE|external quantum efficiency)"
                r"[^.;\n]{0,120}?(\d+(?:\.\d+)?)\s*%",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\d+(?:\.\d+)?)\s*%\s*(?:maximum|max\.?|peak)?\s*"
                r"(?:EQE|external quantum efficiency)",
                re.IGNORECASE,
            ),
        ),
        "ce_max": (
            re.compile(
                r"(?:current efficiency|CE)[^.;\n]{0,120}?(\d+(?:\.\d+)?)\s*"
                r"(?:cd\s*A(?:-1|\^-1|−1|⁻¹)?|cd/A)",
                re.IGNORECASE,
            ),
        ),
        "pe_max": (
            re.compile(
                r"(?:power efficiency|PE)[^.;\n]{0,120}?(\d+(?:\.\d+)?)\s*"
                r"(?:lm\s*W(?:-1|\^-1|−1|⁻¹)?|lm/W)",
                re.IGNORECASE,
            ),
        ),
        "luminance_max": (
            re.compile(
                r"(?:luminance|brightness)[^.;\n]{0,120}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
                r"(?:cd\s*m(?:-2|\^-2|−2|⁻²)?|cd/m2|cd/m²)",
                re.IGNORECASE,
            ),
        ),
        "turn_on_voltage": (
            re.compile(
                r"(?:turn-on|turn on|Von|V_on)[^.;\n]{0,120}?(\d+(?:\.\d+)?)\s*V",
                re.IGNORECASE,
            ),
        ),
        "el_peak": (
            re.compile(
                r"(?:EL peak|emission peak|peak)[^.;\n]{0,120}?(\d+(?:\.\d+)?)\s*nm",
                re.IGNORECASE,
            ),
        ),
        "fwhm": (re.compile(r"(?:FWHM)[^.;\n]{0,120}?(\d+(?:\.\d+)?)\s*nm", re.IGNORECASE),),
        "lifetime": (
            re.compile(
                r"(?:lifetime|LT50|T50)[^.;\n]{0,120}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*h",
                re.IGNORECASE,
            ),
        ),
    }
    CIE_PATTERN = re.compile(
        r"CIE[^.;\n]{0,80}?\(?\s*(0?\.\d+)\s*,\s*(0?\.\d+)\s*\)?",
        re.IGNORECASE,
    )
    DEVICE_LABEL_PATTERNS = (
        re.compile(r"\b(?:device|OLED)\s*[-:]?\s*([A-Z]?\d+[A-Z]?)\b", re.IGNORECASE),
        re.compile(r"\b(D\d+[A-Z]?)\b", re.IGNORECASE),
    )
    STACK_PATTERN = re.compile(
        r"((?:ITO|glass|HATCN|NPB|TAPC|TCTA|EML|mCP|TPBi|BPhen|LiF|Al|Ag|Mg)"
        r"(?:\s*/\s*[A-Za-z0-9().:+_-]+){2,})",
        re.IGNORECASE,
    )

    def extract(self, paper_id: str, blocks: Iterable[DocumentBlock]) -> list[RawDeviceCandidate]:
        relevant_blocks = [block for block in blocks if self._is_relevant(block.text)]
        if not relevant_blocks:
            return []

        field_values: dict[str, str] = {}
        field_evidence: dict[str, FieldEvidence] = {}
        evidence_block_ids: list[str] = []
        evidence_texts: list[str] = []

        for block in relevant_blocks:
            matched_field = False
            for field, patterns in self.FIELD_PATTERNS.items():
                if field in field_values:
                    continue
                value = self._match_first_value(block.text, patterns)
                if not value:
                    continue
                field_values[field] = self._format_value(field, value)
                field_evidence[field] = self._field_evidence(field_values[field], block)
                matched_field = True
            if "cie_x" not in field_values or "cie_y" not in field_values:
                cie_match = self.CIE_PATTERN.search(block.text)
                if cie_match:
                    field_values["cie_x"] = cie_match.group(1)
                    field_values["cie_y"] = cie_match.group(2)
                    field_evidence["cie_x"] = self._field_evidence(field_values["cie_x"], block)
                    field_evidence["cie_y"] = self._field_evidence(field_values["cie_y"], block)
                    matched_field = True
            if "architecture" not in field_values:
                stack_match = self.STACK_PATTERN.search(block.text)
                if stack_match:
                    field_values["architecture"] = stack_match.group(1).strip(" .;,")
                    field_evidence["architecture"] = self._field_evidence(
                        field_values["architecture"], block
                    )
                    matched_field = True
            if matched_field:
                evidence_block_ids.append(block.block_id)
                evidence_texts.append(block.text)

        if not field_values:
            evidence_block_ids = [block.block_id for block in relevant_blocks[:3]]
            evidence_texts = [block.text for block in relevant_blocks[:3]]

        device_label = self._device_label_from_text(
            " ".join(evidence_texts or [relevant_blocks[0].text])
        )
        first_block = relevant_blocks[0]
        evidence_page = (
            field_evidence[next(iter(field_evidence))].page_id
            if field_evidence
            else first_block.page_id
        )
        evidence_text = "\n\n".join(dict.fromkeys(evidence_texts or [first_block.text]))[:4000]

        candidate = RawDeviceCandidate(
            device_label=device_label,
            evidence_text=evidence_text,
            evidence_page=evidence_page,
            evidence_block_ids=list(dict.fromkeys(evidence_block_ids)),
            field_evidence=field_evidence,
            confidence=self._confidence(field_values, field_evidence),
            raw_payload={
                "extractor": self.name,
                "version": self.version,
                "paper_id": paper_id,
                "matched_fields": sorted(field_values),
                "relevant_block_count": len(relevant_blocks),
            },
            **field_values,
        )
        return [candidate]

    def _is_relevant(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in self.KEYWORDS)

    def _match_first_value(self, text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _format_value(self, field: str, value: str) -> str:
        cleaned = value.replace(",", "")
        if field == "eqe_max":
            return f"{cleaned}%"
        if field == "ce_max":
            return f"{cleaned} cd A-1"
        if field == "pe_max":
            return f"{cleaned} lm W-1"
        if field == "luminance_max":
            return f"{cleaned} cd m-2"
        if field == "turn_on_voltage":
            return f"{cleaned} V"
        if field in {"el_peak", "fwhm"}:
            return f"{cleaned} nm"
        if field == "lifetime":
            return f"{cleaned} h"
        return cleaned

    def _field_evidence(self, value: str, block: DocumentBlock) -> FieldEvidence:
        return FieldEvidence(
            value=value,
            block_ids=[block.block_id],
            page_id=block.page_id,
            source_text=block.text,
        )

    def _device_label_from_text(self, text: str) -> str | None:
        for pattern in self.DEVICE_LABEL_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()
        return None

    def _confidence(
        self,
        field_values: dict[str, str],
        field_evidence: dict[str, FieldEvidence],
    ) -> dict[str, object]:
        matched_fields = len(field_values)
        performance_fields = {
            "eqe_max",
            "ce_max",
            "pe_max",
            "luminance_max",
            "turn_on_voltage",
            "cie_x",
            "cie_y",
        }
        performance_count = len(performance_fields.intersection(field_values))
        if matched_fields >= 4 and performance_count >= 2:
            label = "medium"
            score = 0.68
        elif matched_fields >= 2:
            label = "low_medium"
            score = 0.5
        else:
            label = "low"
            score = 0.32
        return {
            "score": score,
            "label": label,
            "components": {
                "matched_field_count": matched_fields,
                "performance_field_count": performance_count,
                "evidence_field_count": len(field_evidence),
                "extractor_confidence": score,
                "evidence_quality": "block_level",
                "schema_validity": "partial",
            },
        }
