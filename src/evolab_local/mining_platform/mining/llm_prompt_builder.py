from __future__ import annotations

import json
from dataclasses import dataclass
from evolab_local.mining_platform.schemas.document import DocumentBlock
from evolab_local.mining_platform.schemas.domain_template import DomainTemplate

KEYWORDS = (
    "oled",
    "device",
    "structure",
    "ito",
    "hil",
    "htl",
    "eml",
    "etl",
    "eil",
    "cathode",
    "anode",
    "eqe",
    "external quantum",
    "current efficiency",
    "power efficiency",
    "luminance",
    "cie",
    "fwhm",
    "lifetime",
    "lt50",
    "lt80",
    "lt95",
    "roll-off",
    "rolloff",
    "table",
    "scheme",
    "figure",
    "fabrication",
    "evaporat",
    "spin",
)


@dataclass(frozen=True)
class PromptSource:
    block_id: str
    page_id: int | None
    source_type: str
    text: str
    bbox: list[float]


def select_prompt_sources(
    sources: list[PromptSource],
    *,
    max_chars: int,
) -> list[PromptSource]:
    if not sources:
        return []
    scored = sorted(
        enumerate(sources),
        key=lambda item: (-_score_source(item[1]), item[0]),
    )
    selected: list[PromptSource] = []
    used_chars = 0
    for _, source in scored:
        source_chars = len(source.text)
        if selected and used_chars + source_chars > max_chars:
            continue
        selected.append(source)
        used_chars += source_chars
        if used_chars >= max_chars:
            break
    return sorted(selected, key=lambda source: (source.page_id or 0, source.block_id))


def sources_from_document_blocks(blocks: list[DocumentBlock]) -> list[PromptSource]:
    return [
        PromptSource(
            block_id=block.block_id,
            page_id=block.page_id,
            source_type=block.block_type,
            text=block.text,
            bbox=block.bbox,
        )
        for block in blocks
        if block.text.strip()
    ]


def build_oled_mining_messages(
    *,
    template: DomainTemplate,
    sources: list[PromptSource],
) -> list[dict[str, str]]:
    field_summary = [
        {
            "field_path": field.field_path,
            "label": field.label,
            "data_type": field.data_type,
            "required": field.required,
            "enum_ref": field.enum_ref,
            "hint": field.extraction_hint,
        }
        for field in template.fields
    ]
    template_payload = {
        "template_id": template.template_id,
        "version": template.version,
        "required_output_keys": template.required_output_keys,
        "vocabularies": template.vocabularies,
        "json_shape": template.llm_output_schema.get("json_shape"),
        "fields": field_summary,
        "example_output": template.example_output,
    }
    source_payload = [
        {
            "block_id": source.block_id,
            "page_id": source.page_id,
            "source_type": source.source_type,
            "bbox": source.bbox,
            "text": source.text,
        }
        for source in sources
    ]
    system = (
        "You are a precise OLED literature mining engine. Output only one JSON object that "
        "matches the provided domain template. Do not output Markdown or explanations. "
        "Prioritize traceability: every extracted value should be auditable against a fine-grained evidence item."
    )
    user = (
        "Extract OLED device data from the provided paper sources.\n\n"
        "Work in this order:\n"
        "A. First create fine-grained evidence[] items.\n"
        "B. Then create paper-local materials[] using those evidence IDs.\n"
        "C. Then create devices[] with layers[] and performance[] using those evidence IDs.\n\n"
        "Evidence rules:\n"
        "1. evidence[] must be fine-grained. Prefer one evidence item for one table row, one device stack statement, one performance statement, one material definition statement, one figure caption, or one relevant paragraph fragment.\n"
        "2. Do not use one broad evidence item to support many unrelated values. Avoid evidence.source_text that combines an entire abstract, entire page, or entire table unless the source block only contains that one row/statement.\n"
        "3. evidence[].block_id must be copied exactly from a provided SOURCE_BLOCKS_JSON block_id.\n"
        "4. evidence[].source_text should be the shortest verbatim source fragment that still supports the value.\n"
        "5. evidence[].source_type must use the template vocabulary. Map source blocks as follows when needed: table -> table, text -> text, image/chart captions -> figure or caption, scheme captions -> scheme, otherwise unknown.\n"
        "6. Create enough evidence items for auditing. As a rule of thumb, each reported device should have at least one stack evidence item and each performance[] item should have its own evidence item unless several metrics are in the same table row.\n\n"
        "7. Create evidence only when at least one extracted material, device, layer, fabrication, or performance field cites it. Do not copy every OLED-related sentence into evidence.\n"
        "8. Reuse one evidence item when the same exact sentence or table row supports several fields or metrics; never duplicate evidence merely because multiple fields cite it.\n"
        "9. Return at most 120 evidence items. If the paper contains more possible evidence, prioritize full device stacks, device-used material definitions, directly measured performance rows, operating conditions, and lifetime data from this paper.\n\n"
        "Extraction rules:\n"
        "1. Extract evidence, materials, and devices only. Do not extract paper metadata.\n"
        "2. Every non-trivial material/device/layer/performance value must cite evidence_refs. Use the most specific evidence IDs available.\n"
        "3. Each performance[] item must cite evidence that contains the metric, value, unit, and condition. If the metric is from a table, evidence.source_text should contain the relevant row or row fragment.\n"
        "4. Each device architecture_text must cite the exact source that reports the full stack. Each layer can reuse that stack evidence unless layer-specific details come from another source.\n"
        "5. Create one material entity for each distinct material mention used in devices. Do not merge host, dopant, sensitizer, transport material, electrode material, or capping material unless the paper explicitly states they are the same material.\n"
        "6. For materials[].full_name_in_paper, use only a full expanded name explicitly written in SOURCE_BLOCKS_JSON. Do not infer, complete, translate, or use external chemical knowledge to expand abbreviations. Use null if the paper only provides an abbreviation, acronym, or paper-specific label.\n"
        "7. Do not invent missing values. Use null, [] or unknown when the source is unclear.\n"
        "8. Do not extract SMILES, InChI, or InChIKey in this phase. Set them to null.\n"
        "9. Keep raw units and raw text when possible, and normalize only obvious numeric values.\n"
        "10. Enum fields must use the exact vocabulary values in DOMAIN_TEMPLATE_JSON. For unsupported lifetime labels, use statistic=measured and preserve the original label in metric_name/raw_text.\n"
        "11. For OLED stacks, preserve architecture_text and also split ordered layers.\n"
        "12. Device-level fabrication.method describes the overall device preparation route. Layer-level fabrication_method should be filled only when the source explicitly reports a layer-specific method or when layers use different methods. Do not repeat the same device-level method for every layer by default.\n"
        "13. Scalar object fields such as layer.thickness must be either null or one object {value, unit}; never output a list. If a composite layer has multiple sublayer thicknesses, keep them in architecture_text/layer_name and set thickness to null.\n"
        "14. For composite electrodes or unresolved composite functional layers, preserve the full original string in architecture_text and layer_name; do not force sublayer information into fields that cannot represent it.\n\n"
        "No-device paper rules:\n"
        "1. Extract only OLED devices fabricated, measured, or explicitly reported as experimental results of the target paper. Do not extract devices mentioned only in the introduction, related work, references, or comparison with prior literature.\n"
        "2. If the target paper contains no extractable experimental OLED device data or no reported device stack, return exactly empty evidence, materials, and devices arrays. Do not invent a partial device from isolated material or performance mentions.\n"
        "3. An empty devices array is a deliberate no-device result. Use it only after checking all provided source blocks for an experimental device architecture or fabrication statement.\n\n"
        "Required shape rules:\n"
        "1. If you create a device, devices[].layers must not be empty when architecture_text or stack evidence exists. Split the reported OLED stack into ordered layers.\n"
        "2. Every layer object must include layer_index as a 1-based integer and layer_role as one exact layer_role vocabulary value. Use unknown only when the source truly cannot support a more specific role.\n"
        "3. Every layer object must include at least one component when layer_name or a material/electrode name is known. Every component must include material_mention copied from the paper text or from the linked materials[].mention_list.\n"
        "4. evidence_refs must contain evidence[].evidence_id values only, such as E001 or E1. Do not put SOURCE_BLOCKS_JSON block_id values such as mineru_p2_b4 into evidence_refs. If an evidence item uses block_id=mineru_p2_b4, cite that evidence item's evidence_id instead.\n\n"
        "DOMAIN_TEMPLATE_JSON:\n"
        f"{json.dumps(template_payload, ensure_ascii=False)}\n\n"
        "SOURCE_BLOCKS_JSON:\n"
        f"{json.dumps(source_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _score_source(source: PromptSource) -> int:
    text = source.text.lower()
    score = sum(5 for keyword in KEYWORDS if keyword in text)
    if source.source_type in {"table", "caption"}:
        score += 20
    if source.source_type in {"image", "chart"} and "chemical" in text:
        score += 8
    return score
