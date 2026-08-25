from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from evolab_local.mining_platform.core.config import DecimerSegmentationConfig, DecimerSmilesConfig


@dataclass(frozen=True)
class DecimerSegment:
    index: int
    bbox: list[float]
    width: int | None = None
    height: int | None = None
    raw_segment: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecimerSegmentationResponse:
    file_name: str
    segment_count: int
    bbox_available: bool
    bbox_source_order: str | None = None
    segments: list[DecimerSegment] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecimerSmilesResponse:
    file_name: str
    smiles: str | None
    confidence_tokens: list[dict[str, Any]] = field(default_factory=list)
    hand_drawn: bool = False
    raw_response: dict[str, Any] = field(default_factory=dict)


class DecimerSegmentationClientProtocol(Protocol):
    def segment_image(
        self,
        image_path: Path,
        *,
        expand: bool,
        return_images: bool,
        max_segments: int,
    ) -> DecimerSegmentationResponse: ...


class DecimerSmilesClientProtocol(Protocol):
    def predict_smiles(self, image_path: Path) -> DecimerSmilesResponse: ...


class DecimerSegmentationClient:
    def __init__(self, config: DecimerSegmentationConfig) -> None:
        self.config = config

    def segment_image(
        self,
        image_path: Path,
        *,
        expand: bool,
        return_images: bool,
        max_segments: int,
    ) -> DecimerSegmentationResponse:
        base_url = self.config.base_url.rstrip("/")
        with image_path.open("rb") as handle:
            files = {"file": (image_path.name, handle, _content_type(image_path))}
            data = {
                "expand": str(expand).lower(),
                "return_images": str(return_images).lower(),
                "max_segments": str(max_segments),
            }
            with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
                response = client.post(f"{base_url}/segment", files=files, data=data)
                response.raise_for_status()
                payload = response.json()
        return parse_decimer_segmentation_response(payload)


class DecimerSmilesClient:
    def __init__(self, config: DecimerSmilesConfig) -> None:
        self.config = config

    def predict_smiles(self, image_path: Path) -> DecimerSmilesResponse:
        base_url = self.config.base_url.rstrip("/")
        with image_path.open("rb") as handle:
            files = {"file": (image_path.name, handle, _content_type(image_path))}
            data = {
                "confidence": str(self.config.confidence).lower(),
                "hand_drawn": str(self.config.hand_drawn).lower(),
            }
            with httpx.Client(timeout=self.config.timeout_seconds, trust_env=False) as client:
                response = client.post(f"{base_url}/predict", files=files, data=data)
                response.raise_for_status()
                payload = response.json()
        tokens = payload.get("confidence_tokens")
        return DecimerSmilesResponse(
            file_name=str(payload.get("file_name") or image_path.name),
            smiles=_str_or_none(payload.get("smiles")),
            confidence_tokens=tokens if isinstance(tokens, list) else [],
            hand_drawn=bool(payload.get("hand_drawn")),
            raw_response=payload,
        )


def parse_decimer_segmentation_response(payload: dict[str, Any]) -> DecimerSegmentationResponse:
    segments: list[DecimerSegment] = []
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list):
        for fallback_index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                continue
            bbox = _parse_bbox(item.get("bbox"))
            if not bbox:
                continue
            segments.append(
                DecimerSegment(
                    index=_int_or_default(item.get("index"), fallback_index),
                    bbox=bbox,
                    width=_int_or_none(item.get("width")),
                    height=_int_or_none(item.get("height")),
                    raw_segment=item,
                )
            )
    return DecimerSegmentationResponse(
        file_name=str(payload.get("file_name") or ""),
        segment_count=_int_or_default(payload.get("segment_count"), len(segments)),
        bbox_available=bool(payload.get("bbox_available")),
        bbox_source_order=_str_or_none(payload.get("bbox_source_order")),
        segments=segments,
        raw_response=payload,
    )


def _parse_bbox(value: object) -> list[float]:
    if isinstance(value, dict):
        coords = [value.get("x0"), value.get("y0"), value.get("x1"), value.get("y1")]
    elif isinstance(value, list):
        coords = value[:4]
    else:
        return []
    if len(coords) < 4:
        return []
    output: list[float] = []
    for coord in coords:
        if not isinstance(coord, int | float):
            return []
        output.append(float(coord))
    return output


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def _int_or_default(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _str_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
