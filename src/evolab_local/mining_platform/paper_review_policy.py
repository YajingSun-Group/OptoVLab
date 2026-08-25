from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NO_DEVICE_REVIEW_REASONS = {
    "no_device_data",
    "no_extractable_device_stack",
    "device_data_validation_failed",
}


def no_device_review_reason(
    mining_result: Mapping[str, Any],
) -> str | None:
    """Return an auditable terminal reason for papers without usable device data."""
    devices = mining_result.get("devices")
    if not isinstance(devices, list):
        return None
    if not devices:
        return "no_device_data"
    device_objects = [item for item in devices if isinstance(item, Mapping)]
    if not device_objects or len(device_objects) != len(devices):
        return None
    if all(not _device_has_reported_stack(device) for device in device_objects):
        return "no_extractable_device_stack"
    return None


def _device_has_reported_stack(device: Mapping[str, Any]) -> bool:
    architecture = device.get("architecture_text")
    if isinstance(architecture, str) and architecture.strip():
        return True
    layers = device.get("layers")
    return isinstance(layers, list) and any(isinstance(layer, Mapping) for layer in layers)
