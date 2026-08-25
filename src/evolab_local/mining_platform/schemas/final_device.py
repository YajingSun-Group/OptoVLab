from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from evolab_local.mining_platform.schemas.device_record import DeviceRecordFields


class OledDeviceFinal(DeviceRecordFields):
    model_config = ConfigDict(from_attributes=True)

    final_device_id: str
    paper_id: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    confirmed_by: str
    confirmed_at: str
    created_at: str | None = None
    updated_at: str | None = None


class ConfirmPaperResult(BaseModel):
    paper_id: str
    final_devices: list[OledDeviceFinal] = Field(default_factory=list)
    final_count: int = 0
