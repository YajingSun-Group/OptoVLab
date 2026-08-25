from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DeviceRecordFields(BaseModel):
    device_label: str | None = None
    architecture: str | None = None
    notes: str | None = None
    substrate: str | None = None
    anode: str | None = None
    hil: str | None = None
    htl: str | None = None
    ebl: str | None = None
    eml_host: str | None = None
    eml_dopant: str | None = None
    eml_emitter: str | None = None
    hbl: str | None = None
    etl: str | None = None
    eil: str | None = None
    cathode: str | None = None
    layer_thicknesses: str | None = None
    eqe_max: str | None = None
    ce_max: str | None = None
    pe_max: str | None = None
    luminance_max: str | None = None
    turn_on_voltage: str | None = None
    cie_x: str | None = None
    cie_y: str | None = None
    el_peak: str | None = None
    fwhm: str | None = None
    lifetime: str | None = None
    evidence_text: str | None = None
    evidence_page: int | None = None


class DeviceRecordCreate(DeviceRecordFields):
    actor: str = "local_user"
    message: str | None = None


class DeviceRecordUpdate(DeviceRecordFields):
    actor: str = "local_user"
    message: str | None = None


class DeviceRecordReviewed(DeviceRecordFields):
    model_config = ConfigDict(from_attributes=True)

    record_id: str
    paper_id: str
    review_status: str = "in_progress"
    created_at: str | None = None
    updated_at: str | None = None
    confirmed_at: str | None = None
