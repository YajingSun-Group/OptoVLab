from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReviewAction(BaseModel):
    actor: str = "local_user"
    message: str | None = None


class ReviewEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    paper_id: str
    record_id: str | None = None
    event_type: str
    actor: str = "local_user"
    message: str | None = None
    before_json: str | None = None
    after_json: str | None = None
    created_at: str | None = None
