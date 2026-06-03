"""Pydantic response models shared by the FastAPI routers."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class SceneSummary(BaseModel):
    id: str
    collection: str
    acquired_at: str
    cloud_cover: float | None = None


class ChangeSummary(BaseModel):
    """Compact change record used by the sidebar and API list endpoint."""

    id: str
    change_type: str
    confidence: float = Field(ge=0, le=1)
    area_m2: float = Field(ge=0)
    monitored_land_cover: str | None = None
    transition: str | None = None
    before_date: str | None = None
    after_date: str | None = None
    reliability: str | None = None
