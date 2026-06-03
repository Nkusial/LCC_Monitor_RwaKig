"""Scene endpoint placeholder for exposing searched Sentinel catalogs."""

from fastapi import APIRouter

from backend.app.models import SceneSummary

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.get("", response_model=list[SceneSummary])
def list_scenes() -> list[SceneSummary]:
    """Return scene summaries once catalog-to-API wiring is enabled."""
    return []
