"""Health endpoint used by smoke tests and local run checks."""

from fastapi import APIRouter

from backend.app.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a minimal status object without touching PostGIS."""
    return HealthResponse()
