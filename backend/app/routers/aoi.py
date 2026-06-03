"""AOI endpoint for the configured 20 km x 20 km study area."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.database import get_settings

router = APIRouter(prefix="/aoi", tags=["aoi"])


@router.get("")
def get_aoi() -> dict[str, Any]:
    """Return the AOI GeoJSON used by processing, API, and WebGIS."""
    aoi_path = Path(get_settings().aoi_path)
    if not aoi_path.exists():
        raise HTTPException(status_code=404, detail=f"AOI file not found: {aoi_path}")

    return json.loads(aoi_path.read_text(encoding="utf-8"))
