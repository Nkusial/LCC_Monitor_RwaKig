"""FastAPI entrypoint for the local-first change monitoring backend.

The API is intentionally small: it exposes health, AOI, scene, and change
endpoints that can be backed by PostGIS locally while still allowing the
frontend to fall back to static demo files on GitHub Pages.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.aoi import router as aoi_router
from backend.app.routers.changes import router as changes_router
from backend.app.routers.health import router as health_router
from backend.app.routers.scenes import router as scenes_router

app = FastAPI(
    title="Confidence-Aware Land-Cover Change Monitoring API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # Local Vite origins used during development; production Pages uses static
    # demo assets and does not need browser calls to the local API.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    """Return a human-readable API index for portfolio reviewers."""
    return {
        "name": "Confidence-Aware Land-Cover Change Monitoring API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/aoi",
            "/changes",
            "/changes/summary",
            "/changes/geojson",
        ],
    }


app.include_router(health_router)
app.include_router(aoi_router)
app.include_router(scenes_router)
app.include_router(changes_router)
