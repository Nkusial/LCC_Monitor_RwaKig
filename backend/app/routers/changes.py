"""Change endpoints backed by PostGIS with static-safe empty fallbacks.

The routes return useful responses even when PostGIS is not running, which
keeps local development and hosted portfolio demos from failing noisily.
"""

from typing import Any

from fastapi import APIRouter, Query
import psycopg
from psycopg.rows import dict_row

from backend.app.database import get_settings
from backend.app.models import ChangeSummary

router = APIRouter(prefix="/changes", tags=["changes"])

AOI_AREA_M2 = 400_000_000

CHANGE_SUMMARY_QUERY = """
    SELECT
        id::text,
        change_type,
        confidence,
        area_m2,
        pre_date::text AS before_date,
        post_date::text AS after_date,
        properties->>'monitored_land_cover' AS monitored_land_cover,
        properties->>'transition' AS transition,
        properties->>'reliability' AS reliability
    FROM change_polygons
    ORDER BY detected_at DESC
    LIMIT %s
"""


def connection_url() -> str:
    """Convert the SQLAlchemy-style DSN into the native psycopg DSN."""
    return get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")


@router.get("", response_model=list[ChangeSummary])
def list_changes(limit: int = Query(default=500, ge=1, le=5000)) -> list[ChangeSummary]:
    """Return recent published changes for the WebGIS sidebar."""
    try:
        with psycopg.connect(connection_url(), connect_timeout=2) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(CHANGE_SUMMARY_QUERY, (limit,))
                return [ChangeSummary(**row) for row in cur.fetchall()]
    except Exception:
        return []


@router.get("/summary")
def change_summary() -> dict[str, Any]:
    """Aggregate changed/no-change quantities for dashboard metrics."""
    query = """
        SELECT
            COUNT(*)::int AS total,
            AVG(confidence)::float AS mean_confidence,
            COALESCE(SUM(area_m2), 0)::float AS changed_area_m2
        FROM change_polygons
    """
    group_queries = {
        "by_monitored_land_cover": """
            SELECT properties->>'monitored_land_cover' AS key, COUNT(*)::int AS count
            FROM change_polygons
            GROUP BY properties->>'monitored_land_cover'
            ORDER BY count DESC
        """,
        "by_reliability": """
            SELECT properties->>'reliability' AS key, COUNT(*)::int AS count
            FROM change_polygons
            GROUP BY properties->>'reliability'
            ORDER BY count DESC
        """,
        "by_monitored_land_cover_area_m2": """
            SELECT
                properties->>'monitored_land_cover' AS key,
                COALESCE(SUM(area_m2), 0)::float AS count
            FROM change_polygons
            GROUP BY properties->>'monitored_land_cover'
            ORDER BY count DESC
        """,
        "by_reliability_area_m2": """
            SELECT
                properties->>'reliability' AS key,
                COALESCE(SUM(area_m2), 0)::float AS count
            FROM change_polygons
            GROUP BY properties->>'reliability'
            ORDER BY count DESC
        """,
    }
    try:
        with psycopg.connect(connection_url(), connect_timeout=2) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query)
                summary = dict(cur.fetchone() or {})
                changed_area_m2 = float(summary.get("changed_area_m2") or 0)
                # The project scope is fixed to a local 20 km x 20 km AOI, so
                # no-change area can be reported even though only changed
                # polygons are stored in PostGIS.
                no_change_area_m2 = max(AOI_AREA_M2 - changed_area_m2, 0)
                summary["aoi_area_m2"] = AOI_AREA_M2
                summary["total_area_m2"] = changed_area_m2
                summary["changed_area_m2"] = changed_area_m2
                summary["no_change_area_m2"] = no_change_area_m2
                summary["changed_percent"] = changed_area_m2 / AOI_AREA_M2 * 100
                summary["no_change_percent"] = no_change_area_m2 / AOI_AREA_M2 * 100
                for output_key, group_query in group_queries.items():
                    cur.execute(group_query)
                    summary[output_key] = {
                        row["key"] or "unknown": row["count"] for row in cur.fetchall()
                    }
                return summary
    except Exception:
        return {
            "total": 0,
            "mean_confidence": None,
            "aoi_area_m2": AOI_AREA_M2,
            "total_area_m2": 0,
            "changed_area_m2": 0,
            "no_change_area_m2": AOI_AREA_M2,
            "changed_percent": 0,
            "no_change_percent": 100,
            "by_monitored_land_cover": {},
            "by_monitored_land_cover_area_m2": {},
            "by_reliability": {},
            "by_reliability_area_m2": {},
        }


@router.get("/geojson")
def changes_geojson(
    limit: int = Query(default=1000, ge=1, le=5000),
    min_confidence: float = Query(default=0.75, ge=0, le=1),
    monitored_land_cover: str = Query(default="all"),
) -> dict[str, Any]:
    """Return map-ready change polygons filtered by confidence and group."""
    filters = ["confidence >= %s"]
    params: list[Any] = [min_confidence]
    if monitored_land_cover != "all":
        filters.append("properties->>'monitored_land_cover' = %s")
        params.append(monitored_land_cover)
    params.append(limit)
    query = f"""
        SELECT
            id::text,
            change_type,
            confidence,
            area_m2,
            pre_date::text AS before_date,
            post_date::text AS after_date,
            properties,
            ST_AsGeoJSON(geom)::json AS geometry
        FROM change_polygons
        WHERE {' AND '.join(filters)}
        ORDER BY confidence DESC, area_m2 DESC
        LIMIT %s
    """
    try:
        with psycopg.connect(connection_url(), connect_timeout=2) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                features = []
                for row in cur.fetchall():
                    props = dict(row["properties"] or {})
                    # Keep canonical database fields duplicated into
                    # properties so MapLibre styling and popups can read them
                    # directly from each GeoJSON feature.
                    props.update(
                        {
                            "id": row["id"],
                            "change_type": row["change_type"],
                            "confidence": row["confidence"],
                            "area_m2": row["area_m2"],
                            "before_date": row["before_date"],
                            "after_date": row["after_date"],
                        }
                    )
                    features.append(
                        {
                            "type": "Feature",
                            "id": row["id"],
                            "properties": props,
                            "geometry": row["geometry"],
                        }
                    )
                return {"type": "FeatureCollection", "features": features}
    except Exception:
        return {"type": "FeatureCollection", "features": []}
