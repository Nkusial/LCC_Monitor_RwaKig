"""Publish scored change polygons into the local PostGIS schema."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import psycopg

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def to_psycopg_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


def database_url(config: dict[str, Any] | None = None) -> str:
    """Prefer DATABASE_URL but keep the local Docker default for reproducibility."""
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return "postgresql+psycopg://geospatial:geospatial@localhost:5432/landcover"


def row_properties(row: Any) -> dict[str, Any]:
    """Move analysis attributes into JSONB while keeping core columns separate."""
    excluded = {"geometry", "change_type", "confidence", "final_confidence", "area_m2"}
    props = {}
    for key, value in row.items():
        if key in excluded:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            value = value.date().isoformat()
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        props[key] = value
    return props


def publish_changes(config: dict[str, Any], replace: bool = True) -> tuple[str, int]:
    """Insert publish-ready polygons and register a pipeline run."""
    ensure_output_dirs(config)
    outputs_dir = Path(config["paths"]["outputs"])
    input_path = outputs_dir / f"{config['change_detection']['output_prefix']}_scored.geojson"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Scored change output not found: {input_path}. Run pipelines/05_score_confidence.py first."
        )

    gdf = gpd.read_file(input_path)
    if "publish_ready" in gdf.columns:
        # Only records that passed confidence scoring become API-visible records.
        gdf = gdf[gdf["publish_ready"] == True].copy()  # noqa: E712
    gdf = gdf.to_crs("EPSG:4326")

    run_id = str(uuid4())
    conninfo = to_psycopg_url(database_url(config))
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            if replace:
                cur.execute("DELETE FROM change_polygons;")
            cur.execute(
                """
                INSERT INTO pipeline_runs (id, started_at, finished_at, status, parameters)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    run_id,
                    datetime.now(UTC),
                    datetime.now(UTC),
                    "published",
                    json.dumps({"stage": "publish_to_postgis", "replace": replace}),
                ),
            )
            for _, row in gdf.iterrows():
                props = row_properties(row)
                pre_date = row.get("before_date")
                post_date = row.get("after_date")
                confidence = float(row.get("final_confidence", row.get("confidence")))
                cur.execute(
                    """
                    INSERT INTO change_polygons (
                        id, run_id, change_type, confidence, area_m2,
                        pre_date, post_date, geom, properties
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326),
                        %s::jsonb
                    )
                    """,
                    (
                        str(uuid4()),
                        run_id,
                        row["change_type"],
                        confidence,
                        float(row["area_m2"]),
                        pre_date,
                        post_date,
                        json.dumps(row.geometry.__geo_interface__),
                        json.dumps(props),
                    ),
                )
        conn.commit()
    return run_id, int(len(gdf))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish scored change polygons to PostGIS.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing change_polygons instead of replacing generated records.",
    )
    args = parser.parse_args()

    run_id, count = publish_changes(load_config(), replace=not args.append)
    print(f"Published {count} change polygons to PostGIS. Run ID: {run_id}")


if __name__ == "__main__":
    main()
