"""Search Sentinel-2 scenes for the configured AOI and register metadata."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import psycopg
from pystac_client import Client
from shapely.geometry import mapping, shape

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def load_aoi_geometry(aoi_path: str) -> dict[str, Any]:
    """Read the AOI as WGS84 GeoJSON geometry for STAC spatial filtering."""
    aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    if aoi.empty:
        raise ValueError(f"AOI file has no features: {aoi_path}")

    return mapping(aoi.geometry.iloc[0])


def item_to_feature(item: Any) -> dict[str, Any]:
    """Convert a STAC item into the lightweight GeoJSON catalog shape."""
    properties = dict(item.properties)
    return {
        "type": "Feature",
        "id": item.id,
        "geometry": item.geometry,
        "properties": {
            "id": item.id,
            "collection": item.collection_id,
            "datetime": properties.get("datetime"),
            "cloud_cover": properties.get("eo:cloud_cover"),
            "assets": {key: asset.href for key, asset in item.assets.items()},
            "stac_properties": properties,
        },
    }


def search_sentinel2_items(
    config: dict[str, Any],
    intersects: dict[str, Any],
    limit: int | None = None,
) -> list[Any]:
    """Find low-cloud Sentinel-2 items that intersect the AOI."""
    sentinel = config["sentinel"]
    date_range = sentinel["date_range"]
    collection = sentinel["collections"]["sentinel2"]
    datetime_range = f"{date_range['start']}/{date_range['end']}"

    client = Client.open(sentinel["stac_api_url"])
    bbox = list(shape(intersects).bounds)
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": sentinel["max_cloud_cover"]}},
        max_items=limit,
        method="POST",
    )
    return list(search.items())


def write_scene_catalog(features: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = {
        "type": "FeatureCollection",
        "features": features,
    }
    output_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def to_psycopg_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def register_scenes_in_postgis(features: list[dict[str, Any]], database_url: str) -> int:
    """Upsert scene metadata so the database can track source imagery."""
    if not features:
        return 0

    sql = """
        INSERT INTO sentinel_scenes (
            id, collection, acquired_at, cloud_cover, footprint, assets, properties
        )
        VALUES (
            %(id)s,
            %(collection)s,
            %(acquired_at)s,
            %(cloud_cover)s,
            ST_SetSRID(ST_GeomFromGeoJSON(%(geometry)s), 4326),
            %(assets)s::jsonb,
            %(properties)s::jsonb
        )
        ON CONFLICT (id) DO UPDATE SET
            cloud_cover = EXCLUDED.cloud_cover,
            assets = EXCLUDED.assets,
            properties = EXCLUDED.properties;
    """

    rows = [
        {
            "id": feature["properties"]["id"],
            "collection": feature["properties"]["collection"],
            "acquired_at": feature["properties"]["datetime"],
            "cloud_cover": feature["properties"]["cloud_cover"],
            "geometry": json.dumps(feature["geometry"]),
            "assets": json.dumps(feature["properties"]["assets"]),
            "properties": json.dumps(feature["properties"]["stac_properties"]),
        }
        for feature in features
    ]

    with psycopg.connect(to_psycopg_url(database_url)) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Sentinel-2 scenes for the configured AOI.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--no-db", action="store_true", help="Skip PostGIS scene registration.")
    return parser.parse_args()


def main() -> None:
    """Search, write the catalog file, and optionally register in PostGIS."""
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    intersects = load_aoi_geometry(config["project"]["aoi_path"])
    items = search_sentinel2_items(config, intersects, limit=args.limit)
    features = [item_to_feature(item) for item in items]

    output_path = Path(config["paths"]["catalog"]) / "scenes.geojson"
    write_scene_catalog(features, output_path)

    print(f"Wrote {len(features)} scenes to {output_path}")

    if args.no_db:
        print("Skipped PostGIS registration.")
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Skipped PostGIS registration because DATABASE_URL is not set.")
        return

    registered = register_scenes_in_postgis(features, database_url)
    print(f"Registered {registered} scenes in PostGIS.")


if __name__ == "__main__":
    main()
