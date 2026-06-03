"""Repair derived products that should be on the project master grid.

This is not a classifier or model-tuning step. It only fixes georeferencing
lineage: direct Sentinel-2 SWIR1 bands are resampled to the same 10 m grid as
the other processed features, and AOI GeoJSON files are regenerated from the
same master-grid corner coordinates used by WebGIS image overlays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform


ROOT = Path(__file__).resolve().parents[1]
MASTER_GRID_PATH = ROOT / "configs" / "master_grid.yaml"
PROCESSED = ROOT / "data" / "processed"
CONFIG_AOI = ROOT / "configs" / "aoi.geojson"
WEB_AOI = ROOT / "web" / "public" / "demo" / "aoi.geojson"


def load_master_grid() -> dict[str, Any]:
    """Load raster dimensions, transform, CRS, and bounds from the grid contract."""
    grid = yaml.safe_load(MASTER_GRID_PATH.read_text(encoding="utf-8"))
    transform_cfg = grid["transform"]
    bounds = grid["bounds"]
    return {
        "crs": grid["crs"],
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "transform": rasterio.Affine(
            float(transform_cfg["a"]),
            float(transform_cfg["b"]),
            float(transform_cfg["c"]),
            float(transform_cfg["d"]),
            float(transform_cfg["e"]),
            float(transform_cfg["f"]),
        ),
        "bounds": (
            float(bounds["left"]),
            float(bounds["bottom"]),
            float(bounds["right"]),
            float(bounds["top"]),
        ),
    }


def master_profile(grid: dict[str, Any]) -> dict[str, Any]:
    """Create a reusable GeoTIFF profile for aligned float features."""
    return {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }


def raster_matches_master(path: Path, grid: dict[str, Any], tolerance: float = 0.001) -> bool:
    """Return True when a raster is already on the project master grid."""
    with rasterio.open(path) as src:
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return (
            str(src.crs) == grid["crs"]
            and src.width == grid["width"]
            and src.height == grid["height"]
            and all(abs(actual - expected) <= tolerance for actual, expected in zip(src.transform[:6], grid["transform"][:6]))
            and all(abs(actual - expected) <= tolerance for actual, expected in zip(bounds, grid["bounds"]))
        )


def repair_swir1(grid: dict[str, Any]) -> list[str]:
    """Resample direct SWIR1 bands from native 20 m products to the 10 m master grid."""
    repaired: list[str] = []
    profile = master_profile(grid)
    swir1_paths = set(PROCESSED.glob("*_s2_swir1.tif")) | set(PROCESSED.glob("s2_*_swir1.tif"))
    for path in sorted(swir1_paths):
        if raster_matches_master(path, grid):
            continue

        with rasterio.open(path) as src:
            aligned = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
            reproject(
                source=src.read(1).astype("float32"),
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(aligned, 1)
            dst.set_band_description(1, "SWIR1 aligned")
        repaired.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return repaired


def master_grid_aoi_geojson(grid: dict[str, Any]) -> dict[str, Any]:
    """Build a WGS84 AOI polygon from the same master-grid corners as raster overlays."""
    left, bottom, right, top = grid["bounds"]
    xs = [left, right, right, left]
    ys = [top, top, bottom, bottom]
    lngs, lats = transform(grid["crs"], "EPSG:4326", xs, ys)
    coordinates = [[float(lngs[index]), float(lats[index])] for index in range(4)]
    return {
        "type": "FeatureCollection",
        "name": "kigali_demo_aoi_master_grid",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Kigali 20 km demo AOI",
                    "description": "Master-grid-aligned 10 m AOI footprint used by analysis rasters and WebGIS.",
                    "source": "configs/master_grid.yaml",
                },
                "geometry": {"type": "Polygon", "coordinates": [[*coordinates, coordinates[0]]]},
            }
        ],
    }


def sync_aoi_files(grid: dict[str, Any]) -> list[str]:
    """Write the same master-grid AOI to backend config and hosted WebGIS data."""
    payload = master_grid_aoi_geojson(grid)
    updated = []
    for path in [CONFIG_AOI, WEB_AOI]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        updated.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair master-grid alignment for derived products.")
    parser.parse_args()

    grid = load_master_grid()
    repaired_swir1 = repair_swir1(grid)
    updated_aoi = sync_aoi_files(grid)
    print(json.dumps({"repaired_swir1": repaired_swir1, "updated_aoi": updated_aoi}, indent=2))


if __name__ == "__main__":
    main()
