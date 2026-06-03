"""Ingest open DEM data and derive topographic features for Kigali AOI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer
import rasterio
import yaml
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from common import load_config
except ModuleNotFoundError:
    from pipelines.common import load_config


TOPO_DIR = ROOT / "data" / "interim" / "topography"
REPORT = ROOT / "data" / "outputs" / "topography_ingestion_report.json"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def aoi_bbox(config: dict[str, Any]) -> list[float]:
    aoi = json.loads((ROOT / config["project"]["aoi_path"]).read_text(encoding="utf-8"))
    return [float(value) for value in shape(aoi["features"][0]["geometry"]).bounds]


def master_profile(config: dict[str, Any]) -> dict[str, Any]:
    grid = load_yaml(ROOT / config["project"]["master_grid_path"])
    transform = Affine(
        float(grid["transform"]["a"]),
        float(grid["transform"]["b"]),
        float(grid["transform"]["c"]),
        float(grid["transform"]["d"]),
        float(grid["transform"]["e"]),
        float(grid["transform"]["f"]),
    )
    return {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": transform,
        "nodata": -9999,
        "compress": "deflate",
    }


def write_raster(path: Path, array: np.ndarray, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


def fetch_dem(config: dict[str, Any], collection: str) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    profile = master_profile(config)
    client = Client.open(config["sentinel"]["stac_api_url"])
    items = list(client.search(collections=[collection], bbox=aoi_bbox(config), limit=12).items())
    if not items:
        raise RuntimeError(f"No DEM items found for collection {collection}.")

    sources = []
    item_ids = []
    try:
        for item in items:
            signed = planetary_computer.sign(item)
            href = signed.assets["data"].href
            src = rasterio.open(href)
            vrt = WarpedVRT(
                src,
                crs=profile["crs"],
                transform=profile["transform"],
                width=profile["width"],
                height=profile["height"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                nodata=-9999,
            )
            sources.append(vrt)
            item_ids.append(item.id)
        mosaic, _ = merge(sources, bounds=None, nodata=-9999)
        return mosaic[0].astype("float32"), profile, item_ids
    finally:
        for source in sources:
            parent = getattr(source, "src_dataset", None)
            source.close()
            if parent is not None:
                parent.close()


def derive_topography(elevation: np.ndarray, resolution_m: float) -> dict[str, np.ndarray]:
    valid = elevation != -9999
    clean = np.where(valid, elevation, np.nan).astype("float32")
    gy, gx = np.gradient(clean, resolution_m, resolution_m)
    slope = np.degrees(np.arctan(np.sqrt((gx**2) + (gy**2)))).astype("float32")
    ruggedness = np.full(clean.shape, -9999, dtype="float32")
    padded = np.pad(clean, 1, mode="edge")
    for row_offset in range(3):
        for col_offset in range(3):
            if row_offset == 1 and col_offset == 1:
                continue
            window = padded[row_offset : row_offset + clean.shape[0], col_offset : col_offset + clean.shape[1]]
            diff = np.abs(clean - window)
            ruggedness = np.where(ruggedness == -9999, diff, np.fmax(ruggedness, diff))
    topographic_position = clean - np.nanmean(
        np.stack(
            [
                padded[row_offset : row_offset + clean.shape[0], col_offset : col_offset + clean.shape[1]]
                for row_offset in range(3)
                for col_offset in range(3)
            ],
            axis=0,
        ),
        axis=0,
    )
    return {
        "dem": np.where(valid, elevation, -9999).astype("float32"),
        "slope": np.where(valid & np.isfinite(slope), slope, -9999).astype("float32"),
        "ruggedness": np.where(valid & np.isfinite(ruggedness), ruggedness, -9999).astype("float32"),
        "tpi": np.where(valid & np.isfinite(topographic_position), topographic_position, -9999).astype("float32"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Copernicus DEM and derive topographic features.")
    parser.add_argument("--collection", default="cop-dem-glo-30")
    args = parser.parse_args()

    config = load_config()
    elevation, profile, item_ids = fetch_dem(config, args.collection)
    resolution_m = abs(float(profile["transform"].a))
    features = derive_topography(elevation, resolution_m)

    outputs = {}
    for name, array in features.items():
        path = TOPO_DIR / f"kigali_{name}.tif"
        write_raster(path, array, profile)
        outputs[name] = str(path.relative_to(ROOT)).replace("\\", "/")

    valid_dem = features["dem"][features["dem"] != -9999]
    valid_slope = features["slope"][features["slope"] != -9999]
    report = {
        "phase": "phase_47_topography_ingestion",
        "source": "Copernicus DEM via Microsoft Planetary Computer",
        "collection": args.collection,
        "license_note": "Copernicus DEM is open data; verify attribution in final publication materials.",
        "item_ids": item_ids,
        "outputs": outputs,
        "summary": {
            "valid_pixel_count": int(valid_dem.size),
            "elevation_min_m": round(float(np.nanmin(valid_dem)), 2),
            "elevation_max_m": round(float(np.nanmax(valid_dem)), 2),
            "elevation_mean_m": round(float(np.nanmean(valid_dem)), 2),
            "slope_mean_degrees": round(float(np.nanmean(valid_slope)), 2),
            "slope_p95_degrees": round(float(np.nanpercentile(valid_slope, 95)), 2),
        },
        "why_it_matters": [
            "Kigali has mountainous terrain, so slope/elevation can explain radar backscatter, shadow, drainage, settlement patterns, and false change signals.",
            "Topography should be used as ancillary model features and as a review cue, not as a land-cover label by itself.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Topography ingestion report: {REPORT}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
