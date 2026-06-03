"""Validate geospatial alignment across rasters and vectors used by the platform.

This report is intentionally stricter than a visual WebGIS check. It verifies
that every analysis raster family used for Sentinel features, weak labels,
change detection, and model review sits on the same master grid before those
products are trusted by the API or frontend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
import yaml
from rasterio.transform import Affine
from rasterio.warp import transform
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
MASTER_GRID_PATH = ROOT / "configs" / "master_grid.yaml"
OUTPUT_REPORT = ROOT / "data" / "outputs" / "geospatial_alignment_report.json"


RASTER_FAMILIES = {
    "processed_sentinel_features": ROOT / "data" / "processed",
    "change_detection_outputs": ROOT / "data" / "outputs",
    "external_weak_sources": ROOT / "data" / "interim" / "external_sources",
    "unet_full_aoi": ROOT / "data" / "interim" / "unet_full_aoi",
    "unet_full_aoi_v3": ROOT / "data" / "interim" / "unet_full_aoi_v3",
    "refined_pseudo_labels": ROOT / "data" / "interim" / "refined_pseudo_labels",
    "topography_features": ROOT / "data" / "interim" / "topography",
}

VECTOR_PRODUCTS = {
    "config_aoi": ROOT / "configs" / "aoi.geojson",
    "webgis_aoi": ROOT / "web" / "public" / "demo" / "aoi.geojson",
    "change_polygons": ROOT / "data" / "outputs" / "monitoring_change_scored.geojson",
    "webgis_change_polygons": ROOT / "web" / "public" / "demo" / "changes.geojson",
}

RAW_SOURCE_RASTER_NAMES = {
    "dynamic_world_label.tif",
    "dynamic_world_label_20260126.tif",
}


def load_master_grid() -> dict[str, Any]:
    """Load the authoritative grid contract from configs/master_grid.yaml."""
    grid = yaml.safe_load(MASTER_GRID_PATH.read_text(encoding="utf-8"))
    transform = grid["transform"]
    bounds = grid["bounds"]
    return {
        "crs": grid["crs"],
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "transform": Affine(
            float(transform["a"]),
            float(transform["b"]),
            float(transform["c"]),
            float(transform["d"]),
            float(transform["e"]),
            float(transform["f"]),
        ),
        "bounds": (
            float(bounds["left"]),
            float(bounds["bottom"]),
            float(bounds["right"]),
            float(bounds["top"]),
        ),
        "tolerance": float(grid.get("alignment_policy", {}).get("tolerance", 0.001)),
    }


def close_enough(actual: list[float] | tuple[float, ...], expected: list[float] | tuple[float, ...], tolerance: float) -> bool:
    """Compare floating-point geospatial metadata with millimetre-scale tolerance."""
    return len(actual) == len(expected) and all(
        abs(actual_value - expected_value) <= tolerance
        for actual_value, expected_value in zip(actual, expected)
    )


def master_grid_coordinates_wgs84(expected: dict[str, Any]) -> list[list[float]]:
    """Return exact WGS84 corners for AOI files that describe the master grid."""
    left, bottom, right, top = expected["bounds"]
    xs = [left, right, right, left]
    ys = [top, top, bottom, bottom]
    lngs, lats = transform(expected["crs"], "EPSG:4326", xs, ys)
    return [[float(lngs[index]), float(lats[index])] for index in range(4)]


def check_raster(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Check one GeoTIFF against CRS, transform, shape, and bounds requirements."""
    with rasterio.open(path) as src:
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        transform = tuple(float(value) for value in src.transform[:6])
        expected_transform = tuple(float(value) for value in expected["transform"][:6])
        mismatches: list[str] = []

        if str(src.crs) != expected["crs"]:
            mismatches.append("crs")
        if src.width != expected["width"] or src.height != expected["height"]:
            mismatches.append("shape")
        if not close_enough(transform, expected_transform, expected["tolerance"]):
            mismatches.append("transform")
        if not close_enough(bounds, expected["bounds"], expected["tolerance"]):
            mismatches.append("bounds")

        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "aligned": not mismatches,
            "mismatches": mismatches,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "transform": [round(value, 6) for value in transform],
            "bounds": [round(value, 6) for value in bounds],
        }


def iter_rasters() -> list[tuple[str, Path]]:
    """Collect only analysis raster families that should already be master-grid aligned."""
    rasters: list[tuple[str, Path]] = []
    for family, directory in RASTER_FAMILIES.items():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.tif")):
            if path.name in RAW_SOURCE_RASTER_NAMES:
                # Raw external exports are preserved in their provider CRS as
                # provenance. The harmonized *_harmonized*.tif products are the
                # rasters consumed by weak-label fusion and model validation.
                continue
            rasters.append((family, path))
    return rasters


def check_vectors(expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify AOI and change polygons are in or inside the master-grid footprint."""
    master_polygon = box(*expected["bounds"])
    expected_aoi_corners = master_grid_coordinates_wgs84(expected)
    records: list[dict[str, Any]] = []
    for name, path in VECTOR_PRODUCTS.items():
        if not path.exists():
            continue

        gdf = gpd.read_file(path)
        if gdf.empty:
            records.append(
                {
                    "name": name,
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "aligned": True,
                    "feature_count": 0,
                    "note": "empty vector product",
                }
            )
            continue

        if name.endswith("aoi"):
            ring = list(gdf.geometry.iloc[0].exterior.coords)[:4]
            actual_corners = [[float(point[0]), float(point[1])] for point in ring]
            aligned = all(
                close_enough(actual, expected_corner, 1e-9)
                for actual, expected_corner in zip(actual_corners, expected_aoi_corners)
            )
            rule = "must match master-grid footprint"
            bounds = tuple(float(value) for value in gdf.to_crs(expected["crs"]).total_bounds)
        else:
            projected = gdf.to_crs(expected["crs"])
            bounds = tuple(float(value) for value in projected.total_bounds)
            vector_box = box(*bounds)
            # Change polygons only need to stay inside the master grid because
            # they represent changed subsets rather than the full AOI footprint.
            aligned = master_polygon.buffer(2.0).contains(vector_box)
            rule = "must stay inside master-grid footprint"

        records.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "aligned": bool(aligned),
                "rule": rule,
                "feature_count": int(len(gdf)),
                "bounds": [round(value, 3) for value in bounds],
            }
        )
    return records


def build_report() -> dict[str, Any]:
    """Build the full geospatial alignment report without modifying source data."""
    expected = load_master_grid()
    raster_records = []
    for family, path in iter_rasters():
        record = check_raster(path, expected)
        record["family"] = family
        raster_records.append(record)

    vector_records = check_vectors(expected)
    raster_family_counts: dict[str, int] = {}
    for record in raster_records:
        raster_family_counts[record["family"]] = raster_family_counts.get(record["family"], 0) + 1
    raster_mismatches = [record for record in raster_records if not record["aligned"]]
    vector_mismatches = [record for record in vector_records if not record["aligned"]]

    return {
        "phase": "geospatial_alignment_validation",
        "master_grid": {
            "path": "configs/master_grid.yaml",
            "crs": expected["crs"],
            "width": expected["width"],
            "height": expected["height"],
            "bounds": [round(value, 3) for value in expected["bounds"]],
            "transform": [round(value, 6) for value in expected["transform"][:6]],
            "tolerance": expected["tolerance"],
        },
        "checked_raster_count": len(raster_records),
        "checked_vector_count": len(vector_records),
        "skipped_raw_source_rasters": sorted(RAW_SOURCE_RASTER_NAMES),
        "raster_mismatch_count": len(raster_mismatches),
        "vector_mismatch_count": len(vector_mismatches),
        "raster_family_counts": raster_family_counts,
        "status": "passed" if not raster_mismatches and not vector_mismatches else "failed",
        "vectors": vector_records,
        "mismatches": {
            "rasters": raster_mismatches,
            "vectors": vector_mismatches,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project raster/vector alignment.")
    parser.add_argument("--allow-fail", action="store_true", help="Write the report without exiting non-zero.")
    args = parser.parse_args()

    report = build_report()
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Geospatial alignment report: {OUTPUT_REPORT}")
    print(
        f"Status: {report['status']} | rasters checked: {report['checked_raster_count']} | "
        f"raster mismatches: {report['raster_mismatch_count']} | "
        f"vector mismatches: {report['vector_mismatch_count']}"
    )
    if report["status"] != "passed" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
