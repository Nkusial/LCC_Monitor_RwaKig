"""Validate that local processing outputs are complete and grid-aligned."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
import yaml

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


REQUIRED_OUTPUTS = [
    "monitoring_change_polygons.geojson",
    "monitoring_change_scored.geojson",
    "monitoring_change_summary.json",
    "monitoring_change_scored_summary.json",
]

ANALYSIS_SUFFIXES = (
    "_s2_ndvi.tif",
    "_s2_ndwi.tif",
    "_s2_ndbi.tif",
    "_s1_vv.tif",
    "_s1_vh.tif",
    "_s1_vv_vh_ratio.tif",
)


def raster_grid(path: Path) -> dict[str, Any]:
    """Return the grid fields that must match across analysis rasters."""
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": [round(value, 10) for value in src.transform[:6]],
            "width": src.width,
            "height": src.height,
            "bounds": [round(value, 3) for value in src.bounds],
            "nodata": "nan" if src.nodata is not None and math.isnan(src.nodata) else src.nodata,
            "dtype": src.dtypes[0],
        }


def load_master_grid(config: dict[str, Any]) -> dict[str, Any] | None:
    """Load the explicit AOI grid contract used by future ML-ready stages."""
    grid_path = config.get("project", {}).get("master_grid_path")
    if not grid_path:
        return None

    path = Path(grid_path)
    if not path.exists():
        return {"missing_path": str(path)}

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expected_grid(master_grid: dict[str, Any]) -> dict[str, Any]:
    """Normalize the YAML master-grid spec into comparable raster metadata."""
    bounds = master_grid["bounds"]
    transform = master_grid["transform"]
    return {
        "crs": master_grid["crs"],
        "transform": [
            round(float(transform[key]), 10)
            for key in ("a", "b", "c", "d", "e", "f")
        ],
        "width": int(master_grid["width"]),
        "height": int(master_grid["height"]),
        "bounds": [
            round(float(bounds[key]), 3)
            for key in ("left", "bottom", "right", "top")
        ],
        "nodata": str(master_grid.get("nodata")),
        "dtype": master_grid.get("dtype"),
    }


def compare_to_master_grid(
    raster_path: Path,
    master_grid: dict[str, Any] | None,
) -> list[str]:
    """Return master-grid fields that differ for a raster."""
    if not master_grid:
        return []
    if "missing_path" in master_grid:
        return ["master_grid_path"]

    actual = raster_grid(raster_path)
    expected = expected_grid(master_grid)
    policy = master_grid.get("alignment_policy", {})

    checks = {
        "crs": policy.get("fail_on_crs_mismatch", True),
        "transform": policy.get("fail_on_transform_mismatch", True),
        "width": policy.get("fail_on_shape_mismatch", True),
        "height": policy.get("fail_on_shape_mismatch", True),
        "bounds": policy.get("fail_on_bounds_mismatch", True),
        "nodata": True,
        "dtype": policy.get("fail_on_dtype_mismatch", False),
    }

    mismatches = []
    for key, should_fail in checks.items():
        if should_fail and actual.get(key) != expected.get(key):
            mismatches.append(key)
    return mismatches


def validate_outputs(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Write a compact report for portfolio reproducibility checks."""
    ensure_output_dirs(config)
    processed_dir = Path(config["paths"]["processed"])
    outputs_dir = Path(config["paths"]["outputs"])

    missing = [name for name in REQUIRED_OUTPUTS if not (outputs_dir / name).exists()]
    monitoring_rasters = sorted(
        path
        for path in processed_dir.glob("monitoring_*.tif")
        if path.name.endswith(ANALYSIS_SUFFIXES)
    )
    training_rasters = sorted(
        path
        for path in processed_dir.glob("training_*.tif")
        if path.name.endswith(ANALYSIS_SUFFIXES)
    )
    if not monitoring_rasters:
        missing.append("data/processed/monitoring analysis rasters")

    reference_grid = raster_grid(monitoring_rasters[0]) if monitoring_rasters else None
    master_grid = load_master_grid(config)
    grid_mismatches = []
    master_grid_mismatches = {}
    for path in monitoring_rasters[1:]:
        # Raster alignment is a hard requirement before change detection,
        # classification, or ML feature stacking can be trusted.
        if raster_grid(path) != reference_grid:
            grid_mismatches.append(path.name)
    for path in monitoring_rasters:
        mismatched_fields = compare_to_master_grid(path, master_grid)
        if mismatched_fields:
            master_grid_mismatches[path.name] = mismatched_fields
    training_master_grid_mismatches = {}
    for path in training_rasters:
        mismatched_fields = compare_to_master_grid(path, master_grid)
        if mismatched_fields:
            training_master_grid_mismatches[path.name] = mismatched_fields

    scored_path = outputs_dir / "monitoring_change_scored.geojson"
    scored = gpd.read_file(scored_path) if scored_path.exists() else gpd.GeoDataFrame()
    publish_ready_count = (
        int(scored["publish_ready"].sum()) if not scored.empty and "publish_ready" in scored else 0
    )

    report = {
        "status": (
            "ok"
            if not missing
            and not grid_mismatches
            and not master_grid_mismatches
            and not training_master_grid_mismatches
            else "needs_attention"
        ),
        "missing_outputs": missing,
        "analysis_raster_count": len(monitoring_rasters),
        "training_analysis_raster_count": len(training_rasters),
        "reference_grid": reference_grid,
        "master_grid": expected_grid(master_grid) if master_grid and "missing_path" not in master_grid else master_grid,
        "grid_mismatches": grid_mismatches,
        "master_grid_mismatches": master_grid_mismatches,
        "training_master_grid_mismatches": training_master_grid_mismatches,
        "scored_change_count": int(len(scored)),
        "publish_ready_count": publish_ready_count,
        "monitored_land_cover_counts": (
            {key: int(value) for key, value in scored.groupby("monitored_land_cover").size().to_dict().items()}
            if not scored.empty and "monitored_land_cover" in scored
            else {}
        ),
        "unsupervised_validation_available": (
            outputs_dir / "unsupervised_validation_report.json"
        ).exists(),
        "weak_labels_available": any(outputs_dir.glob("weak_labels_*_summary.json")),
        "external_weak_sources_available": (
            Path(config["paths"]["interim"]) / "external_sources" / "external_weak_sources_summary.json"
        ).exists(),
        "patch_dataset_available": (
            Path(config["paths"]["interim"]) / "patches" / "patch_manifest.json"
        ).exists(),
        "training_scene_catalog_available": (
            Path(config["paths"]["catalog"]) / "training_scene_pairs_2023_2026.json"
        ).exists(),
        "training_stack_manifest_available": (
            Path(config["paths"]["catalog"]) / "training_stack_manifest_2023_2026.json"
        ).exists(),
        "unet_baseline_report_available": (
            outputs_dir / "unet_baseline_report.json"
        ).exists(),
        "training_preprocessing_summary_available": (
            Path(config["paths"]["catalog"]) / "training_preprocessing_summary_2023_2026.json"
        ).exists(),
        "multi_year_training_patch_manifest_available": (
            Path(config["paths"]["interim"]) / "training_patches" / "training_patch_manifest.json"
        ).exists(),
    }

    report_path = outputs_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local monitoring outputs.")
    parser.parse_args()

    report, report_path = validate_outputs(load_config())
    print(f"Validation report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
