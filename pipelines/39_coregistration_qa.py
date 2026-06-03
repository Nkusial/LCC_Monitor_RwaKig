"""Report diagnostic co-registration quality for multi-source monitoring rasters.

The master grid guarantees identical CRS, transform, bounds, and shape. This
script adds a second, more honest QA layer: it compares image-content gradients
against a Sentinel-2 reference to estimate whether source families appear
shifted after harmonization. The result is diagnostic, not a field-accuracy
claim, because SAR, optical indices, weak labels, and model outputs do not share
the same radiometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import yaml


ROOT = Path(__file__).resolve().parents[1]
MASTER_GRID_PATH = ROOT / "configs" / "master_grid.yaml"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "data" / "outputs"
EXTERNAL_SOURCES = ROOT / "data" / "interim" / "external_sources"
UNET_FULL_AOI_V3 = ROOT / "data" / "interim" / "unet_full_aoi_v3"
UNET_FULL_AOI_BASELINE = ROOT / "data" / "interim" / "unet_full_aoi"
REPORT_PATH = OUTPUTS / "coregistration_qa_report.json"

REFERENCE_RASTER = PROCESSED / "s2_20250220_ndvi.tif"
SAMPLE_STRIDE = 4
MAX_SHIFT_PIXELS = 3


def load_master_grid() -> dict:
    return yaml.safe_load(MASTER_GRID_PATH.read_text(encoding="utf-8"))


def expected_grid() -> dict:
    grid = load_master_grid()
    bounds = grid["bounds"]
    transform = grid["transform"]
    return {
        "crs": grid["crs"],
        "transform": [float(transform[key]) for key in ("a", "b", "c", "d", "e", "f")],
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "bounds": [float(bounds[key]) for key in ("left", "bottom", "right", "top")],
        "tolerance": float(grid.get("alignment_policy", {}).get("tolerance", 0.001)),
    }


def dataset_grid(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        return {
            "crs": str(dataset.crs),
            "transform": [float(value) for value in dataset.transform[:6]],
            "width": int(dataset.width),
            "height": int(dataset.height),
            "bounds": [
                float(dataset.bounds.left),
                float(dataset.bounds.bottom),
                float(dataset.bounds.right),
                float(dataset.bounds.top),
            ],
        }


def grid_matches(path: Path, expected: dict) -> bool:
    actual = dataset_grid(path)
    tolerance = expected["tolerance"]
    if actual["crs"] != expected["crs"]:
        return False
    if actual["width"] != expected["width"] or actual["height"] != expected["height"]:
        return False
    for key in ("transform", "bounds"):
        if any(abs(a - b) > tolerance for a, b in zip(actual[key], expected[key])):
            return False
    return True


def read_sample(path: Path) -> np.ndarray:
    """Read a normalized gradient surface for shift diagnostics."""
    with rasterio.open(path) as dataset:
        height = max(1, dataset.height // SAMPLE_STRIDE)
        width = max(1, dataset.width // SAMPLE_STRIDE)
        array = dataset.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype("float32")

    values = array.compressed()
    if values.size == 0:
        return np.zeros(array.shape, dtype="float32")

    low, high = np.nanpercentile(values, [2, 98])
    if np.isclose(low, high):
        high = low + 1
    normalized = np.clip((array.filled(np.nan) - low) / (high - low), 0, 1)
    gy, gx = np.gradient(np.nan_to_num(normalized, nan=0.0))
    gradient = np.hypot(gx, gy).astype("float32")
    gradient[~np.isfinite(normalized)] = np.nan
    return gradient


def overlap_for_shift(
    reference: np.ndarray,
    target: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref_y0 = max(0, dy)
    ref_y1 = min(reference.shape[0], target.shape[0] + dy)
    tar_y0 = max(0, -dy)
    tar_y1 = tar_y0 + (ref_y1 - ref_y0)
    ref_x0 = max(0, dx)
    ref_x1 = min(reference.shape[1], target.shape[1] + dx)
    tar_x0 = max(0, -dx)
    tar_x1 = tar_x0 + (ref_x1 - ref_x0)
    return reference[ref_y0:ref_y1, ref_x0:ref_x1], target[tar_y0:tar_y1, tar_x0:tar_x1]


def correlation(a: np.ndarray, b: np.ndarray) -> tuple[float | None, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 500:
        return None, int(mask.sum())
    av = a[mask] - float(np.nanmean(a[mask]))
    bv = b[mask] - float(np.nanmean(b[mask]))
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denom == 0:
        return None, int(mask.sum())
    return float(np.sum(av * bv) / denom), int(mask.sum())


def estimate_shift(reference: np.ndarray, target: np.ndarray) -> dict:
    """Estimate best integer shift using local gradient correlation."""
    best = {"dx_pixels": 0, "dy_pixels": 0, "correlation": None, "valid_samples": 0}
    for dy in range(-MAX_SHIFT_PIXELS, MAX_SHIFT_PIXELS + 1):
        for dx in range(-MAX_SHIFT_PIXELS, MAX_SHIFT_PIXELS + 1):
            ref, tar = overlap_for_shift(reference, target, dy, dx)
            score, count = correlation(ref, tar)
            if score is None:
                continue
            if best["correlation"] is None or score > float(best["correlation"]):
                best = {
                    "dx_pixels": dx * SAMPLE_STRIDE,
                    "dy_pixels": dy * SAMPLE_STRIDE,
                    "correlation": score,
                    "valid_samples": count,
                }
    return best


def target_rasters() -> list[dict]:
    """Select representative rasters from optical, radar, weak-label, and ML families."""
    unet_tag = "training_2025_20250829_20250901"
    candidates = [
        ("sentinel2_index", "NDWI moisture index", PROCESSED / "s2_20250220_ndwi.tif"),
        ("sentinel2_index", "NDBI built-up index", PROCESSED / "s2_20250220_ndbi.tif"),
        ("sentinel1_radar", "Sentinel-1 VV backscatter", PROCESSED / "s1_20250221_vv.tif"),
        ("sentinel1_radar", "Sentinel-1 VH backscatter", PROCESSED / "s1_20250221_vh.tif"),
        ("weak_label", "Dynamic World harmonized label", EXTERNAL_SOURCES / "dynamic_world_label_harmonized.tif"),
        ("weak_label", "ESA WorldCover harmonized label", EXTERNAL_SOURCES / "esa_worldcover_harmonized.tif"),
        (
            "change_output",
            "Latest delta NDVI",
            OUTPUTS / "monitoring_20250220_20250221__to__monitoring_20250421_20250419_delta_ndvi.tif",
        ),
        (
            "unet_output",
            "U-Net confidence",
            UNET_FULL_AOI_V3 / f"{unet_tag}_unet_full_aoi_confidence.tif",
        ),
        (
            "test_output",
            "2026 frozen U-Net confidence",
            UNET_FULL_AOI_BASELINE / "test_2026_20260126_20260123_unet_full_aoi_confidence.tif",
        ),
    ]
    return [
        {"family": family, "label": label, "path": path}
        for family, label, path in candidates
        if path.exists()
    ]


def main() -> None:
    expected = expected_grid()
    reference = read_sample(REFERENCE_RASTER)
    records = []
    grid_failures = []

    for target in target_rasters():
        path = target["path"]
        grid_ok = grid_matches(path, expected)
        if not grid_ok:
            grid_failures.append(str(path.relative_to(ROOT)).replace("\\", "/"))

        shift = estimate_shift(reference, read_sample(path))
        abs_shift = max(abs(int(shift["dx_pixels"])), abs(int(shift["dy_pixels"])))
        records.append(
            {
                "family": target["family"],
                "label": target["label"],
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "grid_aligned": grid_ok,
                "diagnostic_shift_pixels": {
                    "dx": shift["dx_pixels"],
                    "dy": shift["dy_pixels"],
                    "max_abs": abs_shift,
                },
                "diagnostic_shift_m": abs_shift * expected["transform"][0],
                "gradient_correlation": shift["correlation"],
                "valid_samples": shift["valid_samples"],
                "interpretation": (
                    "Strong image-content agreement"
                    if shift["correlation"] is not None and shift["correlation"] >= 0.45
                    else "Low cross-source radiometric similarity; use as drift warning, not proof"
                ),
            }
        )

    high_shift_records = [
        record
        for record in records
        if record["diagnostic_shift_pixels"]["max_abs"] > 10
    ]
    report = {
        "phase": "coregistration_qa",
        "status": "warning" if grid_failures or high_shift_records else "passed",
        "method": (
            "Grid metadata validation plus gradient-correlation integer shift diagnostics "
            "against Sentinel-2 NDVI. This is a no-GCP proxy check for prototype QA."
        ),
        "reference": str(REFERENCE_RASTER.relative_to(ROOT)).replace("\\", "/"),
        "master_grid": {
            "path": "configs/master_grid.yaml",
            "crs": expected["crs"],
            "resolution_m": expected["transform"][0],
        },
        "records": records,
        "grid_failure_count": len(grid_failures),
        "high_shift_warning_count": len(high_shift_records),
        "limitations": [
            "This does not replace field GCPs or orthorectification QA.",
            "SAR and optical images can have low gradient correlation even when georeferenced correctly.",
            "OSM is not used as ground-control because its local digitizing offset is unknown.",
        ],
        "recommended_production_method": [
            "Use independent GCPs or high-resolution reference imagery for tie-point registration.",
            "Estimate sub-pixel shifts per date/source before change detection.",
            "Publish WebGIS rasters as Web Mercator tiles, not single image overlays.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Co-registration QA report: {REPORT_PATH}")
    print(f"Status: {report['status']} ({len(records)} targets checked)")


if __name__ == "__main__":
    main()
