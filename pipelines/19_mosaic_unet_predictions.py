"""Mosaic patch-level U-Net inference outputs back to geospatial rasters."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"U-Net config not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def group_records_by_tag(records: list[dict[str, Any]], split: str | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if split and record["split"] != split:
            continue
        grouped[record["training_tag"]].append(record)
    return dict(grouped)


def reference_raster_path(processed_dir: Path, training_tag: str) -> Path:
    return processed_dir / f"{training_tag}_s2_ndvi.tif"


def write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], dtype: str, nodata: float | int) -> None:
    output_profile = profile.copy()
    output_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)


def mosaic_tag(
    tag: str,
    records: list[dict[str, Any]],
    processed_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    reference_path = reference_raster_path(processed_dir, tag)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference raster for U-Net mosaic not found: {reference_path}")
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width

    dominant_sum = np.zeros((height, width), dtype="float32")
    confidence_sum = np.zeros((height, width), dtype="float32")
    entropy_sum = np.zeros((height, width), dtype="float32")
    vote_count = np.zeros((height, width), dtype="uint16")
    class_vote_counts = np.zeros((5, height, width), dtype="uint16")

    for record in records:
        row = int(record["row"])
        col = int(record["col"])
        patch_height = int(record["height"])
        patch_width = int(record["width"])
        row_slice = slice(row, row + patch_height)
        col_slice = slice(col, col + patch_width)
        with np.load(record["path"]) as data:
            dominant = data["dominant_class"].astype("uint8")
            confidence = data["max_probability"].astype("float32")
            entropy = data["entropy"].astype("float32")

        dominant_sum[row_slice, col_slice] += dominant.astype("float32")
        confidence_sum[row_slice, col_slice] += confidence
        entropy_sum[row_slice, col_slice] += entropy
        vote_count[row_slice, col_slice] += 1
        for class_id in range(1, 6):
            class_vote_counts[class_id - 1, row_slice, col_slice] += (dominant == class_id).astype("uint16")

    covered = vote_count > 0
    dominant_class = np.zeros((height, width), dtype="uint8")
    if np.any(covered):
        dominant_class[covered] = (np.argmax(class_vote_counts[:, covered], axis=0) + 1).astype("uint8")
    confidence_mosaic = np.full((height, width), -9999.0, dtype="float32")
    entropy_mosaic = np.full((height, width), -9999.0, dtype="float32")
    confidence_mosaic[covered] = confidence_sum[covered] / vote_count[covered]
    entropy_mosaic[covered] = entropy_sum[covered] / vote_count[covered]

    dominant_path = output_dir / f"{tag}_unet_dominant_class.tif"
    confidence_path = output_dir / f"{tag}_unet_confidence.tif"
    entropy_path = output_dir / f"{tag}_unet_entropy.tif"
    coverage_path = output_dir / f"{tag}_unet_patch_coverage.tif"

    write_raster(dominant_path, dominant_class, profile, "uint8", 0)
    write_raster(confidence_path, confidence_mosaic, profile, "float32", -9999.0)
    write_raster(entropy_path, entropy_mosaic, profile, "float32", -9999.0)
    write_raster(coverage_path, vote_count, profile, "uint16", 0)

    unique, counts = np.unique(dominant_class[covered], return_counts=True)
    class_counts = {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}
    return {
        "training_tag": tag,
        "patch_count": len(records),
        "covered_pixel_count": int(np.count_nonzero(covered)),
        "coverage_fraction": round(float(np.count_nonzero(covered) / covered.size), 6),
        "dominant_class_counts": class_counts,
        "mean_confidence": round(float(np.mean(confidence_mosaic[covered])), 6) if np.any(covered) else None,
        "mean_entropy": round(float(np.mean(entropy_mosaic[covered])), 6) if np.any(covered) else None,
        "rasters": {
            "dominant_class": str(dominant_path).replace("\\", "/"),
            "confidence": str(confidence_path).replace("\\", "/"),
            "entropy": str(entropy_path).replace("\\", "/"),
            "patch_coverage": str(coverage_path).replace("\\", "/"),
        },
    }


def create_mosaics(config: dict[str, Any], settings: dict[str, Any], split: str | None = None) -> Path:
    ensure_output_dirs(config)
    inference_manifest = read_json(settings["outputs"]["inference_manifest"])
    grouped = group_records_by_tag(inference_manifest.get("records", []), split=split)
    if not grouped:
        raise RuntimeError("No U-Net inference records matched the requested mosaic filters.")

    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(settings["outputs"]["mosaic_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mosaics = [
        mosaic_tag(tag, records, processed_dir=processed_dir, output_dir=output_dir)
        for tag, records in sorted(grouped.items())
    ]
    manifest = {
        "phase": "phase_34_unet_prediction_mosaics",
        "source_inference_manifest": settings["outputs"]["inference_manifest"],
        "requested_split": split or "all",
        "mosaic_count": len(mosaics),
        "mosaics": mosaics,
        "notes": [
            "Mosaics are per training date-pair because patch predictions come from different Sentinel-1/2 acquisitions.",
            "Dominant-class rasters use 0 as nodata/uncovered and 1-5 as monitored model classes.",
            "Confidence and entropy rasters use -9999 nodata outside covered patch windows.",
        ],
    }
    manifest_path = Path(settings["outputs"]["mosaic_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create geospatial U-Net prediction mosaics from patch outputs.")
    parser.add_argument("--split", choices=["train", "validation", "test"], help="Optional patch split to mosaic.")
    args = parser.parse_args()

    config = load_config()
    settings = load_yaml("configs/model_unet.yaml")
    manifest_path = create_mosaics(config, settings, split=args.split)
    print(f"U-Net mosaic manifest: {manifest_path}")


if __name__ == "__main__":
    main()
