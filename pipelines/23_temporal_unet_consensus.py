"""Build temporal consensus rasters from full-AOI U-Net predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "unet_full_aoi" / "unet_full_aoi_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "unet_full_aoi"
DEFAULT_REPORT = ROOT / "data" / "outputs" / "unet_temporal_consensus_report.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], dtype: str, nodata: int | float) -> None:
    import rasterio

    output_profile = profile.copy()
    output_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)


def consensus(records: list[dict[str, Any]], split: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import rasterio

    selected = [record for record in records if split is None or record.get("split") == split]
    if not selected:
        raise RuntimeError(f"No full-AOI U-Net records found for split={split!r}.")

    arrays = []
    profile = None
    for record in selected:
        path = repo_path(record["rasters"]["dominant_class"])
        with rasterio.open(path) as src:
            data = src.read(1).astype("uint8")
            current_profile = src.profile.copy()
        if profile is None:
            profile = current_profile
        elif (
            current_profile["width"] != profile["width"]
            or current_profile["height"] != profile["height"]
            or current_profile["transform"] != profile["transform"]
            or current_profile["crs"] != profile["crs"]
        ):
            raise ValueError(f"U-Net dominant-class raster is not aligned: {path}")
        arrays.append(data)

    stack = np.stack(arrays, axis=0)
    # Temporal consensus is a reliability control: pixels that keep the same
    # model class across dates are safer to interpret than one-date predictions.
    valid_count = np.count_nonzero(stack > 0, axis=0).astype("uint8")
    class_votes = np.zeros((5, *stack.shape[1:]), dtype="uint8")
    for class_id in range(1, 6):
        class_votes[class_id - 1] = np.count_nonzero(stack == class_id, axis=0)

    mode_count = np.max(class_votes, axis=0)
    mode_class = (np.argmax(class_votes, axis=0) + 1).astype("uint8")
    mode_class[valid_count == 0] = 0
    # Low consensus does not mean "wrong"; it means the pixel should remain a
    # review zone or require stronger evidence before alert publication.
    consensus_fraction = np.where(valid_count > 0, mode_count / np.maximum(valid_count, 1), -9999).astype("float32")
    instability = ((valid_count > 0) & (consensus_fraction < 0.6)).astype("uint8")

    metadata = {
        "record_count": len(selected),
        "splits_used": sorted(set(record["split"] for record in selected)),
        "years_used": sorted(set(record["year"] for record in selected)),
        "class_counts": {
            str(class_id): int(np.count_nonzero(mode_class == class_id))
            for class_id in range(1, 6)
        },
        "mean_consensus_fraction": round(float(consensus_fraction[valid_count > 0].mean()), 6),
        "unstable_pixel_count": int(np.count_nonzero(instability)),
        "valid_pixel_count": int(np.count_nonzero(valid_count > 0)),
    }
    if profile is None:
        raise RuntimeError("No temporal consensus profile was created.")
    return mode_class, consensus_fraction, instability, {"profile": profile, "metadata": metadata}


def build_temporal_consensus(manifest_path: Path, output_dir: Path, report_path: Path, split: str | None) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    mode_class, consensus_fraction, instability, result = consensus(manifest.get("records", []), split)
    profile = result["profile"]
    suffix = split or "all"

    class_path = output_dir / f"unet_temporal_consensus_{suffix}_class.tif"
    fraction_path = output_dir / f"unet_temporal_consensus_{suffix}_fraction.tif"
    instability_path = output_dir / f"unet_temporal_consensus_{suffix}_instability.tif"
    write_raster(class_path, mode_class, profile, "uint8", 0)
    write_raster(fraction_path, consensus_fraction, profile, "float32", -9999)
    write_raster(instability_path, instability, profile, "uint8", 0)

    report = {
        "phase": "phase_40_temporal_unet_consensus",
        "purpose": "Summarize temporally persistent U-Net predictions and unstable areas across available full-AOI date pairs.",
        "split": split or "all",
        **result["metadata"],
        "outputs": {
            "consensus_class": str(class_path.relative_to(ROOT)).replace("\\", "/"),
            "consensus_fraction": str(fraction_path.relative_to(ROOT)).replace("\\", "/"),
            "instability_mask": str(instability_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "limitations": [
            "This is model temporal stability, not field accuracy.",
            "Dynamic World improves the weak-source agreement side once Earth Engine export is authenticated.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal consensus from full-AOI U-Net outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", default=None, choices=[None, "train", "validation", "test"], help="Optional split filter.")
    args = parser.parse_args()

    report = build_temporal_consensus(args.manifest, args.output_dir, args.report, args.split)
    print(f"Temporal U-Net consensus report: {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
