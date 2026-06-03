"""Generate multi-year ML patches from processed Sentinel-1/2 training stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config

FEATURE_FILE_SUFFIXES = {
    "s2_ndvi": "s2_ndvi",
    "s2_ndwi": "s2_ndwi",
    "s2_ndbi": "s2_ndbi",
    "s2_evi2": "s2_evi2",
    "s2_savi": "s2_savi",
    "s2_mndwi": "s2_mndwi",
    "s2_bsi": "s2_bsi",
    "s1_vv": "s1_vv",
    "s1_vh": "s1_vh",
    "s1_vv_vh_ratio": "s1_vv_vh_ratio",
    "s1_log_ratio": "s1_log_ratio",
}
TOPOGRAPHY_FEATURES = {
    "dem": "data/interim/topography/kigali_dem.tif",
    "slope": "data/interim/topography/kigali_slope.tif",
    "ruggedness": "data/interim/topography/kigali_ruggedness.tif",
    "tpi": "data/interim/topography/kigali_tpi.tif",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def read_feature_stack(processed_dir: Path, tag: str, features: list[str]) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    arrays = []
    reference_profile = None
    valid_mask = None
    for feature in features:
        if feature in TOPOGRAPHY_FEATURES:
            path = Path(TOPOGRAPHY_FEATURES[feature])
        else:
            path = processed_dir / f"{tag}_{FEATURE_FILE_SUFFIXES[feature]}.tif"
        data, profile = read_raster(path)
        data = data.astype("float32")
        if reference_profile is None:
            reference_profile = profile
            valid_mask = np.isfinite(data)
        else:
            if (
                profile["width"] != reference_profile["width"]
                or profile["height"] != reference_profile["height"]
                or profile["transform"] != reference_profile["transform"]
                or profile["crs"] != reference_profile["crs"]
            ):
                raise ValueError(f"Training feature raster is not on the master grid: {path}")
            valid_mask &= np.isfinite(data)
        arrays.append(data)

    if reference_profile is None or valid_mask is None:
        raise RuntimeError("No feature rasters were read.")
    return np.stack(arrays, axis=0).astype("float32"), reference_profile, valid_mask


def iter_patch_windows(height: int, width: int, patch_size: int, stride: int):
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield row, col


def patch_transform(profile: dict[str, Any], row: int, col: int) -> list[float]:
    transform = profile["transform"] * rasterio.Affine.translation(col, row)
    return [round(value, 10) for value in transform[:6]]


def load_settings(config: dict[str, Any], settings_path: Path | None = None) -> dict[str, Any]:
    import yaml

    path = settings_path or Path(config["project"]["patch_dataset_path"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def training_pairs(summary_path: Path, split: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    pairs = summary.get("pairs", [])
    if split:
        pairs = [pair for pair in pairs if pair["split"] == split]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def generate_training_patch_dataset(
    config: dict[str, Any],
    settings_path: Path | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> Path:
    """Create train/validation/test patches from all processed training stacks."""
    ensure_output_dirs(config)
    settings = load_settings(config, settings_path=settings_path)
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(settings["multi_year_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(config["paths"]["catalog"]) / "training_preprocessing_summary_2023_2026.json"
    pairs = training_pairs(summary_path, split=split, limit=limit)
    if not pairs:
        raise RuntimeError("No processed training pairs matched the requested filters.")

    label_sources = settings["label_sources"]
    use_refined_labels = "refined_labels" in label_sources
    if not use_refined_labels:
        agreement_labels, _ = read_raster(Path(label_sources["agreement_labels"]))
        agreement_count, _ = read_raster(Path(label_sources["agreement_count"]))
        disagreement_mask, _ = read_raster(Path(label_sources["disagreement_mask"]))
    else:
        agreement_labels = agreement_count = disagreement_mask = None

    patch_size = int(settings["patch_size"])
    stride = int(settings["stride"])
    max_per_pair = int(settings["max_patches_per_training_pair"])
    minimum_agreement_fraction = float(settings["minimum_agreement_fraction"])
    minimum_valid_feature_fraction = float(settings["minimum_valid_feature_fraction"])
    minimum_training_pixels = int(settings["minimum_training_pixels"])

    records = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    class_pixel_counts: dict[str, int] = {}

    for pair in pairs:
        tag = pair["output_tag"]
        features, profile, valid_feature_mask = read_feature_stack(processed_dir, tag, settings["feature_stack"])
        if use_refined_labels:
            patch_labels, _ = read_raster(Path(label_sources["refined_labels"].format(tag=tag)))
            refined_training_mask, _ = read_raster(Path(label_sources["refined_training_mask"].format(tag=tag)))
            refined_label_confidence, _ = read_raster(Path(label_sources["refined_label_confidence"].format(tag=tag)))
            pair_agreement_count = np.where(refined_training_mask > 0, 3, 0).astype("uint8")
            pair_disagreement_mask = np.zeros(patch_labels.shape, dtype="uint8")
        else:
            patch_labels = agreement_labels
            refined_training_mask = None
            refined_label_confidence = None
            pair_agreement_count = agreement_count
            pair_disagreement_mask = disagreement_mask
        height = int(profile["height"])
        width = int(profile["width"])
        pair_patch_count = 0

        for row, col in iter_patch_windows(height, width, patch_size, stride):
            row_slice = slice(row, row + patch_size)
            col_slice = slice(col, col + patch_size)
            patch_agreement = patch_labels[row_slice, col_slice].astype("uint8")
            patch_valid = valid_feature_mask[row_slice, col_slice]
            disagreement_patch = pair_disagreement_mask[row_slice, col_slice].astype("uint8")
            agreement_count_patch = pair_agreement_count[row_slice, col_slice].astype("uint8")
            agreement_fraction = float(np.count_nonzero(patch_agreement) / patch_agreement.size)
            valid_feature_fraction = float(np.count_nonzero(patch_valid) / patch_valid.size)
            if use_refined_labels:
                refined_mask_patch = refined_training_mask[row_slice, col_slice].astype("uint8")
                label_confidence = refined_label_confidence[row_slice, col_slice].astype("float32")
                label_confidence = np.where(label_confidence > 0, label_confidence, 0.0).astype("float32")
                training_mask = (
                    (patch_agreement > 0)
                    & (refined_mask_patch > 0)
                    & patch_valid
                    & (label_confidence >= float(settings.get("filters", {}).get("minimum_refined_label_confidence", 0.75)))
                ).astype("uint8")
            else:
                label_confidence = np.clip(agreement_count_patch.astype("float32") / 3.0, 0.0, 1.0)
                training_mask = (
                    (patch_agreement > 0)
                    & (disagreement_patch == 0)
                    & patch_valid
                    & (label_confidence >= 0.65)
                ).astype("uint8")
            if agreement_fraction < minimum_agreement_fraction:
                continue
            if valid_feature_fraction < minimum_valid_feature_fraction:
                continue
            if int(np.count_nonzero(training_mask)) < minimum_training_pixels:
                continue

            block_id = f"{tag}_r{row // patch_size:03d}_c{col // patch_size:03d}_{pair_patch_count:05d}"
            patch_path = output_dir / f"{block_id}.npz"
            feature_patch = features[:, row_slice, col_slice]
            uncertainty = 1.0 - label_confidence
            np.savez_compressed(
                patch_path,
                features=feature_patch,
                hard_labels=patch_agreement,
                agreement_labels=patch_agreement,
                label_confidence=label_confidence.astype("float32"),
                uncertainty=uncertainty.astype("float32"),
                training_mask=training_mask,
                agreement_count=agreement_count_patch,
                disagreement_mask=disagreement_patch,
            )

            unique, counts = np.unique(patch_agreement[patch_agreement > 0], return_counts=True)
            class_histogram = {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}
            for key, value in class_histogram.items():
                class_pixel_counts[key] = class_pixel_counts.get(key, 0) + value

            records.append(
                {
                    "patch_id": patch_path.stem,
                    "path": str(patch_path).replace("\\", "/"),
                    "training_tag": tag,
                    "year": int(pair["year"]),
                    "split": pair["split"],
                    "spatial_block_id": block_id,
                    "row": int(row),
                    "col": int(col),
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform(profile, row, col),
                    "feature_names": settings["feature_stack"],
                    "agreement_fraction": round(agreement_fraction, 4),
                    "valid_feature_fraction": round(valid_feature_fraction, 4),
                    "training_pixel_count": int(np.count_nonzero(training_mask)),
                    "disagreement_pixel_count": int(np.count_nonzero(disagreement_patch)),
                    "class_pixel_counts": class_histogram,
                }
            )
            split_counts[pair["split"]] = split_counts.get(pair["split"], 0) + 1
            pair_patch_count += 1
            if pair_patch_count >= max_per_pair:
                break

    manifest = {
        "phase": "phase_32_multi_year_training_patch_dataset",
        "source_summary": str(summary_path).replace("\\", "/"),
        "patch_size": patch_size,
        "stride": stride,
        "feature_stack": settings["feature_stack"],
        "label_sources": settings["label_sources"],
        "patch_count": len(records),
        "split_counts": split_counts,
        "pair_count": len(pairs),
        "total_training_pixel_count": int(sum(record["training_pixel_count"] for record in records)),
        "class_pixel_counts": class_pixel_counts,
        "notes": [
            "Splits inherit the temporal scene-catalog split: 2023/2024 train, 2025 validation, 2026 optional test.",
            "Weak labels come from aligned external agreement labels or strict date-specific refined pseudo-labels.",
            "Patch files are local artifacts under data/interim/training_patches and are not committed to Git.",
        ],
        "patches": records,
    }
    manifest_path = Path(settings["multi_year_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-year weak-label training patches.")
    parser.add_argument("--settings", type=Path, default=None, help="Optional patch settings YAML path.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest_path = generate_training_patch_dataset(load_config(), settings_path=args.settings, split=args.split, limit=args.limit)
    print(f"Training patch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
