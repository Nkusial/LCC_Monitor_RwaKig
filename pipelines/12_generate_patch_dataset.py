"""Generate ML-ready 128 x 128 patches from aligned features and weak labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


FEATURE_FILE_SUFFIXES = {
    "s2_ndvi": "s2_ndvi",
    "s2_ndwi": "s2_ndwi",
    "s2_ndbi": "s2_ndbi",
    "s1_vv": "s1_vv",
    "s1_vh": "s1_vh",
    "s1_vv_vh_ratio": "s1_vv_vh_ratio",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Patch dataset config not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def monitoring_tags(processed_dir: Path) -> list[str]:
    tags = {
        path.name.removesuffix("_s2_ndvi.tif")
        for path in processed_dir.glob("monitoring_*_s2_ndvi.tif")
    }
    return sorted(tags)


def selected_tag(processed_dir: Path, tag: str | None) -> str:
    tags = monitoring_tags(processed_dir)
    if not tags:
        raise FileNotFoundError("No monitoring stacks available for patch generation.")
    if tag is None:
        return tags[-1]
    if tag not in tags:
        raise ValueError(f"Unknown monitoring tag {tag!r}. Available: {tags}")
    return tag


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
                raise ValueError(f"Feature raster is not on the master grid: {path}")
            valid_mask &= np.isfinite(data)
        arrays.append(data)

    if reference_profile is None or valid_mask is None:
        raise ValueError("No features configured for patch generation.")
    return np.stack(arrays, axis=0).astype("float32"), reference_profile, valid_mask


def format_label_path(template: str, tag: str) -> Path:
    return Path(template.format(tag=tag))


def split_for_block(block_id: str, split_policy: dict[str, Any]) -> str:
    modulo = int(block_id.rsplit("_", 1)[-1]) % 10
    if modulo in split_policy["train_modulo_values"]:
        return "train"
    if modulo in split_policy["validation_modulo_values"]:
        return "validation"
    return "test"


def iter_patch_windows(height: int, width: int, patch_size: int, stride: int):
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            yield row, col


def patch_transform(profile: dict[str, Any], row: int, col: int) -> list[float]:
    transform = profile["transform"] * rasterio.Affine.translation(col, row)
    return [round(value, 10) for value in transform[:6]]


def generate_patch_dataset(config: dict[str, Any], tag: str | None = None) -> Path:
    ensure_output_dirs(config)
    settings = load_yaml(config["project"]["patch_dataset_path"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_tag(processed_dir, tag)

    features, profile, valid_feature_mask = read_feature_stack(
        processed_dir,
        selected,
        settings["feature_stack"],
    )
    hard_labels, _ = read_raster(format_label_path(settings["label_sources"]["hard_labels"], selected))
    label_confidence, _ = read_raster(format_label_path(settings["label_sources"]["label_confidence"], selected))
    uncertainty, _ = read_raster(format_label_path(settings["label_sources"]["uncertainty"], selected))
    agreement_labels, _ = read_raster(Path(settings["label_sources"]["agreement_labels"]))
    agreement_count, _ = read_raster(Path(settings["label_sources"]["agreement_count"]))
    disagreement_mask, _ = read_raster(Path(settings["label_sources"]["disagreement_mask"]))

    patch_size = int(settings["patch_size"])
    stride = int(settings["stride"])
    max_patches = int(settings["max_patches"])
    minimum_agreement_fraction = float(settings["minimum_agreement_fraction"])
    minimum_valid_feature_fraction = float(settings["minimum_valid_feature_fraction"])
    minimum_training_pixels = int(settings["minimum_training_pixels"])

    records = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    class_pixel_counts: dict[str, int] = {}
    patch_index = 0
    height = int(profile["height"])
    width = int(profile["width"])

    for row, col in iter_patch_windows(height, width, patch_size, stride):
        row_slice = slice(row, row + patch_size)
        col_slice = slice(col, col + patch_size)
        patch_valid = valid_feature_mask[row_slice, col_slice]
        patch_agreement = agreement_labels[row_slice, col_slice]
        agreement_fraction = float(np.count_nonzero(patch_agreement) / patch_agreement.size)
        valid_feature_fraction = float(np.count_nonzero(patch_valid) / patch_valid.size)
        confidence_patch = label_confidence[row_slice, col_slice].astype("float32")
        disagreement_patch = disagreement_mask[row_slice, col_slice].astype("uint8")
        training_mask = (
            (patch_agreement > 0)
            & (disagreement_patch == 0)
            & patch_valid
            & np.isfinite(confidence_patch)
            & (confidence_patch >= 0.65)
        ).astype("uint8")
        if agreement_fraction < minimum_agreement_fraction:
            continue
        if valid_feature_fraction < minimum_valid_feature_fraction:
            continue
        if int(np.count_nonzero(training_mask)) < minimum_training_pixels:
            continue

        block_id = f"r{row // patch_size:03d}_c{col // patch_size:03d}_{patch_index:05d}"
        split = split_for_block(block_id, settings["split_policy"])
        patch_path = output_dir / f"{selected}_{block_id}.npz"
        feature_patch = features[:, row_slice, col_slice]
        hard_patch = hard_labels[row_slice, col_slice].astype("uint8")
        agreement_patch = patch_agreement.astype("uint8")
        uncertainty_patch = uncertainty[row_slice, col_slice].astype("float32")

        np.savez_compressed(
            patch_path,
            features=feature_patch,
            hard_labels=hard_patch,
            agreement_labels=agreement_patch,
            label_confidence=confidence_patch,
            uncertainty=uncertainty_patch,
            training_mask=training_mask,
            agreement_count=agreement_count[row_slice, col_slice].astype("uint8"),
            disagreement_mask=disagreement_patch,
        )

        unique, counts = np.unique(agreement_patch[agreement_patch > 0], return_counts=True)
        class_histogram = {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}
        for key, value in class_histogram.items():
            class_pixel_counts[key] = class_pixel_counts.get(key, 0) + value

        records.append(
            {
                "patch_id": patch_path.stem,
                "path": str(patch_path).replace("\\", "/"),
                "monitoring_tag": selected,
                "split": split,
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
        split_counts[split] += 1
        patch_index += 1
        if patch_index >= max_patches:
            break

    manifest = {
        "phase": "phase_27_patch_dataset",
        "monitoring_tag": selected,
        "patch_size": patch_size,
        "stride": stride,
        "feature_stack": settings["feature_stack"],
        "label_sources": settings["label_sources"],
        "split_policy": settings["split_policy"],
        "patch_count": len(records),
        "split_counts": split_counts,
        "total_training_pixel_count": int(sum(record["training_pixel_count"] for record in records)),
        "class_pixel_counts": class_pixel_counts,
        "notes": [
            "Splits are spatial-block based, not random pixel based.",
            "Training masks exclude disagreement pixels and low-confidence pixels.",
            "Patch files are local artifacts under data/interim and are not committed to Git.",
        ],
        "patches": records,
    }
    manifest_path = output_dir / "patch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML-ready weak-label patch dataset.")
    parser.add_argument("--tag", default=None, help="Monitoring tag to use. Defaults to latest stack.")
    args = parser.parse_args()
    manifest_path = generate_patch_dataset(load_config(), tag=args.tag)
    print(f"Patch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
