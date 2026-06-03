"""Run full-AOI sliding-window U-Net inference over processed raster stacks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Torch must be imported before NumPy in this Windows Conda geospatial stack.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


FEATURE_FILE_SUFFIXES = {
    # Optical indices, radar features, and later engineered indices are read by
    # name so training/inference stay synchronized through config, not file order.
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
    # Kigali terrain is not flat; these features help explain shadow, moisture,
    # radar geometry, and settlement patterns in mountainous neighborhoods.
    "dem": "data/interim/topography/kigali_dem.tif",
    "slope": "data/interim/topography/kigali_slope.tif",
    "ruggedness": "data/interim/topography/kigali_ruggedness.tif",
    "tpi": "data/interim/topography/kigali_tpi.tif",
}


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


def training_pairs(summary_path: Path, split: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    pairs = summary.get("pairs", [])
    if split:
        pairs = [pair for pair in pairs if pair["split"] == split]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def read_feature_stack(processed_dir: Path, tag: str, features: list[str]) -> tuple["np.ndarray", dict[str, Any], "np.ndarray"]:
    import numpy as np
    import rasterio

    arrays = []
    reference_profile = None
    valid_mask = None
    for feature in features:
        if feature in TOPOGRAPHY_FEATURES:
            path = Path(TOPOGRAPHY_FEATURES[feature])
        else:
            path = processed_dir / f"{tag}_{FEATURE_FILE_SUFFIXES[feature]}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Required feature raster not found: {path}")
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            profile = src.profile.copy()
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
                raise ValueError(f"Feature raster is not aligned to the reference grid: {path}")
            valid_mask &= np.isfinite(data)
        arrays.append(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0))

    if reference_profile is None or valid_mask is None:
        raise RuntimeError("No feature rasters were read.")
    return np.stack(arrays, axis=0).astype("float32"), reference_profile, valid_mask


def iter_full_coverage_windows(height: int, width: int, tile_size: int, stride: int):
    """Yield windows that cover the full raster, including right/bottom edges."""
    rows = list(range(0, max(height - tile_size + 1, 1), stride))
    cols = list(range(0, max(width - tile_size + 1, 1), stride))
    bottom = max(height - tile_size, 0)
    right = max(width - tile_size, 0)
    if not rows or rows[-1] != bottom:
        rows.append(bottom)
    if not cols or cols[-1] != right:
        cols.append(right)
    for row in rows:
        for col in cols:
            yield row, col


def write_raster(path: Path, array: "np.ndarray", profile: dict[str, Any], dtype: str, nodata: float | int) -> None:
    import rasterio

    output_profile = profile.copy()
    output_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)


def run_pair_inference(
    pair: dict[str, Any],
    config: dict[str, Any],
    settings: dict[str, Any],
    model: Any,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    import numpy as np

    tag = pair["output_tag"]
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(settings["outputs"]["full_aoi_dir"])
    features = settings["training"].get("feature_stack") or [
        "s2_ndvi",
        "s2_ndwi",
        "s2_ndbi",
        "s1_vv",
        "s1_vh",
        "s1_vv_vh_ratio",
    ]
    stack, profile, valid_mask = read_feature_stack(processed_dir, tag, features)
    height = int(profile["height"])
    width = int(profile["width"])
    output_classes = int(settings["model"]["output_classes"])
    tile_size = int(settings["inference"]["tile_size"])
    stride = int(settings["inference"]["stride"])
    batch_size = int(settings["inference"]["batch_size"])

    probability_sum = np.zeros((output_classes, height, width), dtype="float32")
    vote_count = np.zeros((height, width), dtype="uint16")
    pending_tiles = []
    pending_windows: list[tuple[int, int]] = []

    def flush_batch() -> None:
        if not pending_tiles:
            return
        # Overlapping windows are averaged so edge pixels are not dominated by a
        # single tile prediction. This is the full-AOI inference mode the user
        # requested after patch-only coverage proved insufficient.
        batch = torch.from_numpy(np.stack(pending_tiles, axis=0)).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(batch), dim=1).detach().cpu().numpy().astype("float32")
        for index, (row, col) in enumerate(pending_windows):
            row_slice = slice(row, row + tile_size)
            col_slice = slice(col, col + tile_size)
            probability_sum[:, row_slice, col_slice] += probabilities[index]
            vote_count[row_slice, col_slice] += 1
        pending_tiles.clear()
        pending_windows.clear()

    for row, col in iter_full_coverage_windows(height, width, tile_size, stride):
        row_slice = slice(row, row + tile_size)
        col_slice = slice(col, col + tile_size)
        pending_tiles.append(stack[:, row_slice, col_slice])
        pending_windows.append((row, col))
        if len(pending_tiles) >= batch_size:
            flush_batch()
    flush_batch()

    covered = vote_count > 0
    averaged = np.zeros_like(probability_sum)
    averaged[:, covered] = probability_sum[:, covered] / vote_count[covered]
    monitored = averaged[1:, :, :]
    # Class 0 is reserved for nodata/ignore. User-facing confidence and entropy
    # are computed only over monitored classes 1-5.
    dominant_class = (np.argmax(monitored, axis=0) + 1).astype("uint8")
    max_probability = np.max(monitored, axis=0).astype("float32")
    clipped = np.clip(monitored, 1.0e-7, 1.0)
    entropy = (-np.sum(clipped * np.log(clipped), axis=0) / np.log(monitored.shape[0])).astype("float32")

    output_mask = covered & valid_mask
    dominant_output = np.where(output_mask, dominant_class, 0).astype("uint8")
    confidence_output = np.full((height, width), -9999.0, dtype="float32")
    entropy_output = np.full((height, width), -9999.0, dtype="float32")
    confidence_output[output_mask] = max_probability[output_mask]
    entropy_output[output_mask] = entropy[output_mask]

    dominant_path = output_dir / f"{tag}_unet_full_aoi_dominant_class.tif"
    confidence_path = output_dir / f"{tag}_unet_full_aoi_confidence.tif"
    entropy_path = output_dir / f"{tag}_unet_full_aoi_entropy.tif"
    coverage_path = output_dir / f"{tag}_unet_full_aoi_coverage.tif"
    write_raster(dominant_path, dominant_output, profile, "uint8", 0)
    write_raster(confidence_path, confidence_output, profile, "float32", -9999.0)
    write_raster(entropy_path, entropy_output, profile, "float32", -9999.0)
    write_raster(coverage_path, vote_count, profile, "uint16", 0)

    unique, counts = np.unique(dominant_output[output_mask], return_counts=True)
    class_counts = {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}
    return {
        "training_tag": tag,
        "year": int(pair["year"]),
        "split": pair["split"],
        "height": height,
        "width": width,
        "tile_size": tile_size,
        "stride": stride,
        "valid_pixel_count": int(np.count_nonzero(output_mask)),
        "coverage_fraction": round(float(np.count_nonzero(output_mask) / output_mask.size), 6),
        "dominant_class_counts": class_counts,
        "mean_confidence": round(float(np.mean(confidence_output[output_mask])), 6) if np.any(output_mask) else None,
        "mean_entropy": round(float(np.mean(entropy_output[output_mask])), 6) if np.any(output_mask) else None,
        "rasters": {
            "dominant_class": str(dominant_path).replace("\\", "/"),
            "confidence": str(confidence_path).replace("\\", "/"),
            "entropy": str(entropy_path).replace("\\", "/"),
            "coverage": str(coverage_path).replace("\\", "/"),
        },
    }


def run_full_aoi_inference(
    config: dict[str, Any],
    settings: dict[str, Any],
    split: str | None = None,
    limit: int | None = None,
) -> Path:
    try:
        import torch
        from ml.unet import build_segmentation_model
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError("PyTorch is required before running full-AOI U-Net inference.") from exc

    ensure_output_dirs(config)
    checkpoint_path = Path(settings["outputs"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {checkpoint_path}. Run `python pipelines/15_train_unet_baseline.py --fit` first."
        )
    summary_path = Path(config["paths"]["catalog"]) / "training_preprocessing_summary_2023_2026.json"
    pairs = training_pairs(summary_path, split=split, limit=limit)
    if not pairs:
        raise RuntimeError("No preprocessed training pairs matched the requested full-AOI inference filters.")

    configured_device = settings["training"]["device"]
    if configured_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configured_device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_segmentation_model(
        architecture=settings["model"].get("architecture", "lightweight_unet"),
        input_channels=int(settings["model"]["input_channels"]),
        output_classes=int(settings["model"]["output_classes"]),
        base_channels=int(settings["model"]["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(settings["outputs"]["full_aoi_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        run_pair_inference(pair, config=config, settings=settings, model=model, torch=torch, device=device)
        for pair in pairs
    ]
    manifest = {
        "phase": "phase_35_full_aoi_unet_inference",
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "requested_split": split or "all",
        "pair_count": len(records),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "outputs": {
            "full_aoi_dir": str(output_dir).replace("\\", "/"),
            "full_aoi_manifest": settings["outputs"]["full_aoi_manifest"],
        },
        "notes": [
            "This is wall-to-wall AOI inference over processed feature rasters, not sampled patch QA.",
            "Dominant class uses 0 as nodata and 1-5 as monitored model classes.",
            "Confidence and entropy use -9999 nodata outside valid feature pixels.",
        ],
        "records": records,
    }
    manifest_path = Path(settings["outputs"]["full_aoi_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-AOI U-Net inference over processed raster stacks.")
    parser.add_argument("--config", type=Path, default=Path("configs/model_unet.yaml"), help="Model config YAML path.")
    parser.add_argument("--split", choices=["train", "validation", "test"], help="Optional pair split to infer.")
    parser.add_argument("--limit", type=int, help="Optional pair limit for quick checks.")
    args = parser.parse_args()

    config = load_config()
    settings = load_yaml(args.config)
    manifest_path = run_full_aoi_inference(config, settings, split=args.split, limit=args.limit)
    print(f"Full-AOI U-Net inference manifest: {manifest_path}")


if __name__ == "__main__":
    main()
