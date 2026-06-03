"""Generate local bootstrap weak labels for future ML-ready land-cover mapping.

This stage does not claim field accuracy. It converts Sentinel-1/2 feature
evidence and existing confidence-scored change polygons into weak hard labels,
soft-label probability rasters, confidence, and uncertainty masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


CLASS_ORDER = [
    "built_up",
    "managed_vegetation",
    "natural_vegetation",
    "bare_soil",
    "water_wetland",
    "uncertain_mixed",
]

FEATURE_SUFFIXES = {
    "ndvi": "s2_ndvi",
    "ndwi": "s2_ndwi",
    "ndbi": "s2_ndbi",
    "vv_vh_ratio": "s1_vv_vh_ratio",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML settings file with a clear missing-file error."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Weak-label settings not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def monitoring_tags(processed_dir: Path) -> list[str]:
    """Return monitoring stack tags that have the required NDVI raster."""
    tags = {
        path.name.removesuffix("_s2_ndvi.tif")
        for path in processed_dir.glob("monitoring_*_s2_ndvi.tif")
    }
    return sorted(tags)


def read_feature(processed_dir: Path, tag: str, feature: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one feature raster from a monitoring stack."""
    path = processed_dir / f"{tag}_{FEATURE_SUFFIXES[feature]}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Required feature raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile.copy()


def normalized_positive(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Convert index values into a stable 0-1 evidence score."""
    score = (values - center) / scale
    return np.clip(score, 0.0, 1.0).astype("float32")


def sentinel_soft_probabilities(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndbi: np.ndarray,
    vv_vh_ratio: np.ndarray,
) -> np.ndarray:
    """Estimate harmonized class probabilities from Sentinel-only evidence."""
    valid = (
        np.isfinite(ndvi)
        & np.isfinite(ndwi)
        & np.isfinite(ndbi)
        & np.isfinite(vv_vh_ratio)
    )

    built = normalized_positive(ndbi, 0.08, 0.28) * (1.0 - normalized_positive(ndwi, 0.15, 0.4))
    green = normalized_positive(ndvi, 0.18, 0.5) * (1.0 - normalized_positive(ndwi, 0.25, 0.4))
    managed = green * 0.45
    natural = green * 0.55
    bare = (1.0 - normalized_positive(ndvi, 0.15, 0.45)) * (
        1.0 - normalized_positive(ndwi, 0.12, 0.4)
    ) * (1.0 - normalized_positive(ndbi, 0.18, 0.3))
    water = normalized_positive(ndwi, 0.12, 0.35)
    radar_mixed = normalized_positive(np.abs(vv_vh_ratio), 1.2, 3.5) * 0.2

    scores = np.stack(
        [
            built,
            managed,
            natural,
            bare,
            water,
            np.full(ndvi.shape, 0.12, dtype="float32") + radar_mixed,
        ],
        axis=0,
    ).astype("float32")
    scores[:, ~valid] = np.nan
    totals = np.nansum(scores, axis=0)
    probabilities = np.divide(
        scores,
        totals,
        out=np.zeros_like(scores, dtype="float32"),
        where=totals > 0,
    )
    probabilities[:, ~valid] = np.nan
    return probabilities


def probability_gap(probabilities: np.ndarray) -> np.ndarray:
    """Return the gap between the two most likely classes for each pixel."""
    sorted_probs = np.sort(np.nan_to_num(probabilities, nan=0.0), axis=0)
    return (sorted_probs[-1] - sorted_probs[-2]).astype("float32")


def dominant_class(probabilities: np.ndarray, class_ids: dict[str, int]) -> np.ndarray:
    """Convert class probabilities into 1-based class ids."""
    dominant_index = np.nanargmax(np.nan_to_num(probabilities, nan=-1.0), axis=0)
    labels = np.zeros(dominant_index.shape, dtype="uint8")
    for index, class_name in enumerate(CLASS_ORDER):
        labels[dominant_index == index] = int(class_ids[class_name])
    labels[~np.isfinite(probabilities).any(axis=0)] = 0
    return labels


def monitored_group_to_class_id(group: str, probabilities: np.ndarray, row: int, col: int, class_ids: dict[str, int]) -> int:
    """Map current monitoring groups into the harmonized class space."""
    if group == "built_up":
        return int(class_ids["built_up"])
    if group == "bare_sparse":
        return int(class_ids["bare_soil"])
    if group == "water_moisture":
        return int(class_ids["water_wetland"])
    if group == "vegetation":
        managed = probabilities[CLASS_ORDER.index("managed_vegetation"), row, col]
        natural = probabilities[CLASS_ORDER.index("natural_vegetation"), row, col]
        return int(class_ids["managed_vegetation"] if managed >= natural else class_ids["natural_vegetation"])
    return int(class_ids["uncertain_mixed"])


def rasterize_monitoring_agreement(
    changes_path: Path,
    profile: dict[str, Any],
    probabilities: np.ndarray,
    class_ids: dict[str, int],
) -> np.ndarray:
    """Rasterize current monitored groups into harmonized ids for agreement checks."""
    output = np.zeros((profile["height"], profile["width"]), dtype="uint8")
    if not changes_path.exists():
        return output

    changes = gpd.read_file(changes_path)
    if changes.empty or "monitored_land_cover" not in changes:
        return output

    changes = changes.to_crs(profile["crs"])
    for _, feature in changes.iterrows():
        point = feature.geometry.representative_point()
        col, row = ~profile["transform"] * (point.x, point.y)
        row_i = int(np.clip(row, 0, output.shape[0] - 1))
        col_i = int(np.clip(col, 0, output.shape[1] - 1))
        class_id = monitored_group_to_class_id(
            str(feature["monitored_land_cover"]),
            probabilities,
            row_i,
            col_i,
            class_ids,
        )
        mask = rasterize(
            [(feature.geometry, class_id)],
            out_shape=output.shape,
            transform=profile["transform"],
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
        output = np.where(mask > 0, mask, output).astype("uint8")
    return output


def combined_change_confidence(outputs_dir: Path, tag: str, shape: tuple[int, int]) -> np.ndarray:
    """Combine confidence rasters touching a monitoring stack."""
    combined = np.zeros(shape, dtype="float32")
    for pattern in (f"*__to__{tag}_confidence.tif", f"{tag}__to__*_confidence.tif"):
        for path in outputs_dir.glob(pattern):
            with rasterio.open(path) as src:
                data = src.read(1).astype("float32")
                if data.shape == shape:
                    combined = np.maximum(combined, np.nan_to_num(data, nan=0.0))
    return combined


def write_raster(path: Path, profile: dict[str, Any], data: np.ndarray, dtype: str, nodata: Any) -> None:
    """Write a single-band GeoTIFF aligned to the source monitoring stack."""
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data.astype(dtype), 1)


def generate_weak_labels(config: dict[str, Any], tag: str | None = None) -> tuple[Path, Path]:
    """Generate weak hard labels, soft labels, confidence, and uncertainty."""
    ensure_output_dirs(config)
    processed_dir = Path(config["paths"]["processed"])
    outputs_dir = Path(config["paths"]["outputs"])
    weak_settings = load_yaml(config["project"]["weak_labels_path"])
    class_ids = weak_settings["class_ids"]
    threshold_settings = weak_settings["thresholds"]

    tags = monitoring_tags(processed_dir)
    if not tags:
        raise FileNotFoundError("No monitoring stacks found for weak-label generation.")
    selected_tag = tag or tags[-1]
    if selected_tag not in tags:
        raise ValueError(f"Unknown monitoring tag {selected_tag!r}. Available: {tags}")

    ndvi, profile = read_feature(processed_dir, selected_tag, "ndvi")
    ndwi, _ = read_feature(processed_dir, selected_tag, "ndwi")
    ndbi, _ = read_feature(processed_dir, selected_tag, "ndbi")
    vv_vh_ratio, _ = read_feature(processed_dir, selected_tag, "vv_vh_ratio")

    probabilities = sentinel_soft_probabilities(ndvi, ndwi, ndbi, vv_vh_ratio)
    dominant = dominant_class(probabilities, class_ids)
    max_probability = np.nanmax(np.nan_to_num(probabilities, nan=0.0), axis=0).astype("float32")
    gap = probability_gap(probabilities)
    uncertainty = (1.0 - max_probability).astype("float32")
    monitoring_agreement = rasterize_monitoring_agreement(
        outputs_dir / "monitoring_change_scored.geojson",
        profile,
        probabilities,
        class_ids,
    )
    change_confidence = combined_change_confidence(outputs_dir, selected_tag, dominant.shape)
    agrees_with_monitoring = (monitoring_agreement == dominant) & (monitoring_agreement > 0)

    label_confidence = (
        0.55 * max_probability
        + 0.25 * agrees_with_monitoring.astype("float32")
        + 0.20 * np.clip(change_confidence, 0.0, 1.0)
    ).astype("float32")
    mixed = gap < float(threshold_settings["mixed_probability_gap"])
    hard_label = np.where(
        (max_probability >= float(threshold_settings["minimum_dominant_probability"]))
        & (label_confidence >= float(threshold_settings["minimum_label_confidence"]))
        & ~mixed,
        dominant,
        0,
    ).astype("uint8")

    prefix = str(weak_settings["outputs"]["prefix"])
    base = outputs_dir / f"{prefix}_{selected_tag}"
    hard_path = base.with_name(f"{base.name}_hard_labels.tif")
    confidence_path = base.with_name(f"{base.name}_label_confidence.tif")
    uncertainty_path = base.with_name(f"{base.name}_uncertainty.tif")
    agreement_path = base.with_name(f"{base.name}_monitoring_agreement.tif")

    write_raster(hard_path, profile, hard_label, "uint8", 0)
    write_raster(confidence_path, profile, label_confidence, "float32", np.nan)
    write_raster(uncertainty_path, profile, uncertainty, "float32", np.nan)
    write_raster(agreement_path, profile, monitoring_agreement, "uint8", 0)

    probability_paths = {}
    if weak_settings["outputs"].get("write_probability_rasters", True):
        for index, class_name in enumerate(CLASS_ORDER):
            probability_path = base.with_name(f"{base.name}_prob_{class_name}.tif")
            write_raster(probability_path, profile, probabilities[index], "float32", np.nan)
            probability_paths[class_name] = str(probability_path).replace("\\", "/")

    labeled_pixels = hard_label > 0
    class_counts = {
        class_name: int(np.count_nonzero(hard_label == int(class_ids[class_name])))
        for class_name in CLASS_ORDER
    }
    candidate_sources = weak_settings["candidate_sources"]
    active_sources = [
        name for name, details in candidate_sources.items() if details.get("enabled") is True
    ]
    registered_sources = list(candidate_sources)

    summary = {
        "phase": "phase_26_weak_label_generation",
        "label_type": "local_bootstrap_weak_labels",
        "monitoring_tag": selected_tag,
        "candidate_sources": candidate_sources,
        "active_sources": active_sources,
        "registered_sources": registered_sources,
        "external_sources_ingested": {
            "dynamic_world": False,
            "esa_worldcover": False,
            "osm_buildings_roads_landuse": False,
        },
        "source_note": (
            "Labels are generated from Sentinel index/radar evidence plus agreement with existing "
            "confidence-scored monitoring polygons. They are not field labels."
        ),
        "class_order": CLASS_ORDER,
        "class_ids": class_ids,
        "total_pixels": int(hard_label.size),
        "weak_labeled_pixels": int(np.count_nonzero(labeled_pixels)),
        "weak_labeled_area_km2": round(float(np.count_nonzero(labeled_pixels) * 100 / 1_000_000), 3),
        "class_pixel_counts": class_counts,
        "mean_label_confidence": round(float(np.nanmean(label_confidence[labeled_pixels])), 4) if np.any(labeled_pixels) else 0.0,
        "mean_uncertainty": round(float(np.nanmean(uncertainty[labeled_pixels])), 4) if np.any(labeled_pixels) else 0.0,
        "outputs": {
            "hard_labels": str(hard_path).replace("\\", "/"),
            "label_confidence": str(confidence_path).replace("\\", "/"),
            "uncertainty": str(uncertainty_path).replace("\\", "/"),
            "monitoring_agreement": str(agreement_path).replace("\\", "/"),
            "probabilities": probability_paths,
        },
    }
    summary_path = outputs_dir / f"{prefix}_{selected_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return hard_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local bootstrap weak labels.")
    parser.add_argument("--tag", default=None, help="Monitoring stack tag. Defaults to latest.")
    args = parser.parse_args()

    hard_labels, summary = generate_weak_labels(load_config(), tag=args.tag)
    print(f"Weak hard labels: {hard_labels}")
    print(f"Weak-label summary: {summary}")


if __name__ == "__main__":
    main()
