"""Detect land-cover change from multi-date Sentinel feature rasters."""

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


CHANGE_CLASSES = {
    1: "vegetation_loss",
    2: "built_up_gain",
    3: "moisture_change",
    4: "radar_change",
}


def load_master_grid(config: dict[str, Any]) -> dict[str, Any]:
    """Load the fixed grid that all change-detection features must share."""
    grid_path = Path(config["project"]["master_grid_path"])
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    transform = grid["transform"]
    bounds = grid["bounds"]
    return {
        "crs": grid["crs"],
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "transform": [float(transform[key]) for key in ("a", "b", "c", "d", "e", "f")],
        "bounds": [float(bounds[key]) for key in ("left", "bottom", "right", "top")],
        "tolerance": float(grid.get("alignment_policy", {}).get("tolerance", 0.001)),
    }


def close_enough(actual: list[float], expected: list[float], tolerance: float) -> bool:
    """Compare raster metadata while allowing tiny floating-point noise."""
    return len(actual) == len(expected) and all(
        abs(actual_value - expected_value) <= tolerance
        for actual_value, expected_value in zip(actual, expected)
    )


def validate_profile_against_master(path: Path, profile: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail before change detection if any feature is off the master grid."""
    transform = [float(value) for value in profile["transform"][:6]]
    bounds = [
        float(profile["transform"].c),
        float(profile["transform"].f + profile["height"] * profile["transform"].e),
        float(profile["transform"].c + profile["width"] * profile["transform"].a),
        float(profile["transform"].f),
    ]
    mismatches = []
    if str(profile["crs"]) != expected["crs"]:
        mismatches.append("crs")
    if int(profile["width"]) != expected["width"] or int(profile["height"]) != expected["height"]:
        mismatches.append("shape")
    if not close_enough(transform, expected["transform"], expected["tolerance"]):
        mismatches.append("transform")
    if not close_enough(bounds, expected["bounds"], expected["tolerance"]):
        mismatches.append("bounds")

    if mismatches:
        raise ValueError(
            "Change-detection feature raster is not aligned to configs/master_grid.yaml: "
            f"{path}; mismatched fields: {', '.join(mismatches)}"
        )


def mean_for_mask(data: np.ndarray, mask: np.ndarray) -> float:
    values = data[mask]
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def infer_landcover_state(ndvi: float, ndbi: float, ndwi: float) -> str:
    """Map simple spectral-index rules into clear monitored land-cover groups."""
    if not all(np.isfinite(value) for value in [ndvi, ndbi, ndwi]):
        return "unknown"
    if ndwi >= 0.2:
        return "water_moisture"
    if ndvi >= 0.45 and ndvi > ndbi:
        return "vegetation"
    if ndbi >= 0.12 and ndbi > ndvi:
        return "built_up"
    if ndvi <= 0.2 and ndbi <= 0.12:
        return "bare_sparse"
    return "mixed"


def change_magnitude(
    delta_ndvi: float,
    delta_ndbi: float,
    delta_ndwi: float,
    delta_radar_ratio: float,
) -> float:
    """Combine optical and radar deltas into one magnitude score."""
    values = np.array([delta_ndvi, delta_ndbi, delta_ndwi, delta_radar_ratio], dtype="float32")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def monitored_land_cover(change_type: str, before_state: str, after_state: str) -> str:
    """Assign each change to the land-cover group being monitored."""
    if change_type == "vegetation_loss":
        return "vegetation"
    if change_type == "built_up_gain":
        return "built_up"
    if change_type == "moisture_change":
        return "water_moisture"
    if before_state == after_state and before_state != "unknown":
        return before_state
    if before_state != "unknown":
        return before_state
    return after_state


def transition_label(before_state: str, after_state: str) -> str:
    return f"{before_state}_to_{after_state}"


def load_monitoring_pairs(path: str | Path) -> list[dict[str, Any]]:
    pairs_path = Path(path)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Monitoring pair catalog not found: {pairs_path}. "
            "Run pipelines/03_build_monitoring_stack.py --mode all first."
        )

    pairs = json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", [])
    return sorted(pairs, key=lambda pair: pair["optical"]["date"])


def raster_path(processed_dir: Path, tag: str, feature: str) -> Path:
    return processed_dir / f"{tag}_{feature}.tif"


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile.copy()


def finite_delta(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    delta = after - before
    delta[~np.isfinite(after) | ~np.isfinite(before)] = np.nan
    return delta.astype("float32")


def classify_change(
    delta_ndvi: np.ndarray,
    delta_ndbi: np.ndarray,
    delta_ndwi: np.ndarray,
    delta_radar_ratio: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Classify changed pixels using transparent threshold rules."""
    classes = np.zeros(delta_ndvi.shape, dtype="uint8")
    valid = (
        np.isfinite(delta_ndvi)
        & np.isfinite(delta_ndbi)
        & np.isfinite(delta_ndwi)
        & np.isfinite(delta_radar_ratio)
    )

    vegetation_loss = valid & (delta_ndvi <= thresholds["ndvi_loss"])
    # Built-up gain is intentionally nested inside vegetation loss here because
    # the monitored peri-urban signal is often vegetation being replaced by
    # brighter built/bare surfaces.
    built_up_gain = vegetation_loss & (delta_ndbi >= thresholds["ndbi_gain"])
    moisture_change = valid & (np.abs(delta_ndwi) >= thresholds["ndwi_abs"])
    radar_change = valid & (np.abs(delta_radar_ratio) >= thresholds["radar_ratio_abs"])

    classes[vegetation_loss] = 1
    classes[built_up_gain] = 2
    classes[moisture_change & (classes == 0)] = 3
    classes[radar_change & (classes == 0)] = 4
    return classes


def confidence_score(
    classes: np.ndarray,
    delta_ndvi: np.ndarray,
    delta_ndbi: np.ndarray,
    delta_ndwi: np.ndarray,
    delta_radar_ratio: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Score changed pixels by how strongly they exceed optical/radar thresholds."""
    confidence = np.zeros(classes.shape, dtype="float32")
    optical_strength = np.maximum(
        np.abs(delta_ndvi) / abs(thresholds["ndvi_loss"]),
        np.maximum(
            np.abs(delta_ndbi) / thresholds["ndbi_gain"],
            np.abs(delta_ndwi) / thresholds["ndwi_abs"],
        ),
    )
    radar_strength = np.abs(delta_radar_ratio) / thresholds["radar_ratio_abs"]
    combined = 0.65 * np.clip(optical_strength, 0, 1) + 0.35 * np.clip(radar_strength, 0, 1)
    confidence[classes > 0] = np.clip(combined[classes > 0], 0, 1)
    return confidence


def write_raster(path: Path, profile: dict[str, Any], data: np.ndarray, dtype: str, nodata: Any) -> None:
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data.astype(dtype), 1)


def polygonize_changes(
    class_raster: np.ndarray,
    confidence: np.ndarray,
    profile: dict[str, Any],
    before_pair: dict[str, Any],
    after_pair: dict[str, Any],
    min_area_m2: float,
    before_features: dict[str, np.ndarray],
    after_features: dict[str, np.ndarray],
    delta_features: dict[str, np.ndarray],
) -> gpd.GeoDataFrame:
    """Convert changed pixels into attributed polygons for API and WebGIS use."""
    features = []
    transform = profile["transform"]
    crs = profile["crs"]

    for geom, value in shapes(class_raster, mask=class_raster > 0, transform=transform):
        class_id = int(value)
        polygon = shape(geom)
        if polygon.is_empty:
            continue
        area_m2 = float(polygon.area)
        if area_m2 < min_area_m2:
            continue

        polygon_mask = geometry_mask(
            [geom],
            out_shape=class_raster.shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )
        class_pixels = polygon_mask & (class_raster == class_id)
        # Polygon attributes preserve before, after, and delta values so users
        # can see what existed before the change and what appears after it.
        mean_confidence = float(np.nanmean(confidence[class_pixels]))
        before_ndvi = mean_for_mask(before_features["ndvi"], class_pixels)
        after_ndvi = mean_for_mask(after_features["ndvi"], class_pixels)
        before_ndbi = mean_for_mask(before_features["ndbi"], class_pixels)
        after_ndbi = mean_for_mask(after_features["ndbi"], class_pixels)
        before_ndwi = mean_for_mask(before_features["ndwi"], class_pixels)
        after_ndwi = mean_for_mask(after_features["ndwi"], class_pixels)
        before_radar_ratio = mean_for_mask(before_features["radar_ratio"], class_pixels)
        after_radar_ratio = mean_for_mask(after_features["radar_ratio"], class_pixels)
        delta_ndvi = mean_for_mask(delta_features["ndvi"], class_pixels)
        delta_ndbi = mean_for_mask(delta_features["ndbi"], class_pixels)
        delta_ndwi = mean_for_mask(delta_features["ndwi"], class_pixels)
        delta_radar_ratio = mean_for_mask(delta_features["radar_ratio"], class_pixels)
        magnitude = change_magnitude(delta_ndvi, delta_ndbi, delta_ndwi, delta_radar_ratio)
        before_state = infer_landcover_state(before_ndvi, before_ndbi, before_ndwi)
        after_state = infer_landcover_state(after_ndvi, after_ndbi, after_ndwi)
        change_type = CHANGE_CLASSES[class_id]
        features.append(
            {
                "geometry": polygon,
                "change_type": change_type,
                "class_id": class_id,
                "confidence": round(mean_confidence, 3),
                "change_magnitude": round(magnitude, 4),
                "area_m2": round(area_m2, 2),
                "monitored_land_cover": monitored_land_cover(change_type, before_state, after_state),
                "transition": transition_label(before_state, after_state),
                "before_state": before_state,
                "after_state": after_state,
                "before_ndvi": round(before_ndvi, 4),
                "after_ndvi": round(after_ndvi, 4),
                "delta_ndvi": round(delta_ndvi, 4),
                "before_ndbi": round(before_ndbi, 4),
                "after_ndbi": round(after_ndbi, 4),
                "delta_ndbi": round(delta_ndbi, 4),
                "before_ndwi": round(before_ndwi, 4),
                "after_ndwi": round(after_ndwi, 4),
                "delta_ndwi": round(delta_ndwi, 4),
                "before_vv_vh": round(before_radar_ratio, 4),
                "after_vv_vh": round(after_radar_ratio, 4),
                "delta_vv_vh": round(delta_radar_ratio, 4),
                "before_tag": before_pair["output_tag"],
                "after_tag": after_pair["output_tag"],
                "before_date": before_pair["optical"]["date"],
                "after_date": after_pair["optical"]["date"],
                "radar_before_date": before_pair["radar"]["date"],
                "radar_after_date": after_pair["radar"]["date"],
            }
        )

    if not features:
        return gpd.GeoDataFrame(
            columns=[
                "change_type",
                "class_id",
                "confidence",
                "change_magnitude",
                "area_m2",
                "monitored_land_cover",
                "transition",
                "before_state",
                "after_state",
                "before_ndvi",
                "after_ndvi",
                "delta_ndvi",
                "before_ndbi",
                "after_ndbi",
                "delta_ndbi",
                "before_ndwi",
                "after_ndwi",
                "delta_ndwi",
                "before_vv_vh",
                "after_vv_vh",
                "delta_vv_vh",
                "before_tag",
                "after_tag",
                "before_date",
                "after_date",
                "radar_before_date",
                "radar_after_date",
                "geometry",
            ],
            geometry="geometry",
            crs=crs,
        )

    return gpd.GeoDataFrame(features, geometry="geometry", crs=crs)


def detect_pair_change(
    config: dict[str, Any],
    before_pair: dict[str, Any],
    after_pair: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Detect change between two already-aligned monitoring stacks."""
    processed_dir = Path(config["paths"]["processed"])
    outputs_dir = Path(config["paths"]["outputs"])
    settings = config["change_detection"]
    thresholds = settings["delta_thresholds"]
    comparison_tag = f"{before_pair['output_tag']}__to__{after_pair['output_tag']}"

    expected_grid = load_master_grid(config)
    feature_paths = {
        "before_ndvi": raster_path(processed_dir, before_pair["output_tag"], "s2_ndvi"),
        "after_ndvi": raster_path(processed_dir, after_pair["output_tag"], "s2_ndvi"),
        "before_ndbi": raster_path(processed_dir, before_pair["output_tag"], "s2_ndbi"),
        "after_ndbi": raster_path(processed_dir, after_pair["output_tag"], "s2_ndbi"),
        "before_ndwi": raster_path(processed_dir, before_pair["output_tag"], "s2_ndwi"),
        "after_ndwi": raster_path(processed_dir, after_pair["output_tag"], "s2_ndwi"),
        "before_radar": raster_path(processed_dir, before_pair["output_tag"], "s1_vv_vh_ratio"),
        "after_radar": raster_path(processed_dir, after_pair["output_tag"], "s1_vv_vh_ratio"),
    }
    arrays: dict[str, np.ndarray] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for name, path in feature_paths.items():
        array, raster_profile = read_raster(path)
        validate_profile_against_master(path, raster_profile, expected_grid)
        arrays[name] = array
        profiles[name] = raster_profile

    profile = profiles["before_ndvi"]
    before_ndvi = arrays["before_ndvi"]
    after_ndvi = arrays["after_ndvi"]
    before_ndbi = arrays["before_ndbi"]
    after_ndbi = arrays["after_ndbi"]
    before_ndwi = arrays["before_ndwi"]
    after_ndwi = arrays["after_ndwi"]
    before_radar = arrays["before_radar"]
    after_radar = arrays["after_radar"]

    delta_ndvi = finite_delta(after_ndvi, before_ndvi)
    delta_ndbi = finite_delta(after_ndbi, before_ndbi)
    delta_ndwi = finite_delta(after_ndwi, before_ndwi)
    delta_radar = finite_delta(after_radar, before_radar)
    class_raster = classify_change(delta_ndvi, delta_ndbi, delta_ndwi, delta_radar, thresholds)
    confidence = confidence_score(class_raster, delta_ndvi, delta_ndbi, delta_ndwi, delta_radar, thresholds)
    class_raster = np.where(
        confidence >= float(settings["confidence_threshold"]), class_raster, 0
    ).astype("uint8")
    confidence = np.where(class_raster > 0, confidence, 0).astype("float32")

    prefix = outputs_dir / comparison_tag
    write_raster(prefix.with_name(f"{comparison_tag}_delta_ndvi.tif"), profile, delta_ndvi, "float32", np.nan)
    write_raster(prefix.with_name(f"{comparison_tag}_delta_ndbi.tif"), profile, delta_ndbi, "float32", np.nan)
    write_raster(prefix.with_name(f"{comparison_tag}_delta_ndwi.tif"), profile, delta_ndwi, "float32", np.nan)
    write_raster(
        prefix.with_name(f"{comparison_tag}_delta_vv_vh_ratio.tif"),
        profile,
        delta_radar,
        "float32",
        np.nan,
    )
    write_raster(prefix.with_name(f"{comparison_tag}_change_class.tif"), profile, class_raster, "uint8", 0)
    write_raster(prefix.with_name(f"{comparison_tag}_confidence.tif"), profile, confidence, "float32", 0)

    polygons = polygonize_changes(
        class_raster,
        confidence,
        profile,
        before_pair,
        after_pair,
        float(settings["min_polygon_area_m2"]),
        before_features={
            "ndvi": before_ndvi,
            "ndbi": before_ndbi,
            "ndwi": before_ndwi,
            "radar_ratio": before_radar,
        },
        after_features={
            "ndvi": after_ndvi,
            "ndbi": after_ndbi,
            "ndwi": after_ndwi,
            "radar_ratio": after_radar,
        },
        delta_features={
            "ndvi": delta_ndvi,
            "ndbi": delta_ndbi,
            "ndwi": delta_ndwi,
            "radar_ratio": delta_radar,
        },
    )

    summary = {
        "comparison_tag": comparison_tag,
        "before_date": before_pair["optical"]["date"],
        "after_date": after_pair["optical"]["date"],
        "change_pixel_count": int(np.count_nonzero(class_raster)),
        "polygon_count": int(len(polygons)),
        "mean_change_magnitude": (
            round(float(polygons["change_magnitude"].mean()), 4) if not polygons.empty else 0
        ),
        "class_counts": {
            CHANGE_CLASSES[class_id]: int(np.count_nonzero(class_raster == class_id))
            for class_id in CHANGE_CLASSES
        },
    }
    return polygons, summary


def detect_changes(config: dict[str, Any]) -> tuple[Path, Path]:
    """Run pairwise change detection across the monitoring date sequence."""
    ensure_output_dirs(config)
    catalog_path = Path(config["paths"]["catalog"]) / "monitoring_pairs.json"
    outputs_dir = Path(config["paths"]["outputs"])
    pairs = load_monitoring_pairs(catalog_path)
    if len(pairs) < 2:
        raise ValueError("At least two monitoring pairs are needed for change detection.")

    all_polygons = []
    summaries = []
    for before_pair, after_pair in zip(pairs, pairs[1:]):
        print(
            f"Detecting change {before_pair['optical']['date']} -> {after_pair['optical']['date']}...",
            flush=True,
        )
        polygons, summary = detect_pair_change(config, before_pair, after_pair)
        all_polygons.append(polygons)
        summaries.append(summary)

    if all_polygons and any(not gdf.empty for gdf in all_polygons):
        combined = gpd.GeoDataFrame(
            pd.concat([gdf for gdf in all_polygons if not gdf.empty], ignore_index=True),
            geometry="geometry",
            crs=all_polygons[0].crs,
        )
    else:
        combined = gpd.GeoDataFrame(geometry=[], crs=config["project"]["local_crs"])

    polygons_path = outputs_dir / f"{config['change_detection']['output_prefix']}_polygons.geojson"
    summary_path = outputs_dir / f"{config['change_detection']['output_prefix']}_summary.json"
    combined.to_crs("EPSG:4326").to_file(polygons_path, driver="GeoJSON")
    summary_path.write_text(json.dumps({"comparisons": summaries}, indent=2), encoding="utf-8")
    return polygons_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect candidate land-cover changes.")
    parser.parse_args()

    polygons_path, summary_path = detect_changes(load_config())
    print(f"Change polygons: {polygons_path}")
    print(f"Change summary: {summary_path}")


if __name__ == "__main__":
    main()
