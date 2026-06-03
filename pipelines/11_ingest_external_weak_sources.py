"""Ingest external weak-label candidate sources onto the AOI master grid.

Phase 26B prepares independent evidence layers before patch generation. ESA
WorldCover and OSM are directly ingested. Dynamic World is supported through a
local Earth Engine export path because the public Dynamic World service is an
Earth Engine ImageCollection, not an unauthenticated COG/STAC endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import planetary_computer
import rasterio
import requests
import yaml
from affine import Affine
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from shapely.geometry import LineString, Polygon, shape

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


HARMONIZED_CLASS_IDS = {
    # External products do not share a taxonomy. This compact schema is the
    # common language used before creating agreement masks or pseudo-labels.
    "nodata_or_unlabeled": 0,
    "built_up": 1,
    "managed_vegetation": 2,
    "natural_vegetation": 3,
    "bare_soil": 4,
    "water_wetland": 5,
    "uncertain_mixed": 6,
}

ESA_TO_HARMONIZED = {
    10: HARMONIZED_CLASS_IDS["natural_vegetation"],  # tree cover
    20: HARMONIZED_CLASS_IDS["natural_vegetation"],  # shrubland
    30: HARMONIZED_CLASS_IDS["natural_vegetation"],  # grassland
    40: HARMONIZED_CLASS_IDS["managed_vegetation"],  # cropland
    50: HARMONIZED_CLASS_IDS["built_up"],
    60: HARMONIZED_CLASS_IDS["bare_soil"],
    70: HARMONIZED_CLASS_IDS["uncertain_mixed"],  # snow/ice, not expected here
    80: HARMONIZED_CLASS_IDS["water_wetland"],
    90: HARMONIZED_CLASS_IDS["water_wetland"],
    95: HARMONIZED_CLASS_IDS["water_wetland"],
    100: HARMONIZED_CLASS_IDS["uncertain_mixed"],
}

DYNAMIC_WORLD_TO_HARMONIZED = {
    0: HARMONIZED_CLASS_IDS["water_wetland"],
    1: HARMONIZED_CLASS_IDS["natural_vegetation"],
    2: HARMONIZED_CLASS_IDS["natural_vegetation"],
    3: HARMONIZED_CLASS_IDS["water_wetland"],
    4: HARMONIZED_CLASS_IDS["managed_vegetation"],
    5: HARMONIZED_CLASS_IDS["natural_vegetation"],
    6: HARMONIZED_CLASS_IDS["built_up"],
    7: HARMONIZED_CLASS_IDS["bare_soil"],
    8: HARMONIZED_CLASS_IDS["uncertain_mixed"],
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def latest_monitoring_tag(processed_dir: Path) -> str:
    tags = sorted(
        path.name.removesuffix("_s2_ndvi.tif")
        for path in processed_dir.glob("monitoring_*_s2_ndvi.tif")
    )
    if not tags:
        raise FileNotFoundError("No monitoring NDVI rasters found for reference grid.")
    return tags[-1]


def reference_profile(config: dict[str, Any], tag: str | None) -> tuple[str, dict[str, Any]]:
    processed_dir = Path(config["paths"]["processed"])
    selected_tag = tag or latest_monitoring_tag(processed_dir)
    ndvi_path = processed_dir / f"{selected_tag}_s2_ndvi.tif"
    with rasterio.open(ndvi_path) as src:
        return selected_tag, src.profile.copy()


def master_grid_profile(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a rasterio profile from the fixed AOI master-grid contract."""
    grid_path = Path(config["project"]["master_grid_path"])
    grid = load_yaml(grid_path)
    transform = Affine(
        float(grid["transform"]["a"]),
        float(grid["transform"]["b"]),
        float(grid["transform"]["c"]),
        float(grid["transform"]["d"]),
        float(grid["transform"]["e"]),
        float(grid["transform"]["f"]),
    )
    profile = {
        "driver": "GTiff",
        "crs": rasterio.crs.CRS.from_string(grid["crs"]),
        "transform": transform,
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
    }
    return "master_grid", profile


def select_reference_profile(
    config: dict[str, Any],
    tag: str | None,
    reference_grid: str,
) -> tuple[str, dict[str, Any], str]:
    if reference_grid == "master":
        selected_tag, profile = master_grid_profile(config)
        return selected_tag, profile, "configs/master_grid.yaml"
    selected_tag, profile = reference_profile(config, tag)
    return selected_tag, profile, f"data/processed/{selected_tag}_s2_ndvi.tif"


def aoi_geometry(config: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    aoi_path = Path(config["project"]["aoi_path"])
    data = json.loads(aoi_path.read_text(encoding="utf-8"))
    geom = data["features"][0]["geometry"]
    bounds = shape(geom).bounds
    return geom, [float(value) for value in bounds]


def aoi_mask_for_profile(geom: dict[str, Any], profile: dict[str, Any]) -> np.ndarray:
    """Project the AOI into the raster grid CRS and return valid AOI pixels."""
    projected = gpd.GeoSeries([shape(geom)], crs="EPSG:4326").to_crs(profile["crs"]).iloc[0]
    return geometry_mask(
        [projected.__geo_interface__],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        invert=True,
    )


def apply_aoi_mask(data: np.ndarray, valid_aoi: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the AOI after source reprojection to the fixed grid."""
    return np.where(valid_aoi, data, 0).astype("uint8")


def raster_signature(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        bounds = src.bounds
        return {
            "path": str(path).replace("\\", "/"),
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "bounds": [round(value, 3) for value in [bounds.left, bounds.bottom, bounds.right, bounds.top]],
            "transform": [round(value, 6) for value in src.transform[:6]],
            "nodata": src.nodata,
        }


def profile_signature(profile: dict[str, Any]) -> dict[str, Any]:
    bounds = array_bounds(profile["height"], profile["width"], profile["transform"])
    return {
        "crs": str(profile["crs"]),
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "bounds": [round(value, 3) for value in bounds],
        "transform": [round(value, 6) for value in profile["transform"][:6]],
    }


def alignment_report(
    profile: dict[str, Any],
    output_paths: list[Path],
    valid_aoi: np.ndarray,
) -> dict[str, Any]:
    """Confirm all generated evidence layers share the same grid and AOI mask."""
    expected = profile_signature(profile)
    rasters = []
    mismatches: dict[str, list[str]] = {}
    outside_aoi_pixels: dict[str, int] = {}
    for path in output_paths:
        if not path.exists():
            continue
        signature = raster_signature(path)
        rasters.append(signature)
        fields = [
            field for field in ["crs", "width", "height", "bounds", "transform"]
            if signature[field] != expected[field]
        ]
        if fields:
            mismatches[path.name] = fields
        with rasterio.open(path) as src:
            data = src.read(1)
        outside_aoi_pixels[path.name] = int(np.count_nonzero((data > 0) & ~valid_aoi))
    return {
        "status": "pass" if not mismatches and all(value == 0 for value in outside_aoi_pixels.values()) else "fail",
        "reference_grid": expected,
        "rasters": rasters,
        "mismatches": mismatches,
        "outside_aoi_nonzero_pixels": outside_aoi_pixels,
        "aoi_valid_pixels": int(np.count_nonzero(valid_aoi)),
    }


def write_raster(path: Path, profile: dict[str, Any], data: np.ndarray, dtype: str = "uint8") -> None:
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", count=1, dtype=dtype, nodata=0, compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data.astype(dtype), 1)


def suffixed_path(output_dir: Path, stem: str, suffix: str, extension: str = ".tif") -> Path:
    """Keep benchmark/test weak-source rasters separate from training/validation outputs."""
    clean_suffix = suffix.strip("_")
    if clean_suffix:
        return output_dir / f"{stem}_{clean_suffix}{extension}"
    return output_dir / f"{stem}{extension}"


def read_optional_raster(path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Read a class raster only when it exists and matches the master grid."""
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype("uint8")
    if data.shape != expected_shape:
        return None
    return data


def remap_classes(data: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    output = np.zeros(data.shape, dtype="uint8")
    for source_id, target_id in mapping.items():
        output[data == source_id] = target_id
    return output


def build_agreement_outputs(
    profile: dict[str, Any],
    output_dir: Path,
    data_outputs_dir: Path,
    tag: str,
    suffix: str = "",
    include_local_bootstrap: bool = True,
) -> dict[str, Any]:
    """Create agreement/disagreement rasters from local and external sources."""
    expected_shape = (profile["height"], profile["width"])
    # Each candidate source is optional, but any source that exists must already
    # be on the selected master/reference grid before it can vote.
    candidate_paths = {
        "dynamic_world": suffixed_path(output_dir, "dynamic_world_harmonized", suffix),
        "esa_worldcover": suffixed_path(output_dir, "esa_worldcover_harmonized", suffix),
        "osm_buildings_roads_landuse": suffixed_path(output_dir, "osm_buildings_roads_landuse_harmonized", suffix),
    }
    if include_local_bootstrap:
        candidate_paths["local_bootstrap_weak_labels"] = data_outputs_dir / f"weak_labels_{tag}_hard_labels.tif"
    source_arrays = {
        name: data
        for name, path in candidate_paths.items()
        if (data := read_optional_raster(path, expected_shape)) is not None
    }
    if not source_arrays:
        return {"status": "skipped", "reason": "No aligned source rasters available."}

    stack = np.stack(list(source_arrays.values()), axis=0)
    source_count = np.count_nonzero(stack > 0, axis=0).astype("uint8")
    class_votes = np.zeros((len(HARMONIZED_CLASS_IDS) - 1, *expected_shape), dtype="uint8")
    for class_id in range(1, len(HARMONIZED_CLASS_IDS)):
        class_votes[class_id - 1] = np.count_nonzero(stack == class_id, axis=0)

    mode_count = np.max(class_votes, axis=0).astype("uint8")
    mode_label = (np.argmax(class_votes, axis=0) + 1).astype("uint8")
    # Agreement labels are intentionally conservative: at least two sources
    # must support the same harmonized class before it becomes training evidence.
    agreement_label = np.where(mode_count >= 2, mode_label, 0).astype("uint8")
    disagreement_mask = ((source_count >= 2) & (mode_count < source_count)).astype("uint8")

    agreement_path = suffixed_path(output_dir, "weak_source_agreement_labels", suffix)
    count_path = suffixed_path(output_dir, "weak_source_agreement_count", suffix)
    disagreement_path = suffixed_path(output_dir, "weak_source_disagreement_mask", suffix)
    write_raster(agreement_path, profile, agreement_label)
    write_raster(count_path, profile, source_count)
    write_raster(disagreement_path, profile, disagreement_mask)

    return {
        "status": "created",
        "sources_used": list(source_arrays),
        "agreement_labels": str(agreement_path).replace("\\", "/"),
        "source_count": str(count_path).replace("\\", "/"),
        "disagreement_mask": str(disagreement_path).replace("\\", "/"),
        "agreement_pixels": int(np.count_nonzero(agreement_label)),
        "disagreement_pixels": int(np.count_nonzero(disagreement_mask)),
    }


def ingest_esa_worldcover(
    settings: dict[str, Any],
    bbox: list[float],
    profile: dict[str, Any],
    output_dir: Path,
    valid_aoi: np.ndarray,
    suffix: str = "",
) -> dict[str, Any]:
    source = settings["candidate_sources"]["esa_worldcover"]
    collection = source["stac_collection"]
    year = str(source["preferred_year"])
    version = str(source["preferred_version"])
    asset_key = source["asset_key"]
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items = list(client.search(collections=[collection], bbox=bbox, limit=20).items())
    selected = [
        item for item in items if year in item.id and version in item.id
    ] or items
    merged = np.zeros((profile["height"], profile["width"]), dtype="uint8")
    signed_ids = []

    for item in selected:
        signed = planetary_computer.sign(item)
        href = signed.assets[asset_key].href
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=profile["crs"],
                transform=profile["transform"],
                width=profile["width"],
                height=profile["height"],
                resampling=Resampling.nearest,
                src_nodata=0,
                nodata=0,
            ) as vrt:
                data = vrt.read(1)
        mapped = remap_classes(data, ESA_TO_HARMONIZED)
        merged = np.where((merged == 0) & (mapped > 0), mapped, merged).astype("uint8")
        signed_ids.append(item.id)

    merged = apply_aoi_mask(merged, valid_aoi)
    output_path = suffixed_path(output_dir, "esa_worldcover_harmonized", suffix)
    write_raster(output_path, profile, merged)
    return {
        "source": "esa_worldcover",
        "status": "ingested" if signed_ids else "not_found",
        "collection": collection,
        "items": signed_ids,
        "output": str(output_path).replace("\\", "/"),
        "nonzero_pixels": int(np.count_nonzero(merged)),
    }


def dynamic_world_status(
    settings: dict[str, Any],
    profile: dict[str, Any],
    output_dir: Path,
    valid_aoi: np.ndarray,
    local_label_raster: str | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    source = settings["candidate_sources"]["dynamic_world"]
    local_path = Path(local_label_raster or source["local_label_raster"])
    output_path = suffixed_path(output_dir, "dynamic_world_harmonized", suffix)
    if not local_path.exists():
        return {
            "source": "dynamic_world",
            "status": "not_ingested_requires_earth_engine_export",
            "earth_engine_asset": source["earth_engine_asset"],
            "expected_local_label_raster": str(local_path).replace("\\", "/"),
            "note": "Export the Dynamic World label band from Earth Engine for the AOI/date, then rerun this stage.",
        }

    with rasterio.open(local_path) as src:
        with WarpedVRT(
            src,
            crs=profile["crs"],
            transform=profile["transform"],
            width=profile["width"],
            height=profile["height"],
            resampling=Resampling.nearest,
            src_nodata=255,
            nodata=255,
        ) as vrt:
            data = vrt.read(1)
    mapped = apply_aoi_mask(remap_classes(data, DYNAMIC_WORLD_TO_HARMONIZED), valid_aoi)
    write_raster(output_path, profile, mapped)
    return {
        "source": "dynamic_world",
        "status": "ingested_from_local_export",
        "earth_engine_asset": source["earth_engine_asset"],
        "input": str(local_path).replace("\\", "/"),
        "output": str(output_path).replace("\\", "/"),
        "nonzero_pixels": int(np.count_nonzero(mapped)),
    }


def osm_class_id(tags: dict[str, Any]) -> int | None:
    if "building" in tags or "highway" in tags:
        return HARMONIZED_CLASS_IDS["built_up"]
    landuse = str(tags.get("landuse", "")).lower()
    if landuse in {"farmland", "orchard", "plant_nursery", "vineyard", "allotments"}:
        return HARMONIZED_CLASS_IDS["managed_vegetation"]
    if landuse in {"forest", "grass", "meadow", "recreation_ground", "village_green"}:
        return HARMONIZED_CLASS_IDS["natural_vegetation"]
    if landuse in {"basin", "reservoir"}:
        return HARMONIZED_CLASS_IDS["water_wetland"]
    if landuse in {"quarry", "construction", "brownfield", "landfill"}:
        return HARMONIZED_CLASS_IDS["bare_soil"]
    if landuse in {"residential", "commercial", "industrial", "retail"}:
        return HARMONIZED_CLASS_IDS["built_up"]
    return None


def overpass_query(bbox: list[float]) -> str:
    minx, miny, maxx, maxy = bbox
    osm_bbox = f"{miny},{minx},{maxy},{maxx}"
    return (
        "[out:json][timeout:90];"
        "("
        f"way[building]({osm_bbox});"
        f"way[highway]({osm_bbox});"
        f"way[landuse]({osm_bbox});"
        ");out tags geom;"
    )


def ingest_osm(
    settings: dict[str, Any],
    bbox: list[float],
    profile: dict[str, Any],
    output_dir: Path,
    valid_aoi: np.ndarray,
    suffix: str = "",
) -> dict[str, Any]:
    source = settings["candidate_sources"]["osm_buildings_roads_landuse"]
    cache_path = output_dir / "osm_overpass_buildings_roads_landuse.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        fetched = False
    else:
        response = requests.get(
            source["overpass_url"],
            params={"data": overpass_query(bbox)},
            headers={"User-Agent": "confidence-aware-lcc-monitor/0.2"},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        fetched = True

    features = []
    skipped = 0
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        class_id = osm_class_id(tags)
        geom_points = element.get("geometry", [])
        if class_id is None or len(geom_points) < 2:
            skipped += 1
            continue
        coords = [(point["lon"], point["lat"]) for point in geom_points]
        try:
            if coords[0] == coords[-1] and len(coords) >= 4:
                geom = Polygon(coords)
            else:
                geom = LineString(coords)
            if geom.is_empty:
                skipped += 1
                continue
            features.append({"geometry": geom, "class_id": class_id})
        except ValueError:
            skipped += 1

    if features:
        gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326").to_crs(profile["crs"])
        buffered = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type in {"LineString", "MultiLineString"}:
                geom = geom.buffer(8)
            buffered.append((geom, int(row.class_id)))
        raster = rasterize(
            buffered,
            out_shape=(profile["height"], profile["width"]),
            transform=profile["transform"],
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
    else:
        raster = np.zeros((profile["height"], profile["width"]), dtype="uint8")

    raster = apply_aoi_mask(raster, valid_aoi)
    output_path = suffixed_path(output_dir, "osm_buildings_roads_landuse_harmonized", suffix)
    write_raster(output_path, profile, raster)
    return {
        "source": "osm_buildings_roads_landuse",
        "status": "ingested",
        "fetched_from_overpass": fetched,
        "cache": str(cache_path).replace("\\", "/"),
        "output": str(output_path).replace("\\", "/"),
        "raw_element_count": int(len(data.get("elements", []))),
        "used_feature_count": int(len(features)),
        "skipped_feature_count": int(skipped),
        "nonzero_pixels": int(np.count_nonzero(raster)),
    }


def ingest_external_sources(
    config: dict[str, Any],
    tag: str | None = None,
    dynamic_world_raster: str | None = None,
    output_suffix: str = "",
    include_local_bootstrap: bool = True,
    reference_grid: str = "processed",
) -> Path:
    ensure_output_dirs(config)
    settings = load_yaml(config["project"]["weak_labels_path"])
    selected_tag, profile, reference_grid_source = select_reference_profile(config, tag, reference_grid)
    aoi_geom, bbox = aoi_geometry(config)
    valid_aoi = aoi_mask_for_profile(aoi_geom, profile)
    output_dir = Path(config["paths"]["interim"]) / "external_sources"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        dynamic_world_status(settings, profile, output_dir, valid_aoi, dynamic_world_raster, output_suffix),
        ingest_esa_worldcover(settings, bbox, profile, output_dir, valid_aoi, output_suffix),
        ingest_osm(settings, bbox, profile, output_dir, valid_aoi, output_suffix),
    ]
    agreement = build_agreement_outputs(
        profile,
        output_dir,
        Path(config["paths"]["outputs"]),
        selected_tag,
        output_suffix,
        include_local_bootstrap,
    )
    produced_paths = [
        suffixed_path(output_dir, "dynamic_world_harmonized", output_suffix),
        suffixed_path(output_dir, "esa_worldcover_harmonized", output_suffix),
        suffixed_path(output_dir, "osm_buildings_roads_landuse_harmonized", output_suffix),
        suffixed_path(output_dir, "weak_source_agreement_labels", output_suffix),
        suffixed_path(output_dir, "weak_source_agreement_count", output_suffix),
        suffixed_path(output_dir, "weak_source_disagreement_mask", output_suffix),
    ]
    summary = {
        "phase": "phase_26b_external_weak_source_ingestion",
        "monitoring_tag": selected_tag,
        "aoi_bbox_wgs84": bbox,
        "grid": {
            "crs": str(profile["crs"]),
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "transform": [round(value, 10) for value in profile["transform"][:6]],
            "reference_source": reference_grid_source,
        },
        "sources": results,
        "agreement_ready_sources": [
            result["source"] for result in results if str(result["status"]).startswith("ingested")
        ],
        "agreement_outputs": agreement,
        "benchmark_guardrails": {
            # The suffix and bootstrap switches are leakage controls. Test-year
            # weak sources can be inspected, but they must not tune training.
            "output_suffix": output_suffix,
            "dynamic_world_override": dynamic_world_raster,
            "local_bootstrap_included": include_local_bootstrap,
            "reference_grid_mode": reference_grid,
            "leakage_policy": "Held-out benchmark sources must not be used for training, tuning, patch balancing, or pseudo-label refinement.",
            "note": "For held-out test years, keep bootstrap labels excluded until final proxy evaluation.",
        },
        "georeference_and_aoi_compatibility": alignment_report(profile, produced_paths, valid_aoi),
    }
    summary_path = suffixed_path(output_dir, "external_weak_sources_summary", output_suffix, ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest external weak-label sources.")
    parser.add_argument("--tag", default=None, help="Monitoring tag used for the reference grid.")
    parser.add_argument(
        "--reference-grid",
        choices=["processed", "master"],
        default="processed",
        help="Use a processed monitoring raster or the fixed master grid as the harmonization target.",
    )
    parser.add_argument("--dynamic-world-raster", default=None, help="Optional local Dynamic World label raster override.")
    parser.add_argument("--output-suffix", default="", help="Suffix for isolated benchmark/test outputs.")
    parser.add_argument(
        "--exclude-local-bootstrap",
        action="store_true",
        help="Exclude local bootstrap labels from agreement outputs to avoid benchmark leakage.",
    )
    args = parser.parse_args()
    summary_path = ingest_external_sources(
        load_config(),
        tag=args.tag,
        dynamic_world_raster=args.dynamic_world_raster,
        output_suffix=args.output_suffix,
        include_local_bootstrap=not args.exclude_local_bootstrap,
        reference_grid=args.reference_grid,
    )
    print(f"External weak-source summary: {summary_path}")


if __name__ == "__main__":
    main()
