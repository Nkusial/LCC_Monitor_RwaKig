"""AOI-windowed Sentinel-1/2 preprocessing.

The important design choice is to clip remote Sentinel assets to the small AOI
before mosaicking or feature creation. That keeps the local portfolio workflow
lightweight and avoids full-tile raster reads.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any

import geopandas as gpd
import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.merge import merge
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, transform_geom

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def load_scene_catalog(path: str | Path) -> list[dict[str, Any]]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Scene catalog not found: {catalog_path}. Run pipelines/01_search_scenes.py first."
        )

    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return data.get("features", [])


def select_sentinel2_scenes(
    features: list[dict[str, Any]], selected_ids: list[str]
) -> list[dict[str, Any]]:
    by_id = {feature.get("id") or feature["properties"]["id"]: feature for feature in features}
    missing = [scene_id for scene_id in selected_ids if scene_id not in by_id]
    if missing:
        raise ValueError(f"Selected Sentinel-2 scene(s) missing from catalog: {missing}")

    return [by_id[scene_id] for scene_id in selected_ids]


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute normalized-difference indices while avoiding divide-by-zero."""
    denominator = a + b
    out = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, denominator, out=out, where=np.abs(denominator) > 1e-6)
    return out


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute radar ratios with NaN where the denominator is invalid."""
    out = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a, b, out=out, where=np.abs(b) > 1e-6)
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_sentinel1_item(config: dict[str, Any], aoi: gpd.GeoDataFrame) -> Any:
    """Find the configured reference Sentinel-1 item or a compatible fallback."""
    pair = config["preprocessing"]["selected_pair"]
    radar = pair["radar"]
    client = Client.open(config["sentinel"]["stac_api_url"])

    search = client.search(
        collections=[radar["collection"]],
        bbox=aoi.to_crs("EPSG:4326").total_bounds.tolist(),
        datetime="2025-02-18/2025-02-23",
        max_items=25,
    )

    candidates = list(search.items())
    for item in candidates:
        if item.id == radar["scene_id"]:
            return planetary_computer.sign(item)

    for item in candidates:
        props = item.properties
        polarizations = props.get("sar:polarizations", [])
        orbit = props.get("sat:orbit_state")
        if orbit == radar["orbit"] and all(pol in polarizations for pol in radar["polarizations"]):
            return planetary_computer.sign(item)

    raise ValueError("No matching Sentinel-1 RTC scene found for the selected reference date.")


def get_sentinel1_item_by_id(config: dict[str, Any], aoi: gpd.GeoDataFrame, pair: dict[str, Any]) -> Any:
    radar = pair["radar"]
    client = Client.open(config["sentinel"]["stac_api_url"])
    date = radar["date"]
    search = client.search(
        collections=[radar["collection"]],
        bbox=aoi.to_crs("EPSG:4326").total_bounds.tolist(),
        datetime=f"{date}/{date}",
        max_items=10,
    )

    for item in search.items():
        if item.id == radar["scene_id"]:
            return planetary_computer.sign(item)

    raise ValueError(f"Selected Sentinel-1 RTC scene not found on {date}: {radar['scene_id']}")


def geometry_for_raster(aoi: gpd.GeoDataFrame, dst_crs: Any) -> list[dict[str, Any]]:
    """Transform the AOI geometry into the raster CRS before windowing."""
    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    return [
        transform_geom("EPSG:4326", dst_crs, geom)
        for geom in json.loads(aoi_wgs84.to_json())["features"][0:1]
        for geom in [geom["geometry"]]
    ]


def clip_asset_to_file(href: str, aoi: gpd.GeoDataFrame, output_path: Path) -> Path:
    """Read only the AOI window from a remote raster asset and save it locally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rasterio_env = {
        # These GDAL options reduce remote listing and retry overhead when
        # reading cloud-hosted GeoTIFFs through rasterio.
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "GDAL_HTTP_MAX_RETRY": "2",
        "GDAL_HTTP_RETRY_DELAY": "1",
    }
    with rasterio.Env(**rasterio_env):
        with rasterio.open(href) as src:
            shapes = geometry_for_raster(aoi, src.crs)
            try:
                window = geometry_window(src, shapes)
            except WindowError as exc:
                raise ValueError(f"AOI does not overlap raster asset: {href}") from exc

            data = src.read(1, window=window)
            transform = window_transform(window, src.transform)
            nodata = src.nodata if src.nodata is not None else 0
            # Mask pixels outside the AOI after windowing so downstream rasters
            # share the same clipped footprint but do not include surrounding tile data.
            mask_array = geometry_mask(
                shapes,
                out_shape=data.shape,
                transform=transform,
                invert=True,
                all_touched=True,
            )
            clipped = np.where(mask_array, data, nodata)[np.newaxis, ...]
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=transform,
                compress="deflate",
                tiled=True,
                count=1,
                nodata=nodata,
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(clipped)
    return output_path


def mosaic_clipped_assets(
    scenes: list[dict[str, Any]],
    band_key: str,
    aoi: gpd.GeoDataFrame,
    output_path: Path,
    temp_dir: Path,
) -> Path:
    """Clip each Sentinel-2 tile to AOI first, then mosaic the small pieces."""
    clipped_paths = []
    for scene in scenes:
        assets = scene["properties"]["assets"]
        href = planetary_computer.sign_url(assets[band_key])
        clipped_paths.append(
            clip_asset_to_file(href, aoi, temp_dir / f"{scene['id']}_{band_key}.tif")
        )

    datasets = [rasterio.open(path) for path in clipped_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
            tiled=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()

    return output_path


def read_reference_profile(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        data = src.read(1).astype("float32")
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=True)
    return profile, data


def reproject_to_match(path: Path, reference_profile: dict[str, Any]) -> np.ndarray:
    """Resample a raster onto the reference optical grid."""
    destination = np.full(
        (reference_profile["height"], reference_profile["width"]), np.nan, dtype="float32"
    )
    with rasterio.open(path) as src:
        reproject(
            source=src.read(1).astype("float32"),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return destination


def write_single_band(path: Path, profile: dict[str, Any], data: np.ndarray, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
        dst.set_band_description(1, description)


def preprocess_pair(
    config: dict[str, Any],
    pair: dict[str, Any],
    output_tag: str,
    metadata_name: str,
    catalog_name: str = "scenes.geojson",
    metadata_only: bool = False,
    mode: str = "all",
) -> Path:
    """Create optical indices and radar features for one monitoring pair."""
    ensure_output_dirs(config)
    processed_dir = Path(config["paths"]["processed"])
    catalog_path = Path(config["paths"]["catalog"]) / catalog_name
    aoi = gpd.read_file(config["project"]["aoi_path"])

    s2_scenes = select_sentinel2_scenes(
        load_scene_catalog(catalog_path), pair["optical"]["scene_ids"]
    )
    if pair["radar"].get("assets"):
        s1_item = SimpleNamespace(
            id=pair["radar"]["scene_id"],
            properties={
                "sat:orbit_state": pair["radar"].get("orbit"),
                "sar:polarizations": pair["radar"].get("polarizations", []),
            },
        )
        s1_assets = {
            key: SimpleNamespace(href=href) for key, href in pair["radar"]["assets"].items()
        }
    elif "scene_id" in pair["radar"]:
        s1_item = get_sentinel1_item_by_id(config, aoi, pair)
        s1_assets = planetary_computer.sign(s1_item).assets
    else:
        s1_item = find_sentinel1_item(config, aoi)
        s1_assets = planetary_computer.sign(s1_item).assets

    manifest = {
        "phase": "phase_4_preprocessing",
        "aoi_path": config["project"]["aoi_path"],
        "sentinel2": {
            "date": pair["optical"]["date"],
            "scene_ids": [scene["id"] for scene in s2_scenes],
            "tiles": pair["optical"]["tiles"],
            "bands": pair["optical"]["bands"],
        },
        "sentinel1": {
            "date": pair["radar"]["date"],
            "scene_id": s1_item.id,
            "orbit": s1_item.properties.get("sat:orbit_state"),
            "polarizations": s1_item.properties.get("sar:polarizations", []),
            "assets": sorted(s1_assets.keys()),
        },
        "outputs": [],
    }

    if metadata_only or mode == "metadata":
        manifest_path = processed_dir / metadata_name
        write_json(manifest_path, manifest)
        return manifest_path

    band_map = {
        name: band_key
        for name, band_key in pair["optical"]["bands"].items()
        if name in {"green", "red", "nir", "swir1"}
    }
    with TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        s2_band_paths = {}
        if mode in {"all", "optical"}:
            for name, band_key in band_map.items():
                print(f"Clipping Sentinel-2 {name} ({band_key})...", flush=True)
                s2_band_paths[name] = mosaic_clipped_assets(
                    s2_scenes,
                    band_key,
                    aoi,
                    processed_dir / f"{output_tag}_s2_{name}.tif",
                    temp_dir,
                )
                manifest["outputs"].append(str(s2_band_paths[name]))
        else:
            s2_band_paths = {
                name: processed_dir / f"{output_tag}_s2_{name}.tif" for name in band_map
            }
            missing = [str(path) for path in s2_band_paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Radar-only mode needs existing Sentinel-2 reference rasters. "
                    f"Run --mode optical first. Missing: {missing}"
                )

        reference_profile, red = read_reference_profile(s2_band_paths["red"])

        if mode in {"all", "optical"}:
            print("Creating Sentinel-2 NDVI, NDWI, and NDBI...", flush=True)
            green = reproject_to_match(s2_band_paths["green"], reference_profile)
            nir = reproject_to_match(s2_band_paths["nir"], reference_profile)
            swir1 = reproject_to_match(s2_band_paths["swir1"], reference_profile)
            # Sentinel-2 SWIR1 is native 20 m. Store the aligned 10 m version
            # back into data/processed so future feature expansion, ML, and
            # audits do not accidentally consume an off-grid source band.
            write_single_band(s2_band_paths["swir1"], reference_profile, swir1, "SWIR1 aligned")

            # The indices become the main optical features for change detection,
            # validation clustering, and WebGIS raster overlays.
            index_outputs = {
                "ndvi": normalized_difference(nir, red),
                "ndwi": normalized_difference(green, nir),
                "ndbi": normalized_difference(swir1, nir),
            }
            for name, data in index_outputs.items():
                output_path = processed_dir / f"{output_tag}_s2_{name}.tif"
                write_single_band(output_path, reference_profile, data, name.upper())
                manifest["outputs"].append(str(output_path))

        if mode not in {"all", "radar"}:
            manifest_path = processed_dir / metadata_name
            write_json(manifest_path, manifest)
            return manifest_path

        s1_outputs = {}
        for polarization in pair["radar"]["polarizations"]:
            asset_key = polarization.lower()
            if asset_key not in s1_assets:
                raise KeyError(f"Sentinel-1 asset missing: {asset_key}")
            print(f"Clipping Sentinel-1 {polarization}...", flush=True)
            clipped_path = clip_asset_to_file(
                s1_assets[asset_key].href,
                aoi,
                temp_dir / f"{s1_item.id}_{asset_key}.tif",
            )
            data = reproject_to_match(clipped_path, reference_profile)
            output_path = processed_dir / f"{output_tag}_s1_{asset_key}.tif"
            write_single_band(output_path, reference_profile, data, polarization)
            s1_outputs[polarization] = data
            manifest["outputs"].append(str(output_path))

        vv_vh = safe_ratio(s1_outputs["VV"], s1_outputs["VH"])
        ratio_path = processed_dir / f"{output_tag}_s1_vv_vh_ratio.tif"
        write_single_band(ratio_path, reference_profile, vv_vh, "VV/VH")
        manifest["outputs"].append(str(ratio_path))

    manifest_path = processed_dir / metadata_name
    write_json(manifest_path, manifest)
    return manifest_path


def preprocess(config: dict[str, Any], metadata_only: bool = False, mode: str = "all") -> Path:
    pair = config["preprocessing"]["selected_pair"]
    return preprocess_pair(
        config,
        pair=pair,
        output_tag="reference_20250220_20250221",
        metadata_name="phase4_manifest.json",
        catalog_name="scenes.geojson",
        metadata_only=metadata_only,
        mode=mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip and prepare Sentinel-1/2 AOI rasters.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate selected scenes and write a manifest without reading raster assets.",
    )
    parser.add_argument(
        "--mode",
        choices=["metadata", "optical", "radar", "all"],
        default="all",
        help="Run only part of Phase 4. Use optical first, then radar, or all.",
    )
    args = parser.parse_args()

    manifest_path = preprocess(load_config(), metadata_only=args.metadata_only, mode=args.mode)
    print(f"Phase 4 preprocessing complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
