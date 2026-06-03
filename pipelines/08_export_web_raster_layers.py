"""Export lightweight georeferenced raster overlays for the hosted WebGIS.

GitHub Pages cannot serve Cloud Optimized GeoTIFFs as a dynamic raster service,
so this script converts selected AOI-clipped GeoTIFF products into transparent
PNG overlays plus a small JSON manifest consumed by the React/MapLibre app.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, transform, transform_geom


ROOT = Path(__file__).resolve().parents[1]
AOI_PATH = ROOT / "configs" / "aoi.geojson"
MASTER_GRID_PATH = ROOT / "configs" / "master_grid.yaml"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "data" / "outputs"
UNET_FULL_AOI_BASELINE = ROOT / "data" / "interim" / "unet_full_aoi"
UNET_FULL_AOI_V3 = ROOT / "data" / "interim" / "unet_full_aoi_v3"
UNET_FULL_AOI = (
    UNET_FULL_AOI_V3
    if (UNET_FULL_AOI_V3 / "unet_full_aoi_manifest_v3.json").exists()
    else UNET_FULL_AOI_BASELINE
)
UNET_MANIFEST_NAME = (
    "unet_full_aoi_manifest_v3.json"
    if UNET_FULL_AOI == UNET_FULL_AOI_V3
    else "unet_full_aoi_manifest.json"
)
EXTERNAL_SOURCES = ROOT / "data" / "interim" / "external_sources"
WEB_DEMO = ROOT / "web" / "public" / "demo"
RASTER_DIR = WEB_DEMO / "rasters"
RASTER_TILE_DIR = WEB_DEMO / "raster_tiles"
MANIFEST_PATH = WEB_DEMO / "raster_layers.json"
BENCHMARK_2026_MANIFEST = UNET_FULL_AOI_BASELINE / "unet_full_aoi_manifest_test_2026.json"
BENCHMARK_2026_REPORT = OUTPUTS / "unet_2026_proxy_benchmark_report.json"
UNET_RELIABILITY_REPORT = (
    OUTPUTS / "unet_v3_full_aoi_validation_report.json"
    if (OUTPUTS / "unet_v3_full_aoi_validation_report.json").exists()
    else OUTPUTS / "unet_full_aoi_validation_report.json"
)
GEOSPATIAL_ALIGNMENT_REPORT = OUTPUTS / "geospatial_alignment_report.json"
COREGISTRATION_QA_REPORT = OUTPUTS / "coregistration_qa_report.json"
MAX_SIZE = 2048
DISPLAY_CRS = "EPSG:3857"
TILE_SIZE = 256
TILE_MIN_ZOOM = 12
TILE_MAX_ZOOM = 14
WEB_MERCATOR_HALF_WORLD = 20_037_508.342789244
_AOI_GEOMETRIES_WGS84: list[dict] | None = None
_MASTER_GRID: dict | None = None


@dataclass(frozen=True)
class SingleBandLayer:
    id: str
    title: str
    group: str
    path: Path
    value_range: tuple[float, float] | None
    palette: tuple[tuple[int, int, int], ...]
    description: str
    opacity: float = 0.72


@dataclass(frozen=True)
class DisplayGrid:
    """Browser display grid for PNG overlays after Web Mercator reprojection."""

    crs: str
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    source_path: Path


def output_shape(width: int, height: int) -> tuple[int, int]:
    """Downsample display overlays so GitHub Pages stays lightweight."""
    scale = min(MAX_SIZE / max(width, height), 1)
    return max(1, round(height * scale)), max(1, round(width * scale))


def load_master_grid() -> dict:
    """Load the fixed analysis grid used by preprocessing, ML, and WebGIS export."""
    global _MASTER_GRID
    if _MASTER_GRID is None:
        _MASTER_GRID = yaml.safe_load(MASTER_GRID_PATH.read_text(encoding="utf-8"))
    return _MASTER_GRID


def expected_master_grid() -> dict:
    """Normalize the master-grid YAML into directly comparable raster metadata."""
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


def master_grid_coordinates_wgs84() -> list[list[float]]:
    """Return exact WGS84 corner coordinates of the authoritative 10 m grid."""
    expected = expected_master_grid()
    left, bottom, right, top = expected["bounds"]
    xs = [left, right, right, left]
    ys = [top, top, bottom, bottom]
    lngs, lats = transform(
        expected["crs"],
        "EPSG:4326",
        xs,
        ys,
    )
    return [[float(lngs[index]), float(lats[index])] for index in range(4)]


def master_grid_aoi_geojson() -> dict:
    """Build the hosted AOI polygon from the same grid footprint as rasters."""
    coordinates = master_grid_coordinates_wgs84()
    return {
        "type": "FeatureCollection",
        "name": "kigali_demo_aoi_master_grid",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Kigali 20 km demo AOI",
                    "description": (
                        "Master-grid-aligned 10 m WebGIS AOI footprint used for "
                        "Sentinel-1/2 overlays, model outputs, and change polygons."
                    ),
                    "source": "configs/master_grid.yaml",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[*coordinates, coordinates[0]]],
                },
            }
        ],
    }


def dataset_grid(dataset: rasterio.DatasetReader) -> dict:
    """Extract coordinate metadata that must match before hosted export."""
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


def close_enough(actual: list[float], expected: list[float], tolerance: float) -> bool:
    """Compare geospatial coordinates while allowing tiny floating-point noise."""
    return len(actual) == len(expected) and all(
        abs(actual_value - expected_value) <= tolerance
        for actual_value, expected_value in zip(actual, expected)
    )


def validate_export_grid(dataset: rasterio.DatasetReader) -> None:
    """Fail fast when a hosted raster is not on the project master grid."""
    expected = expected_master_grid()
    actual = dataset_grid(dataset)
    tolerance = expected["tolerance"]
    mismatches = []

    for key in ("crs", "width", "height"):
        if actual[key] != expected[key]:
            mismatches.append(key)
    for key in ("transform", "bounds"):
        if not close_enough(actual[key], expected[key], tolerance):
            mismatches.append(key)

    if mismatches:
        raise ValueError(
            "Hosted raster is not aligned to configs/master_grid.yaml: "
            f"{dataset.name}; mismatched fields: {', '.join(mismatches)}"
        )


def display_grid_from_dataset(dataset: rasterio.DatasetReader, path: Path) -> DisplayGrid:
    """Create a Web Mercator display grid from a validated analysis raster.

    The analysis products stay on the EPSG:32735 master grid. Only the PNG
    served to MapLibre is reprojected to Web Mercator so the browser does not
    stretch a UTM raster horizontally against the OSM basemap.
    """
    if dataset.crs is None:
        raise ValueError(f"Raster has no CRS and cannot be displayed: {path}")

    dst_transform, dst_width, dst_height = calculate_default_transform(
        dataset.crs,
        DISPLAY_CRS,
        dataset.width,
        dataset.height,
        *dataset.bounds,
    )
    height, width = output_shape(dst_width, dst_height)
    display_transform = dst_transform * Affine.scale(
        dst_width / width,
        dst_height / height,
    )
    right = display_transform.c + display_transform.a * width
    bottom = display_transform.f + display_transform.e * height
    return DisplayGrid(
        crs=DISPLAY_CRS,
        transform=display_transform,
        width=width,
        height=height,
        bounds=(display_transform.c, bottom, right, display_transform.f),
        source_path=path,
    )


def read_band(path: Path) -> tuple[np.ma.MaskedArray, DisplayGrid]:
    """Read a source GeoTIFF into the WebGIS display CRS after grid validation."""
    with rasterio.open(path) as dataset:
        validate_export_grid(dataset)
        grid = display_grid_from_dataset(dataset, path)
        destination = np.full((grid.height, grid.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(dataset, 1),
            destination=destination,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_nodata=dataset.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return np.ma.masked_invalid(destination), grid


def normalize(array: np.ma.MaskedArray, limits: tuple[float, float] | None = None) -> np.ndarray:
    """Scale raster values into 0..1 for PNG color mapping."""
    values = array.compressed()
    if values.size == 0:
        return np.zeros(array.shape, dtype="float32")

    if limits is None:
        low, high = np.nanpercentile(values, [2, 98])
    else:
        low, high = limits

    if np.isclose(low, high):
        high = low + 1

    normalized = (array.filled(low) - low) / (high - low)
    return np.clip(normalized, 0, 1).astype("float32")


def alpha_from_mask(array: np.ma.MaskedArray, opacity: float) -> np.ndarray:
    alpha = np.where(np.ma.getmaskarray(array), 0, round(255 * opacity))
    return alpha.astype("uint8")


def load_aoi_geometries_wgs84() -> list[dict]:
    """Load the master-grid AOI once so display rasters and outline match."""
    global _AOI_GEOMETRIES_WGS84
    if _AOI_GEOMETRIES_WGS84 is not None:
        return _AOI_GEOMETRIES_WGS84

    data = master_grid_aoi_geojson()
    _AOI_GEOMETRIES_WGS84 = [
        feature["geometry"]
        for feature in data.get("features", [])
        if feature.get("geometry") is not None
    ]

    if not _AOI_GEOMETRIES_WGS84:
        raise ValueError(f"No AOI geometry could be derived from {MASTER_GRID_PATH}")
    return _AOI_GEOMETRIES_WGS84


def aoi_mask_for_display(
    grid: DisplayGrid,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize the AOI polygon onto the Web Mercator display grid."""
    geometries = [
        transform_geom("EPSG:4326", grid.crs, geometry, precision=6)
        for geometry in load_aoi_geometries_wgs84()
    ]
    mask = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask.astype(bool)


def apply_aoi_alpha(alpha: np.ndarray, grid: DisplayGrid) -> np.ndarray:
    """Force hosted display overlays to be transparent outside the AOI polygon."""
    aoi_mask = aoi_mask_for_display(grid, alpha.shape)
    return np.where(aoi_mask, alpha, 0).astype("uint8")


def raster_coordinates(grid: DisplayGrid) -> list[list[float]]:
    """Return MapLibre image-source coordinates from the Web Mercator PNG grid."""
    left, bottom, right, top = grid.bounds
    xs = [left, right, right, left]
    ys = [top, top, bottom, bottom]
    lngs, lats = transform(grid.crs, "EPSG:4326", xs, ys)
    return [[float(lngs[index]), float(lats[index])] for index in range(4)]


def save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    rgba = np.dstack([rgb, alpha]).astype("uint8")
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def tile_bounds_mercator(zoom: int, tile_x: int, tile_y: int) -> tuple[float, float, float, float]:
    """Return XYZ tile bounds in EPSG:3857 metres."""
    tile_span = (WEB_MERCATOR_HALF_WORLD * 2) / (2**zoom)
    left = -WEB_MERCATOR_HALF_WORLD + tile_x * tile_span
    right = left + tile_span
    top = WEB_MERCATOR_HALF_WORLD - tile_y * tile_span
    bottom = top - tile_span
    return left, bottom, right, top


def tile_range_for_bounds(
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> tuple[range, range]:
    """Find the XYZ tiles touched by an EPSG:3857 raster footprint."""
    left, bottom, right, top = bounds
    tile_span = (WEB_MERCATOR_HALF_WORLD * 2) / (2**zoom)
    max_tile = 2**zoom - 1
    x_min = max(0, min(max_tile, math.floor((left + WEB_MERCATOR_HALF_WORLD) / tile_span)))
    x_max = max(0, min(max_tile, math.floor((right + WEB_MERCATOR_HALF_WORLD) / tile_span)))
    y_min = max(0, min(max_tile, math.floor((WEB_MERCATOR_HALF_WORLD - top) / tile_span)))
    y_max = max(0, min(max_tile, math.floor((WEB_MERCATOR_HALF_WORLD - bottom) / tile_span)))
    return range(x_min, x_max + 1), range(y_min, y_max + 1)


def source_window_from_mercator_bounds(
    grid: DisplayGrid,
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Convert EPSG:3857 bounds into source image pixel windows."""
    left, bottom, right, top = bounds
    col0 = math.floor((left - grid.transform.c) / grid.transform.a)
    col1 = math.ceil((right - grid.transform.c) / grid.transform.a)
    row0 = math.floor((top - grid.transform.f) / grid.transform.e)
    row1 = math.ceil((bottom - grid.transform.f) / grid.transform.e)
    return (
        max(0, min(grid.width, col0)),
        max(0, min(grid.height, row0)),
        max(0, min(grid.width, col1)),
        max(0, min(grid.height, row1)),
    )


def destination_window_in_tile(
    tile_bounds: tuple[float, float, float, float],
    overlap_bounds: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Map an EPSG:3857 overlap window into a 256 px XYZ tile window."""
    tile_left, tile_bottom, tile_right, tile_top = tile_bounds
    left, bottom, right, top = overlap_bounds
    tile_width = tile_right - tile_left
    tile_height = tile_top - tile_bottom
    x0 = round((left - tile_left) / tile_width * TILE_SIZE)
    x1 = round((right - tile_left) / tile_width * TILE_SIZE)
    y0 = round((tile_top - top) / tile_height * TILE_SIZE)
    y1 = round((tile_top - bottom) / tile_height * TILE_SIZE)
    return (
        max(0, min(TILE_SIZE, x0)),
        max(0, min(TILE_SIZE, y0)),
        max(0, min(TILE_SIZE, x1)),
        max(0, min(TILE_SIZE, y1)),
    )


def write_tile_pyramid(layer_id: str, rgb: np.ndarray, alpha: np.ndarray, grid: DisplayGrid) -> str:
    """Write Web Mercator XYZ tiles so MapLibre avoids single-image warping."""
    rgba = np.dstack([rgb, alpha]).astype("uint8")
    source_image = Image.fromarray(rgba, mode="RGBA")
    layer_tile_dir = RASTER_TILE_DIR / layer_id
    if layer_tile_dir.exists():
        def unlock_and_retry(function, path, _exc_info):
            os.chmod(path, stat.S_IWRITE)
            time.sleep(0.05)
            function(path)

        shutil.rmtree(layer_tile_dir, onerror=unlock_and_retry)

    for zoom in range(TILE_MIN_ZOOM, TILE_MAX_ZOOM + 1):
        x_range, y_range = tile_range_for_bounds(grid.bounds, zoom)
        for tile_x in x_range:
            for tile_y in y_range:
                tile_left, tile_bottom, tile_right, tile_top = tile_bounds_mercator(
                    zoom, tile_x, tile_y
                )
                overlap_left = max(tile_left, grid.bounds[0])
                overlap_bottom = max(tile_bottom, grid.bounds[1])
                overlap_right = min(tile_right, grid.bounds[2])
                overlap_top = min(tile_top, grid.bounds[3])
                if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
                    continue

                src_col0, src_row0, src_col1, src_row1 = source_window_from_mercator_bounds(
                    grid,
                    (overlap_left, overlap_bottom, overlap_right, overlap_top),
                )
                dst_x0, dst_y0, dst_x1, dst_y1 = destination_window_in_tile(
                    (tile_left, tile_bottom, tile_right, tile_top),
                    (overlap_left, overlap_bottom, overlap_right, overlap_top),
                )
                if src_col0 >= src_col1 or src_row0 >= src_row1 or dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
                    continue

                tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                crop = source_image.crop((src_col0, src_row0, src_col1, src_row1))
                resampling = Image.Resampling.BILINEAR
                if "class" in layer_id or "clusters" in layer_id or "review_zones" in layer_id:
                    resampling = Image.Resampling.NEAREST
                crop = crop.resize((dst_x1 - dst_x0, dst_y1 - dst_y0), resampling)
                tile.paste(crop, (dst_x0, dst_y0))

                tile_path = layer_tile_dir / str(zoom) / str(tile_x) / f"{tile_y}.png"
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                tile.save(tile_path, optimize=True)

    return f"raster_tiles/{layer_id}/{{z}}/{{x}}/{{y}}.png"


def display_asset_record(layer_id: str, rgb: np.ndarray, alpha: np.ndarray, grid: DisplayGrid) -> dict:
    """Export both a fallback image and the preferred tiled WebGIS asset."""
    filename = f"{layer_id}.png"
    save_rgba(RASTER_DIR / filename, rgb, alpha)
    return {
        "path": f"rasters/{filename}",
        "tile_template": write_tile_pyramid(layer_id, rgb, alpha, grid),
        "minzoom": TILE_MIN_ZOOM,
        "maxzoom": TILE_MAX_ZOOM,
        "tileSize": TILE_SIZE,
        "coordinates": raster_coordinates(grid),
    }


def interpolate_palette(values: np.ndarray, palette: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    stops = np.linspace(0, 1, len(palette), dtype="float32")
    colors = np.array(palette, dtype="float32")
    channels = [np.interp(values, stops, colors[:, index]) for index in range(3)]
    return np.dstack(channels).astype("uint8")


def export_single_band(layer: SingleBandLayer) -> dict:
    array, grid = read_band(layer.path)
    values = normalize(array, layer.value_range)
    rgb = interpolate_palette(values, layer.palette)
    alpha = apply_aoi_alpha(alpha_from_mask(array, layer.opacity), grid)
    assets = display_asset_record(layer.id, rgb, alpha, grid)
    return {
        "id": layer.id,
        "title": layer.title,
        "group": layer.group,
        "kind": "raster_overlay",
        "opacity": layer.opacity,
        "description": layer.description,
        "source": str(layer.path.relative_to(ROOT)).replace("\\", "/"),
        "satelliteDerived": True,
        **assets,
    }


def export_categorical_layer(
    layer_id: str,
    title: str,
    group: str,
    path: Path,
    class_palette: dict[int, tuple[int, int, int]],
    description: str,
    opacity: float = 0.72,
) -> dict:
    """Export integer class rasters with stable land-cover colors."""
    array, grid = read_band(path)
    classes = array.filled(0).astype("uint8")
    rgb = np.zeros((*classes.shape, 3), dtype="uint8")
    for class_id, color in class_palette.items():
        rgb[classes == class_id] = color
    alpha = np.where(classes == 0, 0, round(255 * opacity)).astype("uint8")
    alpha = apply_aoi_alpha(alpha, grid)
    assets = display_asset_record(layer_id, rgb, alpha, grid)
    return {
        "id": layer_id,
        "title": title,
        "group": group,
        "kind": "raster_overlay",
        "opacity": opacity,
        "description": description,
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "satelliteDerived": True,
        **assets,
    }


def latest_validation_unet_tag() -> str | None:
    """Pick a validation full-AOI U-Net product for hosted portfolio display."""
    manifest_path = UNET_FULL_AOI / UNET_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_records = [
        record for record in manifest.get("records", []) if record.get("split") == "validation"
    ]
    if not validation_records:
        return None
    return sorted(validation_records, key=lambda record: record["training_tag"])[-1]["training_tag"]


def export_unet_full_aoi_layers() -> list[dict]:
    """Export selected full-AOI U-Net products to the hosted WebGIS."""
    tag = latest_validation_unet_tag()
    if not tag:
        return []

    dominant = UNET_FULL_AOI / f"{tag}_unet_full_aoi_dominant_class.tif"
    confidence = UNET_FULL_AOI / f"{tag}_unet_full_aoi_confidence.tif"
    entropy = UNET_FULL_AOI / f"{tag}_unet_full_aoi_entropy.tif"
    if not (dominant.exists() and confidence.exists() and entropy.exists()):
        return []

    pretty_tag = tag.replace("training_", "").replace("_", " / ")
    return [
        export_categorical_layer(
            layer_id="unet_full_aoi_dominant_class",
            title="U-Net dominant class",
            group="U-Net model outputs",
            path=dominant,
            class_palette={
                1: (214, 70, 23),
                2: (47, 137, 86),
                3: (181, 119, 27),
                4: (43, 139, 214),
                5: (112, 76, 219),
            },
            description=(
                f"Weakly supervised full-AOI U-Net dominant land-cover prediction for {pretty_tag}. "
                "This is a model output, not field-validated accuracy."
            ),
            opacity=0.62,
        ),
        export_single_band(
            SingleBandLayer(
                id="unet_full_aoi_confidence",
                title="U-Net confidence",
                group="U-Net model outputs",
                path=confidence,
                value_range=(0, 1),
                palette=((247, 247, 247), (254, 224, 144), (253, 174, 97), (165, 0, 38)),
                description=(
                    f"Maximum class probability from full-AOI U-Net inference for {pretty_tag}. "
                    "Use with entropy before interpreting model predictions."
                ),
                opacity=0.66,
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="unet_full_aoi_entropy",
                title="U-Net entropy",
                group="U-Net model outputs",
                path=entropy,
                value_range=(0, 1),
                palette=((26, 150, 65), (255, 255, 191), (253, 174, 97), (215, 25, 28)),
                description=(
                    f"Normalized uncertainty from full-AOI U-Net inference for {pretty_tag}; "
                    "higher entropy means less confident class separation."
                ),
                opacity=0.66,
            )
        ),
    ]


def export_unet_review_zone_layer() -> dict | None:
    """Export high-review-priority U-Net reliability zones for the WebGIS."""
    tag = latest_validation_unet_tag()
    if not tag:
        return None

    confidence = UNET_FULL_AOI / f"{tag}_unet_full_aoi_confidence.tif"
    entropy = UNET_FULL_AOI / f"{tag}_unet_full_aoi_entropy.tif"
    disagreement = EXTERNAL_SOURCES / "weak_source_disagreement_mask.tif"
    if not (confidence.exists() and entropy.exists() and disagreement.exists()):
        return None

    confidence_array, grid = read_band(confidence)
    entropy_array, _entropy_grid = read_band(entropy)
    disagreement_array, _disagreement_grid = read_band(disagreement)
    low_confidence = confidence_array.filled(-9999) < 0.6
    high_entropy = entropy_array.filled(-9999) >= 0.65
    weak_disagreement = disagreement_array.filled(0) > 0

    classes = np.zeros(confidence_array.shape, dtype="uint8")
    classes[low_confidence] = 1
    classes[high_entropy] = 2
    classes[weak_disagreement] = 3
    classes[np.ma.getmaskarray(confidence_array)] = 0

    rgb = np.zeros((*classes.shape, 3), dtype="uint8")
    rgb[classes == 1] = (245, 158, 11)
    rgb[classes == 2] = (220, 38, 38)
    rgb[classes == 3] = (124, 58, 237)
    alpha = np.where(classes == 0, 0, 185).astype("uint8")
    alpha = apply_aoi_alpha(alpha, grid)

    layer_id = "unet_review_zones"
    assets = display_asset_record(layer_id, rgb, alpha, grid)
    pretty_tag = tag.replace("training_", "").replace("_", " / ")
    return {
        "id": layer_id,
        "title": "U-Net review zones",
        "group": "Reliability review",
        "kind": "raster_overlay",
        "opacity": 0.72,
        "description": (
            f"Candidate review zones for {pretty_tag}: orange is low model confidence, "
            "red is high entropy, and purple is weak-source disagreement. This is a review aid, not accuracy."
        ),
        "source": (
            f"{UNET_FULL_AOI.relative_to(ROOT).as_posix()} confidence/entropy plus "
            "data/interim/external_sources/weak_source_disagreement_mask.tif"
        ),
        "satelliteDerived": True,
        **assets,
    }


def export_cluster_layer() -> dict | None:
    """Export the best unsupervised validation cluster raster, if available."""
    report_path = OUTPUTS / "unsupervised_validation_report.json"
    if not report_path.exists():
        return None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    cluster_path = ROOT / report["outputs"]["cluster_raster"]
    if not cluster_path.exists():
        return None

    array, grid = read_band(cluster_path)
    clusters = array.filled(0).astype("uint8")
    palette = np.array(
        [
            [0, 0, 0],
            [35, 132, 67],
            [253, 174, 97],
            [216, 179, 101],
            [128, 205, 193],
            [94, 79, 162],
            [213, 62, 79],
            [102, 194, 165],
            [230, 245, 152],
            [50, 136, 189],
            [166, 97, 26],
        ],
        dtype="uint8",
    )
    rgb = palette[np.clip(clusters, 0, len(palette) - 1)]
    alpha = np.where(clusters == 0, 0, 178).astype("uint8")
    alpha = apply_aoi_alpha(alpha, grid)
    layer_id = "unsupervised_clusters_best"
    assets = display_asset_record(layer_id, rgb, alpha, grid)
    return {
        "id": layer_id,
        "title": "Unsupervised clusters",
        "group": "Validation",
        "kind": "raster_overlay",
        "opacity": 0.7,
        "description": "Best Phase 20B internal validation cluster overlay. This is not field accuracy.",
        "source": str(cluster_path.relative_to(ROOT)).replace("\\", "/"),
        "satelliteDerived": True,
        **assets,
    }


def export_dashboard_summary() -> None:
    """Publish product-facing summary metrics derived from hosted change polygons."""
    changes_path = (
        OUTPUTS / "monitoring_change_scored.geojson"
        if (OUTPUTS / "monitoring_change_scored.geojson").exists()
        else WEB_DEMO / "changes.geojson"
    )
    if not changes_path.exists():
        return

    data = json.loads(changes_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    by_land_cover: dict[str, int] = {}
    by_land_cover_area: dict[str, float] = {}
    by_reliability: dict[str, int] = {}
    by_reliability_area: dict[str, float] = {}
    by_transition_area: dict[str, float] = {}
    confidences: list[float] = []
    changed_area_m2 = 0.0
    published_count = 0

    for feature in features:
        properties = feature.get("properties") or {}
        if properties.get("publish_ready") is False:
            continue

        published_count += 1
        area_m2 = float(properties.get("area_m2") or 0)
        confidence = properties.get("final_confidence", properties.get("confidence"))
        if confidence is not None:
            confidences.append(float(confidence))

        changed_area_m2 += area_m2
        land_cover = str(properties.get("monitored_land_cover") or "unknown")
        reliability = str(properties.get("reliability") or "unknown")
        transition = str(properties.get("transition") or "")
        if not transition or transition == "None":
            before = str(properties.get("before_state") or "unknown")
            after = str(properties.get("after_state") or "unknown")
            transition = f"{before}_to_{after}"

        by_land_cover[land_cover] = by_land_cover.get(land_cover, 0) + 1
        by_land_cover_area[land_cover] = by_land_cover_area.get(land_cover, 0.0) + area_m2
        by_reliability[reliability] = by_reliability.get(reliability, 0) + 1
        by_reliability_area[reliability] = by_reliability_area.get(reliability, 0.0) + area_m2
        by_transition_area[transition] = by_transition_area.get(transition, 0.0) + area_m2

    aoi_area_m2 = 400_000_000.0
    no_change_area_m2 = max(aoi_area_m2 - changed_area_m2, 0)
    top_transition = max(by_transition_area.items(), key=lambda item: item[1], default=(None, 0.0))

    summary = {
        "total": published_count,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "aoi_area_m2": aoi_area_m2,
        "total_area_m2": changed_area_m2,
        "changed_area_m2": changed_area_m2,
        "no_change_area_m2": no_change_area_m2,
        "changed_percent": changed_area_m2 / aoi_area_m2 * 100,
        "no_change_percent": no_change_area_m2 / aoi_area_m2 * 100,
        "by_monitored_land_cover": by_land_cover,
        "by_monitored_land_cover_area_m2": by_land_cover_area,
        "by_reliability": by_reliability,
        "by_reliability_area_m2": by_reliability_area,
        "top_likely_change_type": top_transition[0],
        "top_likely_change_area_m2": top_transition[1],
    }
    (WEB_DEMO / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def export_validation_summary() -> None:
    """Convert the validation report into frontend-friendly summary JSON."""
    report_path = OUTPUTS / "unsupervised_validation_report.json"
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = report["best_experiment"]
    summary = {
        "phase": "Phase 20B",
        "validation_type": report["validation_type"],
        "ground_truth_available": report["ground_truth_available"],
        "accuracy_claim": report["accuracy_claim"],
        "score_label": "Best unsupervised silhouette",
        "score_scope_note": (
            "This is an internal cluster-coherence score from the unsupervised "
            "validation comparison. It is not a U-Net confidence score and it "
            "changes only when pipelines/09_unsupervised_validation.py is rerun."
        ),
        "source_report": "data/outputs/unsupervised_validation_report.json",
        "silhouette_score": report["silhouette_score"],
        "best_experiment": best,
        "cluster_profiles": report["cluster_profiles"],
        "monitored_group_cluster_alignment": report["monitored_group_cluster_alignment"],
        "comparison_scope": report["comparison_scope"],
        "reference_label_workflow": {
            "status": "planned",
            "planned_steps": [
                "Create stratified validation points inside the AOI.",
                "Label points from field survey or high-resolution image interpretation.",
                "Compare labels with baseline, after, and change outputs.",
                "Report confusion matrix, user accuracy, producer accuracy, and F1 score.",
            ],
        },
    }
    (WEB_DEMO / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def export_unet_reliability_summary() -> None:
    """Convert the Phase 37 reliability report into frontend-friendly JSON."""
    report_path = UNET_RELIABILITY_REPORT
    if not report_path.exists():
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = sorted(
        report.get("records", []),
        key=lambda record: record.get("candidate_review_fraction_of_valid", 0),
        reverse=True,
    )
    summary = {
        "phase": "Phase 37",
        "purpose": report["purpose"],
        "accuracy_claim": "Weak-evidence compatibility only; not field accuracy.",
        "weak_sources_used": report["weak_sources_used"],
        "thresholds": report["thresholds"],
        "summary": report["summary"],
        "limitations": report["limitations"],
        "top_review_records": records[:4],
    }
    (WEB_DEMO / "unet_reliability_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def latest_2026_benchmark_record() -> dict | None:
    """Load the frozen 2026 test record without touching training manifests."""
    if not BENCHMARK_2026_MANIFEST.exists():
        return None
    manifest = json.loads(BENCHMARK_2026_MANIFEST.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest.get("records", [])
        if record.get("split") == "test"
        and str(record.get("training_tag", "")).startswith("test_2026")
    ]
    if not records:
        return None
    return sorted(records, key=lambda record: record["training_tag"])[-1]


def export_2026_benchmark_layers() -> list[dict]:
    """Export frozen held-out 2026 benchmark rasters for transparent review."""
    record = latest_2026_benchmark_record()
    if not record:
        return []

    tag = record["training_tag"]
    dominant = UNET_FULL_AOI_BASELINE / f"{tag}_unet_full_aoi_dominant_class.tif"
    confidence = UNET_FULL_AOI_BASELINE / f"{tag}_unet_full_aoi_confidence.tif"
    entropy = UNET_FULL_AOI_BASELINE / f"{tag}_unet_full_aoi_entropy.tif"
    if not (dominant.exists() and confidence.exists() and entropy.exists()):
        return []

    pretty_tag = tag.replace("test_2026_", "2026 / ").replace("_", " / ")
    layers = [
        export_categorical_layer(
            layer_id="test_2026_unet_dominant_class",
            title="Frozen U-Net dominant class",
            group="2026 frozen benchmark",
            path=dominant,
            class_palette={
                1: (214, 70, 23),
                2: (47, 137, 86),
                3: (181, 119, 27),
                4: (43, 139, 214),
                5: (112, 76, 219),
            },
            description=(
                f"Held-out frozen U-Net prediction for {pretty_tag}. "
                "This 2026 relaxed benchmark was not used for model training or tuning."
            ),
            opacity=0.62,
        ),
        export_single_band(
            SingleBandLayer(
                id="test_2026_unet_confidence",
                title="Frozen U-Net confidence",
                group="2026 frozen benchmark",
                path=confidence,
                value_range=(0, 1),
                palette=((247, 247, 247), (254, 224, 144), (253, 174, 97), (165, 0, 38)),
                description=(
                    f"Maximum class probability for held-out {pretty_tag}. "
                    "Low values should be reviewed before using the prediction."
                ),
                opacity=0.66,
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="test_2026_unet_entropy",
                title="Frozen U-Net entropy",
                group="2026 frozen benchmark",
                path=entropy,
                value_range=(0, 1),
                palette=((26, 150, 65), (255, 255, 191), (253, 174, 97), (215, 25, 28)),
                description=(
                    f"Prediction uncertainty for held-out {pretty_tag}. "
                    "Higher entropy means the model is less certain."
                ),
                opacity=0.66,
            )
        ),
    ]

    review_layer = export_2026_benchmark_review_layer(tag)
    if review_layer:
        layers.append(review_layer)
    return layers


def export_2026_benchmark_review_layer(tag: str) -> dict | None:
    """Export 2026 low-confidence, high-entropy, and weak-disagreement zones."""
    confidence = UNET_FULL_AOI_BASELINE / f"{tag}_unet_full_aoi_confidence.tif"
    entropy = UNET_FULL_AOI_BASELINE / f"{tag}_unet_full_aoi_entropy.tif"
    disagreement = EXTERNAL_SOURCES / "weak_source_disagreement_mask_test_2026.tif"
    if not (confidence.exists() and entropy.exists() and disagreement.exists()):
        return None

    confidence_array, grid = read_band(confidence)
    entropy_array, _entropy_grid = read_band(entropy)
    disagreement_array, _disagreement_grid = read_band(disagreement)
    low_confidence = confidence_array.filled(-9999) < 0.6
    high_entropy = entropy_array.filled(-9999) >= 0.65
    weak_disagreement = disagreement_array.filled(0) > 0

    classes = np.zeros(confidence_array.shape, dtype="uint8")
    classes[low_confidence] = 1
    classes[high_entropy] = 2
    classes[weak_disagreement] = 3
    classes[np.ma.getmaskarray(confidence_array)] = 0

    rgb = np.zeros((*classes.shape, 3), dtype="uint8")
    rgb[classes == 1] = (245, 158, 11)
    rgb[classes == 2] = (220, 38, 38)
    rgb[classes == 3] = (124, 58, 237)
    alpha = np.where(classes == 0, 0, 185).astype("uint8")
    alpha = apply_aoi_alpha(alpha, grid)

    layer_id = "test_2026_unet_review_zones"
    assets = display_asset_record(layer_id, rgb, alpha, grid)
    pretty_tag = tag.replace("test_2026_", "2026 / ").replace("_", " / ")
    return {
        "id": layer_id,
        "title": "Frozen U-Net review zones",
        "group": "2026 frozen benchmark",
        "kind": "raster_overlay",
        "opacity": 0.72,
        "description": (
            f"Held-out 2026 review zones for {pretty_tag}: orange is low model confidence, "
            "red is high entropy, and purple is weak-source disagreement. "
            "This layer supports error review, not accuracy claims."
        ),
        "source": (
            "data/interim/unet_full_aoi test_2026 confidence/entropy plus "
            "data/interim/external_sources/weak_source_disagreement_mask_test_2026.tif"
        ),
        "satelliteDerived": True,
        **assets,
    }


def export_2026_benchmark_summary() -> None:
    """Publish the frozen 2026 benchmark metrics for the WebGIS sidebar."""
    if not BENCHMARK_2026_REPORT.exists():
        return

    report = json.loads(BENCHMARK_2026_REPORT.read_text(encoding="utf-8"))
    records = report.get("records", [])
    record = records[0] if records else {}
    summary = {
        "phase": "Phase 51",
        "title": "2026 frozen U-Net relaxed benchmark",
        "status": "relaxed_proxy_benchmark",
        "accuracy_claim": "Weak-source compatibility only; not field accuracy.",
        "benchmark_caveat": report.get("benchmark_caveat", ""),
        "weak_sources_used": report.get("weak_sources_used", []),
        "thresholds": report.get("thresholds", {}),
        "summary": report.get("summary", {}),
        "record": record,
        "limitations": report.get("limitations", []),
        "non_cheating_protocol": report.get("non_cheating_protocol", {}),
    }
    (WEB_DEMO / "benchmark_2026_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def export_geospatial_alignment_summary() -> None:
    """Publish master-grid validation status for the WebGIS sidebar."""
    if not GEOSPATIAL_ALIGNMENT_REPORT.exists():
        return

    report = json.loads(GEOSPATIAL_ALIGNMENT_REPORT.read_text(encoding="utf-8"))
    summary = {
        "phase": report.get("phase"),
        "status": report.get("status"),
        "master_grid": report.get("master_grid", {}),
        "checked_raster_count": report.get("checked_raster_count", 0),
        "checked_vector_count": report.get("checked_vector_count", 0),
        "raster_mismatch_count": report.get("raster_mismatch_count", 0),
        "vector_mismatch_count": report.get("vector_mismatch_count", 0),
        "raster_family_counts": report.get("raster_family_counts", {}),
        "skipped_raw_source_rasters": report.get("skipped_raw_source_rasters", []),
        "interpretation": (
            "All downstream analysis rasters and WebGIS vectors are aligned to "
            "the same 10 m EPSG:32735 master grid. Raw provider exports are kept "
            "only as provenance and are harmonized before weak-label fusion."
        ),
    }
    (WEB_DEMO / "geospatial_alignment_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def export_coregistration_qa_summary() -> None:
    """Publish image-content registration diagnostics for the WebGIS sidebar."""
    if not COREGISTRATION_QA_REPORT.exists():
        return

    report = json.loads(COREGISTRATION_QA_REPORT.read_text(encoding="utf-8"))
    records = report.get("records", [])
    summary = {
        "phase": report.get("phase"),
        "status": report.get("status"),
        "method": report.get("method"),
        "reference": report.get("reference"),
        "master_grid": report.get("master_grid", {}),
        "checked_target_count": len(records),
        "grid_failure_count": report.get("grid_failure_count", 0),
        "high_shift_warning_count": report.get("high_shift_warning_count", 0),
        "max_diagnostic_shift_m": max(
            [float(record.get("diagnostic_shift_m") or 0) for record in records],
            default=0.0,
        ),
        "records": records[:6],
        "limitations": report.get("limitations", []),
        "recommended_production_method": report.get("recommended_production_method", []),
    }
    (WEB_DEMO / "coregistration_qa_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def export_s2_false_color() -> dict:
    nir, grid = read_band(PROCESSED / "s2_20250220_nir.tif")
    red, _red_grid = read_band(PROCESSED / "s2_20250220_red.tif")
    green, _green_grid = read_band(PROCESSED / "s2_20250220_green.tif")
    rgb = np.dstack(
        [
            normalize(nir) * 255,
            normalize(red) * 255,
            normalize(green) * 255,
        ]
    ).astype("uint8")
    alpha = apply_aoi_alpha(alpha_from_mask(nir, 0.86), grid)
    layer_id = "s2_false_color_20250220"
    assets = display_asset_record(layer_id, rgb, alpha, grid)
    return {
        "id": layer_id,
        "title": "Sentinel-2 false color",
        "group": "Sentinel-2",
        "kind": "raster_overlay",
        "opacity": 0.86,
        "description": "AOI-clipped NIR/red/green composite from the clean 2025-02-20 Sentinel-2 scene.",
        "source": "data/processed/s2_20250220_nir.tif, red.tif, green.tif",
        "satelliteDerived": True,
        **assets,
    }


def main() -> None:
    """Export all hosted raster overlays and their manifest."""
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_TILE_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DEMO / "aoi.geojson").write_text(
        json.dumps(master_grid_aoi_geojson(), indent=2),
        encoding="utf-8",
    )

    export_validation_summary()
    export_unet_reliability_summary()
    export_2026_benchmark_summary()
    export_geospatial_alignment_summary()
    export_coregistration_qa_summary()
    export_dashboard_summary()

    layers = [
        export_s2_false_color(),
        export_single_band(
            SingleBandLayer(
                id="s2_ndvi_20250220",
                title="NDVI vegetation index",
                group="Sentinel-2 indices",
                path=PROCESSED / "s2_20250220_ndvi.tif",
                value_range=(-0.2, 0.8),
                palette=((116, 75, 42), (238, 226, 143), (35, 132, 67), (0, 68, 27)),
                description="Vegetation greenness index derived from Sentinel-2 red and NIR bands.",
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="s2_ndwi_20250220",
                title="NDWI moisture index",
                group="Sentinel-2 indices",
                path=PROCESSED / "s2_20250220_ndwi.tif",
                value_range=(-0.8, 0.4),
                palette=((121, 85, 72), (232, 226, 206), (93, 173, 226), (21, 67, 96)),
                description="Water and surface moisture index derived from Sentinel-2 green and NIR bands.",
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="s2_ndbi_20250220",
                title="NDBI built-up index",
                group="Sentinel-2 indices",
                path=PROCESSED / "s2_20250220_ndbi.tif",
                value_range=(-0.5, 0.5),
                palette=((35, 132, 67), (244, 241, 222), (214, 96, 77), (123, 50, 64)),
                description="Built-up signal index derived from Sentinel-2 SWIR and NIR bands.",
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="s1_vv_20250221",
                title="Sentinel-1 VV backscatter",
                group="Sentinel-1 radar",
                path=PROCESSED / "s1_20250221_vv.tif",
                value_range=None,
                palette=((38, 50, 56), (84, 110, 122), (176, 190, 197), (255, 255, 255)),
                description="AOI-clipped Sentinel-1 RTC VV backscatter from 2025-02-21.",
                opacity=0.68,
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="s1_vh_20250221",
                title="Sentinel-1 VH backscatter",
                group="Sentinel-1 radar",
                path=PROCESSED / "s1_20250221_vh.tif",
                value_range=None,
                palette=((28, 35, 42), (74, 85, 104), (129, 140, 248), (224, 231, 255)),
                description="AOI-clipped Sentinel-1 RTC VH backscatter from 2025-02-21.",
                opacity=0.68,
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="s1_ratio_20250221",
                title="Sentinel-1 VV/VH ratio",
                group="Sentinel-1 radar",
                path=PROCESSED / "s1_20250221_vv_vh_ratio.tif",
                value_range=(0, 12),
                palette=((49, 54, 149), (69, 117, 180), (254, 224, 144), (165, 0, 38)),
                description="Radar structure feature used by the change detector.",
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="delta_ndvi_latest",
                title="Delta NDVI",
                group="Change rasters",
                path=OUTPUTS
                / "monitoring_20250220_20250221__to__monitoring_20250421_20250419_delta_ndvi.tif",
                value_range=(-0.5, 0.5),
                palette=((118, 42, 131), (247, 247, 247), (27, 120, 55)),
                description="Vegetation index change between the February and April monitoring stacks.",
            )
        ),
        export_single_band(
            SingleBandLayer(
                id="change_confidence_latest",
                title="Change confidence",
                group="Change rasters",
                path=OUTPUTS
                / "monitoring_20250220_20250221__to__monitoring_20250421_20250419_confidence.tif",
                value_range=(0, 1),
                palette=((240, 240, 240), (253, 174, 97), (215, 48, 39), (103, 0, 31)),
                description="Pixel-level confidence surface for the latest monitored change comparison.",
            )
        ),
    ]
    cluster_layer = export_cluster_layer()
    if cluster_layer:
        layers.append(cluster_layer)
    layers.extend(export_unet_full_aoi_layers())
    review_layer = export_unet_review_zone_layer()
    if review_layer:
        layers.append(review_layer)
    layers.extend(export_2026_benchmark_layers())

    manifest = {
        "generated_from": "AOI-clipped Sentinel-1/2 and change rasters",
        "grid_alignment": {
            "status": "validated",
            "reference": "configs/master_grid.yaml",
            "checks": ["crs", "transform", "width", "height", "bounds", "hosted_aoi"],
            "display_aoi": "web/public/demo/aoi.geojson",
        },
        "display_alignment": {
            "status": "tiled",
            "display_crs": DISPLAY_CRS,
            "analysis_crs": expected_master_grid()["crs"],
            "tile_scheme": "XYZ",
            "tile_size": TILE_SIZE,
            "minzoom": TILE_MIN_ZOOM,
            "maxzoom": TILE_MAX_ZOOM,
            "reason": (
                "MapLibre raster overlays are served as EPSG:3857 XYZ tiles so they line up "
                "with the Web Mercator basemap instead of stretching one UTM image."
            ),
        },
        "note": (
            "Raster overlays are Web Mercator XYZ display tiles with an AOI-polygon alpha mask; "
            "source GeoTIFFs remain in data/processed and data/outputs. "
            "All exported layers are first validated against the EPSG:32735 "
            "project master grid, then reprojected to EPSG:3857 tiles for browser display."
        ),
        "layers": layers,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(layers)} hosted raster layers to {RASTER_DIR}")


if __name__ == "__main__":
    main()
