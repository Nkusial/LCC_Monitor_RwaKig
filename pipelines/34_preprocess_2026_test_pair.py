"""Preprocess the selected 2026 Sentinel-1/2 pair as held-out test data only.

This stage deliberately does not modify training catalogs, patch manifests,
class weights, thresholds, or pseudo-label refinement inputs. Its only purpose
is to create a frozen-model evaluation stack under a `test_2026_*` tag.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import planetary_computer
from pystac_client import Client

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PACKAGE = ROOT / "data" / "outputs" / "heldout_benchmark_2026_package.json"
TEST_SCENE_CATALOG = ROOT / "data" / "catalog" / "test_scenes_2026.geojson"
TEST_SUMMARY = ROOT / "data" / "catalog" / "test_preprocessing_summary_2026.json"

PREPROCESS_SPEC = importlib.util.spec_from_file_location(
    "phase4_preprocess", ROOT / "pipelines" / "03_preprocess.py"
)
phase4_preprocess = importlib.util.module_from_spec(PREPROCESS_SPEC)
assert PREPROCESS_SPEC.loader is not None
PREPROCESS_SPEC.loader.exec_module(phase4_preprocess)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    """Return project-relative paths when possible, including Windows relative inputs."""
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def selected_candidate(package_path: Path) -> dict[str, Any]:
    package = read_json(package_path)
    candidate = package.get("selected_candidate")
    if not candidate:
        raise RuntimeError("No selected 2026 benchmark candidate is available.")
    if package.get("no_leakage_audit", {}).get("status") != "pass":
        raise RuntimeError("No-leakage audit must pass before preprocessing held-out test data.")
    return candidate


def test_tag(candidate: dict[str, Any]) -> str:
    optical_date = str(candidate["date"]).replace("-", "")
    radar_date = str(candidate["sentinel1_match"]["date"]).replace("-", "")
    return f"test_2026_{optical_date}_{radar_date}"


def build_test_pair(candidate: dict[str, Any]) -> dict[str, Any]:
    scene_ids = [tile["scene_id"] for tile in candidate["tiles"].values()]
    tile_cloud = {
        tile_name: float(tile["cloud_cover_percent"])
        for tile_name, tile in candidate["tiles"].items()
    }
    return {
        "year": 2026,
        "split": "test",
        "cloud_tier": "relaxed_candidate",
        "selection_cloud_threshold": None,
        "output_tag": test_tag(candidate),
        "optical": {
            "collection": "sentinel-2-l2a",
            "date": candidate["date"],
            "tiles": sorted(candidate["tiles"]),
            "scene_ids": scene_ids,
            "bands": {
                "blue": "B02",
                "green": "B03",
                "red": "B04",
                "nir": "B08",
                "swir1": "B11",
            },
            "tile_cloud_cover_percent": tile_cloud,
            "cloud_cover_percent": float(candidate["average_cloud_cover_percent"]),
            "max_tile_cloud_cover_percent": float(candidate["max_tile_cloud_cover_percent"]),
        },
        "radar": {
            "collection": "sentinel-1-rtc",
            "date": candidate["sentinel1_match"]["date"],
            "scene_id": candidate["sentinel1_match"]["scene_id"],
            "orbit": candidate["sentinel1_match"]["orbit"],
            "polarizations": candidate["sentinel1_match"]["polarizations"],
            "time_gap_hours": float(candidate["sentinel1_match"]["time_gap_hours"]),
        },
    }


def item_to_feature(item: Any) -> dict[str, Any]:
    signed = planetary_computer.sign(item)
    return {
        "type": "Feature",
        "id": signed.id,
        "geometry": signed.geometry,
        "properties": {
            "id": signed.id,
            "collection": signed.collection_id,
            "datetime": signed.datetime.isoformat() if signed.datetime else signed.properties.get("datetime"),
            "cloud_cover": signed.properties.get("eo:cloud_cover"),
            "assets": {name: asset.href for name, asset in signed.assets.items()},
            "stac_properties": signed.properties,
        },
    }


def write_test_scene_catalog(config: dict[str, Any], pair: dict[str, Any], output_path: Path) -> Path:
    """Fetch only selected Sentinel-2 items and write the catalog expected by Phase 4."""
    aoi = gpd.read_file(config["project"]["aoi_path"]).to_crs("EPSG:4326")
    bbox = aoi.total_bounds.tolist()
    client = Client.open(config["sentinel"]["stac_api_url"])
    date = pair["optical"]["date"]
    items = list(
        client.search(
            collections=[pair["optical"]["collection"]],
            bbox=bbox,
            datetime=f"{date}/{date}",
            max_items=25,
        ).items()
    )
    selected_ids = set(pair["optical"]["scene_ids"])
    features = [item_to_feature(item) for item in items if item.id in selected_ids]
    found_ids = {feature["id"] for feature in features}
    missing = sorted(selected_ids - found_ids)
    if missing:
        raise RuntimeError(f"Selected 2026 Sentinel-2 items were not found in STAC: {missing}")

    output = {
        "type": "FeatureCollection",
        "metadata": {
            "purpose": "held_out_2026_test_only_scene_catalog",
            "leakage_policy": "Do not add these scenes to training, validation, patch balancing, or pseudo-label refinement.",
            "date": date,
            "scene_ids": sorted(selected_ids),
        },
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def expected_outputs(tag: str) -> list[str]:
    suffixes = [
        "s2_green",
        "s2_red",
        "s2_nir",
        "s2_swir1",
        "s2_ndvi",
        "s2_ndwi",
        "s2_ndbi",
        "s1_vv",
        "s1_vh",
        "s1_vv_vh_ratio",
    ]
    return [f"data/processed/{tag}_{suffix}.tif" for suffix in suffixes]


def preprocess_2026_test_pair(
    config: dict[str, Any],
    package_path: Path,
    mode: str,
) -> Path:
    ensure_output_dirs(config)
    candidate = selected_candidate(package_path)
    pair = build_test_pair(candidate)
    tag = pair["output_tag"]
    catalog_path = write_test_scene_catalog(config, pair, TEST_SCENE_CATALOG)
    manifest_path = phase4_preprocess.preprocess_pair(
        config,
        pair=pair,
        output_tag=tag,
        metadata_name=f"{tag}_manifest.json",
        catalog_name=catalog_path.name,
        metadata_only=(mode == "metadata"),
        mode=mode,
    )
    expected = expected_outputs(tag)
    summary = {
        "phase": "phase_50_heldout_2026_test_preprocessing",
        "status": "metadata_ready" if mode == "metadata" else "processed",
        "mode": mode,
        "output_tag": tag,
        "split": "test",
        "scene_catalog": display_path(catalog_path),
        "manifest": display_path(manifest_path),
        "selected_candidate": candidate,
        "expected_outputs": expected,
        "missing_outputs": [
            path for path in expected if not (ROOT / path).exists()
        ],
        "non_cheating_protocol": {
            "allowed": "Use these rasters only for frozen-model 2026 proxy evaluation.",
            "forbidden": [
                "Do not add this tag to train or validation patch manifests.",
                "Do not tune model architecture, class weights, thresholds, or pseudo-label filters with 2026 results.",
                "Do not claim field accuracy from weak-source agreement.",
            ],
        },
    }
    TEST_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    TEST_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TEST_SUMMARY


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess selected 2026 pair as held-out test data only.")
    parser.add_argument("--package", type=Path, default=BENCHMARK_PACKAGE)
    parser.add_argument("--mode", choices=["metadata", "optical", "radar", "all"], default="metadata")
    args = parser.parse_args()

    summary_path = preprocess_2026_test_pair(load_config(), args.package, args.mode)
    summary = read_json(summary_path)
    print(f"2026 test preprocessing summary: {summary_path}")
    print(json.dumps({"status": summary["status"], "output_tag": summary["output_tag"], "missing_outputs": summary["missing_outputs"]}, indent=2))


if __name__ == "__main__":
    main()
