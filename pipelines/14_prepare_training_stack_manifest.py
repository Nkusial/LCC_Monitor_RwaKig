"""Validate training-scene metadata before heavy multi-year raster preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


REQUIRED_OPTICAL_ASSETS = {"B03", "B04", "B08", "B11"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required catalog file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scene_index(feature_collection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        feature["id"]: feature
        for feature in feature_collection.get("features", [])
    }


def pair_readiness(pair: dict[str, Any], scenes_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return reproducibility checks for one selected Sentinel-1/2 pair."""
    missing_scenes = [
        scene_id for scene_id in pair["optical"]["scene_ids"] if scene_id not in scenes_by_id
    ]
    missing_assets: dict[str, list[str]] = {}
    for scene_id in pair["optical"]["scene_ids"]:
        feature = scenes_by_id.get(scene_id)
        if feature is None:
            continue
        assets = set(feature.get("properties", {}).get("assets", {}))
        missing = sorted(REQUIRED_OPTICAL_ASSETS - assets)
        if missing:
            missing_assets[scene_id] = missing

    return {
        "output_tag": pair["output_tag"],
        "year": pair["year"],
        "split": pair["split"],
        "optical_date": pair["optical"]["date"],
        "radar_date": pair["radar"]["date"],
        "max_tile_cloud_cover_percent": pair["optical"]["max_tile_cloud_cover_percent"],
        "radar_time_gap_hours": pair["radar"]["time_gap_hours"],
        "missing_scenes": missing_scenes,
        "missing_assets": missing_assets,
        "ready_for_preprocessing": not missing_scenes and not missing_assets,
    }


def prepare_training_stack_manifest(config: dict[str, Any]) -> Path:
    """Write a manifest that gates future heavy raster preprocessing."""
    ensure_output_dirs(config)
    catalog_dir = Path(config["paths"]["catalog"])
    pairs_path = catalog_dir / "training_scene_pairs_2023_2026.json"
    scenes_path = catalog_dir / "training_scenes_2023_2026.geojson"
    pairs_catalog = read_json(pairs_path)
    scenes_catalog = read_json(scenes_path)
    scenes_by_id = scene_index(scenes_catalog)

    pair_records = [
        pair_readiness(pair, scenes_by_id)
        for pair in pairs_catalog.get("pairs", [])
    ]
    ready_count = sum(1 for record in pair_records if record["ready_for_preprocessing"])
    split_counts: dict[str, int] = {}
    for record in pair_records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1

    manifest = {
        "phase": "phase_29_training_stack_manifest",
        "purpose": "Preflight selected multi-year Sentinel-1/2 pairs before AOI-windowed raster reads.",
        "source_pairs": str(pairs_path).replace("\\", "/"),
        "source_scenes": str(scenes_path).replace("\\", "/"),
        "required_optical_assets": sorted(REQUIRED_OPTICAL_ASSETS),
        "pair_count": len(pair_records),
        "ready_pair_count": ready_count,
        "split_counts": split_counts,
        "status": "ok" if ready_count == len(pair_records) and pair_records else "needs_attention",
        "pairs": pair_records,
        "next_step": (
            "Run AOI-windowed preprocessing for ready pairs, then generate multi-year "
            "feature/weak-label patches using the same master-grid contract."
        ),
    }
    output_path = catalog_dir / "training_stack_manifest_2023_2026.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-year training stack manifest.")
    parser.parse_args()
    path = prepare_training_stack_manifest(load_config())
    print(f"Training stack manifest: {path}")


if __name__ == "__main__":
    main()
