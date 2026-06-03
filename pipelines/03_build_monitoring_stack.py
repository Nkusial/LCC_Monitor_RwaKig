"""Build the multi-date Sentinel-2/Sentinel-1 monitoring stack.

This stage searches for low-cloud Sentinel-2 tile pairs, matches nearby
Sentinel-1 RTC acquisitions, and can call the Phase 4 preprocessing code for
each selected monitoring date.
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import planetary_computer
from pystac_client import Client

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


PREPROCESS_SPEC = importlib.util.spec_from_file_location(
    "phase4_preprocess", Path("pipelines/03_preprocess.py")
)
# The file name starts with a number, so importlib is used instead of a normal
# Python import. This keeps the phase numbering visible in the repository.
phase4_preprocess = importlib.util.module_from_spec(PREPROCESS_SPEC)
assert PREPROCESS_SPEC.loader is not None
PREPROCESS_SPEC.loader.exec_module(phase4_preprocess)


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def date_key(item: Any) -> str:
    return parse_datetime(item.properties["datetime"]).date().isoformat()


def item_to_feature(item: Any) -> dict[str, Any]:
    data = item.to_dict()
    properties = data.get("properties", {})
    return {
        "type": "Feature",
        "id": item.id,
        "geometry": data.get("geometry"),
        "properties": {
            "id": item.id,
            "collection": item.collection_id,
            "datetime": properties.get("datetime"),
            "cloud_cover": properties.get("eo:cloud_cover"),
            "assets": {key: asset.get("href") for key, asset in data.get("assets", {}).items()},
            "stac_properties": properties,
        },
    }


def write_feature_collection(features: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def search_sentinel2_groups(config: dict[str, Any], aoi: gpd.GeoDataFrame) -> list[list[Any]]:
    """Return complete same-date Sentinel-2 tile groups for the AOI."""
    monitoring = config["preprocessing"]["monitoring_stack"]
    client = Client.open(config["sentinel"]["stac_api_url"])
    search = client.search(
        collections=[config["sentinel"]["collections"]["sentinel2"]],
        bbox=aoi.to_crs("EPSG:4326").total_bounds.tolist(),
        datetime=f"{monitoring['date_range']['start']}/{monitoring['date_range']['end']}",
        query={"eo:cloud_cover": {"lt": monitoring["max_cloud_cover"]}},
        max_items=200,
    )

    by_date: dict[str, list[Any]] = defaultdict(list)
    for item in search.items():
        by_date[date_key(item)].append(item)

    required_tiles = set(monitoring["required_tiles"])
    complete_groups = []
    for items in by_date.values():
        by_tile = {}
        for item in items:
            tile = item.properties.get("s2:mgrs_tile")
            if tile not in required_tiles:
                continue
            current = by_tile.get(tile)
            # Keep the cleanest item for each required tile on the same date.
            if current is None or item.properties.get("eo:cloud_cover", 100) < current.properties.get(
                "eo:cloud_cover", 100
            ):
                by_tile[tile] = item

        if required_tiles.issubset(by_tile):
            complete_groups.append([by_tile[tile] for tile in sorted(required_tiles)])

    complete_groups.sort(
        key=lambda group: (
            sum(item.properties.get("eo:cloud_cover", 100) for item in group) / len(group),
            date_key(group[0]),
        )
    )
    return complete_groups


def find_matching_sentinel1(config: dict[str, Any], aoi: gpd.GeoDataFrame, optical_date: str) -> Any | None:
    """Find the closest VV/VH Sentinel-1 RTC item around an optical date."""
    monitoring = config["preprocessing"]["monitoring_stack"]
    client = Client.open(config["sentinel"]["stac_api_url"])
    center = datetime.fromisoformat(optical_date).replace(tzinfo=timezone.utc)
    days = int(monitoring["sentinel1_search_days"])

    candidates = []
    offsets = sorted(range(-days, days + 1), key=lambda value: (abs(value), value))
    for offset in offsets:
        day = (center + timedelta(days=offset)).date().isoformat()
        search = client.search(
            collections=[config["sentinel"]["collections"]["sentinel1"]],
            bbox=aoi.to_crs("EPSG:4326").total_bounds.tolist(),
            datetime=f"{day}/{day}",
            max_items=20,
        )
        try:
            items = list(search.items())
        except Exception as exc:
            print(f"Skipping Sentinel-1 search day {day}: {exc}", flush=True)
            continue

        for item in items:
            props = item.properties
            polarizations = props.get("sar:polarizations", [])
            if not all(pol in polarizations for pol in ["VV", "VH"]):
                continue
            item_dt = parse_datetime(props["datetime"])
            gap_hours = abs((item_dt - center).total_seconds()) / 3600
            # Prefer the configured orbit, then the smallest optical/radar gap.
            orbit_penalty = 0 if props.get("sat:orbit_state") == monitoring["preferred_orbit"] else 1
            candidates.append((orbit_penalty, gap_hours, item))

    if not candidates:
        return None

    candidates.sort(key=lambda row: (row[0], row[1]))
    return planetary_computer.sign(candidates[0][2])


def build_monitoring_pairs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the catalog of optical/radar date pairs used downstream."""
    aoi = gpd.read_file(config["project"]["aoi_path"])
    monitoring = config["preprocessing"]["monitoring_stack"]
    s2_groups = search_sentinel2_groups(config, aoi)
    pairs = []
    s2_features = []

    for group in s2_groups:
        optical_date = date_key(group[0])
        s1_item = find_matching_sentinel1(config, aoi, optical_date)
        if s1_item is None:
            continue

        scene_ids = [item.id for item in group]
        tiles = [item.properties.get("s2:mgrs_tile") for item in group]
        cloud_values = [item.properties.get("eo:cloud_cover", 100) for item in group]
        radar_dt = parse_datetime(s1_item.properties["datetime"])
        optical_dt = datetime.fromisoformat(optical_date).replace(tzinfo=timezone.utc)

        pairs.append(
            {
                "output_tag": (
                    f"monitoring_{optical_date.replace('-', '')}_"
                    f"{radar_dt.date().isoformat().replace('-', '')}"
                ),
                "optical": {
                    "collection": config["sentinel"]["collections"]["sentinel2"],
                    "date": optical_date,
                    "tiles": tiles,
                    "scene_ids": scene_ids,
                    "cloud_cover_percent": round(sum(cloud_values) / len(cloud_values), 2),
                    "bands": config["preprocessing"]["selected_pair"]["optical"]["bands"],
                },
                "radar": {
                    "collection": config["sentinel"]["collections"]["sentinel1"],
                    "date": radar_dt.date().isoformat(),
                    "scene_id": s1_item.id,
                    "orbit": s1_item.properties.get("sat:orbit_state"),
                    "polarizations": s1_item.properties.get("sar:polarizations", []),
                    "time_gap_hours": round(abs((radar_dt - optical_dt).total_seconds()) / 3600, 1),
                    "assets": {
                        key: asset.href
                        for key, asset in s1_item.assets.items()
                        if key in {"vv", "vh"}
                    },
                },
            }
        )
        s2_features.extend(item_to_feature(item) for item in group)
        if len(pairs) >= int(monitoring["max_pairs"]):
            break

    return pairs, s2_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and preprocess a small Sentinel-1/2 monitoring stack."
    )
    parser.add_argument(
        "--mode",
        choices=["metadata", "optical", "radar", "all"],
        default="metadata",
        help="Use metadata to select pairs only, or process optical/radar/all rasters.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_output_dirs(config)
    catalog_dir = Path(config["paths"]["catalog"])
    pairs_path = catalog_dir / "monitoring_pairs.json"
    scenes_path = catalog_dir / "monitoring_scenes.geojson"

    if args.mode != "metadata" and pairs_path.exists() and scenes_path.exists():
        pairs = json.loads(pairs_path.read_text(encoding="utf-8"))["pairs"]
        print(f"Using {len(pairs)} existing monitoring pairs from {pairs_path}")
    else:
        pairs, s2_features = build_monitoring_pairs(config)
        if not pairs:
            raise RuntimeError("No complete Sentinel-2/Sentinel-1 monitoring pairs found.")

        write_feature_collection(s2_features, scenes_path)
        pairs_path.write_text(json.dumps({"pairs": pairs}, indent=2), encoding="utf-8")
        print(f"Wrote {len(pairs)} monitoring pairs to {pairs_path}")

    if args.mode == "metadata":
        for pair in pairs:
            print(
                f"{pair['output_tag']}: S2 {pair['optical']['date']} "
                f"cloud {pair['optical']['cloud_cover_percent']}%, "
                f"S1 {pair['radar']['date']} {pair['radar']['orbit']} "
                f"gap {pair['radar']['time_gap_hours']}h"
            )
        return

    for pair in pairs:
        print(f"Processing {pair['output_tag']}...", flush=True)
        phase4_preprocess.preprocess_pair(
            config,
            pair=pair,
            output_tag=pair["output_tag"],
            metadata_name=f"{pair['output_tag']}_manifest.json",
            catalog_name="monitoring_scenes.geojson",
            metadata_only=False,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
