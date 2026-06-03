"""Diagnose 2026 Sentinel-1/2 candidates for a held-out test benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
from pystac_client import Client

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def date_key(item: Any) -> str:
    return parse_datetime(item.properties["datetime"]).date().isoformat()


def best_sentinel1_match(
    client: Client,
    collection: str,
    bbox: list[float],
    optical_date: str,
    search_days: int,
    max_gap_hours: float,
    preferred_orbit: str,
) -> dict[str, Any] | None:
    """Return the closest VV/VH Sentinel-1 RTC acquisition near an optical date."""
    center = datetime.fromisoformat(optical_date).replace(tzinfo=timezone.utc)
    candidates = []

    for offset in sorted(range(-search_days, search_days + 1), key=lambda value: (abs(value), value)):
        day = (center + timedelta(days=offset)).date().isoformat()
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{day}/{day}",
            max_items=50,
            method="POST",
        )
        for item in search.items():
            props = item.properties
            polarizations = props.get("sar:polarizations", [])
            if not all(pol in polarizations for pol in ["VV", "VH"]):
                continue
            gap_hours = abs((parse_datetime(props["datetime"]) - center).total_seconds()) / 3600
            if gap_hours > max_gap_hours:
                continue
            orbit_penalty = 0 if props.get("sat:orbit_state") == preferred_orbit else 1
            candidates.append((orbit_penalty, gap_hours, item))

    if not candidates:
        return None

    candidates.sort(key=lambda row: (row[0], row[1]))
    _, gap_hours, item = candidates[0]
    return {
        "collection": collection,
        "date": parse_datetime(item.properties["datetime"]).date().isoformat(),
        "scene_id": item.id,
        "orbit": item.properties.get("sat:orbit_state"),
        "polarizations": item.properties.get("sar:polarizations", []),
        "time_gap_hours": round(gap_hours, 1),
    }


def discover_2026_test_benchmark(config: dict[str, Any]) -> Path:
    """Write exact-cloud 2026 candidate diagnostics for final test selection."""
    ensure_output_dirs(config)
    settings = config["training_scene_discovery"]
    aoi = gpd.read_file(config["project"]["aoi_path"]).to_crs("EPSG:4326")
    bbox = aoi.total_bounds.tolist()
    client = Client.open(config["sentinel"]["stac_api_url"])
    required_tiles = set(settings["required_tiles"])
    start = max("2026-01-01", settings["date_range"]["start"])
    end = settings["date_range"]["end"]

    search = client.search(
        collections=[config["sentinel"]["collections"]["sentinel2"]],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=1000,
        method="POST",
    )

    by_date: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    best_individual_items = []
    for item in search.items():
        tile = item.properties.get("s2:mgrs_tile")
        if tile not in required_tiles:
            continue
        cloud_cover = round(float(item.properties.get("eo:cloud_cover", 100.0)), 2)
        day = date_key(item)
        record = {
            "date": day,
            "tile": tile,
            "scene_id": item.id,
            "cloud_cover_percent": cloud_cover,
            "datetime": item.properties.get("datetime"),
        }
        best_individual_items.append(record)
        current = by_date[day].get(tile)
        if current is None or cloud_cover < current["cloud_cover_percent"]:
            by_date[day][tile] = record

    complete_same_date_groups = []
    incomplete_dates = []
    for day, tile_records in by_date.items():
        if required_tiles.issubset(tile_records):
            clouds = [tile_records[tile]["cloud_cover_percent"] for tile in sorted(required_tiles)]
            s1 = best_sentinel1_match(
                client=client,
                collection=config["sentinel"]["collections"]["sentinel1"],
                bbox=bbox,
                optical_date=day,
                search_days=int(settings["sentinel1_search_days"]),
                max_gap_hours=float(settings["max_radar_time_gap_hours"]),
                preferred_orbit=str(settings["preferred_orbit"]),
            )
            max_cloud = round(max(clouds), 2)
            meets_preferred = max_cloud <= float(settings["preferred_max_cloud_cover"])
            meets_fallback = max_cloud <= float(settings["fallback_max_cloud_cover"])
            complete_same_date_groups.append(
                {
                    "date": day,
                    "average_cloud_cover_percent": round(sum(clouds) / len(clouds), 2),
                    "max_tile_cloud_cover_percent": max_cloud,
                    "tiles": {tile: tile_records[tile] for tile in sorted(required_tiles)},
                    "sentinel1_match": s1,
                    "has_sentinel1_match": s1 is not None,
                    "meets_preferred_cloud_policy": meets_preferred,
                    "meets_fallback_cloud_policy": meets_fallback,
                    "usable_as_strict_test_pair": s1 is not None and meets_fallback,
                    "usable_as_relaxed_candidate": s1 is not None,
                }
            )
        else:
            incomplete_dates.append(
                {
                    "date": day,
                    "available_tiles": sorted(tile_records),
                    "missing_tiles": sorted(required_tiles.difference(tile_records)),
                    "best_available_tile_cloud_percent": min(
                        record["cloud_cover_percent"] for record in tile_records.values()
                    ),
                }
            )

    complete_same_date_groups.sort(
        key=lambda row: (
            row["max_tile_cloud_cover_percent"],
            row["average_cloud_cover_percent"],
            row["date"],
        )
    )
    incomplete_dates.sort(key=lambda row: (row["best_available_tile_cloud_percent"], row["date"]))
    best_individual_items.sort(key=lambda row: (row["cloud_cover_percent"], row["date"], row["tile"]))

    weak_sources = config.get("weak_labels", {})
    weak_config_path = Path(config["project"]["weak_labels_path"])
    if weak_config_path.exists():
        import yaml

        weak_sources = yaml.safe_load(weak_config_path.read_text(encoding="utf-8"))

    report = {
        "phase": "phase_47_2026_test_benchmark_discovery",
        "purpose": "Find best 2026 cloud-free Sentinel-2/Sentinel-1 candidates for a clean held-out benchmark.",
        "date_range": {"start": start, "end": end},
        "required_tiles": sorted(required_tiles),
        "preferred_max_cloud_cover": settings["preferred_max_cloud_cover"],
        "fallback_max_cloud_cover": settings["fallback_max_cloud_cover"],
        "complete_same_date_group_count": len(complete_same_date_groups),
        "strict_test_pair_count": sum(1 for row in complete_same_date_groups if row["usable_as_strict_test_pair"]),
        "relaxed_candidate_pair_count": sum(1 for row in complete_same_date_groups if row["usable_as_relaxed_candidate"]),
        "best_complete_same_date_groups": complete_same_date_groups[:12],
        "best_individual_tile_items": best_individual_items[:20],
        "best_incomplete_dates": incomplete_dates[:20],
        "weak_sources_required_for_benchmark": {
            "dynamic_world": "required for weak-label agreement if Earth Engine export exists for the selected date window",
            "esa_worldcover": "usable as static external weak label; latest available map is not 2026-specific",
            "osm_buildings_roads_landuse": "usable as current vector support, with timestamp caveat",
            "sentinel2_indices": "required from the selected 2026 optical pair",
            "sentinel1_vv_vh": "required from matched Sentinel-1 RTC pair",
            "existing_high_confidence_change_polygons": "optional bootstrap support, not independent final-test truth",
        },
        "configured_weak_sources": weak_sources.get("candidate_sources", {}),
        "recommendation": (
            "Use the lowest max-tile-cloud complete same-date group only if it meets the fallback cloud policy. "
            "If only relaxed candidates exist, keep 2026 as a benchmark candidate but do not claim final "
            "test accuracy until cloud masking and valid-pixel coverage prove the AOI is usable."
        ),
    }

    output_path = Path(config["paths"]["outputs"]) / "test_benchmark_2026_candidates.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover 2026 held-out test benchmark candidates.")
    parser.parse_args()
    path = discover_2026_test_benchmark(load_config())
    print(f"2026 test benchmark candidate report: {path}")


if __name__ == "__main__":
    main()
