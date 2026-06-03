"""Discover multi-year Sentinel-1/2 scene pairs for future ML training."""

from __future__ import annotations

import argparse
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


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def date_key(item: Any) -> str:
    return parse_datetime(item.properties["datetime"]).date().isoformat()


def split_for_year(settings: dict[str, Any], year: int) -> str:
    for split, years in settings["split_by_year"].items():
        if year in years:
            return split
    return "unused"


def item_to_feature(item: Any) -> dict[str, Any]:
    """Return compact STAC metadata needed for later AOI-windowed reads."""
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
            "mgrs_tile": properties.get("s2:mgrs_tile"),
            "assets": {key: asset.get("href") for key, asset in data.get("assets", {}).items()},
            "stac_properties": properties,
        },
    }


def search_sentinel2_groups(
    client: Client,
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    year: int,
    cloud_threshold: float,
) -> list[list[Any]]:
    """Search same-date Sentinel-2 groups that cover all required AOI tiles."""
    settings = config["training_scene_discovery"]
    start = max(f"{year}-01-01", settings["date_range"]["start"])
    end = min(f"{year}-12-31", settings["date_range"]["end"])
    if start > end:
        return []

    search = client.search(
        collections=[config["sentinel"]["collections"]["sentinel2"]],
        bbox=aoi.to_crs("EPSG:4326").total_bounds.tolist(),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_threshold}},
        max_items=500,
        method="POST",
    )

    by_date: dict[str, list[Any]] = defaultdict(list)
    for item in search.items():
        by_date[date_key(item)].append(item)

    required_tiles = set(settings["required_tiles"])
    groups = []
    for items in by_date.values():
        by_tile = {}
        for item in items:
            tile = item.properties.get("s2:mgrs_tile")
            if tile not in required_tiles:
                continue
            current = by_tile.get(tile)
            if current is None or item.properties.get("eo:cloud_cover", 100) < current.properties.get("eo:cloud_cover", 100):
                by_tile[tile] = item
        if required_tiles.issubset(by_tile):
            groups.append([by_tile[tile] for tile in sorted(required_tiles)])

    groups.sort(
        key=lambda group: (
            sum(item.properties.get("eo:cloud_cover", 100) for item in group) / len(group),
            date_key(group[0]),
        )
    )
    return groups[: int(settings["max_optical_groups_per_year"])]


def find_matching_sentinel1(
    client: Client,
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    optical_date: str,
) -> Any | None:
    """Find the closest Sentinel-1 RTC VV/VH item for a Sentinel-2 date."""
    settings = config["training_scene_discovery"]
    center = datetime.fromisoformat(optical_date).replace(tzinfo=timezone.utc)
    days = int(settings["sentinel1_search_days"])
    candidates = []

    for offset in sorted(range(-days, days + 1), key=lambda value: (abs(value), value)):
        day = (center + timedelta(days=offset)).date().isoformat()
        search = client.search(
            collections=[config["sentinel"]["collections"]["sentinel1"]],
            bbox=aoi.to_crs("EPSG:4326").total_bounds.tolist(),
            datetime=f"{day}/{day}",
            max_items=30,
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
            if gap_hours > float(settings["max_radar_time_gap_hours"]):
                continue
            orbit_penalty = 0 if props.get("sat:orbit_state") == settings["preferred_orbit"] else 1
            candidates.append((orbit_penalty, gap_hours, item))

    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]))
    return planetary_computer.sign(candidates[0][2])


def scene_pair_record(
    config: dict[str, Any],
    year: int,
    split: str,
    cloud_threshold: float,
    s2_group: list[Any],
    s1_item: Any,
) -> dict[str, Any]:
    optical_date = date_key(s2_group[0])
    radar_dt = parse_datetime(s1_item.properties["datetime"])
    optical_dt = datetime.fromisoformat(optical_date).replace(tzinfo=timezone.utc)
    cloud_values = [float(item.properties.get("eo:cloud_cover", 100)) for item in s2_group]
    tile_cloud_cover = {
        str(item.properties.get("s2:mgrs_tile")): round(float(item.properties.get("eo:cloud_cover", 100)), 2)
        for item in s2_group
    }
    return {
        "year": year,
        "split": split,
        "cloud_tier": (
            "preferred"
            if cloud_threshold <= float(config["training_scene_discovery"]["preferred_max_cloud_cover"])
            else "fallback"
        ),
        "selection_cloud_threshold": cloud_threshold,
        "output_tag": f"training_{year}_{optical_date.replace('-', '')}_{radar_dt.date().isoformat().replace('-', '')}",
        "optical": {
            "collection": config["sentinel"]["collections"]["sentinel2"],
            "date": optical_date,
            "tiles": [item.properties.get("s2:mgrs_tile") for item in s2_group],
            "scene_ids": [item.id for item in s2_group],
            "bands": config["preprocessing"]["selected_pair"]["optical"]["bands"],
            "tile_cloud_cover_percent": tile_cloud_cover,
            "cloud_cover_percent": round(sum(cloud_values) / len(cloud_values), 2),
            "max_tile_cloud_cover_percent": round(max(cloud_values), 2),
        },
        "radar": {
            "collection": config["sentinel"]["collections"]["sentinel1"],
            "date": radar_dt.date().isoformat(),
            "scene_id": s1_item.id,
            "orbit": s1_item.properties.get("sat:orbit_state"),
            "polarizations": s1_item.properties.get("sar:polarizations", []),
            "time_gap_hours": round(abs((radar_dt - optical_dt).total_seconds()) / 3600, 1),
        },
    }


def match_groups_for_year(
    client: Client,
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    year: int,
    cloud_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Search optical groups for one year and keep only Sentinel-1 matched pairs."""
    groups = search_sentinel2_groups(client, config, aoi, year, cloud_threshold)
    matched = []
    for group in groups:
        s1_item = find_matching_sentinel1(client, config, aoi, date_key(group[0]))
        if s1_item is None:
            continue
        matched.append(
            scene_pair_record(
                config,
                year,
                split_for_year(config["training_scene_discovery"], year),
                cloud_threshold,
                group,
                s1_item,
            )
        )
    return matched[: int(config["training_scene_discovery"]["max_pairs_per_year"])], len(groups)


def discover_training_scenes(config: dict[str, Any]) -> Path:
    """Write a multi-year train/validation/test Sentinel-1/2 scene catalog."""
    ensure_output_dirs(config)
    settings = config["training_scene_discovery"]
    aoi = gpd.read_file(config["project"]["aoi_path"])
    client = Client.open(config["sentinel"]["stac_api_url"])
    required_years = [int(year) for year in settings["required_years"]]
    optional_years = [int(year) for year in settings["optional_years"]]
    years = required_years + optional_years

    pairs = []
    s2_features_by_id = {}
    yearly_status = {}
    for year in years:
        threshold_used = float(settings["preferred_max_cloud_cover"])
        matched, candidate_group_count = match_groups_for_year(
            client,
            config,
            aoi,
            year,
            threshold_used,
        )

        # The portfolio target is <10% cloud when possible. Fallback is allowed
        # only when strict optical scenes cannot produce a Sentinel-1 matched pair.
        if not matched:
            threshold_used = float(settings["fallback_max_cloud_cover"])
            matched, candidate_group_count = match_groups_for_year(
                client,
                config,
                aoi,
                year,
                threshold_used,
            )

        pairs.extend(matched)
        for pair in matched:
            for group in search_sentinel2_groups(client, config, aoi, year, threshold_used):
                if date_key(group[0]) == pair["optical"]["date"]:
                    for item in group:
                        s2_features_by_id[item.id] = item_to_feature(item)
                    break
        yearly_status[str(year)] = {
            "required": year in required_years,
            "optional": year in optional_years,
            "cloud_threshold_used": threshold_used,
            "cloud_tier": "preferred" if threshold_used == float(settings["preferred_max_cloud_cover"]) else "fallback",
            "candidate_optical_groups": candidate_group_count,
            "matched_pairs": len(matched),
            "status": "ok" if matched else ("optional_missing" if year in optional_years else "missing_required"),
        }

    catalog = {
        "phase": "multi_year_training_scene_discovery",
        "purpose": "Sentinel-1/2 scene candidates for future weakly supervised training, validation, and testing.",
        "aoi_path": config["project"]["aoi_path"],
        "date_range": settings["date_range"],
        "preferred_max_cloud_cover": settings["preferred_max_cloud_cover"],
        "fallback_max_cloud_cover": settings["fallback_max_cloud_cover"],
        "required_years": required_years,
        "optional_years": optional_years,
        "status": "ok" if all(yearly_status[str(year)]["matched_pairs"] for year in required_years) else "needs_attention",
        "yearly_status": yearly_status,
        "split_counts": {
            split: sum(1 for pair in pairs if pair["split"] == split)
            for split in ["train", "validation", "test", "unused"]
        },
        "pairs": pairs,
    }
    output_path = Path(config["paths"]["catalog"]) / "training_scene_pairs_2023_2026.json"
    output_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    scenes_path = Path(config["paths"]["catalog"]) / "training_scenes_2023_2026.geojson"
    scenes_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": list(s2_features_by_id.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover multi-year Sentinel-1/2 training scenes.")
    parser.parse_args()
    path = discover_training_scenes(load_config())
    print(f"Training scene catalog: {path}")


if __name__ == "__main__":
    main()
