"""Export Dynamic World labels for the AOI and rerun weak-source ingestion.

Dynamic World is served through Google Earth Engine. This stage downloads a
local AOI GeoTIFF that Phase 26B can harmonize with ESA WorldCover, OSM, and
local Sentinel evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from common import load_config
except ModuleNotFoundError:
    from pipelines.common import load_config


DEFAULT_OUTPUT = ROOT / "data" / "interim" / "external_sources" / "dynamic_world_label.tif"
DEFAULT_SUMMARY = ROOT / "data" / "interim" / "external_sources" / "dynamic_world_export_summary.json"


def project_path(path: str | Path) -> Path:
    """Resolve CLI paths relative to the repository root for stable reports."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    settings_path = project_path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def aoi_geometry(config: dict[str, Any]) -> dict[str, Any]:
    aoi_path = ROOT / config["project"]["aoi_path"]
    data = json.loads(aoi_path.read_text(encoding="utf-8"))
    return data["features"][0]["geometry"]


def parse_monitoring_date(tag: str) -> str:
    """Extract the optical date from monitoring_YYYYMMDD_YYYYMMDD."""
    parts = tag.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return datetime.strptime(parts[1], "%Y%m%d").date().isoformat()
    raise ValueError(f"Could not parse monitoring date from tag: {tag}")


def latest_monitoring_tag(processed_dir: Path) -> str:
    tags = sorted(
        path.name.removesuffix("_s2_ndvi.tif")
        for path in processed_dir.glob("monitoring_*_s2_ndvi.tif")
    )
    if not tags:
        raise FileNotFoundError("No monitoring NDVI rasters found to infer a Dynamic World date.")
    return tags[-1]


def date_window(center_date: str, days_before: int, days_after: int) -> tuple[str, str]:
    center = datetime.strptime(center_date, "%Y-%m-%d")
    start = center - timedelta(days=days_before)
    # Earth Engine filterDate end is exclusive, so add one day.
    end = center + timedelta(days=days_after + 1)
    return start.date().isoformat(), end.date().isoformat()


def initialize_earth_engine(project: str | None) -> None:
    import ee

    ee_project = project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        if ee_project:
            ee.Initialize(project=ee_project)
        else:
            ee.Initialize()
    except Exception as exc:
        message = (
            "Google Earth Engine is installed but not initialized for this machine. "
            "Run `earthengine authenticate`, then rerun this script with "
            "`--ee-project YOUR_GOOGLE_CLOUD_PROJECT` or set EE_PROJECT."
        )
        raise RuntimeError(message) from exc


def export_dynamic_world(
    config: dict[str, Any],
    weak_settings: dict[str, Any],
    monitoring_tag: str,
    output_path: Path,
    date: str | None,
    days_before: int,
    days_after: int,
    project: str | None,
) -> dict[str, Any]:
    import ee
    import geemap

    initialize_earth_engine(project)
    output_path = project_path(output_path)

    selected_date = date or parse_monitoring_date(monitoring_tag)
    start_date, end_date = date_window(selected_date, days_before, days_after)
    asset = weak_settings["candidate_sources"]["dynamic_world"]["earth_engine_asset"]
    region = ee.Geometry(aoi_geometry(config))

    collection = (
        ee.ImageCollection(asset)
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )
    count = int(collection.size().getInfo())
    if count == 0:
        raise RuntimeError(
            f"No Dynamic World images found for AOI between {start_date} and {end_date}."
        )

    label = collection.select("label").mode().clip(region).toUint8()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geemap.download_ee_image(
        label,
        filename=str(output_path),
        region=region,
        crs="EPSG:4326",
        scale=10,
        resampling="near",
        dtype="uint8",
        overwrite=True,
        unmask_value=255,
    )

    return {
        "phase": "dynamic_world_export",
        "monitoring_tag": monitoring_tag,
        "selected_date": selected_date,
        "date_window": {"start": start_date, "end": end_date},
        "earth_engine_asset": asset,
        "image_count": count,
        "output": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "next_step": "Run pipelines/11_ingest_external_weak_sources.py to harmonize Dynamic World and rebuild agreement masks.",
    }


def rerun_external_ingestion(tag: str | None) -> None:
    command = [sys.executable, "pipelines/11_ingest_external_weak_sources.py"]
    if tag:
        command.extend(["--tag", tag])
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Dynamic World labels and optionally rebuild weak-source agreement.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--weak-config", default="configs/weak_labels.yaml")
    parser.add_argument("--tag", default=None, help="Monitoring tag to align with; defaults to latest processed monitoring tag.")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD Dynamic World center date.")
    parser.add_argument("--days-before", type=int, default=15)
    parser.add_argument("--days-after", type=int, default=15)
    parser.add_argument("--ee-project", default=None, help="Google Cloud project enabled for Earth Engine.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rerun-external-ingestion", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    weak_settings = load_yaml(args.weak_config)
    monitoring_tag = args.tag or latest_monitoring_tag(ROOT / config["paths"]["processed"])

    summary = export_dynamic_world(
        config=config,
        weak_settings=weak_settings,
        monitoring_tag=monitoring_tag,
        output_path=args.output,
        date=args.date,
        days_before=args.days_before,
        days_after=args.days_after,
        project=args.ee_project,
    )
    summary_path = project_path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Dynamic World export summary: {summary_path}")
    print(json.dumps(summary, indent=2))

    if args.rerun_external_ingestion:
        rerun_external_ingestion(monitoring_tag)


if __name__ == "__main__":
    main()
