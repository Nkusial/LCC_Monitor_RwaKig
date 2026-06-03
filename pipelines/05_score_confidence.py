"""Apply final confidence and publish-readiness scoring to change polygons."""

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def reliability_label(score: float) -> str:
    """Translate numeric confidence into a reviewer-friendly reliability label."""
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def confidence_adjustment(row: Any) -> float:
    """Apply conservative penalties for ambiguous change situations."""
    adjustment = 0.0
    if row.get("change_type") == "radar_change":
        adjustment -= 0.12
    if row.get("monitored_land_cover") == "mixed":
        adjustment -= 0.08
    if row.get("before_state") == row.get("after_state"):
        adjustment -= 0.05
    if row.get("radar_before_date") and row.get("radar_after_date"):
        # Keep the hook explicit for future orbit/time-gap penalties stored in properties.
        adjustment += 0.0
    return adjustment


def score_changes(config: dict[str, Any]) -> tuple[Path, Path]:
    """Create scored GeoJSON and summary files for publishing and review."""
    ensure_output_dirs(config)
    outputs_dir = Path(config["paths"]["outputs"])
    input_path = outputs_dir / f"{config['change_detection']['output_prefix']}_polygons.geojson"
    if not input_path.exists():
        raise FileNotFoundError(f"Change polygon output not found: {input_path}")

    gdf = gpd.read_file(input_path)
    if gdf.empty:
        scored_path = outputs_dir / f"{config['change_detection']['output_prefix']}_scored.geojson"
        summary_path = outputs_dir / f"{config['change_detection']['output_prefix']}_scored_summary.json"
        gdf.to_file(scored_path, driver="GeoJSON")
        summary_path.write_text(json.dumps({"total": 0}, indent=2), encoding="utf-8")
        return scored_path, summary_path

    threshold = float(config["change_detection"]["confidence_threshold"])
    adjusted_scores = []
    for _, row in gdf.iterrows():
        base = float(row["confidence"])
        magnitude = float(row.get("change_magnitude", 0) or 0)
        # A small magnitude bonus helps strong multi-feature changes rank above
        # weak threshold crossings without overpowering the base confidence.
        magnitude_bonus = min(magnitude, 1.0) * 0.05
        adjusted = base + confidence_adjustment(row) + magnitude_bonus
        adjusted_scores.append(float(np.clip(adjusted, 0, 1)))

    gdf["final_confidence"] = [round(score, 3) for score in adjusted_scores]
    gdf["reliability"] = [reliability_label(score) for score in adjusted_scores]
    gdf["publish_ready"] = gdf["final_confidence"] >= threshold
    gdf["review_reason"] = np.where(
        gdf["publish_ready"],
        "meets_publish_threshold",
        "below_publish_threshold",
    )

    scored_path = outputs_dir / f"{config['change_detection']['output_prefix']}_scored.geojson"
    summary_path = outputs_dir / f"{config['change_detection']['output_prefix']}_scored_summary.json"
    gdf.to_file(scored_path, driver="GeoJSON")
    summary = {
        "total": int(len(gdf)),
        "publish_ready": int(gdf["publish_ready"].sum()),
        "confidence_threshold": threshold,
        "by_reliability": {
            key: int(value)
            for key, value in gdf.groupby("reliability").size().to_dict().items()
        },
        "by_monitored_land_cover": {
            key: int(value)
            for key, value in gdf.groupby("monitored_land_cover").size().to_dict().items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return scored_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply final confidence scoring to change polygons.")
    parser.parse_args()

    scored_path, summary_path = score_changes(load_config())
    print(f"Scored changes: {scored_path}")
    print(f"Scored summary: {summary_path}")


if __name__ == "__main__":
    main()
