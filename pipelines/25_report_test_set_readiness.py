"""Report whether a clean held-out test set is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PAIRS = ROOT / "data" / "catalog" / "training_scene_pairs_2023_2026.json"
DEFAULT_UNET_MANIFEST = ROOT / "data" / "interim" / "unet_full_aoi" / "unet_full_aoi_manifest.json"
DEFAULT_REPORT = ROOT / "data" / "outputs" / "test_set_readiness_report.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(scene_pairs_path: Path, unet_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    scenes = read_json(scene_pairs_path)
    unet = read_json(unet_manifest_path) if unet_manifest_path.exists() else {"records": []}
    scene_test_pairs = [pair for pair in scenes.get("pairs", []) if pair.get("split") == "test"]
    unet_test_records = [record for record in unet.get("records", []) if record.get("split") == "test"]

    report = {
        "phase": "phase_42_test_set_readiness",
        "status": "ready" if scene_test_pairs and unet_test_records else "not_ready",
        "scene_split_counts": scenes.get("split_counts", {}),
        "yearly_status": scenes.get("yearly_status", {}),
        "test_scene_pair_count": len(scene_test_pairs),
        "test_unet_record_count": len(unet_test_records),
        "test_pairs": [
            {
                "output_tag": pair["output_tag"],
                "year": pair["year"],
                "optical_date": pair["optical"]["date"],
                "cloud_cover_percent": pair["optical"]["cloud_cover_percent"],
                "max_tile_cloud_cover_percent": pair["optical"]["max_tile_cloud_cover_percent"],
                "radar_date": pair["radar"]["date"],
                "time_gap_hours": pair["radar"]["time_gap_hours"],
            }
            for pair in scene_test_pairs
        ],
        "blocked_reason": None,
        "recommendation": None,
    }
    if report["status"] != "ready":
        report["blocked_reason"] = (
            "No clean held-out test split is available yet. The 2026 optional catalog currently has no matched usable pairs."
        )
        report["recommendation"] = (
            "Keep 2023/2024 for training, 2025 for validation, and rerun scene discovery when 2026 AOI scenes below the cloud threshold become available."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report held-out test-set readiness.")
    parser.add_argument("--scene-pairs", type=Path, default=DEFAULT_SCENE_PAIRS)
    parser.add_argument("--unet-manifest", type=Path, default=DEFAULT_UNET_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = build_report(args.scene_pairs, args.unet_manifest, args.output)
    print(f"Test-set readiness report: {args.output}")
    print(json.dumps({"status": report["status"], "test_scene_pair_count": report["test_scene_pair_count"]}, indent=2))


if __name__ == "__main__":
    main()
