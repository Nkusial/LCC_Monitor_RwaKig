"""Prepare a review-to-retraining manifest from U-Net reliability outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELIABILITY_REPORT = ROOT / "data" / "outputs" / "unet_full_aoi_validation_report.json"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "retraining" / "review_retraining_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(report_path: Path, output_path: Path, max_items: int) -> dict[str, Any]:
    report = read_json(report_path)
    records = sorted(
        report.get("records", []),
        key=lambda item: item.get("candidate_review_fraction_of_valid", 0),
        reverse=True,
    )[:max_items]

    tasks = [
        {
            "priority": index + 1,
            "training_tag": record["training_tag"],
            "year": record["year"],
            "split": record["split"],
            "review_reason": "High candidate-review fraction from entropy, low confidence, or weak-source disagreement.",
            "candidate_review_fraction_of_valid": record["candidate_review_fraction_of_valid"],
            "compatible_fraction_in_agreement_zone": record["compatible_fraction_in_agreement_zone"],
            "recommended_action": "Digitize or collect reference labels in review zones before adding corrected labels to a retraining patch manifest.",
        }
        for index, record in enumerate(records)
    ]

    manifest = {
        "phase": "phase_41_review_to_retraining_loop",
        "status": "ready_for_reference_label_collection",
        "source_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "label_storage_contract": {
            "future_corrected_labels": "data/interim/retraining/corrected_reference_labels.geojson",
            "future_retraining_patch_manifest": "data/interim/retraining/retraining_patch_manifest.json",
            "do_not_train_on": "unreviewed high-entropy or weak-disagreement pixels",
        },
        "review_tasks": tasks,
        "next_steps": [
            "Inspect U-Net review-zone overlay in the WebGIS.",
            "Create reference labels for the highest-priority records.",
            "Regenerate patches with corrected labels and confidence masks.",
            "Retrain U-Net and compare validation/test reports before replacing hosted model layers.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review tasks for future corrected-label U-Net retraining.")
    parser.add_argument("--report", type=Path, default=DEFAULT_RELIABILITY_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-items", type=int, default=5)
    args = parser.parse_args()

    manifest = build_manifest(args.report, args.output, args.max_items)
    print(f"Review-to-retraining manifest: {args.output}")
    print(json.dumps({"task_count": len(manifest["review_tasks"])}, indent=2))


if __name__ == "__main__":
    main()
