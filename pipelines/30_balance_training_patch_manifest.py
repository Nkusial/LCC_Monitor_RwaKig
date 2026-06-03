"""Create a class-balanced patch manifest for weakly supervised U-Net training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "interim" / "training_patches_v3" / "training_patch_manifest_v3.json"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "training_patches_v3" / "training_patch_manifest_v3_balanced.json"
DEFAULT_REPORT = ROOT / "data" / "outputs" / "training_patch_balance_report_v3.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for class_id, value in record.get("class_pixel_counts", {}).items():
            counts[class_id] = counts.get(class_id, 0) + int(value)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        split = record.get("split", "unknown")
        counts[split] = counts.get(split, 0) + 1
    return counts


def minority_score(record: dict[str, Any], minority_classes: set[str]) -> tuple[int, int, int, str]:
    """Rank patches by rare-class content, then total supervised pixels."""
    histogram = record.get("class_pixel_counts", {})
    minority_pixels = sum(int(histogram.get(class_id, 0)) for class_id in minority_classes)
    total_pixels = sum(int(value) for value in histogram.values())
    class_diversity = len(histogram)
    return (minority_pixels, class_diversity, total_pixels, record["patch_id"])


def build_balanced_manifest(
    manifest: dict[str, Any],
    minority_classes: set[str],
    majority_limit_per_pair: int,
    minority_multiplier: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep rare-class examples and cap majority-only examples per training pair."""
    records = manifest.get("patches", [])
    train_records = [record for record in records if record.get("split") == "train"]
    non_train_records = [record for record in records if record.get("split") != "train"]

    selected_train: list[dict[str, Any]] = []
    dropped_majority_only: list[str] = []
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in train_records:
        by_pair.setdefault(record["training_tag"], []).append(record)

    for pair_records in by_pair.values():
        minority_records = [
            record
            for record in pair_records
            if any(int(record.get("class_pixel_counts", {}).get(class_id, 0)) > 0 for class_id in minority_classes)
        ]
        majority_records = [record for record in pair_records if record not in minority_records]
        minority_records = sorted(
            minority_records,
            key=lambda record: minority_score(record, minority_classes),
            reverse=True,
        )
        majority_records = sorted(
            majority_records,
            key=lambda record: int(record.get("training_pixel_count", 0)),
            reverse=True,
        )
        allowed_majority = max(
            majority_limit_per_pair,
            int(round(len(minority_records) * minority_multiplier)),
        )
        selected_train.extend(minority_records)
        selected_train.extend(majority_records[:allowed_majority])
        dropped_majority_only.extend(record["patch_id"] for record in majority_records[allowed_majority:])

    selected_records = sorted(
        selected_train + non_train_records,
        key=lambda record: (record.get("split", ""), record.get("training_tag", ""), record.get("patch_id", "")),
    )
    balanced = manifest.copy()
    balanced["phase"] = "phase_48_balanced_weak_supervision_patch_manifest"
    balanced["patch_count"] = len(selected_records)
    balanced["split_counts"] = split_counts(selected_records)
    balanced["class_pixel_counts"] = class_counts(selected_records)
    balanced["total_training_pixel_count"] = int(sum(record.get("training_pixel_count", 0) for record in selected_records))
    balanced["patches"] = selected_records
    balanced["balancing"] = {
        "minority_classes": sorted(minority_classes, key=int),
        "majority_limit_per_pair": majority_limit_per_pair,
        "minority_multiplier": minority_multiplier,
        "dropped_majority_only_patch_count": len(dropped_majority_only),
        "strategy": "retain all rare-class train patches, cap majority-only train patches per date, keep validation/test unchanged",
    }
    balanced["notes"] = [
        *manifest.get("notes", []),
        "The balanced manifest controls training exposure without duplicating patch arrays.",
        "Validation/test patches are not downsampled so evaluation still sees the broader weak-label distribution.",
    ]

    report = {
        "phase": "phase_48_class_imbalance_handling",
        "input_patch_count": len(records),
        "output_patch_count": len(selected_records),
        "input_split_counts": split_counts(records),
        "output_split_counts": balanced["split_counts"],
        "input_class_pixel_counts": class_counts(records),
        "output_class_pixel_counts": balanced["class_pixel_counts"],
        "balancing": balanced["balancing"],
        "accuracy_claim": "none; this is weak-supervision dataset balancing before model training",
    }
    return balanced, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance weak-label patch manifest before U-Net training.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--minority-classes", nargs="+", default=["3", "4", "5"])
    parser.add_argument("--majority-limit-per-pair", type=int, default=24)
    parser.add_argument("--minority-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    manifest = read_json(args.input)
    balanced, report = build_balanced_manifest(
        manifest,
        minority_classes=set(args.minority_classes),
        majority_limit_per_pair=args.majority_limit_per_pair,
        minority_multiplier=args.minority_multiplier,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(balanced, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Balanced patch manifest: {args.output}")
    print(json.dumps(report["balancing"], indent=2))


if __name__ == "__main__":
    main()
