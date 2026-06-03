"""Validate that held-out benchmark data cannot leak into model development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PAIRS = ROOT / "data" / "catalog" / "training_scene_pairs_2023_2026.json"
DEFAULT_PATCH_MANIFEST = ROOT / "data" / "interim" / "training_patches" / "training_patch_manifest.json"
DEFAULT_BENCHMARK_REPORT = ROOT / "data" / "outputs" / "test_benchmark_2026_candidates.json"
DEFAULT_OUTPUT = ROOT / "data" / "outputs" / "no_leakage_audit_report.json"


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def tag_splits(records: list[dict[str, Any]], tag_key: str = "output_tag") -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for record in records:
        tag = str(record.get(tag_key, ""))
        split = str(record.get("split", "unknown"))
        if not tag:
            continue
        splits.setdefault(tag, set()).add(split)
    return splits


def validate_scene_pair_splits(scene_pairs: dict[str, Any]) -> list[str]:
    """Return leakage violations from the Sentinel scene-pair catalog."""
    violations: list[str] = []
    pairs = scene_pairs.get("pairs", [])
    for pair in pairs:
        year = int(pair.get("year", 0))
        split = pair.get("split")
        if year >= 2026 and split != "test":
            violations.append(
                f"2026+ scene pair {pair.get('output_tag')} is assigned to {split}, not test."
            )

    for tag, splits in tag_splits(pairs).items():
        if len(splits) > 1:
            violations.append(f"Scene pair {tag} appears in multiple splits: {sorted(splits)}.")
    return violations


def validate_patch_splits(patch_manifest: dict[str, Any]) -> list[str]:
    """Return leakage violations from generated patch records."""
    violations: list[str] = []
    patches = patch_manifest.get("patches", [])
    for record in patches:
        year = int(record.get("year", 0))
        split = record.get("split")
        if year >= 2026 and split != "test":
            violations.append(
                f"2026+ patch {record.get('patch_id')} is assigned to {split}, not test."
            )

    tag_to_splits: dict[str, set[str]] = {}
    for record in patches:
        tag = str(record.get("training_tag", ""))
        split = str(record.get("split", "unknown"))
        if tag:
            tag_to_splits.setdefault(tag, set()).add(split)
    for tag, splits in tag_to_splits.items():
        if len(splits) > 1:
            violations.append(f"Patch source tag {tag} appears in multiple splits: {sorted(splits)}.")
    return violations


def build_report(
    scene_pairs_path: Path,
    patch_manifest_path: Path,
    benchmark_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    scene_pairs = read_json(scene_pairs_path)
    patch_manifest = read_json(patch_manifest_path, default={"patches": [], "split_counts": {}})
    benchmark = read_json(benchmark_report_path, default={})

    violations = []
    violations.extend(validate_scene_pair_splits(scene_pairs))
    violations.extend(validate_patch_splits(patch_manifest))

    scene_pairs_list = scene_pairs.get("pairs", [])
    patch_records = patch_manifest.get("patches", [])
    report = {
        "phase": "phase_48_no_leakage_audit",
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "scene_split_counts": scene_pairs.get("split_counts", {}),
        "patch_split_counts": patch_manifest.get("split_counts", {}),
        "test_scene_pair_count": sum(1 for pair in scene_pairs_list if pair.get("split") == "test"),
        "test_patch_count": sum(1 for record in patch_records if record.get("split") == "test"),
        "benchmark_2026": {
            "strict_test_pair_count": benchmark.get("strict_test_pair_count", 0),
            "relaxed_candidate_pair_count": benchmark.get("relaxed_candidate_pair_count", 0),
            "best_candidate": (benchmark.get("best_complete_same_date_groups") or [None])[0],
        },
        "non_cheating_protocol": {
            "train": "May use 2023/2024 patches only.",
            "validation": "May use 2025 patches for early stopping and model selection only.",
            "test": "May use 2026 held-out candidates only after the model is frozen.",
            "weak_sources": (
                "Dynamic World, ESA WorldCover, OSM, Sentinel index evidence, and Sentinel-1 evidence "
                "are proxy agreement sources. They must not be described as field truth."
            ),
            "bootstrap_polygons": (
                "Existing high-confidence change polygons may guide review or pseudo-label generation, "
                "but must not be used as independent final-test truth."
            ),
        },
        "recommendation": (
            "Run this audit before any model training, calibration, or reported benchmark evaluation. "
            "If status is fail, stop and fix split/catalog leakage first."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate no-leakage rules for held-out test data.")
    parser.add_argument("--scene-pairs", type=Path, default=DEFAULT_SCENE_PAIRS)
    parser.add_argument("--patch-manifest", type=Path, default=DEFAULT_PATCH_MANIFEST)
    parser.add_argument("--benchmark-report", type=Path, default=DEFAULT_BENCHMARK_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(
        scene_pairs_path=args.scene_pairs,
        patch_manifest_path=args.patch_manifest,
        benchmark_report_path=args.benchmark_report,
        output_path=args.output,
    )
    print(f"No-leakage audit report: {args.output}")
    print(json.dumps({"status": report["status"], "violations": report["violations"]}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
