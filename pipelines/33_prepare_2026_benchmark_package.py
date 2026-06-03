"""Prepare an auditable 2026 held-out benchmark package.

This stage does not train, tune, or refine the model. It collects the best
available 2026 Sentinel candidate, aligned external weak-source products, and
the no-leakage audit into one evaluation-readiness report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "outputs" / "test_benchmark_2026_candidates.json"
DEFAULT_WEAK_SOURCES = ROOT / "data" / "interim" / "external_sources" / "external_weak_sources_summary_test_2026.json"
DEFAULT_NO_LEAKAGE = ROOT / "data" / "outputs" / "no_leakage_audit_report.json"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "data" / "outputs" / "heldout_benchmark_2026_package.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def best_candidate(candidates: dict[str, Any]) -> dict[str, Any] | None:
    groups = candidates.get("best_complete_same_date_groups") or []
    return groups[0] if groups else None


def expected_test_tag(candidate: dict[str, Any] | None) -> str | None:
    if not candidate or not candidate.get("sentinel1_match"):
        return None
    optical_date = str(candidate["date"]).replace("-", "")
    radar_date = str(candidate["sentinel1_match"]["date"]).replace("-", "")
    return f"test_2026_{optical_date}_{radar_date}"


def processed_feature_status(processed_dir: Path, tag: str | None) -> dict[str, Any]:
    required_suffixes = [
        "s2_ndvi",
        "s2_ndwi",
        "s2_ndbi",
        "s1_vv",
        "s1_vh",
        "s1_vv_vh_ratio",
    ]
    if tag is None:
        return {"status": "blocked", "reason": "No candidate tag could be derived.", "missing": required_suffixes}
    expected = {suffix: processed_dir / f"{tag}_{suffix}.tif" for suffix in required_suffixes}
    missing = [suffix for suffix, path in expected.items() if not path.exists()]
    return {
        "status": "ready" if not missing else "missing_processed_stack",
        "tag": tag,
        "required_features": required_suffixes,
        "available": [suffix for suffix in required_suffixes if suffix not in missing],
        "missing": missing,
        "expected_paths": {suffix: str(path).replace("\\", "/") for suffix, path in expected.items()},
    }


def weak_source_status(summary: dict[str, Any]) -> dict[str, Any]:
    compatibility = summary.get("georeference_and_aoi_compatibility", {})
    guardrails = summary.get("benchmark_guardrails", {})
    return {
        "status": "ready"
        if compatibility.get("status") == "pass" and guardrails.get("local_bootstrap_included") is False
        else "blocked",
        "agreement_sources": summary.get("agreement_ready_sources", []),
        "agreement_outputs": summary.get("agreement_outputs", {}),
        "georeference_and_aoi_compatibility": compatibility,
        "local_bootstrap_included": guardrails.get("local_bootstrap_included"),
        "leakage_policy": guardrails.get("leakage_policy"),
    }


def build_package(
    candidates_path: Path,
    weak_sources_path: Path,
    no_leakage_path: Path,
    processed_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    candidates = read_json(candidates_path)
    weak_sources = read_json(weak_sources_path)
    no_leakage = read_json(no_leakage_path)
    candidate = best_candidate(candidates)
    tag = expected_test_tag(candidate)
    feature_status = processed_feature_status(processed_dir, tag)
    source_status = weak_source_status(weak_sources)
    strict_count = int(candidates.get("strict_test_pair_count", 0))

    blockers = []
    if no_leakage.get("status") != "pass":
        blockers.append("No-leakage audit failed.")
    if source_status["status"] != "ready":
        blockers.append("2026 weak sources are not aligned, AOI-masked, or leakage-safe.")
    if feature_status["status"] != "ready":
        blockers.append("2026 Sentinel-1/2 processed feature stack is not available yet.")
    if strict_count == 0:
        blockers.append("No strict <10% cloud complete 2026 test pair is available yet; current pair is relaxed.")

    report = {
        "phase": "phase_49_heldout_2026_benchmark_package",
        "status": "ready_for_frozen_model_proxy_evaluation" if not blockers else "benchmark_package_prepared_with_blockers",
        "benchmark_year": 2026,
        "candidate_policy": {
            "strict_test_pair_count": strict_count,
            "relaxed_candidate_pair_count": candidates.get("relaxed_candidate_pair_count", 0),
            "preferred_max_cloud_cover": candidates.get("preferred_max_cloud_cover"),
            "fallback_max_cloud_cover": candidates.get("fallback_max_cloud_cover"),
            "accuracy_claim_allowed": False,
            "reason": "Weak-source agreement is proxy evidence, not field validation truth.",
        },
        "selected_candidate": candidate,
        "expected_test_tag": tag,
        "processed_feature_stack": feature_status,
        "weak_sources": source_status,
        "no_leakage_audit": {
            "status": no_leakage.get("status"),
            "violations": no_leakage.get("violations", []),
            "protocol": no_leakage.get("non_cheating_protocol", {}),
        },
        "blockers": blockers,
        "allowed_next_actions": [
            "Preprocess the selected 2026 Sentinel-1/2 pair into the expected test tag only.",
            "Run frozen-model inference on 2026 after preprocessing is complete.",
            "Report proxy agreement, entropy, confidence calibration, and review zones separately from field accuracy.",
        ],
        "forbidden_actions": [
            "Do not add 2026 patches to train or validation manifests.",
            "Do not tune thresholds or class weights using 2026 benchmark outcomes.",
            "Do not describe Dynamic World, ESA WorldCover, OSM, or Sentinel evidence as field truth.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 2026 held-out benchmark package.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--weak-sources", type=Path, default=DEFAULT_WEAK_SOURCES)
    parser.add_argument("--no-leakage", type=Path, default=DEFAULT_NO_LEAKAGE)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_package(
        candidates_path=args.candidates,
        weak_sources_path=args.weak_sources,
        no_leakage_path=args.no_leakage,
        processed_dir=args.processed_dir,
        output_path=args.output,
    )
    print(f"2026 benchmark package: {args.output}")
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, indent=2))


if __name__ == "__main__":
    main()
