"""Summarize reliability controls for the next improvement track.

This script does not retrain a model or alter the frozen benchmark. It reads the
project's existing QA, temporal-consensus, and reliability reports and turns
them into a single publish-gate summary for change-monitoring decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "improvement_track.yaml"
OUTPUT_PATH = ROOT / "data" / "outputs" / "improvement_track_report.json"
WEB_SUMMARY_PATH = ROOT / "web" / "public" / "demo" / "improvement_track_summary.json"
DOCS_SUMMARY_PATH = ROOT / "docs" / "app" / "demo" / "improvement_track_summary.json"


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_json_if_exists(path_text: str) -> dict[str, Any] | None:
    path = repo_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def fraction(value: int | float, total: int | float) -> float | None:
    if not total:
        return None
    return round(float(value) / float(total), 6)


def summarize_coregistration(config: dict[str, Any]) -> dict[str, Any]:
    report = read_json_if_exists(config["coregistration_qa"]["report"])
    if report is None:
        report = read_json_if_exists(config["coregistration_qa"]["web_summary"])
    if report is None:
        return {"status": "missing", "recommendation": "run pipelines/39_coregistration_qa.py"}

    status = report.get("status", "unknown")
    allowed = set(config["coregistration_qa"]["required_statuses"])
    return {
        "status": status,
        "allowed_for_gate": status in allowed,
        "grid_failure_count": report.get("grid_failure_count", 0),
        "high_shift_warning_count": report.get("high_shift_warning_count", 0),
        "checked_target_count": report.get("checked_target_count", len(report.get("records", []))),
        "method": report.get("method", config["coregistration_qa"]["diagnostic_method"]),
    }


def summarize_temporal_consensus(config: dict[str, Any]) -> dict[str, Any]:
    report = read_json_if_exists(config["temporal_consensus_gate"]["report"])
    if report is None:
        return {"status": "missing", "recommendation": "run pipelines/23_temporal_unet_consensus.py"}

    valid = report.get("valid_pixel_count", 0)
    unstable = report.get("unstable_pixel_count", 0)
    instability_fraction = fraction(unstable, valid)
    mean_consensus = report.get("mean_consensus_fraction")
    min_consensus = config["temporal_consensus_gate"]["minimum_mean_consensus_fraction"]
    max_instability = config["temporal_consensus_gate"]["maximum_temporal_instability_fraction"]

    status = "passed"
    if mean_consensus is None or instability_fraction is None:
        status = "incomplete"
    elif float(mean_consensus) < min_consensus or instability_fraction > max_instability:
        status = "warning"

    return {
        "status": status,
        "record_count": report.get("record_count"),
        "mean_consensus_fraction": mean_consensus,
        "temporal_instability_fraction": instability_fraction,
        "thresholds": {
            "minimum_mean_consensus_fraction": min_consensus,
            "maximum_temporal_instability_fraction": max_instability,
        },
    }


def summarize_reliability(config: dict[str, Any]) -> dict[str, Any]:
    report = read_json_if_exists("data/outputs/reliability_uncertainty_control_report.json")
    if report is None:
        report = read_json_if_exists("web/public/demo/reliability_uncertainty_control_summary.json")
    if report is None:
        return {"status": "missing", "recommendation": "run pipelines/36_report_uncertainty_controls.py"}

    snapshot = report.get("current_reliability_snapshot", {})
    gate = config["change_publish_gate"]
    mean_confidence = snapshot.get("train_validation_mean_confidence")
    weak_compatibility = snapshot.get("train_validation_weak_compatibility")
    review_fraction = snapshot.get("train_validation_review_fraction")

    status = "passed"
    if (
        mean_confidence is None
        or weak_compatibility is None
        or review_fraction is None
        or mean_confidence < gate["minimum_mean_confidence"]
        or weak_compatibility < gate["minimum_weak_source_compatibility"]
        or review_fraction > gate["maximum_review_fraction"]
    ):
        status = "warning"

    return {
        "status": status,
        "mean_confidence": mean_confidence,
        "weak_source_compatibility": weak_compatibility,
        "review_fraction": review_fraction,
        "frozen_2026_mean_confidence": snapshot.get("frozen_2026_mean_confidence"),
        "frozen_2026_review_fraction": snapshot.get("frozen_2026_review_fraction"),
        "accuracy_claim": report.get("accuracy_claim"),
    }


def summarize_ml_comparison(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config["ml_comparison"]
    supervised_available = [path for path in ml_config["supervised_reports"] if repo_path(path).exists()]
    unsupervised_available = [path for path in ml_config["unsupervised_reports"] if repo_path(path).exists()]

    return {
        "supervised_track": ml_config["supervised_track"],
        "supervised_report_count": len(supervised_available),
        "supervised_available_reports": supervised_available,
        "unsupervised_track": ml_config["unsupervised_track"],
        "unsupervised_report_count": len(unsupervised_available),
        "unsupervised_available_reports": unsupervised_available,
        "comparison_metrics": ml_config["comparison_metrics"],
        "boundary": ml_config["boundary"],
        "next_action": (
            "Add self-supervised embedding/deep-clustering outputs, then compare "
            "review burden and weak-source compatibility against the U-Net track."
        ),
    }


def gate_recommendation(
    config: dict[str, Any],
    coregistration: dict[str, Any],
    temporal: dict[str, Any],
    reliability: dict[str, Any],
) -> str:
    labels = config["change_publish_gate"]["recommendation_labels"]
    if coregistration.get("status") == "missing" or temporal.get("status") == "missing":
        return labels["hold"]
    if coregistration.get("allowed_for_gate") is False:
        return labels["hold"]
    if reliability.get("status") == "passed" and temporal.get("status") == "passed":
        return labels["publish"]
    return labels["review"]


def build_report() -> dict[str, Any]:
    config = load_config()
    coregistration = summarize_coregistration(config)
    temporal = summarize_temporal_consensus(config)
    reliability = summarize_reliability(config)
    ml_comparison = summarize_ml_comparison(config)
    recommendation = gate_recommendation(config, coregistration, temporal, reliability)

    return {
        "phase": config["phase"],
        "purpose": config["purpose"],
        "publish_gate": {
            "recommendation": recommendation,
            "coregistration": coregistration,
            "temporal_consensus": temporal,
            "reliability": reliability,
            "interpretation": (
                "This gate ranks candidates for publication or review. It is not "
                "a field-validated accuracy assessment."
            ),
        },
        "ml_comparison_plan": ml_comparison,
        "public_demo_packaging": {
            "manifest": "data/outputs/public_demo_manifest.json",
            "policy": config["public_demo_packaging"]["note"],
            "private_roots": config["public_demo_packaging"]["private_roots"],
        },
        "hosted_backend": {
            "current_mode": config["hosted_backend"]["current_mode"],
            "deploy_when_required": config["hosted_backend"]["deploy_when_required"],
            "production_requirements": config["hosted_backend"]["production_requirements"],
        },
    }


def write_report(report: dict[str, Any]) -> None:
    for path in [OUTPUT_PATH, WEB_SUMMARY_PATH, DOCS_SUMMARY_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    report = build_report()
    write_report(report)
    print(f"Improvement track report: {OUTPUT_PATH}")
    print(json.dumps(report["publish_gate"], indent=2))


if __name__ == "__main__":
    main()
