"""Report reliability and uncertainty controls without field-label overclaims.

This phase turns existing weak-supervision evidence into a user-facing control
summary. It explains what is likely to improve confidence and what remains
unverified until independent reference labels exist. The frozen 2026 benchmark
is read only; no thresholds, labels, or model settings are changed from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "data" / "outputs"
WEB_DEMO = ROOT / "web" / "public" / "demo"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def round6(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def mean_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return mean(values) if values else None


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def risk_level(value: float, limits: tuple[float, float], inverse: bool = False) -> str:
    low, high = limits
    if inverse:
        if value >= high:
            return "low"
        if value >= low:
            return "moderate"
        return "high"
    if value <= low:
        return "low"
    if value <= high:
        return "moderate"
    return "high"


def build_report() -> dict[str, Any]:
    # This report joins model validation, frozen 2026 benchmark, weak-source
    # calibration, temporal stability, class balance, and topography evidence
    # into one user-facing reliability story.
    validation = read_json(OUTPUTS / "unet_v3_full_aoi_validation_report.json")
    benchmark = read_json(OUTPUTS / "unet_2026_proxy_benchmark_report.json")
    calibration = read_json(OUTPUTS / "weak_confidence_calibration_report.json")
    consensus = read_json(OUTPUTS / "unet_temporal_consensus_report.json")
    refined = read_json(OUTPUTS / "refined_pseudo_labels_report.json")
    balance = read_json(OUTPUTS / "training_patch_balance_report_v3.json")
    topography = read_json(OUTPUTS / "topography_ingestion_report.json")

    validation_records = validation.get("records", [])
    validation_records = [
        record for record in validation_records if record.get("split") in {"train", "validation"}
    ]
    benchmark_record = benchmark["records"][0]

    validation_mean_confidence = mean_field(validation_records, "mean_confidence")
    validation_mean_entropy = mean_field(validation_records, "mean_entropy")
    validation_review_fraction = validation["summary"]["mean_candidate_review_fraction_of_valid"]
    validation_compatibility = validation["summary"]["mean_compatible_fraction_in_agreement_zone"]

    test_mean_confidence = benchmark_record["mean_confidence"]
    test_mean_entropy = benchmark_record["mean_entropy"]
    test_review_fraction = benchmark_record["candidate_review_fraction_of_valid"]
    test_compatibility = benchmark_record["compatible_fraction_in_agreement_zone"]

    valid_test_pixels = benchmark_record["valid_model_pixels"]
    driver_metrics = {
        # Driver metrics are diagnostic, not accuracy. They explain why a pixel
        # or date should be trusted, reviewed, or withheld from publication.
        "confidence_gap_validation_to_2026": round6(
            validation_mean_confidence - test_mean_confidence
            if validation_mean_confidence is not None
            else None
        ),
        "entropy_gap_2026_minus_validation": round6(
            test_mean_entropy - validation_mean_entropy
            if validation_mean_entropy is not None
            else None
        ),
        "review_burden_gap_2026_minus_validation": round6(
            test_review_fraction - validation_review_fraction
        ),
        "weak_compatibility_gap_validation_minus_2026": round6(
            validation_compatibility - test_compatibility
        ),
        "test_low_confidence_fraction": round6(
            ratio(benchmark_record["low_confidence_pixels"], valid_test_pixels)
        ),
        "test_high_entropy_fraction": round6(
            ratio(benchmark_record["high_entropy_pixels"], valid_test_pixels)
        ),
        "test_weak_disagreement_fraction": round6(
            ratio(benchmark_record["weak_disagreement_pixels"], valid_test_pixels)
        ),
        "test_weak_agreement_zone_fraction": round6(
            benchmark_record["weak_agreement_zone_fraction_of_valid"]
        ),
        "strict_train_validation_pseudo_label_fraction": round6(
            refined["selected_fraction_of_valid"]
        ),
        "temporal_consensus_fraction_train_validation": round6(
            consensus["mean_consensus_fraction"]
        ),
        "temporal_instability_fraction_train_validation": round6(
            ratio(consensus["unstable_pixel_count"], consensus["valid_pixel_count"])
        ),
        "weak_confidence_support_gap": round6(
            calibration["mean_absolute_confidence_support_gap"]
        ),
        "rare_class_pixel_fraction_before_balancing": round6(
            ratio(
                sum(
                    int(balance["input_class_pixel_counts"].get(class_id, 0))
                    for class_id in ("3", "4", "5")
                ),
                sum(int(value) for value in balance["input_class_pixel_counts"].values()),
            )
        ),
        "slope_p95_degrees": round6(topography["summary"]["slope_p95_degrees"]),
    }

    risk_drivers = [
        # Risk drivers translate numeric diagnostics into actions a reviewer can
        # understand without needing to read every intermediate raster.
        {
            "driver": "2026 domain/test-scene shift",
            "risk_level": risk_level(
                driver_metrics["confidence_gap_validation_to_2026"], (0.15, 0.3)
            ),
            "evidence": (
                "Mean confidence is much lower on the frozen 2026 benchmark than on "
                "train/validation full-AOI records."
            ),
            "safe_action": (
                "Add cleaner train/validation scenes and retrain later, then evaluate once "
                "on the untouched 2026 benchmark."
            ),
        },
        {
            "driver": "High entropy / class ambiguity",
            "risk_level": risk_level(driver_metrics["test_high_entropy_fraction"], (0.25, 0.55)),
            "evidence": "Large 2026 areas have high normalized entropy.",
            "safe_action": (
                "Use entropy as a review-zone mask and avoid publishing low-certainty "
                "alerts as confident changes."
            ),
        },
        {
            "driver": "Weak-source disagreement",
            "risk_level": risk_level(
                driver_metrics["test_weak_disagreement_fraction"], (0.2, 0.45)
            ),
            "evidence": (
                "Dynamic World, ESA WorldCover, and OSM do not support the same class "
                "everywhere in the 2026 AOI."
            ),
            "safe_action": (
                "Require stronger weak-source agreement for high-confidence zones and "
                "keep disagreement zones visible for review."
            ),
        },
        {
            "driver": "Calibration gap",
            "risk_level": risk_level(driver_metrics["weak_confidence_support_gap"], (0.15, 0.3)),
            "evidence": "Model probability and weak-source support are not yet well aligned.",
            "safe_action": (
                "Report calibrated reliability bands separately from raw model probability."
            ),
        },
        {
            "driver": "Rare-class imbalance",
            "risk_level": risk_level(
                driver_metrics["rare_class_pixel_fraction_before_balancing"], (0.02, 0.08), inverse=True
            ),
            "evidence": "Water/moisture, bare/sparse, and mixed classes are much rarer than vegetation/built-up.",
            "safe_action": (
                "Keep class-balanced patch sampling and rare-class loss weighting in the "
                "next train/validation-only model iteration."
            ),
        },
        {
            "driver": "Mountainous terrain effects",
            "risk_level": risk_level(driver_metrics["slope_p95_degrees"], (15, 25)),
            "evidence": "Kigali AOI has steep terrain that can affect radar, shadow, moisture, and settlement signals.",
            "safe_action": (
                "Use DEM, slope, ruggedness, and TPI as model features and as review cues."
            ),
        },
    ]

    likely_changes_without_expert_samples = [
        {
            "change": "More honest confidence bands",
            "expected_direction": "Raw confidence may increase or decrease, but reliability interpretation should improve.",
            "why": "Calibration separates model probability from weak-source support instead of treating probability as truth.",
        },
        {
            "change": "Lower review burden on future validation scenes",
            "expected_direction": "Review-zone fraction should decrease if train/validation-only retraining learns cleaner weak labels.",
            "why": "Strict pseudo-label filtering and temporal consensus remove noisy labels before retraining.",
        },
        {
            "change": "Better stability of likely changes",
            "expected_direction": "Single-date false alarms should decline when temporal persistence gates change publication.",
            "why": "Persistent changes across multiple Sentinel-1/2 dates are more reliable than one-date spikes.",
        },
        {
            "change": "Improved class separation in terrain-affected areas",
            "expected_direction": "Some bare/sparse, moisture, and built-up confusions should reduce.",
            "why": "Slope and topographic position explain shadows, drainage, radar geometry, and settlement patterns.",
        },
        {
            "change": "No field-accuracy claim yet",
            "expected_direction": "The system becomes more useful for screening, but not authoritative mapping.",
            "why": "Expert samples are still required for confusion matrix, user accuracy, producer accuracy, and F1.",
        },
    ]

    return {
        "phase": "phase_53_reliability_uncertainty_control",
        "purpose": (
            "Focus the near-real-time monitor on reliability and uncertainty control "
            "when expert field/reference labels are not yet available."
        ),
        "accuracy_claim": "No field accuracy claim; this is model-risk and weak-evidence guidance.",
        "no_cheating_boundary": {
            "frozen_test_set": benchmark_record["training_tag"],
            "allowed": [
                "Explain 2026 uncertainty and review burden.",
                "Use train/validation reports to plan future model improvements.",
                "Keep 2026 metrics as a frozen benchmark for later comparison.",
            ],
            "forbidden": benchmark["non_cheating_protocol"]["forbidden"],
        },
        "current_reliability_snapshot": {
            "train_validation_mean_confidence": round6(validation_mean_confidence),
            "train_validation_mean_entropy": round6(validation_mean_entropy),
            "train_validation_review_fraction": round6(validation_review_fraction),
            "train_validation_weak_compatibility": round6(validation_compatibility),
            "frozen_2026_mean_confidence": round6(test_mean_confidence),
            "frozen_2026_mean_entropy": round6(test_mean_entropy),
            "frozen_2026_review_fraction": round6(test_review_fraction),
            "frozen_2026_weak_compatibility": round6(test_compatibility),
        },
        "driver_metrics": driver_metrics,
        "risk_drivers": risk_drivers,
        "likely_changes_without_expert_samples": likely_changes_without_expert_samples,
        "recommended_next_controls": [
            "Publish reliability bands: supported, caution, review-required.",
            "Gate change alerts using confidence, entropy, weak agreement, and temporal persistence.",
            "Use only train/validation data for the next retraining cycle.",
            "Reserve the frozen 2026 benchmark for a single later comparison after retraining.",
            "Add a small expert sample later to convert reliability into true accuracy metrics.",
        ],
    }


def main() -> None:
    report = build_report()
    output_path = OUTPUTS / "reliability_uncertainty_control_report.json"
    web_path = WEB_DEMO / "reliability_uncertainty_control_summary.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    web_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Reliability uncertainty control report: {output_path.relative_to(ROOT)}")
    print(f"WebGIS summary: {web_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
