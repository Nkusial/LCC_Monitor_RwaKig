"""Compare full-AOI U-Net outputs with available weak-label evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "unet_full_aoi" / "unet_full_aoi_manifest.json"
DEFAULT_EXTERNAL_SUMMARY = ROOT / "data" / "interim" / "external_sources" / "external_weak_sources_summary.json"
DEFAULT_REPORT = ROOT / "data" / "outputs" / "unet_full_aoi_validation_report.json"

# The U-Net baseline merges managed and natural vegetation into one class. For
# validation, either harmonized vegetation class is treated as compatible.
MODEL_TO_HARMONIZED = {
    1: {1},  # built_up
    2: {2, 3},  # managed_vegetation or natural_vegetation
    3: {4},  # bare_soil
    4: {5},  # water_wetland
    5: {6},  # uncertain_mixed
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(path: Path) -> str:
    """Return a stable project-relative path for reports."""
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.relative_to(ROOT)).replace("\\", "/")


def raster_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def compatible_mask(prediction: "Any", agreement: "Any") -> "Any":
    """Return pixels where model class is compatible with weak-source class."""
    import numpy as np

    compatible = np.zeros(prediction.shape, dtype=bool)
    for model_id, harmonized_ids in MODEL_TO_HARMONIZED.items():
        model_pixels = prediction == model_id
        class_pixels = np.isin(agreement, list(harmonized_ids))
        compatible |= model_pixels & class_pixels
    return compatible


def load_weak_evidence(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    outputs = summary.get("agreement_outputs", {})
    required = ["agreement_labels", "source_count", "disagreement_mask"]
    missing = [key for key in required if key not in outputs]
    if missing:
        raise KeyError(f"External weak-source summary is missing: {', '.join(missing)}")
    return {
        "summary": summary,
        "agreement_labels": raster_path(outputs["agreement_labels"]),
        "source_count": raster_path(outputs["source_count"]),
        "disagreement_mask": raster_path(outputs["disagreement_mask"]),
    }


def validate_record(
    record: dict[str, Any],
    weak: dict[str, Any],
    entropy_threshold: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Summarize one full-AOI U-Net record against static weak evidence rasters."""
    import numpy as np
    import rasterio

    rasters = record["rasters"]
    dominant_path = raster_path(rasters["dominant_class"])
    confidence_path = raster_path(rasters["confidence"])
    entropy_path = raster_path(rasters["entropy"])

    with rasterio.open(dominant_path) as src:
        prediction = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(confidence_path) as src:
        confidence = src.read(1)
    with rasterio.open(entropy_path) as src:
        entropy = src.read(1)
    with rasterio.open(weak["agreement_labels"]) as src:
        agreement = src.read(1)
        weak_profile = src.profile.copy()
    with rasterio.open(weak["source_count"]) as src:
        source_count = src.read(1)
    with rasterio.open(weak["disagreement_mask"]) as src:
        disagreement = src.read(1)

    if (
        profile["width"] != weak_profile["width"]
        or profile["height"] != weak_profile["height"]
        or profile["transform"] != weak_profile["transform"]
        or profile["crs"] != weak_profile["crs"]
    ):
        raise ValueError(f"U-Net output is not aligned with weak evidence grid: {dominant_path}")

    valid_model = prediction > 0
    agreement_zone = valid_model & (agreement > 0) & (source_count >= 2) & (disagreement == 0)
    disagreement_zone = valid_model & (disagreement > 0)
    high_entropy_zone = valid_model & (entropy >= entropy_threshold)
    low_confidence_zone = valid_model & (confidence < confidence_threshold)
    review_zone = high_entropy_zone | low_confidence_zone | disagreement_zone
    compatible = compatible_mask(prediction, agreement) & agreement_zone

    valid_count = int(np.count_nonzero(valid_model))
    agreement_count = int(np.count_nonzero(agreement_zone))
    compatible_count = int(np.count_nonzero(compatible))

    return {
        "training_tag": record["training_tag"],
        "year": record["year"],
        "split": record["split"],
        "valid_model_pixels": valid_count,
        "weak_agreement_zone_pixels": agreement_count,
        "weak_agreement_zone_fraction_of_valid": round(agreement_count / valid_count, 6) if valid_count else 0.0,
        "compatible_pixels_in_agreement_zone": compatible_count,
        "compatible_fraction_in_agreement_zone": round(compatible_count / agreement_count, 6) if agreement_count else 0.0,
        "weak_disagreement_pixels": int(np.count_nonzero(disagreement_zone)),
        "high_entropy_pixels": int(np.count_nonzero(high_entropy_zone)),
        "low_confidence_pixels": int(np.count_nonzero(low_confidence_zone)),
        "candidate_review_pixels": int(np.count_nonzero(review_zone)),
        "candidate_review_fraction_of_valid": round(int(np.count_nonzero(review_zone)) / valid_count, 6)
        if valid_count
        else 0.0,
        "mean_confidence": record["mean_confidence"],
        "mean_entropy": record["mean_entropy"],
        "coverage_fraction": record["coverage_fraction"],
    }


def build_report(
    manifest_path: Path,
    external_summary_path: Path,
    output_path: Path,
    entropy_threshold: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    weak = load_weak_evidence(external_summary_path)

    records = [
        validate_record(record, weak, entropy_threshold, confidence_threshold)
        for record in manifest.get("records", [])
    ]
    if not records:
        raise RuntimeError("No full-AOI U-Net records were found for validation.")

    mean_compatibility = sum(r["compatible_fraction_in_agreement_zone"] for r in records) / len(records)
    mean_review_fraction = sum(r["candidate_review_fraction_of_valid"] for r in records) / len(records)
    ranked_for_review = sorted(records, key=lambda item: item["candidate_review_fraction_of_valid"], reverse=True)

    weak_sources_used = weak["summary"].get("agreement_outputs", {}).get("sources_used", [])
    dynamic_world_note = (
        "Dynamic World is integrated through the local Earth Engine export and contributes to weak-source agreement."
        if "dynamic_world" in weak_sources_used
        else "Dynamic World remains pending until an Earth Engine or local raster export is provided."
    )

    report = {
        "phase": "phase_37_full_aoi_unet_reliability_audit",
        "purpose": "Compare weakly supervised full-AOI U-Net outputs with available weak-source agreement and uncertainty evidence.",
        "input_manifest": project_path(manifest_path),
        "weak_source_summary": project_path(external_summary_path),
        "thresholds": {
            "high_entropy": entropy_threshold,
            "low_confidence": confidence_threshold,
            "minimum_agreement_sources": 2,
        },
        "weak_sources_used": weak_sources_used,
        "limitations": [
            "This is not field accuracy because no independent reference labels are available.",
            dynamic_world_note,
            "The current U-Net class space merges managed and natural vegetation, so vegetation agreement accepts either harmonized vegetation class.",
        ],
        "summary": {
            "record_count": len(records),
            "mean_compatible_fraction_in_agreement_zone": round(mean_compatibility, 6),
            "mean_candidate_review_fraction_of_valid": round(mean_review_fraction, 6),
            "highest_review_priority": ranked_for_review[0]["training_tag"],
        },
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full-AOI U-Net outputs against weak evidence.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--external-summary", type=Path, default=DEFAULT_EXTERNAL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--entropy-threshold", type=float, default=0.65)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    args = parser.parse_args()

    report = build_report(
        manifest_path=args.manifest,
        external_summary_path=args.external_summary,
        output_path=args.output,
        entropy_threshold=args.entropy_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"Full-AOI U-Net reliability report: {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
