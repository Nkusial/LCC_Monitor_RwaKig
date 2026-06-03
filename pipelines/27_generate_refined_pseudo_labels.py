"""Generate strict pseudo-label rasters for confidence-improved retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
UNET = ROOT / "data" / "interim" / "unet_full_aoi"
EXTERNAL = ROOT / "data" / "interim" / "external_sources"
OUTPUT_DIR = ROOT / "data" / "interim" / "refined_pseudo_labels"
REPORT = ROOT / "data" / "outputs" / "refined_pseudo_labels_report.json"

MODEL_TO_HARMONIZED = {
    # Current U-Net classes are compact for the WebGIS, while weak sources use
    # a richer harmonized taxonomy. Compatibility maps prevent false agreement.
    1: {1},
    2: {2, 3},
    3: {4},
    4: {5},
    5: {6},
}
HARMONIZED_TO_MODEL = {
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def write_raster(path: Path, data: np.ndarray, profile: dict[str, Any], dtype: str, nodata: int | float) -> None:
    output_profile = profile.copy()
    output_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data.astype(dtype), 1)


def compatible(prediction: np.ndarray, agreement: np.ndarray) -> np.ndarray:
    mask = np.zeros(prediction.shape, dtype=bool)
    for model_id, harmonized_ids in MODEL_TO_HARMONIZED.items():
        mask |= (prediction == model_id) & np.isin(agreement, list(harmonized_ids))
    return mask


def harmonized_to_model_labels(agreement: np.ndarray) -> np.ndarray:
    """Convert compact harmonized weak-source classes to the current U-Net class space."""
    labels = np.zeros(agreement.shape, dtype="uint8")
    for harmonized_id, model_id in HARMONIZED_TO_MODEL.items():
        labels[agreement == harmonized_id] = model_id
    return labels


def latest_validation_record(manifest: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in manifest.get("records", []) if record.get("split") == "validation"]
    if not records:
        records = manifest.get("records", [])
    if not records:
        raise RuntimeError("No U-Net full-AOI records found.")
    return sorted(records, key=lambda record: record["training_tag"])[-1]


def refine_record(
    record: dict[str, Any],
    confidence_threshold: float,
    entropy_threshold: float,
    consensus_threshold: float,
    agreement: np.ndarray,
    agreement_count: np.ndarray,
    disagreement: np.ndarray,
    include_external_agreement: bool,
) -> dict[str, Any]:
    """Build strict pseudo-label rasters for one full-AOI model record."""
    tag = record["training_tag"]

    prediction, profile = read_raster(UNET / f"{tag}_unet_full_aoi_dominant_class.tif")
    confidence, _ = read_raster(UNET / f"{tag}_unet_full_aoi_confidence.tif")
    entropy, _ = read_raster(UNET / f"{tag}_unet_full_aoi_entropy.tif")
    consensus, _ = read_raster(UNET / "unet_temporal_consensus_all_fraction.tif")

    strict_mask = (
        # A pixel becomes a strict pseudo-label only when model probability,
        # entropy, temporal persistence, and independent weak sources agree.
        (prediction > 0)
        & (confidence >= confidence_threshold)
        & (entropy <= entropy_threshold)
        & (consensus >= consensus_threshold)
        & (agreement > 0)
        & (agreement_count >= 2)
        & (disagreement == 0)
        & compatible(prediction, agreement)
    )
    external_labels = harmonized_to_model_labels(agreement)
    external_agreement_mask = (
        # External-only agreement is allowed to reduce self-training bias; it
        # gives the model new evidence instead of recycling only its own output.
        include_external_agreement
        & (external_labels > 0)
        & (agreement_count >= 2)
        & (disagreement == 0)
    )
    refined_mask = strict_mask | external_agreement_mask
    refined = np.where(strict_mask, prediction, np.where(external_agreement_mask, external_labels, 0)).astype("uint8")
    external_confidence = np.clip(agreement_count.astype("float32") / 4.0, 0.0, 1.0)
    refined_confidence = np.where(
        # Refined confidence is the weakest supporting signal, not raw model
        # probability. This keeps pseudo-labels conservative.
        strict_mask,
        np.minimum.reduce([confidence.astype("float32"), 1.0 - entropy.astype("float32"), consensus.astype("float32")]),
        np.where(external_agreement_mask, external_confidence, -9999),
    ).astype("float32")

    label_path = OUTPUT_DIR / f"{tag}_refined_pseudo_labels.tif"
    mask_path = OUTPUT_DIR / f"{tag}_refined_training_mask.tif"
    confidence_path = OUTPUT_DIR / f"{tag}_refined_label_confidence.tif"
    write_raster(label_path, refined, profile, "uint8", 0)
    write_raster(mask_path, refined_mask.astype("uint8"), profile, "uint8", 0)
    write_raster(confidence_path, refined_confidence, profile, "float32", -9999)

    valid_model_pixels = int(np.count_nonzero(prediction > 0))
    selected_pixels = int(np.count_nonzero(refined_mask))
    return {
        "training_tag": tag,
        "year": int(record.get("year", 0)),
        "split": record.get("split", "unknown"),
        "selected_pixels": selected_pixels,
        "valid_model_pixels": valid_model_pixels,
        "selected_fraction_of_valid": round(selected_pixels / valid_model_pixels, 6) if valid_model_pixels else 0,
        "strict_model_compatible_pixels": int(np.count_nonzero(strict_mask)),
        "external_agreement_pixels": int(np.count_nonzero(external_agreement_mask)),
        "class_counts": {str(class_id): int(np.count_nonzero(refined == class_id)) for class_id in range(1, 6)},
        "outputs": {
            "labels": str(label_path.relative_to(ROOT)).replace("\\", "/"),
            "training_mask": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
            "label_confidence": str(confidence_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }


def build_refined_labels(
    confidence_threshold: float,
    entropy_threshold: float,
    consensus_threshold: float,
    all_records: bool,
    include_external_agreement: bool,
) -> dict[str, Any]:
    manifest = read_json(UNET / "unet_full_aoi_manifest.json")
    agreement, _ = read_raster(EXTERNAL / "weak_source_agreement_labels.tif")
    agreement_count, _ = read_raster(EXTERNAL / "weak_source_agreement_count.tif")
    disagreement, _ = read_raster(EXTERNAL / "weak_source_disagreement_mask.tif")
    records = manifest.get("records", []) if all_records else [latest_validation_record(manifest)]
    if not records:
        raise RuntimeError("No U-Net full-AOI records found.")
    refined_records = [
        refine_record(
            record,
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold,
            consensus_threshold=consensus_threshold,
            agreement=agreement,
            agreement_count=agreement_count,
            disagreement=disagreement,
            include_external_agreement=include_external_agreement,
        )
        for record in records
    ]
    selected_pixels = sum(record["selected_pixels"] for record in refined_records)
    valid_model_pixels = sum(record["valid_model_pixels"] for record in refined_records)
    class_counts = {
        str(class_id): sum(record["class_counts"][str(class_id)] for record in refined_records)
        for class_id in range(1, 6)
    }
    report = {
        "phase": "phase_45_strict_refined_pseudo_labels",
        "record_count": len(refined_records),
        "thresholds": {
            "minimum_unet_confidence": confidence_threshold,
            "maximum_entropy": entropy_threshold,
            "minimum_temporal_consensus": consensus_threshold,
            "minimum_weak_agreement_sources": 2,
            "include_external_agreement_pixels": include_external_agreement,
        },
        "selected_pixels": selected_pixels,
        "valid_model_pixels": valid_model_pixels,
        "selected_fraction_of_valid": round(selected_pixels / valid_model_pixels, 6) if valid_model_pixels else 0,
        "class_counts": class_counts,
        "records": refined_records,
        "notes": [
            "These pseudo-labels are intentionally conservative.",
            "Strict model-compatible pixels require U-Net confidence, low entropy, temporal consensus, weak-source agreement, and no weak-source disagreement.",
            "High-agreement external weak-source pixels are retained to reduce model self-training bias and class collapse.",
            "The all-record mode creates date-specific pseudo-label rasters for weakly supervised multi-date retraining.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strict refined pseudo-labels for retraining.")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--entropy", type=float, default=0.35)
    parser.add_argument("--consensus", type=float, default=0.7)
    parser.add_argument("--all-records", action="store_true", help="Generate pseudo-labels for every full-AOI record.")
    parser.add_argument(
        "--strict-model-only",
        action="store_true",
        help="Disable external agreement fallback and use only model-compatible refined pixels.",
    )
    args = parser.parse_args()

    report = build_refined_labels(
        args.confidence,
        args.entropy,
        args.consensus,
        all_records=args.all_records,
        include_external_agreement=not args.strict_model_only,
    )
    print(f"Refined pseudo-label report: {REPORT}")
    print(json.dumps({"selected_fraction_of_valid": report["selected_fraction_of_valid"]}, indent=2))


if __name__ == "__main__":
    main()
