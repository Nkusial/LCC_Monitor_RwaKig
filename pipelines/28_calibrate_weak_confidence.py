"""Calibrate model confidence against weak-source agreement zones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
UNET = ROOT / "data" / "interim" / "unet_full_aoi"
EXTERNAL = ROOT / "data" / "interim" / "external_sources"
REPORT = ROOT / "data" / "outputs" / "weak_confidence_calibration_report.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def latest_validation_tag() -> str:
    manifest = read_json(UNET / "unet_full_aoi_manifest.json")
    records = [record for record in manifest["records"] if record["split"] == "validation"]
    return sorted(records, key=lambda record: record["training_tag"])[-1]["training_tag"]


def build_bins(confidence: np.ndarray, supported: np.ndarray, valid: np.ndarray, bins: int) -> list[dict]:
    results = []
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = valid & (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        count = int(np.count_nonzero(mask))
        support_rate = float(np.count_nonzero(mask & supported) / count) if count else None
        midpoint = float((low + high) / 2)
        results.append(
            {
                "confidence_low": round(float(low), 3),
                "confidence_high": round(float(high), 3),
                "bin_midpoint": round(midpoint, 3),
                "pixel_count": count,
                "weak_support_rate": round(support_rate, 6) if support_rate is not None else None,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weak-label confidence calibration bins.")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    tag = latest_validation_tag()
    confidence = read(UNET / f"{tag}_unet_full_aoi_confidence.tif").astype("float32")
    prediction = read(UNET / f"{tag}_unet_full_aoi_dominant_class.tif").astype("uint8")
    agreement = read(EXTERNAL / "weak_source_agreement_labels.tif").astype("uint8")
    count = read(EXTERNAL / "weak_source_agreement_count.tif").astype("uint8")
    disagreement = read(EXTERNAL / "weak_source_disagreement_mask.tif").astype("uint8")

    valid = prediction > 0
    supported = (agreement > 0) & (count >= 2) & (disagreement == 0)
    bins = build_bins(confidence, supported, valid, args.bins)
    usable_bins = [item for item in bins if item["weak_support_rate"] is not None]
    mean_abs_gap = float(
        np.mean([abs(item["bin_midpoint"] - item["weak_support_rate"]) for item in usable_bins])
    ) if usable_bins else None

    report = {
        "phase": "phase_46_weak_confidence_calibration",
        "training_tag": tag,
        "calibration_target": "weak-source support rate, not field accuracy",
        "mean_absolute_confidence_support_gap": round(mean_abs_gap, 6) if mean_abs_gap is not None else None,
        "bins": bins,
        "recommendation": "Use calibrated weak support only as model-risk guidance until independent reference labels exist.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Weak confidence calibration report: {REPORT}")
    print(json.dumps({"mean_absolute_confidence_support_gap": report["mean_absolute_confidence_support_gap"]}, indent=2))


if __name__ == "__main__":
    main()
