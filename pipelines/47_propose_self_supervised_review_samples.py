"""Propose review samples from self-supervised embedding anomalies.

This phase turns the label-free embedding layer into concrete review tasks
before any U-Net retraining. It does not create labels, it does not train a
model, and it never uses the frozen 2026 benchmark for sample selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_GEOJSON = ROOT / "data" / "outputs" / "self_supervised_review_patches.geojson"
INPUT_SUMMARY = ROOT / "data" / "outputs" / "self_supervised_review_layer_summary.json"
COMPARISON_REPORT = ROOT / "data" / "outputs" / "model_track_comparison_report.json"
GEOSPATIAL_ALIGNMENT = ROOT / "data" / "outputs" / "geospatial_alignment_report.json"
TEMPORAL_CONSENSUS = ROOT / "data" / "outputs" / "unet_temporal_consensus_report.json"
OUTPUT_MANIFEST = ROOT / "data" / "outputs" / "self_supervised_review_sample_manifest.json"
OUTPUT_GEOJSON = ROOT / "data" / "outputs" / "self_supervised_review_samples.geojson"
WEB_MANIFEST = ROOT / "web" / "public" / "demo" / "self_supervised_review_sample_manifest.json"
WEB_GEOJSON = ROOT / "web" / "public" / "demo" / "self_supervised_review_samples.geojson"
DOCS_MANIFEST = ROOT / "docs" / "app" / "demo" / "self_supervised_review_sample_manifest.json"
DOCS_GEOJSON = ROOT / "docs" / "app" / "demo" / "self_supervised_review_samples.geojson"

PRIORITY_WEIGHTS = {
    "high_anomaly": 1.0,
    "weak_support_review": 0.75,
    "embedding_supported": 0.15,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def review_score(properties: dict[str, Any]) -> float:
    """Rank samples by anomaly, weak-source gap, and review priority.

    The score is a sampling priority only. It is intentionally not an accuracy
    score and should not be used as a class label.
    """
    anomaly = float(properties.get("anomaly_score", 0))
    weak_source_fraction = float(properties.get("weak_source_fraction", 0))
    weak_gap = max(0.0, 1.0 - weak_source_fraction)
    priority = str(properties.get("review_priority", "embedding_supported"))
    priority_weight = PRIORITY_WEIGHTS.get(priority, 0.25)
    return round((0.5 * anomaly) + (0.35 * weak_gap) + (0.15 * priority_weight), 6)


def recommended_action(priority: str) -> str:
    if priority == "high_anomaly":
        return "Inspect first: possible new land-cover regime, sensor artifact, or model blind spot before any retraining."
    if priority == "weak_support_review":
        return "Review with imagery/OSM/indices and either assign a corrected class or mark as mixed/unknown."
    return "Use as a control sample only after visual review confirms the dominant interpretation."


def add_centroid(properties: dict[str, Any], geometry: dict[str, Any]) -> None:
    """Attach a lightweight centroid for review tables without changing geometry."""
    coordinates = geometry.get("coordinates", [])
    ring = coordinates[0] if coordinates else []
    xs = [point[0] for point in ring if isinstance(point, list) and len(point) >= 2]
    ys = [point[1] for point in ring if isinstance(point, list) and len(point) >= 2]
    if xs and ys:
        properties["centroid_lon"] = round((min(xs) + max(xs)) / 2, 9)
        properties["centroid_lat"] = round((min(ys) + max(ys)) / 2, 9)


def select_balanced_samples(
    features: list[dict[str, Any]],
    max_samples: int,
    max_per_stratum: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select review candidates across priority, class interpretation, and year."""
    candidates: list[dict[str, Any]] = []
    excluded_counts = {"frozen_2026_or_later": 0, "missing_geometry": 0}
    for feature in features:
        properties = dict(feature.get("properties", {}))
        geometry = feature.get("geometry")
        year = int(properties.get("year", 0))
        if year >= 2026:
            excluded_counts["frozen_2026_or_later"] += 1
            continue
        if not geometry:
            excluded_counts["missing_geometry"] += 1
            continue
        properties["review_score"] = review_score(properties)
        properties["sampling_role"] = (
            "candidate_review"
            if properties.get("review_priority") != "embedding_supported"
            else "control_after_review"
        )
        properties["recommended_action"] = recommended_action(
            str(properties.get("review_priority", "embedding_supported"))
        )
        add_centroid(properties, geometry)
        candidates.append({**feature, "properties": properties, "geometry": geometry})

    candidates.sort(
        key=lambda item: (
            item["properties"]["review_score"],
            float(item["properties"].get("anomaly_score", 0)),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    stratum_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for feature in candidates:
        props = feature["properties"]
        stratum = (
            str(props.get("review_priority")),
            str(props.get("dominant_interpretation")),
            int(props.get("year", 0)),
        )
        if stratum_counts[stratum] >= max_per_stratum:
            continue
        stratum_counts[stratum] += 1
        props["sample_rank"] = len(selected) + 1
        selected.append(feature)
        if len(selected) >= max_samples:
            break

    return selected, excluded_counts


def build_manifest(max_samples: int = 40, max_per_stratum: int = 8) -> tuple[dict[str, Any], dict[str, Any]]:
    review_layer = read_json(INPUT_GEOJSON)
    review_summary = read_json(INPUT_SUMMARY)
    comparison = read_json(COMPARISON_REPORT) if COMPARISON_REPORT.exists() else {}
    alignment = read_json(GEOSPATIAL_ALIGNMENT) if GEOSPATIAL_ALIGNMENT.exists() else {}
    temporal = read_json(TEMPORAL_CONSENSUS) if TEMPORAL_CONSENSUS.exists() else {}

    selected, excluded_counts = select_balanced_samples(
        review_layer.get("features", []),
        max_samples=max_samples,
        max_per_stratum=max_per_stratum,
    )
    priority_counts: dict[str, int] = defaultdict(int)
    interpretation_counts: dict[str, int] = defaultdict(int)
    for feature in selected:
        props = feature["properties"]
        priority_counts[str(props.get("review_priority"))] += 1
        interpretation_counts[str(props.get("dominant_interpretation"))] += 1

    sample_geojson = {
        "type": "FeatureCollection",
        "name": "self_supervised_review_samples",
        "features": selected,
    }
    manifest = {
        "phase": "phase_64_self_supervised_review_sampling",
        "purpose": "Use label-free embedding anomaly and weak-source support to propose balanced review samples before future U-Net retraining.",
        "accuracy_claim": "No field accuracy claim; these are candidate samples for human or independent review.",
        "source_layer": relative(INPUT_GEOJSON),
        "sample_geojson": relative(OUTPUT_GEOJSON),
        "sample_count": len(selected),
        "selection_strategy": {
            "score": "0.50 * anomaly_score + 0.35 * weak_source_gap + 0.15 * review_priority_weight",
            "balanced_by": ["review_priority", "dominant_interpretation", "year"],
            "max_samples": max_samples,
            "max_per_stratum": max_per_stratum,
            "excluded_counts": excluded_counts,
        },
        "sample_counts": {
            "by_review_priority": dict(priority_counts),
            "by_dominant_interpretation": dict(interpretation_counts),
        },
        "review_label_schema": {
            "future_output": "data/interim/retraining/self_supervised_corrected_review_labels.geojson",
            "required_fields": [
                "sample_rank",
                "reviewer_label",
                "reviewer_confidence",
                "review_decision",
                "evidence_notes",
            ],
            "allowed_review_decisions": [
                "accept_interpretation",
                "correct_class",
                "mixed_or_uncertain",
                "sensor_or_alignment_issue",
                "exclude_from_training",
            ],
        },
        "spatial_encoder_upgrade_path": [
            "Keep current MLP descriptor autoencoder as the CPU baseline.",
            "When compute allows, train a convolutional patch autoencoder on 128 x 128 Sentinel-1/2 feature tensors.",
            "For multi-date modeling, add a temporal transformer or ConvLSTM encoder with date tokens and no weak labels during fitting.",
            "Use weak sources only after fitting to interpret clusters and propose review samples.",
        ],
        "comparison_after_frozen_runs": {
            "metrics": [
                "review_burden_fraction",
                "weak_source_compatibility",
                "temporal_consensus_fraction",
                "temporal_instability_fraction",
                "geospatial_alignment_status",
                "raster_mismatch_count",
                "vector_mismatch_count",
            ],
            "current_self_supervised_metrics": review_summary.get("comparison_metrics", {}),
            "current_u_net_reference": comparison.get("tracks", {}).get("supervised", {}),
            "temporal_status": {
                "record_count": temporal.get("record_count"),
                "mean_consensus_fraction": temporal.get("mean_consensus_fraction"),
                "valid_pixel_count": temporal.get("valid_pixel_count"),
                "unstable_pixel_count": temporal.get("unstable_pixel_count"),
            },
            "geospatial_qa": {
                "status": alignment.get("status"),
                "raster_mismatch_count": alignment.get("raster_mismatch_count"),
                "vector_mismatch_count": alignment.get("vector_mismatch_count"),
                "checked_raster_count": alignment.get("checked_raster_count"),
            },
        },
        "no_cheating_protocol": {
            "frozen_2026_policy": "Do not use 2026 benchmark patches for review-sample selection, encoder fitting, hyperparameter choice, or U-Net retraining.",
            "allowed": [
                "Use 2023-2025 training/validation patches for review sampling.",
                "Use review samples to collect independent labels before retraining.",
                "Run frozen 2026 only after a model version is locked.",
            ],
            "forbidden": [
                "Selecting samples from 2026 to improve training.",
                "Changing thresholds after inspecting 2026 results.",
                "Treating weak-source compatibility as field accuracy.",
            ],
        },
        "next_steps": [
            "Review the selected polygons in the WebGIS alongside Sentinel overlays.",
            "Create corrected labels only for reviewed samples.",
            "Regenerate training patches with corrected labels and confidence masks.",
            "Retrain U-Net with the same no-2026 boundary.",
            "Compare U-Net, unsupervised clustering, and self-supervised embedding metrics after the frozen run.",
        ],
    }
    return manifest, sample_geojson


def write_outputs(manifest: dict[str, Any], sample_geojson: dict[str, Any]) -> None:
    for path in [OUTPUT_MANIFEST, WEB_MANIFEST, DOCS_MANIFEST]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in [OUTPUT_GEOJSON, WEB_GEOJSON, DOCS_GEOJSON]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sample_geojson), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose self-supervised review samples before U-Net retraining.")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-per-stratum", type=int, default=8)
    args = parser.parse_args()

    manifest, sample_geojson = build_manifest(args.max_samples, args.max_per_stratum)
    write_outputs(manifest, sample_geojson)
    print(f"Self-supervised review sample manifest: {OUTPUT_MANIFEST}")
    print(json.dumps({"sample_count": manifest["sample_count"], "sample_counts": manifest["sample_counts"]}, indent=2))


if __name__ == "__main__":
    main()
