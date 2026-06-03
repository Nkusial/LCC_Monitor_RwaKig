"""Compare supervised U-Net and unsupervised/deep-clustering evidence tracks.

The comparison is intentionally reliability-focused. It summarizes what the
current weak-supervised U-Net track and the unsupervised clustering track can
support, then defines the next self-supervised/deep-clustering benchmark without
claiming field accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UNET_VALIDATION = ROOT / "data" / "outputs" / "unet_v3_full_aoi_validation_report.json"
UNET_2026 = ROOT / "data" / "outputs" / "unet_2026_proxy_benchmark_report.json"
UNSUPERVISED = ROOT / "data" / "outputs" / "unsupervised_validation_report.json"
TEMPORAL = ROOT / "data" / "outputs" / "unet_temporal_consensus_report.json"
SELF_SUPERVISED = ROOT / "data" / "outputs" / "self_supervised_embedding_track_report.json"
SELF_SUPERVISED_REVIEW_SAMPLES = ROOT / "data" / "outputs" / "self_supervised_review_sample_manifest.json"
SPATIAL_SELF_SUPERVISED = ROOT / "data" / "outputs" / "spatial_self_supervised_encoder_report.json"
GEOSPATIAL_ALIGNMENT = ROOT / "data" / "outputs" / "geospatial_alignment_report.json"
OUTPUT_PATH = ROOT / "data" / "outputs" / "model_track_comparison_report.json"
WEB_SUMMARY_PATH = ROOT / "web" / "public" / "demo" / "model_track_comparison_summary.json"
DOCS_SUMMARY_PATH = ROOT / "docs" / "app" / "demo" / "model_track_comparison_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT)).replace("\\", "/")}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def round_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def summarize_unet(validation: dict[str, Any], benchmark: dict[str, Any], temporal: dict[str, Any]) -> dict[str, Any]:
    validation_summary = validation.get("summary", {})
    benchmark_summary = benchmark.get("summary", {})
    return {
        "track": "weak_supervised_u_net",
        "purpose": "Class-probability, entropy, and review-zone maps from weak labels.",
        "train_validation": {
            "record_count": validation_summary.get("record_count"),
            "mean_weak_source_compatibility": validation_summary.get("mean_compatible_fraction_in_agreement_zone"),
            "mean_review_fraction": validation_summary.get("mean_candidate_review_fraction_of_valid"),
        },
        "frozen_2026_proxy": {
            "record_count": benchmark_summary.get("record_count"),
            "mean_weak_source_compatibility": benchmark_summary.get("mean_compatible_fraction_in_agreement_zone"),
            "mean_review_fraction": benchmark_summary.get("mean_candidate_review_fraction_of_valid"),
            "caveat": benchmark.get("benchmark_caveat"),
        },
        "temporal_consensus": {
            "record_count": temporal.get("record_count"),
            "mean_consensus_fraction": temporal.get("mean_consensus_fraction"),
            "temporal_instability_fraction": round_or_none(
                safe_get(temporal, "unstable_pixel_count"),
            )
            if temporal.get("valid_pixel_count") in (None, 0)
            else round(float(temporal["unstable_pixel_count"]) / float(temporal["valid_pixel_count"]), 6),
        },
        "strengths": [
            "Produces interpretable class probability, confidence, and entropy surfaces.",
            "Supports wall-to-wall AOI inference after training.",
            "Can use temporal consensus to reduce one-date false alerts.",
        ],
        "limits": [
            "Requires weak labels or expert labels.",
            "May inherit Dynamic World, ESA, OSM, and index-label bias.",
            "The 2026 benchmark shows domain shift and high review burden.",
        ],
    }


def summarize_unsupervised(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_experiment", {})
    return {
        "track": "unsupervised_clustering",
        "purpose": "Feature-space grouping and anomaly/review discovery without training labels.",
        "best_experiment": {
            "monitoring_tag": best.get("monitoring_tag"),
            "feature_set": best.get("feature_set"),
            "features": best.get("features"),
            "reduction": best.get("reduction"),
            "mask_mode": best.get("mask_mode"),
            "cluster_count": best.get("cluster_count"),
            "silhouette_score": best.get("silhouette_score"),
        },
        "ground_truth_available": report.get("ground_truth_available"),
        "accuracy_claim": report.get("accuracy_claim"),
        "strengths": [
            "Works when no labels exist.",
            "Useful for discovering separable spectral/radar regimes and review strata.",
            "Can expose where supervised labels may be too coarse or biased.",
        ],
        "limits": [
            "Clusters are not land-cover classes until interpreted with weak sources or expert review.",
            "Silhouette is feature-space coherence, not map accuracy.",
            "Cluster IDs may change across dates or feature choices.",
        ],
    }


def summarize_self_supervised_track(
    report: dict[str, Any],
    review_samples: dict[str, Any],
    spatial_report: dict[str, Any],
) -> dict[str, Any]:
    if report.get("phase") == "phase_62_self_supervised_embedding_track":
        metrics = report.get("comparison_metrics", {})
        summary = {
            "track": "self_supervised_embedding_deep_clustering",
            "status": "implemented_patch_level",
            "purpose": report.get("purpose"),
            "encoder": report.get("encoder"),
            "deep_clustering": report.get("deep_clustering"),
            "anomaly_detection": report.get("anomaly_detection"),
            "review_comparison": {
                "review_burden_fraction": metrics.get("review_burden_fraction"),
                "weak_source_compatibility": metrics.get("weak_source_compatibility"),
                "mean_cluster_ambiguity": metrics.get("mean_cluster_ambiguity"),
                "high_anomaly_fraction": metrics.get("high_anomaly_fraction"),
            },
            "guardrails": [
                "2026 benchmark is excluded from encoder fitting.",
                "Weak sources interpret clusters after fitting; they do not train the encoder.",
                "Outputs are review-support evidence, not field accuracy.",
            ],
        }
        if review_samples.get("phase") == "phase_64_self_supervised_review_sampling":
            summary["review_sample_proposal"] = {
                "status": "ready_for_review_before_retraining",
                "sample_count": review_samples.get("sample_count"),
                "sample_geojson": review_samples.get("sample_geojson"),
                "sample_counts": review_samples.get("sample_counts"),
                "no_cheating_protocol": review_samples.get("no_cheating_protocol"),
            }
        if spatial_report.get("phase") == "phase_65_spatial_self_supervised_encoder":
            summary["spatial_encoder_upgrade"] = {
                "status": "implemented",
                "encoder": spatial_report.get("encoder"),
                "temporal_encoder_status": spatial_report.get("temporal_encoder_status"),
                "review_comparison": spatial_report.get("comparison_metrics"),
                "quality_gates": spatial_report.get("quality_gates"),
                "no_cheating_protocol": spatial_report.get("no_cheating_protocol"),
            }
        return summary
    return {
        "track": "self_supervised_deep_clustering",
        "status": "planned_not_trained",
        "purpose": (
            "Learn Sentinel-1/2 multi-date embeddings without class labels, then "
            "cluster or score anomalies for comparison against the current U-Net track."
        ),
        "minimum_inputs": [
            "AOI-aligned Sentinel-2 optical indices and bands",
            "AOI-aligned Sentinel-1 VV, VH, and VV/VH features",
            "multi-date temporal windows that exclude the frozen 2026 benchmark during training",
            "weak-source interpretation layers from Dynamic World, ESA WorldCover, OSM, indices, and radar evidence",
        ],
        "expected_outputs": [
            "embedding features per pixel or patch",
            "deep clusters or anomaly scores",
            "cluster interpretation table",
            "review-zone layer for high-uncertainty or weak-source-disagreement areas",
        ],
        "guardrails": [
            "Do not train on the frozen 2026 benchmark.",
            "Do not rename clusters as land-cover classes until weak-source or expert interpretation supports them.",
            "Use the same master-grid alignment checks as the U-Net and raster-publication tracks.",
        ],
    }


def build_recommendation(
    unet: dict[str, Any],
    unsupervised: dict[str, Any],
    self_supervised: dict[str, Any],
    geospatial_alignment: dict[str, Any],
) -> dict[str, Any]:
    train_review = safe_get(unet, "train_validation", "mean_review_fraction")
    test_review = safe_get(unet, "frozen_2026_proxy", "mean_review_fraction")
    best_silhouette = safe_get(unsupervised, "best_experiment", "silhouette_score")
    self_review = safe_get(self_supervised, "review_comparison", "review_burden_fraction")
    self_compat = safe_get(self_supervised, "review_comparison", "weak_source_compatibility")
    review_samples = safe_get(self_supervised, "review_sample_proposal", "sample_count")
    spatial_review = safe_get(self_supervised, "spatial_encoder_upgrade", "review_comparison", "review_burden_fraction")
    spatial_compat = safe_get(
        self_supervised,
        "spatial_encoder_upgrade",
        "review_comparison",
        "weak_source_compatibility",
    )
    spatial_encoder_type = safe_get(self_supervised, "spatial_encoder_upgrade", "encoder", "type")

    return {
        "current_best_use": (
            "Use U-Net for candidate land-cover probability and uncertainty layers, "
            "then use unsupervised and self-supervised embedding tracks as independent review and discovery layers."
        ),
        "do_not_do": [
            "Do not compare U-Net confidence directly with silhouette as if they were the same metric.",
            "Do not treat weak-source compatibility as field accuracy.",
            "Do not tune the model using the frozen 2026 benchmark.",
        ],
        "why": {
            "u_net_review_fraction_train_validation": train_review,
            "u_net_review_fraction_frozen_2026": test_review,
            "best_unsupervised_silhouette": best_silhouette,
            "self_supervised_review_burden_fraction": self_review,
            "self_supervised_weak_source_compatibility": self_compat,
            "self_supervised_review_sample_count": review_samples,
            "spatial_encoder_review_burden_fraction": spatial_review,
            "spatial_encoder_weak_source_compatibility": spatial_compat,
            "spatial_encoder_type": spatial_encoder_type,
            "geospatial_alignment_status": geospatial_alignment.get("status"),
            "raster_mismatch_count": geospatial_alignment.get("raster_mismatch_count"),
            "vector_mismatch_count": geospatial_alignment.get("vector_mismatch_count"),
            "interpretation": (
                "U-Net currently gives usable train/validation screening layers, but "
                "2026 remains a high-review stress case. Unsupervised clustering gives "
                "moderate feature coherence. The self-supervised embedding track is a "
                "label-free anomaly/review layer; it should guide review sampling rather "
                "than replace class labels."
            ),
        },
        "next_self_supervised_track": [
            "Upgrade from patch descriptor embeddings to spatial convolutional or transformer encoders when compute allows.",
            "Use the embedding anomaly score to propose review samples before retraining U-Net.",
            "Compare review burden, weak-source compatibility, temporal stability, and geospatial QA against U-Net after each frozen run.",
            "Keep the frozen 2026 benchmark untouched for final comparison runs only.",
        ],
    }


def build_report() -> dict[str, Any]:
    validation = read_json(UNET_VALIDATION)
    benchmark = read_json(UNET_2026)
    unsupervised_report = read_json(UNSUPERVISED)
    temporal = read_json(TEMPORAL)
    self_supervised_report = read_json(SELF_SUPERVISED)
    review_samples = read_json(SELF_SUPERVISED_REVIEW_SAMPLES)
    spatial_report = read_json(SPATIAL_SELF_SUPERVISED)
    geospatial_alignment = read_json(GEOSPATIAL_ALIGNMENT)
    unet = summarize_unet(validation, benchmark, temporal)
    unsupervised = summarize_unsupervised(unsupervised_report)
    self_supervised = summarize_self_supervised_track(self_supervised_report, review_samples, spatial_report)
    train_review = safe_get(unet, "train_validation", "mean_review_fraction")
    frozen_review = safe_get(unet, "frozen_2026_proxy", "mean_review_fraction")
    self_review = safe_get(self_supervised, "review_comparison", "review_burden_fraction")
    self_compat = safe_get(self_supervised, "review_comparison", "weak_source_compatibility")

    return {
        "phase": "phase_58_model_track_comparison",
        "purpose": "Compare weak-supervised U-Net outputs with unsupervised clustering and self-supervised embedding evidence.",
        "accuracy_claim": "No field accuracy claim; this is reliability, review-burden, and weak-evidence comparison.",
        "tracks": {
            "supervised": unet,
            "unsupervised": unsupervised,
            "self_supervised": self_supervised,
        },
        "recommendation": build_recommendation(unet, unsupervised, self_supervised, geospatial_alignment),
        "comparison_metrics": {
            "review_fraction": {
                "u_net_train_validation": train_review,
                "u_net_frozen_2026": frozen_review,
                "self_supervised_embedding": self_review,
                "spatial_self_supervised_encoder": safe_get(
                    self_supervised,
                    "spatial_encoder_upgrade",
                    "review_comparison",
                    "review_burden_fraction",
                ),
            },
            "weak_source_compatibility": {
                "u_net_train_validation": safe_get(unet, "train_validation", "mean_weak_source_compatibility"),
                "u_net_frozen_2026": safe_get(unet, "frozen_2026_proxy", "mean_weak_source_compatibility"),
                "self_supervised_embedding": self_compat,
                "spatial_self_supervised_encoder": safe_get(
                    self_supervised,
                    "spatial_encoder_upgrade",
                    "review_comparison",
                    "weak_source_compatibility",
                ),
            },
        },
        "quality_gates": {
            "temporal_consensus": {
                "record_count": temporal.get("record_count"),
                "mean_consensus_fraction": temporal.get("mean_consensus_fraction"),
                "temporal_instability_fraction": (
                    None
                    if temporal.get("valid_pixel_count") in (None, 0)
                    else round(
                        float(temporal.get("unstable_pixel_count", 0))
                        / float(temporal["valid_pixel_count"]),
                        6,
                    )
                ),
            },
            "geospatial_alignment": {
                "status": geospatial_alignment.get("status"),
                "checked_raster_count": geospatial_alignment.get("checked_raster_count"),
                "raster_mismatch_count": geospatial_alignment.get("raster_mismatch_count"),
                "vector_mismatch_count": geospatial_alignment.get("vector_mismatch_count"),
            },
            "frozen_2026_boundary": "Use 2026 only for locked-model comparison runs, not selection or retraining.",
        },
        "comparison_dimensions": [
            "review_fraction",
            "weak_source_compatibility",
            "temporal_consensus",
            "entropy_or_cluster_ambiguity",
            "geospatial_alignment_status",
            "frozen_2026_no_cheating_boundary",
        ],
    }


def write_report(report: dict[str, Any]) -> None:
    for path in [OUTPUT_PATH, WEB_SUMMARY_PATH, DOCS_SUMMARY_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    report = build_report()
    write_report(report)
    print(f"Model track comparison report: {OUTPUT_PATH}")
    print(json.dumps(report["recommendation"], indent=2))


if __name__ == "__main__":
    main()
