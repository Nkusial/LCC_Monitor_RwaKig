"""Train a lightweight self-supervised Sentinel patch embedding track.

This phase moves beyond the label-free statistical baseline by fitting a small
neural autoencoder on Sentinel-1/2 and terrain patch descriptors. The encoder is
self-supervised because it learns to reconstruct input features, not weak-label
classes. Weak sources are used only after embedding and clustering to interpret
clusters, estimate review burden, and compare against the U-Net track.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")
warnings.filterwarnings(
    "ignore",
    message=".*Stochastic Optimizer: Maximum iterations.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*MiniBatchKMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "training_patches_v3" / "training_patch_manifest_v3_balanced.json"
UNET_VALIDATION = ROOT / "data" / "outputs" / "unet_v3_full_aoi_validation_report.json"
UNET_2026 = ROOT / "data" / "outputs" / "unet_2026_proxy_benchmark_report.json"
OUTPUT_REPORT = ROOT / "data" / "outputs" / "self_supervised_embedding_track_report.json"
EMBEDDING_OUTPUT = ROOT / "data" / "interim" / "self_supervised_embeddings" / "patch_embeddings_v1.npz"
WEB_SUMMARY = ROOT / "web" / "public" / "demo" / "self_supervised_embedding_track_summary.json"
DOCS_SUMMARY = ROOT / "docs" / "app" / "demo" / "self_supervised_embedding_track_summary.json"
LAST_EMBEDDINGS: np.ndarray | None = None
LAST_RECORDS: list[dict[str, Any]] | None = None
LAST_CLUSTER_LABELS: np.ndarray | None = None
LAST_ANOMALY_SCORES: np.ndarray | None = None

CLASS_NAMES = {
    "1": "vegetation",
    "2": "built_up",
    "3": "water_wetness",
    "4": "bare_or_sparse_ground",
    "5": "mixed_or_uncertain",
}

INTERPRETATION_SOURCES = [
    "Dynamic World",
    "ESA WorldCover",
    "OSM buildings/roads/landuse",
    "Sentinel-2 NDVI/NDWI/NDBI and expanded index evidence",
    "Sentinel-1 VV/VH/log-ratio radar evidence",
    "existing high-confidence change polygons",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT)).replace("\\", "/")}
    return json.loads(path.read_text(encoding="utf-8"))


def round_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def feature_summary(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Create patch descriptors from Sentinel/topography tensors only.

    The neural encoder trains on these descriptors rather than weak-label
    classes. Using the training mask here keeps descriptors focused on valid
    evidence pixels without exposing class IDs to the encoder.
    """
    channels = features.astype("float32")
    if channels.ndim != 3:
        raise ValueError("Expected feature tensor with shape channels x height x width")
    if mask is not None and np.any(mask):
        values = channels[:, mask]
    else:
        values = channels.reshape(channels.shape[0], -1)
    values = np.where(np.isfinite(values), values, np.nan)
    stats = []
    for channel in values:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            stats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        stats.extend(
            [
                float(np.nanmean(finite)),
                float(np.nanstd(finite)),
                float(np.nanpercentile(finite, 5)),
                float(np.nanpercentile(finite, 25)),
                float(np.nanpercentile(finite, 50)),
                float(np.nanpercentile(finite, 75)),
            ]
        )
    return np.asarray(stats, dtype="float32")


def dominant_weak_label(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize weak-label evidence for post-hoc interpretation only."""
    histogram = {str(key): int(value) for key, value in record.get("class_pixel_counts", {}).items()}
    if not histogram:
        return {"class_id": "0", "class_name": "unlabeled", "fraction": 0.0}
    total = sum(histogram.values())
    class_id, count = max(histogram.items(), key=lambda item: item[1])
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "fraction": round(float(count / total), 6) if total else 0.0,
    }


def load_patch_records(manifest: dict[str, Any], max_records: int | None = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load patch descriptors while enforcing the no-2026-training boundary."""
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for record in manifest.get("patches", []):
        if int(record.get("year", 0)) >= 2026:
            continue
        patch_path = ROOT / record["path"]
        if not patch_path.exists():
            continue
        with np.load(patch_path) as patch:
            features = patch["features"]
            training_mask = patch["training_mask"].astype(bool) if "training_mask" in patch else None
        vectors.append(feature_summary(features, training_mask))
        records.append(
            {
                "patch_id": record["patch_id"],
                "path": record["path"],
                "training_tag": record["training_tag"],
                "year": int(record["year"]),
                "split": record["split"],
                "spatial_block_id": record.get("spatial_block_id"),
                "row": record.get("row"),
                "col": record.get("col"),
                "height": int(record.get("height", 128)),
                "width": int(record.get("width", 128)),
                "transform": record.get("transform"),
                "agreement_fraction": round_or_none(record.get("agreement_fraction")),
                "valid_feature_fraction": round_or_none(record.get("valid_feature_fraction")),
                "dominant_weak_label": dominant_weak_label(record),
                "class_pixel_counts": record.get("class_pixel_counts", {}),
            }
        )
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise RuntimeError("No patch records were available for self-supervised embedding.")
    return records, np.vstack(vectors).astype("float32")


def hidden_layer_embedding(model: MLPRegressor, scaled_vectors: np.ndarray) -> np.ndarray:
    """Extract the one-hidden-layer autoencoder bottleneck activations."""
    weights = model.coefs_[0]
    bias = model.intercepts_[0]
    hidden = scaled_vectors @ weights + bias
    if model.activation == "relu":
        hidden = np.maximum(hidden, 0.0)
    elif model.activation == "tanh":
        hidden = np.tanh(hidden)
    return hidden.astype("float32")


def normalized_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray([value / total for value in counts.values() if value > 0], dtype="float64")
    entropy = -np.sum(probabilities * np.log(probabilities))
    maximum = np.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    return round(float(entropy / maximum), 6) if maximum else 0.0


def cluster_profile(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
) -> list[dict[str, Any]]:
    """Profile clusters using weak sources after self-supervised fitting."""
    grouped: dict[int, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for record, label, anomaly in zip(records, labels, anomaly_scores, strict=True):
        grouped[int(label)].append((record, float(anomaly)))

    profiles = []
    for label, cluster_records in sorted(grouped.items()):
        label_counts = Counter(record["dominant_weak_label"]["class_name"] for record, _ in cluster_records)
        year_counts = Counter(str(record["year"]) for record, _ in cluster_records)
        top_name, top_count = label_counts.most_common(1)[0]
        weak_fractions = [record["dominant_weak_label"]["fraction"] for record, _ in cluster_records]
        anomalies = [anomaly for _, anomaly in cluster_records]
        profiles.append(
            {
                "cluster_id": label + 1,
                "patch_count": len(cluster_records),
                "dominant_interpretation": top_name,
                "dominant_interpretation_fraction": round(top_count / len(cluster_records), 6),
                "mean_dominant_weak_label_fraction": round(float(np.mean(weak_fractions)), 6),
                "mean_anomaly_score": round(float(np.mean(anomalies)), 6),
                "weak_label_mix": dict(label_counts),
                "weak_label_mix_entropy": normalized_entropy(dict(label_counts)),
                "year_mix": dict(year_counts),
                "example_patches": [record["patch_id"] for record, _ in cluster_records[:5]],
            }
        )
    return profiles


def summarize_unet_comparison(self_metrics: dict[str, Any]) -> dict[str, Any]:
    validation = read_json(UNET_VALIDATION)
    benchmark = read_json(UNET_2026)
    validation_summary = validation.get("summary", {})
    benchmark_summary = benchmark.get("summary", {})
    train_review = validation_summary.get("mean_candidate_review_fraction_of_valid")
    train_compat = validation_summary.get("mean_compatible_fraction_in_agreement_zone")
    test_review = benchmark_summary.get("mean_candidate_review_fraction_of_valid")
    test_compat = benchmark_summary.get("mean_compatible_fraction_in_agreement_zone")
    return {
        "u_net_track": {
            "train_validation_review_fraction": train_review,
            "train_validation_weak_source_compatibility": train_compat,
            "frozen_2026_review_fraction": test_review,
            "frozen_2026_weak_source_compatibility": test_compat,
        },
        "self_supervised_track": {
            "review_burden_fraction": self_metrics["review_burden_fraction"],
            "weak_source_compatibility": self_metrics["weak_source_compatibility"],
            "cluster_ambiguity": self_metrics["mean_cluster_ambiguity"],
            "high_anomaly_fraction": self_metrics["high_anomaly_fraction"],
        },
        "interpretation": (
            "The self-supervised track is compared as a review and discovery layer, not as a replacement "
            "for class accuracy. Lower review burden and higher weak-source compatibility are useful only "
            "if no-cheating and geospatial-alignment checks remain satisfied."
        ),
    }


def build_report(
    manifest_path: Path = DEFAULT_MANIFEST,
    cluster_count: int = 8,
    embedding_dim: int = 16,
    max_records: int | None = None,
    max_iter: int = 120,
    random_seed: int = 42,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    records, feature_vectors = load_patch_records(manifest, max_records=max_records)
    split_counts = Counter(record["split"] for record in records)
    train_index = np.array([record["split"] == "train" for record in records])
    if np.count_nonzero(train_index) < max(cluster_count, 10):
        raise RuntimeError("Not enough train patches for the requested self-supervised run.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(feature_vectors[train_index])
    all_scaled = scaler.transform(feature_vectors)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(embedding_dim,),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
    )
    autoencoder.fit(train_scaled, train_scaled)
    embedding_train = hidden_layer_embedding(autoencoder, train_scaled)
    embedding_all = hidden_layer_embedding(autoencoder, all_scaled)
    global LAST_EMBEDDINGS, LAST_RECORDS, LAST_CLUSTER_LABELS, LAST_ANOMALY_SCORES
    LAST_EMBEDDINGS = embedding_all
    LAST_RECORDS = records
    reconstructed = autoencoder.predict(all_scaled)
    reconstruction_error = np.mean((all_scaled - reconstructed) ** 2, axis=1)

    model = MiniBatchKMeans(n_clusters=cluster_count, random_state=random_seed, batch_size=8192, n_init="auto")
    model.fit(embedding_train)
    labels = model.predict(embedding_all)
    center_distances = np.linalg.norm(embedding_all - model.cluster_centers_[labels], axis=1)
    raw_anomaly = 0.5 * reconstruction_error + 0.5 * center_distances
    low, high = np.percentile(raw_anomaly, [5, 95])
    anomaly_scores = np.clip((raw_anomaly - low) / (high - low + 1e-9), 0.0, 1.0)
    LAST_CLUSTER_LABELS = labels
    LAST_ANOMALY_SCORES = anomaly_scores

    score = None
    if len(np.unique(labels)) > 1 and len(records) > cluster_count:
        sample_size = min(500, len(records))
        rng = np.random.default_rng(random_seed)
        sample = rng.choice(len(records), size=sample_size, replace=False)
        score = round(float(silhouette_score(embedding_all[sample], labels[sample])), 6)

    profiles = cluster_profile(records, labels, anomaly_scores)
    dominant_fractions = np.asarray([record["dominant_weak_label"]["fraction"] for record in records], dtype="float32")
    weak_compatibility = round(float(np.mean(dominant_fractions)), 6)
    low_support = dominant_fractions < 0.65
    high_anomaly = anomaly_scores >= 0.9
    review_burden = round(float(np.mean(low_support | high_anomaly)), 6)
    high_anomaly_fraction = round(float(np.mean(high_anomaly)), 6)
    coherent_profiles = [profile for profile in profiles if profile["dominant_interpretation_fraction"] >= 0.65]
    mean_cluster_ambiguity = round(float(np.mean([profile["weak_label_mix_entropy"] for profile in profiles])), 6)
    metrics = {
        "silhouette_score": score,
        "weak_source_compatibility": weak_compatibility,
        "review_burden_fraction": review_burden,
        "high_anomaly_fraction": high_anomaly_fraction,
        "mean_cluster_ambiguity": mean_cluster_ambiguity,
        "coherent_cluster_fraction": round(len(coherent_profiles) / len(profiles), 6) if profiles else 0.0,
    }

    return {
        "phase": "phase_62_self_supervised_embedding_track",
        "purpose": "Learn Sentinel-1/2 multi-date patch embeddings without class labels, then use deep clustering and anomaly scoring for review-focused comparison with U-Net.",
        "accuracy_claim": "No field accuracy claim; this is self-supervised representation, cluster interpretation, anomaly screening, and weak-evidence comparison.",
        "no_cheating_protocol": {
            "excluded_years_from_fit": [2026],
            "fit_inputs": "Sentinel-1/2, index, radar, and terrain patch descriptors only; weak-label classes are not used to fit the scaler, autoencoder, or clusters.",
            "weak_labels_role": "post-hoc interpretation from Dynamic World, ESA WorldCover, OSM, Sentinel indices/radar evidence, and existing high-confidence changes",
            "frozen_test_boundary": "2026 benchmark stays evaluation-only and is not used in encoder fitting or cluster selection.",
        },
        "input_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "patch_count": len(records),
        "split_counts": dict(split_counts),
        "feature_names": manifest.get("feature_stack", []),
        "encoder": {
            "type": "self_supervised_mlp_autoencoder",
            "embedding_level": "patch",
            "input_descriptor": "per-channel mean, standard deviation, p05, p25, p50, and p75 from Sentinel/topography patch tensors",
            "embedding_dimension": embedding_dim,
            "training_target": "reconstruct normalized Sentinel/topography descriptors",
            "iterations_completed": int(autoencoder.n_iter_),
            "final_reconstruction_loss": round(float(autoencoder.loss_), 6),
        },
        "deep_clustering": {
            "algorithm": "MiniBatchKMeans on autoencoder bottleneck embeddings",
            "cluster_count": cluster_count,
            "silhouette_score": score,
            "coherent_cluster_count": len(coherent_profiles),
            "coherent_cluster_fraction": metrics["coherent_cluster_fraction"],
        },
        "anomaly_detection": {
            "score_definition": "Normalized blend of autoencoder reconstruction error and embedding distance to assigned cluster center.",
            "high_anomaly_threshold": 0.9,
            "high_anomaly_fraction": high_anomaly_fraction,
        },
        "cluster_interpretation": {
            "sources": INTERPRETATION_SOURCES,
            "profiles": profiles,
        },
        "comparison_metrics": metrics,
        "comparison_against_unet": summarize_unet_comparison(metrics),
        "interpretation": (
            "This track can help discover anomalous or ambiguous patches without labels. It should be used "
            "as an independent reliability layer beside U-Net until expert labels are available."
        ),
    }


def write_outputs(report: dict[str, Any], embeddings: np.ndarray | None = None, records: list[dict[str, Any]] | None = None) -> None:
    for path in [OUTPUT_REPORT, WEB_SUMMARY, DOCS_SUMMARY]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if embeddings is not None and records is not None:
        EMBEDDING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            EMBEDDING_OUTPUT,
            embeddings=embeddings.astype("float32"),
            patch_ids=np.asarray([record["patch_id"] for record in records]),
            training_tags=np.asarray([record["training_tag"] for record in records]),
            years=np.asarray([record["year"] for record in records], dtype="int16"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-supervised Sentinel patch embedding track.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()

    report = build_report(
        manifest_path=args.manifest,
        cluster_count=args.clusters,
        embedding_dim=args.embedding_dim,
        max_records=args.max_records,
        max_iter=args.max_iter,
    )
    write_outputs(report, LAST_EMBEDDINGS, LAST_RECORDS)
    print(f"Self-supervised embedding report: {OUTPUT_REPORT}")
    print(json.dumps(report["comparison_metrics"], indent=2))


if __name__ == "__main__":
    main()
