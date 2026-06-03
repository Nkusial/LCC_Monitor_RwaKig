"""Build a label-free Sentinel patch embedding and cluster baseline.

This phase is the first concrete bridge from the weak-supervised U-Net track
toward self-supervised/deep clustering. It deliberately uses Sentinel-1/2 and
terrain patch features only for fitting. Weak labels are used only after
clustering to interpret clusters and estimate review usefulness.
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
warnings.filterwarnings(
    "ignore",
    message=".*Found Intel OpenMP.*LLVM OpenMP.*",
    category=RuntimeWarning,
)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")
warnings.filterwarnings(
    "ignore",
    message=".*MiniBatchKMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "training_patches_v3" / "training_patch_manifest_v3_balanced.json"
OUTPUT_REPORT = ROOT / "data" / "outputs" / "label_free_embedding_baseline_report.json"
WEB_SUMMARY = ROOT / "web" / "public" / "demo" / "label_free_embedding_baseline_summary.json"
DOCS_SUMMARY = ROOT / "docs" / "app" / "demo" / "label_free_embedding_baseline_summary.json"

CLASS_NAMES = {
    "1": "vegetation",
    "2": "built_up",
    "3": "water_wetness",
    "4": "bare_or_sparse_ground",
    "5": "mixed_or_uncertain",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def feature_summary(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Create compact label-free patch descriptors from feature tensors."""
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
            stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        stats.extend(
            [
                float(np.nanmean(finite)),
                float(np.nanstd(finite)),
                float(np.nanpercentile(finite, 10)),
                float(np.nanpercentile(finite, 50)),
                float(np.nanpercentile(finite, 90)),
            ]
        )
    return np.asarray(stats, dtype="float32")


def dominant_weak_label(record: dict[str, Any]) -> dict[str, Any]:
    """Interpret a patch after clustering using existing weak-label histograms."""
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
    """Load patch feature summaries without using labels for embedding creation."""
    records = []
    vectors = []
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
        interpreted = dominant_weak_label(record)
        records.append(
            {
                "patch_id": record["patch_id"],
                "path": record["path"],
                "training_tag": record["training_tag"],
                "year": int(record["year"]),
                "split": record["split"],
                "dominant_weak_label": interpreted,
                "class_pixel_counts": record.get("class_pixel_counts", {}),
            }
        )
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise RuntimeError("No patch records were available for label-free embedding.")
    return records, np.vstack(vectors).astype("float32")


def cluster_profile(records: list[dict[str, Any]], labels: np.ndarray) -> list[dict[str, Any]]:
    """Summarize clusters using weak labels only for post-hoc interpretation."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record, label in zip(records, labels, strict=True):
        grouped[int(label)].append(record)

    profiles = []
    for label, cluster_records in sorted(grouped.items()):
        label_counts = Counter(record["dominant_weak_label"]["class_name"] for record in cluster_records)
        year_counts = Counter(str(record["year"]) for record in cluster_records)
        top_name, top_count = label_counts.most_common(1)[0]
        profiles.append(
            {
                "cluster_id": label + 1,
                "patch_count": len(cluster_records),
                "dominant_interpretation": top_name,
                "dominant_interpretation_fraction": round(top_count / len(cluster_records), 6),
                "weak_label_mix": dict(label_counts),
                "year_mix": dict(year_counts),
                "example_patches": [record["patch_id"] for record in cluster_records[:5]],
            }
        )
    return profiles


def build_report(
    manifest_path: Path = DEFAULT_MANIFEST,
    cluster_count: int = 8,
    pca_components: int = 12,
    max_records: int | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    records, feature_vectors = load_patch_records(manifest, max_records=max_records)
    split_counts = Counter(record["split"] for record in records)
    train_index = np.array([record["split"] == "train" for record in records])
    if np.count_nonzero(train_index) < cluster_count:
        raise RuntimeError("Not enough train patches for the requested cluster count.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(feature_vectors[train_index])
    all_scaled = scaler.transform(feature_vectors)

    max_components = min(pca_components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    if max_components < 2:
        embedding_train = train_scaled
        embedding_all = all_scaled
        pca_variance = None
    else:
        pca = PCA(n_components=max_components, random_state=random_seed)
        embedding_train = pca.fit_transform(train_scaled)
        embedding_all = pca.transform(all_scaled)
        pca_variance = round(float(np.sum(pca.explained_variance_ratio_)), 6)

    # A larger batch avoids a known Windows/MKL MiniBatchKMeans warning while
    # keeping the prototype lightweight for the local-first workflow.
    model = MiniBatchKMeans(n_clusters=cluster_count, random_state=random_seed, batch_size=8192, n_init="auto")
    model.fit(embedding_train)
    labels = model.predict(embedding_all)

    score = None
    if len(np.unique(labels)) > 1 and len(records) > cluster_count:
        sample_size = min(500, len(records))
        rng = np.random.default_rng(random_seed)
        sample = rng.choice(len(records), size=sample_size, replace=False)
        score = round(float(silhouette_score(embedding_all[sample], labels[sample])), 6)

    profiles = cluster_profile(records, labels)
    coherent_profiles = [
        profile for profile in profiles if profile["dominant_interpretation_fraction"] >= 0.65
    ]

    return {
        "phase": "phase_59_label_free_embedding_baseline",
        "purpose": "Create a label-free Sentinel-1/2 patch embedding baseline for comparison with weak-supervised U-Net and classical unsupervised clustering.",
        "accuracy_claim": "No field accuracy claim; weak labels are used only after clustering for interpretation.",
        "no_cheating_protocol": {
            "excluded_years_from_fit": [2026],
            "fit_inputs": "Sentinel and terrain patch feature statistics only; no weak labels are used to fit scaler, PCA, or clusters.",
            "weak_labels_role": "post-hoc cluster interpretation and review prioritization only",
        },
        "input_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "patch_count": len(records),
        "split_counts": dict(split_counts),
        "feature_names": manifest.get("feature_stack", []),
        "embedding": {
            "descriptor": "per-channel mean, standard deviation, p10, p50, and p90",
            "pca_components": max_components if max_components >= 2 else 0,
            "pca_explained_variance": pca_variance,
        },
        "clustering": {
            "algorithm": "MiniBatchKMeans",
            "cluster_count": cluster_count,
            "silhouette_score": score,
            "coherent_cluster_count": len(coherent_profiles),
            "coherent_cluster_fraction": round(len(coherent_profiles) / len(profiles), 6) if profiles else 0.0,
        },
        "cluster_profiles": profiles,
        "interpretation": (
            "This is a lightweight label-free embedding baseline. It is useful for comparing review burden "
            "and cluster separability, but it is not yet a trained self-supervised deep encoder."
        ),
        "next_step": "Replace the statistical descriptor with a convolutional self-supervised encoder while keeping the same no-2026-leakage and weak-label-post-hoc rules.",
    }


def write_outputs(report: dict[str, Any]) -> None:
    for path in [OUTPUT_REPORT, WEB_SUMMARY, DOCS_SUMMARY]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a label-free Sentinel patch embedding baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    report = build_report(
        manifest_path=args.manifest,
        cluster_count=args.clusters,
        pca_components=args.pca_components,
        max_records=args.max_records,
    )
    write_outputs(report)
    print(f"Label-free embedding baseline report: {OUTPUT_REPORT}")
    print(json.dumps(report["clustering"], indent=2))


if __name__ == "__main__":
    main()
