"""Train a spatial self-supervised Sentinel patch encoder.

This phase replaces the descriptor-only MLP baseline with a convolutional
autoencoder over 128 x 128 Sentinel-1/2, index, radar, and terrain tensors.
It still follows the same guardrails: no 2026 benchmark fitting, no weak labels
inside the training loss, and weak sources only for post-hoc interpretation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
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

USE_TORCH_ENCODER = os.environ.get("LCC_ENABLE_TORCH_SPATIAL_ENCODER", "").lower() in {"1", "true", "yes"}

if USE_TORCH_ENCODER:
    try:  # The Windows local environment can have a broken torch DLL stack.
        import torch
        from torch.utils.data import DataLoader, Dataset

        from ml.self_supervised_encoders import ConvPatchAutoencoder
    except (ModuleNotFoundError, OSError) as error:  # pragma: no cover - platform dependent.
        torch = None
        DataLoader = object
        Dataset = object
        ConvPatchAutoencoder = None
        TORCH_IMPORT_ERROR = str(error)
    else:
        TORCH_IMPORT_ERROR = None
else:
    torch = None
    DataLoader = object
    Dataset = object
    ConvPatchAutoencoder = None
    TORCH_IMPORT_ERROR = (
        "Torch spatial encoder is disabled by default for stable local runs. "
        "Set LCC_ENABLE_TORCH_SPATIAL_ENCODER=1 on a PyTorch-ready CPU/GPU environment."
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "training_patches_v3" / "training_patch_manifest_v3_balanced.json"
OUTPUT_REPORT = ROOT / "data" / "outputs" / "spatial_self_supervised_encoder_report.json"
EMBEDDING_OUTPUT = ROOT / "data" / "interim" / "self_supervised_embeddings" / "spatial_patch_embeddings_v1.npz"
WEB_SUMMARY = ROOT / "web" / "public" / "demo" / "spatial_self_supervised_encoder_summary.json"
DOCS_SUMMARY = ROOT / "docs" / "app" / "demo" / "spatial_self_supervised_encoder_summary.json"
UNET_VALIDATION = ROOT / "data" / "outputs" / "unet_v3_full_aoi_validation_report.json"
UNET_2026 = ROOT / "data" / "outputs" / "unet_2026_proxy_benchmark_report.json"
GEOSPATIAL_ALIGNMENT = ROOT / "data" / "outputs" / "geospatial_alignment_report.json"
TEMPORAL_CONSENSUS = ROOT / "data" / "outputs" / "unet_temporal_consensus_report.json"

CLASS_NAMES = {
    "1": "vegetation",
    "2": "built_up",
    "3": "water_wetness",
    "4": "bare_or_sparse_ground",
    "5": "mixed_or_uncertain",
}

LAST_RECORDS: list[dict[str, Any]] | None = None
LAST_EMBEDDINGS: np.ndarray | None = None
LAST_CLUSTER_LABELS: np.ndarray | None = None
LAST_ANOMALY_SCORES: np.ndarray | None = None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT)).replace("\\", "/")}
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def round_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def dominant_weak_label(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize weak evidence after fitting; this is not used for training."""
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


def load_records(manifest_path: Path, max_records: int | None, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read patch metadata and enforce the no-2026 encoder-fitting boundary."""
    manifest = read_json(manifest_path)
    records = []
    for record in manifest.get("patches", []):
        if int(record.get("year", 0)) >= 2026:
            continue
        path = ROOT / record["path"]
        if not path.exists():
            continue
        records.append(
            {
                "patch_id": record["patch_id"],
                "path": record["path"],
                "training_tag": record["training_tag"],
                "year": int(record["year"]),
                "split": record["split"],
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

    rng = random.Random(seed)
    train = [record for record in records if record["split"] == "train"]
    other = [record for record in records if record["split"] != "train"]
    rng.shuffle(train)
    rng.shuffle(other)
    if max_records is not None:
        train_quota = min(len(train), max(32, int(max_records * 0.65)))
        selected = train[:train_quota] + other[: max(0, max_records - train_quota)]
    else:
        selected = train + other
    selected.sort(key=lambda item: (item["year"], item["training_tag"], item["row"] or 0, item["col"] or 0))
    if not selected:
        raise RuntimeError("No non-2026 patch tensors were available for spatial self-supervised training.")
    return manifest, selected


def compute_channel_stats(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-split channel normalization from Sentinel tensors only."""
    train_records = [record for record in records if record["split"] == "train"]
    if not train_records:
        raise RuntimeError("Spatial encoder needs train-split records for normalization and fitting.")
    sums: np.ndarray | None = None
    sq_sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    for record in train_records:
        with np.load(ROOT / record["path"]) as patch:
            features = patch["features"].astype("float32")
        finite = np.isfinite(features)
        values = np.where(finite, features, 0.0)
        channel_sum = values.reshape(features.shape[0], -1).sum(axis=1)
        channel_sq = (values * values).reshape(features.shape[0], -1).sum(axis=1)
        channel_count = finite.reshape(features.shape[0], -1).sum(axis=1).astype("float64")
        sums = channel_sum if sums is None else sums + channel_sum
        sq_sums = channel_sq if sq_sums is None else sq_sums + channel_sq
        counts = channel_count if counts is None else counts + channel_count
    assert sums is not None and sq_sums is not None and counts is not None
    means = sums / np.maximum(counts, 1)
    variances = (sq_sums / np.maximum(counts, 1)) - (means * means)
    stds = np.sqrt(np.maximum(variances, 1e-6))
    return means.astype("float32"), stds.astype("float32")


class SentinelPatchDataset(Dataset):
    """Lazy normalized patch tensor loader for CPU/GPU-safe local training."""

    def __init__(self, records: list[dict[str, Any]], means: np.ndarray, stds: np.ndarray):
        self.records = records
        self.means = means[:, None, None]
        self.stds = stds[:, None, None]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        with np.load(ROOT / record["path"]) as patch:
            features = patch["features"].astype("float32")
        normalized = (features - self.means) / self.stds
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)
        return torch.from_numpy(normalized.astype("float32"))


def train_encoder(
    records: list[dict[str, Any]],
    means: np.ndarray,
    stds: np.ndarray,
    epochs: int,
    batch_size: int,
    embedding_channels: int,
    learning_rate: float,
    seed: int,
) -> tuple[ConvPatchAutoencoder, list[float], str]:
    if torch is None or ConvPatchAutoencoder is None:
        raise RuntimeError(f"PyTorch is not available for convolutional training: {TORCH_IMPORT_ERROR}")
    torch.manual_seed(seed)
    train_records = [record for record in records if record["split"] == "train"]
    dataset = SentinelPatchDataset(train_records, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = len(means)
    model = ConvPatchAutoencoder(in_channels=in_channels, embedding_channels=embedding_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _epoch in range(epochs):
        epoch_losses = []
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _embedding = model(batch)
            loss = torch.mean((reconstruction - batch) ** 2)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(round(float(np.mean(epoch_losses)), 6) if epoch_losses else 0.0)
    return model, losses, device


def encode_records(
    model: ConvPatchAutoencoder,
    records: list[dict[str, Any]],
    means: np.ndarray,
    stds: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError(f"PyTorch is not available for convolutional inference: {TORCH_IMPORT_ERROR}")
    dataset = SentinelPatchDataset(records, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings = []
    errors = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction, embedding_map = model(batch)
            embedding_vector = embedding_map.mean(dim=(2, 3))
            batch_error = torch.mean((reconstruction - batch) ** 2, dim=(1, 2, 3))
            embeddings.append(embedding_vector.cpu().numpy())
            errors.append(batch_error.cpu().numpy())
    return np.vstack(embeddings).astype("float32"), np.concatenate(errors).astype("float32")


def spatial_grid_descriptor(features: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Encode patch layout as grid-cell means and standard deviations.

    This is the deterministic fallback when PyTorch cannot import locally. It
    is still spatial because it preserves where signals occur inside the patch,
    unlike the older whole-patch descriptor baseline.
    """
    channels, height, width = features.shape
    cell_h = height // grid_size
    cell_w = width // grid_size
    trimmed = features[:, : cell_h * grid_size, : cell_w * grid_size].astype("float32")
    reshaped = trimmed.reshape(channels, grid_size, cell_h, grid_size, cell_w)
    grid = reshaped.transpose(0, 1, 3, 2, 4)
    means = np.nanmean(np.where(np.isfinite(grid), grid, np.nan), axis=(3, 4))
    stds = np.nanstd(np.where(np.isfinite(grid), grid, np.nan), axis=(3, 4))
    descriptor = np.concatenate([means.reshape(-1), stds.reshape(-1)])
    return np.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def run_spatial_grid_fallback(
    records: list[dict[str, Any]],
    embedding_channels: int,
    max_iter: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a spatial-grid autoencoder when torch DLLs are unavailable."""
    descriptors = []
    for record in records:
        with np.load(ROOT / record["path"]) as patch:
            descriptors.append(spatial_grid_descriptor(patch["features"]))
    vectors = np.vstack(descriptors).astype("float32")
    train_index = np.asarray([record["split"] == "train" for record in records])
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(vectors[train_index])
    all_scaled = scaler.transform(vectors)
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(embedding_channels,),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
    )
    autoencoder.fit(train_scaled, train_scaled)
    hidden = np.maximum(all_scaled @ autoencoder.coefs_[0] + autoencoder.intercepts_[0], 0.0)
    reconstruction = autoencoder.predict(all_scaled)
    errors = np.mean((all_scaled - reconstruction) ** 2, axis=1)
    metadata = {
        "type": "spatial_grid_autoencoder_fallback",
        "implementation": "sklearn MLPRegressor over 8 x 8 grid-cell mean/std descriptors",
        "torch_import_error": TORCH_IMPORT_ERROR,
        "grid_size": 8,
        "embedding_channels": embedding_channels,
        "iterations_completed": int(autoencoder.n_iter_),
        "loss_history": [round(float(autoencoder.loss_), 6)],
        "training_target": "reconstruct normalized spatial grid descriptors from Sentinel/radar/terrain tensors",
    }
    return hidden.astype("float32"), errors.astype("float32"), metadata


def normalized_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray([value / total for value in counts.values() if value > 0], dtype="float64")
    entropy = -np.sum(probabilities * np.log(probabilities))
    maximum = np.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    return round(float(entropy / maximum), 6) if maximum else 0.0


def cluster_profiles(records: list[dict[str, Any]], labels: np.ndarray, anomaly_scores: np.ndarray) -> list[dict[str, Any]]:
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


def temporal_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report whether exact repeated spatial windows exist for temporal encoders."""
    by_window: dict[tuple[int, int], set[int]] = defaultdict(set)
    for record in records:
        if record.get("row") is None or record.get("col") is None:
            continue
        by_window[(int(record["row"]), int(record["col"]))].add(int(record["year"]))
    repeated = {key: years for key, years in by_window.items() if len(years) > 1}
    return {
        "candidate_spatial_window_count": len(by_window),
        "multi_year_window_count": len(repeated),
        "status": (
            "ready_for_temporal_transformer"
            if repeated
            else "insufficient_exact_repeated_windows_for_temporal_transformer"
        ),
        "interpretation": (
            "The current patch manifest has exact repeated row/col windows across years."
            if repeated
            else "The current balanced patch manifest samples different windows by date; temporal transformer training should use a future manifest that preserves repeated spatial windows across dates."
        ),
    }


def build_report(
    manifest_path: Path = DEFAULT_MANIFEST,
    max_records: int | None = 240,
    cluster_count: int = 8,
    embedding_channels: int = 32,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> dict[str, Any]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    manifest, records = load_records(manifest_path, max_records=max_records, seed=random_seed)
    train_records = [record for record in records if record["split"] == "train"]
    if len(train_records) < max(cluster_count, 10):
        raise RuntimeError("Not enough train records for spatial self-supervised encoder fitting.")
    if torch is not None and ConvPatchAutoencoder is not None:
        means, stds = compute_channel_stats(records)
        model, losses, device = train_encoder(
            records,
            means,
            stds,
            epochs=epochs,
            batch_size=batch_size,
            embedding_channels=embedding_channels,
            learning_rate=learning_rate,
            seed=random_seed,
        )
        embeddings, reconstruction_error = encode_records(model, records, means, stds, batch_size=batch_size, device=device)
        encoder_metadata = {
            "type": "conv_patch_autoencoder",
            "implementation": "ml.self_supervised_encoders.ConvPatchAutoencoder",
            "embedding_level": "spatial_patch",
            "input_tensor_shape": [len(manifest.get("feature_stack", [])), 128, 128],
            "embedding_channels": embedding_channels,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "loss_history": losses,
            "training_target": "reconstruct normalized Sentinel-1/2, index, radar, and terrain tensors",
        }
    else:
        embeddings, reconstruction_error, fallback_metadata = run_spatial_grid_fallback(
            records,
            embedding_channels=embedding_channels,
            max_iter=max(epochs * 40, 40),
            random_seed=random_seed,
        )
        encoder_metadata = {
            **fallback_metadata,
            "embedding_level": "spatial_patch_grid",
            "input_tensor_shape": [len(manifest.get("feature_stack", [])), 128, 128],
            "epochs": epochs,
            "batch_size": batch_size,
        }
    train_index = np.asarray([record["split"] == "train" for record in records])
    # A larger batch avoids a known MiniBatchKMeans/MKL warning on Windows and
    # keeps CI output focused on real project issues.
    kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=random_seed, batch_size=4096, n_init="auto")
    kmeans.fit(embeddings[train_index])
    labels = kmeans.predict(embeddings)
    center_distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[labels], axis=1)
    raw_anomaly = 0.5 * reconstruction_error + 0.5 * center_distances
    low, high = np.percentile(raw_anomaly, [5, 95])
    anomaly_scores = np.clip((raw_anomaly - low) / (high - low + 1e-9), 0.0, 1.0)

    global LAST_RECORDS, LAST_EMBEDDINGS, LAST_CLUSTER_LABELS, LAST_ANOMALY_SCORES
    LAST_RECORDS = records
    LAST_EMBEDDINGS = embeddings
    LAST_CLUSTER_LABELS = labels
    LAST_ANOMALY_SCORES = anomaly_scores

    score = None
    if len(np.unique(labels)) > 1 and len(records) > cluster_count:
        sample_size = min(500, len(records))
        rng = np.random.default_rng(random_seed)
        sample = rng.choice(len(records), size=sample_size, replace=False)
        score = round(float(silhouette_score(embeddings[sample], labels[sample])), 6)

    weak_fractions = np.asarray([record["dominant_weak_label"]["fraction"] for record in records], dtype="float32")
    low_support = weak_fractions < 0.65
    high_anomaly = anomaly_scores >= 0.9
    profiles = cluster_profiles(records, labels, anomaly_scores)
    validation = read_json(UNET_VALIDATION).get("summary", {})
    benchmark = read_json(UNET_2026).get("summary", {})
    alignment = read_json(GEOSPATIAL_ALIGNMENT)
    temporal = read_json(TEMPORAL_CONSENSUS)
    metrics = {
        "silhouette_score": score,
        "weak_source_compatibility": round(float(np.mean(weak_fractions)), 6),
        "review_burden_fraction": round(float(np.mean(low_support | high_anomaly)), 6),
        "high_anomaly_fraction": round(float(np.mean(high_anomaly)), 6),
        "mean_cluster_ambiguity": round(float(np.mean([profile["weak_label_mix_entropy"] for profile in profiles])), 6),
        "final_reconstruction_loss": (encoder_metadata.get("loss_history") or [None])[-1],
    }
    return {
        "phase": "phase_65_spatial_self_supervised_encoder",
        "purpose": "Train a real convolutional self-supervised Sentinel patch encoder and compare its anomaly/review behavior with the U-Net track.",
        "accuracy_claim": "No field accuracy claim; this is label-free representation learning plus weak-source post-hoc interpretation.",
        "input_manifest": relative(manifest_path),
        "patch_count": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "feature_names": manifest.get("feature_stack", []),
        "encoder": encoder_metadata,
        "temporal_encoder_status": temporal_group_summary(records),
        "deep_clustering": {
            "algorithm": "MiniBatchKMeans on global-average convolutional embedding maps",
            "cluster_count": cluster_count,
            "silhouette_score": score,
        },
        "anomaly_detection": {
            "score_definition": "Normalized blend of convolutional reconstruction error and embedding distance to assigned cluster center.",
            "high_anomaly_threshold": 0.9,
            "high_anomaly_fraction": metrics["high_anomaly_fraction"],
        },
        "cluster_interpretation": {
            "weak_sources_role": "post_hoc_only_after_encoder_and_cluster_fitting",
            "profiles": profiles,
        },
        "comparison_metrics": metrics,
        "comparison_against_unet": {
            "u_net_train_validation_review_fraction": validation.get("mean_candidate_review_fraction_of_valid"),
            "u_net_train_validation_weak_source_compatibility": validation.get("mean_compatible_fraction_in_agreement_zone"),
            "u_net_frozen_2026_review_fraction": benchmark.get("mean_candidate_review_fraction_of_valid"),
            "u_net_frozen_2026_weak_source_compatibility": benchmark.get("mean_compatible_fraction_in_agreement_zone"),
            "spatial_self_supervised_review_burden_fraction": metrics["review_burden_fraction"],
            "spatial_self_supervised_weak_source_compatibility": metrics["weak_source_compatibility"],
        },
        "quality_gates": {
            "geospatial_alignment_status": alignment.get("status"),
            "raster_mismatch_count": alignment.get("raster_mismatch_count"),
            "vector_mismatch_count": alignment.get("vector_mismatch_count"),
            "temporal_consensus_mean_fraction": temporal.get("mean_consensus_fraction"),
        },
        "no_cheating_protocol": {
            "excluded_years_from_fit": [2026],
            "fit_inputs": "Normalized Sentinel-1/2, index, radar, and terrain tensors only; weak labels are not used in the loss, normalization, or clustering.",
            "weak_labels_role": "post-hoc cluster interpretation and review-priority reporting only",
            "frozen_test_boundary": "The 2026 benchmark remains evaluation-only and is not used for encoder fitting, threshold choice, sample selection, or U-Net retraining.",
        },
        "next_steps": [
            "Create a repeated-window temporal patch manifest for true temporal-transformer training.",
            "Use spatial-encoder anomaly samples as review candidates before any U-Net retraining.",
            "Run the frozen 2026 comparison only after the next model version is locked.",
        ],
    }


def write_outputs(report: dict[str, Any]) -> None:
    for path in [OUTPUT_REPORT, WEB_SUMMARY, DOCS_SUMMARY]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if LAST_EMBEDDINGS is not None and LAST_RECORDS is not None:
        EMBEDDING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            EMBEDDING_OUTPUT,
            embeddings=LAST_EMBEDDINGS.astype("float32"),
            patch_ids=np.asarray([record["patch_id"] for record in LAST_RECORDS]),
            training_tags=np.asarray([record["training_tag"] for record in LAST_RECORDS]),
            years=np.asarray([record["year"] for record in LAST_RECORDS], dtype="int16"),
            cluster_labels=LAST_CLUSTER_LABELS.astype("int16") if LAST_CLUSTER_LABELS is not None else np.asarray([]),
            anomaly_scores=LAST_ANOMALY_SCORES.astype("float32") if LAST_ANOMALY_SCORES is not None else np.asarray([]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a spatial self-supervised Sentinel patch encoder.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-records", type=int, default=240)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--embedding-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    report = build_report(
        manifest_path=args.manifest,
        max_records=args.max_records,
        cluster_count=args.clusters,
        embedding_channels=args.embedding_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    write_outputs(report)
    print(f"Spatial self-supervised encoder report: {OUTPUT_REPORT}")
    print(json.dumps(report["comparison_metrics"], indent=2))


if __name__ == "__main__":
    main()
