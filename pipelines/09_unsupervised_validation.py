"""Unsupervised validation for monitored land-cover change outputs.

This phase is for projects without a field validation dataset. It does not
claim classification accuracy. Instead, it clusters AOI Sentinel-1/2 features
and checks whether monitored change groups align with coherent unsupervised
feature clusters.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config

try:
    import umap
except ModuleNotFoundError:
    umap = None


FEATURES = {
    "ndvi": "s2_ndvi",
    "ndwi": "s2_ndwi",
    "ndbi": "s2_ndbi",
    "vv": "s1_vv",
    "vh": "s1_vh",
    "vv_vh_ratio": "s1_vv_vh_ratio",
}

FEATURE_SETS = {
    "sentinel2": ["ndvi", "ndwi", "ndbi"],
    "fusion": ["ndvi", "ndwi", "ndbi", "vv", "vh", "vv_vh_ratio"],
}

DEFAULT_CLUSTER_COUNTS = [4, 5, 6, 8, 10]
DEFAULT_REDUCTIONS = ["none", "pca"] + (["umap"] if umap else [])
DEFAULT_MASKS = ["all", "high_confidence_change"]


def infer_landcover_state(ndvi: float, ndbi: float, ndwi: float) -> str:
    """Infer a human-readable cluster label from Sentinel-2 index means."""
    if not all(np.isfinite(value) for value in [ndvi, ndbi, ndwi]):
        return "unknown"
    if ndwi >= 0.2:
        return "water_moisture"
    if ndvi >= 0.45 and ndvi > ndbi:
        return "vegetation"
    if ndbi >= 0.12 and ndbi > ndvi:
        return "built_up"
    if ndvi <= 0.2 and ndbi <= 0.12:
        return "bare_sparse"
    return "mixed"


def monitoring_tags(processed_dir: Path) -> list[str]:
    """Find monitoring stack tags that have the required NDVI raster."""
    tags = {
        path.name.removesuffix("_s2_ndvi.tif")
        for path in processed_dir.glob("monitoring_*_s2_ndvi.tif")
    }
    return sorted(tags)


def raster_path(processed_dir: Path, tag: str, feature_suffix: str) -> Path:
    path = processed_dir / f"{tag}_{feature_suffix}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Required feature raster not found: {path}")
    return path


def read_feature_stack(
    processed_dir: Path,
    tag: str,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Load selected Sentinel features and verify they share one raster grid."""
    arrays = []
    profile: dict[str, Any] | None = None
    valid_mask: np.ndarray | None = None

    for feature_name in feature_names:
        suffix = FEATURES[feature_name]
        path = raster_path(processed_dir, tag, suffix)
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            if profile is None:
                profile = src.profile.copy()
                valid_mask = np.isfinite(data)
            else:
                if (
                    src.width != profile["width"]
                    or src.height != profile["height"]
                    or src.transform != profile["transform"]
                    or src.crs != profile["crs"]
                ):
                    raise ValueError(f"Raster grid mismatch for {path}")
                valid_mask &= np.isfinite(data)
            arrays.append(data)

    if profile is None or valid_mask is None:
        raise ValueError("No feature rasters found.")

    stack = np.stack(arrays, axis=-1)
    return stack, profile, valid_mask


def sample_valid_features(
    feature_matrix: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    if len(feature_matrix) <= sample_size:
        return feature_matrix
    sample_index = rng.choice(len(feature_matrix), size=sample_size, replace=False)
    return feature_matrix[sample_index]


def sample_indices(total: int, sample_size: int, random_seed: int) -> np.ndarray:
    """Return deterministic sample indices for repeatable validation runs."""
    if total <= sample_size:
        return np.arange(total)
    rng = np.random.default_rng(random_seed)
    return rng.choice(total, size=sample_size, replace=False)


def write_cluster_raster(
    path: Path,
    profile: dict[str, Any],
    valid_mask: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Write cluster labels back onto the AOI raster grid."""
    cluster_raster = np.zeros(valid_mask.shape, dtype="uint8")
    cluster_raster[valid_mask] = labels.astype("uint8") + 1

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(cluster_raster, 1)


def cluster_profiles(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    pixel_area_m2: float,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Summarize each cluster using mean feature values and area."""
    profiles = []
    for label in sorted(np.unique(labels)):
        pixels = feature_matrix[labels == label]
        means = {
            name: round(float(value), 4)
            for name, value in zip(feature_names, np.nanmean(pixels, axis=0), strict=True)
        }
        inferred = infer_landcover_state(
            means.get("ndvi", float("nan")),
            means.get("ndbi", float("nan")),
            means.get("ndwi", float("nan")),
        )
        profiles.append(
            {
                "cluster_id": int(label) + 1,
                "pixel_count": int(len(pixels)),
                "area_m2": round(float(len(pixels) * pixel_area_m2), 2),
                "area_km2": round(float(len(pixels) * pixel_area_m2 / 1_000_000), 3),
                "inferred_land_cover": inferred,
                "mean_features": means,
            }
        )
    return profiles


def confidence_mask_for_tag(
    outputs_dir: Path,
    tag: str,
    target_shape: tuple[int, int],
    threshold: float,
) -> np.ndarray | None:
    """Build a mask from confidence rasters that touch the selected stack."""
    masks = []
    patterns = [f"*__to__{tag}_confidence.tif", f"{tag}__to__*_confidence.tif"]
    for pattern in patterns:
        for path in outputs_dir.glob(pattern):
            with rasterio.open(path) as src:
                data = src.read(1).astype("float32")
                if data.shape != target_shape:
                    continue
                masks.append(np.isfinite(data) & (data >= threshold))
    if not masks:
        return None
    combined = np.zeros(target_shape, dtype=bool)
    for mask in masks:
        combined |= mask
    return combined


def prepare_feature_matrix(
    processed_dir: Path,
    outputs_dir: Path,
    tag: str,
    feature_set: str,
    mask_mode: str,
    confidence_threshold: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, list[str], str]:
    """Prepare the pixel feature matrix for one validation experiment."""
    feature_names = FEATURE_SETS[feature_set]
    stack, profile, valid_mask = read_feature_stack(processed_dir, tag, feature_names)
    effective_mask = valid_mask.copy()
    mask_note = "all valid AOI pixels"

    if mask_mode == "high_confidence_change":
        change_mask = confidence_mask_for_tag(
            outputs_dir,
            tag,
            valid_mask.shape,
            confidence_threshold,
        )
        if change_mask is None or not np.any(change_mask & valid_mask):
            raise ValueError(f"No high-confidence change mask available for {tag}")
        effective_mask &= change_mask
        mask_note = f"pixels with change confidence >= {confidence_threshold}"

    feature_matrix = stack[effective_mask]
    if len(feature_matrix) == 0:
        raise ValueError(f"No valid pixels for {tag} / {feature_set} / {mask_mode}")
    return feature_matrix, profile, effective_mask, feature_names, mask_note


def fit_cluster_experiment(
    feature_matrix: np.ndarray,
    clusters: int,
    reduction: str,
    sample_size: int,
    evaluation_size: int,
    random_seed: int,
) -> dict[str, Any]:
    """Train and score one clustering configuration."""
    train_index = sample_indices(len(feature_matrix), sample_size, random_seed)
    eval_index = sample_indices(len(feature_matrix), evaluation_size, random_seed + 1)
    training_sample = feature_matrix[train_index]
    evaluation_sample = feature_matrix[eval_index]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(training_sample)
    scaled_eval = scaler.transform(evaluation_sample)

    reducer = None
    transformed_train = scaled_train
    transformed_eval = scaled_eval
    explained_variance = None
    if reduction == "pca":
        components = min(3, scaled_train.shape[1])
        reducer = PCA(n_components=components, random_state=random_seed)
        transformed_train = reducer.fit_transform(scaled_train)
        transformed_eval = reducer.transform(scaled_eval)
        explained_variance = round(float(reducer.explained_variance_ratio_.sum()), 4)
    elif reduction == "umap":
        if umap is None:
            raise ValueError("UMAP requested but umap-learn is not installed.")
        # UMAP is optional so the default portfolio environment can stay lighter.
        reducer = umap.UMAP(n_components=3, random_state=random_seed)
        transformed_train = reducer.fit_transform(scaled_train)
        transformed_eval = reducer.transform(scaled_eval)

    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=random_seed,
        batch_size=8192,
        n_init="auto",
    )
    model.fit(transformed_train)
    eval_labels = model.predict(transformed_eval)
    silhouette = (
        float(silhouette_score(transformed_eval, eval_labels))
        if len(set(eval_labels)) > 1
        else None
    )

    return {
        "model": model,
        "scaler": scaler,
        "reducer": reducer,
        "silhouette_score": silhouette,
        "sample_size": int(len(training_sample)),
        "evaluation_size": int(len(evaluation_sample)),
        "pca_explained_variance": explained_variance,
    }


def predict_all_labels(
    feature_matrix: np.ndarray,
    scaler: StandardScaler,
    reducer: PCA | None,
    model: MiniBatchKMeans,
) -> np.ndarray:
    scaled_features = scaler.transform(feature_matrix)
    transformed = reducer.transform(scaled_features) if reducer else scaled_features
    return model.predict(transformed)


def feature_cluster_alignment(
    changes_path: Path,
    cluster_raster_path: Path,
) -> dict[str, Any]:
    """Compare published monitored groups with the best cluster raster."""
    if not changes_path.exists():
        return {"status": "skipped", "reason": f"Missing {changes_path}"}

    changes = gpd.read_file(changes_path)
    if changes.empty or "monitored_land_cover" not in changes:
        return {"status": "skipped", "reason": "No monitored change features found."}

    with rasterio.open(cluster_raster_path) as src:
        cluster_crs = src.crs
        transform = src.transform
        cluster_data = src.read(1)

    changes = changes.to_crs(cluster_crs)
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for _, feature in changes.iterrows():
        # Representative points avoid edge ambiguity when a polygon spans
        # several clusters; this is a lightweight audit, not field validation.
        point = feature.geometry.representative_point()
        row, col = rowcol(transform, point.x, point.y)
        if row < 0 or col < 0 or row >= cluster_data.shape[0] or col >= cluster_data.shape[1]:
            continue
        cluster_id = int(cluster_data[row, col])
        if cluster_id == 0:
            continue
        counts[str(feature["monitored_land_cover"])][cluster_id] += 1

    by_group = {}
    for group, cluster_counts in sorted(counts.items()):
        total = sum(cluster_counts.values())
        dominant_cluster, dominant_count = cluster_counts.most_common(1)[0]
        by_group[group] = {
            "sampled_feature_count": int(total),
            "dominant_cluster": int(dominant_cluster),
            "dominant_cluster_share": round(float(dominant_count / total), 3),
            "cluster_counts": {
                str(cluster_id): int(count)
                for cluster_id, count in sorted(cluster_counts.items())
            },
        }

    return {
        "status": "ok",
        "method": "representative point sampled against unsupervised cluster raster",
        "by_monitored_land_cover": by_group,
    }


def run_unsupervised_validation(
    config: dict[str, Any],
    tag: str | None,
    clusters: int,
    sample_size: int,
    evaluation_size: int,
    random_seed: int,
    compare: bool,
) -> tuple[Path, Path]:
    """Run the Phase 20/20B unsupervised validation workflow."""
    ensure_output_dirs(config)
    processed_dir = Path(config["paths"]["processed"])
    outputs_dir = Path(config["paths"]["outputs"])
    available_tags = monitoring_tags(processed_dir)
    if not available_tags:
        raise FileNotFoundError("No multi-date monitoring rasters found in data/processed.")

    selected_tags = [tag] if tag else [available_tags[0], available_tags[-1]]
    for selected_tag in selected_tags:
        if selected_tag not in available_tags:
            raise ValueError(f"Unknown monitoring tag {selected_tag!r}. Available: {available_tags}")

    experiment_specs = []
    if compare:
        # Phase 20B intentionally compares several cluster counts, feature
        # sets, reductions, masks, and baseline/after stacks before choosing a
        # best internal coherence score.
        for selected_tag in selected_tags:
            for feature_set in FEATURE_SETS:
                for reduction in DEFAULT_REDUCTIONS:
                    for mask_mode in DEFAULT_MASKS:
                        for cluster_count in DEFAULT_CLUSTER_COUNTS:
                            experiment_specs.append(
                                (selected_tag, feature_set, reduction, mask_mode, cluster_count)
                            )
    else:
        experiment_specs.append((selected_tags[-1], "fusion", "none", "all", clusters))

    experiments = []
    best_runtime: dict[str, Any] | None = None
    best_score = -1.0
    confidence_threshold = float(config["change_detection"]["confidence_threshold"])

    for index, (selected_tag, feature_set, reduction, mask_mode, cluster_count) in enumerate(
        experiment_specs
    ):
        try:
            feature_matrix, profile, effective_mask, feature_names, mask_note = prepare_feature_matrix(
                processed_dir,
                outputs_dir,
                selected_tag,
                feature_set,
                mask_mode,
                confidence_threshold,
            )
            result = fit_cluster_experiment(
                feature_matrix,
                cluster_count,
                reduction,
                sample_size,
                evaluation_size,
                random_seed + index,
            )
            score = result["silhouette_score"]
            experiment = {
                "status": "ok",
                "monitoring_tag": selected_tag,
                "feature_set": feature_set,
                "features": feature_names,
                "reduction": reduction,
                "mask_mode": mask_mode,
                "mask_note": mask_note,
                "cluster_count": cluster_count,
                "valid_pixel_count": int(len(feature_matrix)),
                "sample_size": result["sample_size"],
                "evaluation_size": result["evaluation_size"],
                "silhouette_score": round(score, 4) if score is not None else None,
                "pca_explained_variance": result["pca_explained_variance"],
            }
            experiments.append(experiment)
            if score is not None and score > best_score:
                best_score = score
                best_runtime = {
                    **result,
                    "experiment": experiment,
                    "feature_matrix": feature_matrix,
                    "profile": profile,
                    "effective_mask": effective_mask,
                    "feature_names": feature_names,
                }
        except ValueError as error:
            experiments.append(
                {
                    "status": "skipped",
                    "monitoring_tag": selected_tag,
                    "feature_set": feature_set,
                    "reduction": reduction,
                    "mask_mode": mask_mode,
                    "cluster_count": cluster_count,
                    "reason": str(error),
                }
            )

    if best_runtime is None:
        raise ValueError("No unsupervised validation experiment could be completed.")

    best_experiment = best_runtime["experiment"]
    selected_tag = best_experiment["monitoring_tag"]
    feature_matrix = best_runtime["feature_matrix"]
    profile = best_runtime["profile"]
    effective_mask = best_runtime["effective_mask"]
    feature_names = best_runtime["feature_names"]
    labels = predict_all_labels(
        feature_matrix,
        best_runtime["scaler"],
        best_runtime["reducer"],
        best_runtime["model"],
    )
    pixel_area_m2 = abs(float(profile["transform"].a * profile["transform"].e))
    cluster_raster_path = outputs_dir / f"{selected_tag}_unsupervised_clusters.tif"
    write_cluster_raster(cluster_raster_path, profile, effective_mask, labels)

    report = {
        "phase": "phase_20b_unsupervised_validation_comparison",
        "validation_type": "internal_unsupervised_cluster_audit",
        "ground_truth_available": False,
        "accuracy_claim": "No field accuracy is claimed. This report checks feature-space coherence only.",
        "comparison_scope": {
            "cluster_counts": DEFAULT_CLUSTER_COUNTS if compare else [clusters],
            "feature_sets": list(FEATURE_SETS.keys()) if compare else ["fusion"],
            "reductions": DEFAULT_REDUCTIONS if compare else ["none"],
            "masks": DEFAULT_MASKS if compare else ["all"],
            "tags": selected_tags,
            "umap_status": (
                "run" if umap else "not_run_no_umap_dependency; PCA is used as the dependency-light reduction baseline"
            ),
        },
        "monitoring_tag": selected_tag,
        "available_monitoring_tags": available_tags,
        "feature_stack": feature_names,
        "cluster_count": int(best_experiment["cluster_count"]),
        "valid_pixel_count": int(len(feature_matrix)),
        "sample_size": int(best_experiment["sample_size"]),
        "evaluation_size": int(best_experiment["evaluation_size"]),
        "silhouette_score": best_experiment["silhouette_score"],
        "best_experiment": best_experiment,
        "experiments": sorted(
            experiments,
            key=lambda item: (
                item.get("status") != "ok",
                -(item.get("silhouette_score") or -999),
            ),
        ),
        "cluster_profiles": cluster_profiles(feature_matrix, labels, pixel_area_m2, feature_names),
        "monitored_group_cluster_alignment": feature_cluster_alignment(
            outputs_dir / "monitoring_change_scored.geojson",
            cluster_raster_path,
        ),
        "interpretation": (
            "Higher silhouette is better, but this remains an unsupervised feature-space "
            "coherence score. It is not a substitute for field or reference-label accuracy."
        ),
        "outputs": {
            "cluster_raster": str(cluster_raster_path).replace("\\", "/"),
        },
    }
    report_path = outputs_dir / "unsupervised_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return cluster_raster_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unsupervised validation audit.")
    parser.add_argument("--tag", default=None, help="Monitoring tag to cluster. Defaults to latest.")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=40_000)
    parser.add_argument("--evaluation-size", type=int, default=5_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run one fusion/all-pixel configuration instead of the Phase 20B comparison grid.",
    )
    args = parser.parse_args()

    cluster_raster, report_path = run_unsupervised_validation(
        load_config(),
        tag=args.tag,
        clusters=args.clusters,
        sample_size=args.sample_size,
        evaluation_size=args.evaluation_size,
        random_seed=args.random_seed,
        compare=not args.single,
    )
    print(f"Cluster raster: {cluster_raster}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
