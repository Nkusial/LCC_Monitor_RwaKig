"""Export self-supervised embedding review patches for the WebGIS.

The self-supervised encoder produces patch-level embeddings, clusters, and
anomaly scores. This exporter converts those patch diagnostics into lightweight
GeoJSON polygons so users can see where the label-free track supports review,
without confusing those diagnostics with land-cover classes or field accuracy.
weak-label classes are not used to fit the encoder or clusters; weak sources
only interpret the learned embedding space after fitting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
SELF_SUPERVISED_SCRIPT = ROOT / "pipelines" / "45_self_supervised_embedding_track.py"
OUTPUTS = ROOT / "data" / "outputs"
WEB_DEMO = ROOT / "web" / "public" / "demo"
DOCS_DEMO = ROOT / "docs" / "app" / "demo"
OUTPUT_GEOJSON = OUTPUTS / "self_supervised_review_patches.geojson"
WEB_GEOJSON = WEB_DEMO / "self_supervised_review_patches.geojson"
DOCS_GEOJSON = DOCS_DEMO / "self_supervised_review_patches.geojson"
OUTPUT_SUMMARY = OUTPUTS / "self_supervised_review_layer_summary.json"
WEB_SUMMARY = WEB_DEMO / "self_supervised_review_layer_summary.json"
DOCS_SUMMARY = DOCS_DEMO / "self_supervised_review_layer_summary.json"


def load_embedding_module():
    spec = importlib.util.spec_from_file_location("self_supervised_embedding_track", SELF_SUPERVISED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Could not load self-supervised embedding module.")
    spec.loader.exec_module(module)
    return module


def patch_polygon(transform: list[float], width: int, height: int) -> list[list[float]]:
    """Create a WGS84 patch polygon from the master-grid affine transform."""
    a, b, c, d, e, f = transform
    corners = [
        (c, f),
        (c + a * width + b * 0, f + d * width + e * 0),
        (c + a * width + b * height, f + d * width + e * height),
        (c + a * 0 + b * height, f + d * 0 + e * height),
        (c, f),
    ]
    transformer = Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)
    return [[round(lon, 9), round(lat, 9)] for lon, lat in transformer.itransform(corners)]


def review_priority(weak_fraction: float, anomaly_score: float) -> tuple[str, str]:
    """Translate numeric diagnostics into cartographic review categories."""
    if anomaly_score >= 0.9:
        return "high_anomaly", "high embedding anomaly"
    if weak_fraction < 0.65:
        return "weak_support_review", "mixed or weak-source support"
    return "embedding_supported", "embedding cluster has compatible weak-source interpretation"


def build_layer(
    cluster_count: int = 8,
    embedding_dim: int = 16,
    max_iter: int = 120,
    max_records: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = load_embedding_module()
    report = module.build_report(
        cluster_count=cluster_count,
        embedding_dim=embedding_dim,
        max_iter=max_iter,
        max_records=max_records,
    )
    records = module.LAST_RECORDS
    labels = module.LAST_CLUSTER_LABELS
    anomaly_scores = module.LAST_ANOMALY_SCORES
    if records is None or labels is None or anomaly_scores is None:
        raise RuntimeError("Self-supervised diagnostics were not available after report build.")

    features = []
    priority_counts: dict[str, int] = {}
    priority_area_m2: dict[str, float] = {}
    for record, label, anomaly in zip(records, labels, anomaly_scores, strict=True):
        weak = record["dominant_weak_label"]
        priority, reason = review_priority(float(weak["fraction"]), float(anomaly))
        area_m2 = float(record.get("height", 128) * record.get("width", 128) * 100)
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        priority_area_m2[priority] = priority_area_m2.get(priority, 0.0) + area_m2
        features.append(
            {
                "type": "Feature",
                "id": record["patch_id"],
                "properties": {
                    "patch_id": record["patch_id"],
                    "training_tag": record["training_tag"],
                    "year": record["year"],
                    "split": record["split"],
                    "cluster_id": int(label) + 1,
                    "dominant_interpretation": weak["class_name"],
                    "weak_source_fraction": round(float(weak["fraction"]), 6),
                    "anomaly_score": round(float(anomaly), 6),
                    "review_priority": priority,
                    "review_reason": reason,
                    "area_m2": area_m2,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        patch_polygon(
                            record["transform"],
                            int(record.get("width", 128)),
                            int(record.get("height", 128)),
                        )
                    ],
                },
            }
        )

    feature_collection = {
        "type": "FeatureCollection",
        "name": "self_supervised_review_patches",
        "features": features,
    }
    summary = {
        "phase": "phase_63_self_supervised_review_webgis_layer",
        "purpose": "Expose label-free self-supervised embedding support, anomaly, and review-priority patches in the WebGIS.",
        "accuracy_claim": "No field accuracy claim; this layer shows review priority from self-supervised embeddings and weak-source compatibility.",
        "source_report": "data/outputs/self_supervised_embedding_track_report.json",
        "feature_count": len(features),
        "priority_counts": priority_counts,
        "priority_area_m2": {key: round(value, 3) for key, value in priority_area_m2.items()},
        "legend": {
            "embedding_supported": "Embedding-supported patch",
            "weak_support_review": "Needs review: weak-source support is mixed",
            "high_anomaly": "High review priority: embedding anomaly",
        },
        "comparison_metrics": report["comparison_metrics"],
        "no_cheating_protocol": report["no_cheating_protocol"],
    }
    return feature_collection, summary


def write_outputs(feature_collection: dict[str, Any], summary: dict[str, Any]) -> None:
    for path in [OUTPUT_GEOJSON, WEB_GEOJSON, DOCS_GEOJSON]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(feature_collection), encoding="utf-8")
    for path in [OUTPUT_SUMMARY, WEB_SUMMARY, DOCS_SUMMARY]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export self-supervised embedding review patches for WebGIS.")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    feature_collection, summary = build_layer(
        cluster_count=args.clusters,
        embedding_dim=args.embedding_dim,
        max_iter=args.max_iter,
        max_records=args.max_records,
    )
    write_outputs(feature_collection, summary)
    print(f"Self-supervised review layer: {WEB_GEOJSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
