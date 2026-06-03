import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase21_validation_summary_is_published_to_hosted_demo():
    summary_path = ROOT / "web" / "public" / "demo" / "validation_summary.json"
    docs_summary_path = ROOT / "docs" / "app" / "demo" / "validation_summary.json"

    assert summary_path.exists()
    assert docs_summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["phase"] == "Phase 20B"
    assert summary["ground_truth_available"] is False
    assert summary["silhouette_score"] == 0.4459
    assert summary["score_label"] == "Best unsupervised silhouette"
    assert "not a U-Net confidence score" in summary["score_scope_note"]
    assert summary["reference_label_workflow"]["status"] == "planned"
    assert len(summary["reference_label_workflow"]["planned_steps"]) >= 4


def test_phase21_cluster_overlay_is_available_in_layer_manifest():
    manifest = json.loads(
        (ROOT / "web" / "public" / "demo" / "raster_layers.json").read_text(
            encoding="utf-8"
        )
    )
    layer_ids = {layer["id"] for layer in manifest["layers"]}

    assert "unsupervised_clusters_best" in layer_ids
    assert (
        ROOT
        / "docs"
        / "app"
        / "demo"
        / "rasters"
        / "unsupervised_clusters_best.png"
    ).exists()


def test_phase21_frontend_keeps_silhouette_out_of_user_interface():
    app_source = (ROOT / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    api_source = (ROOT / "web" / "src" / "api.ts").read_text(encoding="utf-8")

    assert "Reliability summary" in app_source
    assert "Best unsupervised silhouette" not in app_source
    assert "Future reference-label workflow" not in app_source
    assert "loadValidationSummary" in api_source
