from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_self_supervised_review_exports_exist():
    """Phase 63 should publish the label-free review layer to WebGIS assets."""
    assert (ROOT / "pipelines" / "46_export_self_supervised_review_layer.py").exists()
    assert (ROOT / "web" / "public" / "demo" / "self_supervised_review_patches.geojson").exists()
    assert (ROOT / "web" / "public" / "demo" / "self_supervised_review_layer_summary.json").exists()


def test_self_supervised_review_export_documents_georeference_and_caveat():
    source = (ROOT / "pipelines" / "46_export_self_supervised_review_layer.py").read_text()

    assert "EPSG:32735" in source
    assert "EPSG:4326" in source
    assert "No field accuracy claim" in source
    assert "weak-label classes are not used to fit" in source


def test_webgis_loads_self_supervised_review_assets():
    api_source = (ROOT / "web" / "src" / "api.ts").read_text()

    assert "SelfSupervisedReviewSummary" in api_source
    assert "self_supervised_review_layer_summary.json" in api_source
    assert "self_supervised_review_patches.geojson" in api_source


def test_webgis_renders_self_supervised_review_as_separate_overlay():
    app_source = (ROOT / "web" / "src" / "App.tsx").read_text()
    map_source = (ROOT / "web" / "src" / "map" / "MapView.tsx").read_text()

    assert "Self-supervised review" in app_source
    assert "showSelfSupervisedReview" in app_source
    assert "self-supervised-review" in map_source
    assert "review_priority" in map_source
    assert "label-free review evidence" in app_source
