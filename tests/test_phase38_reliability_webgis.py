import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_phase38_frontend_exposes_reliability_summary_panel() -> None:
    app = read("web/src/App.tsx")
    api = read("web/src/api.ts")
    css = read("web/src/App.css")

    assert "Reliability summary" in app
    assert "Weak-source compatibility" in app
    assert "Likely reliable change" in app
    assert "Needs review" in app
    assert "loadUnetReliabilitySummary" in api
    assert "UnetReliabilitySummary" in api
    assert ".review-key" in css


def test_phase38_raster_manifest_includes_review_zone_layer_when_exported() -> None:
    manifest_path = ROOT / "web" / "public" / "demo" / "raster_layers.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer_ids = {layer["id"] for layer in manifest["layers"]}
    if "unet_review_zones" in layer_ids:
        review_layer = next(layer for layer in manifest["layers"] if layer["id"] == "unet_review_zones")
        assert review_layer["group"] == "Reliability review"
        assert "review aid" in review_layer["description"]
