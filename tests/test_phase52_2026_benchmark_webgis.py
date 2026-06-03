import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_webgis_exporter_publishes_frozen_2026_benchmark_layers() -> None:
    source = read("pipelines/08_export_web_raster_layers.py")

    assert "BENCHMARK_2026_REPORT" in source
    assert "test_2026_unet_dominant_class" in source
    assert "test_2026_unet_review_zones" in source
    assert "This layer supports error review, not accuracy claims." in source


def test_frontend_loads_benchmark_without_accuracy_overclaim() -> None:
    api_source = read("web/src/api.ts")
    app_source = read("web/src/App.tsx")

    assert "Benchmark2026Summary" in api_source
    assert "benchmark_2026_summary.json" in api_source
    assert "2026 frozen benchmark" in app_source
    assert "No-cheating protocol" in app_source
    assert "accuracy_claim" in app_source


def test_generated_2026_benchmark_summary_is_proxy_only_when_available() -> None:
    summary_path = ROOT / "web" / "public" / "demo" / "benchmark_2026_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "relaxed_proxy_benchmark"
    assert summary["accuracy_claim"] == "Weak-source compatibility only; not field accuracy."
    assert "Do not retrain from this result." in summary["non_cheating_protocol"]["forbidden"]
