import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "45_self_supervised_embedding_track.py"
REPORT_PATH = ROOT / "data" / "outputs" / "self_supervised_embedding_track_report.json"


def load_module():
    spec = importlib.util.spec_from_file_location("self_supervised_embedding_track", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_self_supervised_track_uses_no_label_fit_boundary() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == "phase_62_self_supervised_embedding_track"
    assert report["accuracy_claim"].startswith("No field accuracy claim")
    assert report["no_cheating_protocol"]["excluded_years_from_fit"] == [2026]
    assert "weak-label classes are not used" in report["no_cheating_protocol"]["fit_inputs"]
    assert report["encoder"]["type"] == "self_supervised_mlp_autoencoder"
    assert report["encoder"]["embedding_level"] == "patch"


def test_self_supervised_track_interprets_with_all_weak_sources() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    sources = " ".join(report["cluster_interpretation"]["sources"])

    assert "Dynamic World" in sources
    assert "ESA WorldCover" in sources
    assert "OSM" in sources
    assert "NDVI" in sources
    assert "Sentinel-1" in sources
    assert "high-confidence change polygons" in sources
    assert report["cluster_interpretation"]["profiles"]


def test_self_supervised_track_reports_review_metrics_against_unet() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["comparison_metrics"]
    comparison = report["comparison_against_unet"]

    assert metrics["review_burden_fraction"] is not None
    assert metrics["weak_source_compatibility"] is not None
    assert metrics["high_anomaly_fraction"] is not None
    assert "self_supervised_track" in comparison
    assert "u_net_track" in comparison


def test_self_supervised_pipeline_keeps_private_patch_tensors_out_of_ci() -> None:
    module = load_module()

    assert module.DEFAULT_MANIFEST.parts[-2:] == (
        "training_patches_v3",
        "training_patch_manifest_v3_balanced.json",
    )
    assert REPORT_PATH.exists()


def test_generated_self_supervised_summary_when_available_is_not_accuracy_claim() -> None:
    summary_path = ROOT / "web" / "public" / "demo" / "self_supervised_embedding_track_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["accuracy_claim"].startswith("No field accuracy claim")
