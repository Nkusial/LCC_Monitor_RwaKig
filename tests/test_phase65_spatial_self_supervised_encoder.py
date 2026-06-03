import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "48_spatial_self_supervised_encoder.py"
COMPARISON_PATH = ROOT / "pipelines" / "43_compare_model_tracks.py"
REPORT_PATH = ROOT / "data" / "outputs" / "spatial_self_supervised_encoder_report.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def phase65_report():
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_spatial_encoder_report_preserves_no_cheating_boundary() -> None:
    report = phase65_report()

    assert report["phase"] == "phase_65_spatial_self_supervised_encoder"
    assert report["accuracy_claim"].startswith("No field accuracy claim")
    assert report["no_cheating_protocol"]["excluded_years_from_fit"] == [2026]
    assert "weak labels are not used" in report["no_cheating_protocol"]["fit_inputs"]
    assert report["no_cheating_protocol"]["weak_labels_role"].startswith("post-hoc")


def test_spatial_encoder_reports_encoder_and_review_metrics() -> None:
    report = phase65_report()

    assert report["encoder"]["embedding_level"] in {"spatial_patch", "spatial_patch_grid"}
    assert report["comparison_metrics"]["review_burden_fraction"] is not None
    assert report["comparison_metrics"]["weak_source_compatibility"] is not None
    assert report["comparison_against_unet"]["spatial_self_supervised_review_burden_fraction"] is not None


def test_temporal_transformer_status_is_explicit() -> None:
    report = phase65_report()

    assert report["temporal_encoder_status"]["status"] in {
        "ready_for_temporal_transformer",
        "insufficient_exact_repeated_windows_for_temporal_transformer",
    }
    assert "interpretation" in report["temporal_encoder_status"]


def test_model_track_comparison_includes_spatial_upgrade() -> None:
    comparison = load_module(COMPARISON_PATH, "model_track_comparison").build_report()
    self_supervised = comparison["tracks"]["self_supervised"]

    assert "spatial_encoder_upgrade" in self_supervised
    assert self_supervised["spatial_encoder_upgrade"]["review_comparison"]["review_burden_fraction"] is not None
    assert comparison["recommendation"]["why"]["spatial_encoder_type"] is not None


def test_spatial_encoder_pipeline_keeps_private_patch_tensors_out_of_ci() -> None:
    module = load_module(SCRIPT_PATH, "spatial_self_supervised_encoder_contract")

    assert module.DEFAULT_MANIFEST.parts[-2:] == (
        "training_patches_v3",
        "training_patch_manifest_v3_balanced.json",
    )
    assert REPORT_PATH.exists()
