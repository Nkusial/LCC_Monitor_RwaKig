import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLING_SCRIPT = ROOT / "pipelines" / "47_propose_self_supervised_review_samples.py"
COMPARISON_SCRIPT = ROOT / "pipelines" / "43_compare_model_tracks.py"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase64_review_sampling_excludes_frozen_2026() -> None:
    module = load_module(SAMPLING_SCRIPT, "self_supervised_review_sampling")
    manifest, sample_geojson = module.build_manifest(max_samples=12, max_per_stratum=4)

    assert manifest["phase"] == "phase_64_self_supervised_review_sampling"
    assert manifest["accuracy_claim"].startswith("No field accuracy claim")
    assert manifest["sample_count"] == len(sample_geojson["features"])
    assert manifest["no_cheating_protocol"]["frozen_2026_policy"].startswith("Do not use 2026")
    assert all(feature["properties"]["year"] < 2026 for feature in sample_geojson["features"])


def test_phase64_review_samples_have_actionable_properties() -> None:
    module = load_module(SAMPLING_SCRIPT, "self_supervised_review_sampling_properties")
    _manifest, sample_geojson = module.build_manifest(max_samples=8, max_per_stratum=3)

    assert sample_geojson["features"], "Expected review samples from the existing self-supervised layer."
    properties = sample_geojson["features"][0]["properties"]
    for key in [
        "review_score",
        "sampling_role",
        "recommended_action",
        "centroid_lon",
        "centroid_lat",
        "sample_rank",
    ]:
        assert key in properties


def test_spatial_encoder_scaffold_documents_compute_upgrade() -> None:
    source = (ROOT / "ml" / "self_supervised_encoders.py").read_text(encoding="utf-8")

    assert "ConvPatchAutoencoder" in source
    assert "TemporalPatchTransformer" in source
    assert "Weak sources should still be used only for post-hoc" in source
    assert "PyTorch is required" in source


def test_model_track_comparison_includes_phase64_quality_gates() -> None:
    module = load_module(COMPARISON_SCRIPT, "model_track_comparison_phase64")
    report = module.build_report()

    assert "quality_gates" in report
    assert "temporal_consensus" in report["quality_gates"]
    assert "geospatial_alignment" in report["quality_gates"]
    assert "frozen_2026_boundary" in report["quality_gates"]
    assert "comparison_metrics" in report
