import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "43_compare_model_tracks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("model_track_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_model_track_comparison_report_has_clear_boundaries() -> None:
    module = load_module()
    report = module.build_report()

    assert report["phase"] == "phase_58_model_track_comparison"
    assert report["accuracy_claim"].startswith("No field accuracy claim")
    assert "supervised" in report["tracks"]
    assert "unsupervised" in report["tracks"]
    assert report["tracks"]["self_supervised"]["status"] in {
        "planned_not_trained",
        "implemented_patch_level",
    }
    assert any(
        "Do not compare U-Net confidence directly with silhouette" in warning
        for warning in report["recommendation"]["do_not_do"]
    )


def test_model_track_comparison_uses_existing_reliability_metrics() -> None:
    module = load_module()
    report = module.build_report()

    supervised = report["tracks"]["supervised"]
    unsupervised = report["tracks"]["unsupervised"]

    assert supervised["train_validation"]["mean_weak_source_compatibility"] is not None
    assert "mean_consensus_fraction" in supervised["temporal_consensus"]
    assert unsupervised["best_experiment"]["silhouette_score"] is not None
    assert "review_fraction" in report["comparison_metrics"]
    assert report["comparison_metrics"]["review_fraction"]["spatial_self_supervised_encoder"] is not None


def test_model_track_docs_explain_unet_and_unsupervised_roles() -> None:
    doc = read("docs/MODEL_TRACK_COMPARISON.md")

    assert "What U-Net Adds" in doc
    assert "What Unsupervised Clustering Adds" in doc
    assert "Self-Supervised Embedding Track" in doc
    assert "Silhouette is an internal coherence score" in doc


def test_generated_model_track_summary_when_available_is_not_accuracy_claim() -> None:
    summary_path = ROOT / "web" / "public" / "demo" / "model_track_comparison_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["accuracy_claim"].startswith("No field accuracy claim")
