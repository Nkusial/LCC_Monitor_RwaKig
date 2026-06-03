import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "44_label_free_embedding_baseline.py"
REPORT_PATH = ROOT / "data" / "outputs" / "label_free_embedding_baseline_report.json"


def load_module():
    spec = importlib.util.spec_from_file_location("label_free_embedding_baseline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_label_free_embedding_report_uses_no_accuracy_claim() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == "phase_59_label_free_embedding_baseline"
    assert report["accuracy_claim"].startswith("No field accuracy claim")
    assert report["no_cheating_protocol"]["excluded_years_from_fit"] == [2026]
    assert "post-hoc" in report["no_cheating_protocol"]["weak_labels_role"]


def test_label_free_embedding_report_has_clusters_and_interpretation() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["patch_count"] > 0
    assert report["clustering"]["cluster_count"] >= 4
    assert report["clustering"]["silhouette_score"] is not None
    assert len(report["cluster_profiles"]) == report["clustering"]["cluster_count"]
    assert all("dominant_interpretation" in profile for profile in report["cluster_profiles"])


def test_label_free_embedding_pipeline_keeps_private_patch_tensors_out_of_ci() -> None:
    module = load_module()

    assert module.DEFAULT_MANIFEST.parts[-2:] == (
        "training_patches_v3",
        "training_patch_manifest_v3_balanced.json",
    )
    assert REPORT_PATH.exists()


def test_label_free_embedding_docs_define_limitations() -> None:
    doc = (ROOT / "docs" / "LABEL_FREE_EMBEDDING_BASELINE.md").read_text(encoding="utf-8")

    assert "not a deep neural network yet" in doc
    assert "Weak labels are not used to fit" in doc
    assert "No field accuracy is claimed" in doc
