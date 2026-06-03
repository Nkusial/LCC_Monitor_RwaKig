import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = ROOT / "pipelines" / "40_improvement_track_report.py"
MANIFEST_SCRIPT = ROOT / "pipelines" / "41_package_public_demo_manifest.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_improvement_track_config_covers_requested_controls() -> None:
    config = yaml.safe_load(read("configs/improvement_track.yaml"))

    for section in [
        "coregistration_qa",
        "temporal_consensus_gate",
        "ml_comparison",
        "public_demo_packaging",
        "hosted_backend",
    ]:
        assert section in config

    assert config["change_publish_gate"]["minimum_weak_source_compatibility"] >= 0.7
    assert "self-supervised" in config["ml_comparison"]["unsupervised_track"]
    assert "data/raw" in config["public_demo_packaging"]["private_roots"]


def test_improvement_track_doc_explains_controls_without_accuracy_overclaim() -> None:
    doc = read("docs/IMPROVEMENT_TRACK.md")

    for phrase in [
        "Co-Registration QA",
        "Stronger Temporal Consensus",
        "U-Net Versus Self-Supervised Comparison",
        "Lightweight Public Demo Assets",
        "Hosted FastAPI/PostGIS Deployment",
    ]:
        assert phrase in doc

    assert "not field accuracy" in doc.lower()


def test_improvement_track_report_builds_publish_gate() -> None:
    module = load_module(REPORT_SCRIPT, "improvement_track_report")
    report = module.build_report()

    assert report["phase"] == "phase_56_improvement_track_controls"
    assert report["publish_gate"]["recommendation"] in {
        "publish_supported_candidates",
        "publish_with_review_required",
        "hold_for_processing",
    }
    assert "coregistration" in report["publish_gate"]
    assert "temporal_consensus" in report["publish_gate"]
    assert "ml_comparison_plan" in report
    assert report["hosted_backend"]["current_mode"] == "local_first_fastapi_postgis"


def test_public_demo_manifest_keeps_private_roots_out_of_public_package() -> None:
    module = load_module(MANIFEST_SCRIPT, "public_demo_manifest")
    manifest = module.build_manifest()

    assert manifest["phase"] == "phase_56_public_demo_packaging"
    assert "data/raw" in manifest["private_roots"]
    assert "data/interim" in manifest["private_roots"]
    assert all(not root["root"].startswith("data/raw") for root in manifest["roots"])
    assert manifest["status"] in {"passed", "warning"}


def test_generated_web_summaries_when_present_are_consistent() -> None:
    summary_path = ROOT / "web" / "public" / "demo" / "improvement_track_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["publish_gate"]["interpretation"].endswith("accuracy assessment.")
