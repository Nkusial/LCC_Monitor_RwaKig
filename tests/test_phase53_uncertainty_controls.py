import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "36_report_uncertainty_controls.py"
SPEC = importlib.util.spec_from_file_location("uncertainty_controls", SCRIPT_PATH)
uncertainty_controls = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(uncertainty_controls)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_uncertainty_control_report_preserves_no_cheating_boundary() -> None:
    report = uncertainty_controls.build_report()

    assert report["phase"] == "phase_53_reliability_uncertainty_control"
    assert report["accuracy_claim"].startswith("No field accuracy claim")
    assert "Do not retrain from this result." in report["no_cheating_boundary"]["forbidden"]
    assert report["no_cheating_boundary"]["frozen_test_set"].startswith("test_2026")


def test_uncertainty_control_report_tracks_likely_non_field_improvements() -> None:
    report = uncertainty_controls.build_report()
    changes = {item["change"] for item in report["likely_changes_without_expert_samples"]}
    drivers = {item["driver"] for item in report["risk_drivers"]}

    assert "More honest confidence bands" in changes
    assert "Lower review burden on future validation scenes" in changes
    assert "2026 domain/test-scene shift" in drivers
    assert "Mountainous terrain effects" in drivers


def test_frontend_loads_uncertainty_control_summary_when_available() -> None:
    api_source = read("web/src/api.ts")
    app_source = read("web/src/App.tsx")

    assert "ReliabilityControlSummary" in api_source
    assert "reliability_uncertainty_control_summary.json" in api_source
    assert "Reliability controls" in app_source
    assert "Likely changes without expert samples" in app_source
    assert "High caution" in app_source
    assert "high risk" not in app_source.lower()


def test_generated_web_summary_matches_no_accuracy_claim() -> None:
    summary_path = ROOT / "web" / "public" / "demo" / "reliability_uncertainty_control_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["accuracy_claim"] == (
        "No field accuracy claim; this is model-risk and weak-evidence guidance."
    )
    assert summary["current_reliability_snapshot"]["frozen_2026_mean_confidence"] < 0.6
