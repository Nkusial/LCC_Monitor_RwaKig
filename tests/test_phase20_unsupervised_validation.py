from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase20_unsupervised_validation_script_is_present():
    script = ROOT / "pipelines" / "09_unsupervised_validation.py"
    source = script.read_text(encoding="utf-8")

    assert "MiniBatchKMeans" in source
    assert "DEFAULT_CLUSTER_COUNTS = [4, 5, 6, 8, 10]" in source
    assert "FEATURE_SETS" in source
    assert "PCA" in source
    assert "high_confidence_change" in source
    assert "best_experiment" in source
    assert "unsupervised_validation_report.json" in source
    assert "ground_truth_available" in source
    assert "No field accuracy is claimed" in source
    assert "monitoring_change_scored.geojson" in source


def test_phase20_is_documented_as_internal_validation():
    docs = (ROOT / "docs" / "UNSUPERVISED_VALIDATION.md").read_text(encoding="utf-8")
    phases = (ROOT / "docs" / "PHASES.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "internal unsupervised cluster audit" in docs
    assert "Phase 20B" in docs
    assert "0.4459" in docs
    assert "This is not field accuracy" in docs
    assert "Phase 20 unsupervised validation" in phases
    assert "Phase 20B validation comparison" in phases
    assert "09_unsupervised_validation.py" in readme
