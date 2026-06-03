import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


temporal = import_script("temporal_consensus", "pipelines/23_temporal_unet_consensus.py")
retraining = import_script("retraining_review", "pipelines/24_prepare_retraining_review.py")
test_readiness = import_script("test_readiness", "pipelines/25_report_test_set_readiness.py")
no_leakage = import_script("no_leakage", "pipelines/32_validate_no_leakage.py")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_temporal_consensus_module_defines_outputs() -> None:
    assert temporal.DEFAULT_REPORT.name == "unet_temporal_consensus_report.json"
    assert "unet_full_aoi_manifest.json" in str(temporal.DEFAULT_MANIFEST)


def test_retraining_review_manifest_contract_is_explicit() -> None:
    assert retraining.DEFAULT_OUTPUT.name == "review_retraining_manifest.json"
    source = read("pipelines/24_prepare_retraining_review.py")
    assert "corrected_reference_labels.geojson" in source
    assert "do_not_train_on" in source


def test_test_readiness_reports_not_ready_when_test_split_missing() -> None:
    source = read("pipelines/25_report_test_set_readiness.py")
    assert "not_ready" in source
    assert "2026 optional catalog" in source


def test_no_leakage_audit_blocks_test_data_from_training() -> None:
    source = read("pipelines/32_validate_no_leakage.py")

    assert "2026+ scene pair" in source
    assert "2026+ patch" in source
    assert "May use 2026 held-out candidates only after the model is frozen." in source
    assert "must not be described as field truth" in source


def test_deployment_and_docs_are_wired() -> None:
    readme = read("README.md")
    phases = read("docs/PHASES.md")
    hardening = read("docs/RELIABILITY_HARDENING.md")
    runner = read("pipelines/run_local_pipeline.py")
    compose = read("docker-compose.deploy.yml")
    dockerfile = read("backend/Dockerfile")

    assert "23_temporal_unet_consensus.py" in readme
    assert "Phase 43 deployment hardening" in phases
    assert "docker compose -f docker-compose.deploy.yml up --build" in hardening
    assert "temporal-consensus" in runner
    assert "no-leakage" in runner
    assert "No-Leakage Audit" in hardening
    assert "backend" in compose
    assert "uvicorn" in dockerfile
