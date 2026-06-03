import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "33_prepare_2026_benchmark_package.py"
SPEC = importlib.util.spec_from_file_location("benchmark_package", SCRIPT_PATH)
benchmark_package = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark_package)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_expected_test_tag_uses_optical_and_radar_dates() -> None:
    candidate = {
        "date": "2026-01-26",
        "sentinel1_match": {"date": "2026-01-23"},
    }

    assert benchmark_package.expected_test_tag(candidate) == "test_2026_20260126_20260123"


def test_benchmark_package_blocks_missing_processed_stack(tmp_path: Path) -> None:
    status = benchmark_package.processed_feature_status(tmp_path, "test_2026_20260126_20260123")

    assert status["status"] == "missing_processed_stack"
    assert status["missing"] == [
        "s2_ndvi",
        "s2_ndwi",
        "s2_ndbi",
        "s1_vv",
        "s1_vh",
        "s1_vv_vh_ratio",
    ]


def test_benchmark_package_documents_no_cheating_protocol() -> None:
    source = read("pipelines/33_prepare_2026_benchmark_package.py")
    runner = read("pipelines/run_local_pipeline.py")

    assert "accuracy_claim_allowed" in source
    assert "Do not add 2026 patches to train or validation manifests." in source
    assert "Do not tune thresholds or class weights using 2026 benchmark outcomes." in source
    assert "benchmark-package" in runner
