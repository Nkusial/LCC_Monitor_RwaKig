import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "35_run_2026_frozen_unet_benchmark.py"
SPEC = importlib.util.spec_from_file_location("frozen_2026_benchmark", SCRIPT_PATH)
frozen_2026_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(frozen_2026_benchmark)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_build_test_pair_keeps_split_as_test() -> None:
    summary = {
        "output_tag": "test_2026_20260126_20260123",
        "selected_candidate": {
            "date": "2026-01-26",
            "average_cloud_cover_percent": 15.09,
            "max_tile_cloud_cover_percent": 23.07,
            "sentinel1_match": {
                "date": "2026-01-23",
                "time_gap_hours": 55.7,
            },
        },
    }

    pair = frozen_2026_benchmark.build_test_pair(summary)

    assert pair["output_tag"] == "test_2026_20260126_20260123"
    assert pair["split"] == "test"
    assert pair["year"] == 2026


def test_frozen_benchmark_documents_no_tuning_or_accuracy_claims() -> None:
    source = read("pipelines/35_run_2026_frozen_unet_benchmark.py")
    runner = read("pipelines/run_local_pipeline.py")

    assert "Frozen-model inference" in source
    assert "Do not retrain from this result." in source
    assert "Do not tune thresholds, architecture, class weights, or pseudo-label filters" in source
    assert "Do not claim field accuracy from weak-source agreement." in source
    assert "test-2026-unet" in runner
