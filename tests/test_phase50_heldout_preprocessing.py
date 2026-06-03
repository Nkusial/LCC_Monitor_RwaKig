import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "34_preprocess_2026_test_pair.py"
SPEC = importlib.util.spec_from_file_location("test_2026_preprocess", SCRIPT_PATH)
test_2026_preprocess = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(test_2026_preprocess)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_test_pair_uses_test_split_and_tag() -> None:
    candidate = {
        "date": "2026-01-26",
        "average_cloud_cover_percent": 15.09,
        "max_tile_cloud_cover_percent": 23.07,
        "tiles": {
            "35MRT": {"scene_id": "S2_MRT", "cloud_cover_percent": 7.1},
            "35MRU": {"scene_id": "S2_MRU", "cloud_cover_percent": 23.07},
        },
        "sentinel1_match": {
            "date": "2026-01-23",
            "scene_id": "S1_RTC",
            "orbit": "ascending",
            "polarizations": ["VV", "VH"],
            "time_gap_hours": 55.7,
        },
    }

    pair = test_2026_preprocess.build_test_pair(candidate)

    assert pair["split"] == "test"
    assert pair["output_tag"] == "test_2026_20260126_20260123"
    assert pair["optical"]["scene_ids"] == ["S2_MRT", "S2_MRU"]


def test_heldout_preprocessing_documents_no_cheating() -> None:
    source = read("pipelines/34_preprocess_2026_test_pair.py")
    runner = read("pipelines/run_local_pipeline.py")

    assert "held_out_2026_test_only_scene_catalog" in source
    assert "Do not add this tag to train or validation patch manifests." in source
    assert "Do not tune model architecture, class weights, thresholds, or pseudo-label filters" in source
    assert "test-2026-preprocess" in runner


def test_display_path_handles_relative_paths() -> None:
    assert test_2026_preprocess.display_path(Path("data/processed/example.tif")) == "data/processed/example.tif"
