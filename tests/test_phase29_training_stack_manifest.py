import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "14_prepare_training_stack_manifest.py"
SPEC = importlib.util.spec_from_file_location("training_stack_manifest", SCRIPT_PATH)
training_stack_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training_stack_manifest)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pair_readiness_requires_scenes_and_assets() -> None:
    pair = {
        "output_tag": "training_demo",
        "year": 2025,
        "split": "validation",
        "optical": {
            "date": "2025-02-20",
            "scene_ids": ["scene-a"],
            "max_tile_cloud_cover_percent": 3.05,
        },
        "radar": {"date": "2025-02-21", "time_gap_hours": 40.4},
    }
    scenes_by_id = {
        "scene-a": {
            "id": "scene-a",
            "properties": {"assets": {"B03": "...", "B04": "...", "B08": "...", "B11": "..."}},
        }
    }

    record = training_stack_manifest.pair_readiness(pair, scenes_by_id)

    assert record["ready_for_preprocessing"] is True
    assert record["missing_scenes"] == []
    assert record["missing_assets"] == {}


def test_phase29_docs_and_runner_are_wired() -> None:
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    docs = read("docs/TRAINING_SCENES.md")
    runner = read("pipelines/run_local_pipeline.py")

    assert "Phase 29 training stack manifest" in phases
    assert "training_stack_manifest_2023_2026.json" in readme
    assert "B03" in docs and "B11" in docs
    assert "14_prepare_training_stack_manifest.py" in runner
