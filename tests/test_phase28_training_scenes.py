import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "13_discover_training_scenes.py"
SPEC = importlib.util.spec_from_file_location("training_scenes", SCRIPT_PATH)
training_scenes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training_scenes)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_training_scene_discovery_config_uses_temporal_splits() -> None:
    config = load_yaml("configs/pipeline.yaml")
    settings = config["training_scene_discovery"]

    assert settings["preferred_max_cloud_cover"] == 10
    assert settings["fallback_max_cloud_cover"] > settings["preferred_max_cloud_cover"]
    assert settings["max_radar_time_gap_hours"] <= 96
    assert settings["max_pairs_per_year"] == 4
    assert settings["required_years"] == [2023, 2024, 2025]
    assert settings["optional_years"] == [2026]
    assert settings["required_tiles"] == ["35MRT", "35MRU"]
    assert settings["split_by_year"]["train"] == [2023, 2024]
    assert settings["split_by_year"]["validation"] == [2025]
    assert settings["split_by_year"]["test"] == [2026]


def test_training_scene_split_helper_matches_config() -> None:
    settings = load_yaml("configs/pipeline.yaml")["training_scene_discovery"]

    assert training_scenes.split_for_year(settings, 2023) == "train"
    assert training_scenes.split_for_year(settings, 2024) == "train"
    assert training_scenes.split_for_year(settings, 2025) == "validation"
    assert training_scenes.split_for_year(settings, 2026) == "test"
    assert training_scenes.split_for_year(settings, 2027) == "unused"


def test_training_scene_docs_and_runner_are_wired() -> None:
    docs = read("docs/TRAINING_SCENES.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")

    assert "Sentinel-2 L2A cloud cover < 10%" in docs
    assert "35MRT cloud %" in docs
    assert "35MRU cloud %" in docs
    assert "2025-02-20 | 2.97 | 3.05 | 3.01" in docs
    assert "2023 -> train" in docs
    assert "2025 -> validation" in docs
    assert "2026 -> test" in docs
    assert "Phase 28 multi-year training scene discovery" in phases
    assert "Training scene discovery" in readme
    assert "13_discover_training_scenes.py" in runner
