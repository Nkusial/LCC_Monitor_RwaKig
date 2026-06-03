import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "17_generate_training_patch_dataset.py"
SPEC = importlib.util.spec_from_file_location("training_patches", SCRIPT_PATH)
training_patches = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training_patches)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_multi_year_patch_config_points_unet_to_training_manifest() -> None:
    patch_config = load_yaml("configs/patch_dataset.yaml")
    model_config = load_yaml("configs/model_unet.yaml")

    assert patch_config["multi_year_manifest"] == "data/interim/training_patches/training_patch_manifest.json"
    assert patch_config["max_patches_per_training_pair"] > 0
    assert model_config["training"]["patch_manifest"] == patch_config["multi_year_manifest"]


def test_training_pairs_filter_supports_split_and_limit() -> None:
    summary_path = ROOT / "data" / "catalog" / "training_preprocessing_summary_2023_2026.json"

    pairs = training_patches.training_pairs(summary_path, split="validation", limit=2)

    assert len(pairs) == 2
    assert all(pair["split"] == "validation" for pair in pairs)


def test_phase32_docs_and_runner_are_wired() -> None:
    docs = read("docs/MULTI_YEAR_PATCHES.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    validation = read("pipelines/07_validate_outputs.py")

    assert "2023 -> train" in docs
    assert "Phase 32 multi-year training patches" in phases
    assert "17_generate_training_patch_dataset.py" in readme
    assert "training-patches" in runner
    assert "multi_year_training_patch_manifest_available" in validation
