import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "12_generate_patch_dataset.py"
SPEC = importlib.util.spec_from_file_location("patch_dataset", SCRIPT_PATH)
patch_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(patch_dataset)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_patch_dataset_config_uses_master_grid_ready_inputs() -> None:
    settings = load_yaml("configs/patch_dataset.yaml")

    assert settings["patch_size"] == 128
    assert settings["stride"] == 128
    assert settings["minimum_training_pixels"] >= 1
    assert "s2_ndvi" in settings["feature_stack"]
    assert "s1_vv_vh_ratio" in settings["feature_stack"]
    assert "disagreement_mask" in settings["label_sources"]
    assert "agreement_labels" in settings["label_sources"]


def test_patch_split_policy_is_spatial_not_random_pixel() -> None:
    settings = load_yaml("configs/patch_dataset.yaml")
    split_policy = settings["split_policy"]

    assert split_policy["method"] == "spatial_block_modulo"
    assert patch_dataset.split_for_block("r000_c000_00000", split_policy) == "train"
    assert patch_dataset.split_for_block("r000_c000_00006", split_policy) == "validation"
    assert patch_dataset.split_for_block("r000_c000_00009", split_policy) == "test"


def test_patch_dataset_docs_explain_masks_and_local_artifacts() -> None:
    docs = read("docs/PATCH_DATASET.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")

    assert "training_mask" in docs
    assert "spatial block id" in docs
    assert "not committed to Git" in docs
    assert "Phase 27 patch dataset" in phases
    assert "Patch dataset" in readme
