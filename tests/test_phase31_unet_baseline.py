from pathlib import Path

import yaml

from ml.unet import ARCHITECTURE_SUMMARY


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_unet_config_matches_patch_feature_stack() -> None:
    model_config = load_yaml("configs/model_unet.yaml")
    patch_config = load_yaml("configs/patch_dataset.yaml")

    assert model_config["model"]["architecture"] == "lightweight_unet"
    assert model_config["model"]["input_channels"] == len(patch_config["feature_stack"])
    assert model_config["model"]["output_classes"] == len(model_config["model"]["class_names"])
    assert model_config["training"]["ignore_index"] == 0
    assert model_config["training"]["loss"]["class_weight_method"] == "effective_number"
    assert model_config["training"]["sampling"]["class_aware"] is True
    assert model_config["outputs"]["checkpoint_path"] == "data/interim/models/unet_baseline.pt"


def test_unet_architecture_summary_is_explicit() -> None:
    assert ARCHITECTURE_SUMMARY["name"] == "lightweight_unet"
    assert ARCHITECTURE_SUMMARY["input_channels"] == 6
    assert ARCHITECTURE_SUMMARY["output_classes"] == 6
    assert "output" in ARCHITECTURE_SUMMARY


def test_unet_docs_runner_and_report_are_wired() -> None:
    docs = read("docs/SEGMENTATION_BASELINE.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/15_train_unet_baseline.py")

    assert "Phase 31" in docs
    assert "U-Net" in docs
    assert "Phase 31 U-Net segmentation baseline" in phases
    assert "15_train_unet_baseline.py" in readme
    assert "unet-preflight" in runner
    assert "ready_for_training" in script
    assert "accuracy_claim" in script
    assert "validation_loss_by_epoch" in script
    assert "checkpoint_path" in script
    assert "weighted_cross_entropy_plus_generalized_dice" in script
    assert "validation_metrics_by_class" in script
