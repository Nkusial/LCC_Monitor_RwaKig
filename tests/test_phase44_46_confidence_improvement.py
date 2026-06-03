from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_v2_configs_expand_features_without_breaking_baseline() -> None:
    baseline = load_yaml("configs/model_unet.yaml")
    v2_model = load_yaml("configs/model_unet_v2.yaml")
    v2_patches = load_yaml("configs/patch_dataset_v2.yaml")

    assert baseline["model"]["input_channels"] == 6
    assert v2_model["model"]["input_channels"] == len(v2_patches["feature_stack"])
    assert "s2_bsi" in v2_patches["feature_stack"]
    assert "s1_log_ratio" in v2_patches["feature_stack"]
    assert "slope" in v2_patches["feature_stack"]
    assert "tpi" in v2_patches["feature_stack"]


def test_confidence_improvement_scripts_are_wired() -> None:
    runner = read("pipelines/run_local_pipeline.py")
    docs = read("docs/CONFIDENCE_IMPROVEMENT_TRACK.md")
    phases = read("docs/PHASES.md")
    trainer = read("pipelines/15_train_unet_baseline.py")

    assert "feature-expansion" in runner
    assert "refined-pseudo-labels" in runner
    assert "weak-calibration" in runner
    assert "Phase 46 weak confidence calibration" in phases
    assert "Dynamic World" in docs
    assert "early stopping" in trainer


def test_topography_features_are_documented_and_wired() -> None:
    readme = read("README.md")
    phases = read("docs/PHASES.md")
    topo = read("docs/TOPOGRAPHY_FEATURES.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/29_ingest_topography.py")

    assert "29_ingest_topography.py" in readme
    assert "Phase 47 topography ingestion" in phases
    assert "Kigali's mountainous landscape" in topo
    assert "topography" in runner
    assert "cop-dem-glo-30" in script


def test_v3_imbalance_controls_are_wired() -> None:
    v3_model = load_yaml("configs/model_unet_v3.yaml")
    v3_patches = load_yaml("configs/patch_dataset_v3.yaml")
    trainer = read("pipelines/15_train_unet_baseline.py")
    balancer = read("pipelines/30_balance_training_patch_manifest.py")
    runner = read("pipelines/run_local_pipeline.py")
    docs = read("docs/CONFIDENCE_IMPROVEMENT_TRACK.md")
    phases = read("docs/PHASES.md")

    assert v3_model["model"]["input_channels"] == len(v3_patches["feature_stack"])
    assert v3_model["training"]["patch_manifest"].endswith("training_patch_manifest_v3_balanced.json")
    assert v3_model["training"]["loss"]["focal_weight"] > 0
    assert v3_model["training"]["loss"]["tversky_weight"] > 0
    assert "class_weight_overrides" in v3_model["training"]["loss"]
    assert v3_model["model"]["architecture"] in {"lightweight_unet", "lightweight_resunet"}
    assert v3_model["training"]["sampling"]["balanced_manifest"] is True
    assert v3_model["training"]["sampling"]["samples_per_epoch_multiplier"] > 1
    assert v3_model["training"]["sampling"]["minimum_minority_weight"] >= 4
    assert set(v3_model["training"]["sampling"]["minority_classes"]) >= {3, 4, 5}
    assert "weighted_cross_entropy_plus_generalized_dice_plus_focal" in trainer
    assert "tversky_loss" in trainer
    assert "retain all rare-class train patches" in balancer
    assert "balance-patches" in runner
    assert "Imbalance-aware v3 weak supervision" in phases
    assert "Tversky loss" in docs


def test_expert_sample_validation_design_is_present() -> None:
    sample = load_yaml("configs/expert_validation_sample.yaml")
    docs = read("docs/WEAK_SUPERVISION_RETRAINING.md")
    readme = read("README.md")

    assert sample["sample_size"] >= 100
    assert "review_zone" in sample["strata"]
    assert "minority_classes" in sample["strata"]
    assert "Dynamic World" in sample["review_sources"]
    assert "expert sample" in docs
    assert "WEAK_SUPERVISION_RETRAINING.md" in readme
