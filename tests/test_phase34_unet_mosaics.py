import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "19_mosaic_unet_predictions.py"
SPEC = importlib.util.spec_from_file_location("unet_mosaics", SCRIPT_PATH)
unet_mosaics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(unet_mosaics)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_unet_mosaic_config_outputs_are_explicit() -> None:
    config = load_yaml("configs/model_unet.yaml")

    assert config["outputs"]["mosaic_dir"] == "data/interim/unet_mosaics"
    assert config["outputs"]["mosaic_manifest"] == "data/interim/unet_mosaics/unet_mosaic_manifest.json"


def test_group_records_by_tag_supports_split_filter() -> None:
    records = [
        {"training_tag": "a", "split": "train"},
        {"training_tag": "a", "split": "validation"},
        {"training_tag": "b", "split": "validation"},
    ]

    grouped = unet_mosaics.group_records_by_tag(records, split="validation")

    assert set(grouped) == {"a", "b"}
    assert len(grouped["a"]) == 1


def test_phase34_docs_and_runner_are_wired() -> None:
    docs = read("docs/SEGMENTATION_MOSAICS.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/19_mosaic_unet_predictions.py")

    assert "Phase 34" in docs
    assert "Phase 34 U-Net prediction mosaics" in phases
    assert "19_mosaic_unet_predictions.py" in readme
    assert "unet-mosaics" in runner
    assert "dominant_class" in script
    assert "patch_coverage" in script
