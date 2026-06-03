import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "18_run_unet_inference.py"
SPEC = importlib.util.spec_from_file_location("unet_inference", SCRIPT_PATH)
unet_inference = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(unet_inference)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_unet_inference_config_outputs_are_explicit() -> None:
    config = load_yaml("configs/model_unet.yaml")

    assert config["outputs"]["checkpoint_path"] == "data/interim/models/unet_baseline.pt"
    assert config["outputs"]["inference_dir"] == "data/interim/unet_inference"
    assert config["outputs"]["inference_manifest"] == "data/interim/unet_inference/unet_inference_manifest.json"


def test_select_patch_records_supports_split_and_limit() -> None:
    manifest = {
        "patches": [
            {"patch_id": "a", "split": "train"},
            {"patch_id": "b", "split": "validation"},
            {"patch_id": "c", "split": "validation"},
        ]
    }

    selected = unet_inference.select_patch_records(manifest, split="validation", limit=1)

    assert [record["patch_id"] for record in selected] == ["b"]


def test_phase33_docs_and_runner_are_wired() -> None:
    docs = read("docs/SEGMENTATION_INFERENCE.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/18_run_unet_inference.py")

    assert "Phase 33" in docs
    assert "Phase 33 U-Net patch inference" in phases
    assert "18_run_unet_inference.py" in readme
    assert "unet-inference" in runner
    assert "dominant_class" in script
    assert "entropy" in script
