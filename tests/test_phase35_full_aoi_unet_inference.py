import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "20_full_aoi_unet_inference.py"
SPEC = importlib.util.spec_from_file_location("full_aoi_unet", SCRIPT_PATH)
full_aoi_unet = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(full_aoi_unet)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_full_aoi_config_outputs_are_explicit() -> None:
    config = load_yaml("configs/model_unet.yaml")

    assert config["outputs"]["full_aoi_dir"] == "data/interim/unet_full_aoi"
    assert config["outputs"]["full_aoi_manifest"] == "data/interim/unet_full_aoi/unet_full_aoi_manifest.json"
    assert config["inference"]["stride"] <= config["inference"]["tile_size"]


def test_full_coverage_windows_include_edges() -> None:
    windows = list(full_aoi_unet.iter_full_coverage_windows(height=300, width=260, tile_size=128, stride=64))

    assert (0, 0) in windows
    assert (172, 132) in windows


def test_phase35_docs_and_runner_are_wired() -> None:
    docs = read("docs/FULL_AOI_UNET_INFERENCE.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/20_full_aoi_unet_inference.py")

    assert "Phase 35" in docs
    assert "Phase 35 full-AOI U-Net inference" in phases
    assert "20_full_aoi_unet_inference.py" in readme
    assert "unet-full-aoi" in runner
    assert "iter_full_coverage_windows" in script
    assert "full_aoi_manifest" in script
