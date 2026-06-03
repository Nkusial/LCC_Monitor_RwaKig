import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "21_validate_unet_full_aoi.py"
SPEC = importlib.util.spec_from_file_location("unet_reliability", SCRIPT_PATH)
unet_reliability = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(unet_reliability)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_model_to_harmonized_mapping_keeps_vegetation_compatible() -> None:
    assert unet_reliability.MODEL_TO_HARMONIZED[1] == {1}
    assert unet_reliability.MODEL_TO_HARMONIZED[2] == {2, 3}
    assert unet_reliability.MODEL_TO_HARMONIZED[3] == {4}
    assert unet_reliability.MODEL_TO_HARMONIZED[4] == {5}
    assert unet_reliability.MODEL_TO_HARMONIZED[5] == {6}


def test_phase37_docs_and_runner_are_wired() -> None:
    docs = read("docs/FULL_AOI_UNET_RELIABILITY.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/21_validate_unet_full_aoi.py")

    assert "Phase 37" in docs
    assert "Phase 37 full-AOI U-Net reliability audit" in phases
    assert "21_validate_unet_full_aoi.py" in readme
    assert "unet-reliability" in runner
    assert "not field accuracy" in script
