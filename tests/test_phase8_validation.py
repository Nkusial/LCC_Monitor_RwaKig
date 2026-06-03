import importlib.util
from pathlib import Path


VALIDATE_PATH = Path("pipelines/07_validate_outputs.py")
VALIDATE_SPEC = importlib.util.spec_from_file_location("validate_outputs", VALIDATE_PATH)
validate_outputs = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC.loader is not None
VALIDATE_SPEC.loader.exec_module(validate_outputs)

RUNNER_PATH = Path("pipelines/run_local_pipeline.py")
RUNNER_SPEC = importlib.util.spec_from_file_location("run_local_pipeline", RUNNER_PATH)
run_local_pipeline = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(run_local_pipeline)


def test_required_outputs_include_scored_geojson() -> None:
    assert "monitoring_change_scored.geojson" in validate_outputs.REQUIRED_OUTPUTS


def test_validation_checks_analysis_ready_rasters_only() -> None:
    assert "_s2_ndbi.tif" in validate_outputs.ANALYSIS_SUFFIXES
    assert "_s2_swir1.tif" not in validate_outputs.ANALYSIS_SUFFIXES


def test_runner_stage_order_keeps_validation_last() -> None:
    assert list(run_local_pipeline.STAGES)[-1] == "validate"
    assert "publish" in run_local_pipeline.STAGES
    assert "patches" in run_local_pipeline.STAGES
