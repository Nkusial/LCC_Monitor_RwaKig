import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "16_preprocess_training_pairs.py"
SPEC = importlib.util.spec_from_file_location("training_preprocess", SCRIPT_PATH)
training_preprocess = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training_preprocess)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_training_pair_filters_are_explicit() -> None:
    pairs = [
        {"year": 2023, "split": "train", "output_tag": "a"},
        {"year": 2024, "split": "train", "output_tag": "b"},
        {"year": 2025, "split": "validation", "output_tag": "c"},
    ]

    selected = training_preprocess.filter_pairs(pairs, split="train", year=2024, limit=1)

    assert [pair["output_tag"] for pair in selected] == ["b"]


def test_training_preprocess_expected_outputs_are_feature_complete() -> None:
    pair = {"output_tag": "training_demo"}

    outputs = training_preprocess.expected_training_outputs(pair)

    assert "data/processed/training_demo_s2_ndvi.tif" in outputs
    assert "data/processed/training_demo_s2_ndwi.tif" in outputs
    assert "data/processed/training_demo_s2_ndbi.tif" in outputs
    assert "data/processed/training_demo_s1_vv.tif" in outputs
    assert "data/processed/training_demo_s1_vh.tif" in outputs
    assert "data/processed/training_demo_s1_vv_vh_ratio.tif" in outputs


def test_phase30_docs_and_runner_are_wired() -> None:
    docs = read("docs/MULTI_YEAR_PREPROCESSING.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")
    runner = read("pipelines/run_local_pipeline.py")
    validation = read("pipelines/07_validate_outputs.py")

    assert "--mode optical --split train --limit 1" in docs
    assert "Phase 30 multi-year training preprocessing" in phases
    assert "16_preprocess_training_pairs.py" in readme
    assert "training-preprocess" in runner
    assert "training_analysis_raster_count" in validation
    assert "training_master_grid_mismatches" in validation
