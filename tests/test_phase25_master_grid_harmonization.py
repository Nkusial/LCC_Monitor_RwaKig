import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
VALIDATE_PATH = ROOT / "pipelines" / "07_validate_outputs.py"
VALIDATE_SPEC = importlib.util.spec_from_file_location("validate_outputs", VALIDATE_PATH)
validate_outputs = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC.loader is not None
VALIDATE_SPEC.loader.exec_module(validate_outputs)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_master_grid_contract_is_explicit_and_ml_ready() -> None:
    grid = load_yaml("configs/master_grid.yaml")

    assert grid["crs"] == "EPSG:32735"
    assert grid["resolution_m"] == 10
    assert grid["width"] > 0
    assert grid["height"] > 0
    assert grid["transform"]["a"] == 10.0
    assert grid["transform"]["e"] == -10.0
    assert grid["alignment_policy"]["fail_on_crs_mismatch"] is True
    assert grid["alignment_policy"]["fail_on_transform_mismatch"] is True


def test_validation_script_compares_rasters_to_master_grid() -> None:
    grid = load_yaml("configs/master_grid.yaml")
    expected = validate_outputs.expected_grid(grid)

    assert expected["crs"] == "EPSG:32735"
    assert expected["transform"] == [10.0, 0.0, 830650.0, 0.0, -10.0, 9794800.0]
    assert "master_grid_mismatches" in read("pipelines/07_validate_outputs.py")


def test_class_harmonization_taxonomy_preserves_current_groups() -> None:
    harmonization = load_yaml("configs/class_harmonization.yaml")
    target_classes = harmonization["target_classes"]
    current_mapping = harmonization["rules"]["current_group_mapping"]

    assert set(target_classes) == {
        "built_up",
        "managed_vegetation",
        "natural_vegetation",
        "bare_soil",
        "water_wetland",
        "uncertain_mixed",
    }
    assert current_mapping["vegetation"]["requires_additional_evidence"] is True
    assert current_mapping["bare_sparse"]["harmonized"] == ["bare_soil"]
    assert current_mapping["mixed"]["harmonized"] == ["uncertain_mixed"]


def test_phase25_docs_are_implementation_docs_not_blueprint_copy() -> None:
    readme = read("README.md")
    hosted_index = read("docs/index.md")
    phases = read("docs/PHASES.md")
    class_doc = read("docs/CLASS_HARMONIZATION.md")

    assert "Class harmonization" in readme
    assert "Class harmonization" in hosted_index
    assert "Phase 25 master-grid and class harmonization" in phases
    assert "configs/class_harmonization.yaml" in class_doc
    assert "BLUEPRINT_V2_PROMPT_ENGINEERING" not in readme + hosted_index + phases
