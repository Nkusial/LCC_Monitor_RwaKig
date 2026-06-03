import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "10_generate_weak_labels.py"
SPEC = importlib.util.spec_from_file_location("weak_labels", SCRIPT_PATH)
weak_labels = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(weak_labels)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_weak_label_candidate_sources_match_project_decision() -> None:
    settings = load_yaml("configs/weak_labels.yaml")
    candidates = settings["candidate_sources"]

    assert set(candidates) == {
        "dynamic_world",
        "esa_worldcover",
        "osm_buildings_roads_landuse",
        "sentinel2_ndvi_ndwi_ndbi",
        "sentinel1_vv_vh",
        "existing_high_confidence_change_polygons",
    }
    assert candidates["sentinel2_ndvi_ndwi_ndbi"]["enabled"] is True
    assert candidates["sentinel1_vv_vh"]["enabled"] is True
    assert candidates["existing_high_confidence_change_polygons"]["enabled"] is True
    assert candidates["dynamic_world"]["enabled"] is False
    assert candidates["esa_worldcover"]["enabled"] is True
    assert candidates["osm_buildings_roads_landuse"]["enabled"] is True


def test_weak_label_class_order_matches_harmonized_taxonomy() -> None:
    assert weak_labels.CLASS_ORDER == [
        "built_up",
        "managed_vegetation",
        "natural_vegetation",
        "bare_soil",
        "water_wetland",
        "uncertain_mixed",
    ]


def test_phase26_is_documented_without_accuracy_claim() -> None:
    docs = read("docs/WEAK_LABELS.md")
    readme = read("README.md")
    phases = read("docs/PHASES.md")

    assert "Dynamic World" in docs
    assert "ESA WorldCover" in docs
    assert "OSM buildings/roads/landuse" in docs
    assert "does not claim classification accuracy" in docs
    assert "Weak labels" in readme
    assert "Phase 26 weak-label generation" in phases
