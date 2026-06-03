import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "11_ingest_external_weak_sources.py"
SPEC = importlib.util.spec_from_file_location("external_sources", SCRIPT_PATH)
external_sources = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(external_sources)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(read(path))


def test_external_candidate_sources_are_configured_for_ingestion() -> None:
    settings = load_yaml("configs/weak_labels.yaml")
    candidates = settings["candidate_sources"]

    assert candidates["esa_worldcover"]["enabled"] is True
    assert candidates["esa_worldcover"]["stac_collection"] == "esa-worldcover"
    assert candidates["osm_buildings_roads_landuse"]["enabled"] is True
    assert candidates["osm_buildings_roads_landuse"]["overpass_url"].startswith("https://")
    assert candidates["dynamic_world"]["earth_engine_asset"] == "GOOGLE/DYNAMICWORLD/V1"


def test_external_source_class_mappings_use_harmonized_ids() -> None:
    assert external_sources.ESA_TO_HARMONIZED[50] == external_sources.HARMONIZED_CLASS_IDS["built_up"]
    assert external_sources.ESA_TO_HARMONIZED[40] == external_sources.HARMONIZED_CLASS_IDS["managed_vegetation"]
    assert external_sources.DYNAMIC_WORLD_TO_HARMONIZED[6] == external_sources.HARMONIZED_CLASS_IDS["built_up"]
    assert external_sources.DYNAMIC_WORLD_TO_HARMONIZED[7] == external_sources.HARMONIZED_CLASS_IDS["bare_soil"]


def test_osm_tags_map_to_expected_harmonized_classes() -> None:
    assert external_sources.osm_class_id({"building": "yes"}) == external_sources.HARMONIZED_CLASS_IDS["built_up"]
    assert external_sources.osm_class_id({"highway": "residential"}) == external_sources.HARMONIZED_CLASS_IDS["built_up"]
    assert external_sources.osm_class_id({"landuse": "farmland"}) == external_sources.HARMONIZED_CLASS_IDS["managed_vegetation"]
    assert external_sources.osm_class_id({"landuse": "forest"}) == external_sources.HARMONIZED_CLASS_IDS["natural_vegetation"]


def test_phase26b_docs_explain_dynamic_world_export_and_agreement_masks() -> None:
    docs = read("docs/EXTERNAL_WEAK_SOURCES.md")
    phases = read("docs/PHASES.md")
    readme = read("README.md")

    assert "GOOGLE/DYNAMICWORLD/V1" in docs
    assert "weak_source_agreement_labels.tif" in docs
    assert "weak_source_disagreement_mask.tif" in docs
    assert "Phase 26B external weak-source ingestion" in phases
    assert "External weak sources" in readme
