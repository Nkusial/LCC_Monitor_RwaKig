import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_alignment_report_passes_for_analysis_products() -> None:
    report = json.loads(read("data/outputs/geospatial_alignment_report.json"))

    assert report["status"] == "passed"
    assert report["checked_raster_count"] > 0
    assert report["raster_mismatch_count"] == 0
    assert report["vector_mismatch_count"] == 0
    assert "dynamic_world_label.tif" in report["skipped_raw_source_rasters"]


def test_alignment_validator_covers_all_downstream_raster_families() -> None:
    source = read("pipelines/37_validate_geospatial_alignment.py")

    for family in [
        "processed_sentinel_features",
        "change_detection_outputs",
        "external_weak_sources",
        "unet_full_aoi",
        "unet_full_aoi_v3",
        "refined_pseudo_labels",
        "topography_features",
    ]:
        assert family in source


def test_change_detection_validates_every_feature_against_master_grid() -> None:
    source = read("pipelines/04_detect_change.py")

    assert "validate_profile_against_master" in source
    assert "feature_paths" in source
    assert "before_radar" in source
    assert "after_radar" in source
    assert "Change-detection feature raster is not aligned" in source


def test_alignment_report_stays_available_without_sidebar_clutter() -> None:
    api_source = read("web/src/api.ts")
    app_source = read("web/src/App.tsx")
    exporter_source = read("pipelines/08_export_web_raster_layers.py")

    assert "geospatial_alignment_report.json" in api_source
    assert "loadGeospatialAlignmentSummary" not in app_source
    assert "Geospatial alignment" not in app_source
    assert "export_geospatial_alignment_summary" in exporter_source
    assert "DISPLAY_CRS = \"EPSG:3857\"" in exporter_source
    assert "calculate_default_transform" in exporter_source
    assert "reproject(" in exporter_source


def test_coregistration_qa_report_is_packaged_but_not_sidebar_content() -> None:
    report = json.loads(read("web/public/demo/coregistration_qa_report.json"))
    qa_source = read("pipelines/39_coregistration_qa.py")
    app_source = read("web/src/App.tsx")
    api_source = read("web/src/api.ts")

    assert report["status"] in {"passed", "warning", "failed"}
    assert report["checked_target_count"] > 0
    assert "Sentinel-2 NDVI" in report["method"]
    assert "estimate_shift" in qa_source
    assert "Co-registration QA" not in app_source
    assert "loadCoregistrationQaSummary" in api_source
