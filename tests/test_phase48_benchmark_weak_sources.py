import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "11_ingest_external_weak_sources.py"
SPEC = importlib.util.spec_from_file_location("external_weak_sources", SCRIPT_PATH)
external_weak_sources = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(external_weak_sources)


def test_suffixed_path_keeps_benchmark_outputs_separate(tmp_path: Path) -> None:
    path = external_weak_sources.suffixed_path(
        tmp_path,
        "dynamic_world_harmonized",
        "test_2026",
    )

    assert path.name == "dynamic_world_harmonized_test_2026.tif"


def test_benchmark_cli_supports_dynamic_world_override_and_bootstrap_exclusion() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--dynamic-world-raster" in source
    assert "--output-suffix" in source
    assert "--reference-grid" in source
    assert "--exclude-local-bootstrap" in source
    assert "local_bootstrap_included" in source
    assert "held-out test years" in source
    assert "leakage_policy" in source


def test_benchmark_ingestion_validates_georeference_and_aoi_mask() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "aoi_mask_for_profile" in source
    assert "apply_aoi_mask" in source
    assert "alignment_report" in source
    assert "outside_aoi_nonzero_pixels" in source
    assert "georeference_and_aoi_compatibility" in source
