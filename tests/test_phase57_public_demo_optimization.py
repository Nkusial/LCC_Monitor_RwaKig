import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "42_optimize_public_demo_rasters.py"


def load_module():
    spec = importlib.util.spec_from_file_location("public_demo_optimizer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_demo_optimizer_targets_only_manifest_oversized_pngs() -> None:
    source = read("pipelines/42_optimize_public_demo_rasters.py")

    assert "oversized_assets" in source
    assert ".png" in source
    assert "raster_tiles" not in source
    assert "Only fallback/demo PNG files are optimized" in source


def test_public_demo_optimizer_report_preserves_geospatial_safety_note() -> None:
    module = load_module()
    report = module.build_report([])

    assert report["phase"] == "phase_57_public_demo_raster_optimization"
    assert "EPSG:3857 XYZ raster tiles" in report["geospatial_safety_note"]
    assert report["optimized_file_count"] == 0
