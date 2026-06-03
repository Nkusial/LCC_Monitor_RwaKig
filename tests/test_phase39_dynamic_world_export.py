import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "pipelines" / "22_export_dynamic_world.py"
SPEC = importlib.util.spec_from_file_location("dynamic_world_export", SCRIPT_PATH)
dynamic_world_export = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dynamic_world_export)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_date_window_includes_requested_days() -> None:
    start, end = dynamic_world_export.date_window("2025-04-21", 15, 15)
    assert start == "2025-04-06"
    assert end == "2025-05-07"


def test_monitoring_tag_date_parser_uses_optical_date() -> None:
    assert dynamic_world_export.parse_monitoring_date("monitoring_20250421_20250419") == "2025-04-21"


def test_dynamic_world_cli_paths_resolve_from_project_root() -> None:
    relative = dynamic_world_export.project_path("data/interim/external_sources/example.tif")

    assert relative.is_absolute()
    assert relative == ROOT / "data" / "interim" / "external_sources" / "example.tif"


def test_phase39_docs_and_runner_are_wired() -> None:
    readme = read("README.md")
    phases = read("docs/PHASES.md")
    docs = read("docs/EXTERNAL_WEAK_SOURCES.md")
    runner = read("pipelines/run_local_pipeline.py")
    script = read("pipelines/22_export_dynamic_world.py")

    assert "22_export_dynamic_world.py" in readme
    assert "Phase 39 Dynamic World ingestion path" in phases
    assert "YOUR_GOOGLE_CLOUD_PROJECT" in docs
    assert "dynamic-world" in runner
    assert "earth_engine_asset" in script
    assert "project_path" in script
