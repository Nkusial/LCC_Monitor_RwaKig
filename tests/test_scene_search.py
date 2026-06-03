import json
from pathlib import Path

from pipelines import common
import importlib.util


SCRIPT_PATH = Path("pipelines/01_search_scenes.py")
SPEC = importlib.util.spec_from_file_location("scene_search", SCRIPT_PATH)
scene_search = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(scene_search)


def test_to_psycopg_url_removes_sqlalchemy_driver() -> None:
    assert (
        scene_search.to_psycopg_url("postgresql+psycopg://user:pass@localhost/db")
        == "postgresql://user:pass@localhost/db"
    )


def test_write_scene_catalog_creates_feature_collection(tmp_path: Path) -> None:
    output_path = tmp_path / "scenes.geojson"
    feature = {
        "type": "Feature",
        "id": "scene-1",
        "geometry": None,
        "properties": {"id": "scene-1"},
    }

    scene_search.write_scene_catalog([feature], output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert data["features"][0]["id"] == "scene-1"


def test_pipeline_config_loads() -> None:
    config = common.load_config()

    assert config["sentinel"]["collections"]["sentinel2"] == "sentinel-2-l2a"
