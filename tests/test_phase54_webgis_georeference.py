import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def corners_from_aoi() -> list[list[float]]:
    aoi = json.loads((ROOT / "web" / "public" / "demo" / "aoi.geojson").read_text(encoding="utf-8"))
    ring = aoi["features"][0]["geometry"]["coordinates"][0]
    return ring[:4]


def bbox(coordinates: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [coordinate[0] for coordinate in coordinates]
    ys = [coordinate[1] for coordinate in coordinates]
    return min(xs), min(ys), max(xs), max(ys)


def test_webgis_aoi_and_raster_layers_share_the_same_master_grid_footprint() -> None:
    manifest = json.loads(
        (ROOT / "web" / "public" / "demo" / "raster_layers.json").read_text(encoding="utf-8")
    )
    aoi_coordinates = corners_from_aoi()
    aoi_bbox = bbox(aoi_coordinates)

    assert manifest["grid_alignment"]["status"] == "validated"
    assert "hosted_aoi" in manifest["grid_alignment"]["checks"]
    assert manifest["grid_alignment"]["display_aoi"] == "web/public/demo/aoi.geojson"
    assert manifest["display_alignment"]["status"] == "tiled"
    assert manifest["display_alignment"]["display_crs"] == "EPSG:3857"
    assert manifest["display_alignment"]["tile_scheme"] == "XYZ"

    for layer in manifest["layers"]:
        assert len(layer["coordinates"]) == 4
        layer_bbox = bbox(layer["coordinates"])
        assert abs(layer_bbox[0] - aoi_bbox[0]) < 0.001
        assert abs(layer_bbox[1] - aoi_bbox[1]) < 0.001
        assert abs(layer_bbox[2] - aoi_bbox[2]) < 0.001
        assert abs(layer_bbox[3] - aoi_bbox[3]) < 0.001


def test_backend_config_aoi_matches_hosted_webgis_aoi() -> None:
    config_aoi = json.loads((ROOT / "configs" / "aoi.geojson").read_text(encoding="utf-8"))
    hosted_aoi = json.loads((ROOT / "web" / "public" / "demo" / "aoi.geojson").read_text(encoding="utf-8"))

    assert config_aoi["features"][0]["properties"]["source"] == "configs/master_grid.yaml"
    assert (
        config_aoi["features"][0]["geometry"]["coordinates"]
        == hosted_aoi["features"][0]["geometry"]["coordinates"]
    )


def test_exporter_uses_master_grid_as_hosted_display_source_of_truth() -> None:
    source = (ROOT / "pipelines" / "08_export_web_raster_layers.py").read_text(encoding="utf-8")

    assert "def master_grid_aoi_geojson" in source
    assert "display_aoi" in source
    assert "calculate_default_transform" in source
    assert "reproject(" in source
    assert "DISPLAY_CRS = \"EPSG:3857\"" in source
    assert "write_tile_pyramid" in source
