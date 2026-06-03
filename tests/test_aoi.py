import json
from pathlib import Path

import geopandas as gpd


def test_aoi_geojson_exists_and_is_feature_collection() -> None:
    aoi_path = Path("configs/aoi.geojson")

    assert aoi_path.exists()

    data = json.loads(aoi_path.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    assert data["features"][0]["geometry"]["type"] == "Polygon"


def test_aoi_is_portfolio_scale() -> None:
    aoi = gpd.read_file("configs/aoi.geojson").to_crs("EPSG:32735")
    area_km2 = float(aoi.area.iloc[0] / 1_000_000)

    assert 300 <= area_km2 <= 500
