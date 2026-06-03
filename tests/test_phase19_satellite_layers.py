import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase19_raster_layer_manifest_exists():
    manifest_path = ROOT / "web" / "public" / "demo" / "raster_layers.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = manifest["layers"]
    layer_ids = {layer["id"] for layer in layers}

    assert "s2_false_color_20250220" in layer_ids
    assert "s2_ndvi_20250220" in layer_ids
    assert "s2_ndwi_20250220" in layer_ids
    assert "s2_ndbi_20250220" in layer_ids
    assert "s1_vv_20250221" in layer_ids
    assert "s1_vh_20250221" in layer_ids
    assert "s1_ratio_20250221" in layer_ids
    assert "delta_ndvi_latest" in layer_ids
    assert "change_confidence_latest" in layer_ids
    assert "unet_full_aoi_dominant_class" in layer_ids
    assert "unet_full_aoi_confidence" in layer_ids
    assert "unet_full_aoi_entropy" in layer_ids

    for layer in layers:
        assert layer["satelliteDerived"] is True
        assert len(layer["coordinates"]) == 4
        assert layer["tile_template"].endswith("/{z}/{x}/{y}.png")
        assert layer["maxzoom"] == 14
        assert (ROOT / "web" / "public" / "demo" / layer["path"]).exists()


def test_phase19_hosted_app_receives_raster_layers():
    docs_manifest = ROOT / "docs" / "app" / "demo" / "raster_layers.json"
    assert docs_manifest.exists()

    app_source = (ROOT / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    map_source = (ROOT / "web" / "src" / "map" / "MapView.tsx").read_text(
        encoding="utf-8"
    )

    assert "Satellite-derived layer" in app_source
    assert "Change polygon color" in app_source
    assert "baseline" in app_source
    assert "magnitude" in app_source
    assert "satellite-raster" in map_source
    assert "tile_template" in map_source
    assert "tiles: [demoPath(rasterLayer.tile_template)]" in map_source


def test_phase36_unet_full_aoi_layers_are_labeled_as_model_outputs():
    manifest_path = ROOT / "web" / "public" / "demo" / "raster_layers.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = {layer["id"]: layer for layer in manifest["layers"]}

    dominant = layers["unet_full_aoi_dominant_class"]
    confidence = layers["unet_full_aoi_confidence"]
    entropy = layers["unet_full_aoi_entropy"]

    assert dominant["group"] == "U-Net model outputs"
    assert "not field-validated accuracy" in dominant["description"]
    assert "Maximum class probability" in confidence["description"]
    assert "uncertainty" in entropy["description"]


def test_hosted_raster_export_masks_tile_alpha_to_aoi_polygon():
    script = (ROOT / "pipelines" / "08_export_web_raster_layers.py").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (ROOT / "web" / "public" / "demo" / "raster_layers.json").read_text(
            encoding="utf-8"
        )
    )

    assert "aoi_mask_for_display" in script
    assert "rasterize(" in script
    assert script.count("apply_aoi_alpha(") >= 5
    assert "AOI-polygon alpha mask" in manifest["note"]
    assert manifest["display_alignment"]["status"] == "tiled"
    assert manifest["display_alignment"]["display_crs"] == "EPSG:3857"
    assert manifest["display_alignment"]["tile_scheme"] == "XYZ"


def test_hosted_raster_export_validates_master_grid_before_display_masking():
    script = (ROOT / "pipelines" / "08_export_web_raster_layers.py").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (ROOT / "web" / "public" / "demo" / "raster_layers.json").read_text(
            encoding="utf-8"
        )
    )

    assert "MASTER_GRID_PATH" in script
    assert "validate_export_grid(dataset)" in script
    assert "Hosted raster is not aligned to configs/master_grid.yaml" in script
    assert "calculate_default_transform" in script
    assert "reproject(" in script
    assert "write_tile_pyramid" in script
    assert manifest["grid_alignment"]["status"] == "validated"
    assert manifest["grid_alignment"]["reference"] == "configs/master_grid.yaml"


def test_webgis_uses_reader_friendly_change_labels():
    app_source = (ROOT / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    map_source = (ROOT / "web" / "src" / "map" / "MapView.tsx").read_text(
        encoding="utf-8"
    )
    label_source = (ROOT / "web" / "src" / "map" / "layers.ts").read_text(
        encoding="utf-8"
    )

    assert "Water / wetness signal" in label_source
    assert "Bare or sparse ground" in label_source
    assert "Mixed / uncertain" in label_source
    assert "Moisture-related change" in label_source
    assert "Monitoring signal" in map_source
    assert "Before: " in map_source
    assert "After: " in map_source
    assert "labelForClass(key)" in app_source


def test_webgis_exposes_change_observation_dates():
    app_source = (ROOT / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    api_source = (ROOT / "web" / "src" / "api.ts").read_text(encoding="utf-8")
    map_source = (ROOT / "web" / "src" / "map" / "MapView.tsx").read_text(
        encoding="utf-8"
    )
    label_source = (ROOT / "web" / "src" / "map" / "layers.ts").read_text(
        encoding="utf-8"
    )

    assert "changeWindowLabel" in label_source
    assert "Observed: " in map_source
    assert "Optical dates" in map_source
    assert "Radar support" in map_source
    assert "radar_before_date" in api_source
    assert "radar_after_date" in api_source
    assert "change.before_date" in app_source
    assert "change.after_date" in app_source
