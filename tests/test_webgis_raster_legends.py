from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_webgis_has_dynamic_raster_legend_for_selected_layer() -> None:
    app_source = read("web/src/App.tsx")
    layer_source = read("web/src/map/layers.ts")
    css_source = read("web/src/App.css")

    assert "rasterLegendForLayer" in layer_source
    assert "Selected raster layer legend" in app_source
    assert "raster-gradient" in css_source
    assert "raster-legend-items" in css_source


def test_raster_legend_explains_review_and_unet_colors() -> None:
    layer_source = read("web/src/map/layers.ts")

    assert "Low model confidence" in layer_source
    assert "High entropy / ambiguity" in layer_source
    assert "Weak-source disagreement" in layer_source
    assert "Built-up / impervious" in layer_source
    assert "Vegetation" in layer_source
