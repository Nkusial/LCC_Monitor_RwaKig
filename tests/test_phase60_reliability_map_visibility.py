from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_reliability_color_mode_is_exposed_in_sidebar() -> None:
    app = read("web/src/App.tsx")

    assert "value=\"reliability\"" in app
    assert "Reliability" in app
    assert "reliability-legend" in app
    assert "reliability-symbol" in app
    assert "Likely reliable change" in app


def test_reliability_properties_drive_map_style_and_popup() -> None:
    map_view = read("web/src/map/MapView.tsx")
    layers = read("web/src/map/layers.ts")

    assert "reliabilityColorExpression" in map_view
    assert "['get', 'reliability']" in map_view
    assert "labelForReliability" in map_view
    assert "circle-opacity', reliabilityMode ? 0 : 0.92" in map_view
    assert "Review note" in map_view
    assert "reliabilityColors" in layers
    assert "Needs review" in layers


def test_reliability_symbols_do_not_reuse_land_cover_green() -> None:
    css = read("web/src/App.css")
    layers = read("web/src/map/layers.ts")

    assert "reliability-symbol-high" in css
    assert "#005f73" in layers
    assert "high: '#15803d'" not in layers


def test_long_reliability_metric_wraps_without_overlap() -> None:
    app = read("web/src/App.tsx")
    css = read("web/src/App.css")

    assert "metric metric-stacked" in app
    assert "Top likely change type" in app
    assert ".metric-stacked" in css
    assert "overflow-wrap: anywhere" in css
