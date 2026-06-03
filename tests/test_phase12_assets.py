from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase12_portfolio_screenshots_exist() -> None:
    assets_dir = PROJECT_ROOT / "docs" / "assets"
    screenshots = [
        assets_dir / "01_webgis_overview.png",
        assets_dir / "02_api_docs.png",
        assets_dir / "03_change_summary.png",
    ]

    for screenshot in screenshots:
        assert screenshot.exists(), f"Missing portfolio screenshot: {screenshot.name}"
        assert screenshot.stat().st_size > 10_000, f"Screenshot looks empty: {screenshot.name}"


def test_readme_references_phase12_screenshots() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/assets/01_webgis_overview.png" in readme
    assert "docs/assets/02_api_docs.png" in readme
    assert "docs/assets/03_change_summary.png" in readme
