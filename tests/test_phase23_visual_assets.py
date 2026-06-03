from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase23_validation_screenshots_exist():
    assets_dir = ROOT / "docs" / "assets"

    for filename in [
        "08_validation_overlay.png",
        "09_validation_workflow.png",
    ]:
        asset = assets_dir / filename
        assert asset.exists(), f"Missing portfolio screenshot: {filename}"
        assert asset.stat().st_size > 100_000, f"Screenshot looks too small: {filename}"


def test_phase23_docs_reference_validation_screenshots():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    screenshot_guide = (ROOT / "docs" / "SCREENSHOT_GUIDE.md").read_text(encoding="utf-8")
    phases = (ROOT / "docs" / "PHASES.md").read_text(encoding="utf-8")
    package_json = (ROOT / "web" / "package.json").read_text(encoding="utf-8")

    assert "docs/assets/08_validation_overlay.png" in readme
    assert "docs/assets/09_validation_workflow.png" in readme
    assert "npm run screenshots:phase23" in screenshot_guide
    assert "Phase 23 visual asset refresh" in phases
    assert "screenshots:phase23" in package_json
