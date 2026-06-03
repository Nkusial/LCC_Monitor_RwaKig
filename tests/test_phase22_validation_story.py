from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_validation_status_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Validation Status" in readme
    assert "0.4459" in readme
    assert "Accuracy claim:    none" in readme
    assert "Validation: Unsupervised clusters" in readme


def test_portfolio_review_explains_validation_caveat():
    review = (ROOT / "docs" / "PORTFOLIO_REVIEW.md").read_text(encoding="utf-8")

    assert "## Validation Story" in review
    assert "0.2441" in review
    assert "0.4459" in review
    assert "not field accuracy" in review
    assert "planned reference-label workflow" in review


def test_demo_and_screenshot_guides_include_validation_walkthrough():
    demo = (ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
    screenshots = (ROOT / "docs" / "SCREENSHOT_GUIDE.md").read_text(encoding="utf-8")
    phases = (ROOT / "docs" / "PHASES.md").read_text(encoding="utf-8")

    assert "Validation: Unsupervised clusters" in demo
    assert "best current silhouette score is `0.4459`" in demo
    assert "08_validation_overlay.png" in screenshots
    assert "09_validation_workflow.png" in screenshots
    assert "Phase 22 portfolio validation story polish" in phases
