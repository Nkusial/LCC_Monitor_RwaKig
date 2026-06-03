from pathlib import Path


def test_phase11_portfolio_docs_exist() -> None:
    for path in [
        Path("docs/PORTFOLIO_REVIEW.md"),
        Path("docs/SCREENSHOT_GUIDE.md"),
        Path("docs/ML_ROADMAP.md"),
        Path("docs/assets/.gitkeep"),
    ]:
        assert path.exists(), f"Missing Phase 11 portfolio asset: {path}"


def test_ml_roadmap_names_labeling_and_mlops() -> None:
    text = Path("docs/ML_ROADMAP.md").read_text(encoding="utf-8")

    assert "Labeling" in text
    assert "MLOps" in text
    assert "PostGIS" in text
