from pathlib import Path


def test_portfolio_docs_exist() -> None:
    for path in [
        Path("docs/ARCHITECTURE.md"),
        Path("docs/API_EXAMPLES.md"),
        Path("docs/DEMO.md"),
        Path("docs/OPERATIONS.md"),
        Path("docs/PHASES.md"),
    ]:
        assert path.exists(), f"Missing portfolio doc: {path}"


def test_architecture_doc_mentions_core_stack() -> None:
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "Sentinel-2" in text
    assert "PostGIS" in text
    assert "FastAPI" in text
    assert "MapLibre" in text
