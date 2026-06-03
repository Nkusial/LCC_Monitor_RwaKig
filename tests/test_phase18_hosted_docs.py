from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pages_workflow_and_landing_page_exist() -> None:
    assert (PROJECT_ROOT / "docs" / "index.html").exists()
    assert (PROJECT_ROOT / "docs" / "index.md").exists()
    assert (PROJECT_ROOT / "docs" / "app" / "index.html").exists()
    assert (PROJECT_ROOT / "docs" / "HOSTED_DOCS.md").exists()
    assert not (PROJECT_ROOT / ".github" / "workflows" / "pages.yml").exists()


def test_hosted_docs_reference_expected_pages_url() -> None:
    hosted_docs = (PROJECT_ROOT / "docs" / "HOSTED_DOCS.md").read_text(
        encoding="utf-8"
    )
    index = (PROJECT_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://nkusial.github.io/LCC_Monitor_RwaKig/" in hosted_docs
    assert "docs/index.md" in hosted_docs
    assert "docs/app/index.html" in hosted_docs
    assert "Deploy from a branch" in hosted_docs
    assert "/docs" in hosted_docs
    assert "WebGIS overview" in index
    assert "[Hosted docs](docs/HOSTED_DOCS.md)" in readme


def test_hosted_webgis_demo_assets_exist() -> None:
    app_dir = PROJECT_ROOT / "docs" / "app"
    demo_dir = app_dir / "demo"
    app_index = (app_dir / "index.html").read_text(encoding="utf-8")
    root_index = (PROJECT_ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "assets/" in app_index
    assert "url=./app/" in root_index
    assert (demo_dir / "summary.json").exists()
    assert (demo_dir / "aoi.geojson").exists()
    assert (demo_dir / "changes.geojson").exists()
    assert (demo_dir / "changes.geojson").stat().st_size > 1_000_000

