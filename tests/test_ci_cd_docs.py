from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ci_cd_release_workflow_exists() -> None:
    ci_cd_doc = PROJECT_ROOT / "docs" / "CI_CD.md"

    assert (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()
    assert (PROJECT_ROOT / ".github" / "workflows" / "release.yml").exists()
    assert ci_cd_doc.exists()


def test_ci_cd_docs_explain_release_packaging() -> None:
    ci_cd_doc = (PROJECT_ROOT / "docs" / "CI_CD.md").read_text(encoding="utf-8")
    portfolio = (PROJECT_ROOT / "docs" / "PORTFOLIO_REVIEW.md").read_text(
        encoding="utf-8"
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Continuous Integration" in ci_cd_doc
    assert "Continuous Delivery" in ci_cd_doc
    assert "release packaging" in ci_cd_doc
    assert "tag-based CD release packaging" in portfolio
    assert "[CI/CD](docs/CI_CD.md)" in readme
