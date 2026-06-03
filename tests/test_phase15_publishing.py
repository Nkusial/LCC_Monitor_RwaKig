from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase15_publishing_docs_exist() -> None:
    required = [
        PROJECT_ROOT / "docs" / "PUBLISHING.md",
        PROJECT_ROOT / "docs" / "GITHUB_RELEASE_NOTES.md",
        PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.md",
        PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.md",
    ]

    for path in required:
        assert path.exists(), f"Missing publishing artifact: {path}"


def test_release_notes_include_demo_metrics_and_topics() -> None:
    notes = (PROJECT_ROOT / "docs" / "GITHUB_RELEASE_NOTES.md").read_text(
        encoding="utf-8"
    )
    publishing = (PROJECT_ROOT / "docs" / "PUBLISHING.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Published changes: 2,076" in notes
    assert "Changed area:" in notes
    assert "geoai" in notes
    assert "git remote add origin" in publishing
    assert "GitHub release notes" in readme
