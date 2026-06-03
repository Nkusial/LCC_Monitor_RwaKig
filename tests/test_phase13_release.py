from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_checklist_exists_and_mentions_validation() -> None:
    checklist = PROJECT_ROOT / "docs" / "RELEASE_CHECKLIST.md"

    assert checklist.exists()
    text = checklist.read_text(encoding="utf-8")
    assert "pytest -q" in text
    assert "python pipelines/07_validate_outputs.py" in text
    assert "npm run build" in text
    assert "changed and no-change quantities" in text


def test_readme_mentions_current_demo_statistics() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Current Demo Statistics" in readme
    assert "Changed area" in readme
    assert "No-change area" in readme
    assert "Release checklist" in readme
