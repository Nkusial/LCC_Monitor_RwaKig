from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase14_publication_docs_exist() -> None:
    required_docs = [
        PROJECT_ROOT / "CHANGELOG.md",
        PROJECT_ROOT / "docs" / "DEMO.md",
        PROJECT_ROOT / "docs" / "ISSUE_BACKLOG.md",
    ]

    for path in required_docs:
        assert path.exists(), f"Missing publication artifact: {path.name}"


def test_publication_docs_cover_demo_and_future_work() -> None:
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    demo = (PROJECT_ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
    backlog = (PROJECT_ROOT / "docs" / "ISSUE_BACKLOG.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "0.1.0 - Local Portfolio Milestone" in changelog
    assert "changed/no-change quantities" in changelog
    assert "Demo Script" in demo
    assert "Train Baseline Classifier" in backlog
    assert "Demo script" in readme
    assert "Development backlog" in readme
    assert "DEMO_RECORDING.md" not in readme
