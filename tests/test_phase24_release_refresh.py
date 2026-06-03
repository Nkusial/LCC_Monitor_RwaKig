from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_phase24_release_materials_reference_v020():
    changelog = read("CHANGELOG.md")
    release_notes = read("docs/GITHUB_RELEASE_NOTES.md")
    publishing = read("docs/PUBLISHING.md")
    phases = read("docs/PHASES.md")

    assert "0.2.0 - Validation-Aware Hosted WebGIS Milestone" in changelog
    assert "v0.2.0 - Validation-Aware Hosted WebGIS Milestone" in release_notes
    assert "git tag -a v0.2.0" in publishing
    assert "Phase 24 release refresh and demo packaging" in phases


def test_phase24_demo_packaging_highlights_validation_story():
    release_checklist = read("docs/RELEASE_CHECKLIST.md")
    demo = read("docs/DEMO.md")
    hosted_index = read("docs/index.md")

    assert "Validation: Unsupervised clusters" in release_checklist
    assert "0.4459" in release_checklist
    assert "feature-space coherence, not field accuracy" in demo
    assert "Unsupervised validation" in hosted_index
