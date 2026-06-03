from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase16_is_recorded_in_phase_tracker() -> None:
    phases = (PROJECT_ROOT / "docs" / "PHASES.md").read_text(encoding="utf-8")

    assert "16. Local milestone packaging" in phases
    assert "v0.1.0" in phases
    assert "17. CI/CD reflection" in phases
    assert "18. Hosted WebGIS readiness" in phases
    assert "Phase 19" in phases
