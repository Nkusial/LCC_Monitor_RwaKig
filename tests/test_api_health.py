from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_check_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_api_landing_payload() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "/changes/geojson" in payload["endpoints"]


def test_change_summary_exposes_changed_and_no_change_quantities() -> None:
    response = TestClient(app).get("/changes/summary")

    assert response.status_code == 200
    payload = response.json()
    assert "changed_area_m2" in payload
    assert "no_change_area_m2" in payload
    assert "changed_percent" in payload
    assert "no_change_percent" in payload
    assert "by_monitored_land_cover_area_m2" in payload
    assert payload["changed_area_m2"] + payload["no_change_area_m2"] == payload["aoi_area_m2"]
