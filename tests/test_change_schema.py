from backend.app.models import ChangeSummary
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path("pipelines/04_detect_change.py")
SPEC = importlib.util.spec_from_file_location("detect_change", SCRIPT_PATH)
detect_change = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(detect_change)


def test_change_summary_accepts_valid_confidence() -> None:
    change = ChangeSummary(
        id="demo-change",
        change_type="vegetation_loss",
        confidence=0.75,
        area_m2=1500,
    )

    assert change.confidence == 0.75


def test_classify_change_detects_expected_classes() -> None:
    thresholds = {
        "ndvi_loss": -0.18,
        "ndbi_gain": 0.12,
        "ndwi_abs": 0.18,
        "radar_ratio_abs": 0.15,
    }

    classes = detect_change.classify_change(
        delta_ndvi=np.array([[-0.2, -0.2, 0.0, 0.0]], dtype="float32"),
        delta_ndbi=np.array([[0.0, 0.2, 0.0, 0.0]], dtype="float32"),
        delta_ndwi=np.array([[0.0, 0.0, 0.3, 0.0]], dtype="float32"),
        delta_radar_ratio=np.array([[0.0, 0.0, 0.0, 0.2]], dtype="float32"),
        thresholds=thresholds,
    )

    assert classes.tolist() == [[1, 2, 3, 4]]


def test_finite_delta_masks_invalid_values() -> None:
    before = np.array([[1.0, np.nan]], dtype="float32")
    after = np.array([[1.5, 2.0]], dtype="float32")

    delta = detect_change.finite_delta(after, before)

    assert np.isclose(delta[0, 0], 0.5)
    assert np.isnan(delta[0, 1])


def test_infer_landcover_state_from_indices() -> None:
    assert detect_change.infer_landcover_state(ndvi=0.65, ndbi=0.05, ndwi=-0.1) == "vegetation"
    assert detect_change.infer_landcover_state(ndvi=0.1, ndbi=0.2, ndwi=-0.1) == "built_up"
    assert detect_change.infer_landcover_state(ndvi=0.1, ndbi=-0.1, ndwi=0.25) == "water_moisture"
    assert detect_change.infer_landcover_state(ndvi=0.1, ndbi=0.0, ndwi=-0.1) == "bare_sparse"


def test_change_magnitude_uses_feature_deltas() -> None:
    magnitude = detect_change.change_magnitude(
        delta_ndvi=-0.3,
        delta_ndbi=0.2,
        delta_ndwi=0.1,
        delta_radar_ratio=0.4,
    )

    assert np.isclose(magnitude, np.sqrt((0.09 + 0.04 + 0.01 + 0.16) / 4))


def test_monitored_land_cover_aligns_to_change_type() -> None:
    assert (
        detect_change.monitored_land_cover(
            change_type="vegetation_loss",
            before_state="vegetation",
            after_state="mixed",
        )
        == "vegetation"
    )
    assert (
        detect_change.monitored_land_cover(
            change_type="built_up_gain",
            before_state="bare_sparse",
            after_state="built_up",
        )
        == "built_up"
    )
    assert (
        detect_change.monitored_land_cover(
            change_type="moisture_change",
            before_state="mixed",
            after_state="water_moisture",
        )
        == "water_moisture"
    )


def test_transition_label_names_before_and_after_states() -> None:
    assert detect_change.transition_label("vegetation", "mixed") == "vegetation_to_mixed"
