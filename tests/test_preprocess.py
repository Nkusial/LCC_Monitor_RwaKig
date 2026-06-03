import importlib.util
from pathlib import Path

import numpy as np

from pipelines import common


SCRIPT_PATH = Path("pipelines/03_preprocess.py")
SPEC = importlib.util.spec_from_file_location("preprocess", SCRIPT_PATH)
preprocess = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(preprocess)

STACK_SCRIPT_PATH = Path("pipelines/03_build_monitoring_stack.py")
STACK_SPEC = importlib.util.spec_from_file_location("monitoring_stack", STACK_SCRIPT_PATH)
monitoring_stack = importlib.util.module_from_spec(STACK_SPEC)
assert STACK_SPEC.loader is not None
STACK_SPEC.loader.exec_module(monitoring_stack)


def test_phase4_selected_pair_is_configured() -> None:
    config = common.load_config()
    pair = config["preprocessing"]["selected_pair"]
    stack = config["preprocessing"]["monitoring_stack"]

    assert pair["optical"]["date"] == "2025-02-20"
    assert pair["optical"]["tiles"] == ["35MRT", "35MRU"]
    assert pair["radar"]["scene_id"].endswith("_rtc")
    assert pair["radar"]["polarizations"] == ["VV", "VH"]
    assert stack["max_pairs"] >= 3
    assert stack["required_tiles"] == ["35MRT", "35MRU"]


def test_select_sentinel2_scenes_preserves_config_order() -> None:
    features = [
        {"id": "scene-b", "properties": {"id": "scene-b"}},
        {"id": "scene-a", "properties": {"id": "scene-a"}},
    ]

    selected = preprocess.select_sentinel2_scenes(features, ["scene-a", "scene-b"])

    assert [scene["id"] for scene in selected] == ["scene-a", "scene-b"]


def test_index_helpers_handle_zero_denominators() -> None:
    a = np.array([[2.0, 1.0]], dtype="float32")
    b = np.array([[1.0, -1.0]], dtype="float32")

    nd = preprocess.normalized_difference(a, b)
    ratio = preprocess.safe_ratio(a, b)

    assert np.isclose(nd[0, 0], 1 / 3)
    assert np.isnan(nd[0, 1])
    assert np.isclose(ratio[0, 0], 2.0)


def test_monitoring_stack_datetime_helpers() -> None:
    dt = monitoring_stack.parse_datetime("2025-02-21T16:21:04.343502Z")

    assert dt.year == 2025
    assert dt.tzinfo is not None
