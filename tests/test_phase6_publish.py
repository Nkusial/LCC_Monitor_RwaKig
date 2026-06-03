import importlib.util
from pathlib import Path

import pandas as pd


SCORE_PATH = Path("pipelines/05_score_confidence.py")
SCORE_SPEC = importlib.util.spec_from_file_location("score_confidence", SCORE_PATH)
score_confidence = importlib.util.module_from_spec(SCORE_SPEC)
assert SCORE_SPEC.loader is not None
SCORE_SPEC.loader.exec_module(score_confidence)

PUBLISH_PATH = Path("pipelines/06_publish_to_postgis.py")
PUBLISH_SPEC = importlib.util.spec_from_file_location("publish_to_postgis", PUBLISH_PATH)
publish_to_postgis = importlib.util.module_from_spec(PUBLISH_SPEC)
assert PUBLISH_SPEC.loader is not None
PUBLISH_SPEC.loader.exec_module(publish_to_postgis)


def test_reliability_label_groups_scores() -> None:
    assert score_confidence.reliability_label(0.9) == "high"
    assert score_confidence.reliability_label(0.7) == "medium"
    assert score_confidence.reliability_label(0.4) == "low"


def test_confidence_adjustment_penalizes_radar_mixed_no_transition() -> None:
    row = pd.Series(
        {
            "change_type": "radar_change",
            "monitored_land_cover": "mixed",
            "before_state": "mixed",
            "after_state": "mixed",
        }
    )

    assert score_confidence.confidence_adjustment(row) < 0


def test_to_psycopg_url_removes_sqlalchemy_driver() -> None:
    assert (
        publish_to_postgis.to_psycopg_url("postgresql+psycopg://user:pass@localhost/db")
        == "postgresql://user:pass@localhost/db"
    )
