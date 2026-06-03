"""Shared helpers for reproducible local pipeline stages."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/pipeline.yaml") -> dict[str, Any]:
    """Load the YAML pipeline config used by every processing stage."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create configured data directories before a stage writes outputs."""
    for path in config.get("paths", {}).values():
        Path(path).mkdir(parents=True, exist_ok=True)
