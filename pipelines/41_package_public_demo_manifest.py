"""Create a lightweight public-demo asset manifest.

The manifest helps keep the future public repository clean: small WebGIS demo
derivatives can be tracked, while raw Sentinel rasters, local model artifacts,
and credentials stay private.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "improvement_track.yaml"
OUTPUT_PATH = ROOT / "data" / "outputs" / "public_demo_manifest.json"
WEB_SUMMARY_PATH = ROOT / "web" / "public" / "demo" / "public_demo_manifest.json"
DOCS_SUMMARY_PATH = ROOT / "docs" / "app" / "demo" / "public_demo_manifest.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def scan_root(root: Path, max_asset_bytes: int, allowed_suffixes: set[str]) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    total_bytes = sum(path.stat().st_size for path in files)
    by_suffix: dict[str, int] = defaultdict(int)
    oversized = []
    disallowed = []

    for path in files:
        suffix = path.suffix.lower() or "<none>"
        by_suffix[suffix] += 1
        size = path.stat().st_size
        if size > max_asset_bytes:
            oversized.append({"path": relative(path), "size_mb": round(size / 1024 / 1024, 3)})
        if suffix not in allowed_suffixes:
            disallowed.append({"path": relative(path), "suffix": suffix})

    return {
        "root": relative(root),
        "file_count": len(files),
        "total_mb": round(total_bytes / 1024 / 1024, 3),
        "suffix_counts": dict(sorted(by_suffix.items())),
        "oversized_assets": oversized,
        "disallowed_file_types": disallowed,
    }


def build_manifest() -> dict[str, Any]:
    config = load_config()["public_demo_packaging"]
    max_asset_mb = float(config["max_public_asset_mb"])
    max_asset_bytes = int(max_asset_mb * 1024 * 1024)
    allowed_suffixes = {suffix.lower() for suffix in config["allowed_public_file_types"]}

    roots = [scan_root(ROOT / root, max_asset_bytes, allowed_suffixes) for root in config["lightweight_public_roots"]]
    warning_count = sum(len(root["oversized_assets"]) + len(root["disallowed_file_types"]) for root in roots)

    return {
        "phase": "phase_56_public_demo_packaging",
        "purpose": "Track only lightweight public WebGIS/demo derivatives and keep raw data plus model artifacts private.",
        "status": "warning" if warning_count else "passed",
        "max_public_asset_mb": max_asset_mb,
        "roots": roots,
        "private_roots": config["private_roots"],
        "policy": config["note"],
        "warning_count": warning_count,
    }


def write_manifest(manifest: dict[str, Any]) -> None:
    for path in [OUTPUT_PATH, WEB_SUMMARY_PATH, DOCS_SUMMARY_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    manifest = build_manifest()
    write_manifest(manifest)
    print(f"Public demo manifest: {OUTPUT_PATH}")
    print(f"Status: {manifest['status']} ({manifest['warning_count']} warnings)")


if __name__ == "__main__":
    main()
