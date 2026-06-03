"""Optimize oversized public demo raster PNGs.

The WebGIS uses XYZ tiles for map alignment. The larger files in
``web/public/demo/rasters`` and ``docs/app/demo/rasters`` are fallback/demo PNGs,
so they can be palette-optimized without changing the geospatial tile stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "outputs" / "public_demo_manifest.json"
OUTPUT_PATH = ROOT / "data" / "outputs" / "public_demo_optimization_report.json"
WEB_SUMMARY_PATH = ROOT / "web" / "public" / "demo" / "public_demo_optimization_report.json"
DOCS_SUMMARY_PATH = ROOT / "docs" / "app" / "demo" / "public_demo_optimization_report.json"


def repo_path(path_text: str) -> Path:
    return ROOT / path_text


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Run pipelines/41_package_public_demo_manifest.py before optimizing demo rasters."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def oversized_pngs(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for root in manifest.get("roots", []):
        for item in root.get("oversized_assets", []):
            path = repo_path(item["path"])
            if path.suffix.lower() == ".png" and path.exists():
                paths.append(path)
    return sorted(set(paths))


def optimize_png(path: Path, colors: int = 256) -> dict[str, Any]:
    before = path.stat().st_size
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        # Palette quantization is a good fit for WebGIS legend-colored rasters
        # and analytic overlays. It reduces public repo weight while preserving
        # the layer footprint and tile-based georeferencing.
        optimized = rgba.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
        optimized.save(path, optimize=True, compress_level=9)
    after = path.stat().st_size
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "before_mb": round(before / 1024 / 1024, 3),
        "after_mb": round(after / 1024 / 1024, 3),
        "saved_mb": round((before - after) / 1024 / 1024, 3),
        "reduction_fraction": round((before - after) / before, 6) if before else 0,
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    before_total = sum(record["before_mb"] for record in records)
    after_total = sum(record["after_mb"] for record in records)
    return {
        "phase": "phase_57_public_demo_raster_optimization",
        "purpose": "Reduce oversized fallback/demo PNG assets while keeping raw rasters and model artifacts private.",
        "optimized_file_count": len(records),
        "before_total_mb": round(before_total, 3),
        "after_total_mb": round(after_total, 3),
        "saved_total_mb": round(before_total - after_total, 3),
        "method": "RGBA to optimized 256-color PNG palette for oversized public demo raster PNGs.",
        "geospatial_safety_note": (
            "Only fallback/demo PNG files are optimized. WebGIS map alignment remains based on "
            "EPSG:3857 XYZ raster tiles referenced by raster_layers.json."
        ),
        "records": records,
    }


def write_report(report: dict[str, Any]) -> None:
    for path in [OUTPUT_PATH, WEB_SUMMARY_PATH, DOCS_SUMMARY_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    manifest = load_manifest()
    targets = oversized_pngs(manifest)
    records = [optimize_png(path) for path in targets]
    report = build_report(records)
    write_report(report)
    print(f"Optimized {len(records)} oversized public demo PNG files.")
    print(f"Saved {report['saved_total_mb']} MB.")
    print(f"Optimization report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
