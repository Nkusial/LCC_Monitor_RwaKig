"""Run controlled AOI-windowed preprocessing for selected training scene pairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


PREPROCESS_SPEC = importlib.util.spec_from_file_location(
    "phase4_preprocess", Path("pipelines/03_preprocess.py")
)
phase4_preprocess = importlib.util.module_from_spec(PREPROCESS_SPEC)
assert PREPROCESS_SPEC.loader is not None
PREPROCESS_SPEC.loader.exec_module(phase4_preprocess)


def load_training_pairs(catalog_path: Path) -> list[dict[str, Any]]:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Training scene catalog not found: {catalog_path}. "
            "Run pipelines/13_discover_training_scenes.py first."
        )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return catalog.get("pairs", [])


def filter_pairs(
    pairs: list[dict[str, Any]],
    split: str | None = None,
    year: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = pairs
    if split:
        selected = [pair for pair in selected if pair.get("split") == split]
    if year:
        selected = [pair for pair in selected if int(pair.get("year")) == year]
    if limit is not None:
        selected = selected[:limit]
    return selected


def expected_training_outputs(pair: dict[str, Any]) -> list[str]:
    tag = pair["output_tag"]
    return [
        f"data/processed/{tag}_s2_green.tif",
        f"data/processed/{tag}_s2_red.tif",
        f"data/processed/{tag}_s2_nir.tif",
        f"data/processed/{tag}_s2_swir1.tif",
        f"data/processed/{tag}_s2_ndvi.tif",
        f"data/processed/{tag}_s2_ndwi.tif",
        f"data/processed/{tag}_s2_ndbi.tif",
        f"data/processed/{tag}_s1_vv.tif",
        f"data/processed/{tag}_s1_vh.tif",
        f"data/processed/{tag}_s1_vv_vh_ratio.tif",
    ]


def preprocess_training_pairs(
    config: dict[str, Any],
    mode: str = "metadata",
    split: str | None = None,
    year: int | None = None,
    limit: int | None = None,
) -> Path:
    """Run metadata or raster preprocessing for selected training pairs."""
    ensure_output_dirs(config)
    catalog_dir = Path(config["paths"]["catalog"])
    catalog_path = catalog_dir / "training_scene_pairs_2023_2026.json"
    pairs = filter_pairs(load_training_pairs(catalog_path), split=split, year=year, limit=limit)
    if not pairs:
        raise RuntimeError("No training pairs matched the requested filters.")

    processed = []
    for pair in pairs:
        manifest_name = f"{pair['output_tag']}_manifest.json"
        print(f"Preprocessing {pair['output_tag']} in {mode} mode...", flush=True)
        manifest_path = phase4_preprocess.preprocess_pair(
            config,
            pair=pair,
            output_tag=pair["output_tag"],
            metadata_name=manifest_name,
            catalog_name="training_scenes_2023_2026.geojson",
            metadata_only=(mode == "metadata"),
            mode=mode,
        )
        processed.append(
            {
                "output_tag": pair["output_tag"],
                "year": pair["year"],
                "split": pair["split"],
                "mode": mode,
                "manifest": str(manifest_path).replace("\\", "/"),
                "expected_outputs": expected_training_outputs(pair),
            }
        )

    summary = {
        "phase": "phase_30_multi_year_training_preprocessing",
        "mode": mode,
        "catalog": str(catalog_path).replace("\\", "/"),
        "requested_split": split,
        "requested_year": year,
        "requested_limit": limit,
        "processed_pair_count": len(processed),
        "pairs": processed,
        "notes": [
            "Metadata mode validates pair wiring without reading remote rasters.",
            "Use --mode optical, --mode radar, or --mode all with --limit before scaling to all pairs.",
            "Heavy raster outputs remain local under data/processed and are not committed to Git.",
        ],
    }
    summary_path = catalog_dir / "training_preprocessing_summary_2023_2026.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess selected multi-year training scene pairs.")
    parser.add_argument("--mode", choices=["metadata", "optical", "radar", "all"], default="metadata")
    parser.add_argument("--split", choices=["train", "validation", "test"], default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary_path = preprocess_training_pairs(
        load_config(),
        mode=args.mode,
        split=args.split,
        year=args.year,
        limit=args.limit,
    )
    print(f"Training preprocessing summary: {summary_path}")


if __name__ == "__main__":
    main()
