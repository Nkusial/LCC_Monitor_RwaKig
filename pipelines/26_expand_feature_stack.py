"""Create additional Sentinel-derived features for v2 segmentation training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "data" / "outputs" / "feature_expansion_report.json"


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile.copy()


def read_aligned(path: Path, reference: dict[str, Any]) -> np.ndarray:
    """Read any source raster onto the reference grid before index math."""
    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")
    with rasterio.open(path) as src:
        if (
            src.width == reference["width"]
            and src.height == reference["height"]
            and src.transform == reference["transform"]
            and src.crs == reference["crs"]
        ):
            return src.read(1).astype("float32")
        with WarpedVRT(
            src,
            crs=reference["crs"],
            transform=reference["transform"],
            width=reference["width"],
            height=reference["height"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            nodata=np.nan,
        ) as vrt:
            return vrt.read(1).astype("float32")


def write_raster(path: Path, array: np.ndarray, profile: dict[str, Any]) -> None:
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="float32", nodata=-9999, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype("float32"), 1)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, -9999, dtype="float32")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def derive_features(tag: str) -> dict[str, str]:
    red, profile = read_raster(PROCESSED / f"{tag}_s2_red.tif")
    green = read_aligned(PROCESSED / f"{tag}_s2_green.tif", profile)
    nir = read_aligned(PROCESSED / f"{tag}_s2_nir.tif", profile)
    swir1 = read_aligned(PROCESSED / f"{tag}_s2_swir1.tif", profile)
    vv = read_aligned(PROCESSED / f"{tag}_s1_vv.tif", profile)
    vh = read_aligned(PROCESSED / f"{tag}_s1_vh.tif", profile)

    # EVI2 is used because the current processed stack does not include blue.
    evi2 = safe_divide(2.5 * (nir - red), nir + (2.4 * red) + 1.0)
    savi = safe_divide(1.5 * (nir - red), nir + red + 0.5)
    mndwi = safe_divide(green - swir1, green + swir1)
    bsi = safe_divide((swir1 + red) - (nir + green), (swir1 + red) + (nir + green))
    s1_ratio = safe_divide(vv, vh)
    s1_log_ratio = np.full(s1_ratio.shape, -9999, dtype="float32")
    valid_ratio = np.isfinite(s1_ratio) & (s1_ratio > 0)
    s1_log_ratio[valid_ratio] = np.log1p(s1_ratio[valid_ratio]).astype("float32")

    outputs = {
        "s2_evi2": PROCESSED / f"{tag}_s2_evi2.tif",
        "s2_savi": PROCESSED / f"{tag}_s2_savi.tif",
        "s2_mndwi": PROCESSED / f"{tag}_s2_mndwi.tif",
        "s2_bsi": PROCESSED / f"{tag}_s2_bsi.tif",
        "s1_log_ratio": PROCESSED / f"{tag}_s1_log_ratio.tif",
    }
    arrays = {
        "s2_evi2": evi2,
        "s2_savi": savi,
        "s2_mndwi": mndwi,
        "s2_bsi": bsi,
        "s1_log_ratio": s1_log_ratio,
    }
    for name, path in outputs.items():
        write_raster(path, arrays[name], profile)
    return {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in outputs.items()}


def available_tags(prefix: str | None) -> list[str]:
    tags = sorted(path.name.removesuffix("_s2_red.tif") for path in PROCESSED.glob("*_s2_red.tif"))
    return [tag for tag in tags if prefix is None or tag.startswith(prefix)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive extra Sentinel feature rasters for v2 training.")
    parser.add_argument("--prefix", default="training_", help="Tag prefix to process; use empty string for all.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prefix = args.prefix or None
    tags = available_tags(prefix)
    if args.limit:
        tags = tags[: args.limit]
    records = [{"tag": tag, "outputs": derive_features(tag)} for tag in tags]

    report = {
        "phase": "phase_44_feature_expansion",
        "feature_purpose": {
            "s2_evi2": "vegetation vigor without requiring a blue band",
            "s2_savi": "vegetation with soil-background damping",
            "s2_mndwi": "water and wet-surface separation",
            "s2_bsi": "bare soil versus built/sparse surfaces",
            "s1_log_ratio": "radar structure/moisture contrast",
        },
        "record_count": len(records),
        "records": records,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Feature expansion report: {REPORT}")
    print(json.dumps({"record_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
