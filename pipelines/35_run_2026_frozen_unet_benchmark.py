"""Run frozen U-Net inference on the held-out 2026 test stack.

This benchmark is proxy evaluation only. It compares the frozen model output
against aligned weak-source agreement rasters, but it does not train, tune,
rebalance, recalibrate, or generate new pseudo-labels from 2026 outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_CONFIG = ROOT / "configs" / "model_unet.yaml"
DEFAULT_TEST_SUMMARY = ROOT / "data" / "catalog" / "test_preprocessing_summary_2026.json"
DEFAULT_NO_LEAKAGE = ROOT / "data" / "outputs" / "no_leakage_audit_report.json"
DEFAULT_EXTERNAL_SUMMARY = ROOT / "data" / "interim" / "external_sources" / "external_weak_sources_summary_test_2026.json"
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "unet_full_aoi" / "unet_full_aoi_manifest_test_2026.json"
DEFAULT_REPORT = ROOT / "data" / "outputs" / "unet_2026_proxy_benchmark_report.json"

INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "full_aoi_inference", ROOT / "pipelines" / "20_full_aoi_unet_inference.py"
)
full_aoi_inference = importlib.util.module_from_spec(INFERENCE_SPEC)
assert INFERENCE_SPEC.loader is not None
INFERENCE_SPEC.loader.exec_module(full_aoi_inference)

VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "full_aoi_validation", ROOT / "pipelines" / "21_validate_unet_full_aoi.py"
)
full_aoi_validation = importlib.util.module_from_spec(VALIDATION_SPEC)
assert VALIDATION_SPEC.loader is not None
VALIDATION_SPEC.loader.exec_module(full_aoi_validation)

try:
    from common import load_config
except ModuleNotFoundError:
    from pipelines.common import load_config


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required model config is missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_no_leakage(no_leakage_path: Path) -> dict[str, Any]:
    report = read_json(no_leakage_path)
    if report.get("status") != "pass":
        raise RuntimeError("No-leakage audit must pass before 2026 frozen benchmark inference.")
    return report


def build_test_pair(test_summary: dict[str, Any]) -> dict[str, Any]:
    tag = test_summary["output_tag"]
    candidate = test_summary["selected_candidate"]
    return {
        "output_tag": tag,
        "year": 2026,
        "split": "test",
        "optical": {
            "date": candidate["date"],
            "cloud_cover_percent": candidate["average_cloud_cover_percent"],
            "max_tile_cloud_cover_percent": candidate["max_tile_cloud_cover_percent"],
        },
        "radar": {
            "date": candidate["sentinel1_match"]["date"],
            "time_gap_hours": candidate["sentinel1_match"]["time_gap_hours"],
        },
    }


def load_frozen_model(settings: dict[str, Any]) -> tuple[Any, Any, Any, Path]:
    try:
        import torch
        from ml.unet import build_segmentation_model
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError("PyTorch is required for frozen U-Net benchmark inference.") from exc

    checkpoint_path = ROOT / settings["outputs"]["checkpoint_path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen U-Net checkpoint not found: {checkpoint_path}")
    configured_device = settings["training"]["device"]
    device = torch.device("cuda" if configured_device == "auto" and torch.cuda.is_available() else configured_device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_segmentation_model(
        architecture=settings["model"].get("architecture", "lightweight_unet"),
        input_channels=int(settings["model"]["input_channels"]),
        output_classes=int(settings["model"]["output_classes"]),
        base_channels=int(settings["model"]["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch, device, checkpoint_path


def run_benchmark(
    model_config_path: Path,
    test_summary_path: Path,
    no_leakage_path: Path,
    external_summary_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    no_leakage = ensure_no_leakage(no_leakage_path)
    config = load_config()
    settings = load_yaml(model_config_path)
    test_summary = read_json(test_summary_path)
    if test_summary.get("status") != "processed":
        raise RuntimeError("2026 test stack must be fully processed before frozen benchmark inference.")

    pair = build_test_pair(test_summary)
    model, torch, device, checkpoint_path = load_frozen_model(settings)
    record = full_aoi_inference.run_pair_inference(
        pair,
        config=config,
        settings=settings,
        model=model,
        torch=torch,
        device=device,
    )
    manifest = {
        "phase": "phase_51_frozen_2026_unet_inference",
        "purpose": "Frozen-model inference on the held-out 2026 test-only stack.",
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "no_leakage_audit_status": no_leakage.get("status"),
        "records": [record],
        "non_cheating_protocol": {
            "allowed": "Report proxy agreement, confidence, entropy, and review zones.",
            "forbidden": [
                "Do not retrain from this result.",
                "Do not tune thresholds, architecture, class weights, or pseudo-label filters from this result.",
                "Do not claim field accuracy from weak-source agreement.",
            ],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = full_aoi_validation.build_report(
        manifest_path=manifest_path,
        external_summary_path=external_summary_path,
        output_path=report_path,
        entropy_threshold=0.65,
        confidence_threshold=0.60,
    )
    report["phase"] = "phase_51_frozen_2026_proxy_benchmark"
    report["benchmark_caveat"] = (
        "This is a relaxed 2026 proxy benchmark because no complete same-date Sentinel-2 pair "
        "met the preferred <10% cloud policy."
    )
    report["non_cheating_protocol"] = manifest["non_cheating_protocol"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen U-Net inference on the 2026 held-out test stack.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--test-summary", type=Path, default=DEFAULT_TEST_SUMMARY)
    parser.add_argument("--no-leakage", type=Path, default=DEFAULT_NO_LEAKAGE)
    parser.add_argument("--external-summary", type=Path, default=DEFAULT_EXTERNAL_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = run_benchmark(
        model_config_path=args.model_config,
        test_summary_path=args.test_summary,
        no_leakage_path=args.no_leakage,
        external_summary_path=args.external_summary,
        manifest_path=args.manifest,
        report_path=args.output,
    )
    print(f"2026 frozen U-Net proxy benchmark report: {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
