"""Small orchestrator for running selected local pipeline stages in order."""

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = {
    # The default reproducible demo starts at detect because Phase 4 rasters are
    # expensive to rebuild and are treated as prepared local inputs.
    "search": ["pipelines/01_search_scenes.py", "--limit", "10"],
    "stack": ["pipelines/03_build_monitoring_stack.py", "--mode", "all"],
    "detect": ["pipelines/04_detect_change.py"],
    "score": ["pipelines/05_score_confidence.py"],
    "publish": ["pipelines/06_publish_to_postgis.py"],
    "weak-labels": ["pipelines/10_generate_weak_labels.py"],
    "external-sources": ["pipelines/11_ingest_external_weak_sources.py"],
    "patches": ["pipelines/12_generate_patch_dataset.py"],
    "training-scenes": ["pipelines/13_discover_training_scenes.py"],
    "training-stack-manifest": ["pipelines/14_prepare_training_stack_manifest.py"],
    "unet-preflight": ["pipelines/15_train_unet_baseline.py"],
    "training-preprocess": ["pipelines/16_preprocess_training_pairs.py"],
    "training-patches": ["pipelines/17_generate_training_patch_dataset.py"],
    "unet-inference": ["pipelines/18_run_unet_inference.py"],
    "unet-mosaics": ["pipelines/19_mosaic_unet_predictions.py"],
    "unet-full-aoi": ["pipelines/20_full_aoi_unet_inference.py"],
    "unet-reliability": ["pipelines/21_validate_unet_full_aoi.py"],
    "dynamic-world": ["pipelines/22_export_dynamic_world.py"],
    "temporal-consensus": ["pipelines/23_temporal_unet_consensus.py"],
    "retraining-review": ["pipelines/24_prepare_retraining_review.py"],
    "test-readiness": ["pipelines/25_report_test_set_readiness.py"],
    "feature-expansion": ["pipelines/26_expand_feature_stack.py"],
    "refined-pseudo-labels": ["pipelines/27_generate_refined_pseudo_labels.py"],
    "weak-calibration": ["pipelines/28_calibrate_weak_confidence.py"],
    "topography": ["pipelines/29_ingest_topography.py"],
    "balance-patches": ["pipelines/30_balance_training_patch_manifest.py"],
    "benchmark-2026": ["pipelines/31_discover_2026_test_benchmark.py"],
    "no-leakage": ["pipelines/32_validate_no_leakage.py"],
    "benchmark-package": ["pipelines/33_prepare_2026_benchmark_package.py"],
    "test-2026-preprocess": ["pipelines/34_preprocess_2026_test_pair.py"],
    "test-2026-unet": ["pipelines/35_run_2026_frozen_unet_benchmark.py"],
    "uncertainty-controls": ["pipelines/36_report_uncertainty_controls.py"],
    "validate": ["pipelines/07_validate_outputs.py"],
}


def run_stage(name: str) -> None:
    """Run one stage with the current Python interpreter and fail fast."""
    command = [sys.executable, *STAGES[name]]
    print(f"\n==> {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local land-cover monitoring pipeline.")
    parser.add_argument(
        "--from-stage",
        choices=list(STAGES),
        default="detect",
        help="First stage to run. Default starts from existing Phase 4 rasters.",
    )
    parser.add_argument(
        "--to-stage",
        choices=list(STAGES),
        default="validate",
        help="Last stage to run.",
    )
    args = parser.parse_args()

    names = list(STAGES)
    # Keep the CLI simple while preventing accidental backwards stage ranges.
    start = names.index(args.from_stage)
    end = names.index(args.to_stage)
    if start > end:
        raise SystemExit("--from-stage must come before --to-stage")

    for stage in names[start : end + 1]:
        run_stage(stage)


if __name__ == "__main__":
    main()
