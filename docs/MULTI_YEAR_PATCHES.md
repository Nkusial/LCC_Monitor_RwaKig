# Multi-Year Training Patches

Phase 32 expands the weak-label patch dataset from one monitoring stack to the
processed 2023-2025 Sentinel-1/2 training scene catalog.

## Inputs

```text
data/catalog/training_preprocessing_summary_2023_2026.json
data/processed/training_*_s2_ndvi.tif
data/processed/training_*_s2_ndwi.tif
data/processed/training_*_s2_ndbi.tif
data/processed/training_*_s1_vv.tif
data/processed/training_*_s1_vh.tif
data/processed/training_*_s1_vv_vh_ratio.tif
data/interim/external_sources/weak_source_agreement_labels.tif
data/interim/external_sources/weak_source_disagreement_mask.tif
```

## Split Policy

Splits inherit the temporal scene-catalog split:

```text
2023 -> train
2024 -> train
2025 -> validation
2026 -> optional test when usable scenes exist
```

This avoids mixing dates randomly across train and validation. Spatial block IDs
are still stored per patch for future leakage checks.

## Run

```powershell
conda activate realtime_LCC_Rwkig
python pipelines/17_generate_training_patch_dataset.py
```

For a smaller smoke test:

```powershell
python pipelines/17_generate_training_patch_dataset.py --split train --limit 1
```

## Output

```text
data/interim/training_patches/training_patch_manifest.json
data/interim/training_patches/*.npz
```

Patch files remain local and are not committed to Git. The U-Net config now
points at the multi-year manifest by default.
