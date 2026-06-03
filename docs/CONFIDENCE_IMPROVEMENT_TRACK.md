# Confidence Improvement Track

This track implements the seven model-confidence improvements in a way that
fits the current local-first prototype.

## 1. Weak-Label Quality

The strict pseudo-label stage accepts pixels only when all of these conditions
are true:

- U-Net confidence is high
- entropy is low
- temporal consensus is high
- at least two weak sources agree
- weak-source disagreement is absent
- U-Net class is compatible with the weak-source agreement class

Run:

```powershell
python pipelines\27_generate_refined_pseudo_labels.py
```

## 2. Dynamic World

Dynamic World ingestion is scripted in `pipelines/22_export_dynamic_world.py`.
It requires Earth Engine authentication and a Google Cloud project. Once
exported, Dynamic World is harmonized by the existing external-source ingestion
stage.

## 3. Temporal Consistency

`pipelines/23_temporal_unet_consensus.py` writes consensus class, consensus
fraction, and instability rasters across available full-AOI U-Net dates.

## 4. Input Features

`pipelines/26_expand_feature_stack.py` adds:

- EVI2
- SAVI
- MNDWI
- BSI
- Sentinel-1 log ratio

`pipelines/29_ingest_topography.py` adds:

- DEM elevation
- slope
- terrain ruggedness
- topographic position index

These supplement the existing NDVI, NDWI, NDBI, VV, VH, and VV/VH ratio stack.
Topography is important for Kigali because the AOI contains steep terrain.

## 5. Model Training

The baseline trainer now supports early stopping and best-checkpoint saving.
`configs/model_unet_v2.yaml` defines a stricter retraining configuration with
expanded features and a longer training schedule.

`configs/model_unet_v3.yaml` defines the no-field-label model-improvement
track. It trains from date-specific refined pseudo-labels, uses Sentinel-1/2
plus topography features, and handles severe class imbalance with:

- `pipelines/30_balance_training_patch_manifest.py` to retain rare-class
  patches while capping majority-only training patches.
- class-aware weighted sampling during training.
- effective-number class weights.
- optional minority class loss floors for experiments where rare-class recall is more important than precision.
- weighted cross entropy plus generalized Dice plus focal loss plus Tversky loss.
- optional residual U-Net retraining through the `lightweight_resunet` architecture.

This is designed to make the prototype more transferable to other AOIs where
field labels are unavailable, while still reporting weak-evidence compatibility
instead of claiming field accuracy.

## 6. Confidence Calibration

`pipelines/28_calibrate_weak_confidence.py` creates weak-label calibration bins.
This is not field calibration; it compares model confidence with weak-source
support until reference labels exist.

## 7. Pseudo-Label Refinement

`pipelines/27_generate_refined_pseudo_labels.py` produces conservative
pseudo-label, mask, and confidence rasters for future retraining.

## What Still Requires External Access

Dynamic World is the only improvement in this track that cannot be fully
executed without user-side Earth Engine authentication.
