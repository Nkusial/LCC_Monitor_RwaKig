# Patch Dataset

Phase 27 creates local ML-ready patches from the aligned monitoring stack and
weak-label agreement layers.

## Stage

```text
pipelines/12_generate_patch_dataset.py
```

## Inputs

The patch generator reads:

- Sentinel-2 NDVI, NDWI, NDBI
- Sentinel-1 VV, VH, VV/VH ratio
- local weak hard labels
- label confidence
- uncertainty
- external weak-source agreement labels
- external weak-source agreement count
- external weak-source disagreement mask

All rasters must already match the master grid.

## Outputs

Patch files are local artifacts:

```text
data/interim/patches/*.npz
data/interim/patches/patch_manifest.json
```

They are not committed to Git.

Each `.npz` contains:

- `features`
- `hard_labels`
- `agreement_labels`
- `label_confidence`
- `uncertainty`
- `training_mask`
- `agreement_count`
- `disagreement_mask`

## Split Policy

Patches are split by spatial block id, not by random pixels. This reduces
spatial leakage between train, validation, and test sets.

## Training Mask

The training mask excludes:

- disagreement pixels
- low-confidence pixels
- invalid feature pixels
- unlabeled pixels

This keeps the weakly supervised baseline honest: agreement zones can train a
model, while disagreement zones remain uncertainty evidence.
