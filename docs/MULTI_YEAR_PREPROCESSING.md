# Multi-Year Training Preprocessing

Phase 30 bridges the training-scene catalog and deep-learning patches.

## Goal

The project has selected Sentinel-2/Sentinel-1 scene pairs for 2023, 2024, and
2025. Phase 30 runs those pairs through the same AOI-windowed preprocessing
logic used by the original monitoring stack, while keeping the operation
controlled with filters and limits.

## Guardrails

The default mode is metadata-only:

```powershell
python pipelines/16_preprocess_training_pairs.py
```

That validates scene IDs, band configuration, radar IDs, and manifest writing
without reading remote raster data.

For actual raster processing, start small:

```powershell
python pipelines/16_preprocess_training_pairs.py --mode optical --split train --limit 1
python pipelines/16_preprocess_training_pairs.py --mode radar --split train --limit 1
```

Only after one pair is validated should you scale up:

```powershell
python pipelines/16_preprocess_training_pairs.py --mode all --split train
python pipelines/16_preprocess_training_pairs.py --mode all --split validation
```

## Outputs

Metadata mode writes:

```text
data/catalog/training_preprocessing_summary_2023_2026.json
data/processed/training_*_manifest.json
```

Raster modes write local GeoTIFF features such as:

```text
data/processed/training_*_s2_ndvi.tif
data/processed/training_*_s2_ndwi.tif
data/processed/training_*_s2_ndbi.tif
data/processed/training_*_s1_vv.tif
data/processed/training_*_s1_vh.tif
data/processed/training_*_s1_vv_vh_ratio.tif
```

Large raster outputs remain local and are not committed to Git.

## Next Use

After the training rasters exist, patch generation should be expanded to create
multi-year train and validation manifests. The U-Net baseline can then be fit
against a more realistic multi-date training dataset.
