# U-Net Prediction Mosaics

Phase 34 converts patch-level U-Net inference products into geospatial rasters.

## Why Per-Date Mosaics

The patch manifest contains multiple Sentinel-1/2 acquisition pairs from
different years and dates. Predictions should therefore be mosaicked per
`training_tag` so each raster keeps a coherent acquisition date pair and grid.

## Command

```powershell
python pipelines/19_mosaic_unet_predictions.py
```

To mosaic only validation-date patches:

```powershell
python pipelines/19_mosaic_unet_predictions.py --split validation
```

## Outputs

```text
data/interim/unet_mosaics/unet_mosaic_manifest.json
data/interim/unet_mosaics/*_unet_dominant_class.tif
data/interim/unet_mosaics/*_unet_confidence.tif
data/interim/unet_mosaics/*_unet_entropy.tif
data/interim/unet_mosaics/*_unet_patch_coverage.tif
```

The dominant-class raster uses:

```text
0 = nodata / uncovered
1 = built_up
2 = managed_or_natural_vegetation
3 = bare_soil
4 = water_wetland
5 = uncertain_mixed
```

The confidence and entropy rasters use `-9999` as nodata outside covered patch
windows.

## Interpretation

These mosaics are model products, not field-validated maps. They are the right
intermediate layer before WebGIS publication because they preserve georeference,
date-pair identity, confidence, entropy, and patch coverage.
