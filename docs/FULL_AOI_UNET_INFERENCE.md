# Full-AOI U-Net Inference

Phase 35 runs the trained U-Net over complete processed AOI raster stacks.

## Why This Exists

Patch mosaics from Phase 34 only cover the selected training/validation patch
windows. They are useful for model QA, but they are not wall-to-wall maps.

Full-AOI inference slides a `128 x 128` window across the complete processed
Sentinel-1/2 feature stack and averages overlapping predictions. This produces
continuous model outputs across the valid AOI raster extent.

## Command

```powershell
python pipelines/20_full_aoi_unet_inference.py
```

For a quick one-pair check:

```powershell
python pipelines/20_full_aoi_unet_inference.py --split validation --limit 1
```

## Outputs

```text
data/interim/unet_full_aoi/unet_full_aoi_manifest.json
data/interim/unet_full_aoi/*_unet_full_aoi_dominant_class.tif
data/interim/unet_full_aoi/*_unet_full_aoi_confidence.tif
data/interim/unet_full_aoi/*_unet_full_aoi_entropy.tif
data/interim/unet_full_aoi/*_unet_full_aoi_coverage.tif
```

The dominant-class raster uses:

```text
0 = nodata / invalid feature pixel
1 = built_up
2 = managed_or_natural_vegetation
3 = bare_soil
4 = water_wetland
5 = uncertain_mixed
```

## Interpretation

These are still weakly supervised model outputs, not field-validated land-cover
maps. They are suitable for WebGIS visualization if clearly labeled as model
predictions with confidence and entropy layers.
