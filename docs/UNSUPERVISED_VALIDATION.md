# Unsupervised Validation

Phase 20 adds an internal validation path for the project stage where no field
validation dataset is available.

This is an internal unsupervised cluster audit, not a field accuracy assessment.

## What It Does

The script `pipelines/09_unsupervised_validation.py` clusters AOI pixels using
the processed Sentinel feature stack:

- Sentinel-2 NDVI
- Sentinel-2 NDWI
- Sentinel-2 NDBI
- Sentinel-1 VV
- Sentinel-1 VH
- Sentinel-1 VV/VH ratio

It writes:

```text
data/outputs/<monitoring_tag>_unsupervised_clusters.tif
data/outputs/unsupervised_validation_report.json
```

## How To Run

```powershell
conda activate realtime_LCC_Rwkig
python pipelines/09_unsupervised_validation.py
```

Optional single run:

```powershell
python pipelines/09_unsupervised_validation.py --single --clusters 6 --sample-size 80000
```

## Phase 20B Comparison

The default run now compares:

```text
Cluster counts: 4, 5, 6, 8, 10
Feature sets:   Sentinel-2 indices only, Sentinel-1/2 fusion
Reduction:      none, PCA
Masks:          all valid pixels, high-confidence changed pixels
Dates:          baseline stack and latest after-change stack
```

UMAP is supported only when `umap-learn` is installed. It is not required for
the base project because PCA keeps the environment lighter and fully Conda-first.

## Current Local Result

The Phase 20B grid compared 80 experiments. The best current configuration is:

```text
Monitoring stack:  monitoring_20250106_20250104
Feature set:       Sentinel-2 NDVI, NDWI, NDBI
Mask:              high-confidence changed pixels
Clusters:          4
Valid pixels:      863,338
Silhouette score:  0.4459
```

The report also compares monitored change groups against the unsupervised
cluster raster using representative points from the published change polygons.

## Hosted WebGIS Visibility

Phase 21 publishes the best unsupervised cluster raster as a hosted WebGIS
overlay and adds a validation panel to the frontend. The panel intentionally
shows the no-ground-truth caveat beside the silhouette score so the result is
useful without being overstated.

## How To Interpret It

This is not field accuracy and should not be presented as a confusion matrix
against ground truth.

It is useful for:

- checking whether monitored groups occupy coherent Sentinel feature-space
  clusters
- spotting ambiguous groups such as mixed, bare/sparse, and built-up where
  spectral/radar signatures overlap
- documenting uncertainty before a future field or reference-label validation
  dataset is available

The earlier single run scored `0.2441` because it clustered all valid pixels
with a fused Sentinel-1/2 feature set and 6 clusters. The comparison grid shows
that a masked Sentinel-2 index-only configuration separates the strongest
changed land-cover signals more clearly. That is the configuration to report
for internal validation, with the caveat that it is still not field accuracy.

## Next Improvement

When labels become available, replace this internal audit with sampled reference
validation:

- validation points
- class labels from field data or high-resolution interpretation
- confusion matrix
- producer/user accuracy
- class-level F1 score
