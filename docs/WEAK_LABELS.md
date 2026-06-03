# Weak Labels

Phase 26 adds a conservative weak-label layer for future machine-learning work.
It does not create field validation labels and it does not claim classification
accuracy.
This phase does not claim classification accuracy.

## Candidate Evidence Sources

The weak-label workflow is designed around these candidates:

- Dynamic World
- ESA WorldCover
- OSM buildings/roads/landuse
- Sentinel-2 NDVI/NDWI/NDBI evidence
- Sentinel-1 VV/VH evidence
- existing high-confidence change polygons

Current active local sources:

- Sentinel-2 NDVI/NDWI/NDBI evidence
- Sentinel-1 VV/VH evidence
- existing high-confidence change polygons

Registered future external agreement sources:

- Dynamic World
- ESA WorldCover
- OSM buildings/roads/landuse

## Current Implementation

The pipeline stage is:

```text
pipelines/10_generate_weak_labels.py
```

It reads the latest aligned monitoring stack and generates:

- weak hard-label raster
- per-class soft-label probability rasters
- label-confidence raster
- uncertainty raster
- monitoring-agreement raster
- JSON summary

The output paths follow:

```text
data/outputs/weak_labels_<monitoring_tag>_hard_labels.tif
data/outputs/weak_labels_<monitoring_tag>_label_confidence.tif
data/outputs/weak_labels_<monitoring_tag>_uncertainty.tif
data/outputs/weak_labels_<monitoring_tag>_prob_<class>.tif
data/outputs/weak_labels_<monitoring_tag>_summary.json
```

## Why The Labels Are Conservative

The generator only writes a hard label when:

- the dominant soft-label probability is high enough
- label confidence is high enough
- the top two class probabilities are not too close

Ambiguous pixels stay unlabeled or uncertain. This avoids training future models
on visually confusing pixels such as mixed vegetation/bare soil or bare/built
surfaces.

## Class Space

Weak labels use the harmonized class taxonomy from
`configs/class_harmonization.yaml`:

- `built_up`
- `managed_vegetation`
- `natural_vegetation`
- `bare_soil`
- `water_wetland`
- `uncertain_mixed`

## Next Improvement

The next data-quality improvement is to ingest external weak-label sources and
write agreement/disagreement masks. Only agreement zones should become strong
training candidates. Disagreement zones should remain masked or soft-labeled.
