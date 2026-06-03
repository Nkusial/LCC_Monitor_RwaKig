# Validation And Reliability

This project is designed as a confidence-aware screening prototype, not a
field-validated land-cover authority. The validation story therefore separates
engineering correctness, weak-evidence reliability, model uncertainty, and
future accuracy assessment.

## What Is Validated Now

### Engineering Reproducibility

- Pipeline outputs are checked by `pipelines/07_validate_outputs.py`.
- Tests run with `pytest -q`.
- Frontend builds with `npm run build`.
- CI runs tests and the frontend build on pushes and pull requests.

### Geospatial Alignment

The project uses an explicit 10 m AOI master grid in `configs/master_grid.yaml`.
Processed rasters are expected to match:

- CRS
- transform
- resolution
- bounds
- width and height
- NoData behavior

The WebGIS uses Web Mercator XYZ tiles for display, while analysis rasters remain
on the analytical master grid. `pipelines/39_coregistration_qa.py` adds a
prototype co-registration QA check using Sentinel-2 NDVI as the image-content
reference.

### Weak-Source Reliability

Weak-source agreement is used because no field validation dataset is available.
Current candidate evidence sources include:

- Dynamic World, when exported through Earth Engine
- ESA WorldCover
- OSM buildings, roads, and land-use support
- Sentinel-2 NDVI, NDWI, NDBI evidence
- Sentinel-1 VV, VH, and VV/VH evidence
- existing high-confidence change polygons

Agreement zones are useful for screening and review prioritization. They are not
ground-truth labels.

### Unsupervised Validation

The project includes unsupervised cluster comparison for no-field-label stages.
The best current internal feature-space result is:

```text
Feature set:       Sentinel-2 NDVI, NDWI, NDBI
Mask:              high-confidence changed pixels
Clusters:          4
Silhouette score:  0.4459
```

This score is **not accuracy**. It only describes how coherent the selected
feature-space clusters are under the tested configuration.

### U-Net Reliability

The U-Net track is weakly supervised. It supports model-output review through:

- dominant class maps
- confidence maps
- entropy maps
- review-zone overlays
- weak-source compatibility summaries
- train/validation reliability checks
- frozen 2026 proxy benchmark checks

The system intentionally labels risky model conditions as caution signals. In
the WebGIS, these are shown as `High caution`, `Moderate caution`, or `Lower
caution`, rather than as final accuracy judgments.

## What Is Not Validated Yet

The project does not yet include:

- field survey labels
- independent high-resolution visual interpretation labels
- confusion matrix
- producer accuracy
- user accuracy
- F1 score
- calibrated probability-to-accuracy relationship

Until those exist, the correct claim is:

```text
Weak-evidence compatibility only; not field accuracy.
```

## How To Interpret Confidence

Confidence is a reliability signal, not a guarantee. It combines available
model and weak-evidence support, but it can still be affected by:

- mixed Sentinel pixels
- seasonal vegetation change
- cloud, haze, or shadows
- SAR/optical acquisition differences
- mountainous terrain
- weak-source disagreement
- class imbalance
- domain shift between years

High confidence means the current evidence is more internally consistent. It
does not mean the class is field-confirmed.

## Review Zones

Review zones identify places where users should be cautious before trusting a
prediction. Common reasons are:

- low model confidence
- high entropy or class ambiguity
- weak-source disagreement
- 2026 test-scene/domain shift
- calibration gap
- terrain-related confusion

These zones make the prototype more honest: they show where the system is less
certain instead of hiding uncertainty.

## Recommended Future Validation

The next reliability upgrade is a small independent reference-label sample:

1. Create spatially distributed validation points or polygons.
2. Interpret labels from field visits or high-resolution imagery.
3. Keep reference labels separate from training labels.
4. Compute confusion matrix, producer accuracy, user accuracy, F1 score, and
   class-wise uncertainty summaries.
5. Use those results to calibrate confidence and reduce review-zone burden.

This would convert the project from weak-evidence screening toward stronger
accuracy assessment.
