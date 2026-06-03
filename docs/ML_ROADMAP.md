# Future ML/MLOps Roadmap

## Current State

The current `v0.2.0` detector is Version 1: explainable and rule-based. It produces:

- aligned multi-date Sentinel-1/2 features
- candidate change classes
- before/after land-cover states
- change magnitude
- final confidence and reliability
- PostGIS-published geospatial records

This is the right foundation before supervised ML because it creates traceable
features and candidate labels. The next engineering step should harden the data
foundation before any deep learning work: master-grid alignment, class
harmonization, weak labels, soft labels for mixed pixels, and uncertainty-aware
evaluation.

## Phase A: Master Grid And Harmonization

- Create a fixed 10 m master grid for the AOI.
- Validate CRS, transform, bounds, resolution, shape, NoData, and dtype for every raster.
- Add class harmonization across current monitored groups, Dynamic World, ESA WorldCover, OSM, and Sentinel index evidence.
- Treat ambiguous evidence as `uncertain_mixed`.

## Phase B: Weak Labeling And Soft Labels

- Generate weak labels only where independent sources agree.
- Store hard labels, soft-label probabilities, source agreement, disagreement, and label confidence.
- Preserve mixed pixels as probability vectors instead of forcing one class.
- Mask uncertain pixels from supervised loss.

## Phase C: Patch Dataset And Labeling Review

- Review candidate polygons in the WebGIS.
- Generate 128 x 128 pixel patches with feature tensors, labels, confidence masks, date metadata, and spatial block IDs.
- Use spatial and temporal holdout splits, not random pixel splits.

## Phase D: Feature Table

Create a training table with:

```text
delta_ndvi
delta_ndbi
delta_ndwi
delta_vv_vh
before_ndvi
after_ndvi
before_ndbi
after_ndbi
before_state
after_state
area_m2
cloud_cover
radar_orbit_match
label
```

## Phase E: Segmentation Baseline

Start with land-cover segmentation before deep change detection:

- U-Net with a lightweight encoder
- class probability maps
- dominant class raster
- entropy and confidence layers
- soft-label loss with uncertainty masks

## Phase F: Baseline ML For Polygons

Start with lightweight models:

- logistic regression
- random forest
- gradient boosting

Use spatial cross-validation if more AOIs are added.

## Phase G: Deep Change Detection

- Add Siamese U-Net or ChangeFormer-style detection only after segmentation is validated.
- Compare temporally consistent T1/T2 composites.
- Require temporal persistence before publishing alerts.

## Phase H: MLOps

- Track experiments with MLflow or a lightweight local registry.
- Version features and labels.
- Add model evaluation reports.
- Publish model-scored polygons to PostGIS.
- Monitor drift in class balance, confidence, and false positives.

## Phase I: Production-Like Expansion

- Add more dates and seasons.
- Add more AOIs.
- Add cloud/shadow masking from SCL.
- Add orbit-aware Sentinel-1 comparison rules.
- Add tile cache or COG outputs for faster WebGIS rendering.
