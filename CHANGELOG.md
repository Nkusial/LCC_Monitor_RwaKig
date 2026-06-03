# Changelog

## Unreleased

### Changed

- Added a WebGIS reliability color mode and legend so likely reliable and review-required change polygons are visible on the map.
- Removed developer QA cards for geospatial alignment and co-registration from the operational WebGIS sidebar.
- Simplified `README.md` into a public-facing project overview.
- Moved detailed implementation history and validation explanation into
  dedicated documentation files.
- Added `docs/VALIDATION.md` for confidence, uncertainty, weak-source
  reliability, review zones, and future field/reference-label validation.
- Added improvement-track controls for co-registration QA, temporal consensus,
  U-Net versus self-supervised comparison planning, public-demo packaging, and
  hosted FastAPI/PostGIS deployment readiness.
- Added a repeatable public-demo raster optimizer for oversized fallback PNG
  assets while preserving the tiled WebGIS display path.
- Added model-track comparison reporting for weak-supervised U-Net,
  unsupervised clustering, and the planned self-supervised/deep-clustering path.
- Added a label-free Sentinel patch embedding baseline to compare cluster
  separability against U-Net reliability without using weak labels during fitting.
- Added a self-supervised Sentinel patch autoencoder track for embedding
  clustering, anomaly scoring, weak-source interpretation, and U-Net comparison.
- Added a WebGIS self-supervised review overlay so embedding-supported,
  weak-support, and high-anomaly patch footprints can be inspected without
  treating them as land-cover classes or field accuracy.
- Added a self-supervised review-sample proposal workflow that uses anomaly
  scores and weak-source gaps to prioritize review before U-Net retraining
  while keeping the frozen 2026 benchmark untouched.
- Added a spatial self-supervised encoder pipeline that trains a convolutional
  patch autoencoder when PyTorch is available and uses a spatial-grid
  autoencoder fallback on the current Windows DLL-constrained environment.

## 0.2.0 - Validation-Aware Hosted WebGIS Milestone

### Added

- Hosted GitHub Pages WebGIS as the public portfolio entrypoint.
- Satellite-derived WebGIS overlays for Sentinel-2 false color, NDVI, NDWI, NDBI, Sentinel-1 VV, VH, VV/VH ratio, delta NDVI, and change confidence.
- Unsupervised validation workflow for no-field-label project stages.
- Phase 20B validation comparison across cluster counts, Sentinel-2-only versus Sentinel-1/2 fusion features, PCA reduction, masks, and baseline/after stacks.
- Frontend validation panel showing the best unsupervised cluster result and the no-ground-truth accuracy caveat.
- Reference-label planning panel for later field or high-resolution interpretation labels.
- Playwright-based screenshot capture for validation overlay and validation workflow portfolio assets.
- Explanatory comments across backend, frontend, and geospatial pipeline code.

### Validation

- Python test suite: `55 passed`.
- Frontend production build: succeeds.
- Best internal unsupervised validation score: silhouette `0.4459`.
- Hosted WebGIS exposes the validation overlay and validation summary panel.

### Known Limits

- No field accuracy is claimed until reference labels or field validation points are added.
- Hosted GitHub Pages is a static WebGIS demo; local FastAPI/PostGIS remains the dynamic backend workflow.
- Source GeoTIFF rasters stay local and are represented on the hosted app by lightweight PNG overlays.

## 0.1.0 - Local Portfolio Milestone

### Added

- Local 20 km x 20 km Kigali AOI configuration.
- Conda-based backend/geospatial environment.
- Docker Compose PostGIS service with spatial schema.
- FastAPI backend with health, AOI, scene, and change endpoints.
- Sentinel-2 and Sentinel-1 scene discovery and matching workflow.
- AOI-windowed preprocessing for multi-date monitoring rasters.
- Explainable change detection with before/after land-cover context.
- Confidence scoring and publish-ready filtering.
- PostGIS publication for change polygons.
- MapLibre WebGIS with monitored-group filters, changed/no-change quantities, and confidence threshold control.
- Reproducibility validation report.
- Portfolio docs, screenshots, release checklist, and future ML/MLOps roadmap.

### Validation

- Python test suite: `35+ passed`.
- Frontend production build: succeeds.
- Output validation report: `status: ok`.

### Known Limits

- This is a local portfolio system, not national-scale production monitoring.
- The current detector is an explainable baseline, not a trained supervised model.
- Raw Sentinel assets and derived raster outputs are intentionally excluded from Git.
