# GitHub Release Notes

## v0.2.0 - Validation-Aware Hosted WebGIS Milestone

This release refreshes the portfolio around the hosted WebGIS and the no-field-label validation story. The public app now shows satellite-derived raster overlays, unsupervised cluster validation, changed/no-change quantities, and a clear future path toward reference-label accuracy assessment.

## Highlights

- GitHub Pages publishes the interactive MapLibre WebGIS from `docs/app/`.
- Sentinel-derived overlays are available in the hosted app:
  - Sentinel-2 false color
  - NDVI, NDWI, NDBI
  - Sentinel-1 VV, VH, VV/VH ratio
  - delta NDVI
  - change confidence
  - best unsupervised cluster overlay
- Phase 20B compares cluster counts, Sentinel-2-only versus Sentinel-1/2 fusion features, PCA, changed-pixel masks, and baseline/after stacks.
- Best internal validation configuration:

```text
Feature set:       Sentinel-2 NDVI, NDWI, NDBI
Mask:              high-confidence changed pixels
Clusters:          4
Silhouette score:  0.4459
Accuracy claim:    no field accuracy; feature-space coherence only
```

- Portfolio screenshots now include the validation overlay and reference-label workflow panel.
- Code comments were added across backend, frontend, and pipeline source files for reviewer readability.

## Demo Links

```text
Hosted WebGIS:
https://nkusial.github.io/LCC_Monitor_RwaKig/app/

Repository:
https://github.com/Nkusial/LCC_Monitor_RwaKig
```

## Validation Commands

```powershell
conda activate realtime_LCC_Rwkig
pytest -q
cd web
npm run build
```

Expected result:

```text
55 tests pass
frontend build succeeds
```

## Known Limits

- This remains a local-first portfolio system, not national-scale production monitoring.
- No field validation dataset is available yet, so the validation panel reports internal clustering coherence, not map accuracy.
- The hosted app is static; local dynamic development still uses FastAPI and PostGIS.
- Raw Sentinel assets and derived GeoTIFF stacks are intentionally excluded from Git.

## v0.1.0 - Local Portfolio Milestone

This release packages a local-first, confidence-aware land-cover change monitoring platform for a 20 km x 20 km Kigali peri-urban AOI.

## Highlights

- Sentinel-2 optical indices and Sentinel-1 radar features are aligned into a multi-date monitoring stack.
- AOI-windowed raster reads keep preprocessing practical on a local machine.
- Change records include baseline state, after-change state, monitored land-cover group, magnitude, confidence, reliability, and area.
- PostGIS stores publish-ready change polygons for FastAPI and WebGIS access.
- MapLibre WebGIS shows changed/no-change quantities, monitored-group filters, confidence thresholding, and clickable change records.
- Reproducibility checks validate raster grids, required outputs, scored changes, and publish-ready counts.

## Demo Snapshot

```text
AOI:               400 km²
Published changes: 2,076
Changed area:      60.5 km²
No-change area:    339.5 km²
Mean confidence:   0.88
```

## Validation

```text
pytest -q
npm run build
python pipelines/07_validate_outputs.py
```

Expected result:

```text
tests pass
frontend build succeeds
validation_report.json status: ok
```

## Known Limits

- This is a local portfolio project, not a national-scale production system.
- The current detector is an explainable baseline rather than a supervised deep learning model.
- Raw Sentinel assets and derived rasters are intentionally excluded from Git.

