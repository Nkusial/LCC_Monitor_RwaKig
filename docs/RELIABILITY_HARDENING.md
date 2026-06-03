# Reliability Hardening

This project now separates four different ideas that are often confused in
early GeoAI prototypes:

- model prediction
- weak-source agreement
- temporal model stability
- field/reference-label accuracy

## Implemented Now

### WebGIS Review Zones

The hosted WebGIS includes `U-Net review zones`:

- orange: low model confidence
- red: high entropy
- purple: weak-source disagreement

These zones are map-review aids, not accuracy labels.

### Temporal Consensus

`pipelines/23_temporal_unet_consensus.py` builds wall-to-wall temporal
consensus rasters from all available full-AOI U-Net predictions.

Outputs:

- consensus class
- consensus fraction
- instability mask
- `data/outputs/unet_temporal_consensus_report.json`

### Review-To-Retraining Loop

`pipelines/24_prepare_retraining_review.py` creates a manifest of the highest
priority review records. This is the bridge from current weak supervision to a
future corrected-label retraining cycle.

It does not train on unreviewed uncertain pixels.

### Test-Set Readiness

`pipelines/25_report_test_set_readiness.py` reports whether a clean held-out
test split exists. At the current project date, the 2026 optional test split is
not available, so the project should not claim final test accuracy.

### No-Leakage Audit

`pipelines/32_validate_no_leakage.py` prevents benchmark cheating:

- 2026+ scenes and patches must be assigned only to `test`.
- A scene or patch source tag cannot appear in more than one split.
- 2026 test candidates are for final evaluation only after model freezing.
- Dynamic World, ESA WorldCover, OSM, Sentinel indices, Sentinel-1 evidence,
  and bootstrap polygons are proxy agreement sources, not field truth.

This audit must pass before training, calibration, or any benchmark claim.

### Deployment Hardening

The project remains local-first, but now includes a containerized backend path:

```powershell
docker compose -f docker-compose.deploy.yml up --build
```

This runs PostGIS plus the FastAPI backend. The hosted GitHub Pages WebGIS is
still static, while local development can use the live FastAPI/PostGIS stack.

## Still Not Claimed

- no field accuracy
- no confusion matrix without reference labels
- no national-scale production deployment
- no Dynamic World fusion until Earth Engine authentication/export succeeds
