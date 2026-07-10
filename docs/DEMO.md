# Demo Script

## Goal

Show a local-first, confidence-aware land-cover change monitoring workflow for a 20 km x 20 km Kigali AOI.

## Before the Demo

Use `LCC_Monitor_RwaKig` as the local project folder and `realtime_LCC_Rwkig` as the Conda environment.

```powershell
cd path\to\LCC_Monitor_RwaKig
conda activate realtime_LCC_Rwkig
docker compose up -d postgis
python pipelines/07_validate_outputs.py
python pipelines/09_unsupervised_validation.py
```

Expected:

```text
status: ok
grid_mismatches: []
publish_ready_count: 2076
best silhouette score: 0.4459
```

## Start Apps

Terminal 1:

```powershell
uvicorn backend.app.main:app --reload
```

Terminal 2:

```powershell
cd web
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Walkthrough

1. Show the AOI and explain local-first scope.
2. Open the monitored land-cover filter.
3. Filter to `vegetation`, then `built_up`, then `water_moisture`.
4. Click a polygon and explain:
   - monitored land-cover group
   - transition
   - confidence
   - change magnitude
   - area
5. Change the satellite-derived layer to `Validation: Unsupervised clusters`.
6. Explain the validation panel:
   - no field labels are available yet
   - the app reports internal feature-space coherence, not field accuracy
   - the best current silhouette score is `0.4459`
   - the best configuration uses Sentinel-2 NDVI, NDWI, and NDBI on high-confidence changed pixels
7. Open the reference-label plan in the validation panel and explain how it will support later accuracy assessment.
8. Open API docs at `http://127.0.0.1:8000/docs`.
9. Show `/changes/summary` and `/changes/geojson`.
10. Mention validation report and test suite.

## Current Numbers To Say Out Loud

- Published changes: `2,076`
- Changed area: `60.5 km²`
- No-change area: `339.5 km²`
- Mean confidence: `0.88`
- Best unsupervised silhouette: `0.4459`

## Reviewer Talking Points

- The system avoids full-tile downloads by reading only AOI raster windows.
- Change records preserve before/after context, not only a binary changed/not-changed label.
- PostGIS stores publish-ready polygons for backend and WebGIS consumption.
- The current detector is explainable and ready to evolve toward stronger ML once reviewed labels are added.
- The validation story is honest: no field accuracy is claimed until reference labels exist.
