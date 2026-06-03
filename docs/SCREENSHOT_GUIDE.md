# Screenshot Guide

Save screenshots or short recordings under `docs/assets/`.

## Captured Screenshots

1. `01_webgis_overview.png`
   - WebGIS overview with AOI, sidebar metrics, and recent change records.

2. `02_api_docs.png`
   - FastAPI Swagger documentation.

3. `03_change_summary.png`
   - `/changes/summary` response for the published change layer.

## Optional Additional Screenshots

1. `04_vegetation_filter.png`
   - Set monitored land cover to `vegetation`.
   - Keep confidence at `0.75`.

2. `05_built_up_filter.png`
   - Set monitored land cover to `built_up`.
   - Show how the polygon distribution changes.

3. `06_change_popup.png`
   - Click a polygon.
   - Capture the popup showing transition, confidence, magnitude, and area.

4. `07_validation_report.png`
   - Run `python pipelines/07_validate_outputs.py`.
   - Capture the `status: ok` output.

5. `08_validation_overlay.png`
   - Open the WebGIS.
   - Set satellite-derived layer to `Validation: Unsupervised clusters`.
   - Keep the validation panel visible with the `0.4459` silhouette score.

6. `09_validation_workflow.png`
   - Open the reference-label workflow details in the validation panel.
   - Capture the no-ground-truth caveat and planned reference-label steps.

Regenerate the validation screenshots with Playwright:

```powershell
cd web
conda activate realtime_LCC_Rwkig
npm run screenshots:phase23
```

## Recording Note

A recorded walkthrough is intentionally not part of this repository yet. Add one
later when the WebGIS, georeference QA, and reliability controls are ready for a
polished public demo.
