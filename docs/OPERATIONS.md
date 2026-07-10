# Local Operations

## Naming Note

`LCC_Monitor_RwaKig` is the public GitHub repository and the recommended local clone folder name. `realtime_LCC_Rwkig` is only the Conda environment used to run the local geospatial stack; it is not a second repository.

## Start Services

```powershell
cd path\to\LCC_Monitor_RwaKig
conda activate realtime_LCC_Rwkig
docker compose up -d postgis
uvicorn backend.app.main:app --reload
```

In a second terminal:

```powershell
cd path\to\LCC_Monitor_RwaKig\web
conda activate realtime_LCC_Rwkig
npm run dev
```

## Useful URLs

```text
API landing:      http://127.0.0.1:8000/
API docs:         http://127.0.0.1:8000/docs
Health:           http://127.0.0.1:8000/health
Change summary:   http://127.0.0.1:8000/changes/summary
WebGIS:           http://127.0.0.1:5173
```

## Re-run Local Pipeline

Use existing Phase 4 rasters:

```powershell
python pipelines/run_local_pipeline.py --from-stage detect --to-stage validate
```

Rebuild monitoring rasters too:

```powershell
python pipelines/run_local_pipeline.py --from-stage stack --to-stage validate
```

## Quality Gates

```powershell
pytest -q
python pipelines/07_validate_outputs.py
python pipelines/39_coregistration_qa.py
python pipelines/40_improvement_track_report.py
python pipelines/41_package_public_demo_manifest.py
python pipelines/42_optimize_public_demo_rasters.py
python pipelines/43_compare_model_tracks.py
python pipelines/44_label_free_embedding_baseline.py
cd web
npm run build
```

Expected validation status:

```text
status: ok
grid_mismatches: []
improvement track: publish gate and public-demo packaging summarized
```

## CI/CD

GitHub Actions provides:

- CI on pushes and pull requests through `.github/workflows/ci.yml`.
- Tag-based release packaging through `.github/workflows/release.yml`.

The delivery layer is intentionally static-artifact focused: CI validates the project, GitHub Pages serves the WebGIS/docs from `docs/`, and tagged releases package docs plus frontend build artifacts. The FastAPI/PostGIS backend remains local-first until a hosted dynamic backend is required.

