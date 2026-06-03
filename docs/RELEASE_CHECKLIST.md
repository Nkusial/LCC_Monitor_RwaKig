# Release Checklist

Use this before sharing the portfolio repo or tagging a local milestone.

## Local Services

- Start PostGIS:
  `docker compose up -d postgis`
- Start FastAPI:
  `uvicorn backend.app.main:app --reload`
- Start WebGIS:
  `cd web`
  `npm run dev`

## Validation Commands

```powershell
conda activate realtime_LCC_Rwkig
pytest -q
python pipelines/07_validate_outputs.py
cd web
npm run build
npm run screenshots:phase23
```

Expected status:

```text
tests pass
validation_report.json status: ok
frontend build succeeds
validation screenshots regenerate
```

## Demo Checks

- `http://127.0.0.1:8000/health` returns `{"status": "ok"}`.
- `http://127.0.0.1:8000/changes/summary` reports changed and no-change quantities.
- `http://127.0.0.1:5173` shows the AOI, monitored groups, changed area, and no-change area.
- `https://nkusial.github.io/LCC_Monitor_RwaKig/app/` opens the hosted WebGIS.
- The hosted satellite layer selector includes `Validation: Unsupervised clusters`.
- The validation panel shows silhouette `0.4459` and the no-field-accuracy caveat.
- The WebGIS legend includes:
  - no change
  - vegetation
  - built up
  - water moisture
  - bare sparse
  - mixed

## Git Hygiene

- Keep raw/interim/processed/output rasters out of Git.
- Keep `.env` out of Git.
- Commit source code, config examples, docs, tests, and screenshot assets.
- Review `git status --short` before every milestone commit.

## Portfolio Talking Points

- Local-first, reproducible geospatial stack.
- AOI-windowed Sentinel processing instead of full-tile reads.
- Sentinel-2 optical indices plus Sentinel-1 radar features.
- Baseline and after-change attribution for each change record.
- Confidence scoring before PostGIS publication.
- API and WebGIS expose changed/no-change quantities for interpretation.
- Hosted WebGIS exposes satellite-derived overlays and unsupervised validation evidence.
- Current validation is honest about the lack of field labels and explains the planned reference-label workflow.

