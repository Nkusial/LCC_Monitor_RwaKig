# Publishing Guide

This guide prepares the project for a public GitHub portfolio release.

## Repository Name

Suggested name:

```text
LCC_Monitor_RwaKig
```

Alternative shorter name:

```text
geoai-change-monitor
```

## Repository Description

```text
Local-first GeoAI platform for confidence-aware near-real-time land-cover change monitoring with Sentinel-1/2, PostGIS, FastAPI, and MapLibre.
```

## Suggested Topics

```text
geoai
geospatial
remote-sensing
sentinel-1
sentinel-2
postgis
fastapi
maplibre
land-cover-change
mlops
```

## Before First Push

Run:

```powershell
git status --short
pytest -q
python pipelines/07_validate_outputs.py
cd web
npm run build
```

Confirm:

```text
.env is not tracked
raw/interim/processed/output rasters are not tracked
docs/assets contains portfolio screenshots
README renders the WebGIS screenshot
```

## First Remote Setup

Create an empty GitHub repository, then run:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If your local branch is intentionally named `master`, you can keep it, but `main` is the common public default.

## Milestone Tags

For the original local portfolio milestone:

```powershell
git tag -a v0.1.0 -m "Local portfolio milestone"
git push origin v0.1.0
```

For the validation-aware hosted WebGIS milestone:

```powershell
git tag -a v0.2.0 -m "Validation-aware hosted WebGIS milestone"
git push origin v0.2.0
```

Use [GITHUB_RELEASE_NOTES.md](GITHUB_RELEASE_NOTES.md) as the release body and copy the matching version section.

## Public Demo Check

After GitHub Pages refreshes, verify:

```text
https://nkusial.github.io/LCC_Monitor_RwaKig/app/
```

Confirm the app opens with the MapLibre interface, and that the satellite-derived layer selector includes `Validation: Unsupervised clusters`.

## What Not To Publish

- `.env`
- raw Sentinel assets
- derived raster stacks
- local database volumes
- personal API keys or tokens
- very large generated media files unless intentionally added through Git LFS or releases

