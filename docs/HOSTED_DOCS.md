# Hosted Documentation

This project uses GitHub Pages branch publishing for the public WebGIS demo and portfolio documentation.

Landing page redirect:

```text
docs/index.html
```

Interactive app:

```text
docs/app/index.html
```

## How To Enable GitHub Pages

In the GitHub repository:

1. Open **Settings**.
2. Open **Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Set **Branch** to `main`.
5. Set the folder to `/docs`.
6. Save.

Expected public URL:

```text
https://nkusial.github.io/LCC_Monitor_RwaKig/
```

## What Gets Published

GitHub Pages publishes the `docs/` folder directly:

- `docs/index.html`
- `docs/index.md`
- `docs/app/`
- documentation pages under `docs/`
- screenshot assets under `docs/assets/`

The hosted app is a static WebGIS demo. It uses snapshot files under `web/public/demo/` copied into `docs/app/demo/` during the Pages build. Local development still uses the live FastAPI/PostGIS API when it is available.

## Hosted WebGIS Layers

The public app includes display layers exported from AOI-clipped satellite rasters:

- Sentinel-2 false color for the 2025-02-20 optical reference.
- Sentinel-2 NDVI, NDWI, and NDBI index overlays.
- Sentinel-1 VV, VH, and VV/VH radar overlays for the 2025-02-21 radar reference.
- Latest delta NDVI and change-confidence raster overlays.
- Best unsupervised cluster overlay from the Phase 20B validation comparison.
- Change polygons that can be colored by monitored group, before state, after state, or change magnitude.

The PNG overlays are lightweight web display products. The source GeoTIFFs remain local under `data/processed/` and `data/outputs/`.

## Hosted Validation Panel

The app also publishes `validation_summary.json`, a compact frontend summary of
the Phase 20B unsupervised validation report. The panel shows:

- best silhouette score
- best experiment configuration
- cluster profile summaries
- no-ground-truth accuracy caveat
- reference-label planning notes for later independent validation

