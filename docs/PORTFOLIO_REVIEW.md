# Portfolio Review Guide

## One-Minute Pitch

This project is a local-first near-real-time land-cover change monitoring system for a 20 km x 20 km peri-urban AOI. It combines Sentinel-2 optical indices, Sentinel-1 RTC radar features, PostGIS, FastAPI, and a MapLibre WebGIS to detect, score, publish, and inspect candidate land-cover changes.

## Reviewer Links

- Hosted WebGIS: <https://nkusial.github.io/LCC_Monitor_RwaKig/app/>
- GitHub repository: <https://github.com/Nkusial/LCC_Monitor_RwaKig>
- Local WebGIS, when running the dev server: <http://127.0.0.1:5173>
- Local API docs, when running FastAPI: <http://127.0.0.1:8000/docs>

## What To Show First

1. Open the hosted WebGIS at <https://nkusial.github.io/LCC_Monitor_RwaKig/app/> or the local WebGIS at `http://127.0.0.1:5173`.
2. Show the Kigali AOI.
3. Point to the published change count and mean confidence.
4. Filter by monitored land cover:
   - `vegetation`
   - `built_up`
   - `water_moisture`
5. Switch the satellite-derived layer to `Validation: Unsupervised clusters`.
6. Point to the validation panel:
   - best silhouette score: `0.4459`
   - Sentinel-2 index-only feature set
   - high-confidence changed-pixel mask
   - no field accuracy claim
7. Click a change polygon and explain:
   - monitored land-cover group
   - before/after transition
   - confidence
   - change magnitude
   - area

## What Makes It Portfolio-Quality

- AOI-windowed raster reads avoid full Sentinel tile downloads.
- Multi-date Sentinel-1/2 features are aligned to a common CRS/grid.
- Change records preserve baseline and after-change context.
- Confidence scoring separates detected candidates from publish-ready records.
- PostGIS stores final geospatial outputs for API and WebGIS use.
- The project has repeatable validation, CI checks, and tag-based CD release packaging.
- The hosted WebGIS exposes validation evidence instead of hiding it in local logs.

## Validation Story

This project currently has no field validation dataset, so it does not claim
ground-truth classification accuracy. That is intentional and reviewer-safe.

Instead, Phase 20B runs an unsupervised cluster audit over the Sentinel feature
stack. The first single configuration produced a weak silhouette score of
`0.2441`, so the workflow was improved by comparing cluster counts, feature
sets, PCA reduction, all-pixel versus high-confidence-change masks, and
baseline/after monitoring stacks.

Best current result:

```text
Feature set:       Sentinel-2 NDVI, NDWI, NDBI
Mask:              high-confidence changed pixels
Clusters:          4
Silhouette score:  0.4459
```

How to present it:

- Say this is internal feature-space coherence, not field accuracy.
- Show the `Validation: Unsupervised clusters` overlay.
- Explain that Sentinel-2 index-only clustering separated the strongest changed
  land-cover signals better than all-pixel Sentinel-1/2 fusion.
- Point to the planned reference-label workflow as the path toward confusion
  matrices, producer accuracy, user accuracy, and F1 score.

## Frequently Asked Questions

| Question | Answer |
| --- | --- |
| Why not deep learning first? | This phase establishes explainable, reproducible baselines before labels exist. |
| What makes it near-real-time? | The pipeline is designed around repeatable scene discovery, AOI-windowed preprocessing, and publishable change records. |
| Why Sentinel-1 and Sentinel-2? | Optical indices help interpret land cover; radar helps when clouds or surface moisture complicate optical signals. |
| What is local-first? | The system runs locally with open-source tools and stores outputs in local folders/PostGIS. |
| Does this project have CI/CD? | Yes. CI runs tests and the frontend build on GitHub pushes and pull requests. CD runs on version tags and packages docs plus the WebGIS build as a release artifact. |
| Is the `0.4459` silhouette score field accuracy? | No. It is an unsupervised feature-space coherence score used because no field labels exist yet. |
| Why did Sentinel-2-only clustering score better than Sentinel-1/2 fusion? | For this AOI and mask, NDVI/NDWI/NDBI separated the strongest changed pixels more cleanly; radar remains useful for detection and cloud-resilience but can add noise to unsupervised clustering. |
| What is the next ML step? | Harden the data foundation first: master grid, class harmonization, weak labels, soft labels, patch splits, then supervised segmentation. |

## Demo Success Criteria

- `GET /health` returns `{"status": "ok"}`.
- `GET /changes/summary` returns published records.
- WebGIS shows colored change polygons.
- WebGIS shows the `Validation: Unsupervised clusters` overlay and validation panel.
- `python pipelines/07_validate_outputs.py` reports `status: ok`.
- `python pipelines/09_unsupervised_validation.py` reports the best Phase 20B configuration.
- `pytest -q` passes.

