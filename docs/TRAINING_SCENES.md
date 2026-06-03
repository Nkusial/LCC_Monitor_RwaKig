# Multi-Year Training Scene Discovery

Phase 28 prepares the project for weakly supervised deep learning by finding
multi-date Sentinel-2/Sentinel-1 scene pairs before model training starts.

## Purpose

The patch dataset from Phase 27 proves that the local feature and weak-label
pipeline works for the current monitoring stack. Deep learning needs a broader
scene base, so this phase searches 2023, 2024, 2025, and optional 2026 scenes
for train/validation/test splits.

## Selection Policy

Sentinel-2 is searched first because cloud cover controls optical feature
quality. The preferred rule is strict:

```text
Sentinel-2 L2A cloud cover < 10%
required tiles: 35MRT and 35MRU
same acquisition date
```

If a required year cannot produce a Sentinel-1 matched pair under the strict
rule, the script may use the configured fallback threshold. Fallback scenes are
not hidden; each catalog record includes `cloud_tier` and
`selection_cloud_threshold`.

Sentinel-1 RTC is then matched around each Sentinel-2 date:

```text
polarizations: VV and VH
preferred orbit: ascending
search window: +/- 6 days
maximum optical/radar gap: 96 hours
```

## Split Plan

```text
2023 -> train
2024 -> train
2025 -> validation
2026 -> test, optional because the project date is 2026-05-07
```

This is a temporal split foundation, not a random pixel split. Later model
training should still keep spatial block IDs from Phase 27 to avoid leakage
inside each year.

## Output

The discovery script writes:

```text
data/catalog/training_scene_pairs_2023_2026.json
data/catalog/training_scenes_2023_2026.geojson
```

The catalog stores item IDs, dates, tiles, cloud cover, radar orbit,
polarizations, and optical/radar time gap. It intentionally stores metadata,
not large raster assets, so the repo remains light and reproducible.

Phase 29 then writes a lightweight preprocessing-readiness manifest:

```text
data/catalog/training_stack_manifest_2023_2026.json
```

This manifest checks that every selected Sentinel-2 item is present in the
scene catalog and has the required optical assets `B03`, `B04`, `B08`, and
`B11` before any heavy raster reads begin.

## Selected Scene Cloud Cover

Cloud cover below is the exact Sentinel-2 STAC `eo:cloud_cover` metadata for
each selected tile, not a visual estimate from the map. Because the AOI uses two
MGRS tiles, the catalog records both tile values plus the mean and maximum tile
cloud cover used for transparent screening.

| Year | Split | Sentinel-2 date | 35MRT cloud % | 35MRU cloud % | Mean cloud % | Max tile cloud % | Sentinel-1 date | Gap hours |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2023 | train | 2023-07-21 | 0.01 | 0.02 | 0.01 | 0.02 | 2023-07-23 | 51.6 |
| 2023 | train | 2023-07-26 | 0.13 | 0.19 | 0.16 | 0.19 | 2023-07-26 | 16.4 |
| 2023 | train | 2023-07-16 | 0.09 | 1.53 | 0.81 | 1.53 | 2023-07-14 | 31.6 |
| 2023 | train | 2023-10-04 | 1.23 | 2.05 | 1.64 | 2.05 | 2023-10-06 | 64.4 |
| 2024 | train | 2024-08-04 | 0.06 | 0.08 | 0.07 | 0.08 | 2024-08-01 | 55.6 |
| 2024 | train | 2024-07-25 | 0.00 | 1.02 | 0.51 | 1.02 | 2024-07-22 | 68.2 |
| 2024 | train | 2024-06-30 | 0.81 | 6.56 | 3.69 | 6.56 | 2024-06-26 | 79.6 |
| 2024 | train | 2024-06-25 | 5.30 | 4.02 | 4.66 | 5.30 | 2024-06-26 | 40.4 |
| 2025 | validation | 2025-07-30 | 0.00 | 0.00 | 0.00 | 0.00 | 2025-07-27 | 55.6 |
| 2025 | validation | 2025-08-29 | 2.01 | 1.17 | 1.59 | 2.01 | 2025-09-01 | 88.4 |
| 2025 | validation | 2025-02-20 | 2.97 | 3.05 | 3.01 | 3.05 | 2025-02-21 | 40.4 |
| 2025 | validation | 2025-08-06 | 2.85 | 3.31 | 3.08 | 3.31 | 2025-08-08 | 64.4 |

No 2026 test pair was selected because no complete `35MRT` plus `35MRU`
Sentinel-2 pair was available through 2026-05-07 under the configured preferred
or fallback cloud policy.

## Run

```powershell
conda activate realtime_LCC_Rwkig
python pipelines/13_discover_training_scenes.py
python pipelines/14_prepare_training_stack_manifest.py
```

Or through the local runner:

```powershell
python pipelines/run_local_pipeline.py --from-stage training-scenes --to-stage validate
```

## Next Use

The next deep-learning phase should preprocess selected catalog pairs onto the
master grid, generate multi-date feature/weak-label patches, and only then
train a weakly supervised segmentation baseline.
