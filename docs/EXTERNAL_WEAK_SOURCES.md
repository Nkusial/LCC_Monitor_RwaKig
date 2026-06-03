# External Weak Sources

Phase 26B ingests external weak-label candidates before patch generation.

## Sources

Ingested now:

- ESA WorldCover from the Planetary Computer `esa-worldcover` collection
- OSM buildings/roads/landuse from Overpass API

Supported through Earth Engine:

- Dynamic World from Google Earth Engine asset `GOOGLE/DYNAMICWORLD/V1`

Dynamic World is handled differently because it is accessed through Google
Earth Engine. If this machine is authenticated with Earth Engine and a Google
Cloud project is enabled, export the Dynamic World label band for the AOI/date
with:

```powershell
conda activate realtime_LCC_Rwkig
earthengine authenticate
earthengine set_project YOUR_GOOGLE_CLOUD_PROJECT
python pipelines\22_export_dynamic_world.py `
  --ee-project YOUR_GOOGLE_CLOUD_PROJECT `
  --rerun-external-ingestion
```

The exporter writes the local label raster to:

```text
data/interim/external_sources/dynamic_world_label.tif
```

Then it reruns, or you can manually rerun:

```powershell
python pipelines/11_ingest_external_weak_sources.py
```

## Outputs

The stage writes local rasters to `data/interim/external_sources/`:

- `esa_worldcover_harmonized.tif`
- `osm_buildings_roads_landuse_harmonized.tif`
- `dynamic_world_harmonized.tif` when a local Dynamic World export exists
- `weak_source_agreement_labels.tif`
- `weak_source_agreement_count.tif`
- `weak_source_disagreement_mask.tif`
- `external_weak_sources_summary.json`

These files are local processing artifacts and are not committed to Git.

## Agreement Logic

Each source is mapped to the harmonized class ids:

- `built_up`
- `managed_vegetation`
- `natural_vegetation`
- `bare_soil`
- `water_wetland`
- `uncertain_mixed`

Agreement labels are written only where at least two aligned sources vote for
the same harmonized class. Disagreement pixels are preserved so future patch
generation can mask or soft-label uncertain areas instead of training on them as
hard truth.
