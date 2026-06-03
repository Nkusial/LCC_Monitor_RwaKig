# Topography Features

Kigali's mountainous landscape affects this monitoring problem. Slope and
elevation influence settlement patterns, radar backscatter, surface moisture,
shadow, exposed soil, and false change signals.

Phase 47 ingests open Copernicus DEM data from the Microsoft Planetary Computer
and aligns it to the project 10 m master grid.

## Outputs

Local rasters are written to:

```text
data/interim/topography/kigali_dem.tif
data/interim/topography/kigali_slope.tif
data/interim/topography/kigali_ruggedness.tif
data/interim/topography/kigali_tpi.tif
```

The report is written to:

```text
data/outputs/topography_ingestion_report.json
```

## Current AOI Summary

The Kigali AOI has:

- elevation range: about 1,338 m to 2,073 m
- mean elevation: about 1,514 m
- mean slope: about 11.6 degrees
- 95th percentile slope: about 29.0 degrees

## Model Use

Topography is added only to the Version 2 confidence-improvement feature stack.
It does not change the existing six-channel baseline checkpoint.

Topography should help the model distinguish:

- built-up expansion on slopes
- bare soil and construction surfaces
- radar backscatter changes caused by terrain
- moisture/drainage patterns
- shadow-related uncertainty
