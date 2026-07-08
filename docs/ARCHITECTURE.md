# Architecture

## Architecture Goals

This project is a local-first GeoAI monitoring platform for a 20 km x 20 km
Kigali peri-urban AOI. The architecture is designed to make land-cover change
candidates reproducible, inspectable, and honest about uncertainty when field
reference data is not yet available.

The system separates five concerns:

- open satellite and weak-reference data acquisition
- AOI-windowed geospatial processing on a shared master grid
- change detection, weak-supervised ML, and reliability scoring
- local API/database services for dynamic inspection
- lightweight public WebGIS/docs delivery for portfolio review

The diagrams below are intentionally split into compact views so they remain
readable in GitHub preview, even when the page is zoomed.

## System Context

```mermaid
flowchart TB
    analyst["Analyst / reviewer"]
    github["GitHub repository<br/>code, docs, CI, release history"]
    pages["GitHub Pages<br/>static WebGIS and documentation"]
    local["Local workstation<br/>Conda, Docker, processing workspace"]
    postgis["Local PostGIS<br/>change polygons, AOI, summaries"]
    api["FastAPI backend<br/>health, AOI, scenes, changes"]
    webgis["MapLibre WebGIS<br/>filters, popups, reliability views"]

    analyst --> pages
    analyst --> webgis
    github --> pages
    local --> github
    local --> postgis
    postgis --> api
    api --> webgis

    classDef human fill:#f8fafc,stroke:#475569,color:#0f172a
    classDef public fill:#e0f2fe,stroke:#0369a1,color:#0f172a
    classDef local fill:#ecfdf5,stroke:#047857,color:#0f172a
    classDef service fill:#fff7ed,stroke:#c2410c,color:#0f172a

    class analyst human
    class github,pages public
    class local,postgis local
    class api,webgis service
```

**Notes:** GitHub Pages is the public static delivery target. FastAPI and
PostGIS are part of the local dynamic runtime, so the public site can be viewed
without exposing a hosted database or backend service.

## Processing Architecture: Data Foundation

```mermaid
flowchart TB
    subgraph sources["Open and weak-reference sources"]
        s2["Sentinel-2 L2A<br/>blue, green, red, NIR, SWIR"]
        s1["Sentinel-1 RTC<br/>VV, VH"]
        dw["Dynamic World<br/>weak class probabilities"]
        esa["ESA WorldCover<br/>weak land-cover prior"]
        osm["OpenStreetMap<br/>buildings, roads, landuse"]
        dem["Terrain features<br/>elevation, slope"]
    end

    subgraph harmonize["Geospatial harmonization"]
        aoi["AOI definition<br/>configs/aoi.geojson"]
        grid["Master grid<br/>CRS, bounds, transform, resolution"]
        preprocess["AOI-windowed preprocessing<br/>clip, reproject, resample, mask"]
        qa["Alignment QA<br/>CRS, transform, extent, shift checks"]
    end

    subgraph features["Analysis-ready features"]
        optical["Optical indices<br/>NDVI, NDWI, NDBI"]
        radar["Radar features<br/>VV, VH, VV/VH"]
        temporal["Temporal deltas<br/>T1 to T2 feature change"]
        weak["Weak-label evidence<br/>agreement, disagreement, soft labels"]
    end

    s2 --> preprocess
    s1 --> preprocess
    dem --> preprocess
    aoi --> grid
    grid --> preprocess
    preprocess --> qa
    qa --> optical
    qa --> radar
    optical --> temporal
    radar --> temporal
    dw --> weak
    esa --> weak
    osm --> weak
    optical --> weak
    radar --> weak

    classDef source fill:#eff6ff,stroke:#2563eb,color:#0f172a
    classDef harmonized fill:#ecfdf5,stroke:#059669,color:#0f172a
    classDef feature fill:#fefce8,stroke:#ca8a04,color:#0f172a

    class s2,s1,dw,esa,osm,dem source
    class aoi,grid,preprocess,qa harmonized
    class optical,radar,temporal,weak feature
```

**Notes:** All raster-derived evidence must pass through the same AOI and master
grid logic before comparison. This is the control point that prevents apparent
change from being caused by CRS, transform, resolution, or extent mismatch.

## Processing Architecture: Detection And Reliability

```mermaid
flowchart TB
    subgraph inputs["Aligned feature evidence"]
        temporal["Temporal deltas<br/>optical and radar change"]
        weak["Weak-label evidence<br/>Dynamic World, ESA, OSM, Sentinel support"]
        features["Feature stack<br/>NDVI, NDWI, NDBI, VV, VH, VV/VH"]
    end

    subgraph models["Detection and model tracks"]
        rules["Explainable change detector<br/>thresholds and magnitude"]
        unet["Weak-supervised U-Net baseline<br/>class probability, entropy, review zones"]
        ssl["Label-free embedding baseline<br/>self-supervised review candidates"]
    end

    subgraph gate["Reliability gate"]
        score["Publish/review decision<br/>confidence, compatibility, temporal consensus"]
        review["Review zones<br/>low confidence, high entropy, disagreement"]
    end

    subgraph outputs["Publishable outputs"]
        geojson["Change GeoJSON<br/>before/after class, date window, magnitude"]
        reports["Validation reports<br/>QA, uncertainty, benchmark summaries"]
        tiles["Static WebGIS assets<br/>small demo tiles and summaries"]
        db["PostGIS tables<br/>AOI, scenes, change_polygons"]
    end

    temporal --> rules
    features --> unet
    weak --> unet
    features --> ssl
    rules --> score
    unet --> score
    ssl --> score
    weak --> score
    score --> geojson
    score --> reports
    score --> tiles
    score --> db
    score --> review
    review --> reports
    review --> tiles

    classDef input fill:#fefce8,stroke:#ca8a04,color:#0f172a
    classDef model fill:#fdf2f8,stroke:#be185d,color:#0f172a
    classDef gate fill:#fff7ed,stroke:#c2410c,color:#0f172a
    classDef output fill:#f8fafc,stroke:#475569,color:#0f172a

    class temporal,weak,features input
    class rules,unet,ssl model
    class score,review gate
    class geojson,reports,tiles,db output
```

**Notes:** The platform does not treat a U-Net class as truth. Rule-based change
signals, weak-supervised U-Net evidence, self-supervised review candidates, and
weak-source compatibility all feed the reliability gate. The gate decides what
is publishable and what remains a review priority.

## Runtime And Delivery Architecture

```mermaid
flowchart TB
    subgraph local_runtime["Local dynamic runtime"]
        docker["Docker Compose"]
        postgis["PostGIS service"]
        fastapi["FastAPI app"]
        local_web["Vite WebGIS dev server"]
    end

    subgraph static_release["Public static release"]
        docs["docs/<br/>architecture, operations, validation"]
        app["docs/app/<br/>prebuilt WebGIS"]
        demo["docs/app/demo/<br/>small public demo assets"]
        pages["GitHub Pages"]
    end

    subgraph automation["Quality and release automation"]
        ci["CI workflow<br/>pytest, build checks"]
        release["Release workflow<br/>tagged artifact packaging"]
        checks["Geospatial checks<br/>master grid, co-registration, public-safe assets"]
    end

    docker --> postgis
    postgis --> fastapi
    fastapi --> local_web
    docs --> pages
    app --> pages
    demo --> pages
    ci --> checks
    checks --> release
    release --> pages

    classDef runtime fill:#ecfdf5,stroke:#047857,color:#0f172a
    classDef static fill:#e0f2fe,stroke:#0369a1,color:#0f172a
    classDef automation fill:#fff7ed,stroke:#c2410c,color:#0f172a

    class docker,postgis,fastapi,local_web runtime
    class docs,app,demo,pages static
    class ci,release,checks automation
```

**Notes:** CI validates the code and build. CD is static-artifact delivery: the
WebGIS and documentation are published through GitHub Pages, while tagged
releases package reproducible artifacts. Hosted FastAPI/PostGIS deployment is a
future step, not part of the current public release.

## Key Data Contracts

| Contract | Purpose | Main fields or assets |
| --- | --- | --- |
| AOI GeoJSON | Defines the project boundary and display fit | geometry, CRS assumptions |
| Master grid | Forces consistent raster alignment | EPSG, transform, bounds, resolution, width, height |
| Scene catalog | Records selected Sentinel-1/2 acquisitions | date, sensor, tile, cloud cover, orbit, asset links |
| Feature stack | Analysis-ready raster inputs | NDVI, NDWI, NDBI, VV, VH, VV/VH, deltas |
| Weak-label stack | Independent evidence and review masks | Dynamic World, ESA, OSM, Sentinel index evidence |
| Change GeoJSON | Public WebGIS and API change records | before class, after class, date window, magnitude, confidence, reliability |
| Validation reports | Reviewer-facing reliability evidence | alignment QA, weak-source compatibility, review burden, benchmark notes |

## Quality Gates

The platform does not publish change candidates only because a model produced a
class. It applies staged reliability controls:

1. **Geospatial alignment gate:** rasters must match the AOI master grid before
   change detection or model inference.
2. **Temporal gate:** change should be tied to an explicit before/after date
   window, not an unspecified event date.
3. **Weak-source gate:** Dynamic World, ESA WorldCover, OSM, and Sentinel
   evidence are used as compatibility checks, not field accuracy labels.
4. **Uncertainty gate:** low confidence, high entropy, and weak-source
   disagreement are routed to review zones.
5. **Benchmark gate:** held-out periods, such as the 2026 benchmark, must not be
   used for model tuning and final evaluation at the same time.

## Design Choices

- **Local-first by default:** raw processing, PostGIS, and FastAPI run locally so
  the project remains reproducible without cloud infrastructure.
- **Static public demo:** GitHub Pages serves lightweight WebGIS assets and
  documentation. This is continuous delivery for public artifacts, not full
  backend cloud deployment.
- **Master-grid discipline:** every raster-derived layer should be aligned to
  the same CRS, transform, resolution, and AOI bounds before comparison.
- **Explainability before complexity:** rule-based deltas, weak-source checks,
  U-Net probabilities, entropy, and review zones are exposed to users rather
  than hidden behind a single accuracy number.
- **No field-accuracy claim:** weak labels and public demo outputs support
  screening and prioritization; independent expert samples are still required
  for formal accuracy assessment.

## Current Monitoring Groups

```text
vegetation
built_up
water_moisture
bare_sparse
mixed
unknown
```

## Known Limitations

- The public WebGIS currently demonstrates selected monitoring intervals rather
  than a complete operational 2023-2026 time-window alert history.
- The U-Net baseline is weak-supervised and should be treated as a review aid,
  not a field-validated classifier.
- Mixed Sentinel pixels, topographic effects, moisture, haze, and seasonal
  vegetation can produce ambiguous change evidence.
- Full hosted backend deployment is not part of the current public release; the
  FastAPI/PostGIS stack remains local-first until a dynamic hosted service is
  required.
