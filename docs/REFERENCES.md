# References And Scientific Basis

This page documents the scientific and technical basis for the Confidence-Aware Land-Cover Change Monitor. It is not a generic bibliography. Each reference is included because it supports a specific design choice in the project: Sentinel-1/2 data selection, weak-source evidence, spectral indices, uncertainty handling, validation limits, or the open-source geospatial stack.

The project should be interpreted as a decision-support prototype. These references strengthen the methodology, but they do not replace independent field or expert reference samples for final accuracy assessment.

## Earth Observation Data Foundation

### Sentinel-2 optical imagery

**Reference:** European Space Agency. Sentinel-2 mission documentation.  
**Traceability:** <https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2>

**Why this source matters:** Sentinel-2 is the main optical data source for the prototype. ESA documents Sentinel-2 as a high-resolution multispectral Copernicus mission with 13 spectral bands, 10 m class observations, wide swath coverage, and frequent revisit capability. Those characteristics make it suitable for local land-cover monitoring over a 20 km x 20 km AOI.

**How it is used in this project:**

- Search and selection of cloud-minimized Sentinel-2 L2A scenes.
- AOI-windowed raster reads to avoid full-tile processing.
- NDVI, NDWI, and NDBI feature generation.
- False-color WebGIS raster overlays.
- Optical evidence in weak-label compatibility and reliability checks.

**Important limitation:** Sentinel-2 is optical. Clouds, shadows, haze, topographic illumination, and mixed pixels can weaken class confidence. This is one reason the project combines optical evidence with Sentinel-1 radar, weak sources, and review zones.

### Sentinel-1 radar imagery

**Reference:** European Space Agency. Sentinel-1 mission documentation.  
**Traceability:** <https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1>

**Reference:** Microsoft Planetary Computer. Sentinel-1 RTC dataset documentation.  
**Traceability:** <https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc>

**Why these sources matter:** Sentinel-1 provides radar observations that complement Sentinel-2. ESA describes Sentinel-1 as an all-weather, day-and-night radar mission. The Planetary Computer Sentinel-1 RTC collection provides radiometrically terrain-corrected data that is better suited to analysis-ready workflows than raw SAR scenes.

**How it is used in this project:**

- VV, VH, and VV/VH radar features.
- Radar support for land-cover change candidates.
- Cross-checking optical-only signals in cloudy or moisture-sensitive areas.
- Future near-real-time monitoring where optical scenes are not always usable.

**Important limitation:** Radar backscatter is sensitive to geometry, moisture, surface roughness, and terrain. In Kigali's hilly landscape, radar evidence must be interpreted carefully and aligned to a common grid before being used in confidence scoring.

## Weak Reference And Context Sources

### Dynamic World

**Reference:** Brown, C. F. et al. (2022). *Dynamic World, Near real-time global 10 m land use land cover mapping*. Scientific Data, 9, 251.  
**Traceability:** <https://www.nature.com/articles/s41597-022-01307-4>

**Why this source matters:** Dynamic World is a near-real-time global 10 m land-use/land-cover product derived from Sentinel-2 and deep learning. It is highly relevant to a near-real-time land-cover monitoring prototype.

**How it is used in this project:**

- Candidate weak-label source.
- Weak-source agreement and disagreement checks.
- Review-zone generation where Dynamic World conflicts with Sentinel or other auxiliary evidence.
- Confidence-support scoring.

**Important limitation:** Dynamic World is not field validation. It is model-derived evidence. In this repository it is therefore used as weak support, not as ground truth.

### ESA WorldCover

**Reference:** ESA WorldCover. Worldwide land-cover mapping.  
**Traceability:** <https://esa-worldcover.org/en>

**Why this source matters:** ESA WorldCover provides global land-cover products at 10 m resolution developed and validated using Sentinel-1 and Sentinel-2 data. It is a useful independent weak-reference layer for broad land-cover categories.

**How it is used in this project:**

- Harmonized weak prior for broad land-cover classes.
- Agreement checks with Dynamic World, OSM, Sentinel indices, and radar features.
- Caution flags when the weak-reference sources disagree.

**Important limitation:** WorldCover products are not near-real-time and may not capture recent local changes. They are useful for prior context, not for confirming every candidate change event.

### OpenStreetMap

**Reference:** OpenStreetMap contributors. OpenStreetMap data and copyright.  
**Traceability:** <https://www.openstreetmap.org/copyright>

**Why this source matters:** OSM provides human-mapped buildings, roads, and land-use context. This is particularly useful for interpreting built-up expansion and transport-related development.

**How it is used in this project:**

- Built-up and road-context support.
- Weak evidence for built-up/impervious candidates.
- Context for reviewer interpretation of change polygons.

**Important limitation:** OSM completeness varies by location. Absence of an OSM feature is not evidence that a building, road, or land-use class is absent.

## Spectral Indices And Feature Evidence

### NDVI

**Reference:** Rouse, J. W., Haas, R. H., Schell, J. A., and Deering, D. W. (1974). *Monitoring vegetation systems in the Great Plains with ERTS*. NASA Technical Reports Server.  
**Traceability:** <https://ntrs.nasa.gov/citations/19740022614>

**Why this source matters:** NDVI is a foundational vegetation index. It supports vegetation condition analysis and vegetation-loss interpretation in multispectral imagery.

**How it is used in this project:**

- Vegetation signal detection.
- Vegetation-loss candidates.
- Optical evidence for weak-label compatibility.

**Important limitation:** NDVI can be affected by soil background, atmospheric conditions, seasonal phenology, and mixed pixels.

### NDWI

**Reference:** McFeeters, S. K. (1996). *The use of the Normalized Difference Water Index in the delineation of open water features*. International Journal of Remote Sensing.  
**Traceability:** <https://doi.org/10.1080/01431169608948714>

**Why this source matters:** NDWI supports water and wetness signal interpretation from optical imagery.

**How it is used in this project:**

- Water/wetness signal detection.
- Moisture-related change screening.
- Caution flags where water, wet soil, shadow, and mixed surfaces may be confused.

**Important limitation:** In urban and peri-urban landscapes, wetness signals can be ambiguous. The project labels this class as water/wetness signal rather than claiming pure open water.

### NDBI

**Reference:** Zha, Y., Gao, J., and Ni, S. (2003). *Use of normalized difference built-up index in automatically mapping urban areas from TM imagery*. International Journal of Remote Sensing.  
**Traceability:** <https://doi.org/10.1080/01431160304987>

**Why this source matters:** NDBI supports built-up and impervious-surface evidence from multispectral imagery.

**How it is used in this project:**

- Built-up/impervious signal detection.
- Built-up expansion candidates.
- Cross-checking with OSM buildings/roads and Sentinel-1 radar.

**Important limitation:** Bare soil and bright built surfaces can be spectrally similar. This is why the project distinguishes built-up evidence from bare or sparse ground and uses reliability controls before publishing candidate changes.

## Validation, Confidence, And Uncertainty

### Land-change accuracy assessment

**Reference:** Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., and Wulder, M. A. (2014). *Good practices for estimating area and assessing accuracy of land change*. Remote Sensing of Environment.  
**Traceability:** <https://doi.org/10.1016/j.rse.2014.02.015>

**Why this source matters:** This is the key reference behind the project's cautious accuracy language. It supports the need for reference samples, response design, error matrices, area-adjusted estimates, and confidence intervals before claiming accuracy.

**How it is used in this project:**

- Separates model confidence from field accuracy.
- Justifies the statement that weak-source compatibility is not ground-truth validation.
- Frames future expert-sample collection as a required step for true accuracy assessment.

**Important limitation:** Until expert or field reference samples are added, the project can report reliability and evidence agreement, but it should not report final classification accuracy.

### Neural-network calibration

**Reference:** Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.  
**Traceability:** <https://arxiv.org/abs/1706.04599>

**Why this source matters:** Modern neural networks can be overconfident. This supports the project's decision to avoid treating raw model probabilities as reliable confidence by themselves.

**How it is used in this project:**

- Motivates confidence calibration checks.
- Supports reliability controls for U-Net outputs trained on weak labels.
- Encourages reporting review zones and uncertainty rather than a single unqualified confidence score.

**Important limitation:** Calibration itself requires validation data or a defensible proxy. Weak labels can help diagnose behavior, but they cannot prove field accuracy.

### Deep-learning uncertainty

**Reference:** Gal, Y. and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. Proceedings of Machine Learning Research.  
**Traceability:** <https://proceedings.mlr.press/v48/gal16.html>

**Why this source matters:** The paper supports the principle that uncertainty should be represented explicitly in deep-learning workflows.

**How it is used in this project:**

- Supports uncertainty-aware review zones.
- Supports the idea of separating likely reliable changes from needs-review candidates.
- Frames entropy and disagreement as operational review signals.

**Important limitation:** The current prototype uses lightweight uncertainty and reliability controls. More rigorous uncertainty estimation would require a stronger trained model, calibrated validation data, or repeated inference methods.

## Machine Learning And Model Tracks

### U-Net segmentation baseline

**Reference:** Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.  
**Traceability:** <https://arxiv.org/abs/1505.04597>

**Why this source matters:** U-Net is a widely used encoder-decoder segmentation architecture. It provides a clear supervised baseline for pixel-level land-cover segmentation experiments.

**How it is used in this project:**

- Supervised segmentation baseline.
- Trained with weak labels because field labels are not yet available.
- Compared against unsupervised and label-free tracks using review burden, weak-source compatibility, and temporal stability.

**Important limitation:** U-Net is supervised. If trained on weak labels, it can learn weak-label bias. The project therefore treats U-Net output as candidate evidence, not final truth.

### Label-free embedding baseline

**Reference basis:** The label-free track is included as an engineering baseline rather than a field-validated model. It tests whether Sentinel-1/2 patch features form meaningful groups before weak labels are applied.

**How it is used in this project:**

- Patch-level embedding exploration.
- Anomaly/review-sample proposal.
- Comparison against the weakly supervised U-Net track.

**Important limitation:** Label-free clusters need interpretation. Dynamic World, ESA WorldCover, OSM, Sentinel indices, and radar evidence are used post hoc to interpret clusters, not to claim supervised accuracy.

## Geospatial Engineering Stack

### STAC

**Reference:** SpatioTemporal Asset Catalogs. STAC specification.  
**Traceability:** <https://stacspec.org/en>

**Why this source matters:** STAC provides a common language for describing, cataloging, and discovering spatiotemporal geospatial assets.

**How it is used in this project:**

- Scene search and metadata organization.
- Reproducible monitoring-pair documentation.
- Transparent date, sensor, and asset tracking.

### Cloud Optimized GeoTIFF

**Reference:** Cloud Optimized GeoTIFF project.  
**Traceability:** <https://www.cogeo.org/>

**Why this source matters:** COG practices support efficient raster access and future web/cloud raster serving.

**How it is used in this project:**

- Design guidance for keeping heavy rasters local while publishing lightweight demo assets.
- Future direction for scalable raster delivery.

### PostGIS

**Reference:** PostGIS project documentation.  
**Traceability:** <https://postgis.net/>

**Why this source matters:** PostGIS extends PostgreSQL with spatial storage, indexing, and querying. It is suitable for storing AOIs, scenes, change polygons, summaries, and future API-backed map layers.

**How it is used in this project:**

- Spatial schema design.
- Change polygon storage.
- Future dynamic backend deployment path.

### FastAPI

**Reference:** FastAPI official documentation.  
**Traceability:** <https://fastapi.tiangolo.com/>

**Why this source matters:** FastAPI is the local API framework used to expose health checks, AOI metadata, scenes, changes, summaries, and GeoJSON endpoints.

**How it is used in this project:**

- Local-first backend service.
- API documentation through OpenAPI/Swagger UI.
- Future hosted backend path.

### MapLibre

**Reference:** MapLibre project.  
**Traceability:** <https://maplibre.org/>

**Why this source matters:** MapLibre provides the open-source web mapping library used for the hosted static WebGIS.

**How it is used in this project:**

- Interactive map rendering.
- Change polygon visualization.
- Reliability styling and popups.
- Static public demo delivery through GitHub Pages.

## Reviewer Interpretation

The references above support the design and credibility of the prototype. They should be read with the following interpretation:

- Sentinel-1/2, Dynamic World, ESA WorldCover, OSM, and spectral indices provide evidence.
- Evidence agreement can improve confidence but cannot replace independent validation.
- U-Net trained on weak labels is a model baseline, not a ground-truth classifier.
- Calibration, uncertainty, temporal consensus, and co-registration QA reduce risk but do not prove accuracy.
- Field samples or high-quality expert reference labels are still required for final accuracy metrics.

This framing is intentional: the project is designed to be useful and transparent without overstating what can be proven from weak labels alone.
