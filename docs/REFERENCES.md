# References And Scientific Basis

This page lists the main scientific and technical sources behind the project. The links are kept for traceability, while the notes explain the part of the prototype each reference supports.

## Satellite Data And Reference Products

| Topic | Reference | Project relevance |
| --- | --- | --- |
| Sentinel-2 optical imagery | European Space Agency, [Sentinel-2 mission](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) | Supports the use of multispectral Sentinel-2 data for NDVI, NDWI, NDBI, false-color visualization, and optical land-cover evidence. |
| Sentinel-1 radar imagery | European Space Agency, [Sentinel-1 mission](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1) | Supports the use of radar observations as cloud-resilient evidence for surface change and moisture-sensitive monitoring. |
| Sentinel-1 RTC | Microsoft Planetary Computer, [Sentinel-1 RTC](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) | Supports the use of radiometrically terrain-corrected Sentinel-1 assets for VV, VH, and VV/VH features. |
| Dynamic World | Brown et al. (2022), [Dynamic World, near real-time global 10 m land use land cover mapping](https://www.nature.com/articles/s41597-022-01307-4) | Supports Dynamic World as a weak-label and weak-evidence source for agreement, disagreement, and review-zone analysis. |
| ESA WorldCover | ESA WorldCover, [Worldwide land-cover mapping](https://esa-worldcover.org/en) | Supports the use of ESA WorldCover as an independent weak-reference layer for broad land-cover classes. |
| OpenStreetMap | OpenStreetMap contributors, [copyright and contributors](https://www.openstreetmap.org/copyright) | Supports the use of mapped buildings, roads, and land-use context as auxiliary evidence, especially for built-up interpretation. |

## Spectral Indices

| Index | Reference | Project relevance |
| --- | --- | --- |
| NDVI | Rouse et al. (1974), [Monitoring vegetation systems in the Great Plains with ERTS](https://ntrs.nasa.gov/citations/19740022614) | Supports vegetation signal and vegetation-loss interpretation. |
| NDWI | McFeeters (1996), [Normalized Difference Water Index](https://doi.org/10.1080/01431169608948714) | Supports water and wetness signal interpretation. |
| NDBI | Zha, Gao, and Ni (2003), [Normalized Difference Built-up Index](https://doi.org/10.1080/01431160304987) | Supports built-up and impervious-surface evidence. |

## Validation, Confidence, And Uncertainty

| Topic | Reference | Project relevance |
| --- | --- | --- |
| Land-change accuracy assessment | Olofsson et al. (2014), [Good practices for estimating area and assessing accuracy of land change](https://doi.org/10.1016/j.rse.2014.02.015) | Supports the project's separation between model confidence, weak-source agreement, and true accuracy assessment. |
| Neural-network calibration | Guo et al. (2017), [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Supports caution around raw model probabilities and motivates calibration-aware confidence reporting. |
| Deep-learning uncertainty | Gal and Ghahramani (2016), [Dropout as a Bayesian Approximation](https://proceedings.mlr.press/v48/gal16.html) | Supports explicit uncertainty and review-zone reporting in deep-learning workflows. |

## Machine Learning And Geospatial Stack

| Component | Reference | Project relevance |
| --- | --- | --- |
| U-Net | Ronneberger, Fischer, and Brox (2015), [U-Net](https://arxiv.org/abs/1505.04597) | Supports the supervised segmentation baseline used for weak-label experiments. |
| STAC | [SpatioTemporal Asset Catalog specification](https://stacspec.org/en) | Supports reproducible scene metadata and asset tracking. |
| Cloud Optimized GeoTIFF | [COG project](https://www.cogeo.org/) | Supports efficient raster access and future raster-serving design. |
| PostGIS | [PostGIS](https://postgis.net/) | Supports spatial storage and querying for AOIs, scenes, summaries, and change polygons. |
| FastAPI | [FastAPI documentation](https://fastapi.tiangolo.com/) | Supports the local API layer for health checks, AOI metadata, scenes, and change outputs. |
| MapLibre | [MapLibre](https://maplibre.org/) | Supports the interactive WebGIS interface and hosted static map delivery. |

## Interpretation Boundary

These sources support the project's methodology and implementation choices. They do not replace field or expert reference samples. The current prototype should therefore present confidence, weak-source agreement, and review-zone evidence as decision-support information, not as final field-validated accuracy.
