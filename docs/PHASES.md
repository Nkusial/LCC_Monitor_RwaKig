# Project Phases

## Completed

1. Project skeleton, Conda environment, repo structure.
2. Docker and PostGIS validation.
3. Sentinel scene discovery.
4. AOI-windowed Sentinel-1/2 preprocessing.
5. Multi-date change detection with before/after land-cover attribution.
6. Confidence scoring and PostGIS publishing.
7. FastAPI/WebGIS visualization of published changes.
8. Local reproducibility and validation report.
9. Portfolio hardening: CI workflow, API landing endpoint, and operations docs.
10. Portfolio presentation: architecture, API examples, and demo walkthrough.
11. Visual portfolio packaging: review guide, screenshot guide, and ML/MLOps roadmap.
12. Captured visual assets: WebGIS overview, API docs, and change summary screenshots.
13. Release readiness: README statistics, release checklist, and final validation guidance.
14. Publication polish: changelog, demo walkthrough, and development backlog.
15. External publishing prep: GitHub release notes, publishing guide, and issue templates.
16. Local milestone packaging: publishable files staged, milestone commit created, and `v0.1.0` tag prepared.
17. CI/CD reflection: tag-based release packaging workflow and CI/CD documentation.
18. Hosted WebGIS readiness: GitHub Pages branch publishing, interactive static WebGIS demo, and documentation setup.
19. Phase 19 satellite-derived WebGIS layers: Sentinel-2 false color, NDVI, NDWI, NDBI, Sentinel-1 radar overlays, change confidence, and baseline/after/magnitude polygon modes.
20. Phase 20 unsupervised validation: Sentinel feature clustering, internal cluster-coherence report, and no-ground-truth accuracy caveat.
20B. Phase 20B validation comparison: tested cluster counts, Sentinel-2 vs fusion features, PCA, high-confidence masks, and baseline/after stacks; best silhouette improved to 0.4459.
21. Phase 21 validation visibility: unsupervised cluster overlay, frontend validation summary panel, and reference-label planning notes.
22. Phase 22 portfolio validation story polish: README validation status, reviewer FAQ, demo walkthrough, and screenshot guidance.
23. Phase 23 visual asset refresh: Playwright-captured validation overlay and reference-label workflow screenshots for the hosted WebGIS.
24. Phase 24 release refresh and demo packaging: changelog, release notes, publishing guide, checklist, and demo walkthrough updated for the validation-aware hosted WebGIS milestone.
25. Phase 25 master-grid and class harmonization: explicit AOI grid contract, raster alignment checks, and harmonized class taxonomy added before weak-label generation.
26. Phase 26 weak-label generation: conservative local bootstrap weak hard labels, soft-label probability rasters, uncertainty, and confidence outputs generated from Sentinel evidence plus high-confidence monitoring-polygon agreement.
26B. Phase 26B external weak-source ingestion: ESA WorldCover and OSM buildings/roads/landuse are ingested onto the master grid; Dynamic World is supported through a required Earth Engine/local raster export; agreement and disagreement masks are generated.
27. Phase 27 patch dataset: local 128 x 128 feature/label patches generated with confidence, uncertainty, agreement, disagreement, spatial block IDs, and train/validation/test splits.
28. Phase 28 multi-year training scene discovery: 2023, 2024, 2025, and optional 2026 Sentinel-2 scenes are searched with a preferred <10% cloud policy and matched to Sentinel-1 RTC VV/VH acquisitions for future deep-learning train/validation/test data.
29. Phase 29 training stack manifest: selected multi-year scene pairs are converted into preprocessing-ready metadata catalogs with Sentinel-2 asset checks before heavy AOI-windowed raster reads.
30. Phase 30 multi-year training preprocessing: controlled metadata/optical/radar/all runner added for AOI-windowed preprocessing of selected training scene pairs before patch expansion.
31. Phase 31 U-Net segmentation baseline: lightweight U-Net architecture, weak-label training config, GPU training path, imbalance-aware loss/sampling, checkpoint output, and weak-label validation-loss reporting.
32. Phase 32 multi-year training patches: processed 2023-2025 Sentinel-1/2 stacks are converted into temporal-split weak-label patches for U-Net training.
33. Phase 33 U-Net patch inference: trained checkpoint inference writes patch-level class probabilities, dominant class, maximum probability, and entropy for downstream mosaicking and WebGIS publication.
34. Phase 34 U-Net prediction mosaics: patch-level model outputs are mosaicked into per-date geospatial rasters for dominant class, confidence, entropy, and patch coverage.
35. Phase 35 full-AOI U-Net inference: sliding-window model inference runs across complete valid AOI raster stacks, producing wall-to-wall dominant class, confidence, entropy, and coverage rasters.
36. Phase 36 full-AOI U-Net WebGIS publication: selected full-AOI dominant class, confidence, and entropy outputs are exported as hosted WebGIS PNG overlays with weak-supervision caveats.
37. Phase 37 full-AOI U-Net reliability audit: full-AOI model outputs are compared with weak-source agreement zones, disagreement masks, confidence, and entropy to identify compatible predictions and candidate review areas.
38. Phase 38 reliability-aware WebGIS: U-Net review zones and reliability summary metrics are exposed in the hosted frontend so users can separate supported predictions from areas needing review.
39. Phase 39 Dynamic World ingestion path: Earth Engine export tooling added so Dynamic World labels can be downloaded, harmonized, and included in weak-source agreement masks after local Earth Engine authentication.
40. Phase 40 temporal U-Net consensus: full-AOI model predictions across dates are summarized into consensus class, consensus fraction, and instability rasters.
41. Phase 41 review-to-retraining loop: reliability outputs are converted into prioritized review tasks and a corrected-label retraining manifest contract.
42. Phase 42 test-set readiness: the project reports whether a clean held-out test split exists before any final accuracy claim.
43. Phase 43 deployment hardening: a containerized FastAPI backend deployment path is added alongside the existing PostGIS service.
44. Phase 44 feature expansion: EVI2, SAVI, MNDWI, BSI, and Sentinel-1 log-ratio rasters are derived for confidence-improved retraining.
45. Phase 45 strict refined pseudo-labels: high-confidence, low-entropy, temporally stable, weak-source-compatible pixels are exported for future retraining.
46. Phase 46 weak confidence calibration: model confidence is compared with weak-source support bins so confidence can be treated as model-risk guidance.
47. Phase 47 topography ingestion: open Copernicus DEM is aligned to the master grid and slope, ruggedness, and topographic position features are derived for Kigali's mountainous AOI.
48. Phase 48 benchmark weak-source guardrails: 2026 Dynamic World, ESA WorldCover, and OSM weak sources are harmonized onto the master grid with AOI masking, georeference checks, and bootstrap-label exclusion.
49. Phase 49 held-out benchmark package: the 2026 candidate, aligned weak sources, and no-leakage audit are assembled into an evaluation-only package with explicit blockers and forbidden actions.
50. Phase 50 held-out 2026 test preprocessing: the selected Sentinel-2/Sentinel-1 candidate is processed under a `test_2026_*` tag with no training/validation catalog edits or pseudo-label leakage.
51. Phase 51 frozen 2026 U-Net proxy benchmark: the frozen checkpoint is run on the test-only stack and compared with aligned 2026 weak-source agreement without retraining or tuning.
52. Phase 52 2026 benchmark WebGIS visibility: the frozen 2026 proxy benchmark is published as clearly labeled WebGIS layers and sidebar metrics without turning weak-source agreement into an accuracy claim.
53. Phase 53 reliability and uncertainty control: train/validation reliability, frozen 2026 stress metrics, calibration gaps, temporal stability, terrain effects, and likely no-field-label improvements are summarized without touching the frozen test set.
54. Phase 54 WebGIS georeference correction: hosted raster overlays are exported as Web Mercator tile layers so browser display uses the same spatial reference as the basemap.
55. Phase 55 geospatial alignment hardening: downstream analysis families are checked against the master grid before change detection or WebGIS publication.
56. Phase 56 improvement-track controls: co-registration QA, temporal consensus, U-Net versus self-supervised comparison planning, public-demo packaging, and hosted-backend deployment requirements are consolidated into runnable reports.
57. Phase 57 public demo raster optimization: oversized fallback PNG overlays are palette-optimized while the WebGIS continues to use georeferenced XYZ tiles for map display.
58. Phase 58 model-track comparison: weak-supervised U-Net outputs are compared with unsupervised clustering evidence, and the self-supervised/deep-clustering benchmark path is defined without field-accuracy overclaiming.
59. Phase 59 label-free embedding baseline: Sentinel-1/2 and terrain patch descriptors are clustered without weak labels during fitting, then interpreted post hoc for review-burden comparison.
60. Phase 60 reliability map visibility: likely reliable and review-required change polygons are exposed as a dedicated WebGIS color mode, legend, and popup fields.
61. Phase 61 WebGIS sidebar simplification: developer geospatial-alignment and co-registration QA diagnostics remain packaged for review but are removed from the operational map sidebar.
62. Phase 62 self-supervised embedding track: Sentinel-1/2, index, radar, and terrain patch descriptors are encoded with a label-free autoencoder, clustered, anomaly-scored, interpreted with weak sources post hoc, and compared against the U-Net reliability track.
63. Phase 63 self-supervised WebGIS review layer: label-free embedding support, weak-support review, and high-anomaly patch footprints are exported to the WebGIS as a distinct review overlay, not as land-cover classes or field accuracy.
64. Phase 64 self-supervised review sampling: label-free embedding anomaly scores and weak-source gaps are converted into balanced review-sample candidates before any future U-Net retraining, while excluding the frozen 2026 benchmark.
65. Phase 65 spatial self-supervised encoder: the descriptor-only baseline is upgraded with a spatial patch encoder path, using a convolutional autoencoder when PyTorch is healthy and a spatial-grid autoencoder fallback on the current Windows DLL-constrained environment.

## Current Capabilities

- Local 20 km x 20 km Kigali AOI.
- Multi-date Sentinel-2/Sentinel-1 monitoring stack.
- Change outputs aligned to clear monitored groups:
  `vegetation`, `built_up`, `water_moisture`, `bare_sparse`, `mixed`, `unknown`.
- Changed/no-change quantities are visible in the API summary, WebGIS statistics, and legend.
- Hosted WebGIS now exposes satellite-derived raster overlays, not only change-feature snapshots.
- Unsupervised validation is available for no-field-label project stages.
- Validation evidence is visible in the hosted WebGIS through the cluster overlay and summary panel.
- Portfolio materials explain the validation caveat and best unsupervised result clearly.
- Portfolio screenshots now show the validation-aware WebGIS, including the unsupervised overlay and reference-label plan.
- Release materials now package the hosted validation-aware WebGIS as the `v0.2.0` milestone.
- Master-grid validation now checks processed analysis rasters against an explicit 10 m AOI grid contract.
- Class harmonization rules are defined before weak-label or segmentation work.
- Weak-label generation now registers Dynamic World, ESA WorldCover, OSM buildings/roads/landuse, Sentinel-2 NDVI/NDWI/NDBI, Sentinel-1 VV/VH, and existing high-confidence change polygons as candidate evidence sources.
- External-source ingestion writes agreement/disagreement rasters from available aligned weak sources before patch generation.
- Patch generation writes local ML-ready `.npz` patches and a manifest while excluding disagreement and low-confidence pixels from training masks.
- Multi-year training scene discovery now prepares 2023/2024 train, 2025 validation, and optional 2026 test Sentinel-1/2 pairs before expanding the patch dataset for deep learning.
- Training stack readiness now checks selected scene IDs and required Sentinel-2 assets before raster preprocessing is expanded beyond the demo dates.
- Multi-year training preprocessing can now be run safely in metadata mode first, then scaled to optical/radar/all raster modes by split, year, or limit.
- A lightweight U-Net segmentation baseline is available as a GPU-trainable checkpoint workflow with class-aware sampling, effective-number class weights, and weighted CE + Dice loss.
- Multi-year training patches now use the processed training stacks and inherit temporal train/validation splits from the scene catalog.
- U-Net patch inference writes local model probability and entropy products before map publication.
- U-Net prediction mosaics preserve georeference and date-pair identity before WebGIS publication.
- Full-AOI U-Net inference now provides wall-to-wall model outputs instead of sampled patch-only coverage.
- The hosted WebGIS now exposes full-AOI U-Net dominant class, confidence, and entropy overlays as model outputs.
- Full-AOI U-Net reliability reporting now separates weak-evidence compatibility from true field accuracy.
- Reliability review zones and U-Net compatibility/review metrics are now visible in the WebGIS.
- Dynamic World ingestion is now scripted through Earth Engine export and plugs into the existing external weak-source harmonization stage.
- Temporal U-Net consensus and instability reporting are available for stronger multi-date reliability checks.
- Review-to-retraining and held-out test-readiness reports prevent unsupported accuracy claims.
- A backend container path is available for local deployment hardening.
- Expanded Sentinel features, strict pseudo-label refinement, and weak calibration reports are available for confidence-improved U-Net retraining.
- DEM, slope, ruggedness, and topographic position are now available as Version 2 model features.
- Imbalance-aware v3 weak supervision now balances rare-class patches before U-Net training and adds focal loss to weighted CE + Dice training.
- 2026 benchmark weak sources are now isolated from train/validation data, checked against the master grid, and packaged for future frozen-model proxy evaluation.
- The selected 2026 Sentinel-1/2 candidate is now available as a test-only processed stack for frozen-model inference.
- Frozen 2026 U-Net inference now reports weak-source proxy compatibility and review-zone burden while preserving the no-field-accuracy caveat.
- The hosted WebGIS now exposes 2026 frozen benchmark dominant-class, confidence, entropy, and review-zone overlays with a no-cheating protocol in the sidebar.
- Reliability and uncertainty controls now explain why the 2026 confidence stress is low, what can improve without expert samples, and which actions are forbidden to avoid benchmark leakage.
- PostGIS-backed API and MapLibre WebGIS.
- The WebGIS now includes a toggleable self-supervised review layer that shows
  where label-free embeddings support or question current change evidence.
- Self-supervised anomaly scores now propose balanced review samples for future
  label correction before U-Net retraining, without using the frozen 2026
  benchmark for selection or tuning.
- A spatial self-supervised encoder report now compares patch-layout embeddings
  against U-Net review burden and weak-source compatibility.
- Reproducible local validation through `pipelines/07_validate_outputs.py`.

## Next Candidate Phase

The next phase can make the self-supervised track temporal instead of only
spatial:

- generate a repeated-window multi-date patch manifest
- train the temporal transformer scaffold on matched row/column windows
- use the Phase 64 review-sample manifest to guide independent review before any retraining
- keep the Phase 63 review overlay as the map comparison surface
- compare against U-Net only with frozen no-cheating runs
- keep Dynamic World, ESA, OSM, Sentinel indices, and radar evidence as post-hoc interpretation sources

Before that comparison is treated as an alert source, Phase 56 provides the
publish-gate controls that decide whether outputs should be published,
published with review labels, or held for more processing.
