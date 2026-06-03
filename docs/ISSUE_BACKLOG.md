# Development Backlog

These are Alfred Nkusi's next engineering tasks for strengthening the Kigali
land-cover change monitoring prototype before a polished public release.

## ML And Labeling

### Add Label Schema For Supervised Learning

Define a label table or GeoJSON format for human-reviewed change/no-change samples.

Acceptance criteria:

- Label schema includes geometry, date pair, class, reviewer, and confidence.
- Labels can be exported to a model-ready feature table.

### Train Baseline Classifier

Train a simple Random Forest or Gradient Boosting model using Sentinel-2 indices, Sentinel-1 features, and existing confidence outputs.

Acceptance criteria:

- Training script reads reproducible feature tables.
- Metrics include precision, recall, F1, and confusion matrix.
- Model artifact is versioned outside Git or through a lightweight registry.

## MLOps

### Add Experiment Tracking

Add local MLflow or a simple file-based experiment log for model runs.

Acceptance criteria:

- Each experiment records parameters, metrics, input dates, and model path.
- Validation outputs remain reproducible from config.

### Add Data Version Manifest

Create a manifest that records scene IDs, acquisition dates, asset URLs, and processing hashes.

Acceptance criteria:

- Manifest is written after preprocessing.
- Validation fails when required scene/date metadata is missing.

## WebGIS

### Add Time Pair Selector

Let users switch between monitoring date pairs on the map.

Acceptance criteria:

- API supports date-pair filtering.
- WebGIS displays active baseline and comparison dates.

### Add Downloadable GeoJSON

Expose filtered change results as downloadable GeoJSON from the WebGIS.

Acceptance criteria:

- Download respects confidence and monitored-group filters.
- File name includes date pair and filter settings.

## Operations

### Add One-Command Local Demo

Create a script that starts PostGIS, FastAPI, and WebGIS for a local demo.

Acceptance criteria:

- Script checks required ports.
- Script prints the backend and frontend URLs.
