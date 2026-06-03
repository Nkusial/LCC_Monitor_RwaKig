# Full-AOI U-Net Reliability Audit

Phase 37 compares the full-AOI U-Net model outputs against the weak evidence
that is already available in the project.

This phase does not claim field accuracy. It answers a narrower engineering
question: where do weakly supervised model predictions agree with current
agreement zones, and where should a human review the map before trusting it?

## Inputs

- `data/interim/unet_full_aoi/unet_full_aoi_manifest.json`
- `data/interim/external_sources/external_weak_sources_summary.json`
- weak-source agreement labels
- weak-source source-count raster
- weak-source disagreement mask

## Checks

The audit reports:

- compatible U-Net pixels inside weak-source agreement zones
- weak-source disagreement overlap
- high-entropy pixels
- low-confidence pixels
- candidate review pixels

The U-Net model currently merges managed and natural vegetation into one
vegetation class. For reliability checking, either harmonized vegetation class
is counted as compatible with the model vegetation prediction.

## Run

```powershell
conda activate realtime_LCC_Rwkig
python pipelines\21_validate_unet_full_aoi.py
```

The report is written to:

```text
data/outputs/unet_full_aoi_validation_report.json
```

## Interpretation

Use this report as a model-risk layer, not an accuracy statement.

High entropy, low confidence, and weak-source disagreement zones are the best
places to collect field labels, digitize reference labels, or inspect
high-resolution imagery before publishing stronger claims.
