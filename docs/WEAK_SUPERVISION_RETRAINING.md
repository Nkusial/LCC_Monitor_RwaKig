# Confidence-Aware Weak-Supervision Retraining

This project now treats weak supervision as a model-improvement workflow, not
only a reporting workflow.

## Workflow

1. Weak labels and Sentinel-1/2 imagery are aligned to the master grid.
2. Dynamic World, ESA WorldCover, OSM, Sentinel indices, Sentinel-1 evidence,
   temporal consensus, and prior high-confidence polygons are used to clean and
   score labels.
3. Class-balanced patch sampling retains rare-class patches and caps
   majority-only patches.
4. The v3 model config uses a residual U-Net option for weak-label noise.
5. Training combines weighted cross entropy, Dice, focal loss, and Tversky loss.
6. Pseudo-label refinement is rerun after inference to support another
   correction/retraining cycle.
7. A small independent expert sample is defined in
   `configs/expert_validation_sample.yaml` for human review where field data do
   not exist.

## Current Caveat

This still does not claim field accuracy. The system is designed to be useful
and replicable in AOIs without field labels by separating reliable predictions
from caution/review zones and by making minority-class weakness visible.

## Current Production Choice

The accepted v3 configuration uses the lightweight U-Net with class-aware
sampling, focal loss, and Tversky loss. A residual U-Net option is implemented,
but the first ResUNet experiment collapsed minority-class predictions, so it is
kept as an experimental architecture rather than the published default.

An aggressive minority loss-floor experiment improved rare-class recall but
overpredicted rare classes and reduced overall weak-source compatibility. The
production config keeps loss-floor overrides available but disabled by default.
