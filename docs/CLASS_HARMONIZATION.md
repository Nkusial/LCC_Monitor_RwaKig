# Class Harmonization

This project keeps the current WebGIS monitoring groups for the Version 1
prototype, but future weak-label and ML work must use a cleaner harmonized
taxonomy. The purpose is to reduce confusing labels such as `bare_sparse`,
`mixed`, or ambiguous `vegetation` when multiple evidence sources disagree.

## Current Groups

The current detection output uses:

- `vegetation`
- `built_up`
- `water_moisture`
- `bare_sparse`
- `mixed`
- `unknown`

These groups remain valid for the current WebGIS and API because they describe
the existing rule-based monitoring output.

## Harmonized Classes

Future weak labels, soft labels, and segmentation models should map evidence
into:

- `built_up`
- `managed_vegetation`
- `natural_vegetation`
- `bare_soil`
- `water_wetland`
- `uncertain_mixed`

The split between `managed_vegetation` and `natural_vegetation` is important.
Sentinel indices can show greenness, but they do not always distinguish crop,
garden, grass, shrub, and tree cover without additional temporal or external
evidence.

## Evidence Rules

Use hard labels only where independent sources agree. Good weak-label candidates
should combine at least two of these evidence streams:

- Dynamic World probability
- ESA WorldCover class agreement
- OSM land-use or feature support
- Sentinel-2 index support
- Sentinel-1 backscatter or ratio support

When evidence is mixed, keep probability-style labels instead of forcing a
single class. Mixed pixels should become `uncertain_mixed` or soft-label vectors,
not overconfident polygons.

## Implementation Contract

The machine-readable rules live in `configs/class_harmonization.yaml`.
Any future weak-label generation stage should read that file and write:

- dominant harmonized class
- source agreement score
- soft-label probability vector
- uncertainty mask
- notes for disagreement or low-confidence evidence
