# U-Net Segmentation Baseline

Phase 31 starts the deep-learning track with a small U-Net land-cover
segmentation baseline.

## Why U-Net First

U-Net is the first model because it is explainable, lightweight, and appropriate
for limited weak labels. It predicts per-pixel land-cover probabilities from a
single-date Sentinel-1/2 feature stack before the project attempts direct
change-detection models.

## Architecture

```text
Input:
  6 channels, 128 x 128
  NDVI, NDWI, NDBI, VV, VH, VV/VH

Encoder:
  double convolution blocks with max-pooling

Bottleneck:
  compact high-level feature representation

Decoder:
  transposed convolutions plus U-Net skip connections

Output:
  6 class logits per pixel
```

Configured classes:

```text
background_or_unlabeled
built_up
managed_or_natural_vegetation
bare_soil
water_wetland
uncertain_mixed
```

## Training Policy

The baseline uses Phase 27 patch files and only trains on pixels where the
`training_mask` is valid. The mask excludes low-confidence and disagreement
pixels, so the model is weakly supervised rather than field-validated.

The current training loop is imbalance-aware because weak-label satellite
patches are dominated by common classes. It uses:

```text
class-aware patch sampling
effective-number class weights
maximum class-weight cap to avoid unstable rare-class gradients
weighted cross entropy + generalized Dice loss
per-class weak-label IoU, Dice, and recall reporting
```

This is especially important for minority classes such as `water_wetland` and
`uncertain_mixed`, which can disappear from dominant-class predictions when a
plain cross-entropy U-Net is trained on imbalanced weak labels.

The first command is a preflight:

```powershell
conda activate realtime_LCC_Rwkig
python pipelines/15_train_unet_baseline.py
```

It writes:

```text
data/outputs/unet_baseline_report.json
```

Actual fitting requires PyTorch:

```powershell
python pipelines/15_train_unet_baseline.py --fit
```

The current tested setup uses:

```text
torch 2.6.0+cu124
NVIDIA GeForce RTX 4060 Laptop GPU
```

The fitted baseline writes a local checkpoint and a weak-label validation
report:

```text
data/interim/models/unet_baseline.pt
data/outputs/unet_baseline_report.json
```

The checkpoint is a local artifact and is not committed to Git.

On this Windows Conda geospatial environment, PyTorch is installed from the
official CUDA wheel index because the Conda install path timed out. The model
module sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing PyTorch to avoid the
known OpenMP runtime conflict between the CUDA wheel and Conda numeric
libraries.

The training script also imports the model/Torch module before NumPy. In this
Windows Conda stack, importing NumPy first can make PyTorch fail while loading
`fbgemm.dll`.

## Current Limitation

The validation loss is measured against weak labels and masks, not independent
field labels. Treat it as an engineering signal that the model can learn from
the current weak-label dataset, not as field accuracy.

Per-class validation metrics are also weak-label metrics. They help diagnose
model collapse or minority-class under-prediction, but they are not a substitute
for field or expert reference labels.
