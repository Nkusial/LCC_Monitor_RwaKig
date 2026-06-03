"""Lightweight U-Net architecture for weakly supervised land-cover segmentation."""

from __future__ import annotations

import os
from typing import Any


ARCHITECTURE_SUMMARY = {
    # Stored in reports so reviewers can see which model family produced a
    # checkpoint without opening the Python source.
    "name": "lightweight_unet",
    "input_channels": 6,
    "output_classes": 6,
    "encoder": [
        "double_conv(input, 32)",
        "max_pool + double_conv(32, 64)",
        "max_pool + double_conv(64, 128)",
        "max_pool + double_conv(128, 256)",
    ],
    "decoder": [
        "upsample + skip + double_conv(256+128, 128)",
        "upsample + skip + double_conv(128+64, 64)",
        "upsample + skip + double_conv(64+32, 32)",
    ],
    "output": "1x1 convolution to class logits",
}

SUPPORTED_ARCHITECTURES = {
    # The registry keeps model choice explicit as the project compares U-Net,
    # ResUNet, and later nested-decoder options under the same pipeline.
    "lightweight_unet": "Baseline encoder-decoder U-Net.",
    "lightweight_resunet": "Residual encoder-decoder U-Net for noisier weak-label retraining.",
    "unet_plus_plus": "Planned nested-decoder option for a later implementation.",
}


def _load_torch() -> tuple[Any, Any]:
    """Import PyTorch only when a model is actually built.

    Keeping Torch lazy prevents the normal API and validation tests from
    loading CUDA DLLs after GDAL/NumPy libraries are already active.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch
        from torch import nn
    except (ModuleNotFoundError, OSError) as exc:  # pragma: no cover - setup dependent.
        raise RuntimeError(
            "PyTorch is required for U-Net training. Install the tested CUDA "
            "wheel in the `realtime_LCC_Rwkig` environment before fitting. "
            f"Import error: {exc}"
        ) from exc
    return torch, nn


def build_unet(input_channels: int, output_classes: int, base_channels: int) -> Any:
    """Build the lightweight U-Net after lazily importing PyTorch."""
    torch, nn = _load_torch()

    class DoubleConv(nn.Module):
        """Two 3x3 convolution blocks used by both encoder and decoder."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class LightweightUNet(nn.Module):
        """A small U-Net suitable for 128 x 128 Sentinel-1/2 feature patches."""

        def __init__(self) -> None:
            super().__init__()
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 8

            self.enc1 = DoubleConv(input_channels, c1)
            self.enc2 = DoubleConv(c1, c2)
            self.enc3 = DoubleConv(c2, c3)
            self.bottleneck = DoubleConv(c3, c4)
            self.pool = nn.MaxPool2d(2)

            self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(c4, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(c3, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(c2, c1)
            self.classifier = nn.Conv2d(c1, output_classes, kernel_size=1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))

            # Skip connections preserve local edges and small settlement/water
            # signals that can vanish in the encoder bottleneck.
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.classifier(d1)

    return LightweightUNet()


def build_resunet(input_channels: int, output_classes: int, base_channels: int) -> Any:
    """Build a small residual U-Net after lazily importing PyTorch."""
    torch, nn = _load_torch()

    class ResidualBlock(nn.Module):
        """Residual block used when weak labels are noisy or imbalanced."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x: Any) -> Any:
            # The residual path makes the deeper option less brittle when weak
            # labels include mixed pixels or source disagreement.
            return self.activation(self.block(x) + self.skip(x))

    class LightweightResUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 8
            self.enc1 = ResidualBlock(input_channels, c1)
            self.enc2 = ResidualBlock(c1, c2)
            self.enc3 = ResidualBlock(c2, c3)
            self.bottleneck = ResidualBlock(c3, c4)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
            self.dec3 = ResidualBlock(c4, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.dec2 = ResidualBlock(c3, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.dec1 = ResidualBlock(c2, c1)
            self.classifier = nn.Conv2d(c1, output_classes, kernel_size=1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.classifier(d1)

    return LightweightResUNet()


def build_segmentation_model(
    architecture: str,
    input_channels: int,
    output_classes: int,
    base_channels: int,
) -> Any:
    """Build the configured segmentation architecture."""
    if architecture == "lightweight_unet":
        return build_unet(input_channels, output_classes, base_channels)
    if architecture == "lightweight_resunet":
        return build_resunet(input_channels, output_classes, base_channels)
    if architecture == "unet_plus_plus":
        raise NotImplementedError("UNet++ is registered for comparison but is not implemented yet.")
    raise ValueError(f"Unsupported architecture: {architecture}")
