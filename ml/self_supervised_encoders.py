"""Future self-supervised spatial encoders for Sentinel patch tensors.

The current production prototype uses a CPU-friendly MLP autoencoder over patch
descriptors. These modules define the next compute-heavy step without changing
the current pipeline: train on Sentinel/radar/topography tensors, reconstruct
or contrast patches, and interpret embeddings only after fitting.
"""

from __future__ import annotations

import os


# The local Conda stack mixes MKL/LLVM OpenMP with PyTorch on Windows. Keeping
# one thread and allowing duplicate OpenMP runtimes prevents import-time failure;
# the spatial encoder pipeline remains a prototype-quality CPU/GPU training path.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:  # pragma: no cover - tests do not require a GPU or PyTorch installation.
    import torch
    from torch import nn
except (ModuleNotFoundError, OSError):  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for spatial self-supervised encoders. "
            "Use the existing MLP descriptor track until GPU/torch compute is available."
        )


if torch is not None and nn is not None:

    class ConvPatchAutoencoder(nn.Module):
        """Small convolutional autoencoder for 128 x 128 Sentinel feature patches."""

        def __init__(self, in_channels: int, embedding_channels: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, embedding_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embedding_channels),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embedding_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            embedding = self.encoder(x)
            reconstruction = self.decoder(embedding)
            return reconstruction, embedding


    class TemporalPatchTransformer(nn.Module):
        """Transformer encoder for multi-date patch embeddings.

        A convolutional stem first converts each date into a compact vector.
        The transformer then models date-to-date context without using class
        labels. Weak sources should still be used only for post-hoc cluster
        interpretation and review-sample selection.
        """

        def __init__(
            self,
            in_channels: int,
            embedding_dim: int = 128,
            heads: int = 4,
            layers: int = 2,
        ):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, embedding_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                batch_first=True,
                dim_feedforward=embedding_dim * 4,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: batch x time x channels x height x width
            batch, time, channels, height, width = x.shape
            flattened = x.reshape(batch * time, channels, height, width)
            date_embeddings = self.stem(flattened).reshape(batch, time, -1)
            encoded = self.temporal_encoder(date_embeddings)
            return encoded.mean(dim=1)

else:

    class ConvPatchAutoencoder:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            require_torch()


    class TemporalPatchTransformer:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            require_torch()
