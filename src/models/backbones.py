"""Backbone networks and modality adapters."""
import torch
import torch.nn as nn


class ModalityAdapter(nn.Module):
    """Lightweight adapter layer to absorb modality-specific speckle/noise and contrast profiles."""

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleSharedBackbone(nn.Module):
    """A standard feature extractor with multi-scale feature output for landmark heatmaps."""

    def __init__(self, in_channels: int = 64, feature_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat + self.res_blocks(feat)
        return feat
